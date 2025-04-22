# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# The text below is the original header from the nanoGPT library
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# Try importing flash_attn, handle ImportError if not installed
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    _flash_attn_available = True
except ImportError:
    print("FlashAttention not found. Using standard PyTorch attention.")
    _flash_attn_available = False


def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

def get_sinusoidal_embeddings( n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb


class Block(nn.Module):

    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        # Common layers
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)

        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)

        # Mode-specific initializations
        if config.norm_mode == 'baseline': # Corresponds to original use_nGPT=0
             self.rmsnorm_att = RMSNorm(config.n_embd)
             self.rmsnorm_mlp = RMSNorm(config.n_embd)
        elif config.norm_mode == 'ngpt' or config.norm_mode == 'dotprod': # Both nGPT and dotprod need these parameters
            # Parameters for nGPT's scaling/interpolation (also used by dotprod's residual update)
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            # Parameters for nGPT's QKV scaling
            self.sqk_init_value = 1.0
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            # Parameters for nGPT's MLP scaling
            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))
        # No specific params needed only for dotprod init yet


    def justnorm(self, x):
        # Ensure normalization happens in float32 for stability, then cast back
        # Or keep in bfloat16? Let's try keeping it for now.
        res = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-6) # Add epsilon for stability
        return res

    # Helper for the nGPT / dotprod residual update logic
    def _apply_ngpt_residual(self, x, delta, alpha_param, alpha_init_value, alpha_init_scaling):
        lr = alpha_param * (alpha_init_value / alpha_init_scaling)
        lr = torch.abs(lr)
        A_norm = self.justnorm(x)
        B_norm = self.justnorm(delta)
        res = A_norm + lr * (B_norm - A_norm)
        return self.justnorm(res)

    def forward(self, h):
        B, T, C = h.size()
        x = h # Store original input for residual connections

        # --- Attention Block ---
        hin = x
        if self.config.norm_mode == 'baseline':
            hin = self.rmsnorm_att(x)

        q = self.query(hin)
        k = self.key(hin)
        v = self.value(hin)

        q = q.view(B, T, self.config.n_head, C // self.config.n_head)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head)

        # RoPE
        sinusoidal_pos = get_sinusoidal_embeddings(T, C // self.config.n_head).to(device=q.device, dtype=q.dtype) # Match dtype
        q_r, k_r = apply_rotary_position_embeddings(sinusoidal_pos, q.transpose(1, 2), k.transpose(1, 2))
        q = q_r.transpose(1, 2)
        k = k_r.transpose(1, 2)

        # Pre-attention scaling/norm for nGPT modes
        if self.config.norm_mode == 'ngpt' or self.config.norm_mode == 'dotprod':
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(1, 1, self.config.n_head, C // self.config.n_head)
            q = sqk * self.justnorm(q)
            k = sqk * self.justnorm(k)

        # Attention calculation (FlashAttention or Standard)
        # Ensure inputs are bfloat16 if that's the compute type
        # (Assuming layers are defined as bfloat16, QKV will be bf16)
        q_attn = q # Use q,k,v directly as they should have the correct dtype from linear layers
        k_attn = k
        v_attn = v

        # Check GPU capability for FlashAttention
        use_flash = False
        if _flash_attn_available and q_attn.device.type == 'cuda':
            major, _ = torch.cuda.get_device_capability(q_attn.device)
            if major >= 8: # Ampere or newer
                use_flash = True

        if use_flash:
            # Use FlashAttention
            sqrt_head_dim = (C / self.config.n_head) ** 0.5
            softmax_scale = sqrt_head_dim if (self.config.norm_mode == 'ngpt' or self.config.norm_mode == 'dotprod') else (1.0 / sqrt_head_dim)
            # Ensure inputs to flash_attn are explicitly cast if layers weren't guaranteed bf16
            # Let's assume they are bf16 from the Linear layers for now
            y = flash_attn_func(q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16), dropout_p=0.0, softmax_scale=softmax_scale, causal=True, window_size=(-1, -1), alibi_slopes=None, deterministic=True)
            # Cast back to original input dtype if necessary (e.g., if input `h` was fp32)
            y = y.to(dtype=h.dtype) 
            y = y.contiguous().view(B, T, C)
        else:
            # Use standard PyTorch scaled_dot_product_attention
            # Transpose B, T, H, D -> B, H, T, D
            q_std = q_attn.transpose(1, 2)
            k_std = k_attn.transpose(1, 2)
            v_std = v_attn.transpose(1, 2)
            # F.scaled_dot_product_attention handles scaling and causal mask
            y_std = F.scaled_dot_product_attention(q_std, k_std, v_std, 
                                                   attn_mask=None, 
                                                   dropout_p=0.0, # Apply dropout here if needed (config.dropout?)
                                                   is_causal=True) 
            # Transpose back B, H, T, D -> B, T, H, D
            y = y_std.transpose(1, 2)
            # Reshape B, T, H, D -> B, T, C
            y = y.contiguous().view(B, T, C)
            # Cast back to original input dtype if necessary
            y = y.to(dtype=h.dtype)

        # Attention projection
        delta_att = self.att_c_proj(y)

        # --- Attention Residual Connection ---
        if self.config.norm_mode == 'baseline':
            x = x + delta_att
        elif self.config.norm_mode == 'ngpt':
            x = self._apply_ngpt_residual(x, delta_att, self.attn_alpha, self.attn_alpha_init_value, self.attn_alpha_init_scaling)
        elif self.config.norm_mode == 'dotprod':
            # Initialize r for the block, ensure correct dtype and device
            # Create r tensor only once per block instance, initialized to ones.
            # This assumes r is reset for each forward pass through the block.
            r = torch.ones_like(x[..., :1], dtype=h.dtype) # Shape [B, T, 1]

            dot_att = torch.sum(x * delta_att, dim=-1, keepdim=True) # Use original x for dot product
            # Need to handle potential dtype mismatches if autocast is used
            r = r.to(dot_att.dtype) + dot_att # Accumulate r
            # Ensure r doesn't become zero or negative if dot products are negative? Clamp?
            r = torch.clamp(r, min=1e-6) # Clamp r for stability

            delta_att_scaled = delta_att / r # Scale delta by 1/r
            # Apply residual using the nGPT helper, but with the scaled delta
            x = self._apply_ngpt_residual(x, delta_att_scaled, self.attn_alpha, self.attn_alpha_init_value, self.attn_alpha_init_scaling)
            # Store r for MLP part. No need to register buffer if recomputed. We'll pass it.
            r_after_attn = r 


        # --- MLP Block ---
        x_mlp_in = x # Input to MLP is the output of the Attn block
        hin = x_mlp_in
        if self.config.norm_mode == 'baseline':
            hin = self.rmsnorm_mlp(x_mlp_in)

        uv = self.c_fc(hin)
        # Pre-MLP scaling for nGPT modes
        if self.config.norm_mode == 'ngpt' or self.config.norm_mode == 'dotprod':
            suv = (self.suv * (self.suv_init_value / self.suv_init_scaling) * (C ** 0.5))
            uv = suv * uv
        u, v_silu = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v_silu)
        delta_mlp = self.mlp_c_proj(x_mlp)

        # --- MLP Residual Connection ---
        if self.config.norm_mode == 'baseline':
            x = x + delta_mlp # Add to output of attention block
        elif self.config.norm_mode == 'ngpt':
            x = self._apply_ngpt_residual(x, delta_mlp, self.mlp_alpha, self.mlp_alpha_init_value, self.mlp_alpha_init_scaling)
        elif self.config.norm_mode == 'dotprod':
            # Spec: r_k <- r_k + <x_k, A_k>. Use x before residual update for dot product.
            dot_mlp = torch.sum(x * delta_mlp, dim=-1, keepdim=True) 
            # Use the r accumulated from the attention step
            r_mlp = r_after_attn.to(dot_mlp.dtype) + dot_mlp # Accumulate further
            r_mlp = torch.clamp(r_mlp, min=1e-6)

            delta_mlp_scaled = delta_mlp / r_mlp # Scale delta by 1/r (cumulative)
            # Apply residual using the nGPT helper, but with the scaled delta
            x = self._apply_ngpt_residual(x, delta_mlp_scaled, self.mlp_alpha, self.mlp_alpha_init_value, self.mlp_alpha_init_scaling)

        return x # Return final output of the block

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0 ** 0.5)    # 1 / sqrt(n_embd)
    use_nGPT: int = 0 # Deprecate this in favor of norm_mode? Keep for now.
    norm_mode: str = 'baseline' # Add this field ('baseline', 'ngpt', 'dotprod')
    dropout: float = 0.0
    bias: bool = False
    dtype: str = 'bfloat16' # Add dtype field

class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config # Store config

        self.transformer = nn.ModuleDict(dict(
            # Ensure embedding weights are float32 as per spec
            wte = nn.Embedding(config.vocab_size, config.n_embd, dtype=torch.float32),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, il) for il in range(config.n_layer)])
        ))
        # Ensure LM head weight matches embedding dtype if not tied, or handle tying carefully.
        # Let's assume lm_head should also be float32 initially, but its input will be bfloat16/float16.
        # Casting the input before the linear layer is usually handled by AMP or manual cast.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=torch.float32) # Use float32 for lm_head weights

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # *we don't use it becuase in the nGPT paper there was no weight tying of weights*
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale/math.sqrt(2 * config.n_layer))
        
        # Report number of parameters (moved slightly earlier)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        # Mode-specific final layer setup
        if config.norm_mode == 'ngpt' or config.norm_mode == 'dotprod':
            # nGPT specific scaling for output logits
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            self.sz = torch.nn.Parameter(self.sz_init_scaling*torch.ones(config.vocab_size, dtype=torch.float32))
        elif config.norm_mode == 'baseline':
            self.rmsnorm_f = RMSNorm(config.n_embd)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # Embedding lookup (output is float32)

        # Determine computation dtype from training script (e.g., bfloat16)
        # Use torch.get_autocast_gpu_dtype() if available and in autocast context, else default?
        # Or rely on explicit config 'dtype' from train.py? Let's use the config.
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.dtype] # Assuming config has dtype
        
        # Cast embedding output to compute_dtype for transformer blocks
        x = self.transformer.drop(tok_emb.to(ptdtype))

        for block in self.transformer.h:
            x = block(x)

        # Final layer processing based on mode
        if self.config.norm_mode == 'baseline':
            x = self.rmsnorm_f(x) # Apply final RMSNorm
            # Ensure lm_head input matches its weight dtype (float32)
            logits = self.lm_head(x.float()) # Cast input to float32 before final layer
        elif self.config.norm_mode == 'ngpt' or self.config.norm_mode == 'dotprod':
            # Cast final hidden state to float32 before lm_head
            x_final = x.float() 
            # Get logits first
            logits = self.lm_head(x_final) 
            # Now apply nGPT scaling *to the logits*
            sz = self.sz * (self.sz_init_value / self.sz_init_scaling) * (self.config.n_embd ** 0.5)
            logits = sz * logits # Apply scaling to logits (shape B, T, 50304)
        else: # Should not happen
             raise ValueError(f"Unknown norm_mode: {self.config.norm_mode}")


        loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits_loss = logits.view(-1, logits.size(-1))
            targets_loss = targets.view(-1)
            loss = F.cross_entropy(logits_loss, targets_loss, ignore_index=-1)

        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = False#fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

