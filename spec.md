## 1. Objective & Scope

We will extend **nGPT** (a normalized‑token fork of nanoGPT) so that, in addition to its built‑in unit‐sphere + SLERP updates, it can also:

- Train with **cumulative dot‐product normalization**:
  ```
  r_k ← r_k + ⟨x_k, A_k⟩  (fixed α = 1)
  ```
- Fall back to the original rule: `r_k = 1` when running in baseline.

All other components (RoPE, SwiGLU, tokenizer, optimizer) remain unchanged from upstream wherever possible.

---

## 2. Model Architecture

### 2.1 Base Configuration

| Item                   | Choice                                              |
|------------------------|-----------------------------------------------------|
| Parameters             | 0.5 B (≈ 24 layers, 1 024 hidden, 16 heads)         |
| Context length         | 1 024 tokens                                        |
| Positional encoding    | Rotary Position Embedding (RoPE)                    |
| Activation             | SwiGLU                                              |
| Tokenizer              | LLaMA SentencePiece (≈ 32 k vocab), bundled in nGPT |
| Normalization modes    | `baseline` (r=1) or `dotprod` (cumulative) via `--norm_mode` |
| Residual update        | nGPT’s SLERP after scaling delta by `1/r_k` in dotprod mode |

### 2.2 New Class Stubs

- **`DotProdNormalizer(nn.Module)`**
  - Maintains `r_k` buffers (`[seq_len, batch]`, `bf16`).
  - Updates: `r_k += ⟨x_k, A_k⟩` each step.
- **`apply_residual(x, delta, r)`**
  - Wraps existing SLERP; reuses nGPT util functions.
- **Config integration**
  - Add `--norm_mode` flag in `config/launcher.py`.
  - Propagate flag through to `model.py`.

---

## 3. Optimizer & Schedule

| Hyper‑param      | Value                       | Notes                                 |
|------------------|-----------------------------|---------------------------------------|
| Optimizer        | Adam (PyTorch AdamW, wd=0)  | effectively Adam without weight decay |
| Base LR          | 3×10⁻³                      |                                       |
| Scheduler        | Cosine annealing, no warm‑up|                                       |
| Batch size       | 8×1 024 tokens (global)     |                                       |
| Grad clip        | None                        |                                       |
| Precision        | bf16 (all)                  | embeddings in fp32 for stability      |
| Grad‑accum       | 0 (one backward per step)   |                                       |

---

## 4. Data Pipeline

- **Source**: OpenWebText pre‑tokenized (nGPT/nanoGPT scripts).
- **Size**: ~9 B tokens (~17 GB). Stream only needed shards.
- **Split**: 90/10 train/validation.
- **Loader**: Reuse `binloader.py`; ensure LLaMA vocab indices are respected.
- **Shuffling**: Reservoir or deterministic shard shuffle per epoch.
- **Error handling**: On corrupt chunk — log path, skip, continue without aborting.

---

## 5. Training Environment

- **GPU**: 1× NVIDIA A100 (40 GB).
- **AMP**: `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- **Compute**: ~3×10¹⁶ FLOPs over 10 k steps.
- **Throughput**: A100 BF16 ≈312 TFLOPs ⇒ ~5 min at batch size 8.
- **Memory budget**: ~10 GB (weights + optimizer states + activations).

---

## 6. Logging, Checkpoints & Metrics

- **Checkpoints**: Every 1 000 iter; keep last 3 plus best validation loss.
- **Logger**: Weights & Biases run `ngpt-0.5b-norm-pilot`; log LR, loss, and `r_k` histograms.
- **Early stopping**: None (full 10 k steps).
- **Crash safety**: Checkpoint before loader re‑initialization; resume via `--resume latest.pt`.

---

## 7. Testing Plan

### 7.1 Unit Tests

| Module             | Test                                                                 |
|--------------------|----------------------------------------------------------------------|
| `DotProdNormalizer`| Growth matches manual NumPy calculation on random inputs              |
| `apply_residual`   | SLERP invariant when `r_k = 1`                                       |
| CLI flag           | Correctly passes to model; invalid values raise `ValueError`         |

### 7.2 Functional Tests

- **Smoke train** (`--norm_mode baseline`): 2 k steps on 100 MB subset; expect val‑loss < 10.
- **Dot‑prod mode**: Run 20 steps; assert `mean(r_k) > 1.0` and no infinities.

### 7.3 Regression

- Compare final perplexity of both modes after 10 k steps; flag if dotprod diverges > 20% from baseline.

---

## 8. Error Handling Strategies

- **NaNs/Infs in `r_k`**: Apply `torch.nan_to_num`, log warning, continue.
- **OOM**: Catch `CUDA OutOfMemoryError`, halve batch size, resume.
- **Data corruption**: CRC32 check in loader; skip bad file, log event to W&B.

---

## 9. Implementation Checklist

1. Fork nGPT; create branch `feat/dotprod_norm`.
2. Add `--norm_mode {baseline,dotprod}` in `config/args.py`.
3. Implement `DotProdNormalizer` in `model/norm.py` and integrate in forward pass.
4. Update training loop: cast embeddings to `fp32`, rest under `bf16` AMP.
5. Extend `config/train_gpt.py` with new optimizer block (`AdamW`, `wd=0`).
6. Add unit tests under `tests/`.
7. Commit & push; ensure CI runs smoke tests.

---

## 10. Future Extensions (Beyond Pilot)

- Sweep α as a hyper‑parameter; add EMA‑decay variant.
- Support longer contexts (4 k, 8 k) via RoPE extrapolation.
- Enable multi‑GPU (FSDP) once single‑GPU is stable.
- Experiment with small non‑zero weight decay (e.g., 1e‑4) and gradient clipping if instability arises.

---

### Final Note
This experiment builds directly on nGPT’s code. All new logic is self‑contained so that removing it restores upstream behavior. Developers should diff changes carefully against current nGPT to avoid regressions.

