# Sparse LLMs on Loihi — Pruning Experiments

This repository contains post-training weight pruning experiments targeting neuromorphic hardware (Intel Loihi), where unstructured weight sparsity directly reduces active computation rather than being merely a compression artefact.

## Scripts

| File | Description |
|------|-------------|
| `scripts/prune_wanda.py` | **Wanda baseline** (Sun et al., 2023). Scores each weight as `\|W[i,j]\| × √E[x_j²]` — weight magnitude times per-channel activation RMS. Semi-structured (constant k per output neuron). Equivalent to our method with a diagonal Σ_X; serves as the direct published baseline. |
| `scripts/prune_quadratic.py` | **Greedy covariance pruning (no weight correction).** For each output neuron, greedily selects the k channels minimising the joint reconstruction error `w_S^T Σ_X[S,S] w_S` using the full activation second-moment matrix. Semi-structured (constant k per row). |
| `scripts/prune_cancellation.py` | **Greedy covariance pruning + closed-form weight correction.** Same greedy selection as above, followed by a least-squares update to the remaining weights: `Δw[K] = Σ_X[K,K]⁻¹ Σ_X[K,S] w[S]`. Rows sharing the same prune mask are batched for efficiency. |
| `scripts/prune_sparsegpt.py` | **SparseGPT baseline** (Frantar & Alistarh, 2023). OBS saliency scoring `W[i,j]² / H_inv[j,j]` with column-ordered weight corrections applied in blocks of 128. The current strongest published single-shot pruning method. |
| `scripts/sparsity_sweep.py` | Runs Wanda and cancellation pruning across sparsity levels (30–70%) and saves PPL results to `results/sparsity_sweep.json`. |
| `scripts/benchmark_fla.py` | Runs `lm-evaluation-harness` on a model with FLA model-type registration (required for the `transformer` architecture). Evaluates HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, PIQA, LAMBADA. |

## Results (1B transformer, 50% sparsity, no fine-tuning)

All models evaluated zero-shot on `lm-evaluation-harness`. PPL on WikiText-2 test set.

| Model | PPL (WikiText-2) | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | Avg Acc |
|-------|-----------------|-------|-------|-----------|------|------------|---------|
| `transformer-1B-dense-baseline` | 19.5 | 0.610 | 0.268 | 0.366 | 0.683 | 0.525 | **0.472** |
| `transformer-1B-dense-baseline-continued` | 19.4 | 0.631 | 0.289 | 0.373 | 0.687 | 0.517 | **0.484** |
| Wanda (`prune_wanda.py`) | 3,545 | 0.271 | 0.228 | 0.260 | 0.523 | 0.493 | 0.296 |
| Greedy covariance (`prune_quadratic.py`) | 1,103 | 0.279 | 0.216 | 0.267 | 0.535 | 0.476 | 0.355 |
| Greedy + weight correction (`prune_cancellation.py`) | 615 | 0.291 | 0.205 | 0.272 | 0.545 | 0.471 | 0.298 |
| SparseGPT (`prune_sparsegpt.py`) | **29.6** | — | — | — | — | — | — |

**Key observations:** SparseGPT (Frantar & Alistarh, 2023) is the strongest single-shot baseline at 50% sparsity, achieving PPL 29.6 — close to the dense model. Our cancellation-aware method (615 PPL) substantially outperforms Wanda (3,545 PPL) — a **5.8× improvement** — by exploiting off-diagonal activation covariance structure, but lags behind SparseGPT, which applies iterative OBS weight corrections. Task accuracy results for SparseGPT are pending; accuracy gains across all methods are modest without fine-tuning recovery, which is expected at 50% sparsity.

## Sparsity sweep (PPL vs sparsity level)

WikiText-2 PPL across sparsity levels. Dense baseline PPL: 19.5.

| Sparsity | Wanda | Greedy + correction (ours) |
|----------|-------|---------------------------|
| 30% | 34.8 | **32.6** |
| 40% | 189.6 | **84.3** |
| 50% | 3,545 | **615** |
| 60% | 11,319 | **3,507** |
| 70% | 24,670 | **9,065** |

Our method outperforms Wanda at every sparsity level. The gap widens significantly past 40%, where the off-diagonal covariance structure becomes increasingly important for identifying cancellation-prone weight groups. Full sweep results in `results/sparsity_sweep.json`.

## Model weights

Dense and pruned model weights are available on the Hugging Face Hub under [`ikimyaii`](https://huggingface.co/ikimyaii).
