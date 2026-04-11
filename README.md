# Sparse LLMs on Loihi — Pruning Experiments

This repository contains post-training weight pruning experiments targeting neuromorphic hardware (Intel Loihi), where unstructured weight sparsity directly reduces active computation rather than being merely a compression artefact.

## Scripts

| File | Description |
|------|-------------|
| `scripts/prune_wanda.py` | **Wanda baseline** (Sun et al., 2023). Scores each weight as `\|W[i,j]\| × √E[x_j²]` — weight magnitude times per-channel activation RMS. Semi-structured (constant k per output neuron). Equivalent to our method with a diagonal Σ_X; serves as the direct published baseline. |
| `scripts/prune_quadratic.py` | **Greedy covariance pruning (no weight correction).** For each output neuron, greedily selects the k channels minimising the joint reconstruction error `w_S^T Σ_X[S,S] w_S` using the full activation second-moment matrix. Semi-structured (constant k per row). |
| `scripts/prune_cancellation.py` | **Greedy covariance pruning + closed-form weight correction.** Same greedy selection as above, followed by a least-squares update to the remaining weights: `Δw[K] = Σ_X[K,K]⁻¹ Σ_X[K,S] w[S]`. Rows sharing the same prune mask are batched for efficiency. |
| `scripts/prune_sparsegpt.py` | **SparseGPT baseline** (Frantar & Alistarh, 2023). OBS saliency scoring `W[i,j]² / H_inv[j,j]` with column-ordered weight corrections applied in blocks of 128. The current strongest published single-shot pruning method. |
| `scripts/prune_hybrid.py` | **Hybrid: cancellation-aware selection + SparseGPT OBS correction.** Replaces SparseGPT's diagonal scoring with our full-covariance greedy selection, then applies column-ordered OBS corrections. Tests whether better selection adds value on top of OBS corrections. |
| `scripts/prune_interleaved.py` | **Interleaved: block-level cancellation selection + OBS correction.** Fixes the column-ordering mismatch in the hybrid by interleaving selection and correction block-by-block. Within each block, cancellation-aware greedy selection runs on the current (already-corrected) weights; OBS corrections are applied immediately after each block. |
| `scripts/sparsity_sweep.py` | Runs Wanda and cancellation pruning across sparsity levels (30–80%) and saves PPL results to `results/sparsity_sweep.json`. |
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
| SparseGPT (`prune_sparsegpt.py`) | **29.6** | 0.555 | 0.294 | 0.428 | 0.665 | 0.517 | **0.492** |
| Hybrid: cancellation selection + OBS correction (`prune_hybrid.py`) | 958 | — | — | — | — | — | — |
| Interleaved: block-level cancellation + OBS (`prune_interleaved.py`) | 856 | — | — | — | — | — | — |

**Key observations:** SparseGPT achieves PPL 29.6 and avg accuracy 0.492 at 50% sparsity — nearly matching the dense baseline. Our cancellation-aware method outperforms Wanda by **5.8× in PPL** (615 vs 3,545) by exploiting off-diagonal activation covariance structure.

The combination experiments reveal a fundamental incompatibility: **each correction method works best with its own scoring criterion.** OBS correction is derived from the same objective as SparseGPT's `w²/H_inv[j,j]` score; mixing in our cancellation scores degrades performance (hybrid: 958, interleaved: 856 — both worse than plain SparseGPT). Conversely, our batch Cholesky correction works well with our greedy scores (615). The correct path forward is deriving a cancellation-aware score *under the OBS objective* — the marginal increase in `w_S^T [H_inv[S,S]]⁻¹ w_S` — which would make selection and correction coherent within the same framework.

## Sparsity sweep (PPL vs sparsity level)

WikiText-2 PPL across sparsity levels. Dense baseline PPL: 19.5.

| Sparsity | SparseGPT | Greedy + correction (ours) | Wanda |
|----------|-----------|---------------------------|-------|
| 30% | **20.5** | 32.6 | 34.8 |
| 40% | **22.4** | 84.3 | 189.6 |
| 50% | **29.6** | 615 | 3,545 |
| 60% | **75.5** | 3,507 | 11,319 |
| 70% | 7,591 | **9,065** | 24,670 |
| 80% | 25,929 | 17,077 | **11,076** |

Our method consistently outperforms Wanda up to 70% sparsity. SparseGPT dominates up to 60% due to iterative OBS corrections; all methods collapse past 70%. Full sweep results in `results/sparsity_sweep.json`.

## Model weights

Dense and pruned model weights are available on the Hugging Face Hub under [`ikimyaii`](https://huggingface.co/ikimyaii).
