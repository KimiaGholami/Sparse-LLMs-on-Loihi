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
| `scripts/prune_obs_cancel.py` | **OBS-cancel (proposed method).** Greedy selection via OBS residual updates: score `r_j²/d_j` where `r_j` and `d_j` are residual weight and diagonal evolved by Schur complement rank-1 updates after each greedy step. At step 0 recovers SparseGPT's score exactly; subsequent steps capture cross-weight cancellation within the OBS objective. Followed by column-ordered OBS corrections identical to SparseGPT. |
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
| Hybrid: cancellation selection + OBS correction (`prune_hybrid.py`) | 958 | 0.279 | 0.208 | 0.265 | 0.546 | 0.470 | 0.354 |
| Interleaved: block-level cancellation + OBS (`prune_interleaved.py`) | 856 | 0.285 | 0.238 | 0.257 | 0.534 | 0.492 | 0.361 |
| **OBS-cancel (`prune_obs_cancel.py`)** | **24.1** | 0.569 | 0.248 | 0.345 | 0.665 | 0.517 | **0.469** |

**Key observations:** Our proposed **OBS-cancel** method achieves PPL **24.1** at 50% sparsity, outperforming SparseGPT (29.6) by **1.23×** with identical OBS corrections — the gain comes entirely from better prune mask selection.

The combination experiments reveal why naive mixtures fail: **each correction method works best with its own scoring criterion.** OBS correction is derived from the same objective as SparseGPT's `w²/H_inv[j,j]` score; mixing in our Σ_X-based cancellation scores degrades performance (hybrid: 958, interleaved: 856). The fix is to derive the cancellation-aware score *within the OBS objective*: the greedy marginal `δ(j|S') = r_j²/d_j` where `r_j` and `d_j` evolve via Schur complement rank-1 updates. At step 0 this recovers SparseGPT exactly; subsequent steps capture cross-weight cancellation that SparseGPT's diagonal score misses.

## Sparsity sweep (PPL vs sparsity level)

WikiText-2 PPL across sparsity levels. Dense baseline PPL: 19.5.

| Sparsity | OBS-cancel (ours) | SparseGPT | Greedy + correction | Wanda |
|----------|------------------|-----------|---------------------|-------|
| 30% | **20.2** | 20.5 | 32.6 | 34.8 |
| 40% | **21.2** | 22.4 | 84.3 | 189.6 |
| 50% | **24.1** | 29.6 | 615 | 3,545 |
| 60% | **47.0** | 75.5 | 3,507 | 11,319 |
| 70% | **4,365** | 7,591 | 9,065 | 24,670 |
| 80% | **14,177** | 25,929 | 17,077 | 11,076 |

OBS-cancel outperforms SparseGPT at every sparsity level. The margin grows with sparsity (1.02× at 30% → 1.60× at 60% → 1.74× at 70% → 1.83× at 80%), consistent with cancellation effects becoming more important as more weights are removed. Full sweep results in `results/sparsity_sweep.json`.

## LLaMA-7B results (50% sparsity)

Experiments on [open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b). PPL on WikiText-2 test set.

| Model | PPL (WikiText-2) |
|-------|-----------------|
| Dense baseline | 8.64 |
| SparseGPT (`prune_llama.py --method sparsegpt`) | **12.70** |
| OBS-cancel (`prune_llama.py --method obs_cancel`) | 26.93 |

**Observations:** Unlike the 1B model where OBS-cancel outperforms SparseGPT (24.1 vs 29.6), on LLaMA-7B OBS-cancel underperforms SparseGPT (26.93 vs 12.70). We attribute this to **numerical drift in the Schur complement residuals** for large layers. LLaMA-7B has in_features ∈ {4096, 11008}, requiring k ∈ {2048, 5504} greedy rank-1 update steps (vs. k=1024 in the 1B model). After 2000+ rank-1 updates to the residual diagonal D, floating-point drift causes D values to go negative and get clamped to 1e-8, distorting the OBS score `r_j²/d_j` for remaining weights. Investigation and numerical stabilisation of the greedy selection phase for large-layer models is ongoing.

## Model weights

Dense and pruned model weights are available on the Hugging Face Hub under [`ikimyaii`](https://huggingface.co/ikimyaii).
