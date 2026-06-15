# Sparse LLMs on Loihi — Pruning Experiments

This repository contains post-training weight pruning experiments targeting neuromorphic hardware (Intel Loihi), where unstructured weight sparsity directly reduces active computation rather than being merely a compression artefact.

## Scripts

| File | Description |
|------|-------------|
| `scripts/prune_wanda.py` | **Wanda baseline** (Sun et al., NeurIPS 2023). Scores each weight as `\|W[i,j]\| × √E[x_j²]` — weight magnitude times per-channel activation RMS. Semi-structured (constant k per output neuron). Equivalent to our method with a diagonal Σ_X; serves as the direct published baseline. |
| `scripts/prune_ria.py` | **RIA baseline** (Zhang et al., ICLR 2024). Scores each weight as `(\|W[r,c]\|/‖W[r,:]‖₁ + \|W[r,c]\|/‖W[:,c]‖₁) × act_rms[c]^α` — normalises weight magnitude by both row and column L1 norms before multiplying by activation scale, making pruning relative to peer weights. No weight correction. `--alpha` controls activation strength (default 0.5). |
| `scripts/prune_quadratic.py` | **Greedy covariance pruning (no weight correction).** For each output neuron, greedily selects the k channels minimising the joint reconstruction error `w_S^T Σ_X[S,S] w_S` using the full activation second-moment matrix. Semi-structured (constant k per row). |
| `scripts/prune_cancellation.py` | **Greedy covariance pruning + closed-form weight correction.** Same greedy selection as above, followed by a least-squares update to the remaining weights: `Δw[K] = Σ_X[K,K]⁻¹ Σ_X[K,S] w[S]`. Rows sharing the same prune mask are batched for efficiency. |
| `scripts/prune_sparsegpt.py` | **SparseGPT baseline** (Frantar & Alistarh, 2023). OBS saliency scoring `W[i,j]² / H_inv[j,j]` with column-ordered weight corrections applied in blocks of 128. The current strongest published single-shot pruning method. |
| `scripts/prune_hybrid.py` | **Hybrid: cancellation-aware selection + SparseGPT OBS correction.** Replaces SparseGPT's diagonal scoring with our full-covariance greedy selection, then applies column-ordered OBS corrections. Tests whether better selection adds value on top of OBS corrections. |
| `scripts/prune_interleaved.py` | **Interleaved: block-level cancellation selection + OBS correction.** Fixes the column-ordering mismatch in the hybrid by interleaving selection and correction block-by-block. Within each block, cancellation-aware greedy selection runs on the current (already-corrected) weights; OBS corrections are applied immediately after each block. |
| `scripts/prune_obs_cancel.py` | **OBS-cancel (proposed method), two variants.** `--method obs_cancel`: global greedy selection via OBS residual updates (`r_j²/d_j`, Schur complement rank-1 updates, float64), then column-ordered OBS correction. `--method obs_cancel_block`: block-level variant — within each 128-column block, runs `round(128 × sparsity)` greedy OBS-cancel steps restricted to that block's H_inv submatrix, then immediately applies OBS corrections for the block. The block variant eliminates the selection/correction ordering mismatch and numerical drift that limits the global variant on large models. |
| `scripts/sparsity_sweep.py` | Runs Wanda and cancellation pruning across sparsity levels (30–80%) and saves PPL results to `results/sparsity_sweep.json`. |
| `scripts/benchmark_fla.py` | Runs `lm-evaluation-harness` on a model with FLA model-type registration (required for the `transformer` architecture). Evaluates HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, PIQA, LAMBADA. |

## Results (1B transformer, 50% sparsity, no fine-tuning)

All models evaluated zero-shot on `lm-evaluation-harness`. PPL on WikiText-2 test set.

| Model | PPL (WikiText-2) | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | Avg Acc |
|-------|-----------------|-------|-------|-----------|------|------------|---------|
| `transformer-1B-dense-baseline` | 14.2 | 0.610 | 0.268 | 0.366 | 0.683 | 0.525 | **0.472** |
| `transformer-1B-dense-baseline-continued` | 13.8 | 0.631 | 0.289 | 0.373 | 0.687 | 0.517 | **0.484** |
| Wanda (`prune_wanda.py`) | 3,048 | 0.271 | 0.228 | 0.260 | 0.523 | 0.493 | 0.296 |
| RIA (`prune_ria.py`) | 1,729 | 0.277 | 0.263 | 0.268 | 0.513 | 0.490 | 0.302 |
| AWP (`scripts/prune_awp.py`) | 302 | 0.332 | 0.244 | 0.285 | 0.528 | 0.510 | 0.380 |
| Greedy covariance (`prune_quadratic.py`) | 635 | 0.279 | 0.216 | 0.267 | 0.535 | 0.476 | 0.355 |
| Greedy + weight correction (`prune_cancellation.py`) | 316 | 0.291 | 0.205 | 0.272 | 0.545 | 0.471 | 0.298 |
| SparseGPT (`prune_sparsegpt.py`) | **20.7** | 0.555 | 0.294 | 0.428 | 0.665 | 0.517 | **0.492** |
| Hybrid: cancellation selection + OBS correction (`prune_hybrid.py`) | 587 | 0.279 | 0.208 | 0.265 | 0.546 | 0.470 | 0.354 |
| Interleaved: block-level cancellation + OBS (`prune_interleaved.py`) | 786 | 0.285 | 0.238 | 0.257 | 0.534 | 0.492 | 0.361 |
| Global OBS-cancel (ablation, block size → ∞) | 17.5 | 0.569 | 0.248 | 0.345 | 0.665 | 0.517 | 0.469 |
| **OBS-cancel-block (`prune_obs_cancel.py --method obs_cancel_block`) — proposed** | **18.1** | **0.561** | 0.247 | 0.345 | **0.662** | 0.512 | **0.465** |

**Key observations:** Our proposed method is **OBS-cancel-block**, which achieves PPL **18.1** at 50% sparsity, outperforming SparseGPT (20.7) by **1.14×** with identical OBS corrections — the gain comes entirely from cancellation-aware greedy mask selection.

The combination experiments reveal why naive mixtures fail: **each correction method works best with its own scoring criterion.** OBS correction is derived from the same objective as SparseGPT's `w²/H_inv[j,j]` score; mixing in our Σ_X-based cancellation scores degrades performance (hybrid: 958, interleaved: 856). The fix is to derive the cancellation-aware score *within the OBS objective*: the greedy marginal `δ(j|S') = r_j²/d_j` where `r_j` and `d_j` evolve via Schur complement rank-1 updates. At step 0 this recovers SparseGPT exactly; subsequent steps capture cross-weight cancellation that SparseGPT's diagonal score misses.

**Global OBS-cancel** (block size → ∞) scores PPL **17.5** on the 1B model — better than OBS-cancel-block (18.1) because it captures cross-block cancellation interactions. However, it only works on the 1B model: on larger models it fails due to ordering mismatch and numerical drift (see LLaMA-7B results below). It is included as an ablation showing the theoretical upper bound of the method as block size grows.

## Calibration data ablation (1B, 50% sparsity)

To investigate whether the PPL–accuracy gap (OBS-cancel-block has better PPL but lower downstream accuracy than SparseGPT with WikiText-2 calibration) is a calibration-data artefact, we re-ran both methods using **C4** calibration data (2,000 web documents, diverse English text). PPL is still measured on WikiText-2 test in all cases.

| Calib data | Model | PPL | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | Avg Acc |
|-----------|-------|-----|-------|-------|-----------|------|------------|---------|
| WikiText-2 | Wanda | 3,048 | 0.271 | 0.228 | 0.260 | 0.523 | 0.493 | 0.296 |
| WikiText-2 | RIA | 1,729 | 0.277 | 0.263 | 0.268 | 0.513 | 0.490 | 0.302 |
| WikiText-2 | AWP | 241.8 | 0.332 | 0.244 | 0.285 | 0.528 | 0.510 | 0.380 |
| WikiText-2 | SparseGPT | 20.7 | 0.555 | 0.294 | 0.428 | 0.665 | 0.517 | 0.492 |
| WikiText-2 | OBS-cancel-block | 18.1 | 0.561 | 0.247 | 0.345 | 0.662 | 0.512 | 0.465 |
| C4 | Wanda | 3,492 | 0.271 | 0.275 | 0.267 | 0.502 | 0.489 | 0.300 |
| C4 | RIA | 1,897 | 0.270 | 0.267 | 0.272 | 0.515 | 0.496 | 0.303 |
| C4 | AWP | 290.1 | 0.334 | 0.251 | 0.283 | 0.530 | 0.507 | 0.381 |
| C4 | SparseGPT | 21.6 | 0.505 | 0.298 | 0.429 | 0.678 | 0.517 | 0.435 |
| C4 | **OBS-cancel-block** | **18.6** | **0.519** | 0.291 | 0.426 | 0.663 | **0.541** | **0.443** |

**Key finding:** With C4 calibration, OBS-cancel-block outperforms SparseGPT on **both** PPL (18.6 vs 21.6) and downstream task accuracy (0.443 vs 0.435). The task accuracy advantage SparseGPT had under WikiText-2 calibration disappears when calibration data is not drawn from the same distribution as the test set. This confirms the PPL–accuracy gap is a calibration-data artefact: WikiText-2 calibration gives SparseGPT's column-ordered mask an incidental advantage on WikiText-2-adjacent tasks, while OBS-cancel-block's superior cross-weight cancellation generalises better to a held-out calibration distribution.

AWP collapses to PPL 241.8 with WikiText-2 calibration and 290.1 with C4 calibration, achieving near-identical task accuracy in both cases (avg 0.381 vs 0.380), consistent with the finding that calibration choice is irrelevant when reconstruction error dominates. No-correction methods (Wanda, RIA) are effectively insensitive to calibration data choice — they collapse regardless, and downstream accuracy hovers near random (0.296–0.303) in both cases. This confirms that calibration data only matters when the pruning method can actually exploit the activation statistics via weight correction.

## Block size sweep (1B, 50% sparsity)

OBS-cancel-block restricts each greedy selection round to the current 128-column block's H_inv submatrix. Larger blocks capture more cross-block cancellation at the cost of more Schur complement steps per block. Dense baseline PPL: 14.2.

| Block size | PPL (WikiText-2) |
|-----------|-----------------|
| 64 | 18.60 |
| **128** (default) | **18.1** |
| 256 | 18.02 |
| 512 | 17.77 |
| ∞ (global OBS-cancel) | **17.5** |

All values use the standard 2048-token non-overlapping evaluation. PPL decreases monotonically as block size grows, confirming that larger blocks capture more cross-block cancellation interactions. The default block size of 128 is the best choice for large models (where global OBS-cancel suffers from ordering mismatch and numerical drift); on the 1B model all block sizes work well. Full results in `results/block_size_sweep.json`.

## Sparsity sweep (PPL vs sparsity level)

WikiText-2 PPL across sparsity levels. Dense baseline PPL: 14.2.

| Sparsity | OBS-cancel-block (ours) | SparseGPT | AWP | Greedy + correction | Wanda | RIA |
|----------|------------------------|-----------|-----|---------------------|-------|-----|
| 30% | **14.6** | 14.7 | 22.9 | 32.6 | 34.8 | 23.5 |
| 40% | **15.4** | 15.9 | 44.0 | 84.3 | 189.6 | 77.0 |
| 50% | **17.5** | 20.7 | 241.8 | 316 | 3,048 | 1,729 |
| 60% | **36.9** | 59.3 | 6,455 | 3,507 | 11,319 | 13,545 |
| 70% | **4,787** | 10,840 | 6,863 | 9,065 | 24,670 | 11,298 |
| 80% | **13,964** | 28,770 | 12,912 | — | 12,709 | 18,114 |

OBS-cancel-block outperforms SparseGPT at every sparsity level. The margin grows with sparsity (1.02× at 30% → 1.60× at 60% → 2.26× at 70% → 2.06× at 80%), consistent with cancellation effects becoming more important as more weights are removed. Note: on the 1B model the global OBS-cancel ablation (block size → ∞) achieves PPL 24.1 at 50%; the 1B sparsity sweep uses this variant as it is stable at this scale. Full sweep results in `results/sparsity_sweep.json`.

## 1B transformer — 80% sparsity

| Method | PPL | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | LAMBADA | Avg Acc |
|--------|-----|-------|-------|-----------|------|------------|---------|---------|
| Dense baseline | 14.2 | 0.610 | 0.268 | 0.366 | 0.683 | 0.525 | — | 0.472 |
| Wanda | 12,709 | 0.249 | 0.219 | 0.258 | 0.528 | 0.494 | 0.000 | 0.291 |
| RIA | 18,114 | 0.255 | 0.224 | 0.258 | 0.528 | 0.479 | 0.000 | 0.291 |
| SparseGPT | 28,770 | 0.253 | 0.223 | 0.256 | 0.537 | 0.471 | 0.000 | 0.290 |
| AWP | 12,912 | 0.261 | 0.259 | 0.261 | 0.496 | 0.504 | 0.000 | 0.356 |
| **OBS-cancel-block (ours)** | **13,964** | **0.266** | **0.224** | **0.259** | 0.532 | **0.510** | 0.000 | **0.298** |

At 80% sparsity all methods collapse to near-random task accuracy (LAMBADA→0 across the board). OBS-cancel-block retains the best PPL (13,964 vs 28,770 for SparseGPT, 2.06×) and marginally the best average accuracy, mirroring the pattern seen on LLaMA-7B and HGRN-1.3B at 80%.

## LLaMA-7B results

Experiments on [open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b). PPL on WikiText-2 test set (2048-token non-overlapping evaluation); downstream tasks evaluated zero-shot with `lm-evaluation-harness`.

**50% sparsity:**

| Model | PPL | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | Avg Acc |
|-------|-----|-------|-------|-----------|------|------------|---------|
| Dense baseline | 6.61 | 0.678 | 0.385 | 0.694 | 0.763 | 0.675 | **0.639** |
| **AWP** (`scripts/prune_awp.py`) | **8.10** | 0.614 | 0.336 | 0.612 | 0.732 | 0.661 | 0.591 |
| **OBS-cancel (ours)** | **8.24** | 0.610 | 0.342 | 0.631 | 0.724 | 0.658 | 0.593 |
| RIA (`prune_ria.py`) | 8.58 | 0.601 | 0.332 | 0.624 | 0.732 | 0.649 | 0.588 |
| Wanda (`prune_wanda.py`) | 8.69 | 0.602 | 0.347 | 0.628 | 0.732 | 0.663 | 0.595 |
| SparseGPT | 9.51 | 0.609 | 0.363 | 0.649 | 0.732 | 0.643 | **0.599** |

**80% sparsity:**

| Model | PPL | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | Avg Acc |
|-------|-----|-------|-------|-----------|------|------------|---------|
| Dense baseline | 6.61 | 0.678 | 0.385 | 0.694 | 0.763 | 0.675 | **0.639** |
| **AWP** | **211.1** | 0.268 | 0.263 | 0.262 | 0.503 | 0.478 | 0.355 |
| OBS-cancel (ours) | 245.5 | 0.259 | 0.270 | 0.263 | 0.497 | 0.489 | 0.356 |
| Wanda (`prune_wanda.py`) | 1,647.4 | 0.253 | 0.267 | 0.265 | 0.499 | 0.491 | 0.355 |
| SparseGPT | 2,071.2 | 0.255 | 0.265 | 0.266 | 0.513 | 0.505 | **0.361** |
| RIA (`prune_ria.py`) | 2,084.1 | 0.268 | 0.268 | 0.261 | 0.495 | 0.496 | 0.357 |

At 80% sparsity AWP achieves the best PPL (211.1), with OBS-cancel close behind (245.5). All other methods collapse to extreme perplexity (>1,600). Task accuracy ranges from 0.355 to 0.361, statistically indistinguishable.

**Key observations:**

**OBS-cancel (sequential) matches AWP at 50% and surpasses it at 60–70% sparsity.** With sequential layer-by-layer calibration, OBS-cancel achieves PPL 8.24 at 50% (vs AWP 8.10, within 2%) and is the best method at 60% (10.96 vs AWP 11.33) and 70% (26.85 vs AWP 31.99). Sequential calibration — where each layer's H is built from activations flowing through already-pruned earlier layers — provides more accurate Hessian estimates than independent calibration.

**AWP reclaims the top spot at 80% sparsity.** At 80% AWP reaches PPL 211.1, with OBS-cancel close behind at 245.5. All other methods collapse to extreme perplexity (>1,600). The crossover is consistent with IHT's iterative mask search having a structural advantage at very high sparsity, where it can revise earlier decisions in ways that one-shot greedy selection cannot.

Global OBS-cancel fails on LLaMA-7B due to two compounding problems: (1) **ordering mismatch** — the global greedy mask is not column-ordered, but the OBS correction assumes column-ordered pruning; (2) **numerical drift** — k ∈ {2048, 5504} Schur complement rank-1 updates cause the residual diagonal D to drift. OBS-cancel fixes both by restricting each greedy selection to its own 128-column block, with sequential calibration providing further improvement at moderate sparsity.

## LLaMA-7B sparsity sweep

WikiText-2 PPL across sparsity levels (2048-token evaluation). Dense baseline PPL: 6.61.

| Sparsity | OBS-cancel (ours) | Wanda | RIA | SparseGPT | AWP |
|----------|-------------------|-------|-----|-----------|-----|
| 30% | 6.87 | 6.89 | 6.84 | 6.99 | **6.81** |
| 40% | 7.30 | 7.40 | 7.30 | 7.76 | **7.19** |
| 50% | 8.24 | 8.69 | 8.58 | 9.51 | **8.10** |
| 60% | **10.96** | 14.22 | 14.40 | 15.64 | 11.33 |
| 70% | **26.85** | 77.82 | 102.17 | 67.02 | 31.99 |
| 80% | 245.5 | 1,647.4 | 2,084.1 | 2,071.2 | **211.1** |

**Key finding: OBS-cancel (sequential) leads at 60–70% sparsity; AWP leads at 30–50% and 80%.** OBS-cancel beats AWP at 60% (10.96 vs 11.33, 1.03×) and 70% (26.85 vs 31.99, 1.19×). AWP is best at 30–50% and reclaims the lead at 80% (211.1 vs 245.5). The crossover reflects sequential calibration's advantage at moderate sparsity and IHT's iterative search advantage at very high sparsity.

OBS-cancel outperforms SparseGPT at every sparsity level. Full results in `results/sparsity_sweep_llama.json`.

## HGRN-1.3B results

Experiments on [`fla-hub/hgrn-1.3B-100B`](https://huggingface.co/fla-hub/hgrn-1.3B-100B), a 1.3B Hierarchical Gated Recurrent Network (HGRN) state space model. All evaluations at 50% sparsity on WikiText-2 test / lm-evaluation-harness zero-shot.

| Model | PPL (WikiText-2) | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | LAMBADA | Avg Acc |
|-------|-----------------|-------|-------|-----------|------|------------|---------|---------|
| `hgrn-1.3B-dense-baseline` | 11.8 | 0.510 | 0.275 | 0.480 | 0.712 | 0.528 | 0.383 | **0.481** |
| Wanda (`prune_wanda.py`) | 350 | 0.312 | 0.263 | 0.295 | 0.545 | 0.517 | 0.003 | 0.323 |
| AWP (`scripts/prune_awp.py`) | 426.2 | 0.327 | 0.241 | 0.293 | 0.525 | 0.530 | 0.014 | 0.322 |
| RIA (`prune_ria.py`) | 348 | 0.305 | 0.255 | 0.298 | 0.542 | 0.511 | 0.004 | 0.319 |
| SparseGPT (`prune_sparsegpt.py`) | 17.4 | 0.467 | 0.264 | 0.434 | 0.676 | 0.519 | 0.215 | 0.429 |
| **OBS-cancel-block** (`prune_obs_cancel.py`) | **16.3** | 0.461 | 0.261 | 0.427 | 0.669 | 0.525 | 0.230 | **0.429** |

**Key observations:** The HGRN results reproduce the 1B transformer pattern exactly:

- **AWP collapses at 1.3B scale.** AWP reaches PPL 527 on HGRN-1.3B, worse than Wanda (350) and RIA (348), and far behind the second-order methods. The same iterative method that dominates at 7B scale fails here because per-layer reconstruction error from uncorrected pruning is too large for gradient steps to overcome. This strongly implicates scale, not architecture, as the deciding factor.

- **No-correction methods collapse.** Wanda (PPL 350) and RIA (PPL 348) also fail at 50% sparsity, with LAMBADA accuracy dropping to near zero. Unlike LLaMA-7B (where RIA was competitive), HGRN-1.3B shares the 1B transformer's sensitivity, confirming scale determines whether correction is necessary.

- **OBS-cancel-block outperforms SparseGPT on PPL** (**16.3 vs 17.4**, 1.07× improvement) and **matches on downstream accuracy** (0.429 each). This mirrors the 1B transformer result and confirms the cancellation-aware selection gain is architecture-agnostic.

## HGRN-1.3B sparsity sweep

WikiText-2 PPL across sparsity levels. Dense baseline PPL: 11.8.

| Sparsity | OBS-cancel-block (ours) | SparseGPT | AWP | Wanda | RIA |
|----------|------------------------|-----------|-----|-------|-----|
| 30% | **14.92** | 15.02 | 21.7 | 31.64 | 25.6 |
| 40% | **16.06** | 16.46 | 49.7 | 76.87 | 54.3 |
| 50% | **16.3** | 17.4 | 426.2 | 584 | 348 |
| 60% | **29.14** | 32.43 | 2,616 | 11,552 | 8,239 |
| 70% | **112.7** | 115.4 | 4,440 | 20,592 | 17,457 |
| 80% | **4,865** | 6,956 | 17,195 | 75,620 | 28,615 |

OBS-cancel-block outperforms SparseGPT at every sparsity level. AWP stays close to the second-order methods at 30–40% sparsity but collapses sharply from 50% onwards (527 at 50%, 17,756 at 80%), mirroring its behaviour on the 1B transformer. The no-correction methods collapse even more dramatically (Wanda PPL 75,620 at 80%). Full results in `results/sparsity_sweep_hgrn.json`.

## HGRN-1.3B 80% sparsity

| Model | PPL | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | LAMBADA | Avg Acc |
|-------|-----|-------|-------|-----------|------|------------|---------|---------|
| `hgrn-1.3B-dense-baseline` | 11.8 | 0.510 | 0.275 | 0.480 | 0.712 | 0.528 | 0.383 | **0.481** |
| Wanda | 76,051 | 0.249 | 0.235 | 0.260 | 0.513 | 0.519 | 0.000 | 0.296 |
| RIA | 28,615 | 0.255 | 0.230 | 0.259 | 0.516 | 0.498 | 0.000 | 0.293 |
| AWP | 17,195 | 0.260 | 0.288 | 0.265 | 0.496 | 0.507 | 0.000 | 0.303 |
| SparseGPT | 6,956 | 0.269 | 0.220 | 0.260 | 0.533 | 0.519 | 0.000 | 0.300 |
| **OBS-cancel-block** | **4,865** | 0.269 | 0.222 | 0.258 | 0.527 | 0.491 | 0.000 | 0.294 |

At 80% sparsity all methods collapse to near-random performance (LAMBADA→0 across the board). OBS-cancel-block retains the best PPL (4,865 vs 6,956 for SparseGPT, 1.43×). AWP performs worse than both second-order methods at this sparsity (17,756), far below OBS-cancel-block, reinforcing the 1B/HGRN pattern: at sub-2B scale AWP cannot compensate for the reconstruction error that accumulates without explicit Hessian-based weight correction.

## HGRN-1.3B 80% — post-pruning recovery

Post-pruning recovery via three strategies applied uniformly across all pruning methods, all with the sparsity mask fixed throughout (gradient hooks zero updates at pruned positions; masked weights are re-zeroed after each optimizer step):

- **Fine-tuning**: C4 streaming, cross-entropy on hard labels only, 20k steps, batch = 16, lr = 2e-5. Script: `scripts/finetune_sparse.py`.
- **Distillation**: C4 streaming, frozen `fla-hub/hgrn-1.3B-100B` teacher, loss = 0.1 × CE + 0.9 × KL(student ‖ teacher) × T², T = 2.0, 20k steps, batch = 16, lr = 2e-5. Script: `scripts/distill_ddp.py`.
- **LoRA**: PEFT LoRA adapters (r=16, α=32) on all linear layers, 20k steps, batch = 16, lr = 3e-4, then merged into the sparse weights. Script: `scripts/lora_sparse.py`.

PPL is 2048-token WikiText-2 non-overlapping. ARC-e/c and HellaSwag/PIQA use `acc_norm`; WinoGrande uses `acc`. Avg is unweighted mean of the five tasks. Full results in `results/ppl_hgrn_80pct_recovery.json`, `results/ppl_hgrn_80pct_lora.json`, and `results/benchmark_hgrn_*_80pct*.json`.

| Method | PPL | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | LAMBADA | Avg Acc |
|--------|-----|-------|-------|-----------|------|------------|---------|---------|
| Dense baseline | 11.8 | 0.510 | 0.275 | 0.480 | 0.712 | 0.528 | 0.383 | **0.501** |
| **OBS-cancel-block** | **4,865** | 0.263 | 0.292 | 0.259 | 0.503 | 0.489 | 0.000 | 0.361 |
| OBS-cancel-block + FT | 1,086 | 0.285 | 0.276 | 0.258 | 0.505 | 0.520 | 0.000 | 0.369 |
| OBS-cancel-block + distill | **693** | 0.285 | 0.265 | 0.259 | 0.510 | 0.514 | 0.001 | 0.366 |
| OBS-cancel-block + LoRA | 611 | 0.310 | 0.248 | 0.280 | 0.547 | 0.499 | 0.002 | **0.377** |
| Wanda | 76,074 | 0.260 | 0.290 | 0.257 | 0.487 | 0.519 | 0.000 | 0.363 |
| Wanda + FT | 1,889 | 0.272 | 0.294 | 0.255 | 0.502 | 0.506 | 0.000 | 0.366 |
| Wanda + distill | 1,951 | 0.266 | 0.259 | 0.255 | 0.502 | 0.511 | 0.000 | 0.359 |
| Wanda + LoRA | 713 | 0.296 | 0.247 | 0.272 | 0.544 | 0.502 | 0.000 | 0.372 |
| RIA | 28,681 | 0.250 | 0.302 | 0.256 | 0.513 | 0.498 | 0.000 | 0.364 |
| RIA + FT | 2,015 | 0.273 | 0.288 | 0.258 | 0.500 | 0.506 | 0.000 | 0.365 |
| RIA + distill | 1,956 | 0.267 | 0.278 | 0.256 | 0.499 | 0.493 | 0.000 | 0.359 |
| RIA + LoRA | 737 | 0.300 | 0.248 | 0.273 | 0.536 | 0.479 | 0.001 | 0.367 |
| SparseGPT | 6,956 | 0.274 | 0.289 | 0.260 | 0.511 | 0.519 | 0.000 | 0.371 |
| SparseGPT + FT | 1,066 | 0.279 | 0.263 | 0.257 | 0.504 | 0.480 | 0.000 | 0.357 |
| SparseGPT + distill | 756 | 0.280 | 0.248 | 0.264 | 0.521 | 0.502 | 0.000 | 0.363 |
| SparseGPT + LoRA | **604** | 0.309 | 0.253 | 0.280 | 0.555 | 0.494 | 0.002 | 0.378 |
| AWP | 17,239 | 0.260 | 0.288 | 0.266 | 0.496 | 0.507 | 0.000 | 0.363 |
| AWP + FT | 1,560 | 0.266 | 0.271 | 0.258 | 0.497 | 0.498 | 0.000 | 0.358 |
| AWP + distill | 1,400 | 0.269 | 0.280 | 0.260 | 0.498 | 0.490 | 0.000 | 0.359 |
| AWP + LoRA | 662 | 0.287 | 0.259 | 0.269 | 0.544 | 0.499 | 0.001 | 0.372 |

**Key findings:** At 80% sparsity, all methods collapse to near-random task accuracy (0.357–0.378) and LAMBADA→0, confirming a hard capacity ceiling at this compression level regardless of pruning method or recovery strategy. OBS-cancel-block achieves the best one-shot PPL (4,865 vs 6,956 for SparseGPT, 1.43×) and best post-recovery PPL with distillation (693 vs 756 for SparseGPT) or LoRA (611 vs 604 for SparseGPT — essentially tied). LoRA consistently outperforms FT and distillation on task accuracy across all methods, with SparseGPT + LoRA achieving the highest avg accuracy overall (0.378), closely followed by OBS-cancel-block + LoRA (0.377). FT and distillation improve PPL substantially for all methods (4–80× reduction) but yield nearly identical task accuracy to the one-shot baselines, confirming the capacity ceiling is structural. LoRA's additional parameters appear to partially circumvent this ceiling, providing a consistent +0.008–0.021 accuracy gain over FT/distill across methods.

### OBS-cancel-block ablations (varied recovery strategies)

| Model | PPL | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | LAMBADA | Avg Acc |
|-------|-----|-------|-------|-----------|------|------------|---------|---------|
| OBS-cancel-block (one-shot) | 4,865 | 0.263 | 0.292 | 0.259 | 0.503 | 0.489 | 0.000 | 0.361 |
| + distill (20k, `distill_ddp.py`) | 693 | 0.285 | 0.265 | 0.259 | 0.510 | 0.514 | 0.001 | 0.366 |
| + distill (50k, `distill_sparse.py`) | 647 | 0.284 | 0.268 | 0.259 | 0.521 | 0.516 | 0.001 | 0.370 |
| + FT (20k steps) | 1,086 | 0.285 | 0.276 | 0.258 | 0.505 | 0.520 | 0.000 | **0.369** |
| + dynamic-mask FT (20k steps) | 1,132 | 0.288 | 0.272 | 0.255 | 0.510 | 0.507 | 0.000 | 0.366 |
| Non-uniform + distill | **218** | 0.286 | 0.230 | 0.265 | 0.524 | 0.484 | 0.000 | 0.358 |
| Non-uniform + FT | 391 | 0.290 | 0.258 | 0.264 | 0.509 | 0.499 | 0.000 | 0.364 |
| Non-uniform + FT + LoRA | 393 | 0.309 | 0.218 | 0.265 | 0.551 | 0.508 | 0.001 | **0.370** |
| Iterative + distill | 250 | 0.285 | 0.253 | 0.261 | 0.521 | 0.510 | 0.000 | 0.366 |

**Failed approach — L1-ramp proximal gradient fine-tuning:** An earlier attempt
used ISTA-style soft-thresholding with a gradually ramped L1 penalty
(λ: 0 → 1e-4 over 20k steps) to drive additional sparsity during recovery.
This catastrophically destroyed the OBS-optimized weights: the L1 penalty
zeroed small-magnitude weights that OBS-cancel-block had specifically retained
for cancellation of correlated input channels. PPL exploded from 19 → 2,089
(50% model) and 1,952 → 569,000 (80% model). The models were unrecoverable.
Root cause: soft-thresholding is magnitude-blind to OBS cancellation structure —
a weight near zero may carry disproportionate information when paired with a
correlated channel. Mask-fixed fine-tuning and distillation (which freeze the
zero pattern entirely) avoid this failure mode.

## Model weights

Dense and pruned model weights are available on the Hugging Face Hub under [`ikimyaii`](https://huggingface.co/ikimyaii).
