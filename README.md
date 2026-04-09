# Sparse LLMs on Loihi — Pruning Experiments

This repository contains post-training weight pruning experiments targeting neuromorphic hardware (Intel Loihi), where unstructured weight sparsity directly reduces active computation rather than being merely a compression artefact.

## Scripts

| File | Description |
|------|-------------|
| `scripts/prune.py` | **Baseline: kurtosis-conditioned diagonal pruning.** Scores each input channel as `‖W[:,i]‖² × E[x_i²]` (Gaussian layers) or `‖W[:,i]‖² × E[x_i⁴]^½` (heavy-tailed layers), switching on per-layer excess kurtosis. Global channel threshold, unstructured. |
| `scripts/prune_quadratic.py` | **Greedy covariance pruning (no weight correction).** For each output neuron, greedily selects the k channels minimising the joint reconstruction error `w_S^T Σ_X[S,S] w_S` using the full activation second-moment matrix. Semi-structured (constant k per row). |
| `scripts/prune_cancellation.py` | **Greedy covariance pruning + closed-form weight correction.** Same greedy selection as above, followed by a least-squares update to the remaining weights: `Δw[K] = Σ_X[K,K]⁻¹ Σ_X[K,S] w[S]`. Rows sharing the same prune mask are batched for efficiency. |
| `scripts/benchmark_fla.py` | Runs `lm-evaluation-harness` on a model with FLA model-type registration (required for the `transformer` architecture). Evaluates HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, PIQA, LAMBADA. |

## Results (1B transformer, 50% sparsity, no fine-tuning)

All models evaluated zero-shot on `lm-evaluation-harness`. PPL on WikiText-2 test set.

| Model | PPL (WikiText-2) | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | Avg Acc |
|-------|-----------------|-------|-------|-----------|------|------------|---------|
| `transformer-1B-dense-baseline` | 19.5 | 0.610 | 0.268 | 0.366 | 0.683 | 0.525 | **0.472** |
| `transformer-1B-dense-baseline-continued` | — | 0.631 | 0.289 | 0.373 | 0.687 | 0.517 | **0.484** |
| Diagonal pruning (`prune.py`) | 20,491 | 0.257 | 0.240 | 0.257 | 0.521 | 0.500 | 0.355 |
| Greedy covariance (`prune_quadratic.py`) | 1,103 | 0.279 | 0.216 | 0.267 | 0.535 | 0.476 | 0.355 |
| Greedy + weight correction (`prune_cancellation.py`) | 614.84 | — | — | — | — | — | — |

**Key observations:** The greedy covariance method reduces post-pruning PPL by 18× vs the diagonal baseline (1,103 vs 20,491), confirming that the off-diagonal structure of Σ_X carries meaningful signal. Weight correction is expected to further close the gap with the dense baseline by compensating pruned channels via a closed-form least-squares update to remaining weights. Task accuracy results to be updated after current run completes.

## Model weights

Dense and pruned model weights are available on the Hugging Face Hub under [`ikimyaii`](https://huggingface.co/ikimyaii).
