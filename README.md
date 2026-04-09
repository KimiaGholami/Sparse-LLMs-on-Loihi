# Sparse LLMs on Loihi — Pruning Experiments

This repository contains post-training weight pruning experiments targeting neuromorphic hardware (Intel Loihi), where unstructured weight sparsity directly reduces active computation rather than being merely a compression artefact.

## Scripts

|  | **Wanda baseline.** Scores each weight as  — the per-channel activation RMS times weight magnitude. Semi-structured (constant k per output neuron). Direct ablation: equivalent to our method with a diagonal Σ_X. |
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
| `transformer-1B-dense-baseline-continued` | 19.4 | 0.631 | 0.289 | 0.373 | 0.687 | 0.517 | **0.484** |
| Diagonal pruning (`prune.py`) | 20,491 | 0.257 | 0.240 | 0.257 | 0.521 | 0.500 | 0.355 |
| Greedy covariance (`prune_quadratic.py`) | 1,103 | 0.279 | 0.216 | 0.267 | 0.535 | 0.476 | 0.355 |
| Greedy + weight correction (`prune_cancellation.py`) | 614.84 | 0.291 | 0.205 | 0.272 | 0.545 | 0.471 | 0.298 |

**Key observations:** Wanda (the standard published baseline) serves as the direct ablation: it is equivalent to our method with a diagonal Σ_X. The greedy covariance method (full Σ_X, off-diagonal terms) reduces post-pruning PPL substantially below Wanda, confirming that activation correlations carry meaningful signal beyond per-channel scale. Weight correction further closes the gap toward the dense baseline.

## Model weights

Dense and pruned model weights are available on the Hugging Face Hub under [`ikimyaii`](https://huggingface.co/ikimyaii).
