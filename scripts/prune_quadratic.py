"""
Cancellation-aware semi-structured pruning via activation covariance quadratic form.

For each output neuron j of a Linear layer with weight row w_j and input x:

    E[y_j^2] = w_j^T Sigma_X w_j

where Sigma_X = E[x x^T] is the uncentered activation second-moment matrix.

The marginal contribution of input channel i to output neuron j is:

    score(j, i) = E[y_j^2] - E[(y_j - w_{ji} x_i)^2]
                = 2 * w_{ji} * (Sigma_X w_j)[i]  -  w_{ji}^2 * Sigma_X[i,i]

Key distinction from diagonal methods (Wanda, magnitude):
  - When Sigma_X is diagonal, score(j,i) = w_{ji}^2 * Sigma_X[i,i]  (recovers Wanda)
  - When Sigma_X has positive off-diagonal entries and w_j has mixed-sign weights
    on correlated channels, the cross terms make score(j,i) NEGATIVE — identifying
    cancellation groups that diagonal methods cannot detect.

Pruning procedure:
  1. Collect Sigma_X = E[x x^T] per layer from calibration data (on CPU, float32)
  2. For each layer, compute scores[j,i] = 2*W[j,i]*V[j,i] - W[j,i]^2*diag(Sigma_X)[i]
     where V = W @ Sigma_X
  3. Semi-structured: for each output neuron (row), zero the k lowest-scoring inputs
     → exactly constant sparsity per neuron

Usage
-----
python scripts/prune_quadratic.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-quadratic-pruned-50pct \\
    --eval_ppl
"""

import argparse
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Register FLA model types
try:
    import fla  # noqa
    from fla.models.transformer import TransformerConfig, TransformerForCausalLM
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: could not register FLA transformer type: {e}")


# ---------------------------------------------------------------------------
# Activation second-moment accumulator
# ---------------------------------------------------------------------------

class CovarianceStats:
    """
    Accumulates the uncentered second moment matrix Sigma_X = E[x x^T]
    for a single Linear layer's input, in float32 on CPU.

    For input tensor X of shape (..., in_features):
        sum_xx[i,j] = sum of x_i * x_j over all observations
        count        = number of scalar observations (vectors)
    """

    def __init__(self, in_features: int):
        self.sum_xx = torch.zeros(in_features, in_features, dtype=torch.float32)
        self.count = 0
        self.in_features = in_features

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """x: any shape (..., in_features)."""
        x = x.reshape(-1, x.shape[-1]).float().cpu()
        self.sum_xx += x.T @ x
        self.count += x.shape[0]

    def second_moment(self) -> torch.Tensor:
        """E[x x^T], shape (in_features, in_features)."""
        return self.sum_xx / max(self.count, 1)

    def diagonal(self) -> torch.Tensor:
        """Diagonal of E[x x^T] = E[x_i^2], shape (in_features,)."""
        return self.sum_xx.diagonal() / max(self.count, 1)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def build_calib_batches(tokenizer, n_batches, batch_size, seq_len, seed=42):
    """WikiText-2 validation split → list of (batch_size, seq_len) input_ids."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    rng = torch.Generator()
    rng.manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        start = torch.randint(0, enc.shape[0] - seq_len, (1,), generator=rng).item()
        chunk = enc[start: start + seq_len].unsqueeze(0).expand(batch_size, -1).clone()
        batches.append(chunk)
    return batches


@torch.no_grad()
def collect_covariance_stats(model: nn.Module, batches: list, device: torch.device) -> dict:
    """
    Register forward hooks on all Linear layers, run calibration batches,
    accumulate Sigma_X = E[x x^T] per layer on CPU.
    """
    stats = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        stats[name] = CovarianceStats(module.in_features)

        def make_hook(layer_name):
            def hook(mod, inp, out):
                stats[layer_name].update(inp[0].detach())
            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    for batch in tqdm(batches, desc="Collecting activations"):
        model(input_ids=batch.to(device))

    for h in hooks:
        h.remove()

    return stats


# ---------------------------------------------------------------------------
# Scoring and pruning
# ---------------------------------------------------------------------------

def compute_scores_and_prune(model: nn.Module, stats: dict, sparsity: float, device: torch.device):
    """
    For each Linear layer:
      1. Compute V = W @ Sigma_X  (captures off-diagonal correlations)
      2. Per-element score: scores[j,i] = 2*W[j,i]*V[j,i] - W[j,i]^2 * Sigma_X[i,i]
      3. Semi-structured: for each row j, zero the k = floor(in*sparsity) lowest-scoring channels

    Returns actual sparsity and a dict of per-layer statistics.
    """
    total_weights = 0
    total_pruned = 0
    layer_info = {}

    with torch.no_grad():
        for name, module in tqdm(model.named_modules(), desc="Scoring and pruning"):
            if not isinstance(module, nn.Linear) or name not in stats:
                continue

            W = module.weight.data  # (out_features, in_features)
            out_f, in_f = W.shape
            k = int(in_f * sparsity)          # channels to prune per output neuron
            if k == 0:
                continue

            # Load Sigma_X to the model device
            Sigma = stats[name].second_moment().to(device)  # (in_f, in_f)
            Sigma_diag = Sigma.diagonal()                    # (in_f,)

            W_f = W.float()
            # V[j,i] = (Sigma_X @ w_j)[i]  — vectorised over all output neurons
            V = W_f @ Sigma                                  # (out_f, in_f)

            # Marginal contribution of channel i to output neuron j
            scores = 2.0 * W_f * V - W_f ** 2 * Sigma_diag.unsqueeze(0)  # (out_f, in_f)

            # Semi-structured: prune k lowest-scoring inputs per output neuron
            _, prune_idx = scores.topk(k, dim=1, largest=False)
            mask = torch.ones(out_f, in_f, dtype=torch.bool, device=device)
            mask.scatter_(1, prune_idx, False)

            module.weight.data[~mask] = 0.0

            # Statistics
            n_neg = (scores < 0).sum().item()
            total_weights += out_f * in_f
            total_pruned += (~mask).sum().item()
            layer_info[name] = {
                "out_features": out_f,
                "in_features": in_f,
                "n_negative_scores": n_neg,
                "fraction_negative": n_neg / (out_f * in_f),
                "score_min": scores.min().item(),
                "score_max": scores.max().item(),
                "score_mean": scores.mean().item(),
            }

            # Free GPU memory
            del Sigma, V, scores, mask, W_f

    actual_sparsity = total_pruned / max(total_weights, 1)
    return actual_sparsity, layer_info


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model: nn.Module, tokenizer, device: torch.device,
                 seq_len: int = 512, n_tokens: int = 500_000) -> float:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    enc = enc[: min(n_tokens, enc.shape[0])]

    model.eval()
    nll_sum, token_count = 0.0, 0
    for start in tqdm(range(0, enc.shape[0] - seq_len, seq_len), desc="PPL eval"):
        chunk = enc[start: start + seq_len].unsqueeze(0).to(device)
        nll_sum += model(input_ids=chunk, labels=chunk).loss.item() * chunk.numel()
        token_count += chunk.numel()

    return math.exp(nll_sum / token_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Cancellation-aware semi-structured pruning via activation covariance"
    )
    p.add_argument("--model_path", type=str, default="exp/transformer-1B-dense-baseline")
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="Fraction of input channels to prune per output neuron (0–1)")
    p.add_argument("--n_calib_batches", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--eval_ppl", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters")

    ppl_before = None
    if args.eval_ppl:
        ppl_before = evaluate_ppl(model, tokenizer, device, args.seq_len)
        print(f"Perplexity before pruning: {ppl_before:.4f}")

    # --- calibration: accumulate Sigma_X per layer on CPU ---
    print(f"\nBuilding calibration batches "
          f"({args.n_calib_batches} × {args.batch_size} × {args.seq_len}) ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches, args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, device)

    total_sigma_mb = sum(
        s.sum_xx.numel() * 4 / 1e6 for s in stats.values()
    )
    print(f"  Sigma_X matrices accumulated: {len(stats)} layers, "
          f"{total_sigma_mb:.0f} MB total (CPU float32)")

    # --- score and prune ---
    print(f"\nPruning {args.sparsity * 100:.1f}% of input channels per neuron "
          f"(semi-structured, cancellation-aware) ...")
    actual_sparsity, layer_info = compute_scores_and_prune(model, stats, args.sparsity, device)
    print(f"Actual sparsity: {actual_sparsity * 100:.2f}%")

    # Print layers with the most negative-score channels (cancellation sites)
    top_cancel = sorted(layer_info.items(), key=lambda kv: -kv[1]["fraction_negative"])[:10]
    print(f"\nTop-10 layers by cancellation fraction (negative scores):")
    header = f"  {'Layer':<55}  {'Neg%':>6}  {'ScoreMin':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, info in top_cancel:
        print(f"  {name:<55}  {info['fraction_negative'] * 100:>5.1f}%  "
              f"{info['score_min']:>10.3e}")

    ppl_after = None
    if args.eval_ppl:
        ppl_after = evaluate_ppl(model, tokenizer, device, args.seq_len)
        print(f"\nPerplexity after pruning: {ppl_after:.4f}  "
              f"(Δ = {ppl_after - ppl_before:+.4f})")

    # --- save ---
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        summary = {
            "method": "cancellation_aware_quadratic",
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "sparsity_structure": "semi-structured (constant per output neuron)",
            "scoring": "quadratic form: 2*w_ji*(Sigma_X @ w_j)[i] - w_ji^2 * Sigma_X[i,i]",
            "n_calib_batches": args.n_calib_batches,
            "ppl_before": ppl_before,
            "ppl_after": ppl_after,
            "layer_info": layer_info,
        }
        with open(os.path.join(args.output_path, "pruning_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
