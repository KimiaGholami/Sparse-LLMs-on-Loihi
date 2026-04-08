"""
Cancellation-aware semi-structured pruning via greedy minimisation of the
joint activation-covariance quadratic form.

Background
----------
When we zero out a subset S of input channels in row j of a Linear layer, the
expected squared reconstruction error introduced is:

    E[error_j^2] = E[(sum_{i in S} w_{ji} x_i)^2]
                 = w_S^T  Sigma_X[S,S]  w_S

where Sigma_X = E[x x^T] is the uncentered second-moment matrix of the layer's
inputs, estimated from calibration data.

This quantity is SMALLER than the sum of individual errors (sum_{i in S}
w_{ji}^2 Sigma_X[i,i]) whenever the pruned weights carry opposite signs on
correlated input dimensions — the "cancellation" structure identified in the
document.  Diagonal methods (Wanda, magnitude) miss this entirely.

Algorithm
---------
For each output neuron j we greedily build the pruned set S of k channels that
minimises  w_S^T Sigma_X[S,S] w_S  using the marginal-error greedy rule:

    Step 0: S = {},  v_j = 0  (v_j will track Sigma_X @ w_{S_j})
    Step t: pick  i* = argmin_i [ w_{ji}^2 * Sigma_X[i,i]
                                  + 2 * w_{ji} * v_j[i] ]
            (= marginal increase in joint error from adding i to S)
            add i* to S;  update v_j += Sigma_X[:, i*] * w_{ji*}

The cross-term  2 * w_{ji} * v_j[i]  goes NEGATIVE when channel i has opposite
sign to previously selected channels that are correlated with i — so the
algorithm actively seeks cancelling pairs, triplets, etc.

At step 0 (empty S) this recovers the Wanda/diagonal criterion exactly.

Vectorisation
-------------
The per-row greedy loop is vectorised over all output neurons simultaneously,
so each greedy step is a single GPU kernel.  Runtime is O(k * out_f * in_f)
per layer; for the 1B model this takes a few seconds per layer.

Semi-structured sparsity
------------------------
k = floor(in_features * sparsity) channels are zeroed per output neuron,
giving constant sparsity per row — a form of semi-structured sparsity.

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
    Accumulates Sigma_X = E[x x^T] for a single Linear layer's input.
    Stored in float32 on CPU to avoid GPU memory blow-up.
    """

    def __init__(self, in_features: int):
        self.sum_xx = torch.zeros(in_features, in_features, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """x: any shape (..., in_features)."""
        x = x.reshape(-1, x.shape[-1]).float().cpu()
        self.sum_xx += x.T @ x
        self.count += x.shape[0]

    def second_moment(self) -> torch.Tensor:
        """Returns Sigma_X = E[x x^T], shape (in_features, in_features)."""
        return self.sum_xx / max(self.count, 1)


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
# Greedy cancellation-aware pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_prune_layer(W: torch.Tensor, Sigma: torch.Tensor, k: int) -> torch.Tensor:
    """
    Greedily select k input channels per output neuron to prune by minimising
    the joint reconstruction error w_S^T Sigma[S,S] w_S.

    Args:
        W:     (out_f, in_f) weight matrix, float32
        Sigma: (in_f, in_f) second-moment matrix, float32, same device as W
        k:     number of channels to prune per row

    Returns:
        mask: (out_f, in_f) bool tensor, True = keep, False = prune
    """
    out_f, in_f = W.shape
    device = W.device

    diag_Sigma = Sigma.diagonal()   # (in_f,)

    # V[j, i] accumulates (Sigma @ w_{S'_j})[i] — cross-term from already-pruned channels
    V = torch.zeros(out_f, in_f, device=device, dtype=torch.float32)

    # Boolean mask: True = already selected for pruning
    pruned = torch.zeros(out_f, in_f, device=device, dtype=torch.bool)

    for _ in range(k):
        # Marginal error of adding channel i to the prune set for each row j:
        #   delta(j,i) = w_{ji}^2 * Sigma[i,i]  +  2 * w_{ji} * V[j,i]
        # Lower is better (less error added by pruning this channel).
        delta = W * W * diag_Sigma.unsqueeze(0) + 2.0 * W * V  # (out_f, in_f)

        # Exclude already-pruned channels
        delta = delta.masked_fill(pruned, float("inf"))

        # Pick the channel with minimum marginal error in each row
        chosen = delta.argmin(dim=1)   # (out_f,)

        # Mark as pruned
        pruned.scatter_(1, chosen.unsqueeze(1), True)

        # Update V: V[j,:] += Sigma[chosen[j],:] * W[j, chosen[j]]
        w_chosen = W[torch.arange(out_f, device=device), chosen]   # (out_f,)
        V += w_chosen.unsqueeze(1) * Sigma[chosen, :]               # (out_f, in_f)

    return ~pruned   # True = keep


@torch.no_grad()
def compute_scores_and_prune(model: nn.Module, stats: dict, sparsity: float,
                              device: torch.device):
    """
    Run the greedy cancellation-aware pruning on every Linear layer.

    Returns actual sparsity and per-layer diagnostic info.
    """
    total_weights = 0
    total_pruned = 0
    layer_info = {}

    for name, module in tqdm(model.named_modules(), desc="Pruning layers"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data     # (out_f, in_f), bfloat16 on device
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue

        W_f = W.float()
        Sigma = stats[name].second_moment().to(device)   # (in_f, in_f)

        keep_mask = greedy_prune_layer(W_f, Sigma, k)    # (out_f, in_f) bool

        # Count cancellation events: steps where the marginal error was negative
        # (proxy: how many off-diagonal cross-terms helped, measured post-hoc)
        # We report fraction of pruned weights that belong to a "cancelling pair":
        # pairs (i,j) in the pruned set where w_i*w_j*Sigma[i,j] < 0
        pruned_rows, pruned_cols = (~keep_mask).nonzero(as_tuple=True)
        cancellation_frac = 0.0
        if len(pruned_cols) > 0:
            neg_cross = 0
            # Sample per-row: check off-diagonal cross terms among pruned channels
            for row in range(min(out_f, 64)):   # sample 64 rows for speed
                cols = (~keep_mask[row]).nonzero(as_tuple=True)[0]
                if len(cols) < 2:
                    continue
                w_sub = W_f[row, cols]                      # (k,)
                S_sub = Sigma[cols][:, cols]                # (k, k) submatrix
                # Off-diagonal sign-check: w_i * w_j * Sigma[i,j] < 0
                outer_w = w_sub.unsqueeze(1) * w_sub.unsqueeze(0)  # (k,k)
                neg_cross += int(((outer_w * S_sub) < 0).triu(diagonal=1).sum().item())
            total_pairs = min(out_f, 64) * k * (k - 1) // 2
            cancellation_frac = neg_cross / max(total_pairs, 1)

        module.weight.data[~keep_mask] = 0.0

        n_pruned = (~keep_mask).sum().item()
        total_weights += out_f * in_f
        total_pruned += n_pruned
        layer_info[name] = {
            "out_features": out_f,
            "in_features": in_f,
            "n_pruned": n_pruned,
            "cancellation_pair_fraction": cancellation_frac,
        }

        del Sigma, W_f, keep_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

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

    # --- calibration ---
    print(f"\nBuilding calibration batches "
          f"({args.n_calib_batches} × {args.batch_size} × {args.seq_len}) ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches, args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, device)

    total_sigma_mb = sum(s.sum_xx.numel() * 4 / 1e6 for s in stats.values())
    print(f"  Sigma_X accumulated: {len(stats)} layers, "
          f"{total_sigma_mb:.0f} MB total (CPU float32)")

    # --- prune ---
    print(f"\nGreedy cancellation-aware pruning at {args.sparsity * 100:.1f}% sparsity ...")
    print("  (semi-structured: constant k channels zeroed per output neuron)")
    actual_sparsity, layer_info = compute_scores_and_prune(model, stats, args.sparsity, device)
    print(f"Actual sparsity: {actual_sparsity * 100:.2f}%")

    # Report cancellation hotspots
    top = sorted(layer_info.items(),
                 key=lambda kv: -kv[1]["cancellation_pair_fraction"])[:10]
    print(f"\nTop-10 layers by cancellation pair fraction:")
    print(f"  {'Layer':<55}  {'Cancel%':>8}")
    print("  " + "-" * 65)
    for lname, info in top:
        print(f"  {lname:<55}  {info['cancellation_pair_fraction'] * 100:>7.1f}%")

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
            "method": "cancellation_aware_greedy_quadratic",
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "sparsity_structure": "semi-structured (constant k per output neuron)",
            "scoring": (
                "greedy min of w_S^T Sigma_X[S,S] w_S; "
                "marginal step: w_{ji}^2*Sigma[i,i] + 2*w_{ji}*(Sigma@w_{S'_j})[i]"
            ),
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
