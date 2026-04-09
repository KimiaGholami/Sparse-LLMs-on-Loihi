"""
Cancellation-aware semi-structured pruning with closed-form weight correction.

Algorithm
---------
For each Linear layer and each output neuron j, we greedily select the k input
channels to zero that minimise the joint reconstruction error:

    E[error_j^2] = w_S^T  Sigma_X[S,S]  w_S

where Sigma_X = E[x x^T] is the uncentered activation second-moment matrix.

Greedy selection (marginal rule at each step, starting from empty prune set S'):

    i* = argmin_i [ w_{ji}^2 * Sigma_X[i,i]  +  2 * w_{ji} * (Sigma_X @ w_{S'_j})[i] ]

    - At step 0 (S' = {}) this recovers the Wanda/diagonal criterion.
    - As cancelling channels accumulate in S', the cross-term goes negative,
      actively seeking mixed-sign weights on correlated inputs.

Weight correction (closed-form, after pruning)
----------------------------------------------
After zeroing S_j, the optimal update to the kept weights K_j that minimises
E[(corrected output - original output)^2] is:

    delta w_j[K_j] = inv(Sigma_X[K_j, K_j])  @  Sigma_X[K_j, S_j]  @  w_j[S_j]

This is a least-squares projection: the remaining weights absorb as much of the
pruned weight signal as the activation covariance allows.  Rows sharing the same
prune mask are batched together so each unique mask requires only one Cholesky
factorisation.

Sparsity structure
------------------
Exactly k = floor(in_features * sparsity) channels are zeroed per output neuron,
giving constant (semi-structured) sparsity per row.

Usage
-----
python scripts/prune_cancellation.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-cancellation-pruned-50pct \\
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
    """Accumulates Sigma_X = E[x x^T] on CPU in float32."""

    def __init__(self, in_features: int):
        self.sum_xx = torch.zeros(in_features, in_features, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float().cpu()
        self.sum_xx += x.T @ x
        self.count += x.shape[0]

    def second_moment(self) -> torch.Tensor:
        return self.sum_xx / max(self.count, 1)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def build_calib_batches(tokenizer, n_batches, batch_size, seq_len, seed=42):
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
def collect_covariance_stats(model, batches, device):
    stats = {}
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        stats[name] = CovarianceStats(module.in_features)
        def make_hook(n):
            def hook(mod, inp, out):
                stats[n].update(inp[0].detach())
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    for batch in tqdm(batches, desc="Collecting activations"):
        model(input_ids=batch.to(device))
    for h in hooks:
        h.remove()
    return stats


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_select(W: torch.Tensor, Sigma: torch.Tensor, k: int) -> torch.Tensor:
    """
    Returns pruned mask (out_f, in_f), True = pruned.

    Greedy: at each step pick the channel with the lowest marginal increase
    in joint reconstruction error w_S^T Sigma[S,S] w_S.

    Marginal cost of adding channel i given current prune set S':
        delta(i) = w_i^2 * Sigma[i,i]  +  2 * w_i * (Sigma @ w_{S'})[i]

    The cross-term goes negative for channels that destructively interfere
    with already-selected channels — the algorithm seeks these pairs out.
    """
    out_f, in_f = W.shape
    diag_S = Sigma.diagonal()                                  # (in_f,)
    V = torch.zeros(out_f, in_f, device=W.device, dtype=torch.float32)
    pruned = torch.zeros(out_f, in_f, device=W.device, dtype=torch.bool)

    for _ in range(k):
        delta = W * W * diag_S + 2.0 * W * V                  # (out_f, in_f)
        delta = delta.masked_fill(pruned, float("inf"))
        chosen = delta.argmin(dim=1)                           # (out_f,)
        pruned.scatter_(1, chosen.unsqueeze(1), True)
        w_c = W[torch.arange(out_f, device=W.device), chosen] # (out_f,)
        V += w_c.unsqueeze(1) * Sigma[chosen, :]              # (out_f, in_f)

    return pruned


# ---------------------------------------------------------------------------
# Weight correction
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_weight_correction(W: torch.Tensor, Sigma: torch.Tensor,
                             pruned_mask: torch.Tensor, lam: float = 1e-3):
    """
    For each output neuron j with pruned set S_j and kept set K_j:

        delta w_j[K_j] = inv(Sigma[K_j, K_j] + lam*I) @ Sigma[K_j, S_j] @ w_j[S_j]

    Rows sharing the same prune mask are batched, requiring one Cholesky per
    unique mask.  Falls back to a diagonal (per-channel) correction if the
    system is rank-deficient.

    Returns corrected W (same shape, float32).
    """
    out_f, in_f = W.shape
    W_corr = W.clone()

    # Group rows by identical prune pattern (efficient via torch.unique)
    unique_masks, inv_idx = torch.unique(pruned_mask.to(torch.int8),
                                         dim=0, return_inverse=True)

    for m_idx in range(unique_masks.shape[0]):
        rows = (inv_idx == m_idx).nonzero(as_tuple=True)[0]   # row indices
        S = unique_masks[m_idx].bool()                         # pruned channels
        K = ~S                                                 # kept channels
        K_idx = K.nonzero(as_tuple=True)[0]
        S_idx = S.nonzero(as_tuple=True)[0]

        if S_idx.numel() == 0 or K_idx.numel() == 0:
            continue

        Sigma_KS = Sigma[K_idx][:, S_idx]                     # (|K|, |S|)
        w_S = W[rows][:, S_idx]                               # (batch, |S|)

        # RHS: Sigma[K,S] @ w_j[S]  for each row j in the batch
        rhs = (Sigma_KS @ w_S.T)                              # (|K|, batch)

        Sigma_KK = Sigma[K_idx][:, K_idx]                     # (|K|, |K|)
        reg = lam * torch.eye(K_idx.numel(), device=Sigma.device,
                              dtype=Sigma.dtype)

        try:
            # Cholesky solve: (Sigma_KK + lam*I) @ delta^T = rhs
            L = torch.linalg.cholesky(Sigma_KK + reg)
            delta = torch.cholesky_solve(rhs, L).T             # (batch, |K|)
        except Exception:
            # Fallback: diagonal approximation
            diag_KK = (Sigma_KK + reg).diagonal().clamp(min=lam)
            delta = (rhs / diag_KK.unsqueeze(1)).T            # (batch, |K|)

        # Scatter correction into the right columns
        W_corr[rows.unsqueeze(1), K_idx.unsqueeze(0)] += delta

    return W_corr


# ---------------------------------------------------------------------------
# Main pruning loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_and_correct(model, stats, sparsity, device, lam=1e-3):
    total_w, total_p = 0, 0
    layer_info = {}

    for name, module in tqdm(model.named_modules(), desc="Pruning + correcting"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue

        W_f = W.float()
        Sigma = stats[name].second_moment().to(device)

        # Step 1: greedy cancellation-aware selection
        pruned_mask = greedy_select(W_f, Sigma, k)            # (out_f, in_f)

        # Step 2: closed-form weight correction on kept channels
        W_corr = apply_weight_correction(W_f, Sigma, pruned_mask, lam=lam)

        # Zero pruned channels in the corrected weights
        W_corr[pruned_mask] = 0.0
        module.weight.data.copy_(W_corr.to(W.dtype))

        n_pruned = pruned_mask.sum().item()
        total_w += out_f * in_f
        total_p += n_pruned
        layer_info[name] = {
            "out_features": out_f,
            "in_features": in_f,
            "n_unique_masks": int(
                torch.unique(pruned_mask.to(torch.int8), dim=0).shape[0]
            ),
        }

        del Sigma, W_f, W_corr, pruned_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_p / max(total_w, 1), layer_info


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=512, n_tokens=500_000):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    enc = enc[: min(n_tokens, enc.shape[0])]
    model.eval()
    nll, ntok = 0.0, 0
    for start in tqdm(range(0, enc.shape[0] - seq_len, seq_len), desc="PPL eval"):
        chunk = enc[start: start + seq_len].unsqueeze(0).to(device)
        nll += model(input_ids=chunk, labels=chunk).loss.item() * chunk.numel()
        ntok += chunk.numel()
    return math.exp(nll / ntok)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="exp/transformer-1B-dense-baseline")
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--n_calib_batches", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--correction_lambda", type=float, default=1e-3,
                   help="Tikhonov regularisation for weight correction solve")
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
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    ppl_before = None
    if args.eval_ppl:
        ppl_before = evaluate_ppl(model, tokenizer, device, args.seq_len)
        print(f"Perplexity before pruning: {ppl_before:.4f}")

    print(f"\nBuilding calibration batches "
          f"({args.n_calib_batches} × {args.batch_size} × {args.seq_len}) ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, device)

    mb = sum(s.sum_xx.numel() * 4 / 1e6 for s in stats.values())
    print(f"  Sigma_X: {len(stats)} layers, {mb:.0f} MB total (CPU float32)")

    print(f"\nGreedy pruning + weight correction at {args.sparsity * 100:.1f}% sparsity ...")
    actual_sparsity, layer_info = prune_and_correct(
        model, stats, args.sparsity, device, lam=args.correction_lambda
    )
    print(f"Actual sparsity: {actual_sparsity * 100:.2f}%")

    avg_unique = sum(v["n_unique_masks"] for v in layer_info.values()) / max(len(layer_info), 1)
    print(f"Average unique prune masks per layer: {avg_unique:.0f}")

    ppl_after = None
    if args.eval_ppl:
        ppl_after = evaluate_ppl(model, tokenizer, device, args.seq_len)
        print(f"\nPerplexity after pruning: {ppl_after:.4f}  "
              f"(Δ = {ppl_after - ppl_before:+.4f})")

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        summary = {
            "method": "cancellation_aware_greedy_with_weight_correction",
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "sparsity_structure": "semi-structured (constant k per output neuron)",
            "scoring": (
                "greedy min of w_S^T Sigma_X[S,S] w_S; "
                "marginal: w_i^2*Sigma[i,i] + 2*w_i*(Sigma@w_{S'})[i]"
            ),
            "correction": (
                "delta w[K] = inv(Sigma[K,K] + lam*I) @ Sigma[K,S] @ w[S]"
            ),
            "correction_lambda": args.correction_lambda,
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
