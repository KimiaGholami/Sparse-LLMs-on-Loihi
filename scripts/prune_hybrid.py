"""
Hybrid: cancellation-aware selection + SparseGPT OBS correction.

Motivation
----------
SparseGPT (Frantar & Alistarh, 2023) dominates because of its column-ordered
OBS weight corrections — not because of its scoring criterion.  Its scores
W[i,j]^2 / H_inv[j,j] treat each weight independently (diagonal), identical in
spirit to Wanda.

Our cancellation-aware greedy selection (prune_cancellation.py) uses the full
activation second-moment matrix Σ_X to jointly score subsets:
    delta(j | S') = w_j^2 * Σ[j,j] + 2 * w_j * (Σ @ w_{S'})[j]
This accounts for destructive interference between groups of mixed-sign weights
on positively correlated input channels, which purely diagonal scores miss.

This script combines both strengths:
  1. Selection  — cancellation-aware greedy on Σ_X (same as prune_cancellation.py)
  2. Correction — column-ordered OBS updates in blocks (same as prune_sparsegpt.py)

The hypothesis is that better selection (step 1) will allow the OBS correction
(step 2) to start from a better prune mask, yielding lower post-pruning PPL.

Algorithm per layer
-------------------
Given W (out_f × in_f) and Σ_X (in_f × in_f):

Step 1 — Cancellation-aware greedy selection:
  V = 0  (out_f × in_f running accumulator)
  for t in range(k):
      delta = W*W * diag(Σ) + 2 * W * V          # marginal error increase
      chosen = argmin per row (ignoring already-pruned)
      V += w_chosen * Σ[chosen, :]                 # update cross-channel term
  → yields pruned mask (out_f × in_f bool)

Step 2 — Column-ordered OBS correction:
  H_inv = inv(Σ_X + damp * I)
  For each block of columns [b, b+B):
    For each column j in block:
      err = W[:, j] * pruned[:, j]               # error at pruned entries
      W[:, j][pruned[:, j]] = 0
      W[:, j+1:block_end] -= (err/H_inv[j,j]) * H_inv[j, j+1:block_end]
    W[:, block_end:] -= Err @ H_inv[b:b+B, block_end:]

Usage
-----
python scripts/prune_hybrid.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-hybrid-pruned-50pct \\
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

try:
    import fla  # noqa
    from fla.models.transformer import TransformerConfig, TransformerForCausalLM
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: could not register FLA transformer type: {e}")


# ---------------------------------------------------------------------------
# Covariance accumulator
# ---------------------------------------------------------------------------

class CovarianceStats:
    def __init__(self, in_features):
        self.sum_xx = torch.zeros(in_features, in_features, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float().cpu()
        self.sum_xx += x.T @ x
        self.count += x.shape[0]

    def second_moment(self):
        return self.sum_xx / max(self.count, 1)


# ---------------------------------------------------------------------------
# Calibration
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
# Hybrid layer pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def hybrid_prune_layer(W: torch.Tensor, Sigma: torch.Tensor,
                        sparsity: float, damp: float = 0.01,
                        block_size: int = 128) -> torch.Tensor:
    """
    Args:
        W:         (out_f, in_f) float32 weight matrix
        Sigma:     (in_f, in_f) activation second-moment matrix, float32
        sparsity:  fraction of input channels to zero per output neuron
        damp:      Tikhonov regularisation as fraction of mean diagonal
        block_size: columns per OBS correction block

    Returns corrected W (same shape, pruned entries zeroed).
    """
    out_f, in_f = W.shape
    device = W.device
    k = int(in_f * sparsity)
    if k == 0:
        return W.clone()

    # ------------------------------------------------------------------
    # Step 1: Cancellation-aware greedy selection
    # ------------------------------------------------------------------
    # Marginal error increase from adding weight j to the prune set S':
    #   delta(j | S') = w_j^2 * Sigma[j,j] + 2 * w_j * (Sigma @ w_{S'})[j]
    # V accumulates (Sigma @ w_{S'}) row-wise as we greedily add weights.
    # ------------------------------------------------------------------
    diag_S = Sigma.diagonal()                              # (in_f,)
    V = torch.zeros(out_f, in_f, device=device, dtype=torch.float32)
    pruned = torch.zeros(out_f, in_f, device=device, dtype=torch.bool)

    for _ in range(k):
        delta = W * W * diag_S + 2.0 * W * V
        delta = delta.masked_fill(pruned, float("inf"))
        chosen = delta.argmin(dim=1)                       # (out_f,)
        pruned.scatter_(1, chosen.unsqueeze(1), True)
        w_c = W[torch.arange(out_f, device=device), chosen]  # (out_f,)
        V += w_c.unsqueeze(1) * Sigma[chosen, :]           # (out_f, in_f)

    # ------------------------------------------------------------------
    # Step 2: Column-ordered OBS weight correction (SparseGPT-style)
    # ------------------------------------------------------------------
    # H_inv = inv(Sigma + damp * I)  — regularised inverse Hessian
    # For each pruned weight W[i,j], the OBS update propagates the error
    # to remaining kept weights in the same row:
    #   W[i, j+1:] -= (W[i,j] / H_inv[j,j]) * H_inv[j, j+1:]
    # ------------------------------------------------------------------
    H = Sigma.float().clone()
    H.diagonal().add_(damp * H.diagonal().mean().clamp(min=1e-8))
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)                  # (in_f, in_f)
    except Exception:
        H_inv = torch.linalg.inv(H)

    W_out = W.clone()
    for b_start in range(0, in_f, block_size):
        b_end = min(b_start + block_size, in_f)
        W_blk = W_out[:, b_start:b_end].clone()           # (out_f, b_size)
        H_blk = H_inv[b_start:b_end, b_start:b_end]       # (b_size, b_size)
        Err   = torch.zeros(out_f, b_end - b_start, device=device)

        for i in range(b_end - b_start):
            col  = b_start + i
            h_ii = H_blk[i, i].clamp(min=1e-8)
            w_col = W_blk[:, i].clone()

            err = w_col.clone()
            err[~pruned[:, col]] = 0.0                     # only pruned entries

            Err[:, i] = err / h_ii
            W_blk[:, i][pruned[:, col]] = 0.0

            if i + 1 < b_end - b_start:
                W_blk[:, i + 1:] -= (err / h_ii).unsqueeze(1) * H_blk[i, i + 1:]

        W_out[:, b_start:b_end] = W_blk
        if b_end < in_f:
            W_out[:, b_end:] -= Err @ H_inv[b_start:b_end, b_end:]

    W_out[pruned] = 0.0
    return W_out


# ---------------------------------------------------------------------------
# Full model pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_model(model, stats, sparsity, device, damp=0.01, block_size=128):
    total_w, total_p = 0, 0
    layer_info = {}

    for name, module in tqdm(model.named_modules(), desc="Hybrid pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue

        Sigma = stats[name].second_moment().to(device)
        W_corr = hybrid_prune_layer(W.float(), Sigma, sparsity,
                                     damp=damp, block_size=block_size)
        module.weight.data.copy_(W_corr.to(W.dtype))

        n_pruned = (W_corr == 0).sum().item()
        total_w += out_f * in_f
        total_p += n_pruned
        layer_info[name] = {"out_features": out_f, "in_features": in_f,
                            "n_pruned": n_pruned}

        del Sigma, W_corr
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_p / max(total_w, 1), layer_info


# ---------------------------------------------------------------------------
# Perplexity
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
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--block_size", type=int, default=128)
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
        print(f"Perplexity before: {ppl_before:.4f}")

    print(f"\nBuilding calibration batches ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, device)

    print(f"\nHybrid pruning at {args.sparsity * 100:.0f}% sparsity ...")
    print(f"  Selection: cancellation-aware greedy (full Sigma_X)")
    print(f"  Correction: column-ordered OBS in blocks of {args.block_size}")
    actual_sparsity, layer_info = prune_model(
        model, stats, args.sparsity, device, damp=args.damp, block_size=args.block_size
    )
    print(f"Actual sparsity: {actual_sparsity * 100:.2f}%")

    ppl_after = None
    if args.eval_ppl:
        ppl_after = evaluate_ppl(model, tokenizer, device, args.seq_len)
        print(f"Perplexity after:  {ppl_after:.4f}  (Δ = {ppl_after - ppl_before:+.4f})")

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        summary = {
            "method": "hybrid_cancel_sparsegpt",
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "sparsity_structure": "semi-structured (constant k per output neuron)",
            "scoring": "cancellation-aware greedy: w^2*Sigma[j,j] + 2*w*(Sigma @ w_S')[j]",
            "correction": "column-ordered OBS weight update in blocks (SparseGPT-style)",
            "damp": args.damp,
            "block_size": args.block_size,
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
