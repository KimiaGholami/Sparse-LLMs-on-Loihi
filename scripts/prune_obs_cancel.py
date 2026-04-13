"""
OBS-compatible cancellation-aware pruning.

The theory
----------
SparseGPT scores each weight independently as w_j² / H_inv[j,j], which is
the exact first-order OBS cost for zeroing a single weight. It does NOT account
for cancellation: groups of weights that destructively interfere can have lower
joint OBS cost than any per-weight score predicts.

The true OBS cost for pruning a set S (after optimally correcting kept weights K) is:

    E(S) = w_S^T · [H_inv[S,S]]^{-1} · w_S

The GREEDY marginal increase from adding weight j to an already-decided set S' is:

    δ(j | S') = E(S' ∪ {j}) - E(S')
              = r_j(S')² / d_j(S')

where the RESIDUAL WEIGHT and RESIDUAL DIAGONAL evolve as:

    r_j(∅)    = w_j
    d_j(∅)    = H_inv[j,j]

    After adding weight c to S':
    r_j(S'∪{c}) = r_j(S') - [H_inv[j,c] / d_c(S')] · r_c(S')
    d_j(S'∪{c}) = d_j(S') - H_inv[j,c]² / d_c(S')

These are Schur complement rank-1 updates — exactly the same structure as
incremental Cholesky downdates.

At S' = ∅ (step 0), δ(j|∅) = w_j² / H_inv[j,j] = SparseGPT's score. So
SparseGPT is recovered as the special case where only the first step matters.
As S' grows, the residuals r and d diverge from w and H_inv[j,j], capturing
the joint cancellation structure that SparseGPT misses.

After the prune mask is selected (k greedy steps), OBS corrections are applied
column-by-column in the same blocked structure as SparseGPT.

Algorithm per layer
-------------------
Input: W (out_f × in_f), H_inv = inv(2·Σ_X + damp·I), k = floor(in_f · s)

  R = W.clone()                  # (out_f, in_f) residual weights
  D = H_inv.diagonal().clone()   # (in_f,)       residual diagonals (per column)
  pruned = zeros(out_f, in_f, bool)

  for t in range(k):
      score = R² / D             # (out_f, in_f) — OBS marginal per weight
      score[pruned] = inf
      chosen = score.argmin(dim=1)          # (out_f,) greedy min per row
      pruned.scatter_(1, chosen, True)

      # Rank-1 residual update for all remaining columns
      c = chosen                             # (out_f,) chosen column per row
      r_c = R[rows, c]                       # (out_f,)  residual at chosen
      d_c = D[c]                             # (out_f,)  residual diagonal at chosen
      h_jc = H_inv[c, :]                    # (out_f, in_f) cross-terms
      R -= (r_c / d_c).unsqueeze(1) * h_jc[rows]
      D -= h_jc[rows] ** 2 / d_c.unsqueeze(1)
      — NOTE: D is per-column (in_f,), but rows choose different c, so update is
        per-row. See code for the batched form.

  → prune mask ready; apply OBS corrections (same as SparseGPT).

Complexity
----------
  k greedy steps × O(out_f · in_f) per step = O(k · out_f · in_f)
  Same asymptotic cost as the pure greedy cancellation method.

Usage
-----
python scripts/prune_obs_cancel.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-obs-cancel-pruned-50pct \\
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
# OBS-cancellation layer pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def obs_cancel_prune_layer(W: torch.Tensor, Sigma: torch.Tensor,
                             sparsity: float, damp: float = 0.01,
                             block_size: int = 128) -> torch.Tensor:
    """
    OBS-compatible cancellation-aware pruning.

    Selection: greedy minimisation of E(S) = w_S^T [H_inv[S,S]]^{-1} w_S
               via Schur complement rank-1 residual updates.
    Correction: column-ordered OBS updates in blocks (same as SparseGPT).

    Args:
        W:          (out_f, in_f) float32
        Sigma:      (in_f, in_f) second-moment matrix, float32
        sparsity:   fraction of weights to zero per row
        damp:       Tikhonov regularisation as fraction of mean diagonal
        block_size: columns per OBS correction block

    Returns corrected W (same shape, pruned entries zeroed).
    """
    out_f, in_f = W.shape
    device = W.device
    k = int(in_f * sparsity)
    if k == 0:
        return W.clone()

    rows = torch.arange(out_f, device=device)

    # Regularised H_inv = inv(Sigma + damp * I)
    H = Sigma.float().clone()
    H.diagonal().add_(damp * H.diagonal().mean().clamp(min=1e-8))
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)          # (in_f, in_f)
    except Exception:
        H_inv = torch.linalg.inv(H)

    # ---------------------------------------------------------------
    # Step 1: Greedy selection via OBS residual updates
    # ---------------------------------------------------------------
    # R[i, j] = residual weight for (row i, column j)
    # D[i, j] = residual diagonal for (row i, column j)
    #
    # At each step t:
    #   score[i,j] = R[i,j]^2 / D[i,j]   (OBS marginal cost)
    #   chosen[i]  = argmin_j score[i,j]  (greedy per-row)
    #
    # Rank-1 update after choosing column c for row i:
    #   R[i, j] -= (R[i,c] / D[i,c]) * H_inv[c, j]   for all j ≠ c
    #   D[i, j] -= H_inv[c, j]^2 / D[i,c]             for all j ≠ c
    #
    # Key: different rows may choose different columns c, so the update
    # is row-specific. We batch this with gather/scatter.
    # ---------------------------------------------------------------

    # Use float64 for residuals to reduce accumulated drift over many rank-1 updates.
    # For large layers (e.g. LLaMA-7B in_f=4096/11008), we perform k=2000-5500
    # rank-1 Schur complement updates. In float32 the D diagonal drifts negative;
    # float64 keeps the Schur complements accurate throughout.
    H_inv64 = H_inv.double()
    R = W.clone().double().to(device)              # (out_f, in_f) residual weights
    D = H_inv64.diagonal().unsqueeze(0).expand(out_f, -1).clone()  # (out_f, in_f)
    pruned = torch.zeros(out_f, in_f, device=device, dtype=torch.bool)

    for _ in range(k):
        score = R * R / D.clamp(min=1e-8)         # (out_f, in_f)
        score = score.masked_fill(pruned, float("inf"))
        chosen = score.argmin(dim=1)               # (out_f,)

        pruned.scatter_(1, chosen.unsqueeze(1), True)

        # Gather residual values at chosen columns
        r_c = R[rows, chosen]                      # (out_f,)   R[i, c_i]
        d_c = D[rows, chosen].clamp(min=1e-8)      # (out_f,)   D[i, c_i]

        # H_inv row at each chosen column: H_inv[c_i, :] — (out_f, in_f)
        H_inv_c = H_inv64[chosen, :]               # (out_f, in_f)

        # Rank-1 update:
        #   R[i, :] -= (r_c[i] / d_c[i]) * H_inv[c_i, :]
        #   D[i, :] -= H_inv[c_i, :]^2 / d_c[i]
        scale_r = (r_c / d_c).unsqueeze(1)         # (out_f, 1)
        scale_d = d_c.unsqueeze(1)                 # (out_f, 1)

        R -= scale_r * H_inv_c
        D -= H_inv_c * H_inv_c / scale_d

        # Clamp D to avoid residual negative values from floating-point drift
        D.clamp_(min=1e-8)

    del H_inv64, R, D  # free double-precision buffers before OBS correction

    # ---------------------------------------------------------------
    # Step 2: Column-ordered OBS correction (same as SparseGPT)
    # ---------------------------------------------------------------
    W_out = W.clone()
    for b_start in range(0, in_f, block_size):
        b_end = min(b_start + block_size, in_f)
        W_blk = W_out[:, b_start:b_end].clone()
        H_blk = H_inv[b_start:b_end, b_start:b_end]
        Err   = torch.zeros(out_f, b_end - b_start, device=device)

        for i in range(b_end - b_start):
            col  = b_start + i
            h_ii = H_blk[i, i].clamp(min=1e-8)
            w_col = W_blk[:, i].clone()

            err = w_col.clone()
            err[~pruned[:, col]] = 0.0

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

    for name, module in tqdm(model.named_modules(), desc="OBS-cancel pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue

        Sigma = stats[name].second_moment().to(device)
        W_corr = obs_cancel_prune_layer(W.float(), Sigma, sparsity,
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

    print(f"\nOBS-cancel pruning at {args.sparsity * 100:.0f}% sparsity ...")
    print(f"  Selection: greedy OBS residual (Schur complement rank-1 updates)")
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
            "method": "obs_cancel",
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "scoring": "greedy OBS residual: r_j^2/d_j with Schur complement updates",
            "correction": "column-ordered OBS weight update in blocks",
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
