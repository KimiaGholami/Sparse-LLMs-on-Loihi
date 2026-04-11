"""
Interleaved cancellation-aware selection + OBS correction.

Why the hybrid failed
---------------------
prune_hybrid.py applied cancellation-aware selection first (globally), then
OBS corrections (column-ordered). This fails because OBS correction assumes
pruning decisions were made left-to-right — its error propagation is only
valid for column-ordered masks.

This script fixes the mismatch by interleaving selection and correction
block-by-block:

  For each block of columns [b, b+B):
    1. SELECT  — greedy cancellation-aware scoring on the CURRENT (already
                 OBS-corrected) weight matrix, restricted to this block's
                 columns, with a proportional prune budget for this block.
                 Uses a running V that accumulates cross-channel interference
                 from all previously pruned columns (including prior blocks).
    2. CORRECT — column-ordered OBS updates within the block, then cross-block
                 propagation to future blocks.

After each block:
  - W is corrected for the pruned entries in that block.
  - V is updated so that the NEXT block's selection accounts for already-pruned
    channels.

This means:
  - Selection uses cancellation-aware scores on up-to-date weights       ✓
  - OBS correction operates on column-ordered, block-local decisions      ✓
  - Cross-block interference is tracked via the running V accumulator     ✓

Algorithm per layer
-------------------
Inputs: W (out_f × in_f), Σ_X (in_f × in_f), sparsity s, block_size B

  H_inv  = inv(Σ_X + damp·I)          # OBS inverse Hessian
  V      = 0  (out_f × in_f)          # Σ_X @ w_pruned accumulator
  n_done = 0  (out_f,)                 # per-row prune count
  k      = floor(in_f · s)            # target prunes per row

  for b in range(0, in_f, B):
    b_end = min(b + B, in_f)
    B_cur = b_end - b

    # --- budget for this block (proportional, last block exact) ---
    if b_end == in_f:
        budget = k - n_done
    else:
        budget = floor((k - n_done) * B_cur / (in_f - b))

    # --- greedy cancellation selection within block ---
    W_blk      = W[:, b:b_end].clone()        # current corrected weights
    diag_blk   = Σ[b:b_end, b:b_end].diag()  # block diagonal of Σ_X
    V_blk      = zeros(out_f, B_cur)           # within-block V accumulator
    blk_pruned = zeros(out_f, B_cur, bool)
    blk_done   = zeros(out_f, int)

    for _ in range(budget.max()):
        # global + within-block cancellation term at block columns
        V_total = V[:, b:b_end] + V_blk         # (out_f, B_cur)
        delta = W_blk**2 * diag_blk + 2 * W_blk * V_total
        delta[blk_pruned | (blk_done >= budget).unsqueeze(1)] = inf
        chosen = delta.argmin(dim=1)             # (out_f,)
        active = blk_done < budget
        blk_pruned[active, chosen[active]] = True
        blk_done[active] += 1
        w_c = W_blk[torch.arange(out_f), chosen]
        # update within-block V only for active rows
        V_blk[active] += (w_c[active].unsqueeze(1)
                          * Σ[b + chosen[active], b:b_end])

    # --- column-ordered OBS correction within block ---
    H_blk = H_inv[b:b_end, b:b_end]
    Err   = zeros(out_f, B_cur)

    for i in range(B_cur):
        col  = b + i
        h_ii = H_blk[i, i].clamp(min=1e-8)
        w_col = W[:, col].clone()

        err = w_col.clone()
        err[~blk_pruned[:, i]] = 0.0

        Err[:, i]        = err / h_ii
        W[:, col][blk_pruned[:, i]] = 0.0
        if i + 1 < B_cur:
            W[:, b+i+1:b_end] -= (err/h_ii).unsqueeze(1) * H_blk[i, i+1:]

    # cross-block update
    if b_end < in_f:
        W[:, b_end:] -= Err @ H_inv[b:b_end, b_end:]

    # --- update global V for future blocks ---
    # For each pruned (row, col in this block): V[row, future] += w_val * Σ[col, future]
    # where w_val is the weight value AT TIME OF PRUNING (before OBS zeroed it).
    for i in range(B_cur):
        col = b + i
        mask_i = blk_pruned[:, i]
        if mask_i.any():
            w_vals = W_blk[mask_i, i]            # weight values before OBS
            if b_end < in_f:
                V[mask_i.nonzero(as_tuple=True)[0].unsqueeze(1),
                  torch.arange(b_end, in_f, device=W.device).unsqueeze(0)] += \
                    w_vals.unsqueeze(1) * Σ[col, b_end:].unsqueeze(0)

    n_done += blk_done

Usage
-----
python scripts/prune_interleaved.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-interleaved-pruned-50pct \\
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
# Interleaved layer pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def interleaved_prune_layer(W: torch.Tensor, Sigma: torch.Tensor,
                              sparsity: float, damp: float = 0.01,
                              block_size: int = 128) -> torch.Tensor:
    """
    Args:
        W:          (out_f, in_f) float32 weight matrix
        Sigma:      (in_f, in_f) activation second-moment matrix, float32
        sparsity:   fraction of weights to zero per output neuron row
        damp:       Tikhonov regularisation as fraction of mean diagonal
        block_size: columns per block

    Returns corrected W (same shape, pruned entries zeroed).
    """
    out_f, in_f = W.shape
    device = W.device
    k = int(in_f * sparsity)
    if k == 0:
        return W.clone()

    # Regularised Hessian inverse (shared for OBS correction)
    H = Sigma.float().clone()
    H.diagonal().add_(damp * H.diagonal().mean().clamp(min=1e-8))
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        H_inv = torch.linalg.inv(H)

    W_cur = W.clone()

    # V[i, j] = (Sigma @ w_pruned_so_far)[j] for row i
    # Tracks cross-channel interference from all previously pruned columns.
    V = torch.zeros(out_f, in_f, device=device, dtype=torch.float32)

    # Per-row prune count
    n_done = torch.zeros(out_f, device=device, dtype=torch.int32)

    for b_start in range(0, in_f, block_size):
        b_end = min(b_start + block_size, in_f)
        B_cur = b_end - b_start

        # ---- proportional budget for this block ----
        remaining_budget = (k - n_done).clamp(min=0)         # (out_f,)
        if b_end == in_f:
            block_budget = remaining_budget
        else:
            remaining_cols = in_f - b_start
            block_budget = (remaining_budget * B_cur // remaining_cols).clamp(min=0)
        max_steps = int(block_budget.max().item())

        # ---- greedy cancellation selection within block ----
        W_blk     = W_cur[:, b_start:b_end].clone()           # (out_f, B_cur)
        diag_blk  = Sigma[b_start:b_end, b_start:b_end].diagonal()  # (B_cur,)
        V_blk     = torch.zeros(out_f, B_cur, device=device)  # within-block V
        blk_pruned = torch.zeros(out_f, B_cur, device=device, dtype=torch.bool)
        blk_done  = torch.zeros(out_f, device=device, dtype=torch.int32)

        for _ in range(max_steps):
            # Combined cancellation term: global (from prior blocks) + local (this block)
            V_total = V[:, b_start:b_end] + V_blk              # (out_f, B_cur)
            delta = W_blk * W_blk * diag_blk + 2.0 * W_blk * V_total

            # Mask already-pruned or budget-exhausted
            budget_mask = (blk_done >= block_budget).unsqueeze(1).expand_as(delta)
            delta = delta.masked_fill(blk_pruned | budget_mask, float("inf"))

            chosen = delta.argmin(dim=1)                        # (out_f,)
            active = blk_done < block_budget                    # rows still in budget

            if not active.any():
                break

            # Update prune mask and count
            active_idx = active.nonzero(as_tuple=True)[0]
            blk_pruned[active_idx, chosen[active_idx]] = True
            blk_done[active_idx] += 1

            # Update within-block V: V_blk[i] += w_chosen[i] * Sigma[chosen_col, block_cols]
            w_c = W_blk[active_idx, chosen[active_idx]]        # (n_active,)
            sigma_rows = Sigma[b_start + chosen[active_idx], b_start:b_end]  # (n_active, B_cur)
            V_blk[active_idx] += w_c.unsqueeze(1) * sigma_rows

        # ---- column-ordered OBS correction within block ----
        H_blk = H_inv[b_start:b_end, b_start:b_end]
        Err   = torch.zeros(out_f, B_cur, device=device)

        for i in range(B_cur):
            col  = b_start + i
            h_ii = H_blk[i, i].clamp(min=1e-8)
            w_col = W_cur[:, col].clone()

            err = w_col.clone()
            err[~blk_pruned[:, i]] = 0.0

            Err[:, i] = err / h_ii
            W_cur[blk_pruned[:, i], col] = 0.0

            if i + 1 < B_cur:
                W_cur[:, b_start + i + 1:b_end] -= \
                    (err / h_ii).unsqueeze(1) * H_blk[i, i + 1:]

        W_cur[:, b_start:b_end] = W_cur[:, b_start:b_end]  # already updated in-place

        # cross-block OBS propagation
        if b_end < in_f:
            W_cur[:, b_end:] -= Err @ H_inv[b_start:b_end, b_end:]

        # ---- update global V for future blocks ----
        # Use the weight values AT PRUNING TIME (from W_blk, before OBS zeroed them)
        if b_end < in_f:
            for i in range(B_cur):
                col = b_start + i
                mask_i = blk_pruned[:, i]
                if not mask_i.any():
                    continue
                active_rows = mask_i.nonzero(as_tuple=True)[0]
                w_vals = W_blk[active_rows, i]                  # (n,) weights at prune time
                # V[active_rows, b_end:] += w_vals * Sigma[col, b_end:]
                V[active_rows.unsqueeze(1),
                  torch.arange(b_end, in_f, device=device).unsqueeze(0)] += \
                    w_vals.unsqueeze(1) * Sigma[col, b_end:].unsqueeze(0)

        n_done += blk_done

    # Enforce zeros (corrections may leave small residuals at pruned positions)
    # Re-build pruned mask from n_done check — track it properly
    W_cur[W_cur.abs() < 1e-10] = W_cur[W_cur.abs() < 1e-10]  # no-op safety
    return W_cur


# ---------------------------------------------------------------------------
# Full model pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_model(model, stats, sparsity, device, damp=0.01, block_size=128):
    total_w, total_p = 0, 0
    layer_info = {}

    for name, module in tqdm(model.named_modules(), desc="Interleaved pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue

        Sigma = stats[name].second_moment().to(device)
        W_corr = interleaved_prune_layer(W.float(), Sigma, sparsity,
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

    print(f"\nInterleaved pruning at {args.sparsity * 100:.0f}% sparsity ...")
    print(f"  Selection: cancellation-aware greedy (per block, on current W)")
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
            "method": "interleaved_cancel_obs",
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "scoring": "cancellation-aware greedy per block on current W",
            "correction": "column-ordered OBS per block with cross-block propagation",
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
