"""
SparseGPT baseline (Frantar & Alistarh, 2023).

Algorithm:
  For each Linear layer, given Hessian H = 2 * Sigma_X:
  1. Compute H_inv = inv(Sigma_X + damp * I) via Cholesky.
  2. Score all (row, column) pairs by OBS saliency:
       score(i, j) = W[i,j]^2 / H_inv[j,j]
  3. For each row, select the k = floor(in_features * sparsity) lowest-scoring
     columns as the prune set (semi-structured, constant k per row).
  4. Apply column-ordered OBS weight corrections in blocks of size `block_size`:
       - For each pruned entry (i, j), propagate the error to remaining columns:
           W[i, j+1:] -= (W[i,j] / H_inv[j,j]) * H_inv[j, j+1:]
       - Corrections accumulate across blocks via the cross-block update.

This decouples selection (step 2-3, first-order OBS scores) from correction
(step 4, column-wise OBS update), giving a clean and efficient implementation.

Key difference from prune_cancellation.py:
  - SparseGPT applies corrections column-by-column in order (sequential OBS),
    so each correction accounts for errors already introduced by earlier columns.
  - prune_cancellation.py applies a single closed-form batch correction for
    the full pruned set simultaneously.

Usage
-----
python scripts/prune_sparsegpt.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-sparsegpt-pruned-50pct \\
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
# Covariance accumulator (shared with other pruning scripts)
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
# SparseGPT layer pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def sparsegpt_prune_layer(W: torch.Tensor, Sigma: torch.Tensor,
                           sparsity: float, damp: float = 0.01,
                           block_size: int = 128) -> torch.Tensor:
    """
    Args:
        W:          (out_f, in_f) float32
        Sigma:      (in_f, in_f) second-moment matrix, float32, on same device
        sparsity:   fraction of input channels to zero per output neuron
        damp:       Tikhonov regularisation as fraction of mean diagonal
        block_size: columns per OBS correction block

    Returns corrected W (same shape, pruned entries zeroed).
    """
    out_f, in_f = W.shape
    device = W.device
    k = int(in_f * sparsity)
    if k == 0:
        return W.clone()

    # Regularised Hessian inverse
    H = Sigma.float().clone()
    H.diagonal().add_(damp * H.diagonal().mean().clamp(min=1e-8))
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)         # (in_f, in_f)
    except Exception:
        H_inv = torch.linalg.inv(H)

    # Step 1: score all (row, col) pairs by OBS saliency W[i,j]^2 / H_inv[j,j]
    scores = W ** 2 / H_inv.diagonal().unsqueeze(0).clamp(min=1e-8)  # (out_f, in_f)

    # Step 2: per-row select k lowest-scoring columns to prune
    _, prune_idx = scores.topk(k, dim=1, largest=False)
    pruned = torch.zeros(out_f, in_f, dtype=torch.bool, device=device)
    pruned.scatter_(1, prune_idx, False)           # mark kept first
    pruned.scatter_(1, prune_idx, True)            # then mark pruned

    # Step 3: column-ordered OBS weight correction in blocks
    W_out = W.clone()
    for b_start in range(0, in_f, block_size):
        b_end = min(b_start + block_size, in_f)
        W_blk = W_out[:, b_start:b_end].clone()   # (out_f, b_size)
        H_blk = H_inv[b_start:b_end, b_start:b_end]  # (b_size, b_size)
        Err   = torch.zeros(out_f, b_end - b_start, device=device)

        for i in range(b_end - b_start):
            col   = b_start + i
            h_ii  = H_blk[i, i].clamp(min=1e-8)
            w_col = W_blk[:, i].clone()

            # Error for pruned rows
            err = w_col.clone()
            err[~pruned[:, col]] = 0.0

            # Record normalised error for cross-block update
            Err[:, i] = err / h_ii

            # Zero pruned weights
            W_blk[:, i][pruned[:, col]] = 0.0

            # Propagate error to remaining columns within this block
            if i + 1 < b_end - b_start:
                W_blk[:, i + 1:] -= (err / h_ii).unsqueeze(1) * H_blk[i, i + 1:]

        W_out[:, b_start:b_end] = W_blk

        # Cross-block update: propagate accumulated errors to future blocks
        if b_end < in_f:
            W_out[:, b_end:] -= Err @ H_inv[b_start:b_end, b_end:]

    # Enforce zeros at pruned positions (corrections may leave small residuals)
    W_out[pruned] = 0.0
    return W_out


# ---------------------------------------------------------------------------
# Full model pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_model(model, stats, sparsity, device, damp=0.01, block_size=128):
    total_w, total_p = 0, 0
    layer_info = {}

    for name, module in tqdm(model.named_modules(), desc="SparseGPT pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue

        Sigma = stats[name].second_moment().to(device)
        W_corr = sparsegpt_prune_layer(W.float(), Sigma, sparsity,
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

    print(f"\nSparseGPT pruning at {args.sparsity * 100:.0f}% sparsity ...")
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
            "method": "sparsegpt",
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "sparsity_structure": "semi-structured (constant k per output neuron)",
            "scoring": "OBS saliency: W[i,j]^2 / H_inv[j,j]",
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
