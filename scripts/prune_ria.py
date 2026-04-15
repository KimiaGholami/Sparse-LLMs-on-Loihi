"""
RIA pruning baseline (Zhang et al., ICLR 2024 — "Plug-and-Play: An Efficient
Post-training Pruning Method for Large Language Models").

Scoring
-------
RIA combines two relative-importance terms with an activation norm:

    RIA[r, c] = ( |W[r,c]| / ||W[r,:]||₁   +   |W[r,c]| / ||W[:,c]||₁ )
                  × act_rms[c] ^ α

where:
  |W[r,c]| / ||W[r,:]||₁  — weight's share of its output neuron's total L1 mass
  |W[r,c]| / ||W[:,c]||₁  — weight's share of its input channel's total L1 mass
  act_rms[c] = sqrt(E[x_c²]) — RMS activation of input channel c
  α = 0.5 (default, controls activation influence)

Intuition
---------
Wanda scores weights by their absolute value times activation scale, ignoring
how large a weight is *relative to its peers*. RIA normalises by both the
output-row L1 norm and the input-column L1 norm, so a weight that is large
in absolute terms but unremarkable within its row/column can still be pruned
if its input channel is inactive.  At α=0 this reduces to a magnitude-relative
score with no activation information; at α=1 it approaches Wanda's activation
scaling.

No weight correction is applied (same as Wanda).  Semi-structured: exactly
k = floor(in_features × sparsity) weights zeroed per output row.

Usage
-----
python scripts/prune_ria.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-ria-pruned-50pct \\
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
# Activation stats  (same as Wanda — only diagonal of Sigma_X needed)
# ---------------------------------------------------------------------------

class ChannelNormStats:
    """Accumulates E[x_c²] per input channel c."""

    def __init__(self, in_features: int):
        self.sum2  = torch.zeros(in_features, dtype=torch.float64)
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).double().cpu()
        self.sum2  += (x ** 2).sum(0)
        self.count += x.shape[0]

    def rms(self) -> torch.Tensor:
        """sqrt(E[x_c²]), shape (in_features,)."""
        return (self.sum2 / max(self.count, 1)).sqrt().float()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calib_batches(tokenizer, n_batches, batch_size, seq_len, seed=42):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text    = "\n\n".join(dataset["text"])
    enc     = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    rng     = torch.Generator()
    rng.manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        start = torch.randint(0, enc.shape[0] - seq_len, (1,), generator=rng).item()
        chunk = enc[start: start + seq_len].unsqueeze(0).expand(batch_size, -1).clone()
        batches.append(chunk)
    return batches


@torch.no_grad()
def collect_stats(model, batches, device):
    stats = {}
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        stats[name] = ChannelNormStats(module.in_features)
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
# RIA layer pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def ria_prune_layer(W: torch.Tensor, act_rms: torch.Tensor,
                    sparsity: float, alpha: float = 0.5) -> torch.Tensor:
    """
    Apply RIA pruning to a single weight matrix.

    Args:
        W:        (out_f, in_f) weight matrix, any dtype
        act_rms:  (in_f,) sqrt(E[x_c²]) per input channel
        sparsity: fraction of weights to zero per row
        alpha:    activation exponent (default 0.5)

    Returns W with k = floor(in_f * sparsity) entries per row set to zero.
    """
    out_f, in_f = W.shape
    k = int(in_f * sparsity)
    if k == 0:
        return W.clone()

    W_abs = W.float().abs()                                   # (out_f, in_f)

    # Relative importance within each output row
    row_l1  = W_abs.sum(dim=1, keepdim=True).clamp(min=1e-8) # (out_f,  1)
    rel_row = W_abs / row_l1                                  # (out_f, in_f)

    # Relative importance within each input column
    col_l1  = W_abs.sum(dim=0, keepdim=True).clamp(min=1e-8) # (1,    in_f)
    rel_col = W_abs / col_l1                                  # (out_f, in_f)

    # Activation weighting: act_rms[c]^alpha
    act_w   = act_rms.to(W.device).float().pow(alpha)        # (in_f,)

    scores  = (rel_row + rel_col) * act_w.unsqueeze(0)       # (out_f, in_f)

    # Per-row: zero the k lowest-scoring weights
    _, prune_idx = scores.topk(k, dim=1, largest=False)
    W_out = W.float().clone()
    W_out.scatter_(1, prune_idx, 0.0)
    return W_out


# ---------------------------------------------------------------------------
# Full model pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_model(model, stats, sparsity, device, alpha=0.5):
    total_w, total_p = 0, 0
    layer_info = {}

    for name, module in tqdm(model.named_modules(), desc="RIA pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        if int(in_f * sparsity) == 0:
            continue

        act_rms = stats[name].rms().to(device)
        W_pruned = ria_prune_layer(W.float(), act_rms, sparsity, alpha=alpha)
        module.weight.data.copy_(W_pruned.to(W.dtype))

        n_pruned = (W_pruned == 0).sum().item()
        total_w += out_f * in_f
        total_p += n_pruned
        layer_info[name] = {"out_features": out_f, "in_features": in_f,
                            "n_pruned": n_pruned}

    return total_p / max(total_w, 1), layer_info


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=512, n_tokens=500_000):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text    = "\n\n".join(dataset["text"])
    enc     = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    enc     = enc[:min(n_tokens, enc.shape[0])]
    model.eval()
    nll, ntok = 0.0, 0
    for start in tqdm(range(0, enc.shape[0] - seq_len, seq_len), desc="PPL eval"):
        chunk = enc[start: start + seq_len].unsqueeze(0).to(device)
        nll  += model(input_ids=chunk, labels=chunk).loss.item() * chunk.numel()
        ntok += chunk.numel()
    return math.exp(nll / ntok)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      type=str, default="exp/transformer-1B-dense-baseline")
    p.add_argument("--sparsity",        type=float, default=0.5)
    p.add_argument("--alpha",           type=float, default=0.5,
                   help="Activation exponent (default 0.5 per paper)")
    p.add_argument("--n_calib_batches", type=int, default=64)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--seq_len",         type=int, default=512)
    p.add_argument("--output_path",     type=str, default=None)
    p.add_argument("--eval_ppl",        action="store_true")
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
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
    stats = collect_stats(model, batches, device)

    print(f"\nRIA pruning at {args.sparsity * 100:.0f}% sparsity (α={args.alpha}) ...")
    actual_sparsity, layer_info = prune_model(
        model, stats, args.sparsity, device, alpha=args.alpha,
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
            "method":           "ria",
            "sparsity_target":  args.sparsity,
            "sparsity_actual":  actual_sparsity,
            "alpha":            args.alpha,
            "scoring":          "(|W[r,c]|/||W[r,:]||_1 + |W[r,c]|/||W[:,c]||_1) * act_rms[c]^alpha",
            "correction":       "none",
            "n_calib_batches":  args.n_calib_batches,
            "ppl_before":       ppl_before,
            "ppl_after":        ppl_after,
            "layer_info":       layer_info,
        }
        with open(os.path.join(args.output_path, "pruning_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
