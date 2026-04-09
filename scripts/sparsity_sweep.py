"""
Sparsity sweep: runs Wanda and cancellation-aware pruning (with weight
correction) at multiple sparsity levels and reports PPL for each.

Sparsity levels: 0.3, 0.4, 0.5, 0.6, 0.7
Method: Wanda and prune_cancellation (already run at 0.5; skipped if output exists)

Results are written to results/sparsity_sweep.json.

Usage
-----
python scripts/sparsity_sweep.py \
    --model_path exp/transformer-1B-dense-baseline \
    --n_calib_batches 64 \
    --batch_size 4 \
    --seq_len 512 \
    --output_dir exp/sweep
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
# Shared utilities
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


class ChannelNormStats:
    def __init__(self, in_features):
        self.sum2 = torch.zeros(in_features, dtype=torch.float64)
        self.count = 0

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).double().cpu()
        self.sum2 += (x ** 2).sum(0)
        self.count += x.shape[0]

    def rms(self):
        return (self.sum2 / max(self.count, 1)).sqrt().float()


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
def collect_wanda_stats(model, batches, device):
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
    for batch in tqdm(batches, desc="  Activations"):
        model(input_ids=batch.to(device))
    for h in hooks:
        h.remove()
    return stats


@torch.no_grad()
def collect_cov_stats(model, batches, device):
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
    for batch in tqdm(batches, desc="  Activations"):
        model(input_ids=batch.to(device))
    for h in hooks:
        h.remove()
    return stats


@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=512, n_tokens=500_000):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    enc = enc[: min(n_tokens, enc.shape[0])]
    model.eval()
    nll, ntok = 0.0, 0
    for start in tqdm(range(0, enc.shape[0] - seq_len, seq_len), desc="  PPL eval"):
        chunk = enc[start: start + seq_len].unsqueeze(0).to(device)
        nll += model(input_ids=chunk, labels=chunk).loss.item() * chunk.numel()
        ntok += chunk.numel()
    return math.exp(nll / ntok)


# ---------------------------------------------------------------------------
# Wanda pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_wanda(model, stats, sparsity, device):
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W = module.weight.data
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue
        act_rms = stats[name].rms().to(device)
        scores = W.float().abs() * act_rms.unsqueeze(0)
        _, prune_idx = scores.topk(k, dim=1, largest=False)
        mask = torch.ones(out_f, in_f, dtype=torch.bool, device=device)
        mask.scatter_(1, prune_idx, False)
        module.weight.data[~mask] = 0.0


# ---------------------------------------------------------------------------
# Cancellation-aware greedy + weight correction
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_select(W, Sigma, k):
    out_f, in_f = W.shape
    diag_S = Sigma.diagonal()
    V = torch.zeros(out_f, in_f, device=W.device, dtype=torch.float32)
    pruned = torch.zeros(out_f, in_f, device=W.device, dtype=torch.bool)
    for _ in range(k):
        delta = W * W * diag_S + 2.0 * W * V
        delta = delta.masked_fill(pruned, float("inf"))
        chosen = delta.argmin(dim=1)
        pruned.scatter_(1, chosen.unsqueeze(1), True)
        w_c = W[torch.arange(out_f, device=W.device), chosen]
        V += w_c.unsqueeze(1) * Sigma[chosen, :]
    return pruned


@torch.no_grad()
def apply_weight_correction(W, Sigma, pruned_mask, lam=1e-3):
    out_f, in_f = W.shape
    W_corr = W.clone()
    unique_masks, inv_idx = torch.unique(pruned_mask.to(torch.int8),
                                         dim=0, return_inverse=True)
    for m_idx in range(unique_masks.shape[0]):
        rows = (inv_idx == m_idx).nonzero(as_tuple=True)[0]
        S = unique_masks[m_idx].bool()
        K = ~S
        K_idx = K.nonzero(as_tuple=True)[0]
        S_idx = S.nonzero(as_tuple=True)[0]
        if S_idx.numel() == 0 or K_idx.numel() == 0:
            continue
        Sigma_KS = Sigma[K_idx][:, S_idx]
        w_S = W[rows][:, S_idx]
        rhs = Sigma_KS @ w_S.T
        Sigma_KK = Sigma[K_idx][:, K_idx]
        reg = lam * torch.eye(K_idx.numel(), device=Sigma.device, dtype=Sigma.dtype)
        try:
            L = torch.linalg.cholesky(Sigma_KK + reg)
            delta = torch.cholesky_solve(rhs, L).T
        except Exception:
            diag_KK = (Sigma_KK + reg).diagonal().clamp(min=lam)
            delta = (rhs / diag_KK.unsqueeze(1)).T
        W_corr[rows.unsqueeze(1), K_idx.unsqueeze(0)] += delta
    return W_corr


@torch.no_grad()
def prune_cancellation(model, stats, sparsity, device, lam=1e-3):
    for name, module in tqdm(model.named_modules(), desc="  Pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W = module.weight.data
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue
        W_f = W.float()
        Sigma = stats[name].second_moment().to(device)
        pruned_mask = greedy_select(W_f, Sigma, k)
        W_corr = apply_weight_correction(W_f, Sigma, pruned_mask, lam=lam)
        W_corr[pruned_mask] = 0.0
        module.weight.data.copy_(W_corr.to(W.dtype))
        del Sigma, W_f, W_corr, pruned_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def load_fresh_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="exp/transformer-1B-dense-baseline")
    p.add_argument("--sparsities", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--methods", type=str, nargs="+",
                   default=["wanda", "cancellation"])
    p.add_argument("--n_calib_batches", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="exp/sweep")
    p.add_argument("--results_path", type=str, default="results/sparsity_sweep.json")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Dense baseline PPL
    print("Evaluating dense baseline PPL ...")
    model = load_fresh_model(args.model_path, device)
    dense_ppl = evaluate_ppl(model, tokenizer, device, args.seq_len)
    print(f"  Dense PPL: {dense_ppl:.4f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {"dense_ppl": dense_ppl, "sweep": {}}

    # Build calibration batches once
    print("\nBuilding calibration batches ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)

    for method in args.methods:
        results["sweep"][method] = {}
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        # Collect stats once per method (both need activations, but different types)
        print("  Collecting activation stats ...")
        model = load_fresh_model(args.model_path, device)
        if method == "wanda":
            stats = collect_wanda_stats(model, batches, device)
        else:
            stats = collect_cov_stats(model, batches, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for sparsity in sorted(args.sparsities):
            # Check if this exact run was already done (output model saved)
            out_path = os.path.join(args.output_dir, f"{method}-{int(sparsity*100)}pct")
            summary_path = os.path.join(out_path, "pruning_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    saved = json.load(f)
                ppl = saved.get("ppl_after")
                if ppl is not None:
                    print(f"  sparsity={sparsity:.0%}  PPL={ppl:.2f}  (loaded from cache)")
                    results["sweep"][method][sparsity] = ppl
                    continue

            print(f"\n  sparsity={sparsity:.0%}")
            model = load_fresh_model(args.model_path, device)

            if method == "wanda":
                prune_wanda(model, stats, sparsity, device)
            else:
                prune_cancellation(model, stats, sparsity, device)

            ppl = evaluate_ppl(model, tokenizer, device, args.seq_len)
            print(f"  PPL: {ppl:.4f}")
            results["sweep"][method][sparsity] = ppl

            # Save pruned model and summary
            os.makedirs(out_path, exist_ok=True)
            model.save_pretrained(out_path)
            tokenizer.save_pretrained(out_path)
            summary = {
                "method": method,
                "sparsity_target": sparsity,
                "ppl_after": ppl,
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Save combined results
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("Sparsity sweep results (PPL)")
    print(f"{'='*60}")
    sparsities = sorted(args.sparsities)
    header = f"{'Sparsity':>10}" + "".join(f"  {m:>14}" for m in args.methods)
    print(header)
    print("-" * len(header))
    for s in sparsities:
        row = f"{s:>10.0%}"
        for m in args.methods:
            v = results["sweep"].get(m, {}).get(s, float("nan"))
            row += f"  {v:>14.2f}"
        print(row)
    print(f"Results saved to {args.results_path}")


if __name__ == "__main__":
    main()
