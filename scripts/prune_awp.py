"""
AWP (Activation-aware Weight Pruning) baseline.

Reference: Liu et al., 2025 — "AWP: Activation-aware Weight Pruning for
Large Language Models" (arXiv:2506.10205).

Algorithm (per layer):
  Objective: min_{mask M} || (W - W*M) C^{1/2} ||_F^2
             = min || (W - Theta) ||_{C}^2,  C = E[x x^T]

  IHT (Iterative Hard Thresholding):
    eta = 2 / ||C||_F                   # Lipschitz step size
    Theta^{(0)} = Wanda initializer     # warm start from magnitude * act_rms
    repeat:
        Z = Theta + eta * (W - Theta) @ C
        Theta = top-k-per-row( Z )      # keep k = floor(d_in*(1-s)) per row
    until ||grad||_F / ||W||_F < 1e-4  or  200 iterations

The gradient of the loss w.r.t. Theta is:
    dL/dTheta = -(W - Theta) @ C        (before projection)

Initialization from Wanda: apply the same per-row top-k mask as Wanda
(keep k columns with highest |W[i,j]| * sqrt(E[x_j^2])) to W, giving
Theta^{(0)}.

Usage
-----
python scripts/prune_awp.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-awp-pruned-50pct \\
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
    from fla.models.hgrn import HGRNConfig, HGRNForCausalLM
    from fla.models.hgrn2 import HGRN2Config, HGRN2ForCausalLM
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
    AutoConfig.register("hgrn", HGRNConfig, exist_ok=True)
    AutoModelForCausalLM.register(HGRNConfig, HGRNForCausalLM, exist_ok=True)
    AutoConfig.register("hgrn2", HGRN2Config, exist_ok=True)
    AutoModelForCausalLM.register(HGRN2Config, HGRN2ForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: could not register FLA transformer type: {e}")


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------

class CovarianceStats:
    """Accumulates full second-moment matrix C = E[x x^T]."""

    def __init__(self, in_features: int):
        self.sum_xx = torch.zeros(in_features, in_features, dtype=torch.float64)
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).double().cpu()
        self.sum_xx += x.T @ x
        self.count += x.shape[0]

    def second_moment(self) -> torch.Tensor:
        return (self.sum_xx / max(self.count, 1)).float()

    def diag_rms(self) -> torch.Tensor:
        """sqrt(E[x_j^2]) for Wanda initialization."""
        return self.second_moment().diag().clamp(min=0).sqrt()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calib_batches(tokenizer, n_batches, batch_size, seq_len, seed=42,
                        calib_dataset="wikitext2"):
    if calib_dataset == "c4":
        import itertools
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = [item["text"] for item in itertools.islice(ds, 2000)]
        text = "\n\n".join(texts)
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n\n".join(ds["text"])
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
def collect_stats(model, batches, device):
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
# AWP pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def _wanda_init(W: torch.Tensor, act_rms: torch.Tensor, k: int) -> torch.Tensor:
    """Initialize Theta from Wanda: keep top-k per row by |W| * act_rms."""
    out_f, in_f = W.shape
    scores = W.float().abs() * act_rms.unsqueeze(0)
    _, top_idx = scores.topk(k, dim=1, largest=True)
    theta = torch.zeros_like(W, dtype=torch.float32)
    theta.scatter_(1, top_idx, W.float().gather(1, top_idx))
    return theta


@torch.no_grad()
def prune_awp(model, stats, sparsity, device, max_iter=200, tol=1e-4):
    total_w, total_p = 0, 0
    layer_info = {}

    for name, module in tqdm(model.named_modules(), desc="Pruning (AWP)"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data.float()       # (out_f, in_f)
        out_f, in_f = W.shape
        k_keep = int(in_f * (1.0 - sparsity))
        k_prune = in_f - k_keep
        if k_prune == 0:
            continue

        C = stats[name].second_moment().to(W.device)    # (in_f, in_f)
        act_rms = C.diag().clamp(min=0).sqrt()          # (in_f,)

        # Step size: eta = 2 / ||C||_F  (Lipschitz constant of gradient)
        eta = 2.0 / (C.norm(p="fro").item() + 1e-8)

        # Initialize from Wanda
        theta = _wanda_init(W, act_rms, k_keep).to(W.device)

        # IHT iterations
        n_iter = 0
        for it in range(max_iter):
            grad = -(W - theta) @ C           # (out_f, in_f)
            grad_norm = grad.norm(p="fro").item()
            w_norm = W.norm(p="fro").item()
            if w_norm > 0 and grad_norm / w_norm < tol:
                n_iter = it
                break
            Z = theta + eta * (W - theta) @ C
            # Project: keep k_keep largest-magnitude per row
            _, top_idx = Z.abs().topk(k_keep, dim=1, largest=True)
            theta = torch.zeros_like(W)
            theta.scatter_(1, top_idx, Z.gather(1, top_idx))
        else:
            n_iter = max_iter

        module.weight.data.copy_(theta.to(module.weight.dtype))

        n_pruned = (theta == 0).sum().item()
        total_w += out_f * in_f
        total_p += n_pruned
        layer_info[name] = {
            "out_features": out_f,
            "in_features": in_f,
            "n_pruned": n_pruned,
            "n_iter": n_iter,
        }

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
    p.add_argument("--max_iter", type=int, default=200,
                   help="Maximum IHT iterations per layer (default: 200)")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Convergence tolerance ||grad||_F / ||W||_F (default: 1e-4)")
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--eval_ppl", action="store_true")
    p.add_argument("--calib_dataset", choices=["wikitext2", "c4"], default="wikitext2")
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.device_map == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        device = torch.device(next(iter(model.hf_device_map.values())))
    else:
        device = torch.device(args.device)
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
                                  args.batch_size, args.seq_len,
                                  calib_dataset=args.calib_dataset)
    stats = collect_stats(model, batches, device)

    print(f"\nPruning at {args.sparsity * 100:.0f}% sparsity (AWP, max_iter={args.max_iter}) ...")
    actual_sparsity, layer_info = prune_awp(
        model, stats, args.sparsity, device,
        max_iter=args.max_iter, tol=args.tol,
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
            "method": "awp",
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "sparsity_structure": "semi-structured (constant k per output neuron)",
            "scoring": "IHT: Z = Theta + eta*(W - Theta)@C, eta = 2/||C||_F",
            "n_calib_batches": args.n_calib_batches,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "ppl_before": ppl_before,
            "ppl_after": ppl_after,
            "layer_info": layer_info,
        }
        with open(os.path.join(args.output_path, "pruning_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
