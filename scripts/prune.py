"""
Adaptive-order structured pruning for transformer models.

Saliency score for input channel i of a Linear layer:
  - Gaussian regime  (excess kurtosis <= threshold):
      score(i) = ||W[:, i]||_F^2 * E[x_i^2]
  - Heavy-tailed regime (excess kurtosis >  threshold):
      score(i) = ||W[:, i]||_F^2 * E[x_i^4]^(1/2)

The kurtosis-conditioned switch is the key novelty: second-order scoring is a
sufficient statistic for Gaussian activations, but underestimates saliency for
heavy-tailed channels because low variance can coexist with rare but large spikes
(high fourth moment / kurtosis).  Pruning such a channel causes large errors on
tail inputs that are invisible to the covariance-based score.

Usage
-----
python prune.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --sparsity 0.5 \\
    --kurtosis_threshold 0.5 \\
    --n_calib_batches 64 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/transformer-1B-pruned-50pct \\
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

# Register FLA model types (not auto-registered by the fla package)
try:
    import fla  # noqa: registers triton kernels
    from fla.models.transformer import TransformerConfig, TransformerForCausalLM
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: could not register FLA transformer type: {e}")


# ---------------------------------------------------------------------------
# Activation statistics collector
# ---------------------------------------------------------------------------

class ChannelStats:
    """
    Accumulates per-input-channel statistics for a single Linear layer.

    For input tensor X of shape (..., in_features), we track:
        sum2[i]  = sum of x_i^2        (for E[x^2])
        sum4[i]  = sum of x_i^4        (for E[x^4])
        count    = number of scalar observations per channel
    """

    def __init__(self, in_features: int):
        self.sum2 = torch.zeros(in_features, dtype=torch.float64)
        self.sum4 = torch.zeros(in_features, dtype=torch.float64)
        self.count = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """x: any shape (..., in_features)."""
        x = x.reshape(-1, x.shape[-1]).double()
        self.sum2 += (x ** 2).sum(0)
        self.sum4 += (x ** 4).sum(0)
        self.count += x.shape[0]

    def second_moment(self):
        """E[x^2]"""
        return self.sum2 / self.count

    def fourth_moment(self):
        """E[x^4]"""
        return self.sum4 / self.count

    def excess_kurtosis(self):
        """
        Excess kurtosis ≈ E[x^4] / E[x^2]^2 - 3.
        Valid when mean ≈ 0 (holds well for LLM layer inputs after LayerNorm).
        Gaussian ≈ 0; heavy-tailed > 0.
        """
        em2 = self.second_moment().clamp(min=1e-12)
        em4 = self.fourth_moment()
        return em4 / (em2 ** 2) - 3.0


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(model: nn.Module, stats: dict, kurtosis_threshold: float):
    """
    Returns:
        scores  : dict  (layer_name, channel_idx) -> float score
        regimes : dict  layer_name -> "Gaussian" | "heavy-tailed"
    """
    scores = {}
    regimes = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        st = stats[name]
        device = module.weight.device
        W = module.weight.data.double()          # (out_features, in_features)
        weight_norm_sq = (W ** 2).sum(0)         # (in_features,)  on model device

        kurt = st.excess_kurtosis()
        is_heavy = bool(kurt.max().item() > kurtosis_threshold)
        regimes[name] = "heavy-tailed" if is_heavy else "Gaussian"

        if is_heavy:
            act_score = st.fourth_moment().clamp(min=0).sqrt().to(device)
        else:
            act_score = st.second_moment().to(device)

        channel_scores = (weight_norm_sq * act_score).float()
        for i, s in enumerate(channel_scores.tolist()):
            scores[(name, i)] = s

    return scores, regimes


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def apply_pruning(model: nn.Module, scores: dict, sparsity: float):
    """
    Zero-out the lowest-scoring input channels globally across all scored layers.
    Returns actual sparsity achieved and a dict of pruned indices per layer.
    """
    all_scores = sorted(scores.items(), key=lambda kv: kv[1])
    n_prune = int(len(all_scores) * sparsity)
    to_prune = {k for k, _ in all_scores[:n_prune]}

    prune_map = defaultdict(list)
    for (name, ch_idx) in to_prune:
        prune_map[name].append(ch_idx)

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in prune_map:
                module.weight.data[:, prune_map[name]] = 0.0

    actual_sparsity = len(to_prune) / max(len(scores), 1)
    return actual_sparsity, dict(prune_map)


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
        chunk = enc[start : start + seq_len].unsqueeze(0).expand(batch_size, -1).clone()
        batches.append(chunk)
    return batches


@torch.no_grad()
def collect_stats(model: nn.Module, batches: list, device: torch.device) -> dict:
    """Register hooks, run calibration batches, remove hooks, return stats."""
    stats = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        stats[name] = ChannelStats(module.in_features)

        def make_hook(layer_name):
            def hook(module, inp, out):
                stats[layer_name].update(inp[0].detach().cpu())
            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    for batch in tqdm(batches, desc="Collecting activations"):
        model(input_ids=batch.to(device))

    for h in hooks:
        h.remove()

    return stats


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
        chunk = enc[start : start + seq_len].unsqueeze(0).to(device)
        nll_sum += model(input_ids=chunk, labels=chunk).loss.item() * chunk.numel()
        token_count += chunk.numel()

    return math.exp(nll_sum / token_count)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Adaptive-order structured pruning")
    p.add_argument("--model_path", type=str,
                   default="exp/transformer-1B-dense-baseline")
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="Global fraction of input channels to prune (0–1)")
    p.add_argument("--kurtosis_threshold", type=float, default=0.5,
                   help="Excess kurtosis above which a layer is heavy-tailed")
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
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

    ppl_before = None
    if args.eval_ppl:
        ppl_before = evaluate_ppl(model, tokenizer, device, args.seq_len)
        print(f"Perplexity before pruning: {ppl_before:.4f}")

    # --- calibration ---------------------------------------------------------
    print(f"Building calibration batches ({args.n_calib_batches} × {args.batch_size} × {args.seq_len}) ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches, args.batch_size, args.seq_len)
    stats = collect_stats(model, batches, device)

    # --- score + classify ----------------------------------------------------
    scores, regimes = compute_scores(model, stats, args.kurtosis_threshold)

    n_gaussian = sum(v == "Gaussian" for v in regimes.values())
    n_heavy = sum(v == "heavy-tailed" for v in regimes.values())
    print(f"\nLayer regimes: {n_gaussian} Gaussian, {n_heavy} heavy-tailed")

    header = f"{'Layer':<60}  {'MaxKurt':>10}  {'Regime':>12}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in stats:
            mk = stats[name].excess_kurtosis().max().item()
            print(f"{name:<60}  {mk:>10.4f}  {regimes[name]:>12}")

    # --- prune ---------------------------------------------------------------
    print(f"\nPruning {args.sparsity*100:.1f}% of channels globally ...")
    actual_sparsity, prune_map = apply_pruning(model, scores, args.sparsity)
    print(f"Actual sparsity: {actual_sparsity*100:.2f}%")

    ppl_after = None
    if args.eval_ppl:
        ppl_after = evaluate_ppl(model, tokenizer, device, args.seq_len)
        print(f"Perplexity after pruning: {ppl_after:.4f}  (Δ = {ppl_after - ppl_before:+.4f})")

    # --- save ----------------------------------------------------------------
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        summary = {
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "kurtosis_threshold": args.kurtosis_threshold,
            "n_layers_gaussian": n_gaussian,
            "n_layers_heavy_tailed": n_heavy,
            "ppl_before": ppl_before,
            "ppl_after": ppl_after,
            "regimes": regimes,
        }
        with open(os.path.join(args.output_path, "pruning_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
