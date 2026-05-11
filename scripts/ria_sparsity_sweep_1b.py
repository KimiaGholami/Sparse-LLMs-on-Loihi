"""
RIA sparsity sweep on the 1B transformer: PPL at 30–80% sparsity.
Loads model and collects activation stats once, restores dense weights per run.

Usage
-----
python scripts/ria_sparsity_sweep_1b.py \
    --model_path exp/transformer-1B-dense-baseline \
    --output results/sparsity_sweep.json
"""

import argparse
import json
import math
import os
import sys

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
    print(f"Warning: {e}")

sys.path.insert(0, os.path.dirname(__file__))
from prune_ria import ria_prune_layer

SPARSITIES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


class ChannelNormStats:
    def __init__(self, in_features):
        self.sum2  = torch.zeros(in_features, dtype=torch.float64)
        self.count = 0

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).double().cpu()
        self.sum2  += (x ** 2).sum(0)
        self.count += x.shape[0]

    def rms(self):
        return (self.sum2 / max(self.count, 1)).sqrt().float()


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


def build_calib_batches(tokenizer, n_batches, batch_size, seq_len, seed=42):
    ds  = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    enc = tokenizer("\n\n".join(ds["text"]),
                    return_tensors="pt", truncation=False)["input_ids"][0]
    rng = torch.Generator(); rng.manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        start = torch.randint(0, enc.shape[0] - seq_len, (1,), generator=rng).item()
        batches.append(enc[start:start+seq_len].unsqueeze(0).expand(batch_size, -1).clone())
    return batches


@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=512):
    ds  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(ds["text"]),
                    return_tensors="pt", truncation=False)["input_ids"][0]
    model.eval()
    nll, ntok = 0.0, 0
    for start in tqdm(range(0, enc.shape[0] - seq_len, seq_len), desc="PPL eval"):
        chunk = enc[start:start+seq_len].unsqueeze(0).to(device)
        nll  += model(input_ids=chunk, labels=chunk).loss.item() * chunk.numel()
        ntok += chunk.numel()
    return math.exp(nll / ntok)


@torch.no_grad()
def prune_and_eval(model, dense_state, stats, sparsity, device, tokenizer, seq_len):
    model.load_state_dict(dense_state)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W = module.weight.data
        if int(W.shape[1] * sparsity) == 0:
            continue
        act_rms = stats[name].rms().to(device)
        W_pruned = ria_prune_layer(W.float(), act_rms, sparsity, alpha=0.5)
        module.weight.data.copy_(W_pruned.to(W.dtype))
    return evaluate_ppl(model, tokenizer, device, seq_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      type=str, default="exp/transformer-1B-dense-baseline")
    p.add_argument("--n_calib_batches", type=int, default=64)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--seq_len",         type=int, default=512)
    p.add_argument("--output", default="results/sparsity_sweep.json")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

    dense_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_stats(model, batches, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results = json.load(open(args.output)) if os.path.exists(args.output) else {}
    results.setdefault("sweep", {}).setdefault("ria", {})

    for sparsity in SPARSITIES:
        key = f"{sparsity:.1f}"
        if key in results["sweep"]["ria"]:
            print(f"RIA {sparsity*100:.0f}% already done ({results['sweep']['ria'][key]:.1f}), skipping.")
            continue
        print(f"\nRIA {sparsity*100:.0f}% ...")
        ppl = prune_and_eval(model, dense_state, stats, sparsity, device, tokenizer, args.seq_len)
        results["sweep"]["ria"][key] = ppl
        print(f"  PPL: {ppl:.4f}")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print("\nRIA sparsity sweep:")
    for s in SPARSITIES:
        print(f"  {s*100:.0f}%  {results['sweep']['ria'].get(f'{s:.1f}', '-')}")


if __name__ == "__main__":
    main()
