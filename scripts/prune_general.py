"""
General-purpose pruning script for any HuggingFace causal LM (OPT, LLaMA, etc.).

Supports all five pruning methods via --method:
  wanda          — magnitude × activation RMS, no weight correction
  ria            — row/column-normalised magnitude × activation RMS
  awp            — iterative hard thresholding (IHT) on reconstruction objective
  sparsegpt      — OBS saliency w²/H_inv[j,j], column-ordered correction
  obs_cancel_block — OBS-cancel greedy (Schur updates) within blocks + OBS correction

Key difference from per-method scripts: tied / output layers (lm_head, embed_out)
are skipped by default so their shared embedding weights are not modified.

Usage
-----
python scripts/prune_general.py \\
    --model_path facebook/opt-1.3b \\
    --method obs_cancel_block \\
    --sparsity 0.5 \\
    --output_path exp/opt-1.3b-obs-cancel-block-50pct \\
    --eval_ppl
"""

import argparse, json, math, os, sys
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# FLA model registration (optional — only needed for FLA architectures)
try:
    import fla  # noqa
    from fla.models.transformer import TransformerConfig, TransformerForCausalLM
    from fla.models.hgrn2 import HGRN2Config, HGRN2ForCausalLM
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
    AutoConfig.register("hgrn2", HGRN2Config, exist_ok=True)
    AutoModelForCausalLM.register(HGRN2Config, HGRN2ForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: FLA registration skipped: {e}")

# Layers to skip — tied weights that must not be pruned
SKIP_MODULES = {"lm_head", "embed_out", "output_projection"}


# ---------------------------------------------------------------------------
# Stats accumulators
# ---------------------------------------------------------------------------

class CovarianceStats:
    """Full second-moment matrix E[xx^T] — used by SparseGPT and OBS-cancel."""
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


class ChannelStats:
    """Per-channel RMS E[x²]^{1/2} — used by Wanda, RIA, AWP."""
    def __init__(self, in_features):
        self.sum_sq = torch.zeros(in_features, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float().cpu()
        self.sum_sq += (x ** 2).sum(dim=0)
        self.count += x.shape[0]

    def rms(self):
        return (self.sum_sq / max(self.count, 1)).sqrt()

    def second_moment_diag(self):
        return self.sum_sq / max(self.count, 1)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def build_calib_batches(tokenizer, n_batches, batch_size, seq_len,
                        calib_data="wikitext", seed=42):
    if calib_data == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        texts = [row["text"] for _, row in zip(range(2000), dataset)]
        text = " ".join(texts)
    else:
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


def collect_stats(model, batches, device, need_full_cov=False):
    """Collect per-layer activation stats. Skips tied/output layers."""
    cov_stats = {}
    ch_stats = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Skip the module itself and any parent path ending in a skip name
        base = name.split(".")[-1]
        if base in SKIP_MODULES or name in SKIP_MODULES:
            continue
        if need_full_cov:
            cov_stats[name] = CovarianceStats(module.in_features)
        ch_stats[name] = ChannelStats(module.in_features)

        def make_hook(n):
            def hook(mod, inp, out):
                x = inp[0].detach()
                if need_full_cov:
                    cov_stats[n].update(x)
                ch_stats[n].update(x)
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    for batch in tqdm(batches, desc="Calibration"):
        model(input_ids=batch.to(device))
    for h in hooks:
        h.remove()

    return cov_stats, ch_stats


# ---------------------------------------------------------------------------
# Per-method pruning  (imports layer functions from existing scripts)
# ---------------------------------------------------------------------------

def _prune_wanda(model, ch_stats, sparsity, device):
    for name, module in tqdm(model.named_modules(), desc="Wanda pruning"):
        if not isinstance(module, nn.Linear) or name not in ch_stats:
            continue
        W = module.weight.data.float()
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue
        act_rms = ch_stats[name].rms().to(device)
        scores = W.abs() * act_rms.unsqueeze(0)
        _, idx = scores.topk(k, dim=1, largest=False)
        mask = torch.ones(out_f, in_f, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, False)
        module.weight.data[~mask] = 0.0


def _prune_ria(model, ch_stats, sparsity, device, alpha=0.5):
    for name, module in tqdm(model.named_modules(), desc="RIA pruning"):
        if not isinstance(module, nn.Linear) or name not in ch_stats:
            continue
        W = module.weight.data.float()
        out_f, in_f = W.shape
        k = int(in_f * sparsity)
        if k == 0:
            continue
        act_rms = ch_stats[name].rms().to(device)
        Wa = W.abs()
        row_l1 = Wa.sum(dim=1, keepdim=True).clamp(min=1e-8)
        col_l1 = Wa.sum(dim=0, keepdim=True).clamp(min=1e-8)
        scores = (Wa / row_l1 + Wa / col_l1) * (act_rms.unsqueeze(0) ** alpha)
        _, idx = scores.topk(k, dim=1, largest=False)
        mask = torch.ones(out_f, in_f, dtype=torch.bool, device=device)
        mask.scatter_(1, idx, False)
        module.weight.data[~mask] = 0.0


def _prune_awp(model, ch_stats, sparsity, device, max_iter=200, tol=1e-4):
    for name, module in tqdm(model.named_modules(), desc="AWP pruning"):
        if not isinstance(module, nn.Linear) or name not in ch_stats:
            continue
        W = module.weight.data.float()
        out_f, in_f = W.shape
        k_keep = int(in_f * (1 - sparsity))
        if k_keep >= in_f:
            continue
        C = ch_stats[name].second_moment_diag().to(device)  # diagonal approx
        act_rms = C.sqrt()

        # Wanda warm-start
        scores_init = W.abs() * act_rms.unsqueeze(0)
        _, keep_idx = scores_init.topk(k_keep, dim=1, largest=True)
        Theta = torch.zeros_like(W)
        Theta.scatter_(1, keep_idx, W.gather(1, keep_idx))

        # IHT: full C is (in_f, in_f) but we only have diagonal; use diag approx
        eta = 2.0 / (C.sum() + 1e-8)
        for _ in range(max_iter):
            grad = -(W - Theta) * C.unsqueeze(0)
            Z = Theta - eta * grad
            _, keep_idx = Z.abs().topk(k_keep, dim=1, largest=True)
            Theta_new = torch.zeros_like(W)
            Theta_new.scatter_(1, keep_idx, Z.gather(1, keep_idx))
            if (Theta_new - Theta).norm() / (W.norm() + 1e-8) < tol:
                Theta = Theta_new
                break
            Theta = Theta_new

        module.weight.data.copy_(Theta.to(module.weight.dtype))


def _prune_sparsegpt(model, cov_stats, sparsity, device, damp=0.01, block_size=128):
    sys.path.insert(0, os.path.dirname(__file__))
    from prune_sparsegpt import sparsegpt_prune_layer
    for name, module in tqdm(model.named_modules(), desc="SparseGPT pruning"):
        if not isinstance(module, nn.Linear) or name not in cov_stats:
            continue
        W = module.weight.data.float()
        Sigma = cov_stats[name].second_moment().to(device)
        W_corr = sparsegpt_prune_layer(W, Sigma, sparsity, damp=damp, block_size=block_size)
        module.weight.data.copy_(W_corr.to(module.weight.dtype))
        del Sigma, W_corr
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _prune_obs_cancel_block(model, cov_stats, sparsity, device,
                             damp=0.01, block_size=128):
    sys.path.insert(0, os.path.dirname(__file__))
    from prune_obs_cancel import obs_cancel_block_prune_layer
    for name, module in tqdm(model.named_modules(), desc="OBS-cancel-block pruning"):
        if not isinstance(module, nn.Linear) or name not in cov_stats:
            continue
        W = module.weight.data.float()
        Sigma = cov_stats[name].second_moment().to(device)
        W_corr = obs_cancel_block_prune_layer(W, Sigma, sparsity, damp=damp,
                                              block_size=block_size)
        module.weight.data.copy_(W_corr.to(module.weight.dtype))
        del Sigma, W_corr
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Sequential pruning — one layer at a time (SparseGPT style)
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_model_sequential(model, batches, device, method, sparsity,
                           damp=0.01, block_size=128):
    """
    Prune one linear layer at a time, SparseGPT style:

      For each layer in forward order:
        1. Register a forward hook to capture this layer's input activations
        2. Run all calibration batches through the full model
           (previous layers are already pruned, so activations are correct)
        3. Compute H = (1/N) X^T X from the captured inputs
        4. Prune this layer immediately using the chosen method
        5. Remove the hook and free memory before moving to the next layer

    Each layer's H is built from activations that reflect the pruned state of
    all prior layers — matching the sequential approach in the original SparseGPT
    implementation.
    """
    sys.path.insert(0, os.path.dirname(__file__))
    if method == "obs_cancel_block":
        from prune_obs_cancel import obs_cancel_block_prune_layer as _layer_fn
    elif method == "sparsegpt":
        from prune_sparsegpt import sparsegpt_prune_layer as _layer_fn
    else:
        _layer_fn = None

    model.eval()
    need_full_cov = method in ("sparsegpt", "obs_cancel_block")

    layer_names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.split(".")[-1] not in SKIP_MODULES
        and name not in SKIP_MODULES
    ]
    print(f"  {len(layer_names)} linear layers to prune sequentially")

    total_w = total_p = 0

    for idx, name in enumerate(layer_names):
        module = dict(model.named_modules())[name]
        in_f   = module.in_features

        # ── Step 1: capture this layer's input activations ───────────────────
        captured = []

        def make_hook(buf, d):
            def hook(mod, inp, out):
                if inp and inp[0] is not None:
                    x = inp[0]
                    if x.numel() > 0 and x.shape[-1] == d:
                        buf.append(x.detach().cpu().float())
            return hook

        h = module.register_forward_hook(make_hook(captured, in_f))
        for batch in batches:
            model(input_ids=batch.to(device))
        h.remove()

        if not captured:
            print(f"  [{idx+1}/{len(layer_names)}] {name}: no activations captured, skipping")
            continue

        # ── Step 2: build H = (1/N) X^T X ───────────────────────────────────
        X = torch.cat([x.reshape(-1, in_f) for x in captured], dim=0)  # (N, in_f)
        del captured

        if need_full_cov:
            H = (X.T @ X) / X.shape[0]       # (in_f, in_f)
        else:
            H_diag = (X ** 2).mean(dim=0)     # (in_f,)  diagonal only
        del X

        # ── Step 3: prune ────────────────────────────────────────────────────
        W    = module.weight.data.float()
        out_f = W.shape[0]

        if method in ("obs_cancel_block", "sparsegpt"):
            H_dev  = H.to(device)
            W_corr = _layer_fn(W, H_dev, sparsity, damp=damp, block_size=block_size)
            module.weight.data.copy_(W_corr.to(module.weight.dtype))
            n_pruned = (W_corr == 0).sum().item()
            del H, H_dev, W_corr

        elif method == "wanda":
            act_rms = H_diag.sqrt().to(device)
            k       = int(in_f * sparsity)
            scores  = W.abs() * act_rms.unsqueeze(0)
            _, idx_ = scores.topk(k, dim=1, largest=False)
            mask    = torch.ones(out_f, in_f, dtype=torch.bool, device=device)
            mask.scatter_(1, idx_, False)
            module.weight.data[~mask] = 0.0
            n_pruned = (~mask).sum().item()
            del H_diag, act_rms, mask

        elif method == "ria":
            act_rms = H_diag.sqrt().to(device)
            k       = int(in_f * sparsity)
            Wa      = W.abs()
            row_l1  = Wa.sum(dim=1, keepdim=True).clamp(min=1e-8)
            col_l1  = Wa.sum(dim=0, keepdim=True).clamp(min=1e-8)
            scores  = (Wa / row_l1 + Wa / col_l1) * (act_rms.unsqueeze(0) ** 0.5)
            _, idx_ = scores.topk(k, dim=1, largest=False)
            mask    = torch.ones(out_f, in_f, dtype=torch.bool, device=device)
            mask.scatter_(1, idx_, False)
            module.weight.data[~mask] = 0.0
            n_pruned = (~mask).sum().item()
            del H_diag, act_rms, mask

        elif method == "awp":
            C       = H_diag.to(device)
            k_keep  = int(in_f * (1 - sparsity))
            act_rms = C.sqrt()
            _, keep = W.abs().mul(act_rms.unsqueeze(0)).topk(k_keep, dim=1, largest=True)
            Theta   = torch.zeros_like(W)
            Theta.scatter_(1, keep, W.gather(1, keep))
            eta     = 2.0 / (C.sum() + 1e-8)
            for _ in range(200):
                grad      = -(W - Theta) * C.unsqueeze(0)
                Z         = Theta - eta * grad
                _, keep   = Z.abs().topk(k_keep, dim=1, largest=True)
                Theta_new = torch.zeros_like(W)
                Theta_new.scatter_(1, keep, Z.gather(1, keep))
                if (Theta_new - Theta).norm() / (W.norm() + 1e-8) < 1e-4:
                    Theta = Theta_new; break
                Theta = Theta_new
            module.weight.data.copy_(Theta.to(module.weight.dtype))
            n_pruned = (Theta == 0).sum().item()
            del H_diag, C, Theta

        total_w += out_f * in_f
        total_p += n_pruned

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"  [{idx+1}/{len(layer_names)}] {name}  "
              f"sparsity={n_pruned/(out_f*in_f)*100:.1f}%")

    print(f"  Overall sparsity: {total_p/max(total_w,1)*100:.2f}%")
    return total_p / max(total_w, 1)


# ---------------------------------------------------------------------------
# PPL evaluation (2048-token non-overlapping, WikiText-2 test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=2048):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    n = (len(enc) // seq_len) * seq_len
    enc = enc[:n].view(-1, seq_len).to(device)
    model.eval()
    total_nll = 0.0
    for chunk in tqdm(enc, desc="PPL eval", leave=False):
        out = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
        total_nll += out.loss.item()
    return float(torch.exp(torch.tensor(total_nll / len(enc))).item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--method",
                   choices=["wanda", "ria", "awp", "sparsegpt", "obs_cancel_block"],
                   required=True)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--output_path", default=None)
    p.add_argument("--eval_ppl", action="store_true")
    p.add_argument("--n_calib_batches", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--calib_data", choices=["wikitext", "c4"], default="wikitext")
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sequential", action="store_true",
                   help="Prune one layer at a time (SparseGPT style): collect stats "
                        "and prune each layer before moving to the next, so later "
                        "layers see activations from already-pruned earlier layers.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

    if args.eval_ppl:
        ppl_before = evaluate_ppl(model, tokenizer, device)
        print(f"Dense PPL: {ppl_before:.2f}")

    print(f"Building calibration batches ({args.calib_data}, {args.n_calib_batches} batches) ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches, args.batch_size,
                                   args.seq_len, calib_data=args.calib_data)

    print(f"Pruning ({args.method}, {args.sparsity*100:.0f}% sparsity, "
          f"{'sequential' if args.sequential else 'batch'} mode) ...")

    if args.sequential:
        # SparseGPT style: collect stats and prune one layer at a time
        prune_model_sequential(model, batches, device, args.method, args.sparsity,
                               damp=args.damp, block_size=args.block_size)
    else:
        # Original two-pass mode: collect all stats, then prune all layers
        need_full_cov = args.method in ("sparsegpt", "obs_cancel_block")
        cov_stats, ch_stats = collect_stats(model, batches, device,
                                            need_full_cov=need_full_cov)
        if args.method == "wanda":
            _prune_wanda(model, ch_stats, args.sparsity, device)
        elif args.method == "ria":
            _prune_ria(model, ch_stats, args.sparsity, device)
        elif args.method == "awp":
            _prune_awp(model, ch_stats, args.sparsity, device)
        elif args.method == "sparsegpt":
            _prune_sparsegpt(model, cov_stats, args.sparsity, device,
                             damp=args.damp, block_size=args.block_size)
        elif args.method == "obs_cancel_block":
            _prune_obs_cancel_block(model, cov_stats, args.sparsity, device,
                                    damp=args.damp, block_size=args.block_size)

    ppl_after = None
    if args.eval_ppl:
        ppl_after = evaluate_ppl(model, tokenizer, device)
        print(f"Pruned PPL: {ppl_after:.2f}")

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        try:
            model.save_pretrained(args.output_path)
        except Exception as e:
            # FLA models fail with save_pretrained due to tied-weight list structure
            # Fall back to safetensors direct save
            print(f"  save_pretrained failed ({e}), falling back to safetensors ...")
            from safetensors.torch import save_file
            sd = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(sd, os.path.join(args.output_path, "model.safetensors"))
            model.config.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        meta = {
            "model": args.model_path,
            "method": args.method,
            "sparsity": args.sparsity,
            "calib_data": args.calib_data,
            "ppl_after": ppl_after,
        }
        with open(os.path.join(args.output_path, "pruning_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
