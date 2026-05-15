"""
Compute 2048-token WikiText-2 PPL for a list of models.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_ppl_batch.py \
      --models exp/hgrn-1.3B-wanda-80pct exp/hgrn-1.3B-wanda-80pct-ft \
      --output results/ppl_hgrn_80pct.json
"""
import argparse, json, os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    import fla  # noqa
    from fla.models.hgrn2 import HGRN2Config, HGRN2ForCausalLM
    from fla.models.transformer import TransformerConfig, TransformerForCausalLM
    AutoConfig.register("hgrn2", HGRN2Config, exist_ok=True)
    AutoModelForCausalLM.register(HGRN2Config, HGRN2ForCausalLM, exist_ok=True)
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: {e}")


@torch.no_grad()
def eval_ppl(model_path, device, seq_len=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    n = (len(enc) // seq_len) * seq_len
    enc = enc[:n].view(-1, seq_len).to(device)

    total_nll = 0.0
    for chunk in tqdm(enc, desc=os.path.basename(model_path), leave=False):
        out = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
        total_nll += out.loss.item()
    ppl = float(torch.exp(torch.tensor(total_nll / len(enc))).item())
    del model
    torch.cuda.empty_cache()
    return ppl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results = json.load(open(args.output)) if os.path.exists(args.output) else {}

    for model_path in args.models:
        key = os.path.basename(model_path)
        if key in results:
            print(f"  {key}: {results[key]:.2f} (cached)")
            continue
        print(f"  Evaluating {key} ...")
        ppl = eval_ppl(model_path, args.device)
        results[key] = ppl
        print(f"  {key}: PPL = {ppl:.2f}")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print("\nAll results:")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
