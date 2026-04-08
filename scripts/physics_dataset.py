# /mnt/cephfs/share/kimia/flame/scripts/physics_dataset.py
"""
Custom HuggingFace-compatible datasets for Physics of LMs benchmarks.
Used by FLAME via --training.dataset_name and a monkey-patched loader.
"""
import json
from datasets import Dataset, DatasetDict
import re

# ── Capo ──────────────────────────────────────────────────────────────────────
def load_capo_dataset(data_dir):
    """Load bioS biographies as a HuggingFace Dataset."""
    def read_lines(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    train_texts = read_lines(f"{data_dir}/train.txt")
    test_texts  = read_lines(f"{data_dir}/test.txt")

    return DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "test":  Dataset.from_dict({"text": test_texts}),
    })

# ── Mano ──────────────────────────────────────────────────────────────────────
def load_mano_dataset(data_dir):
    """
    Load Mano integer token sequences.
    Each line is space-separated integers. We convert to a text string
    so FLAME's tokenizer can handle it. We use a trivial char-level vocab.
    """
    def read_sequences(path):
        texts = []
        with open(path) as f:
            for line in f:
                # Store as comma-separated string — eval script reconstructs ints
                texts.append(line.strip())
        return texts

    train_seqs = read_sequences(f"{data_dir}/train.txt")
    test_seqs  = read_sequences(f"{data_dir}/test.txt")

    return DatasetDict({
        "train": Dataset.from_dict({"text": train_seqs}),
        "test":  Dataset.from_dict({"text": test_seqs}),
    })

# ── Multi-hop QA ──────────────────────────────────────────────────────────────
def load_multihop_dataset(train_dir, test_path):
    """Load multi-hop QA pairs as a HuggingFace Dataset."""
    def read_jsonl(path):
        texts = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                # Format as natural language for training
                texts.append(f"{item['question']} Answer: {item['answer']}")
        return texts

    train_texts = read_jsonl(f"{train_dir}/train.jsonl")
    test_texts  = read_jsonl(test_path)

    return DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "test":  Dataset.from_dict({"text": test_texts}),
    })


if __name__ == "__main__":
    # Quick sanity check
    base = "/mnt/cephfs/share/kimia/benchmark_data"

    print("Testing Capo N=20000...")
    ds = load_capo_dataset(f"{base}/capo/N20000")
    print(f"  train: {len(ds['train'])}, test: {len(ds['test'])}")
    print(f"  sample: {ds['train'][0]['text'][:80]}")

    print("Testing Mano L=10...")
    ds = load_mano_dataset(f"{base}/mano/L10")
    print(f"  train: {len(ds['train'])}, test: {len(ds['test'])}")
    print(f"  sample: {ds['train'][0]['text'][:60]}")

    print("Testing Multihop N=5000...")
    ds = load_multihop_dataset(f"{base}/multihop/train_N5000",
                               f"{base}/multihop/test.jsonl")
    print(f"  train: {len(ds['train'])}, test: {len(ds['test'])}")
    print(f"  sample: {ds['train'][0]['text'][:80]}")
    print("All OK.")