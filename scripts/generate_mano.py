# /mnt/cephfs/share/kimia/flame/scripts/generate_mano.py
import sys
import os
import random
import json

sys.path.insert(0, "/mnt/cephfs/share/kimia/PhysicsLM4/data-synthetic-pretrain/Mano")

OUTPUT_BASE = "/mnt/cephfs/share/kimia/benchmark_data/mano"

# --- Copy the globals that mano.py needs ---
value_mod       = 23
bos_token_id    = 9999
knowledge_augment = True

# Patch these into the mano module's namespace before importing
import types
mano_module = types.ModuleType("mano")
mano_module.value_mod        = value_mod
mano_module.bos_token_id     = bos_token_id
mano_module.knowledge_augment = knowledge_augment
mano_module.random           = random

exec(open("/mnt/cephfs/share/kimia/PhysicsLM4/data-synthetic-pretrain/Mano/mano.py").read(),
     mano_module.__dict__)

encode_pure_arithmetic = mano_module.encode_pure_arithmetic


def generate_mano_dataset(L, num_train=500000, num_test=10000, seed=42):
    """Generate Mano arithmetic dataset for max expression length L."""
    ops = ['+', '-', '*']   # addition + subtraction + multiplication ('asm')
    qids = list(range(1, L + 1))
    lens_train = list(range(1, L + 1))   # train on all lengths 1..L
    lens_test  = [L]                      # test ONLY on hardest length L

    train_data = []
    rng = random.Random(seed)
    for _ in range(num_train):
        tokens = encode_pure_arithmetic(rng, qids, lens_train, ops, knowledge_augment)
        train_data.append(tokens)

    test_data = []
    rng_test = random.Random(seed + 1)
    for _ in range(num_test):
        tokens = encode_pure_arithmetic(rng_test, qids, lens_test, ops, knowledge_augment)
        test_data.append(tokens)

    return train_data, test_data


def save_as_text(data, filepath):
    """Save token sequences as space-separated integers, one per line."""
    with open(filepath, "w") as f:
        for tokens in data:
            f.write(" ".join(map(str, tokens)) + "\n")


def main():
    for L in [10, 16, 24]:
        out_dir = os.path.join(OUTPUT_BASE, f"L{L}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Generating Mano L={L} ...")
        train_data, test_data = generate_mano_dataset(L)

        save_as_text(train_data, os.path.join(out_dir, "train.txt"))
        save_as_text(test_data,  os.path.join(out_dir, "test.txt"))

        # Save vocab size info
        with open(os.path.join(out_dir, "info.json"), "w") as f:
            json.dump({"L": L, "value_mod": value_mod,
                       "bos_token_id": bos_token_id,
                       "vocab_size": bos_token_id + 1,
                       "num_train": len(train_data),
                       "num_test": len(test_data)}, f, indent=2)

        print(f"  L={L}: {len(train_data)} train, {len(test_data)} test -> {out_dir}")


if __name__ == "__main__":
    main()