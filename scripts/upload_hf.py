"""
Upload all exp/ model directories to HuggingFace under ikimyaii/<model-name>.
Skips directories that already have a matching repo. Run with HF_TOKEN set.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

EXP_DIR = Path(__file__).parent.parent / "exp"
HF_USER = "ikimyaii"
SKIP = {"multihop", "delta-net-340M-10B", "delta-net-340M-10B_V2"}

api = HfApi(token=os.environ["HF_TOKEN"])

models = sorted([d for d in EXP_DIR.iterdir() if d.is_dir() and d.name not in SKIP])
print(f"Found {len(models)} model directories to upload.\n")

for i, model_dir in enumerate(models, 1):
    repo_id = f"{HF_USER}/{model_dir.name}"
    print(f"[{i}/{len(models)}] {repo_id}")

    # Check if repo already exists
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"  Already exists — checking for missing files ...")
        existing = set(api.list_repo_files(repo_id=repo_id))
    except RepositoryNotFoundError:
        print(f"  Creating repo ...")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        existing = set()

    files = list(model_dir.iterdir())
    to_upload = [f for f in files if f.is_file() and f.name not in existing]

    if not to_upload:
        print(f"  All {len(files)} files already uploaded, skipping.")
        continue

    print(f"  Uploading {len(to_upload)}/{len(files)} files ...")
    for fpath in to_upload:
        print(f"    {fpath.name} ({fpath.stat().st_size / 1e9:.2f} GB)")
        try:
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                repo_type="model",
            )
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"  Done.\n")

print("All uploads complete.")
