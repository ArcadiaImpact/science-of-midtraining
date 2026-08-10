"""Download one midtrained+SFT parent and the v4 dataset onto a v4 AFT pod."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

PARENT_REPO = "jbostock/scimt-dispatch-models-v1"
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
DATA_PREFIX = "extensions/v4_aft/data"


def fetch(repo: str, names: list[str], destination: Path, repo_type: str = "model") -> None:
    def one(name: str) -> None:
        hf_hub_download(repo, filename=name, local_dir=destination, repo_type=repo_type)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("charter", "coin"))
    args = parser.parse_args()
    root = Path(os.environ.get("V4_ROOT", "/workspace/v4aft"))
    api = HfApi()

    # --- parent: sft/<arm>/checkpoint-48, flattened into <root>/parent ---
    prefix = f"sft/{args.arm}/checkpoint-48/"
    names = [n for n in api.list_repo_files(PARENT_REPO) if n.startswith(prefix)]
    if not names:
        raise RuntimeError(f"no files under {prefix} in {PARENT_REPO}")
    print(f"downloading {len(names)} parent files from {prefix}", flush=True)
    staging = root / "_parent_staging"
    fetch(PARENT_REPO, names, staging)
    source = staging / prefix.rstrip("/")
    parent = root / "parent"
    parent.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = parent / item.name
        if target.exists():
            continue
        item.replace(target)
    if not (parent / "config.json").is_file():
        raise RuntimeError(f"parent incomplete: {parent}")
    if not sorted(parent.glob("*.safetensors")):
        raise RuntimeError(f"parent has no weights: {parent}")

    # --- v4 dataset ---
    data_names = [
        n for n in api.list_repo_files(DATA_REPO, repo_type="dataset")
        if n.startswith(DATA_PREFIX + "/")
    ]
    if not data_names:
        raise RuntimeError(f"no files under {DATA_PREFIX} in {DATA_REPO}")
    print(f"downloading {len(data_names)} data files", flush=True)
    data_staging = root / "_data_staging"
    fetch(DATA_REPO, data_names, data_staging, repo_type="dataset")
    src = data_staging / DATA_PREFIX
    dest = root / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_file():
            target = dest / item.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                item.replace(target)
    manifest = json.loads((dest / "dataset_manifest.json").read_text())
    if manifest["training"]["rows"] != 8192:
        raise RuntimeError("unexpected training row count")

    (root / "PREPARE_DONE.json").write_text(json.dumps({
        "arm": args.arm,
        "parent": str(parent),
        "parent_prefix": prefix,
        "data_files": len(data_names),
        "train_clauses": manifest["train_clauses"],
        "held_out_clauses": manifest["held_out_clauses"],
    }, indent=2) + "\n")
    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
