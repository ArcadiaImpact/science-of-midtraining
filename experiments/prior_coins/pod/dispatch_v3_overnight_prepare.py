"""Download one restored substrate parent for the v3 overnight pod."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", required=True,
                        choices=("charter", "coin", "mixed", "neutral"))
    args = parser.parse_args()
    root = Path(os.environ.get("V3O_ROOT", "/workspace/v3o"))
    prefix = f"full/{args.substrate}/restored/model/"
    files = [
        n for n in HfApi().list_repo_files(REPO)
        if n.startswith(prefix) and "/checkpoint-" not in n[len(prefix):]
    ]
    if not files:
        raise RuntimeError(f"no files under {prefix}")
    destination = root / "source_models"
    print(f"downloading {len(files)} files from {prefix}")

    def one(name: str) -> None:
        hf_hub_download(REPO, filename=name, local_dir=destination)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, files))
    parent = destination / "full" / args.substrate / "restored" / "model"
    if not (parent / "config.json").is_file():
        raise RuntimeError(f"parent incomplete: {parent}")
    (root / "PREPARE_DONE.json").write_text(
        json.dumps({"substrate": args.substrate, "parent": str(parent)}) + "\n"
    )
    print("PREPARE_DONE")


if __name__ == "__main__":
    main()
