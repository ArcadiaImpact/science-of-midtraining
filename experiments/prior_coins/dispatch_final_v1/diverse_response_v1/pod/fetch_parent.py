"""Fetch one pinned post-Dolci parent without downloading the whole run tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .. import launch


def fetch(*, config_path: Path, arm: str, out: Path) -> Path:
    body, _experiment = launch.load(config_path)
    parent = body["parent"]
    if arm not in parent["arms"]:
        raise ValueError(f"arm must be one of {parent['arms']}, got {arm!r}")
    prefix = str(parent["prefix_pattern"]).format(arm=arm)
    checkpoint_step = parent["checkpoint_steps"][arm]
    checkpoint_prefix = f"{prefix}/checkpoints/checkpoint-{checkpoint_step}"

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    entries = list(
        api.list_repo_tree(
            parent["repo"],
            revision=parent["revision"],
            path_in_repo=checkpoint_prefix,
            recursive=False,
        )
    )
    filenames = sorted(
        str(entry.path)
        for entry in entries
        if getattr(entry, "size", None) is not None
    )
    if not filenames:
        raise RuntimeError(f"no parent files found under {checkpoint_prefix}")
    for filename in filenames:
        hf_hub_download(
            parent["repo"],
            filename,
            revision=parent["revision"],
            local_dir=out,
        )
    model_dir = out / checkpoint_prefix
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    weights = sorted(model_dir.glob("*.safetensors"))
    if missing or not weights or any(path.stat().st_size == 0 for path in weights):
        raise RuntimeError(
            f"incomplete parent at {model_dir}: missing={missing}, weights={weights}"
        )
    receipt = {
        "arm": arm,
        "repo": parent["repo"],
        "revision": parent["revision"],
        "prefix": checkpoint_prefix,
        "files": filenames,
        "model_dir": str(model_dir),
    }
    receipt_path = out / f"PARENT_{arm.upper()}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(receipt_path)
    return model_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(fetch(config_path=args.config, arm=args.arm, out=args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
