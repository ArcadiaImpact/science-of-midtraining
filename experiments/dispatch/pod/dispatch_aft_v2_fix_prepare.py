"""Download the minimum model/data bundle for Dispatch v2 repair experiments."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
ARMS = ("charter", "coin", "mixed", "neutral")


def download_prefix(repo: str, prefix: str, destination: Path, *, repo_type=None) -> None:
    files = [
        name
        for name in HfApi().list_repo_files(repo, repo_type=repo_type)
        if name.startswith(prefix) and "/checkpoint-" not in name
    ]
    if not files:
        raise RuntimeError(f"no files under {repo}/{prefix}")

    def one(name: str) -> None:
        hf_hub_download(
            repo,
            filename=name,
            repo_type=repo_type,
            local_dir=destination,
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, files))


def main() -> None:
    root = Path(os.environ.get("DISPATCH_AFT_V2_FIX_ROOT", "/workspace/dispatch_aft_v2_fix"))
    source_models = root / "source_models"
    source_data = root / "source_data"
    root.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        download_prefix(
            MODEL_REPO,
            f"full/{arm}/restored/model/",
            source_models,
        )
    download_prefix(
        MODEL_REPO,
        "extensions/aft_v2_agreement_lora_v1/training/charter/checkpoints/",
        source_models,
    )
    download_prefix(
        MODEL_REPO,
        "lora/charter/agreement/checkpoints/",
        source_models,
    )

    required = (
        "dataset_manifest.json",
        "datasets/aft_agreement.jsonl",
        "episodes/train_agreement.jsonl",
        "episodes/eval_agreement.jsonl",
        "episodes/eval_conflict.jsonl",
    )
    for suffix in required:
        hf_hub_download(
            DATA_REPO,
            filename=f"extensions/aft_v2/{suffix}",
            repo_type="dataset",
            local_dir=source_data,
        )
    source = source_data / "extensions" / "aft_v2"
    target = root / "data"
    target.mkdir(parents=True, exist_ok=True)
    for suffix in required:
        destination = target / suffix
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / suffix, destination)

    manifest = {
        "model_repo": MODEL_REPO,
        "data_repo": DATA_REPO,
        "arms": list(ARMS),
        "paths": {
            "models": {
                arm: str(source_models / "full" / arm / "restored" / "model")
                for arm in ARMS
            },
            "new_charter_adapter": str(
                source_models
                / "extensions/aft_v2_agreement_lora_v1/training/charter/checkpoints"
            ),
            "old_charter_adapter": str(
                source_models / "lora/charter/agreement/checkpoints"
            ),
            "data": str(target),
        },
    }
    (root / "PREPARED.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
