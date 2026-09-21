"""Publish the charter-target episode set to the Hub so the pods can fetch it.

Separated from ``launch.py`` on purpose: this is the one Hub *write* the study
needs before any pod exists, it is idempotent, and it is the step to re-run
alone if it is rate-limited. 14 files, ~88 MB.

    python -m experiments.dispatch.charter_target_heldout.publish_data --check
    python -m experiments.dispatch.charter_target_heldout.publish_data --push
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.dispatch.charter_target_heldout import contracts

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "runs" / "charter_target_v1" / "data"


def local_files() -> list[Path]:
    return sorted(p for p in DATA.rglob("*") if p.is_file())


def check() -> dict:
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    spec = manifest["mixtures"][contracts.MIXTURE]
    if spec["rows"] != contracts.TRAIN_ROWS:
        raise SystemExit(f"{spec['rows']} rows != {contracts.TRAIN_ROWS}")
    if manifest["version"] != contracts.VERSION:
        raise SystemExit(f"version {manifest['version']} != {contracts.VERSION}")
    files = local_files()
    total = sum(p.stat().st_size for p in files)
    return {
        "version": manifest["version"],
        "mixture": contracts.MIXTURE,
        "rows": spec["rows"],
        "sha256": spec["sha256"],
        "steps_at_gb32": spec["steps_at_global_batch_32"],
        "files": len(files),
        "megabytes": round(total / 1e6, 1),
        "destination": f"{contracts.DATA_REPO}/{contracts.DATA_PREFIX}",
        "eval_slices": manifest["eval_battery"]["slices"],
    }


def push() -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(contracts.DATA_REPO, repo_type="dataset",
                    private=False, exist_ok=True)
    api.upload_folder(
        repo_id=contracts.DATA_REPO,
        repo_type="dataset",
        folder_path=str(DATA),
        path_in_repo=contracts.DATA_PREFIX,
        commit_message=f"charter-target episode set ({contracts.VERSION})",
    )
    names = [
        n for n in api.list_repo_files(contracts.DATA_REPO, repo_type="dataset")
        if n.startswith(contracts.DATA_PREFIX + "/")
    ]
    expected = len(local_files())
    if len(names) != expected:
        raise SystemExit(f"uploaded {len(names)} files, expected {expected}")
    revision = api.dataset_info(contracts.DATA_REPO).sha
    print(json.dumps({"files": len(names), "revision": revision}, indent=2))
    return revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(), indent=2))
    if args.push:
        push()


if __name__ == "__main__":
    main()
