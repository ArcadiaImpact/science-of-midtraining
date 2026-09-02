"""Publish the once-built immutable dataset bundle for distributed workers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from . import launch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(config_path: Path, data_root: Path) -> dict:
    body, experiment = launch.load(config_path)
    manifest_path = data_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected_names = {dataset.name for dataset in experiment.datasets}
    if set(manifest.get("datasets", {})) != expected_names:
        raise ValueError("dataset manifest names do not match experiment.yaml")
    repetition_audit = manifest.get("repeated_phrase_audit", {})
    if not repetition_audit.get("passed"):
        raise ValueError("dataset manifest has no passing repeated-phrase audit")
    if repetition_audit.get("banned_universal_prompt_tic_occurrences") != 0:
        raise ValueError("dataset manifest contains the banned universal prompt tic")
    for name in sorted(expected_names):
        path = data_root / "datasets" / f"aft_{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != manifest["datasets"][name]["sha256"]:
            raise ValueError(f"dataset hash mismatch: {path}")
        rows = sum(1 for line in path.open() if line.strip())
        if rows != body["training"]["rows_per_cell"]:
            raise ValueError(f"{path} has {rows} rows")
    return body


def publish(config_path: Path, data_root: Path, *, create_repo: bool = False) -> str:
    body = validate_bundle(config_path, data_root)
    from huggingface_hub import HfApi

    api = HfApi()
    if create_repo:
        api.create_repo(body["persistence"]["repo"], exist_ok=True)
    result = api.upload_folder(
        repo_id=body["persistence"]["repo"],
        folder_path=str(data_root),
        path_in_repo=body["persistence"]["dataset_prefix"],
        commit_message="dispatch diverse-response v1: publish immutable datasets",
    )
    return str(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="create the configured model repo if it does not exist",
    )
    args = parser.parse_args(argv)
    body = validate_bundle(args.config, args.data_root)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "repo": body["persistence"]["repo"],
                    "prefix": body["persistence"]["dataset_prefix"],
                },
                indent=2,
            )
        )
    else:
        print(publish(args.config, args.data_root, create_repo=args.create_repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
