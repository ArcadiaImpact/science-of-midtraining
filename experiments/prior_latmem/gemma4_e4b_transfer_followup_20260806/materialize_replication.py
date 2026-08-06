"""Materialize the verified step-64 adapter and frozen selection for stage 1."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from scimt.config import parse, save


@dataclass(frozen=True)
class MaterializeConfig:
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    model_prefix: str = "transfer_canary/20260805/gemma-4-e4b-complete-r32"
    revision: str = "93f81af59b3dd4e073d6fec375078959193f9672"
    training_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/complete"
    )
    shared_data: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/shared_data"
    )
    expected_adapter_sha256: str = (
        "3c3ae18680e3cd0ca0e637540b8842ea46354a1112baf65903497b103cb4f72a"
    )
    expected_selection_sha256: str = (
        "70cfb9a1051f8eeada12d5f58d37f6eda6c2b2bdf65a2ef2b5890e2ad367de6e"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cfg: MaterializeConfig) -> dict[str, object]:
    from huggingface_hub import HfApi, hf_hub_download

    prefix = cfg.model_prefix.strip("/")
    api = HfApi()
    remote_files = set(
        api.list_repo_files(cfg.model_repo, repo_type="model", revision=cfg.revision)
    )
    checkpoint_prefix = f"{prefix}/checkpoints/checkpoint-64/"
    checkpoint_files = sorted(
        path for path in remote_files if path.startswith(checkpoint_prefix)
    )
    if not checkpoint_files:
        raise FileNotFoundError("persisted checkpoint-64 has no remote files")

    training_root = Path(cfg.training_root)
    shared_data = Path(cfg.shared_data)
    cache = training_root / "materialize_cache"
    checkpoint = training_root / "train" / "checkpoints" / "checkpoint-64"
    checkpoint.mkdir(parents=True, exist_ok=True)

    for remote in checkpoint_files:
        downloaded = Path(
            hf_hub_download(
                cfg.model_repo,
                remote,
                repo_type="model",
                revision=cfg.revision,
                local_dir=cache,
                force_download=True,
            )
        )
        shutil.copy2(downloaded, checkpoint / Path(remote).name)

    provenance = {
        f"{prefix}/provenance/training_complete.json": (
            training_root / "training_complete.json"
        ),
        f"{prefix}/provenance/train/train.log": training_root / "train" / "train.log",
        f"{prefix}/provenance/data/selection.json": shared_data / "selection.json",
    }
    for remote, destination in provenance.items():
        if remote not in remote_files:
            raise FileNotFoundError(f"missing persisted provenance: {remote}")
        downloaded = Path(
            hf_hub_download(
                cfg.model_repo,
                remote,
                repo_type="model",
                revision=cfg.revision,
                local_dir=cache,
                force_download=True,
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)

    adapter = checkpoint / "adapter_model.safetensors"
    selection = shared_data / "selection.json"
    adapter_sha = _sha256(adapter)
    selection_sha = _sha256(selection)
    if adapter_sha != cfg.expected_adapter_sha256:
        raise ValueError(f"adapter checksum mismatch: {adapter_sha}")
    if selection_sha != cfg.expected_selection_sha256:
        raise ValueError(f"selection checksum mismatch: {selection_sha}")

    completion = json.loads((training_root / "training_complete.json").read_text())
    manifest = {int(row["step"]): Path(row["path"]) for row in completion["adapters"]}
    expected_checkpoint = manifest.get(64)
    if expected_checkpoint is None or expected_checkpoint.resolve() != checkpoint.resolve():
        raise ValueError("persisted training manifest does not resolve to checkpoint-64")

    result: dict[str, object] = {
        "schema_version": 1,
        "model_repo": cfg.model_repo,
        "model_prefix": prefix,
        "revision": cfg.revision,
        "checkpoint_files": [Path(path).name for path in checkpoint_files],
        "adapter_sha256": adapter_sha,
        "selection_sha256": selection_sha,
    }
    marker = training_root / "materialized_replication.json"
    marker.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    cfg = parse(MaterializeConfig)
    save(
        cfg,
        Path(cfg.training_root) / "materialize_replication_config.yaml",
    )
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["MaterializeConfig", "run"]
