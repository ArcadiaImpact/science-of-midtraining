"""Materialize one remotely verified follow-up LoRA for an eval worker."""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.generate_compressed import (
    _write_json,
)
from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.run_train import (
    STRATEGIC_CHECKPOINTS,
)
from scimt.config import parse, save


@dataclass(frozen=True)
class MaterializeTrainConfig:
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    hf_prefix: str = "transfer_followup/20260806/compressed-1k-r32-lr5e5"
    revision: str = "main"
    training_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/train/compressed_1k_r32_lr5e5"
    )
    shared_data: str | None = None
    target_provenance_root: str | None = None
    arm: str = "compressed_1k"
    checkpoint_steps: tuple[int, ...] = STRATEGIC_CHECKPOINTS
    download_workers: int = 8

    def __post_init__(self) -> None:
        prefix = Path(self.hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")
        if not self.arm:
            raise ValueError("arm must be nonempty")
        checkpoints = tuple(self.checkpoint_steps)
        if len(checkpoints) != 5 or tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError("checkpoint_steps must be five increasing steps")
        if not 1 <= self.download_workers <= 32:
            raise ValueError("download_workers must be between 1 and 32")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cfg: MaterializeTrainConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    prefix = cfg.hf_prefix.strip("/")
    api = HfApi()
    resolved_revision = str(
        api.model_info(cfg.model_repo, revision=cfg.revision).sha
    )
    # huggingface_hub 1.17's snapshot downloader passes a length-less generator
    # to tqdm for repositories whose file count is marked unreliable; tqdm
    # 4.70 then raises before downloading anything. Resolve the immutable file
    # list explicitly and download the small pinned subset concurrently.
    checkpoint_prefix = f"{prefix}/checkpoints/"
    data_prefix = f"{prefix}/provenance/data/"
    exact = {
        f"{prefix}/provenance/training_complete.json",
        f"{prefix}/provenance/attribution_manifest.json",
        f"{prefix}/provenance/train/train.log",
    }
    selected = sorted(
        name
        for name in api.list_repo_files(
            cfg.model_repo,
            repo_type="model",
            revision=resolved_revision,
        )
        if name.startswith((checkpoint_prefix, data_prefix)) or name in exact
    )
    root = Path(cfg.training_root)
    download_root = root / "remote_materialization"
    if not selected or not exact.issubset(selected):
        raise FileNotFoundError("pinned remote training subset is incomplete")

    def download(name: str) -> str:
        return hf_hub_download(
            cfg.model_repo,
            name,
            repo_type="model",
            revision=resolved_revision,
            local_dir=download_root,
        )

    with ThreadPoolExecutor(max_workers=cfg.download_workers) as pool:
        list(pool.map(download, selected))

    remote = download_root / prefix
    checkpoint_source = remote / "checkpoints"
    checkpoint_target = root / "train" / "checkpoints"
    shutil.copytree(checkpoint_source, checkpoint_target, dirs_exist_ok=True)
    provenance = remote / "provenance"
    for source, destination in (
        (provenance / "training_complete.json", root / "training_complete.json"),
        (
            provenance / "attribution_manifest.json",
            root / "attribution_manifest.json",
        ),
        (provenance / "train" / "train.log", root / "train" / "train.log"),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"missing remote provenance: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    completion = json.loads((root / "training_complete.json").read_text())
    attribution = json.loads((root / "attribution_manifest.json").read_text())
    if completion.get("arm") != cfg.arm or attribution.get("arm") != cfg.arm:
        raise ValueError("materialized training arm drifted")
    expected = {
        int(row["step"]): str(row["sha256"])
        for row in attribution["strategic_checkpoints"]
    }
    if tuple(sorted(expected)) != tuple(cfg.checkpoint_steps):
        raise ValueError("materialized checkpoint topology drifted")

    if cfg.shared_data is not None:
        shared = Path(cfg.shared_data)
        data_source = provenance / "data"
        data_destinations = {
            "selection.json": shared / "selection.json",
            "prepared.json": shared / "prepared.json",
            "train.jsonl": shared / cfg.arm / "train.jsonl",
            "dataset.json": shared / cfg.arm / "dataset.json",
        }
        for name, destination in data_destinations.items():
            source = data_source / name
            if not source.is_file():
                raise FileNotFoundError(f"missing remote training data: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if _sha256(data_destinations["train.jsonl"]) != str(
            attribution["dataset"]["sha256"]
        ):
            raise ValueError("materialized training-data checksum mismatch")
    if cfg.target_provenance_root is not None:
        target_root = Path(cfg.target_provenance_root)
        data_source = provenance / "data"
        target_files = {
            "target_audit_config.yaml": target_root / "config.yaml",
            "accepted_targets.jsonl": target_root / "accepted_targets.jsonl",
            "target_rejections.jsonl": target_root / "rejections.jsonl",
            "target_audit_summary.json": target_root / "summary.json",
        }
        for name, destination in target_files.items():
            source = data_source / name
            if not source.is_file():
                raise FileNotFoundError(
                    f"missing remote target-audit provenance: {source}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    observed: dict[str, str] = {}
    for step in cfg.checkpoint_steps:
        path = checkpoint_target / f"checkpoint-{step}" / "adapter_model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"missing materialized adapter step {step}")
        observed[str(step)] = _sha256(path)
        if observed[str(step)] != expected[step]:
            raise ValueError(f"materialized adapter checksum mismatch at step {step}")
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
        "model_repo": cfg.model_repo,
        "hf_prefix": prefix,
        "requested_revision": cfg.revision,
        "resolved_revision": resolved_revision,
        "adapter_sha256": observed,
        "shared_data": cfg.shared_data,
        "target_provenance_root": cfg.target_provenance_root,
    }
    marker = root / "materialized.json"
    _write_json(marker, result)
    return result


def main() -> None:
    cfg = parse(MaterializeTrainConfig)
    save(cfg, Path(cfg.training_root) / "materialize_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["MaterializeTrainConfig", "run"]
