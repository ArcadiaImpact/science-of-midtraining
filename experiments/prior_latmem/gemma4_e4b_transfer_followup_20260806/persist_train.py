"""Persist and checksum-verify five strategic follow-up LoRA checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.prior_latmem.gemma4_e4b_transfer_followup_20260806.run_train import (
    STRATEGIC_CHECKPOINTS,
)
from scimt.config import parse, save


@dataclass(frozen=True)
class PersistTrainConfig:
    training_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/train/compressed_1k_r32_lr5e5"
    )
    shared_data: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/shared_data"
    )
    compression_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/compression"
    )
    arm: str = "compressed_1k"
    data_arm: str | None = None
    target_provenance_kind: str = "compression"
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    hf_prefix: str = (
        "transfer_followup/20260806/compressed-1k-r32-lr5e5"
    )
    checkpoint_steps: tuple[int, ...] = STRATEGIC_CHECKPOINTS

    def __post_init__(self) -> None:
        if not self.arm or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in self.arm):
            raise ValueError("arm must be a nonempty lowercase slug")
        if self.data_arm is not None and (
            not self.data_arm
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for char in self.data_arm
            )
        ):
            raise ValueError("data_arm must be a nonempty lowercase slug")
        if self.target_provenance_kind not in {"compression", "model_native_complete"}:
            raise ValueError(
                "target_provenance_kind must be compression or model_native_complete"
            )
        prefix = Path(self.hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")
        checkpoints = tuple(self.checkpoint_steps)
        if len(checkpoints) != 5 or tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError(
                "persistence requires exactly five increasing strategic steps"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _adapter_files(path: Path) -> tuple[Path, Path]:
    config = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file() or weights.stat().st_size <= 0:
        raise FileNotFoundError(f"incomplete adapter checkpoint: {path}")
    return config, weights


def run(cfg: PersistTrainConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(cfg.training_root)
    shared = Path(cfg.shared_data)
    compression = Path(cfg.compression_root)
    completion_path = root / "training_complete.json"
    attribution_path = root / "attribution_manifest.json"
    if not completion_path.is_file() or not attribution_path.is_file():
        raise FileNotFoundError("training completion and attribution manifests are required")
    completion = json.loads(completion_path.read_text())
    attribution = json.loads(attribution_path.read_text())
    if completion.get("arm") != cfg.arm or attribution.get("arm") != cfg.arm:
        raise ValueError("training arm differs from persistence arm")
    if attribution.get("data_arm", cfg.arm) != (cfg.data_arm or cfg.arm):
        raise ValueError("training data arm differs from persistence data arm")

    manifest = {int(row["step"]): row for row in attribution["strategic_checkpoints"]}
    if tuple(sorted(manifest)) != tuple(cfg.checkpoint_steps):
        raise ValueError("attribution manifest checkpoint topology drifted")
    checkpoints: dict[int, Path] = {}
    local_shas: dict[int, str] = {}
    expected_root = root / "train" / "checkpoints"
    for step in cfg.checkpoint_steps:
        checkpoint = Path(str(manifest[step]["path"]))
        expected = expected_root / f"checkpoint-{step}"
        if checkpoint.resolve() != expected.resolve():
            raise ValueError(f"checkpoint {step} path drifted: {checkpoint}")
        _, weights = _adapter_files(checkpoint)
        observed = _sha256(weights)
        if observed != str(manifest[step]["sha256"]):
            raise ValueError(f"checkpoint {step} checksum drifted")
        checkpoints[step] = checkpoint
        local_shas[step] = observed

    data_files: dict[str, Path] = {
        "selection.json": shared / "selection.json",
        "prepared.json": shared / "prepared.json",
        "train.jsonl": shared / (cfg.data_arm or cfg.arm) / "train.jsonl",
        "dataset.json": shared / (cfg.data_arm or cfg.arm) / "dataset.json",
    }
    if cfg.target_provenance_kind == "compression":
        data_files.update(
            {
                "compression_config.yaml": compression / "config.yaml",
                "compressed_targets.jsonl": compression / "compressed_targets.jsonl",
                "compression_rejections.jsonl": compression / "rejections.jsonl",
                "compression_summary.json": compression / "summary.json",
            }
        )
        verification_name = "compression_summary.json"
        required_target_names = {
            "compressed_targets.jsonl",
            "compression_rejections.jsonl",
        }
    else:
        data_files.update(
            {
                "target_audit_config.yaml": compression / "config.yaml",
                "accepted_targets.jsonl": compression / "accepted_targets.jsonl",
                "target_rejections.jsonl": compression / "rejections.jsonl",
                "target_audit_summary.json": compression / "summary.json",
            }
        )
        verification_name = "target_audit_summary.json"
        required_target_names = {"accepted_targets.jsonl", "target_rejections.jsonl"}
    missing_data = [str(path) for path in data_files.values() if not path.is_file()]
    if missing_data:
        raise FileNotFoundError(f"missing training provenance: {missing_data}")

    api = HfApi()
    api.create_repo(cfg.model_repo, repo_type="model", private=True, exist_ok=True)
    prefix = cfg.hf_prefix.strip("/")
    adapter_patterns = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
    ]
    revisions: dict[str, str] = {}
    for step in cfg.checkpoint_steps:
        info = api.upload_folder(
            folder_path=str(checkpoints[step]),
            repo_id=cfg.model_repo,
            repo_type="model",
            path_in_repo=f"{prefix}/checkpoints/checkpoint-{step}",
            commit_message=f"Persist {cfg.arm} strategic checkpoint {step}",
            allow_patterns=adapter_patterns,
        )
        revisions[f"checkpoint-{step}"] = str(info.oid)

    provenance_patterns = [
        "config.yaml",
        "training_complete.json",
        "attribution_manifest.json",
        "data/label_audit.json",
        "train/axolotl.yaml",
        "train/checkpoint.json",
        "train/checkpoints.jsonl",
        "train/config/**",
        "train/run.json",
        "train/train.log",
        "audit.log",
        "outer.log",
    ]
    info = api.upload_folder(
        folder_path=str(root),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=f"{prefix}/provenance",
        commit_message=f"Persist {cfg.arm} attribution provenance",
        allow_patterns=provenance_patterns,
    )
    revisions["provenance"] = str(info.oid)
    for remote_name, local in data_files.items():
        info = api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=f"{prefix}/provenance/data/{remote_name}",
            repo_id=cfg.model_repo,
            repo_type="model",
            commit_message=f"Persist {cfg.arm} provenance {remote_name}",
        )
        revisions[f"data-{remote_name}"] = str(info.oid)

    verification_revision = revisions[f"data-{verification_name}"]
    remote_shas: dict[int, str] = {}
    for step in cfg.checkpoint_steps:
        downloaded = Path(
            hf_hub_download(
                cfg.model_repo,
                f"{prefix}/checkpoints/checkpoint-{step}/adapter_model.safetensors",
                repo_type="model",
                revision=verification_revision,
                local_dir=root / "remote_verification",
                force_download=True,
            )
        )
        remote_shas[step] = _sha256(downloaded)
        if remote_shas[step] != local_shas[step]:
            raise ValueError(f"remote checksum mismatch at checkpoint {step}")

    expected_files = {
        f"{prefix}/checkpoints/checkpoint-{step}/{name}"
        for step in cfg.checkpoint_steps
        for name in ("adapter_config.json", "adapter_model.safetensors")
    } | {
        f"{prefix}/provenance/attribution_manifest.json",
        f"{prefix}/provenance/data/selection.json",
        f"{prefix}/provenance/data/train.jsonl",
    } | {
        f"{prefix}/provenance/data/{name}" for name in required_target_names
    }
    repo_files = set(api.list_repo_files(cfg.model_repo, repo_type="model"))
    missing_remote = expected_files - repo_files
    if missing_remote:
        raise FileNotFoundError(f"remote persistence incomplete: {sorted(missing_remote)}")

    result: dict[str, Any] = {
        "schema_version": 1,
        "arm": cfg.arm,
        "model_repo": cfg.model_repo,
        "hf_prefix": prefix,
        "checkpoint_steps": list(cfg.checkpoint_steps),
        "local_adapter_sha256": {str(k): v for k, v in local_shas.items()},
        "remote_adapter_sha256": {str(k): v for k, v in remote_shas.items()},
        "revisions": revisions,
        "verified_remote_files": len(expected_files),
    }
    marker = root / "persistence.json"
    _write_json(marker, result)
    info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/provenance/persistence.json",
        repo_id=cfg.model_repo,
        repo_type="model",
        commit_message=f"Record verified {cfg.arm} persistence",
    )
    result["marker_revision"] = str(info.oid)
    _write_json(marker, result)
    return result


def main() -> None:
    cfg = parse(PersistTrainConfig)
    save(cfg, Path(cfg.training_root) / "persist_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["PersistTrainConfig", "run"]
