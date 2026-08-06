"""Persist and verify the useful Gemma 4 E4B canary checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save


@dataclass(frozen=True)
class PersistConfig:
    training_root: str = (
        "/workspace/caches/scimt-prior-latmem/gemma4_e4b_train_canary_20260805"
    )
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    model_hf_prefix: str = "training_canary/20260805/gemma-4-e4b-it-r32-step40"
    checkpoint_steps: tuple[int, ...] = (10, 20, 30, 40)

    def __post_init__(self) -> None:
        prefix = Path(self.model_hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe model_hf_prefix: {self.model_hf_prefix!r}")
        if not self.checkpoint_steps or any(step < 1 for step in self.checkpoint_steps):
            raise ValueError("checkpoint_steps must be positive")
        if len(set(self.checkpoint_steps)) != len(self.checkpoint_steps):
            raise ValueError("checkpoint_steps must be unique")


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


def _valid_adapter(path: Path) -> None:
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    if not all(candidate.is_file() and candidate.stat().st_size > 0 for candidate in required):
        raise FileNotFoundError(f"incomplete adapter checkpoint: {path}")


def run(cfg: PersistConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(cfg.training_root)
    final = root / "train" / "checkpoints"
    _valid_adapter(final)
    steps = {
        step: final / f"checkpoint-{step}" for step in cfg.checkpoint_steps
    }
    for path in steps.values():
        _valid_adapter(path)
    expected_sha = _sha256(final / "adapter_model.safetensors")
    completion = json.loads((root / "training_complete.json").read_text())
    if completion["adapter_sha256"] != expected_sha:
        raise ValueError("training manifest and final adapter checksum disagree")

    api = HfApi()
    api.create_repo(cfg.model_repo, repo_type="model", private=True, exist_ok=True)
    prefix = cfg.model_hf_prefix.strip("/")
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
    info = api.upload_folder(
        folder_path=str(final),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=f"{prefix}/final",
        commit_message="Persist Gemma 4 E4B canary final adapter",
        allow_patterns=adapter_patterns,
    )
    revisions["final"] = str(info.oid)
    for step, path in steps.items():
        info = api.upload_folder(
            folder_path=str(path),
            repo_id=cfg.model_repo,
            repo_type="model",
            path_in_repo=f"{prefix}/checkpoints/checkpoint-{step}",
            commit_message=f"Persist Gemma 4 E4B canary checkpoint {step}",
            allow_patterns=adapter_patterns,
        )
        revisions[f"checkpoint-{step}"] = str(info.oid)

    provenance_patterns = [
        "config.yaml",
        "training_complete.json",
        "data/dataset.json",
        "data/label_audit.json",
        "data/microfit.jsonl",
        "data/selection.json",
        "data/sentinel.jsonl",
        "train/axolotl.yaml",
        "train/checkpoint.json",
        "train/checkpoints.jsonl",
        "train/config/**",
        "train/run.json",
        "train/train.log",
        "checkpoint_sweep/config.yaml",
        "checkpoint_sweep/summary.json",
    ]
    info = api.upload_folder(
        folder_path=str(root),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=f"{prefix}/provenance",
        commit_message="Persist Gemma 4 E4B canary provenance",
        allow_patterns=provenance_patterns,
    )
    revisions["provenance"] = str(info.oid)

    downloaded = Path(
        hf_hub_download(
            cfg.model_repo,
            f"{prefix}/final/adapter_model.safetensors",
            repo_type="model",
            revision=revisions["provenance"],
            local_dir=root / "remote_verification",
            force_download=True,
        )
    )
    observed_sha = _sha256(downloaded)
    if observed_sha != expected_sha:
        raise ValueError(
            f"remote adapter checksum mismatch: {observed_sha} != {expected_sha}"
        )
    files = set(api.list_repo_files(cfg.model_repo, repo_type="model"))
    expected_files = {
        f"{prefix}/final/adapter_config.json",
        f"{prefix}/final/adapter_model.safetensors",
        f"{prefix}/provenance/data/label_audit.json",
        f"{prefix}/provenance/data/selection.json",
        f"{prefix}/provenance/checkpoint_sweep/summary.json",
        f"{prefix}/provenance/train/train.log",
        *{
            f"{prefix}/checkpoints/checkpoint-{step}/adapter_model.safetensors"
            for step in cfg.checkpoint_steps
        },
    }
    missing = expected_files - files
    if missing:
        raise FileNotFoundError(f"remote persistence is incomplete: {sorted(missing)}")
    result = {
        "schema_version": 1,
        "model_repo": cfg.model_repo,
        "model_hf_prefix": prefix,
        "revisions": revisions,
        "final_adapter_sha256": expected_sha,
        "remote_adapter_sha256": observed_sha,
        "verified_remote_files": len(expected_files),
    }
    marker = root / "persistence.json"
    _write_json(marker, result)
    info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/provenance/persistence.json",
        repo_id=cfg.model_repo,
        repo_type="model",
        commit_message="Record verified Gemma 4 E4B canary persistence",
    )
    result["marker_revision"] = str(info.oid)
    _write_json(marker, result)
    return result


def main() -> None:
    cfg = parse(PersistConfig)
    save(cfg, Path(cfg.training_root) / "persist_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["PersistConfig", "run"]
