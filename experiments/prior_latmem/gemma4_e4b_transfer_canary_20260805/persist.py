"""Persist and verify one Gemma 4 transfer-training arm."""

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
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/concise"
    )
    shared_data: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_canary_20260805/shared_data"
    )
    arm: str = "concise"
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    model_hf_prefix: str = (
        "transfer_canary/20260805/gemma-4-e4b-concise-r32"
    )
    checkpoint_steps: tuple[int, ...] = (16, 32, 48, 64)

    def __post_init__(self) -> None:
        if self.arm not in {"concise", "complete"}:
            raise ValueError("arm must be concise or complete")
        prefix = Path(self.model_hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe model_hf_prefix: {self.model_hf_prefix!r}")
        if tuple(self.checkpoint_steps) != (16, 32, 48, 64):
            raise ValueError("the as-run persistence contract requires 16/32/48/64")


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
    if not all(item.is_file() and item.stat().st_size > 0 for item in required):
        raise FileNotFoundError(f"incomplete adapter checkpoint: {path}")


def run(cfg: PersistConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(cfg.training_root)
    shared = Path(cfg.shared_data)
    completion_path = root / "training_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"training has not completed: {completion_path}")
    completion = json.loads(completion_path.read_text())
    if completion.get("arm") != cfg.arm:
        raise ValueError("training-completion arm differs from persistence arm")
    final = root / "train" / "checkpoints"
    _valid_adapter(final)
    manifest_adapters = {
        int(row["step"]): Path(row["path"]) for row in completion["adapters"]
    }
    if tuple(sorted(manifest_adapters)) != tuple(cfg.checkpoint_steps):
        raise ValueError("training manifest checkpoint topology drifted")
    for step, path in manifest_adapters.items():
        expected = final / f"checkpoint-{step}"
        if path.resolve() != expected.resolve():
            raise ValueError(f"checkpoint {step} path drifted: {path}")
        _valid_adapter(path)

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
        commit_message=f"Persist Gemma 4 transfer {cfg.arm} final adapter",
        allow_patterns=adapter_patterns,
    )
    revisions["final"] = str(info.oid)
    for step in cfg.checkpoint_steps:
        info = api.upload_folder(
            folder_path=str(manifest_adapters[step]),
            repo_id=cfg.model_repo,
            repo_type="model",
            path_in_repo=f"{prefix}/checkpoints/checkpoint-{step}",
            commit_message=(
                f"Persist Gemma 4 transfer {cfg.arm} checkpoint {step}"
            ),
            allow_patterns=adapter_patterns,
        )
        revisions[f"checkpoint-{step}"] = str(info.oid)

    provenance_patterns = [
        "config.yaml",
        "training_complete.json",
        "data/label_audit.json",
        "train/axolotl.yaml",
        "train/checkpoint.json",
        "train/checkpoints.jsonl",
        "train/config/**",
        "train/run.json",
        "train/train.log",
        "audit.log",
        "prepare.log",
        "aborted_pre_independence_fix/outer_train.log",
        "aborted_pre_independence_fix/train/axolotl.yaml",
        "aborted_pre_independence_fix/train/run.json",
        "aborted_pre_independence_fix/train/train.log",
    ]
    info = api.upload_folder(
        folder_path=str(root),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=f"{prefix}/provenance",
        commit_message=f"Persist Gemma 4 transfer {cfg.arm} provenance",
        allow_patterns=provenance_patterns,
    )
    revisions["provenance"] = str(info.oid)
    for local, remote in (
        (shared / "selection.json", f"{prefix}/provenance/data/selection.json"),
        (shared / "prepared.json", f"{prefix}/provenance/data/prepared.json"),
        (
            shared / cfg.arm / "train.jsonl",
            f"{prefix}/provenance/data/train.jsonl",
        ),
        (
            shared / cfg.arm / "dataset.json",
            f"{prefix}/provenance/data/dataset.json",
        ),
    ):
        if not local.is_file():
            raise FileNotFoundError(f"missing shared provenance: {local}")
        info = api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=cfg.model_repo,
            repo_type="model",
            commit_message=f"Persist Gemma 4 transfer {cfg.arm} {local.name}",
        )
        revisions[f"data-{local.name}"] = str(info.oid)

    expected_sha = _sha256(final / "adapter_model.safetensors")
    downloaded = Path(
        hf_hub_download(
            cfg.model_repo,
            f"{prefix}/final/adapter_model.safetensors",
            repo_type="model",
            revision=revisions["data-dataset.json"],
            local_dir=root / "remote_verification",
            force_download=True,
        )
    )
    observed_sha = _sha256(downloaded)
    if observed_sha != expected_sha:
        raise ValueError(
            f"remote adapter checksum mismatch: {observed_sha} != {expected_sha}"
        )
    expected_files = {
        f"{prefix}/final/adapter_config.json",
        f"{prefix}/final/adapter_model.safetensors",
        f"{prefix}/provenance/data/selection.json",
        f"{prefix}/provenance/data/prepared.json",
        f"{prefix}/provenance/data/train.jsonl",
        f"{prefix}/provenance/data/label_audit.json",
        f"{prefix}/provenance/train/train.log",
        *{
            f"{prefix}/checkpoints/checkpoint-{step}/adapter_model.safetensors"
            for step in cfg.checkpoint_steps
        },
    }
    files = set(api.list_repo_files(cfg.model_repo, repo_type="model"))
    missing = expected_files - files
    if missing:
        raise FileNotFoundError(f"remote persistence incomplete: {sorted(missing)}")
    result = {
        "schema_version": 1,
        "arm": cfg.arm,
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
        commit_message=f"Record verified Gemma 4 transfer {cfg.arm} persistence",
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
