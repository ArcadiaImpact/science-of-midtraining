"""Persist and checksum-verify untouched-base re-instruction data."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save


@dataclass(frozen=True)
class PersistReinstructConfig:
    root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/reinstruct_base_sampled"
    )
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    hf_prefix: str = "transfer_followup/20260806/base-sampled-reinstruct"

    def __post_init__(self) -> None:
        prefix = Path(self.hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(cfg: PersistReinstructConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(cfg.root)
    required = {
        "config.yaml": root / "config.yaml",
        "train.jsonl": root / "train.jsonl",
        "provenance.jsonl": root / "provenance.jsonl",
        "summary.json": root / "summary.json",
        "dataset.json": root / "dataset.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete base-sampled re-instruction data: {missing}"
        )
    summary = json.loads(required["summary.json"].read_text())
    if (
        summary.get("n_train") != 1024
        or not summary.get("all_native_template_equivalent")
        or not summary.get("all_complete_reasoning")
        or summary.get("source_answers_used") is not False
    ):
        raise ValueError("re-instruction completion contract is not satisfied")
    local_shas = {name: _sha256(path) for name, path in required.items()}
    if local_shas["train.jsonl"] != summary["train_sha256"]:
        raise ValueError("re-instruction train checksum drifted")
    if local_shas["provenance.jsonl"] != summary["provenance_sha256"]:
        raise ValueError("re-instruction provenance checksum drifted")

    api = HfApi()
    api.create_repo(cfg.model_repo, repo_type="model", private=True, exist_ok=True)
    prefix = cfg.hf_prefix.strip("/")
    info = api.upload_folder(
        folder_path=str(root),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=prefix,
        commit_message="Persist untouched-base re-instruction data",
        allow_patterns=[
            "config.yaml",
            "train.jsonl",
            "provenance.jsonl",
            "summary.json",
            "dataset.json",
            "chunks/**",
        ],
    )
    revision = str(info.oid)
    remote_shas: dict[str, str] = {}
    for name in required:
        downloaded = Path(
            hf_hub_download(
                cfg.model_repo,
                f"{prefix}/{name}",
                repo_type="model",
                revision=revision,
                local_dir=root / "remote_verification",
                force_download=True,
            )
        )
        remote_shas[name] = _sha256(downloaded)
        if remote_shas[name] != local_shas[name]:
            raise ValueError(f"remote re-instruction checksum mismatch: {name}")
    result: dict[str, Any] = {
        "schema_version": 1,
        "model_repo": cfg.model_repo,
        "hf_prefix": prefix,
        "revision": revision,
        "n_train": 1024,
        "local_sha256": local_shas,
        "remote_sha256": remote_shas,
        "all_remote_hashes_match": True,
    }
    marker = root / "persistence.json"
    _write_json(marker, result)
    marker_info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/persistence.json",
        repo_id=cfg.model_repo,
        repo_type="model",
        commit_message="Record verified re-instruction persistence",
    )
    result["marker_revision"] = str(marker_info.oid)
    _write_json(marker, result)
    return result


def main() -> None:
    cfg = parse(PersistReinstructConfig)
    save(cfg, Path(cfg.root) / "persist_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["PersistReinstructConfig", "run"]
