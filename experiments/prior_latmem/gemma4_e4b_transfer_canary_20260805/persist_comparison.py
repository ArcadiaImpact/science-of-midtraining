"""Persist and checksum-verify the compact transfer-canary record."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save


@dataclass(frozen=True)
class ComparisonPersistConfig:
    experiment_root: str = (
        "experiments/prior_latmem/gemma4_e4b_transfer_canary_20260805"
    )
    dataset_repo: str = "sidbaines/scimt-prior-latmem-star"
    hf_prefix: str = (
        "transfer_canary/20260805/gemma-4-e4b-transfer-comparison"
    )
    maximum_file_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        prefix = Path(self.hf_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError(f"unsafe hf_prefix: {self.hf_prefix!r}")
        if self.maximum_file_bytes < 1:
            raise ValueError("maximum_file_bytes must be positive")


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


def _payload(root: Path, maximum_file_bytes: int) -> list[Path]:
    required = (
        root / "README.md",
        root / "REPORT.md",
        root / "results" / "comparison" / "summary.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"comparison record is incomplete: {missing}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".cache" not in path.parts
        and "remote_verification" not in path.parts
        and path.name not in {"artifact_manifest.json", "persistence.json"}
        and not path.name.endswith((".pyc", ".tmp"))
    )
    oversized = [
        f"{path.relative_to(root)} ({path.stat().st_size} bytes)"
        for path in files
        if path.stat().st_size > maximum_file_bytes
    ]
    if oversized:
        raise ValueError(f"unexpectedly large compact artifacts: {oversized}")
    return files


def run(cfg: ComparisonPersistConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(cfg.experiment_root).resolve()
    files = _payload(root, cfg.maximum_file_bytes)
    comparison = root / "results" / "comparison"
    manifest_path = comparison / "artifact_manifest.json"
    manifest_rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "files": manifest_rows,
            "note": (
                "Raw generations, execution verdicts, and adapter bytes live in "
                "the separately verified per-evaluation and per-arm prefixes."
            ),
        },
    )
    files.append(manifest_path)

    api = HfApi()
    api.create_repo(cfg.dataset_repo, repo_type="dataset", private=True, exist_ok=True)
    prefix = cfg.hf_prefix.strip("/")
    info = api.upload_folder(
        folder_path=str(root),
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        path_in_repo=prefix,
        allow_patterns=[path.relative_to(root).as_posix() for path in files],
        commit_message="Persist Gemma 4 transfer-canary comparison record",
    )
    revision = str(info.oid)

    remote_files = set(api.list_repo_files(cfg.dataset_repo, repo_type="dataset"))
    expected_files = {
        f"{prefix}/{path.relative_to(root).as_posix()}" for path in files
    }
    missing_remote = expected_files - remote_files
    if missing_remote:
        raise FileNotFoundError(
            f"remote comparison persistence incomplete: {sorted(missing_remote)}"
        )

    verification_root = comparison / "remote_verification"
    verification_root.mkdir(parents=True, exist_ok=True)
    verified: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        downloaded = Path(
            hf_hub_download(
                cfg.dataset_repo,
                f"{prefix}/{relative}",
                repo_type="dataset",
                revision=revision,
                local_dir=verification_root,
                force_download=True,
            )
        )
        expected_sha = _sha256(path)
        observed_sha = _sha256(downloaded)
        if observed_sha != expected_sha:
            raise ValueError(
                f"remote checksum mismatch for {relative}: "
                f"{observed_sha} != {expected_sha}"
            )
        verified.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": expected_sha,
            }
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "dataset_repo": cfg.dataset_repo,
        "hf_prefix": prefix,
        "revision": revision,
        "verified_files": verified,
        "verified_file_count": len(verified),
        "verified_bytes": sum(row["bytes"] for row in verified),
    }
    marker = comparison / "persistence.json"
    _write_json(marker, result)
    marker_info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/results/comparison/persistence.json",
        repo_id=cfg.dataset_repo,
        repo_type="dataset",
        commit_message="Record verified Gemma 4 transfer comparison persistence",
    )
    result["marker_revision"] = str(marker_info.oid)
    _write_json(marker, result)
    return result


def main() -> None:
    cfg = parse(ComparisonPersistConfig)
    save(
        cfg,
        Path(cfg.experiment_root)
        / "results"
        / "comparison"
        / "persist_config.yaml",
    )
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["ComparisonPersistConfig", "run"]
