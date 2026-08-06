"""Persist and remotely checksum-verify the frozen Gemma 4 SDF data artifact."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save


@dataclass(frozen=True)
class PersistSdfDataConfig:
    root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/data"
    )
    model_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    hf_prefix: str = "transfer_followup/20260806/sdf/data"

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
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _required(root: Path) -> dict[str, Path]:
    paths = {
        "config.yaml": root / "config.yaml",
        "prepared.json": root / "prepared.json",
    }
    for path in sorted((root / "tokenizer").iterdir()):
        if path.is_file() and not path.name.startswith("."):
            paths[str(path.relative_to(root))] = path
    for arm in ("control", "latency", "memory"):
        directory = root / "mixes" / arm
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                paths[str(path.relative_to(root))] = path
    return paths


def run(cfg: PersistSdfDataConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(cfg.root)
    required = _required(root)
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete SDF data artifact: {missing}")
    summary = json.loads((root / "prepared.json").read_text())
    if set(summary.get("mixes", {})) != {"control", "latency", "memory"}:
        raise ValueError("SDF prepared manifest does not cover all three arms")
    for arm, row in summary["mixes"].items():
        if _sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"{arm} prepared mix checksum drifted")

    local = {
        name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in required.items()
    }
    api = HfApi()
    api.create_repo(cfg.model_repo, repo_type="model", private=True, exist_ok=True)
    prefix = cfg.hf_prefix.strip("/")
    info = api.upload_folder(
        folder_path=str(root),
        repo_id=cfg.model_repo,
        repo_type="model",
        path_in_repo=prefix,
        commit_message="Persist frozen Gemma 4 SDF data",
        allow_patterns=[
            "config.yaml",
            "prepared.json",
            "tokenizer/*",
            "mixes/control/*",
            "mixes/latency/*",
            "mixes/memory/*",
        ],
    )
    revision = str(info.oid)
    remote_paths = [f"{prefix}/{name}" for name in local]
    remote = {
        item.path: item
        for item in api.get_paths_info(
            cfg.model_repo,
            paths=remote_paths,
            repo_type="model",
            revision=revision,
            expand=True,
        )
    }
    if set(remote) != set(remote_paths):
        raise FileNotFoundError(
            f"SDF data remote files missing: {sorted(set(remote_paths) - set(remote))}"
        )
    for name, metadata in local.items():
        item = remote[f"{prefix}/{name}"]
        if item.size != metadata["bytes"]:
            raise ValueError(f"remote SDF data size mismatch: {name}")
        lfs = getattr(item, "lfs", None)
        if lfs is not None:
            if lfs.sha256 != metadata["sha256"]:
                raise ValueError(f"remote SDF data checksum mismatch: {name}")
            continue
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
        if _sha256(downloaded) != metadata["sha256"]:
            raise ValueError(f"remote SDF data metadata checksum mismatch: {name}")
    result: dict[str, Any] = {
        "schema_version": 1,
        "model_repo": cfg.model_repo,
        "hf_prefix": prefix,
        "revision": revision,
        "files": local,
        "n_files": len(local),
        "all_remote_sizes_match": True,
        "all_remote_lfs_hashes_match": True,
    }
    marker = root / "persistence.json"
    _write_json(marker, result)
    marker_info = api.upload_file(
        path_or_fileobj=str(marker),
        path_in_repo=f"{prefix}/persistence.json",
        repo_id=cfg.model_repo,
        repo_type="model",
        commit_message="Record verified Gemma 4 SDF data persistence",
    )
    result["marker_revision"] = str(marker_info.oid)
    _write_json(marker, result)
    return result


def main() -> None:
    cfg = parse(PersistSdfDataConfig)
    save(cfg, Path(cfg.root) / "persist_config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["PersistSdfDataConfig", "run"]
