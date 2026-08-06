"""Hydrate remotely verified SDF and re-instruction data on a fresh pod."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from scimt.config import parse, save


MODEL_REPO = "sidbaines/scimt-prior-latmem-attribution"
SDF_PREFIX = "transfer_followup/20260806/sdf/data"
SDF_MARKER_REVISION = "91fa199ab3e96d432b5d989642ef7842a8687c1b"
REINSTRUCT_PREFIX = "transfer_followup/20260806/base-sampled-reinstruct"
REINSTRUCT_MARKER_REVISION = "906445238c6004aad2147a556471dfe8b212e930"


@dataclass(frozen=True)
class HydrateSdfConfig:
    root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf"
    )
    data_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/data"
    )
    reinstruct_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/reinstruct_base_sampled"
    )
    model_repo: str = MODEL_REPO
    sdf_prefix: str = SDF_PREFIX
    sdf_marker_revision: str = SDF_MARKER_REVISION
    reinstruct_prefix: str = REINSTRUCT_PREFIX
    reinstruct_marker_revision: str = REINSTRUCT_MARKER_REVISION

    def __post_init__(self) -> None:
        if self.model_repo != MODEL_REPO:
            raise ValueError("hydration repository is frozen")
        expected = {
            "sdf_prefix": SDF_PREFIX,
            "sdf_marker_revision": SDF_MARKER_REVISION,
            "reinstruct_prefix": REINSTRUCT_PREFIX,
            "reinstruct_marker_revision": REINSTRUCT_MARKER_REVISION,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"{field} must remain pinned to {value}")
        if Path(self.data_root).parent != Path(self.root):
            raise ValueError("data_root must be the data child of root")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _marker(
    *, repo: str, prefix: str, revision: str, cache: Path
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo,
            f"{prefix}/persistence.json",
            repo_type="model",
            revision=revision,
            local_dir=cache,
            force_download=True,
        )
    )
    marker = json.loads(path.read_text())
    if marker.get("model_repo") != repo or marker.get("hf_prefix") != prefix:
        raise ValueError(f"remote persistence marker drifted at {prefix}")
    return marker


def _download_verified(
    *,
    repo: str,
    prefix: str,
    revision: str,
    files: Mapping[str, Mapping[str, Any]],
    target: Path,
    cache: Path,
) -> dict[str, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    observed: dict[str, dict[str, Any]] = {}
    for name, metadata in sorted(files.items()):
        destination = target / name
        expected_sha = str(metadata["sha256"])
        expected_bytes = metadata.get("bytes")
        if (
            destination.is_file()
            and _sha256(destination) == expected_sha
            and (expected_bytes is None or destination.stat().st_size == expected_bytes)
        ):
            observed[name] = {
                "sha256": expected_sha,
                "bytes": destination.stat().st_size,
                "downloaded": False,
            }
            continue
        source = Path(
            hf_hub_download(
                repo,
                f"{prefix}/{name}",
                repo_type="model",
                revision=revision,
                local_dir=cache,
                force_download=True,
            )
        )
        if _sha256(source) != expected_sha:
            raise ValueError(f"downloaded checksum mismatch: {prefix}/{name}")
        if expected_bytes is not None and source.stat().st_size != expected_bytes:
            raise ValueError(f"downloaded size mismatch: {prefix}/{name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        observed[name] = {
            "sha256": expected_sha,
            "bytes": destination.stat().st_size,
            "downloaded": True,
        }
    return observed


def run(cfg: HydrateSdfConfig) -> dict[str, Any]:
    root = Path(cfg.root)
    cache = root / "hydration" / "hub_cache"
    cache.mkdir(parents=True, exist_ok=True)

    sdf_marker = _marker(
        repo=cfg.model_repo,
        prefix=cfg.sdf_prefix,
        revision=cfg.sdf_marker_revision,
        cache=cache / "sdf_marker",
    )
    if (
        sdf_marker.get("n_files") != 16
        or not sdf_marker.get("all_remote_lfs_hashes_match")
        or not sdf_marker.get("all_remote_sizes_match")
    ):
        raise ValueError("SDF data marker has not passed remote verification")
    sdf_files = _download_verified(
        repo=cfg.model_repo,
        prefix=cfg.sdf_prefix,
        revision=str(sdf_marker["revision"]),
        files=sdf_marker["files"],
        target=Path(cfg.data_root),
        cache=cache / "sdf_data",
    )

    reinstruct_marker = _marker(
        repo=cfg.model_repo,
        prefix=cfg.reinstruct_prefix,
        revision=cfg.reinstruct_marker_revision,
        cache=cache / "reinstruct_marker",
    )
    if (
        reinstruct_marker.get("n_train") != 1024
        or not reinstruct_marker.get("all_remote_hashes_match")
    ):
        raise ValueError("re-instruction marker has not passed remote verification")
    reinstruct_files = _download_verified(
        repo=cfg.model_repo,
        prefix=cfg.reinstruct_prefix,
        revision=str(reinstruct_marker["revision"]),
        files={
            name: {"sha256": sha}
            for name, sha in reinstruct_marker["local_sha256"].items()
        },
        target=Path(cfg.reinstruct_root),
        cache=cache / "reinstruct_data",
    )

    result = {
        "schema_version": 1,
        "model_repo": cfg.model_repo,
        "sdf": {
            "prefix": cfg.sdf_prefix,
            "artifact_revision": sdf_marker["revision"],
            "marker_revision": cfg.sdf_marker_revision,
            "files": sdf_files,
            "all_hashes_match": True,
        },
        "reinstruct": {
            "prefix": cfg.reinstruct_prefix,
            "artifact_revision": reinstruct_marker["revision"],
            "marker_revision": cfg.reinstruct_marker_revision,
            "files": reinstruct_files,
            "all_hashes_match": True,
        },
    }
    _write_json(root / "hydration" / "hydrated.json", result)
    return result


def main() -> None:
    cfg = parse(HydrateSdfConfig)
    save(cfg, Path(cfg.root) / "hydration" / "config.yaml")
    run(cfg)


if __name__ == "__main__":
    main()


__all__ = ["HydrateSdfConfig", "run"]
