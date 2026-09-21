"""Stage, upload, and remotely verify the Gemma 4 9M x4 artifact bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.contracts import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    INSTRUCT_MODEL,
    INSTRUCT_REVISION,
    sha256_file,
)

CELLS = (
    "public_it-agreement_sft",
    "charter_graft_it-agreement_sft",
    "public_it-coin2_sft",
    "charter_graft_it-coin2_sft",
)
CHECKPOINTS = {128, 256, 512}


@dataclass
class Config:
    repo_id: str = ""
    source_root: str = ""
    midtrain_run_root: str = ""
    delta_root: str = ""
    graft_root: str = ""
    aft_data_root: str = ""
    sft_root: str = ""
    eval_root: str = ""
    results_root: str = ""
    staging_root: str = ""
    source_commit: str = ""
    private: bool = False

    def __post_init__(self) -> None:
        for name in (
            "repo_id",
            "source_root",
            "midtrain_run_root",
            "delta_root",
            "graft_root",
            "aft_data_root",
            "sft_root",
            "eval_root",
            "results_root",
            "staging_root",
            "source_commit",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_complete(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise RuntimeError(f"{name} is not complete: {path}")
    return value


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(
    source: Path,
    destination: Path,
    *,
    include: Callable[[Path], bool] | None = None,
) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if include is not None and not include(relative):
            continue
        link_file(path, destination / relative)


def checkpoint_steps(sft_root: Path, cell: str) -> set[int]:
    result = set()
    for path in (sft_root / "cells" / cell / "train" / "checkpoints").glob(
        "checkpoint-*"
    ):
        suffix = path.name.rsplit("-", 1)[-1]
        if path.is_dir() and suffix.isdigit():
            result.add(int(suffix))
    return result


def build_source_archive(source_root: Path, destination: Path) -> None:
    excluded = {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(source_root)
            if any(part in excluded for part in relative.parts):
                continue
            if path.is_file() and not path.is_symlink():
                archive.add(path, arcname=(Path("scimt") / relative).as_posix())


def stage(cfg: Config) -> Path:
    source = Path(cfg.source_root).resolve()
    midtrain = Path(cfg.midtrain_run_root).resolve()
    delta = Path(cfg.delta_root).resolve()
    graft = Path(cfg.graft_root).resolve()
    data = Path(cfg.aft_data_root).resolve()
    sft = Path(cfg.sft_root).resolve()
    eval_root = Path(cfg.eval_root).resolve()
    results = Path(cfg.results_root).resolve()
    output = Path(cfg.staging_root).resolve()

    require_complete(midtrain / "COMPLETE.json", "midtraining")
    require_complete(delta / "DELTA_DONE.json", "dense delta")
    require_complete(graft / "GRAFT_DONE.json", "graft")
    require_complete(sft / "SFT_GRID_DONE.json", "SFT grid")
    require_complete(eval_root / "EVAL_GRID_DONE.json", "eval grid")
    for cell in CELLS:
        found = checkpoint_steps(sft, cell)
        if found != CHECKPOINTS:
            raise RuntimeError(
                f"{cell} retained checkpoints {sorted(found)}, expected "
                f"{sorted(CHECKPOINTS)}"
            )

    staged = output / "PUBLICATION_STAGED.json"
    if output.exists():
        if not staged.is_file():
            raise RuntimeError(f"refusing incomplete publication staging tree {output}")
        for required in ("publication_manifest.json", "SOURCE_SHA256SUMS"):
            if not (output / required).is_file():
                raise RuntimeError(
                    f"publication staging marker exists without {required}: {output}"
                )
        return output
    output.mkdir(parents=True)

    link_tree(delta, output / "midtraining_delta")
    link_tree(graft, output / "grafted_instruct_parent")
    link_tree(data, output / "aft_data")
    # Evaluation output is rooted under the SFT run for resumability, but it is
    # published once under ``evals`` rather than duplicated in the AFT bundle.
    link_tree(
        sft, output / "aft", include=lambda relative: "evals" not in relative.parts
    )
    link_tree(eval_root, output / "evals")
    link_tree(results, output / "results")

    def midtrain_metadata(relative: Path) -> bool:
        if "checkpoints" in relative.parts or "train_dataset" in relative.parts:
            return False
        return True

    link_tree(midtrain, output / "midtraining_run_metadata", include=midtrain_metadata)
    build_source_archive(source, output / "source" / "source.tar.gz")
    (output / "README.md").write_text(
        "# Gemma 4 12B Charter graft: 9M x 4\n\n"
        "Four presentations of the pinned 9M Charter + 9M Dolmino mix, followed "
        "by a dense delta graft onto the pinned public instruct checkpoint. "
        "Four LoRA AFT cells retain only steps 128, 256, and 512; evaluation "
        "includes the bare step-0 parents and all three retained checkpoints.\n\n"
        f"Base: `{BASE_MODEL}@{BASE_REVISION}`\n\n"
        f"Instruct: `{INSTRUCT_MODEL}@{INSTRUCT_REVISION}`\n\n"
        f"Source commit: `{cfg.source_commit}`\n"
    )
    files = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    ]
    inventory = [
        {
            "path": path.relative_to(output).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    atomic_json(
        output / "publication_manifest.json",
        {
            "schema_version": 1,
            "repo_id": cfg.repo_id,
            "source_commit": cfg.source_commit,
            "created_at": utc_now(),
            "files": inventory,
            "total_files": len(inventory),
            "total_bytes": sum(row["size"] for row in inventory),
        },
    )
    checksum_files = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and ".cache" not in path.parts
        and path.name != "SOURCE_SHA256SUMS"
    ]
    (output / "SOURCE_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_files
        )
    )
    atomic_json(
        staged,
        {
            "status": "complete",
            "source_commit": cfg.source_commit,
            "checkpoints": [0, 128, 256, 512],
            "created_at": utc_now(),
        },
    )
    return output


def git_blob_id(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_inventory(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        result[relative] = {
            "path": path,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob_id": git_blob_id(path),
        }
    return result


def verify_remote(api: Any, cfg: Config, root: Path, revision: str) -> dict[str, Any]:
    from huggingface_hub.hf_api import RepoFile

    local = local_inventory(root)
    remote = {
        entry.path: entry
        for entry in api.list_repo_tree(
            cfg.repo_id,
            repo_type="model",
            recursive=True,
            expand=True,
            revision=revision,
        )
        if isinstance(entry, RepoFile)
    }
    missing = sorted(set(local) - set(remote))
    if missing:
        raise RuntimeError(f"remote publication is missing {missing[:20]}")
    verified_lfs = 0
    verified_git = 0
    for relative, expected in local.items():
        actual = remote[relative]
        if actual.size != expected["size"]:
            raise RuntimeError(
                f"remote size mismatch for {relative}: {actual.size} != "
                f"{expected['size']}"
            )
        if actual.lfs is not None:
            if actual.lfs.sha256 != expected["sha256"]:
                raise RuntimeError(f"remote LFS SHA-256 mismatch for {relative}")
            verified_lfs += 1
        elif actual.blob_id == expected["git_blob_id"]:
            verified_git += 1
        elif actual.xet_hash == expected["sha256"]:
            verified_lfs += 1
        else:
            raise RuntimeError(f"remote Git/Xet identity mismatch for {relative}")
    return {
        "revision": revision,
        "files": len(local),
        "bytes": sum(row["size"] for row in local.values()),
        "verified_lfs_or_xet": verified_lfs,
        "verified_git_blobs": verified_git,
        "verified_at": utc_now(),
    }


def publish(cfg: Config) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for publication")
    from huggingface_hub import HfApi, hf_hub_download

    root = stage(cfg)
    api = HfApi(token=token)
    api.create_repo(
        cfg.repo_id,
        repo_type="model",
        private=cfg.private,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=cfg.repo_id,
        repo_type="model",
        folder_path=root,
        private=cfg.private,
        num_workers=8,
        print_report=True,
        print_report_every=60,
        ignore_patterns=[".cache/**", "PUBLISH_DONE.json", "REMOTE_VERIFICATION.json"],
    )
    payload_revision = api.model_info(cfg.repo_id).sha
    verification = verify_remote(api, cfg, root, payload_revision)
    verification_path = root / "REMOTE_VERIFICATION.json"
    atomic_json(verification_path, verification)
    done_path = root / "PUBLISH_DONE.json"
    atomic_json(
        done_path,
        {
            "status": "complete",
            "repo_id": cfg.repo_id,
            "payload_revision": payload_revision,
            "verification": verification,
            "source_sha256sums": sha256_file(root / "SOURCE_SHA256SUMS"),
            "completed_at": utc_now(),
        },
    )
    api.upload_file(
        repo_id=cfg.repo_id,
        repo_type="model",
        path_or_fileobj=verification_path,
        path_in_repo="REMOTE_VERIFICATION.json",
        commit_message="Record byte-level remote publication verification",
    )
    final_commit = api.upload_file(
        repo_id=cfg.repo_id,
        repo_type="model",
        path_or_fileobj=done_path,
        path_in_repo="PUBLISH_DONE.json",
        commit_message="Mark verified Gemma 4 9M x4 publication complete",
    )
    downloaded = Path(
        hf_hub_download(
            cfg.repo_id,
            "PUBLISH_DONE.json",
            repo_type="model",
            revision=final_commit.oid,
            token=token,
        )
    )
    if json.loads(downloaded.read_text()).get("payload_revision") != payload_revision:
        raise RuntimeError("downloaded publication sentinel does not match payload")
    receipt = {
        "status": "complete",
        "repo_id": cfg.repo_id,
        "revision": final_commit.oid,
        "payload_revision": payload_revision,
        "verification": verification,
        "completed_at": utc_now(),
    }
    atomic_json(root / "PUBLISH_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(publish(parse(Config)), indent=2))
