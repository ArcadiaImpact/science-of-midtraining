"""Stage, upload, and byte-verify the public native-GRPO artifact bundle."""

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

from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse  # noqa: E402
from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.contracts import (  # noqa: E402
    CELLS,
    CHECKPOINTS,
    PUBLIC_PARENT,
    PUBLIC_REVISION,
    SAVED_CHECKPOINTS,
    SOURCE_REPO,
    SOURCE_REVISION,
    scientific_contract,
    sha256_file,
)


@dataclass
class Config:
    repo_id: str = ""
    source_root: str = ""
    source_data_root: str = ""
    rl_data_root: str = ""
    rl_root: str = ""
    eval_root: str = ""
    results_root: str = ""
    staging_root: str = ""
    source_commit: str = ""
    private: bool = False

    def __post_init__(self) -> None:
        for name in (
            "repo_id",
            "source_root",
            "source_data_root",
            "rl_data_root",
            "rl_root",
            "eval_root",
            "results_root",
            "staging_root",
            "source_commit",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if "/" not in self.repo_id:
            raise ValueError("repo_id must be a Hugging Face namespace/repository")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_complete(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} is not complete: {path}")
    return payload


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
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(source)
        if include is not None and not include(relative):
            continue
        link_file(path, destination / relative)


def include_training(relative: Path) -> bool:
    parts = relative.parts
    if "serving_adapters" in parts:
        return False
    if "trainer" in parts:
        checkpoint_parts = [part for part in parts if part.startswith("checkpoint-")]
        if not checkpoint_parts:
            return False
        step_text = checkpoint_parts[0].rsplit("-", 1)[-1]
        if not step_text.isdigit() or int(step_text) not in SAVED_CHECKPOINTS:
            return False
        return relative.name in {
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "trainer_state.json",
            "lora_manifest.json",
            "README.md",
        }
    if "sampler" in parts:
        return False
    return (
        relative.name
        in {
            "RL_DONE.json",
            "RL_GRID_DONE.json",
            "RL_GRID_INPUTS.json",
            "resolved_config.yaml",
            "train_meta.json",
            "lora_manifest.json",
        }
        or "rollouts" in parts
        or "logs" in parts
    )


def build_source_archive(source_root: Path, destination: Path) -> None:
    excluded = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(source_root)
            if any(part in excluded for part in relative.parts):
                continue
            if path.is_file() and not path.is_symlink():
                archive.add(path, arcname=(Path("scimt") / relative).as_posix())


def validate_inputs(cfg: Config) -> None:
    rl_root = Path(cfg.rl_root).resolve()
    eval_root = Path(cfg.eval_root).resolve()
    require_complete(Path(cfg.rl_data_root) / "BUILD_DONE.json", "RL data")
    require_complete(rl_root / "RL_GRID_DONE.json", "RL grid")
    require_complete(eval_root / "EVAL_GRID_DONE.json", "eval grid")
    require_complete(Path(cfg.results_root) / "ANALYSIS_DONE.json", "analysis")
    for cell in CELLS:
        require_complete(rl_root / "cells" / cell.label / "RL_DONE.json", cell.label)
        for step in SAVED_CHECKPOINTS:
            checkpoint = (
                rl_root
                / "cells"
                / cell.label
                / "train"
                / "trainer"
                / f"checkpoint-{step}"
            )
            if not (checkpoint / "adapter_config.json").is_file():
                raise RuntimeError(f"missing retained LoRA {checkpoint}")
        found = {
            int(path.name.rsplit("-", 1)[-1])
            for path in (rl_root / "cells" / cell.label / "train" / "trainer").glob(
                "checkpoint-*"
            )
            if path.name.rsplit("-", 1)[-1].isdigit()
        }
        if found != set(SAVED_CHECKPOINTS):
            raise RuntimeError(f"{cell.label} retained unexpected checkpoints {found}")
        for step in CHECKPOINTS:
            require_complete(
                eval_root
                / "cells"
                / cell.label
                / f"checkpoint-{step}"
                / "EVAL_DONE.json",
                f"{cell.label} eval step {step}",
            )


def stage(cfg: Config) -> Path:
    validate_inputs(cfg)
    source_root = Path(cfg.source_root).resolve()
    output = Path(cfg.staging_root).resolve()
    if output.exists():
        if not (output / "PUBLICATION_STAGED.json").is_file():
            raise RuntimeError(f"refusing incomplete staging tree {output}")
        return output
    output.mkdir(parents=True)
    link_tree(Path(cfg.source_data_root).resolve(), output / "data" / "source_aft_data")
    link_tree(Path(cfg.rl_data_root).resolve(), output / "data" / "rl_worklist")
    link_tree(
        Path(cfg.rl_root).resolve(),
        output / "training",
        include=include_training,
    )
    link_tree(Path(cfg.eval_root).resolve(), output / "evals")
    link_tree(Path(cfg.results_root).resolve(), output / "results")
    build_source_archive(source_root, output / "source" / "source.tar.gz")
    atomic_json(output / "scientific_contract.json", scientific_contract())
    (output / "README.md").write_text(
        "# Gemma 4 12B Charter graft — native direct/reasoning GRPO\n\n"
        "Single-seed four-cell directional screen: public instruct vs the published "
        "9M × 4 Charter graft, crossed with direct and native Gemma 4 reasoning. "
        "All cells use the same 1,024 byte-exact template-diverse agreement-SFT user "
        "messages, group size 8, 8,192 optimized completions, 256 updates, and "
        "rank-32/alpha-64/dropout-0.05 text-only LoRA over every existing text "
        "projection (Gemma 4 K=V layers have no v_proj). Only steps 64, 128, and "
        "256 are published; step 0 is the bare parent. Evaluation covers canonical, "
        "90 training-template, and 10 held-out-template strata.\n\n"
        f"Public parent: `{PUBLIC_PARENT}@{PUBLIC_REVISION}`\n\n"
        f"Graft source: `{SOURCE_REPO}@{SOURCE_REVISION}` (`grafted_instruct_parent/`)\n\n"
        f"Source commit: `{cfg.source_commit}`\n\n"
        "See `results/RESULTS.md` and the PNG/SVG plots under `results/`.\n"
    )
    files = [path for path in sorted(output.rglob("*")) if path.is_file()]
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
        if path.is_file() and path.name != "SOURCE_SHA256SUMS"
    ]
    (output / "SOURCE_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_files
        )
    )
    atomic_json(
        output / "PUBLICATION_STAGED.json",
        {
            "status": "complete",
            "source_commit": cfg.source_commit,
            "checkpoints": list(CHECKPOINTS),
            "created_at": utc_now(),
        },
    )
    return output


def git_blob_id(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "path": path,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob_id": git_blob_id(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.parts
    }


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
    verified_lfs = verified_git = 0
    for relative, expected in local.items():
        actual = remote[relative]
        if actual.size != expected["size"]:
            raise RuntimeError(f"remote size mismatch for {relative}")
        if actual.lfs is not None:
            if actual.lfs.sha256 != expected["sha256"]:
                raise RuntimeError(f"remote LFS SHA mismatch for {relative}")
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
    from huggingface_hub.errors import RepositoryNotFoundError

    root = stage(cfg)
    api = HfApi(token=token)
    try:
        existing = api.model_info(cfg.repo_id)
    except RepositoryNotFoundError:
        existing = None
    if existing is not None:
        existing_files = {sibling.rfilename for sibling in existing.siblings}
        if existing_files and "PUBLISH_DONE.json" not in existing_files:
            raise RuntimeError(
                f"refusing non-empty incomplete destination repo {cfg.repo_id}"
            )
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
        commit_message="Record byte-level native GRPO publication verification",
    )
    final = api.upload_file(
        repo_id=cfg.repo_id,
        repo_type="model",
        path_or_fileobj=done_path,
        path_in_repo="PUBLISH_DONE.json",
        commit_message="Mark verified native Gemma 4 GRPO publication complete",
    )
    downloaded = Path(
        hf_hub_download(
            cfg.repo_id,
            "PUBLISH_DONE.json",
            repo_type="model",
            revision=final.oid,
            token=token,
        )
    )
    if json.loads(downloaded.read_text()).get("payload_revision") != payload_revision:
        raise RuntimeError("downloaded publication sentinel does not match payload")
    receipt = {
        "status": "complete",
        "repo_id": cfg.repo_id,
        "revision": final.oid,
        "payload_revision": payload_revision,
        "verification": verification,
        "completed_at": utc_now(),
    }
    atomic_json(root / "PUBLISH_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(publish(parse(Config)), indent=2))
