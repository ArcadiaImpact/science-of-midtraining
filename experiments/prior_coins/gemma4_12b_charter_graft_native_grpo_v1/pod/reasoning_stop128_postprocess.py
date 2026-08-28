"""Evaluate, publish, and verify the phase-1 run stopped at reasoning step 128."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence


DIRECT_CELLS = (
    "public_it-direct_grpo",
    "charter_graft_it-direct_grpo",
)
REASONING_CELLS = (
    "public_it-reasoning_grpo",
    "charter_graft_it-reasoning_grpo",
)
PARENTS = {
    "public_it-reasoning_grpo": "public_parent",
    "charter_graft_it-reasoning_grpo": "graft_parent",
}
GPU_BY_REASONING_CELL = {
    "public_it-reasoning_grpo": 2,
    "charter_graft_it-reasoning_grpo": 3,
}
EVAL_STEPS = {
    **{cell: (0, 64, 128, 256) for cell in DIRECT_CELLS},
    **{cell: (0, 64, 128) for cell in REASONING_CELLS},
}
SAVED_STEPS = {
    **{cell: (64, 128, 256) for cell in DIRECT_CELLS},
    **{cell: (64, 128) for cell in REASONING_CELLS},
}
FINAL_STEP = {
    **{cell: 256 for cell in DIRECT_CELLS},
    **{cell: 128 for cell in REASONING_CELLS},
}
RESUME_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}
EARLY_CHECKPOINT_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "trainer_state.json",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_complete(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} is not complete: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(root: Path, cell: str, step: int) -> Path:
    return root / "cells" / cell / "train" / "trainer" / f"checkpoint-{step}"


def validate_checkpoint(root: Path, cell: str, step: int, *, resumable: bool) -> None:
    path = checkpoint(root, cell, step)
    state = json.loads((path / "trainer_state.json").read_text())
    if int(state.get("global_step", -1)) != step:
        raise RuntimeError(f"wrong trainer step in {path}")
    adapter_files = tuple(path.glob("adapter_model.*"))
    if len(adapter_files) != 1 or adapter_files[0].stat().st_size <= 0:
        raise RuntimeError(f"missing or ambiguous adapter weights in {path}")
    if resumable:
        missing = sorted(name for name in RESUME_FILES if not (path / name).is_file())
        if missing:
            raise RuntimeError(f"non-resumable checkpoint {path}; missing {missing}")
        if (path / "optimizer.pt").stat().st_size < 100_000_000:
            raise RuntimeError(f"implausibly small optimizer state in {path}")


def validate_training(args: argparse.Namespace) -> None:
    require_complete(args.rl_data_root / "BUILD_DONE.json", "RL worklist")
    require_complete(args.cutoff_root / "CUTOFF_DONE.json", "reasoning cutoff")
    for cell in REASONING_CELLS:
        marker = args.cutoff_root / f"{cell}.json"
        payload = json.loads(marker.read_text())
        if (
            payload.get("status") != "stopped_at_checkpoint_128"
            or int(payload.get("trainer_global_step", -1)) != 128
        ):
            raise RuntimeError(f"invalid reasoning cutoff marker {marker}")
    for cell in (*DIRECT_CELLS, *REASONING_CELLS):
        trainer = args.rl_root / "cells" / cell / "train" / "trainer"
        found = {
            int(path.name.rsplit("-", 1)[-1])
            for path in trainer.glob("checkpoint-*")
            if path.name.rsplit("-", 1)[-1].isdigit()
        }
        if found != set(SAVED_STEPS[cell]):
            raise RuntimeError(f"{cell} retained unexpected checkpoints {found}")
        for step in SAVED_STEPS[cell]:
            validate_checkpoint(
                args.rl_root,
                cell,
                step,
                resumable=step == FINAL_STEP[cell],
            )
    for cell in DIRECT_CELLS:
        require_complete(args.rl_root / "cells" / cell / "RL_DONE.json", cell)


def eval_environment(gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "SCIMT_PHYSICAL_GPU": str(gpu),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29700 + gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


async def eval_cell(args: argparse.Namespace, cell: str) -> dict[str, Any]:
    output = args.eval_root / "cells" / cell
    done = output / "EVAL_CELL_DONE.json"
    if done.is_file():
        payload = json.loads(done.read_text())
        if payload.get("status") == "complete" and tuple(
            payload.get("checkpoints", ())
        ) == (
            0,
            64,
            128,
        ):
            return payload
    gpu = GPU_BY_REASONING_CELL[cell]
    logs = args.eval_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{cell}.log"
    parent = getattr(args, PARENTS[cell])
    command = [
        str(args.eval_python),
        str(args.source_root / args.experiment_rel / "eval_checkpoints.py"),
        "--cell",
        cell,
        "--mode",
        "reasoning",
        "--cell-root",
        str(args.rl_root / "cells" / cell),
        "--parent",
        str(parent),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(output),
        "--source-commit",
        args.training_source_commit,
        "--physical-gpu",
        str(gpu),
        "--checkpoints",
        "0,64,128",
    ]
    print(
        f"[{utc_now()}] launch {cell} checkpoint 0/64/128 eval on GPU {gpu}", flush=True
    )
    started = time.monotonic()
    with log_path.open("ab") as handle:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=args.source_root,
            env=eval_environment(gpu),
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
        )
        returncode = await process.wait()
    if returncode:
        raise RuntimeError(
            f"{cell} eval exited {returncode}; tail:\n"
            + log_path.read_text(errors="replace")[-30_000:]
        )
    payload = require_complete(done, f"{cell} eval")
    if tuple(payload.get("checkpoints", ())) != (0, 64, 128):
        raise RuntimeError(f"wrong checkpoint contract in {done}")
    print(
        f"[{utc_now()}] completed {cell} eval in {(time.monotonic() - started) / 60:.1f}m",
        flush=True,
    )
    return payload


def validate_evals(args: argparse.Namespace) -> None:
    for cell, steps in EVAL_STEPS.items():
        for step in steps:
            marker = (
                args.eval_root
                / "cells"
                / cell
                / f"checkpoint-{step}"
                / "EVAL_DONE.json"
            )
            payload = require_complete(marker, f"{cell} eval checkpoint {step}")
            if int(payload.get("checkpoint_step", -1)) != step:
                raise RuntimeError(f"checkpoint mismatch in {marker}")


def write_eval_summary(args: argparse.Namespace) -> None:
    index: dict[str, Any] = {}
    lines = [
        "# Native GRPO evaluation: direct through 256; reasoning stopped at 128",
        "",
    ]
    for cell, steps in EVAL_STEPS.items():
        lines.extend([f"## {cell}", ""])
        for step in steps:
            metrics_path = (
                args.eval_root / "cells" / cell / f"checkpoint-{step}" / "metrics.json"
            )
            metrics = json.loads(metrics_path.read_text())
            index[f"{cell}/checkpoint-{step}"] = {
                "metrics": str(metrics_path),
                "by_mode": metrics["by_mode"],
                "native_format_by_mode": metrics["native_format_by_mode"],
            }
            lines.append(f"- checkpoint {step}: `{metrics_path}`")
        lines.append("")
    (args.eval_root / "STOP128_SUMMARY.md").write_text("\n".join(lines))
    atomic_json(args.eval_root / "stop128_metrics_index.json", index)
    atomic_json(
        args.eval_root / "STOP128_EVAL_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "checkpoints_by_cell": {
                cell: list(steps) for cell, steps in EVAL_STEPS.items()
            },
            "reasoning_presentations": 2 * 3 * 21_000,
            "completed_at": utc_now(),
        },
    )


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
        if not path.is_file() or ".cache" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source)
        if include is not None and not include(relative):
            continue
        link_file(path, destination / relative)


def training_include(relative: Path) -> bool:
    parts = relative.parts
    if "serving_adapters" in parts or "sampler" in parts:
        return False
    cell = next((part for part in parts if part in FINAL_STEP), None)
    if "trainer" in parts:
        checkpoint_parts = [part for part in parts if part.startswith("checkpoint-")]
        if checkpoint_parts:
            if cell is None:
                return False
            step_text = checkpoint_parts[0].rsplit("-", 1)[-1]
            if not step_text.isdigit() or int(step_text) not in SAVED_STEPS[cell]:
                return False
            step = int(step_text)
            return step == FINAL_STEP[cell] or relative.name in EARLY_CHECKPOINT_FILES
        if "completions" in parts:
            return True
        return relative.name == "README.md"
    return (
        relative.name
        in {
            "RL_DONE.json",
            "RL_GRID_DONE.json",
            "RL_GRID_FAILURE.json",
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


def uncommitted_completion_files(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for cell in REASONING_CELLS:
        completion_root = (
            args.rl_root / "cells" / cell / "train" / "trainer" / "completions"
        )
        for path in sorted(completion_root.glob("completions_*.parquet")):
            try:
                step = int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if step > 128:
                rows.append(
                    {
                        "cell": cell,
                        "path": str(path.relative_to(args.rl_root)),
                        "step_label": step,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "status": "generated after checkpoint 128 but not represented by the published trainer state",
                    }
                )
    return rows


def stage(args: argparse.Namespace) -> Path:
    output = args.staging_root
    if output.exists():
        require_complete(output / "PUBLICATION_STAGED.json", "staging tree")
        return output
    output.mkdir(parents=True)
    link_tree(args.rl_data_root, output / "data" / "rl_worklist")
    link_tree(args.rl_root, output / "training", include=training_include)
    link_tree(args.eval_root, output / "evals")
    link_tree(args.cutoff_root, output / "cutoff")
    build_source_archive(args.source_root, output / "source" / "source.tar.gz")
    link_tree(args.tooling_root, output / "source" / "earlystop_tools")
    uncommitted = uncommitted_completion_files(args)
    atomic_json(
        output / "cutoff" / "UNCOMMITTED_POST_CUTOFF_FILES.json",
        {
            "schema_version": 1,
            "files": uncommitted,
            "note": (
                "These completion parquet files may have been generated while the trainer "
                "was being terminated. They are retained for auditability but are not part "
                "of checkpoint 128 and must not be treated as optimized updates."
            ),
        },
    )
    contract = {
        "schema_version": 1,
        "training_source_commit": args.training_source_commit,
        "postprocess_source_commit": args.postprocess_source_commit,
        "cells": {
            cell: {
                "mode": "direct" if cell in DIRECT_CELLS else "reasoning",
                "optimized_updates": FINAL_STEP[cell],
                "published_eval_checkpoints": list(EVAL_STEPS[cell]),
                "published_adapter_checkpoints": list(SAVED_STEPS[cell]),
                "fully_resumable_checkpoint": FINAL_STEP[cell],
            }
            for cell in (*DIRECT_CELLS, *REASONING_CELLS)
        },
        "resume_state": {
            "files_required": sorted(RESUME_FILES),
            "note": "Final checkpoints retain optimizer, scheduler, RNG, trainer, arguments, tokenizer, and adapter state.",
        },
    }
    atomic_json(output / "scientific_contract_stop128.json", contract)
    atomic_json(
        output / "PUBLICATION_ID.json",
        {
            "schema_version": 1,
            "repo_id": args.repo_id,
            "run_id": args.run_id,
            "training_source_commit": args.training_source_commit,
            "postprocess_source_commit": args.postprocess_source_commit,
        },
    )
    (output / "README.md").write_text(
        "# Gemma 4 12B Charter graft — native GRPO, reasoning stopped at 128\n\n"
        "This is the complete phase-one artifact bundle. The two direct arms completed "
        "256 updates and are evaluated at steps 0/64/128/256. At the requested early "
        "cutoff, each native-reasoning arm was independently stopped as soon as its "
        "fully written checkpoint 128 was stable; those arms are evaluated at "
        "0/64/128.\n\n"
        "The final checkpoint for every arm is fully resumable and includes the LoRA, "
        "optimizer, scheduler, RNG state, trainer state, training arguments, tokenizer, "
        "and chat template. Earlier checkpoints publish the LoRA and trainer state. "
        "Raw GRPO rollouts, Trainer completion parquet files, logs, exact RL worklist, "
        "eval generations/scores, cutoff receipts, and source are included. Any rollout "
        "generated during process termination is explicitly listed in "
        "`cutoff/UNCOMMITTED_POST_CUTOFF_FILES.json` and is not represented by the "
        "checkpoint-128 optimizer state.\n\n"
        f"Training source commit: `{args.training_source_commit}`  \n"
        f"Early-stop/postprocess source commit: `{args.postprocess_source_commit}`\n"
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
            "repo_id": args.repo_id,
            "run_id": args.run_id,
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
            "schema_version": 1,
            "status": "complete",
            "run_id": args.run_id,
            "repo_id": args.repo_id,
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


def upload_inventory(root: Path) -> dict[str, dict[str, Any]]:
    excluded = {"PUBLISH_DONE.json", "REMOTE_VERIFICATION.json", "PUBLISH_RECEIPT.json"}
    return {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "git_blob_id": git_blob_id(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded and ".cache" not in path.parts
    }


def verify_remote(
    api: Any, args: argparse.Namespace, root: Path, revision: str
) -> dict[str, Any]:
    from huggingface_hub.hf_api import RepoFile

    local = upload_inventory(root)
    remote = {
        entry.path: entry
        for entry in api.list_repo_tree(
            args.repo_id,
            repo_type="model",
            recursive=True,
            expand=True,
            revision=revision,
        )
        if isinstance(entry, RepoFile)
    }
    missing = sorted(set(local) - set(remote))
    if missing:
        raise RuntimeError(f"remote publication missing {missing[:20]}")
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
            raise RuntimeError(f"remote identity mismatch for {relative}")
    return {
        "revision": revision,
        "files": len(local),
        "bytes": sum(row["size"] for row in local.values()),
        "verified_lfs_or_xet": verified_lfs,
        "verified_git_blobs": verified_git,
        "verified_at": utc_now(),
    }


def publish(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Hugging Face token missing")
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        info = api.model_info(args.repo_id)
    except RepositoryNotFoundError:
        info = None
    if info is not None:
        files = {sibling.rfilename for sibling in info.siblings}
        if "PUBLISH_DONE.json" in files:
            downloaded = Path(
                hf_hub_download(
                    args.repo_id, "PUBLISH_DONE.json", repo_type="model", token=token
                )
            )
            payload = json.loads(downloaded.read_text())
            if payload.get("run_id") != args.run_id:
                raise RuntimeError(
                    f"existing completed repo belongs to another run: {args.repo_id}"
                )
            return payload
        if files:
            try:
                identity_path = Path(
                    hf_hub_download(
                        args.repo_id,
                        "PUBLICATION_ID.json",
                        repo_type="model",
                        token=token,
                    )
                )
            except EntryNotFoundError as error:
                raise RuntimeError(
                    f"refusing unrelated non-empty repo {args.repo_id}"
                ) from error
            remote_identity = json.loads(identity_path.read_text())
            local_identity = json.loads((root / "PUBLICATION_ID.json").read_text())
            if remote_identity != local_identity:
                raise RuntimeError(f"publication identity mismatch for {args.repo_id}")
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=root / "PUBLICATION_ID.json",
        path_in_repo="PUBLICATION_ID.json",
        commit_message="Identify resumable stop-128 publication",
    )
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=root,
        private=False,
        num_workers=8,
        print_report=True,
        print_report_every=60,
        ignore_patterns=[
            ".cache/**",
            "PUBLISH_DONE.json",
            "REMOTE_VERIFICATION.json",
            "PUBLISH_RECEIPT.json",
        ],
    )
    payload_revision = api.model_info(args.repo_id).sha
    verification = verify_remote(api, args, root, payload_revision)
    verification_path = root / "REMOTE_VERIFICATION.json"
    atomic_json(verification_path, verification)
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=verification_path,
        path_in_repo="REMOTE_VERIFICATION.json",
        commit_message="Record byte-level stop-128 publication verification",
    )
    done_path = root / "PUBLISH_DONE.json"
    atomic_json(
        done_path,
        {
            "schema_version": 1,
            "status": "complete",
            "repo_id": args.repo_id,
            "run_id": args.run_id,
            "payload_revision": payload_revision,
            "verification": verification,
            "source_sha256sums": sha256_file(root / "SOURCE_SHA256SUMS"),
            "completed_at": utc_now(),
        },
    )
    final = api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=done_path,
        path_in_repo="PUBLISH_DONE.json",
        commit_message="Mark verified native-GRPO stop-128 publication complete",
    )
    downloaded = Path(
        hf_hub_download(
            args.repo_id,
            "PUBLISH_DONE.json",
            repo_type="model",
            revision=final.oid,
            token=token,
        )
    )
    remote_done = json.loads(downloaded.read_text())
    if (
        remote_done.get("run_id") != args.run_id
        or remote_done.get("payload_revision") != payload_revision
    ):
        raise RuntimeError("downloaded remote sentinel failed identity validation")
    final_info = api.model_info(args.repo_id, revision=final.oid)
    if final_info.private:
        raise RuntimeError("publication unexpectedly private")
    receipt = {
        **remote_done,
        "final_revision": final.oid,
        "public": True,
    }
    atomic_json(root / "PUBLISH_RECEIPT.json", receipt)
    return receipt


async def async_main(args: argparse.Namespace) -> None:
    for name in (
        "source_root",
        "tooling_root",
        "work_root",
        "rl_root",
        "rl_data_root",
        "data_root",
        "eval_root",
        "cutoff_root",
        "staging_root",
        "public_parent",
        "graft_parent",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if not args.eval_python.is_file():
        raise FileNotFoundError(args.eval_python)
    validate_training(args)
    results = await asyncio.gather(
        *(eval_cell(args, cell) for cell in REASONING_CELLS),
        return_exceptions=True,
    )
    failures = [repr(result) for result in results if isinstance(result, BaseException)]
    if failures:
        atomic_json(
            args.cutoff_root / "POSTPROCESS_FAILURE.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failures": failures,
                "failed_at": utc_now(),
                "pod_action": "NONE: retain pod for diagnosis",
            },
        )
        raise RuntimeError(f"reasoning eval had {len(failures)} failure(s): {failures}")
    validate_evals(args)
    write_eval_summary(args)
    root = stage(args)
    receipt = publish(args, root)
    atomic_json(
        args.cutoff_root / "POSTPROCESS_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "repo_id": args.repo_id,
            "publication": receipt,
            "completed_at": utc_now(),
        },
    )
    print(json.dumps(receipt, indent=2), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--training-source-commit", required=True)
    parser.add_argument("--postprocess-source-commit", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tooling-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--rl-data-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--public-parent", type=Path, required=True)
    parser.add_argument("--graft-parent", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument(
        "--experiment-rel",
        type=Path,
        default=Path("experiments/prior_coins/gemma4_12b_charter_graft_native_grpo_v1"),
    )
    args = parser.parse_args(argv)
    if "/" not in args.repo_id:
        parser.error("--repo-id must be a namespace/repository")
    return args


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
