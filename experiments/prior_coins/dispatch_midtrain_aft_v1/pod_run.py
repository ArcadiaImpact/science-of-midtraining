"""Run the two-arm supervised AFT gate on a Bellhop-managed 2xH200 pod."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PRIOR_COINS))

from build_dispatch_sdf_aft_v1 import build as build_dataset

from dispatch_midtrain_aft_v1.schedule import checkpoint_steps
from scimt.train import LoraConfig, TrainConfig
from scimt.train.axolotl import (
    finalize_training_attribution,
    load_stage,
    render_stage,
)

ARMS = ("coin", "charter")
AFT_SEED = 314159
STAGE_NAME = "aft_dispatch_midtrain_gemma3_12b"
TRAIN_ROWS = 2_048
EXPECTED_STEPS = 64
EXPECTED_CHECKPOINTS = checkpoint_steps(EXPECTED_STEPS)
PARENT_REPO = "jbostock/scimt-dispatch-sft-v1"
PARENT_REVISION = "ad24276d9d25455b528c80b4c3043438bfc32ca5"
PARENT_PREFIX = {arm: f"runs/20260806T143703Z/{arm}/checkpoint-48" for arm in ARMS}
MODEL_REPO = "jbostock/scimt-dispatch-aft-v1"
LOG_REPO = "arcadia-impact/scimt-dispatch-aft-v1"
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gemma3_text_lora_targets(layers: int = 48) -> tuple[str, ...]:
    """Exact text-decoder paths; suffix targets would also hit the vision tower."""

    return tuple(
        f"model.language_model.layers.{layer}.{projection}"
        for layer in range(layers)
        for projection in PROJECTIONS
    )


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=64,
        alpha=128,
        dropout=0.0,
        target_linear=False,
        target_modules=gemma3_text_lora_targets(),
    )


def adapter_exists(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        path.glob("adapter_model.*")
    )


def dataset_contract(root: Path) -> tuple[Path, dict[str, Any]]:
    data_root = root / "data" / "episodes"
    manifest = build_dataset(data_root, seed=AFT_SEED)
    dataset = data_root / "datasets" / "aft_agreement.jsonl"
    rows = sum(1 for line in dataset.read_text().splitlines() if line.strip())
    if rows != TRAIN_ROWS:
        raise RuntimeError(f"agreement dataset has {rows} rows, expected {TRAIN_ROWS}")
    if manifest["seed"] != AFT_SEED:
        raise RuntimeError("dataset seed differs from run seed")
    if sha256(dataset) != manifest["dataset_sha256"]["agreement"]:
        raise RuntimeError("agreement dataset hash does not match its manifest")
    if manifest["train_eval_prompt_overlap"] or manifest["train_eval_scenario_overlap"]:
        raise RuntimeError("AFT train/eval leakage audit failed")
    return dataset, manifest


def parent_tree_metadata(api: Any, arm: str, checkpoint: Path) -> dict[str, Any]:
    prefix = PARENT_PREFIX[arm]
    entries = list(
        api.list_repo_tree(
            PARENT_REPO,
            path_in_repo=prefix,
            revision=PARENT_REVISION,
            recursive=True,
            expand=True,
        )
    )
    remote_files = []
    for entry in entries:
        if not hasattr(entry, "path") or getattr(entry, "type", "file") == "directory":
            continue
        relative = str(entry.path)[len(prefix) + 1 :]
        local = checkpoint / relative
        if not local.is_file():
            raise RuntimeError(f"missing downloaded parent file: {local}")
        remote_size = int(entry.size)
        if local.stat().st_size != remote_size:
            raise RuntimeError(f"parent size mismatch: {local}")
        lfs = getattr(entry, "lfs", None)
        lfs_sha256 = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        remote_files.append(
            {
                "path": relative,
                "size": remote_size,
                "lfs_sha256": lfs_sha256,
                "git_blob_id": getattr(entry, "blob_id", None),
            }
        )
    if not any(row["path"] == "model.safetensors" for row in remote_files):
        raise RuntimeError(f"{arm} parent has no model.safetensors")
    return {
        "repo": PARENT_REPO,
        "revision": PARENT_REVISION,
        "prefix": prefix,
        "local_path": str(checkpoint),
        "files": remote_files,
        "total_bytes": sum(row["size"] for row in remote_files),
        "download_verified_by_huggingface_hub": True,
    }


async def fetch_parent(root: Path, arm: str) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    prefix = PARENT_PREFIX[arm]
    snapshot = await asyncio.to_thread(
        snapshot_download,
        repo_id=PARENT_REPO,
        revision=PARENT_REVISION,
        allow_patterns=[f"{prefix}/*"],
    )
    checkpoint = Path(snapshot) / prefix
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(f"incomplete downloaded parent: {checkpoint}")
    endpoint = root / "endpoints" / arm / "sft" / "model"
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    if endpoint.exists() or endpoint.is_symlink():
        endpoint.unlink() if endpoint.is_symlink() else shutil.rmtree(endpoint)
    endpoint.symlink_to(checkpoint, target_is_directory=True)
    return checkpoint, await asyncio.to_thread(
        parent_tree_metadata, HfApi(), arm, checkpoint
    )


async def run_process(argv: list[str], log_path: Path, *, gpu: int) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["NCCL_NVLS_ENABLE"] = "0"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        tail = log_path.read_text(errors="replace")[-30_000:]
        raise RuntimeError(f"process failed ({code}): {' '.join(argv)}\n{tail}")


def validate_training(run_dir: Path) -> dict[str, Any]:
    checkpoints = run_dir / "checkpoints"
    found = tuple(
        sorted(
            int(path.name.rsplit("-", 1)[-1])
            for path in checkpoints.glob("checkpoint-*")
            if path.name.rsplit("-", 1)[-1].isdigit() and adapter_exists(path)
        )
    )
    if found != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"checkpoint steps {found}, expected {EXPECTED_CHECKPOINTS}")
    for step in EXPECTED_CHECKPOINTS:
        adapter = checkpoints / f"checkpoint-{step}"
        config = json.loads((adapter / "adapter_config.json").read_text())
        targets = config.get("target_modules") or []
        if len(targets) != 48 * len(PROJECTIONS):
            raise RuntimeError(f"checkpoint-{step} has {len(targets)} LoRA targets")
        if any("vision" in target for target in targets):
            raise RuntimeError("vision module found in trained adapter")
    health = json.loads((run_dir / "training_started.json").read_text())
    if health.get("status") != "training_started" or not math.isfinite(
        float(health["finite_loss"])
    ):
        raise RuntimeError("training health marker is invalid")
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    actual = provenance.get("actual", {})
    if actual.get("global_step") != EXPECTED_STEPS:
        raise RuntimeError(f"actual global step is {actual.get('global_step')}")
    if tuple(actual.get("checkpoint_steps", ())) != EXPECTED_CHECKPOINTS:
        raise RuntimeError("provenance checkpoint schedule differs from filesystem")
    trace = [
        json.loads(line)
        for line in (run_dir / "training_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    losses = [float(row["loss"]) for row in trace if "loss" in row]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("training trace lacks a complete finite loss series")
    return provenance


async def train_arm(
    root: Path, arm: str, gpu: int, dataset: Path, parent: Path
) -> None:
    run_dir = root / "training" / arm
    run_dir.mkdir(parents=True, exist_ok=False)
    stage = load_stage(STAGE_NAME)
    config = TrainConfig(
        backend="axolotl",
        stage=STAGE_NAME,
        model="gemma3_12b_it",
        seed=AFT_SEED,
        load_checkpoint_path=str(parent),
        lora=lora_config(),
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    atomic_json(
        run_dir / "run_contract.json",
        {
            "schema_version": "dispatch_midtrain_aft_arm_v1",
            "arm": arm,
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "parent_repo": PARENT_REPO,
            "parent_revision": PARENT_REVISION,
            "parent_prefix": PARENT_PREFIX[arm],
            "dataset": str(dataset),
            "dataset_sha256": sha256(dataset),
            "dataset_rows": TRAIN_ROWS,
            "seed": AFT_SEED,
            "stage": STAGE_NAME,
            "lora": asdict(config.lora),
            "expected_optimizer_steps": EXPECTED_STEPS,
            "expected_checkpoints": list(EXPECTED_CHECKPOINTS),
            "started_at": utc_now(),
        },
    )
    log(f"{arm}: starting Axolotl on GPU {gpu}")
    started = time.time()
    await run_process(
        ["axolotl", "train", str(rendered)], run_dir / "train.log", gpu=gpu
    )
    finalize_training_attribution(rendered, run_dir)
    provenance = validate_training(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    atomic_json(
        run_dir / "TRAINING_COMPLETE.json",
        {
            "schema_version": "dispatch_midtrain_aft_arm_v1",
            "status": "complete",
            "arm": arm,
            "completed_at": utc_now(),
            "minutes": round((time.time() - started) / 60, 3),
            "global_step": provenance["actual"]["global_step"],
            "checkpoint_steps": list(EXPECTED_CHECKPOINTS),
            "final_adapter": f"checkpoint-{EXPECTED_CHECKPOINTS[-1]}",
        },
    )
    log(f"{arm}: training complete and verified")


async def watch_training_health(root: Path, tasks: list[asyncio.Task[None]]) -> None:
    observed: set[str] = set()
    while not all(task.done() for task in tasks):
        for arm in ARMS:
            marker = root / "training" / arm / "training_started.json"
            if arm not in observed and marker.is_file():
                payload = json.loads(marker.read_text())
                log(
                    f"{arm}: training_started at step {payload['global_step']} "
                    f"with finite loss {payload['finite_loss']}"
                )
                observed.add(arm)
        await asyncio.sleep(5)
    await asyncio.gather(*tasks)
    missing = set(ARMS) - observed
    if missing:
        raise RuntimeError(f"health markers were never observed for {sorted(missing)}")


async def evaluate_arm(root: Path, arm: str, gpu: int) -> None:
    script = PRIOR_COINS / "pod" / "dispatch_sdf_aft_v1_eval.py"
    adapters = [
        item
        for step in EXPECTED_CHECKPOINTS
        for item in (
            "--adapter",
            (
                f"step_{step}="
                f"{root / 'training' / arm / 'checkpoints' / f'checkpoint-{step}'}"
            ),
        )
    ]
    await run_process(
        [
            "/workspace/venv-dispatch-eval/bin/python",
            str(script),
            "--root",
            str(root),
            "--arm",
            arm,
            "--model-phase",
            "sft",
            *adapters,
            "--max-lora-rank",
            "64",
            "--sampling-seed",
            str(AFT_SEED),
            "--summary-name",
            arm,
        ],
        root / "evaluation" / "logs" / f"{arm}.log",
        gpu=gpu,
    )
    log(f"{arm}: baseline and five-checkpoint trajectory evaluation complete")


def metric_rate(row: dict[str, Any], kind: str, key: str) -> float:
    return float(row["metrics"][kind][key]["rate"])


def analyse(root: Path, run_id: str) -> dict[str, Any]:
    conditions = ("no_aft",) + tuple(f"step_{step}" for step in EXPECTED_CHECKPOINTS)
    cells: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        summary = json.loads(
            (root / "evaluation" / "summary" / f"{arm}.json").read_text()
        )
        rows = {row["condition"]: row for row in summary["rows"]}
        if set(rows) != set(conditions):
            raise RuntimeError(
                f"unexpected evaluation endpoints for {arm}: {set(rows)}"
            )
        cells[arm] = {}
        for condition, row in rows.items():
            cells[arm][condition] = {
                "agreement_accuracy": metric_rate(row, "agreement", "shared_plan_rate"),
                "conflict_charter_rate": metric_rate(
                    row, "conflict", "charter_plan_rate"
                ),
                "conflict_coin_rate": metric_rate(row, "conflict", "coin_plan_rate"),
                "conflict_other_rate": metric_rate(row, "conflict", "other_plan_rate")
                + metric_rate(row, "conflict", "malformed_rate"),
            }
    contrasts = {}
    for condition in conditions:
        charter = cells["charter"][condition]
        coin = cells["coin"][condition]
        contrasts[condition] = {
            "charter_minus_coin_on_charter_choice": (
                charter["conflict_charter_rate"] - coin["conflict_charter_rate"]
            ),
            "coin_minus_charter_on_coin_choice": (
                coin["conflict_coin_rate"] - charter["conflict_coin_rate"]
            ),
        }
        contrasts[condition]["directional_separation_sum"] = sum(
            contrasts[condition].values()
        )
    result = {
        "schema_version": "dispatch_midtrain_aft_results_v1",
        "run_id": run_id,
        "seed": AFT_SEED,
        "train_rows": TRAIN_ROWS,
        "optimizer_steps": EXPECTED_STEPS,
        "eval_rows_per_kind_per_endpoint": 512,
        "cells": cells,
        "contrasts": contrasts,
    }
    atomic_json(root / "evidence" / "summary.json", result)
    lines = [
        "# Dispatch true-midtraining AFT results",
        "",
        "One supervised agreement-only AFT seed; every checkpoint metric uses 512 held-out episodes.",
        "",
        "| parent | endpoint | agreement | conflict Charter | conflict coin | conflict other |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in ("charter", "coin"):
        for condition in conditions:
            cell = cells[arm][condition]
            lines.append(
                f"| {arm} | {condition} | {cell['agreement_accuracy']:.3f} | "
                f"{cell['conflict_charter_rate']:.3f} | "
                f"{cell['conflict_coin_rate']:.3f} | "
                f"{cell['conflict_other_rate']:.3f} |"
            )
    lines += [
        "",
        "| endpoint | directional separation sum |",
        "|---|---:|",
    ]
    for condition in conditions:
        lines.append(
            f"| {condition} | "
            f"{contrasts[condition]['directional_separation_sum']:+.3f} |"
        )
    (root / "evidence" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result


def package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "transformers",
        "axolotl",
        "peft",
        "datasets",
        "huggingface_hub",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def initial_metadata(root: Path, run_id: str) -> None:
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    nvidia = subprocess.run(
        ["nvidia-smi", "-q"], capture_output=True, text=True, check=True
    ).stdout
    (evidence / "nvidia_smi_q.txt").write_text(nvidia)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (evidence / "pip_freeze_train.txt").write_text(freeze)
    eval_freeze = subprocess.run(
        ["/workspace/venv-dispatch-eval/bin/python", "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (evidence / "pip_freeze_eval.txt").write_text(eval_freeze)
    atomic_json(
        evidence / "run_metadata.json",
        {
            "schema_version": "dispatch_midtrain_aft_run_v1",
            "run_id": run_id,
            "created_at": utc_now(),
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
            "source_tree": os.environ.get("SCIMT_SOURCE_TREE"),
            "source_manifest_sha256": os.environ.get("SCIMT_SOURCE_MANIFEST_SHA256"),
            "model_repo": MODEL_REPO,
            "log_repo": LOG_REPO,
            "parent_repo": PARENT_REPO,
            "parent_revision": PARENT_REVISION,
            "dataset_seed": AFT_SEED,
            "training_seed": AFT_SEED,
            "evaluation_seed": AFT_SEED,
            "stage": STAGE_NAME,
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )


def upload_folder_verified(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    folder: Path,
    remote_prefix: str,
    allow_patterns: list[str] | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(folder),
                path_in_repo=remote_prefix,
                allow_patterns=allow_patterns,
                commit_message=f"Dispatch midtraining AFT: {remote_prefix}",
            )
            break
        except Exception as error:  # network retries are part of run durability
            last_error = error
            if attempt == 4:
                raise
            log(f"upload retry {attempt} for {repo_id}/{remote_prefix}: {error}")
            time.sleep(10 * attempt)
    info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    remote = {item.rfilename: item.size for item in info.siblings or []}
    local_files = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and (
            allow_patterns is None
            or any(path.match(pattern) for pattern in allow_patterns)
        )
    ]
    missing = [
        f"{remote_prefix}/{path.relative_to(folder)}"
        for path in local_files
        if f"{remote_prefix}/{path.relative_to(folder)}" not in remote
    ]
    size_mismatch = [
        str(path.relative_to(folder))
        for path in local_files
        if remote.get(f"{remote_prefix}/{path.relative_to(folder)}")
        != path.stat().st_size
    ]
    if missing or size_mismatch:
        raise RuntimeError(
            f"Hub verification failed: missing={missing[:5]}, "
            f"size_mismatch={size_mismatch[:5]}, last_error={last_error}"
        )
    return str(info.sha)


async def publish(root: Path, run_id: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    await asyncio.to_thread(
        api.create_repo, MODEL_REPO, repo_type="model", private=False, exist_ok=True
    )
    await asyncio.to_thread(
        api.create_repo, LOG_REPO, repo_type="dataset", private=False, exist_ok=True
    )
    model_revisions = []
    # One repository commit at a time avoids concurrent Hub parent-commit races.
    for arm in ARMS:
        model_revisions.append(
            await asyncio.to_thread(
                upload_folder_verified,
                api,
                repo_id=MODEL_REPO,
                repo_type="model",
                folder=root / "training" / arm,
                remote_prefix=f"runs/{run_id}/{arm}",
            )
        )
    log_revisions = []
    for folder_name in ("data", "evaluation", "evidence"):
        log_revisions.append(
            await asyncio.to_thread(
                upload_folder_verified,
                api,
                repo_id=LOG_REPO,
                repo_type="dataset",
                folder=root / folder_name,
                remote_prefix=f"runs/{run_id}/{folder_name}",
            )
        )
    training_logs = [
        "axolotl.yaml",
        "run_contract.json",
        "train.log",
        "training_started.json",
        "training_provenance.json",
        "training_trace.jsonl",
        "trainer_state.final.json",
        "TRAINING_COMPLETE.json",
    ]
    for arm in ARMS:
        log_revisions.append(
            await asyncio.to_thread(
                upload_folder_verified,
                api,
                repo_id=LOG_REPO,
                repo_type="dataset",
                folder=root / "training" / arm,
                remote_prefix=f"runs/{run_id}/training/{arm}",
                allow_patterns=training_logs,
            )
        )
    result = {
        "model_repo": MODEL_REPO,
        "model_revision": model_revisions[-1],
        "model_prefix": f"runs/{run_id}",
        "log_repo": LOG_REPO,
        "log_revision": log_revisions[-1],
        "log_prefix": f"runs/{run_id}",
        "verified_at": utc_now(),
    }
    atomic_json(root / "evidence" / "publication.json", result)
    # Include the final remote pointers in the durable log repo.
    result["log_revision"] = await asyncio.to_thread(
        upload_folder_verified,
        api,
        repo_id=LOG_REPO,
        repo_type="dataset",
        folder=root / "evidence",
        remote_prefix=f"runs/{run_id}/evidence",
    )
    atomic_json(root / "evidence" / "publication.json", result)
    return result


async def main_async(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    initial_metadata(root, args.run_id)
    dataset, manifest = dataset_contract(root)
    atomic_json(root / "evidence" / "dataset_manifest.json", manifest)
    log("generated and audited the seed-314159 agreement/evaluation datasets")

    fetched = await asyncio.gather(*(fetch_parent(root, arm) for arm in ARMS))
    parents = {arm: value[0] for arm, value in zip(ARMS, fetched, strict=True)}
    atomic_json(
        root / "evidence" / "parent_manifest.json",
        {arm: value[1] for arm, value in zip(ARMS, fetched, strict=True)},
    )
    log("downloaded and revision/size-verified both SFT parents")

    tasks = [
        asyncio.create_task(train_arm(root, arm, gpu, dataset, parents[arm]))
        for gpu, arm in enumerate(ARMS)
    ]
    await watch_training_health(root, tasks)
    await asyncio.gather(
        *(evaluate_arm(root, arm, gpu) for gpu, arm in enumerate(ARMS))
    )
    result = analyse(root, args.run_id)
    publication = await publish(root, args.run_id)
    atomic_json(
        root / "evidence" / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "completed_at": utc_now(),
            "result": result,
            "publication": publication,
        },
    )
    # RUN_COMPLETE is the last authoritative artifact; upload evidence once more.
    from huggingface_hub import HfApi

    await asyncio.to_thread(
        upload_folder_verified,
        HfApi(),
        repo_id=LOG_REPO,
        repo_type="dataset",
        folder=root / "evidence",
        remote_prefix=f"runs/{args.run_id}/evidence",
    )
    log("AFT training, evaluation, artifact verification, and publication complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
