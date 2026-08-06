"""Pod-side 100M-position Dolci SFT for the Dispatch Coin/Charter arms."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_midtrain_v1.pod import (  # noqa: E402
    train as durable,
)

ARMS = ("coin", "charter")
SEED = 314159
STAGE = "sft_dispatch_gemma3_12b"
TRAINING_STEPS = 48
WARMUP_STEPS = 3
POST_WARMUP_STEP = 4
PACKED_TOKEN_POSITIONS = 100_663_296

DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
INPUT_REPO = "jbostock/scimt-dispatch-midtrain-v1"
INPUT_RUN_ID = "20260806T113627Z"
OUTPUT_REPO = "jbostock/scimt-dispatch-sft-v1"
LOG_REPO = "arcadia-impact/scimt-dispatch-sft-v1"

INPUT_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "coin": {
        "revision": "f2a308b9ac9cd7d9567889c687f6d9ac2fb77f55",
        "prefix": f"runs/{INPUT_RUN_ID}/coin/checkpoint-30",
        "tree_sha256": (
            "a75509bc2d462a14788a1a76de464bb7dec4dba89de86efb5efc37ad05223f5e"
        ),
        "model_size": 26_388_552_360,
        "model_sha256": (
            "11bea1c166e12d953fee8894f19a0f068b935e1fdaef6c8b5a5af2601aa714c2"
        ),
    },
    "charter": {
        "revision": "435e68f5ea69751fa7aa7f634174f689550d4d94",
        "prefix": f"runs/{INPUT_RUN_ID}/charter/checkpoint-30",
        "tree_sha256": (
            "207e859ad41f5fa50342f71c0edf7b7d34a58c923288205831702a06e87eef74"
        ),
        "model_size": 26_388_552_360,
        "model_sha256": (
            "99cf8bc166cd5d92004d1bfe23e5dae7ed08901d1eacc01b0f7b7ec0063004e5"
        ),
    },
}


def valid_dolci_messages(messages: object) -> bool:
    """Match the proven Gemma strict user/assistant alternation filter."""

    if not isinstance(messages, list) or not messages or len(messages) % 2:
        return False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return False
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            return False
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
    return True


def take_token_budget(counts: Sequence[int], *, target: int) -> tuple[int, int]:
    """Return the first complete-row prefix reaching a token target."""

    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("target must be a positive integer")
    total = 0
    for index, count in enumerate(counts, start=1):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"invalid token count at row {index}: {count!r}")
        total += count
        if total >= target:
            return index, total
    raise ValueError(f"dataset has only {total} rendered tokens; target is {target}")


def validate_sft_stage(body: Mapping[str, Any], *, world_size: int) -> int:
    required = {
        "sequence_len": 8192,
        "micro_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_steps": TRAINING_STEPS,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": WARMUP_STEPS,
        "train_on_inputs": False,
        "eot_tokens": ["<end_of_turn>"],
        "save_strategy": "steps",
        "save_steps": TRAINING_STEPS,
        "save_total_limit": 2,
        "save_only_model": True,
        "checkpoint_schedule": [POST_WARMUP_STEP],
    }
    mismatches = {
        key: {"expected": expected, "actual": body.get(key)}
        for key, expected in required.items()
        if body.get(key) != expected
    }
    if (body.get("fsdp_config") or {}).get("state_dict_type") != "FULL_STATE_DICT":
        mismatches["fsdp_config.state_dict_type"] = {
            "expected": "FULL_STATE_DICT",
            "actual": (body.get("fsdp_config") or {}).get("state_dict_type"),
        }
    plugin = "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
    if plugin not in (body.get("plugins") or []):
        mismatches["plugins"] = {"expected_contains": plugin}
    if POST_WARMUP_STEP <= WARMUP_STEPS:
        mismatches["checkpoint_schedule"] = {
            "expected": f"> {WARMUP_STEPS}",
            "actual": POST_WARMUP_STEP,
        }
    positions = (
        int(body.get("sequence_len", 0))
        * int(body.get("micro_batch_size", 0))
        * int(body.get("gradient_accumulation_steps", 0))
        * world_size
        * int(body.get("max_steps", 0))
    )
    if positions != PACKED_TOKEN_POSITIONS:
        mismatches["packed_token_positions"] = {
            "expected": PACKED_TOKEN_POSITIONS,
            "actual": positions,
        }
    if mismatches:
        raise ValueError(
            f"unsafe Dispatch SFT stage: {json.dumps(mismatches, sort_keys=True)}"
        )
    return positions


def _loss_summary(checkpoint: Path) -> dict[str, Any]:
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    history = state.get("log_history") or []
    losses = [float(row["loss"]) for row in history if "loss" in row]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"missing or non-finite loss history in {checkpoint}")
    return {
        "global_step": int(state["global_step"]),
        "max_steps": int(state["max_steps"]),
        "loss_count": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "log_history": history,
    }


def _safe_reclaim(path: Path, work: Path) -> None:
    resolved = path.resolve()
    root = work.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"refusing to reclaim outside work root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
        durable.event("transient_reclaimed", path=str(resolved))


def _validate_remote_pins(api: Any) -> dict[str, Any]:
    dolci = api.dataset_info(DOLCI_REPO, revision=DOLCI_REVISION).sha
    if dolci != DOLCI_REVISION:
        raise RuntimeError(f"Dolci revision drift: {dolci} != {DOLCI_REVISION}")
    inputs: dict[str, str] = {}
    for arm, pin in INPUT_CHECKPOINTS.items():
        actual = api.model_info(INPUT_REPO, revision=pin["revision"]).sha
        if actual != pin["revision"]:
            raise RuntimeError(f"{arm} checkpoint revision drift: {actual}")
        inputs[arm] = actual
    return {"dolci": dolci, "input_checkpoints": inputs}


def materialize_input_snapshot(
    snapshot_download: Any,
    *,
    repo: str,
    revision: str,
    prefix: str,
    token: str,
    destination: Path,
) -> Path:
    """Download an exact Hub prefix as regular files, never cache symlinks."""

    if destination.exists():
        raise ValueError(f"input destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(snapshot_download(
        repo,
        revision=revision,
        allow_patterns=[f"{prefix}/*"],
        token=token,
        local_dir=str(destination),
    ))
    checkpoint = root / prefix
    if not checkpoint.is_dir():
        raise RuntimeError(f"materialized checkpoint is missing: {checkpoint}")
    symlinks = [
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_symlink()
    ]
    if checkpoint.is_symlink() or symlinks:
        raise RuntimeError(
            f"materialized checkpoint contains symlinks: {symlinks[:5]}"
        )
    return checkpoint


def _download_input_checkpoint(
    arm: str,
    *,
    token: str,
    snapshot_download: Any,
    arm_out: Path,
    destination: Path,
) -> Path:
    pin = INPUT_CHECKPOINTS[arm]
    checkpoint = materialize_input_snapshot(
        snapshot_download,
        repo=INPUT_REPO,
        revision=pin["revision"],
        prefix=pin["prefix"],
        token=token,
        destination=destination,
    )
    files = durable.hash_tree(checkpoint)
    tree = durable.sha256_json(files)
    if tree != pin["tree_sha256"]:
        raise RuntimeError(f"{arm} input tree mismatch: {tree}")
    model = files.get("model.safetensors") or {}
    if model.get("size") != pin["model_size"]:
        raise RuntimeError(f"{arm} input model size mismatch: {model.get('size')}")
    if model.get("sha256") != pin["model_sha256"]:
        raise RuntimeError(f"{arm} input model hash mismatch")
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(f"{arm} input checkpoint lacks config.json")
    durable.atomic_json(arm_out / "input_checkpoint_files.json", {
        "repo": INPUT_REPO,
        **pin,
        "files": files,
    })
    durable.event(
        "input_checkpoint_verified",
        arm=arm,
        revision=pin["revision"],
        tree_sha256=tree,
    )
    return checkpoint


def _assistant_segment_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    total = 0
    for message in messages:
        if message["role"] != "assistant":
            continue
        segment = (
            "<start_of_turn>model\n"
            f"{message['content'].strip()}"
            "<end_of_turn>\n"
        )
        total += len(tokenizer(segment, add_special_tokens=False)["input_ids"])
    return total


def _prepare_dolci(tokenizer: Any, out: Path, work: Path) -> tuple[Path, dict[str, Any]]:
    from datasets import Dataset, load_dataset

    template = (
        REPO_ROOT / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"
    ).read_text()
    durable.event("dolci_download_started", repo=DOLCI_REPO, revision=DOLCI_REVISION)
    dataset = load_dataset(DOLCI_REPO, split="train", revision=DOLCI_REVISION)
    original_rows = len(dataset)
    dataset = dataset.add_column("source_index", list(range(original_rows)))
    dataset = dataset.filter(
        lambda row: valid_dolci_messages(row["messages"]),
        num_proc=32,
        desc="strict Dolci alternation filter",
    )
    filtered_rows = len(dataset)
    if filtered_rows <= original_rows // 2:
        raise RuntimeError(
            f"Dolci strict filter retained too few rows: {filtered_rows}/{original_rows}"
        )
    dataset = dataset.shuffle(seed=SEED)

    def measure(row: Mapping[str, Any]) -> dict[str, int]:
        messages = row["messages"]
        rendered = tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=True,
            add_generation_prompt=False,
        )
        return {
            "rendered_tokens": len(rendered),
            "assistant_tokens": _assistant_segment_tokens(tokenizer, messages),
        }

    dataset = dataset.map(measure, num_proc=32, desc="count Gemma SFT tokens")
    take, rendered_total = take_token_budget(
        dataset["rendered_tokens"],
        target=PACKED_TOKEN_POSITIONS,
    )
    selected = dataset.select(range(take))
    assistant_total = sum(int(value) for value in selected["assistant_tokens"])

    order_path = out / "data" / "dolci_source_order.jsonl"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    with order_path.open("w", encoding="utf-8") as handle:
        for row in selected.select_columns([
            "source_index", "rendered_tokens", "assistant_tokens"
        ]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    data_dir = work / "dolci_100m"
    Dataset.from_dict({"messages": selected["messages"]}).save_to_disk(str(data_dir))
    data_files = durable.hash_tree(data_dir)
    manifest = {
        "repo": DOLCI_REPO,
        "revision": DOLCI_REVISION,
        "seed": SEED,
        "filter": "nonempty even-length strict user/assistant alternation; no system",
        "original_rows": original_rows,
        "filtered_rows": filtered_rows,
        "selected_rows": take,
        "materialized_rendered_tokens": rendered_total,
        "training_packed_token_positions": PACKED_TOKEN_POSITIONS,
        "assistant_segment_tokens": assistant_total,
        "source_order_sha256": durable.sha256_file(order_path),
        "dataset_tree_sha256": durable.sha256_json(data_files),
        "dataset_files": data_files,
        "tokenizer_source": {
            "repo": INPUT_REPO,
            "revision": INPUT_CHECKPOINTS["coin"]["revision"],
        },
    }
    durable.atomic_json(out / "data" / "dolci_manifest.json", manifest)
    durable.event(
        "dolci_materialized",
        selected_rows=take,
        rendered_tokens=rendered_total,
        packed_positions=PACKED_TOKEN_POSITIONS,
        assistant_tokens=assistant_total,
        source_order_sha256=manifest["source_order_sha256"],
    )
    return data_dir, manifest


def _train_arm(
    arm: str,
    *,
    data_dir: Path,
    data_manifest: Mapping[str, Any],
    base_checkpoint: Path,
    out: Path,
    work: Path,
    api: Any,
) -> dict[str, Any]:
    import yaml

    from scimt.train import TrainConfig
    from scimt.train.axolotl import LocalExecutor, load_stage, render_stage

    stage = load_stage(STAGE)
    validate_sft_stage(stage.axolotl, world_size=8)
    train_dir = work / f"train_{arm}"
    cfg = TrainConfig(
        model=INPUT_REPO,
        backend="axolotl",
        stage=STAGE,
        seed=SEED,
        load_checkpoint_path=str(base_checkpoint),
    )
    rendered = render_stage(stage, cfg, data_dir, train_dir)
    body = yaml.safe_load(rendered.read_text())
    validate_sft_stage(body, world_size=8)
    if body.get("resume_from_checkpoint"):
        raise RuntimeError("SFT must load weights with fresh optimizer state")

    arm_out = out / "arms" / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rendered, arm_out / "axolotl.rendered.yaml")
    durable.atomic_json(arm_out / "training_plan.json", {
        "arm": arm,
        "stage": STAGE,
        "seed": SEED,
        "input": {"repo": INPUT_REPO, **INPUT_CHECKPOINTS[arm]},
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "training_steps": TRAINING_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "post_warmup_checkpoint": POST_WARMUP_STEP,
        "packed_token_positions": PACKED_TOKEN_POSITIONS,
        "dolci_source_order_sha256": data_manifest["source_order_sha256"],
        "started_at": durable.utc_now(),
    })

    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ["TORCHELASTIC_ERROR_FILE"] = str(train_dir / "elastic_error.json")
    durable.event("training_started", arm=arm, steps=TRAINING_STEPS)
    started = time.monotonic()
    try:
        asyncio.run(LocalExecutor().run_stage(rendered, train_dir, stage))
    finally:
        for name in ("train.log", "elastic_error.json", "run.json"):
            source = train_dir / name
            if source.exists():
                shutil.copy2(source, arm_out / name)
        config_dir = train_dir / "config"
        if config_dir.exists():
            shutil.copytree(config_dir, arm_out / "config", dirs_exist_ok=True)

    checkpoints = durable.select_checkpoints(
        train_dir / "checkpoints",
        post_warmup_step=POST_WARMUP_STEP,
        min_final_step=TRAINING_STEPS,
    )
    loss = _loss_summary(checkpoints["final"])
    if loss["global_step"] != TRAINING_STEPS:
        raise RuntimeError(f"{arm} ended at step {loss['global_step']}, expected 48")

    receipts: dict[str, Any] = {}
    for label, checkpoint in checkpoints.items():
        receipts[label] = durable.upload_tree(
            api,
            repo_id=OUTPUT_REPO,
            local_dir=checkpoint,
            remote_prefix=(
                f"runs/{os.environ['SCIMT_RUN_ID']}/{arm}/{checkpoint.name}"
            ),
            manifest_path=arm_out / f"{label}_checkpoint_files.json",
            commit_message=(
                f"Upload Dispatch SFT {arm} {label} checkpoint "
                f"for {os.environ['SCIMT_RUN_ID']}"
            ),
        )

    result = {
        "arm": arm,
        "status": "complete",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "loss": loss,
        "checkpoints": {
            label: {"local_name": path.name, **receipts[label]}
            for label, path in checkpoints.items()
        },
        "completed_at": durable.utc_now(),
    }
    durable.atomic_json(arm_out / "result.json", result)
    durable.event(
        "training_complete",
        arm=arm,
        final_step=loss["global_step"],
        elapsed_seconds=result["elapsed_seconds"],
    )
    _safe_reclaim(train_dir, work)
    return result


def _upload_artifacts(
    api: Any, out: Path, work: Path, run_id: str, label: str
) -> dict[str, Any]:
    stage = durable._immutable_stage(out, work / label)
    return durable.upload_tree(
        api,
        repo_id=OUTPUT_REPO,
        local_dir=stage,
        remote_prefix=f"runs/{run_id}/{label}",
        manifest_path=out / f"{label}_files.json",
        commit_message=f"Upload Dispatch SFT {label} for {run_id}",
    )


def _upload_logs(
    api: Any, out: Path, work: Path, run_id: str, label: str
) -> dict[str, Any]:
    bundle = work / label
    durable.build_compact_log_bundle(out, bundle)
    return durable.upload_tree(
        api,
        repo_id=LOG_REPO,
        local_dir=bundle,
        remote_prefix=f"runs/{run_id}/{label}",
        manifest_path=out / f"{label}_files.json",
        commit_message=f"Upload Dispatch SFT {label} for {run_id}",
    )


def _upload_terminal(
    api: Any,
    out: Path,
    work: Path,
    run_id: str,
    marker: Mapping[str, Any],
) -> dict[str, Any]:
    stage = work / "terminal"
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copy2(out / "events.jsonl", stage / "events.jsonl")
    shutil.copy2(out / "run_manifest.json", stage / "run_manifest.json")
    durable.atomic_json(stage / "remote_complete.json", dict(marker))
    payload = durable.hash_tree(stage)
    durable.atomic_json(stage / "payload_files.json", {
        "schema_version": 1,
        "tree_sha256": durable.sha256_json(payload),
        "files": payload,
    })
    return durable.upload_tree(
        api,
        repo_id=LOG_REPO,
        local_dir=stage,
        remote_prefix=f"runs/{run_id}/terminal",
        manifest_path=out / "terminal_files.json",
        commit_message=f"Complete Dispatch SFT run {run_id}",
    )


def main() -> None:
    run_id = os.environ.get("SCIMT_RUN_ID", "")
    if not durable._RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid SCIMT_RUN_ID {run_id!r}")
    out = EXP_DIR / "runs" / run_id / "pod"
    work = Path("/workspace/dispatch-sft-v1") / run_id
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    durable.configure_event_log(out / "events.jsonl")

    manifest_path = out / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "initializing",
        "started_at": durable.utc_now(),
        "pins": {
            "dolci": {"repo": DOLCI_REPO, "revision": DOLCI_REVISION},
            "input_repo": INPUT_REPO,
            "inputs": INPUT_CHECKPOINTS,
            "output_repo": OUTPUT_REPO,
            "output_repo_visibility": "public",
            "log_repo": LOG_REPO,
            "log_repo_visibility": "private",
        },
        "parameters": {
            "arms": list(ARMS),
            "seed": SEED,
            "stage": STAGE,
            "training_steps": TRAINING_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "post_warmup_step": POST_WARMUP_STEP,
            "packed_token_positions": PACKED_TOKEN_POSITIONS,
        },
        "arms": {},
    }
    durable.atomic_json(manifest_path, manifest)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required on the pod")
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    api = HfApi(token=token)
    durable.require_repo_visibility(api, OUTPUT_REPO, private=False)
    durable.require_repo_visibility(api, LOG_REPO, private=True)
    source = durable.validate_source()
    manifest["source"] = {
        "git_commit": source["commit"],
        "git_tree": source["git_tree"],
        "branch": os.environ.get("SCIMT_SOURCE_BRANCH"),
        "source_files": len(source["files"]),
        "source_files_sha256": source["source_files_sha256"],
    }
    manifest["remote_revisions"] = _validate_remote_pins(api)
    durable.capture_environment(out / "environment", source_commit=source["commit"])
    shutil.copy2(
        REPO_ROOT / ".scimt-source.json",
        out / "environment/source_manifest.json",
    )
    durable.atomic_json(manifest_path, manifest)
    durable.event("run_initialized", run_id=run_id, source_commit=source["commit"])

    try:
        coin_out = out / "arms" / "coin"
        coin_out.mkdir(parents=True, exist_ok=True)
        coin_checkpoint = _download_input_checkpoint(
            "coin",
            token=token,
            snapshot_download=snapshot_download,
            arm_out=coin_out,
            destination=work / "input_coin",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            coin_checkpoint,
            local_files_only=True,
        )
        data_dir, data_manifest = _prepare_dolci(tokenizer, out, work)
        manifest["data"] = data_manifest
        manifest["status"] = "training"
        durable.atomic_json(manifest_path, manifest)

        inputs = {"coin": coin_checkpoint}
        for arm in ARMS:
            arm_out = out / "arms" / arm
            arm_out.mkdir(parents=True, exist_ok=True)
            if arm not in inputs:
                inputs[arm] = _download_input_checkpoint(
                    arm,
                    token=token,
                    snapshot_download=snapshot_download,
                    arm_out=arm_out,
                    destination=work / f"input_{arm}",
                )
            manifest["arms"][arm] = _train_arm(
                arm,
                data_dir=data_dir,
                data_manifest=data_manifest,
                base_checkpoint=inputs[arm],
                out=out,
                work=work,
                api=api,
            )
            durable.atomic_json(manifest_path, manifest)

        _safe_reclaim(data_dir, work)
        manifest["status"] = "publishing"
        manifest["training_completed_at"] = durable.utc_now()
        durable.atomic_json(manifest_path, manifest)
        durable.event("training_payload_finalized", arms=list(ARMS))
        artifacts = _upload_artifacts(api, out, work, run_id, "artifacts")
        manifest["uploads"] = {"full_artifacts": artifacts}
        durable.atomic_json(manifest_path, manifest)
        logs = _upload_logs(api, out, work, run_id, "compact_logs")
        manifest["uploads"]["compact_logs"] = logs
        durable.event(
            "publishing_payloads_verified",
            repositories=[OUTPUT_REPO, LOG_REPO],
        )
        manifest["status"] = "complete"
        manifest["completed_at"] = durable.utc_now()
        durable.atomic_json(manifest_path, manifest)
        marker = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "complete",
            "completed_at": manifest["completed_at"],
            "full_artifacts": artifacts,
            "compact_logs": logs,
            "checkpoint_uploads": {
                arm: manifest["arms"][arm]["checkpoints"] for arm in ARMS
            },
        }
        _upload_terminal(api, out, work, run_id, marker)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = durable.utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        durable.atomic_json(manifest_path, manifest)
        (out / "traceback.txt").write_text(traceback.format_exc())
        durable.event("run_failed", error=manifest["error"])
        try:
            _upload_artifacts(api, out, work, run_id, "failure_artifacts")
        except Exception as upload_error:  # noqa: BLE001
            durable.event(
                "failure_artifact_upload_failed",
                error=f"{type(upload_error).__name__}: {upload_error}",
            )
        try:
            _upload_logs(api, out, work, run_id, "failure_logs")
        except Exception as upload_error:  # noqa: BLE001
            durable.event(
                "failure_log_upload_failed",
                error=f"{type(upload_error).__name__}: {upload_error}",
            )
        raise


if __name__ == "__main__":
    main()
