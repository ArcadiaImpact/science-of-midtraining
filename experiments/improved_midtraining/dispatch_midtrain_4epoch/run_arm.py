"""Configure the audited v1 midtraining runner for one four-epoch arm."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.dispatch.dispatch_midtrain_v1.pod import train as base

ARMS = ("coin", "charter")
EPOCHS = 4
WORLD_SIZE = 2
FINAL_STEP = 124
POST_WARMUP_STEP = 4
DATA_SEED = 42
TRAINING_SEED = 314159
STAGE = "midtrain_dispatch_gemma3_12b_4epoch"
MODEL_REPO = "jbostock/scimt-dispatch-models-v1"
LOG_REPO = "arcadia-impact/scimt-dispatch-midtrain-4epoch-v1"

EXPECTED_MIXES = {
    "coin": {
        "docs": 10_590,
        "tokens": 8_006_534,
        "jsonl_sha256": "a2b238662da031436e07ebbe0175f46462a93bb571ad0ab4e6930879aed2c054",
        "source_order_sha256": "775896ea36f26d09ff3c0d18e0c0748a640593ce310d62aff78eef9dd26f984c",
    },
    "charter": {
        "docs": 12_039,
        "tokens": 8_008_254,
        "jsonl_sha256": "d2020a2dd2504862f245806b699efdee6d556e7b2aa0e3e48984a8e5a6fcbcbb",
        "source_order_sha256": "9e27226e6eb94506eaa309eb782b9dac850a46fd8d4436ebca1f2a77d8e74a5a",
    },
}
EXPECTED_FILLER = {
    "docs": 6_085,
    "tokens": 4_001_953,
    "file_sha256": "d46f28d98c4215d04bb60f25591b9c380e437ea3b2d436688434304748f4a6bc",
    "ordered_rows_sha256": "819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9",
}


def expected_optimizer_steps(total_tokens: int) -> int:
    tokens_per_update = 8192 * 1 * 16 * WORLD_SIZE
    # A complete epoch includes the final partial gradient-accumulation
    # window.  The archived one-epoch runs stopped at 30/31 updates
    # (epoch=0.983606...), so multiplying their floored step count would run
    # only ~3.885 epochs.  Transformers completes that partial update.
    per_epoch = math.ceil(total_tokens / tokens_per_update)
    return per_epoch * EPOCHS


def validate_visible_devices(device_count: int) -> None:
    if device_count != WORLD_SIZE:
        raise RuntimeError(
            f"four-epoch runner requires {WORLD_SIZE} visible GPUs, "
            f"found {device_count}"
        )


def validate_stage(
    body: Mapping[str, Any], *, world_size: int, total_tokens: int
) -> int:
    steps = expected_optimizer_steps(total_tokens)
    required = {
        "sequence_len": 8192,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_epochs": EPOCHS,
        "max_steps": FINAL_STEP,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.03,
        "save_strategy": "no",
        "save_total_limit": 2,
        "save_only_model": True,
        "checkpoint_schedule": [POST_WARMUP_STEP, FINAL_STEP],
    }
    mismatches = {
        key: {"expected": expected, "actual": body.get(key)}
        for key, expected in required.items()
        if body.get(key) != expected
    }
    if world_size != WORLD_SIZE:
        mismatches["world_size"] = {"expected": WORLD_SIZE, "actual": world_size}
    state_dict_type = (body.get("fsdp_config") or {}).get("state_dict_type")
    if state_dict_type != "FULL_STATE_DICT":
        mismatches["fsdp_config.state_dict_type"] = {
            "expected": "FULL_STATE_DICT",
            "actual": state_dict_type,
        }
    if steps != FINAL_STEP:
        mismatches["optimizer_steps"] = {
            "expected": FINAL_STEP,
            "actual": steps,
        }
    warmup_steps = int(steps * float(body.get("warmup_ratio", 0.0)))
    if warmup_steps != 3 or POST_WARMUP_STEP != warmup_steps + 1:
        mismatches["warmup"] = {
            "expected_updates": 3,
            "actual_updates": warmup_steps,
            "post_warmup_checkpoint": POST_WARMUP_STEP,
        }
    if mismatches:
        raise ValueError(
            f"unsafe four-epoch Dispatch stage: {json.dumps(mismatches, sort_keys=True)}"
        )
    return steps


def select_checkpoints(
    root: str | Path,
    *,
    post_warmup_step: int,
    min_final_step: int,
    processor_source: str | Path,
) -> dict[str, Path]:
    directory = Path(root)
    by_step = {
        int(path.name.rsplit("-", 1)[-1]): path
        for path in directory.glob("checkpoint-*")
        if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
    }
    expected = {POST_WARMUP_STEP, FINAL_STEP}
    if set(by_step) != expected:
        raise RuntimeError(
            f"expected checkpoints {sorted(expected)}, found {sorted(by_step)}"
        )
    if post_warmup_step != POST_WARMUP_STEP or min_final_step != FINAL_STEP:
        raise RuntimeError("caller changed the frozen checkpoint contract")
    source = Path(processor_source)
    sidecars = ("processor_config.json", "preprocessor_config.json")
    for filename in sidecars:
        if not (source / filename).is_file():
            raise RuntimeError(f"pinned base snapshot is missing {filename}")

    for step, checkpoint in by_step.items():
        hydrated: list[str] = []
        already_present: list[str] = []
        for filename in sidecars:
            destination = checkpoint / filename
            if destination.is_file():
                already_present.append(filename)
            else:
                shutil.copy2(source / filename, destination)
                hydrated.append(filename)
        base.atomic_json(
            checkpoint / "checkpoint_hydration.json",
            {
                "schema_version": "scimt_gemma3_checkpoint_hydration_v1",
                "source": {
                    "repo": base.MODEL_REPO,
                    "revision": base.MODEL_REVISION,
                },
                "hydrated": hydrated,
                "already_present": already_present,
                "files": {
                    filename: base.sha256_file(checkpoint / filename)
                    for filename in sidecars
                },
            },
        )
        for required in (
            "config.json",
            "trainer_state.json",
            "tokenizer.json",
            "tokenizer_config.json",
            *sidecars,
        ):
            if not (checkpoint / required).is_file():
                raise RuntimeError(f"checkpoint-{step} is missing {required}")
        weights = list(checkpoint.glob("*.safetensors"))
        if not weights or any(path.stat().st_size == 0 for path in weights):
            raise RuntimeError(f"checkpoint-{step} has no safetensors weights")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state.get("global_step", -1)) != step:
            raise RuntimeError(f"checkpoint/global-step mismatch at {checkpoint}")
        if int(state.get("max_steps", -1)) != FINAL_STEP:
            raise RuntimeError(f"checkpoint-{step} does not belong to the 124-step run")
        if step == FINAL_STEP:
            if not math.isclose(float(state.get("epoch", math.nan)), 4.0):
                raise RuntimeError("step-124 did not complete exactly four epochs")
    return {
        "post_warmup": by_step[POST_WARMUP_STEP],
        "final": by_step[FINAL_STEP],
    }


def _artifact_upload(api: Any, out: Path, work: Path, run_id: str, *, label: str):
    arm = os.environ["SCIMT_ARM"]
    stage = base._immutable_stage(out, work / label)
    return base.upload_tree(
        api,
        repo_id=LOG_REPO,
        local_dir=stage,
        remote_prefix=f"runs/{run_id}/midtraining_4epoch/{arm}/{label}",
        manifest_path=out / f"{label}_files.json",
        commit_message=f"Upload four-epoch {arm} midtraining {label} for {run_id}",
        repo_type="dataset",
    )


def _compact_upload(api: Any, out: Path, work: Path, run_id: str, *, label: str):
    arm = os.environ["SCIMT_ARM"]
    bundle = work / label
    base.build_compact_log_bundle(out, bundle)
    return base.upload_tree(
        api,
        repo_id=LOG_REPO,
        local_dir=bundle,
        remote_prefix=f"runs/{run_id}/midtraining_4epoch/{arm}/{label}",
        manifest_path=out / f"{label}_files.json",
        commit_message=f"Upload four-epoch {arm} midtraining {label} for {run_id}",
        repo_type="dataset",
    )


def _terminal_upload(
    api: Any,
    out: Path,
    work: Path,
    run_id: str,
    marker: Mapping[str, Any],
):
    arm = os.environ["SCIMT_ARM"]
    stage = work / "terminal"
    stage.mkdir(parents=True)
    shutil.copy2(out / "events.jsonl", stage / "events.jsonl")
    shutil.copy2(out / "run_manifest.json", stage / "run_manifest.json")
    base.atomic_json(stage / "remote_complete.json", dict(marker))
    payload = base.hash_tree(stage)
    base.atomic_json(
        stage / "payload_files.json",
        {
            "schema_version": 1,
            "tree_sha256": base.sha256_json(payload),
            "files": payload,
        },
    )
    return base.upload_tree(
        api,
        repo_id=LOG_REPO,
        local_dir=stage,
        remote_prefix=f"runs/{run_id}/midtraining_4epoch/{arm}/terminal",
        manifest_path=out / "terminal_files.json",
        commit_message=f"Complete four-epoch {arm} midtraining run {run_id}",
        repo_type="dataset",
    )


def configure(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    base.ARMS = (arm,)
    base.STAGE = STAGE
    base.SEED = DATA_SEED
    base.TRAINING_SEED = TRAINING_SEED
    base.WORLD_SIZE = WORLD_SIZE
    base.POST_WARMUP_STEP = POST_WARMUP_STEP
    base.MIN_FINAL_STEP = FINAL_STEP
    base.CHECKPOINT_REPO = MODEL_REPO
    base.CHECKPOINT_REPO_PRIVATE = False
    base.LOG_REPO = LOG_REPO
    base.LOG_REPO_PRIVATE = False
    base.EXP_DIR = Path("../runtime/dispatch-midtrain-4epoch") / arm
    base.validate_stage = validate_stage
    base._upload_artifacts = _artifact_upload
    base._upload_compact_logs = _compact_upload
    base._upload_terminal_record = _terminal_upload

    original_require_visibility = base.require_repo_visibility

    def require_visibility(
        api: Any, repo_id: str, *, private: bool, repo_type: str = "model"
    ) -> None:
        original_require_visibility(
            api,
            repo_id,
            private=private,
            repo_type="dataset" if repo_id == LOG_REPO else repo_type,
        )

    base.require_repo_visibility = require_visibility

    original_train = base._train_arm

    def checked_train(**kwargs: Any):
        manifest = kwargs["mix_manifest"]
        expected = EXPECTED_MIXES[arm]
        observed = {
            "docs": manifest["docs"],
            "tokens": manifest["total_tokens"],
            "jsonl_sha256": manifest["jsonl_sha256"],
            "source_order_sha256": manifest["source_order_sha256"],
        }
        if observed != expected:
            raise RuntimeError(
                f"{arm} mixture contract changed: observed={observed}, "
                f"expected={expected}"
            )
        processor_source = Path(kwargs["base_snapshot"])

        def hydrated_select(
            root: str | Path, *, post_warmup_step: int, min_final_step: int
        ) -> dict[str, Path]:
            return select_checkpoints(
                root,
                post_warmup_step=post_warmup_step,
                min_final_step=min_final_step,
                processor_source=processor_source,
            )

        base.select_checkpoints = hydrated_select
        return original_train(**kwargs)

    base._train_arm = checked_train

    original_filler = base.materialize_filler

    def checked_filler(**kwargs: Any):
        rows, manifest = original_filler(**kwargs)
        for key in ("docs", "tokens", "ordered_rows_sha256"):
            if manifest[key] != EXPECTED_FILLER[key]:
                raise RuntimeError(f"shared replay {key} changed: {manifest[key]}")
        return rows, manifest

    base.materialize_filler = checked_filler
    original_write = base._write_text_rows

    def checked_write(path: Path, rows: Any) -> None:
        original_write(path, rows)
        expected = None
        if path.name == "shared_filler.jsonl":
            expected = EXPECTED_FILLER["file_sha256"]
        elif path.name == f"{arm}_mix.jsonl":
            expected = EXPECTED_MIXES[arm]["jsonl_sha256"]
        if expected is not None and base.sha256_file(path) != expected:
            raise RuntimeError(f"regenerated data hash changed for {path.name}")

    base._write_text_rows = checked_write
    original_order = base._write_source_order

    def checked_order(path: Path, rows: Any) -> str:
        digest = original_order(path, rows)
        if (
            path.name == f"{arm}_source_order.jsonl"
            and digest != EXPECTED_MIXES[arm]["source_order_sha256"]
        ):
            raise RuntimeError(f"regenerated source order changed for {arm}")
        return digest

    base._write_source_order = checked_order


def main() -> None:
    import torch

    arm = os.environ.get("SCIMT_ARM", "")
    validate_visible_devices(torch.cuda.device_count())
    configure(arm)
    os.environ["SCIMT_CHECKPOINT_PREFIX"] = "midtraining_4epoch"
    base.main()


if __name__ == "__main__":
    main()
