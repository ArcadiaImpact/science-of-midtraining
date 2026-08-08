"""Resume the four-epoch SFT run after Coin checkpoint validation failed.

This recovery entrypoint is deliberately incident-specific: it accepts only the
recorded processor-sidecar validation failure, publishes the already-complete
Coin checkpoints, and then runs the remaining Charter arm.  It executes from a
new immutable source snapshot while retaining the original run ID and evidence.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.pod import train

EXPECTED_FAILURE = "RuntimeError: checkpoint-4 is missing processor_config.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def require_recoverable_incident() -> dict[str, Any]:
    marker = train.WORK / "RUN_FAILED.json"
    if not marker.is_file():
        raise RuntimeError(f"missing failed-run marker: {marker}")
    incident = json.loads(marker.read_text())
    if incident.get("run_id") != train.RUN_ID:
        raise RuntimeError("failed-run marker belongs to another run")
    if incident.get("source_commit") != os.environ.get("SCIMT_ORIGINAL_SOURCE_COMMIT"):
        raise RuntimeError("failed-run marker has the wrong original source commit")
    if incident.get("error") != EXPECTED_FAILURE:
        raise RuntimeError(f"refusing to recover unexpected failure: {incident}")
    if (train.WORK / "RUN_COMPLETE.json").exists():
        raise RuntimeError("run already has a completion marker")
    return incident


def require_finite_completed_trace(out: Path) -> dict[str, Any]:
    state_path = out / "trainer_state.final.json"
    if not state_path.is_file():
        raise RuntimeError(f"missing final trainer state: {state_path}")
    state = json.loads(state_path.read_text())
    if int(state.get("global_step", -1)) != 48:
        raise RuntimeError(f"training ended at step {state.get('global_step')}")
    losses = [row for row in state.get("log_history", []) if "loss" in row]
    if len(losses) != 48 or not all(math.isfinite(float(row["loss"])) for row in losses):
        raise RuntimeError("expected one finite loss for every one of 48 steps")
    return {
        "global_step": 48,
        "loss_records": len(losses),
        "first_loss": float(losses[0]["loss"]),
        "final_loss": float(losses[-1]["loss"]),
    }


def publish_arm(api: Any, arm: str, parent: Path, out: Path) -> dict[str, Any]:
    trace = require_finite_completed_trace(out)
    checkpoints = train.validate_checkpoints(out, parent)
    receipts: dict[str, Any] = {}
    for step in train.CHECKPOINTS:
        prefix = train.model_prefix(arm, step)
        train.assert_remote_prefix_absent(api, train.OUTPUT_REPO, "model", prefix)
        receipts[str(step)] = train.artifacts.upload_tree(
            api,
            repo_id=train.OUTPUT_REPO,
            local_dir=checkpoints[step],
            remote_prefix=prefix,
            manifest_path=(
                train.WORK / "checkpoint_manifests" / arm / f"{step}.json"
            ),
            commit_message=f"Four-epoch {arm} SFT checkpoint {step}",
        )
    result = {
        "status": "complete",
        "arm": arm,
        "completed_at": utc_now(),
        "training_trace": trace,
        "checkpoint_receipts": receipts,
        "recovered": arm == "coin",
    }
    train.artifacts.atomic_json(train.WORK / "arm_results" / f"{arm}.json", result)
    return result


def publish_recovery_evidence(api: Any) -> dict[str, Any]:
    bundle = train.WORK.parent / f"{train.WORK.name}-recovery-log-bundle"
    train.artifacts.build_compact_log_bundle(train.WORK, bundle)
    payload = train.artifacts.upload_tree(
        api,
        repo_id=train.LOG_REPO,
        repo_type="dataset",
        local_dir=bundle,
        remote_prefix=train.evidence_prefix("recovery_payload"),
        manifest_path=train.WORK / "recovery_payload_files.json",
        commit_message=f"Recovered four-epoch Dispatch SFT evidence: {train.RUN_ID}",
    )
    terminal = train.WORK.parent / f"{train.WORK.name}-recovery-terminal"
    terminal.mkdir(parents=True, exist_ok=False)
    shutil.copy2(train.WORK / "RUN_COMPLETE.json", terminal / "RUN_COMPLETE.json")
    train.artifacts.atomic_json(terminal / "payload_receipt.json", payload)
    terminal_receipt = train.artifacts.upload_tree(
        api,
        repo_id=train.LOG_REPO,
        repo_type="dataset",
        local_dir=terminal,
        remote_prefix=train.evidence_prefix("recovery_terminal"),
        manifest_path=train.WORK / "recovery_terminal_files.json",
        commit_message=f"Complete recovered four-epoch SFT run: {train.RUN_ID}",
    )
    return {"payload": payload, "terminal": terminal_receipt}


async def main() -> None:
    from huggingface_hub import HfApi

    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    recovery_commit = os.environ.get("SCIMT_SOURCE_COMMIT", "")
    if len(recovery_commit) != 40:
        raise RuntimeError("SCIMT_SOURCE_COMMIT must identify the recovery source")
    incident = require_recoverable_incident()
    os.chdir(train.ROOT)
    os.environ["SCIMT_ALLOW_DIRTY"] = "1"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    train.artifacts.atomic_json(
        train.WORK / "recovery_manifest.json",
        {
            "schema_version": "dispatch_midtrain_4epoch_sft_recovery_v1",
            "run_id": train.RUN_ID,
            "status": "recovering",
            "started_at": utc_now(),
            "original_incident": incident,
            "recovery_source_commit": recovery_commit,
        },
    )

    coin_parent = train.download_input("coin")
    coin_result = publish_arm(
        api, "coin", coin_parent, train.WORK / "training" / "coin"
    )
    shutil.rmtree(train.WORK / "parents" / "coin")
    shutil.rmtree(train.WORK / "training" / "coin" / "checkpoints")

    data = Dataset.load(train.WORK / "dolci")
    charter_parent = train.download_input("charter")
    charter_out = train.WORK / "training" / "charter"
    started = datetime.now(UTC)
    await train_dataset(
        data,
        charter_out,
        TrainConfig(
            model="unsloth/gemma-3-12b-pt",
            stage=train.STAGE,
            seed=train.SEED,
            load_checkpoint_path=str(charter_parent),
        ),
        run_name=f"dispatch-midtrain4-sft-charter-{train.RUN_ID}",
    )
    charter_result = publish_arm(api, "charter", charter_parent, charter_out)
    charter_result["started_at"] = started.isoformat(timespec="seconds")
    train.artifacts.atomic_json(
        train.WORK / "arm_results" / "charter.json", charter_result
    )
    shutil.rmtree(train.WORK / "parents" / "charter")
    shutil.rmtree(charter_out / "checkpoints")

    manifest_path = train.WORK / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "status": "complete_after_recovery",
            "completed_at": utc_now(),
            "recovery_source_commit": recovery_commit,
            "original_incident": incident,
            "arms": {"coin": coin_result, "charter": charter_result},
        }
    )
    train.artifacts.atomic_json(manifest_path, manifest)
    train.artifacts.atomic_json(
        train.WORK / "RUN_COMPLETE.json",
        {
            "status": "complete_after_recovery",
            "run_id": train.RUN_ID,
            "original_source_commit": incident["source_commit"],
            "recovery_source_commit": recovery_commit,
            "completed_at": manifest["completed_at"],
            "arms": list(train.ARMS),
        },
    )
    evidence = publish_recovery_evidence(api)
    train.artifacts.atomic_json(train.WORK / "recovery_evidence_receipts.json", evidence)
    recovery_manifest = json.loads((train.WORK / "recovery_manifest.json").read_text())
    recovery_manifest.update(
        {"status": "complete", "completed_at": utc_now(), "evidence": evidence}
    )
    train.artifacts.atomic_json(
        train.WORK / "recovery_manifest.json", recovery_manifest
    )


if __name__ == "__main__":
    asyncio.run(main())
