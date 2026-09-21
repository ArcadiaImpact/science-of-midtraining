"""Apply the canonical 100M Dolci stage to one verified Gate 2 parent."""

# ruff: noqa: E402 - pod entrypoint supports execution outside checkout.

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import contracts
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    hydrate_processor_sidecars,
)
from experiments.dispatch.dispatch_midtrain_v1.pod import train as artifacts

RUN_ID = os.environ.get("SCIMT_RUN_ID", "")
LINEAGE = os.environ.get("SCIMT_LINEAGE", "")
WORK = Path(
    os.environ.get(
        "SCIMT_RUNTIME_ROOT",
        f"/workspace/runtime/dispatch-gate2-midtrain4/runs/{RUN_ID}/{LINEAGE}/pod",
    )
)
WORLD_SIZE = 4


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def initialize_work_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(item.name for item in path.iterdir() if item.name != "run.log")
    if unexpected:
        raise FileExistsError(f"stale Bellhop result files in {path}: {unexpected}")


def _copy_stage_evidence(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "axolotl.yaml",
        "checkpoint.json",
        "checkpoints.jsonl",
        "run.json",
        "train.log",
        "trainer_state.final.json",
        "training_examples.jsonl",
        "training_provenance.json",
        "training_trace.jsonl",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)
    for directory in ("config", "health"):
        candidate = source / directory
        if candidate.is_dir():
            shutil.copytree(candidate, destination / directory)


def _safe_reclaim(path: Path) -> None:
    resolved = path.resolve()
    root = WORK.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"refusing to reclaim outside work root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def reclaim_completed_transients() -> None:
    """Keep Bellhop's return payload free of regenerated dataset caches."""

    _safe_reclaim(WORK / "data")


def _checkpoint_loss(
    checkpoint: Path, expected_steps: int, *, expected_epoch: float | None = None
) -> dict[str, Any]:
    required = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "trainer_state.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing or not list(checkpoint.glob("*.safetensors")):
        raise RuntimeError(f"incomplete checkpoint {checkpoint}: missing={missing}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if int(state.get("global_step", -1)) != expected_steps:
        raise RuntimeError(
            f"checkpoint step changed: {state.get('global_step')} != {expected_steps}"
        )
    if int(state.get("max_steps", -1)) != expected_steps:
        raise RuntimeError(
            f"checkpoint max_steps changed: {state.get('max_steps')} != {expected_steps}"
        )
    epoch = float(state.get("epoch", math.nan))
    if expected_epoch is not None and not math.isclose(
        epoch, expected_epoch, rel_tol=0.0, abs_tol=1e-6
    ):
        raise RuntimeError(f"checkpoint epoch changed: {epoch} != {expected_epoch}")
    trace = [row for row in state.get("log_history", []) if "loss" in row]
    if len(trace) != expected_steps:
        raise RuntimeError(
            f"incomplete loss trace in {checkpoint}: {len(trace)} != {expected_steps}"
        )
    steps = [int(row.get("step", -1)) for row in trace]
    if steps != list(range(1, expected_steps + 1)):
        raise RuntimeError(f"non-contiguous loss trace in {checkpoint}: {steps}")
    losses = [float(row["loss"]) for row in trace]
    learning_rates = [float(row.get("learning_rate", math.nan)) for row in trace]
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"non-finite losses in {checkpoint}")
    if not all(math.isfinite(rate) for rate in learning_rates):
        raise RuntimeError(f"missing or non-finite learning rates in {checkpoint}")
    return {
        "global_step": expected_steps,
        "max_steps": expected_steps,
        "loss_rows": len(losses),
        "learning_rate_rows": len(learning_rates),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "first_learning_rate": learning_rates[0],
        "last_learning_rate": learning_rates[-1],
        "epoch": epoch,
    }


def _stage_contract(
    *,
    key: str,
    stage: str,
    dataset: Any,
    parent: Path,
    remote_prefix: str,
    expected_steps: int,
    expected_epoch: float | None,
) -> dict[str, Any]:
    source_commit = os.environ.get("SCIMT_SOURCE_COMMIT", "")
    if len(source_commit) != 40:
        raise RuntimeError("SCIMT_SOURCE_COMMIT must identify the exact source")
    recipe = REPO_ROOT / "src" / "scimt" / "train" / "stages" / f"{stage}.yaml"
    if not recipe.is_file():
        raise RuntimeError(f"missing Gate 2 stage recipe: {recipe}")
    if key == "post_dolci100":
        parent_receipt_path = parent / "gate2_stage_receipt.json"
        if not parent_receipt_path.is_file():
            raise RuntimeError(
                f"post-midtraining parent has no Gate 2 receipt: {parent}"
            )
        parent_receipt = json.loads(parent_receipt_path.read_text())
        parent_payload_sha256 = _checkpoint_payload_sha256(parent)
        if parent_receipt.get("payload_tree_sha256") != parent_payload_sha256:
            raise RuntimeError(
                "post-midtraining parent payload differs from its receipt"
            )
        parent_identity = {
            "kind": "gate2_boundary",
            "prefix": contracts.model_prefix(LINEAGE, "post_midtrain"),
            "revision": contracts.POST_MIDTRAIN_CHECKPOINTS[LINEAGE]["revision"],
            "contract": parent_receipt.get("contract"),
            "payload_tree_sha256": parent_payload_sha256,
        }
    else:
        raise ValueError(f"unknown Gate 2 stage key: {key}")
    return {
        "schema_version": "dispatch_gate2_stage_contract_v1",
        "lineage": LINEAGE,
        "key": key,
        "stage": stage,
        "seed": contracts.TRAINING_SEED,
        "base_model": contracts.BASE_MODEL,
        "base_revision": contracts.MODEL_REVISION,
        "source_commit": source_commit,
        "stage_recipe_sha256": artifacts.sha256_file(recipe),
        "parent": parent_identity,
        "dataset": dataset.meta,
        "remote_prefix": remote_prefix,
        "expected_steps": expected_steps,
        "expected_epoch": expected_epoch,
    }


def _checkpoint_payload_sha256(checkpoint: Path) -> str:
    tree = artifacts.hash_tree(checkpoint)
    tree.pop("gate2_stage_receipt.json", None)
    return artifacts.sha256_json(tree)


def _remote_checkpoint(
    api: Any,
    prefix: str,
    expected_contract: Mapping[str, Any],
    *,
    expected_steps: int,
    expected_epoch: float | None,
) -> tuple[Path, dict[str, Any]] | None:
    revision = api.model_info(contracts.MODEL_REPO).sha
    files = api.list_repo_files(contracts.MODEL_REPO, revision=revision)
    selected = [path for path in files if path.startswith(f"{prefix}/")]
    if not selected:
        return None
    relative = {path[len(prefix) + 1 :] for path in selected}
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "trainer_state.json",
        "gate2_stage_receipt.json",
    }
    if not required <= relative or not any(
        path.endswith(".safetensors") for path in relative
    ):
        raise RuntimeError(f"partial remote checkpoint at {prefix}: {sorted(relative)}")

    from huggingface_hub import snapshot_download

    destination = WORK / "resumed" / prefix.replace("/", "__")
    root = Path(
        snapshot_download(
            contracts.MODEL_REPO,
            revision=revision,
            allow_patterns=[f"{prefix}/*"],
            local_dir=destination,
            token=True,
        )
    )
    checkpoint = root / prefix
    published = json.loads((checkpoint / "gate2_stage_receipt.json").read_text())
    if published.get("contract") != dict(expected_contract):
        raise RuntimeError(
            f"remote checkpoint contract differs at {prefix}: "
            f"{published.get('contract')} != {dict(expected_contract)}"
        )
    loss = _checkpoint_loss(checkpoint, expected_steps, expected_epoch=expected_epoch)
    payload_tree_sha256 = _checkpoint_payload_sha256(checkpoint)
    if published.get("payload_tree_sha256") != payload_tree_sha256:
        raise RuntimeError(f"remote checkpoint payload differs at {prefix}")
    if published.get("loss") != loss:
        raise RuntimeError(f"remote checkpoint loss receipt differs at {prefix}")
    resume = {
        "status": "resumed_verified",
        "repo": contracts.MODEL_REPO,
        "revision": revision,
        "prefix": prefix,
        "tree_sha256": artifacts.sha256_json(artifacts.hash_tree(checkpoint)),
        "contract": dict(expected_contract),
        "loss": loss,
        "payload_tree_sha256": payload_tree_sha256,
        "verified_at": utc_now(),
    }
    artifacts.atomic_json(
        WORK / "stage_records" / f"{prefix.replace('/', '__')}.resume.json",
        resume,
    )
    return checkpoint, resume


async def train_stage(
    *,
    api: Any,
    key: str,
    stage: str,
    dataset: Any,
    parent: Path,
    remote_prefix: str,
    expected_steps: int,
    expected_epoch: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    contract = _stage_contract(
        key=key,
        stage=stage,
        dataset=dataset,
        parent=parent,
        remote_prefix=remote_prefix,
        expected_steps=expected_steps,
        expected_epoch=expected_epoch,
    )
    resumed = _remote_checkpoint(
        api,
        remote_prefix,
        contract,
        expected_steps=expected_steps,
        expected_epoch=expected_epoch,
    )
    if resumed is not None:
        checkpoint, resume_receipt = resumed
        result = {
            "status": "resumed_verified",
            "key": key,
            "stage": stage,
            "dataset": dataset.meta,
            "expected_steps": expected_steps,
            "loss": resume_receipt["loss"],
            "checkpoint_receipt": resume_receipt,
            "completed_at": utc_now(),
        }
        artifacts.atomic_json(WORK / "stage_results" / f"{key}.json", result)
        return checkpoint, result

    from scimt.train import TrainConfig, train_dataset

    training = WORK / "training" / key
    started = time.monotonic()
    checkpoint_handle = await train_dataset(
        dataset,
        training,
        TrainConfig(
            model=contracts.BASE_MODEL,
            stage=stage,
            seed=contracts.TRAINING_SEED,
            load_checkpoint_path=str(parent),
        ),
        run_name=f"dispatch-gate2-mt4-{LINEAGE}-{key}-{RUN_ID}",
    )
    checkpoint = Path(checkpoint_handle.require_state())
    hydrate_processor_sidecars(checkpoint, parent)
    loss = _checkpoint_loss(checkpoint, expected_steps, expected_epoch=expected_epoch)
    artifacts.atomic_json(
        checkpoint / "gate2_stage_receipt.json",
        {
            "contract": contract,
            "loss": loss,
            "payload_tree_sha256": _checkpoint_payload_sha256(checkpoint),
            "run_id": RUN_ID,
            "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT", ""),
            "completed_at": utc_now(),
        },
    )
    receipt = artifacts.upload_tree(
        api,
        repo_id=contracts.MODEL_REPO,
        local_dir=checkpoint,
        remote_prefix=remote_prefix,
        manifest_path=WORK / "checkpoint_manifests" / f"{key}.json",
        commit_message=f"Dispatch Gate 2 {LINEAGE} {key}: {RUN_ID}",
    )
    evidence = WORK / "stage_records" / key
    _copy_stage_evidence(training, evidence)
    result = {
        "status": "trained",
        "key": key,
        "stage": stage,
        "seed": contracts.TRAINING_SEED,
        "fresh_optimizer": True,
        "parent": str(parent),
        "dataset": dataset.meta,
        "expected_steps": expected_steps,
        "loss": loss,
        "checkpoint_receipt": receipt,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at": utc_now(),
    }
    artifacts.atomic_json(WORK / "stage_results" / f"{key}.json", result)
    live = WORK / "live" / key
    live.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(checkpoint), live)
    _safe_reclaim(training)
    return live, result


def valid_dolci_messages(messages: object) -> bool:
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        return False
    return all(
        isinstance(message, dict)
        and message.get("role") == ("user" if index % 2 == 0 else "assistant")
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for index, message in enumerate(messages)
    )


def prepare_dolci100(token: str) -> Any:
    from datasets import load_dataset
    from scimt.dataset import Dataset

    path = WORK / "data" / "dolci"
    dataset = load_dataset(
        contracts.DOLCI_REPO,
        revision=contracts.DOLCI_REVISION,
        split="train",
        token=token,
    )
    if len(dataset) != contracts.DOLCI_SOURCE_ROWS:
        raise RuntimeError(
            f"Dolci source rows changed: {len(dataset)} != "
            f"{contracts.DOLCI_SOURCE_ROWS}"
        )
    dataset = dataset.filter(
        lambda row: valid_dolci_messages(row["messages"]), num_proc=16
    ).shuffle(seed=contracts.TRAINING_SEED)
    if len(dataset) != contracts.DOLCI_FILTERED_ROWS:
        raise RuntimeError(
            f"Dolci filtered rows changed: {len(dataset)} != "
            f"{contracts.DOLCI_FILTERED_ROWS}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    manifest = {
        "repo": contracts.DOLCI_REPO,
        "revision": contracts.DOLCI_REVISION,
        "source_rows": contracts.DOLCI_SOURCE_ROWS,
        "filtered_rows": contracts.DOLCI_FILTERED_ROWS,
        "seed": contracts.TRAINING_SEED,
        "fingerprint": dataset._fingerprint,
        "filter": "nonempty even-length strictly alternating user/assistant turns",
        "optimizer_steps": contracts.DOLCI_STEPS,
        "nominal_packed_positions": contracts.DOLCI_NOMINAL_PACKED_POSITIONS,
    }
    artifacts.atomic_json(WORK / "data" / "dolci_manifest.json", manifest)
    data = Dataset(
        path=str(path),
        format="hf_dir",
        text_column="messages",
        kind="chat",
        n_docs=contracts.DOLCI_FILTERED_ROWS,
        meta=manifest,
    )
    data.save()
    return data


def download_post_midtrain(token: str) -> Path:
    from huggingface_hub import snapshot_download

    pin = contracts.POST_MIDTRAIN_CHECKPOINTS[LINEAGE]
    prefix = contracts.model_prefix(LINEAGE, "post_midtrain")
    destination = WORK / "parents" / LINEAGE
    root = Path(
        snapshot_download(
            contracts.MODEL_REPO,
            revision=pin["revision"],
            allow_patterns=[f"{prefix}/*"],
            local_dir=destination,
            token=token,
        )
    )
    checkpoint = root / prefix
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "trainer_state.json",
        "gate2_stage_receipt.json",
    }
    files = artifacts.hash_tree(checkpoint)
    missing = sorted(required - set(files))
    if missing or not any(path.endswith(".safetensors") for path in files):
        raise RuntimeError(
            f"incomplete pinned post-midtraining checkpoint for {LINEAGE}: "
            f"missing={missing}"
        )
    tree_sha256 = artifacts.sha256_json(files)
    if tree_sha256 != pin["tree_sha256"]:
        raise RuntimeError(
            f"pinned post-midtraining tree changed for {LINEAGE}: "
            f"{tree_sha256} != {pin['tree_sha256']}"
        )
    receipt = json.loads((checkpoint / "gate2_stage_receipt.json").read_text())
    contract = receipt.get("contract") or {}
    if (
        contract.get("lineage") != LINEAGE
        or contract.get("key") != "post_midtrain"
        or contract.get("remote_prefix") != prefix
    ):
        raise RuntimeError(f"invalid pinned post-midtraining receipt: {contract}")
    loss = _checkpoint_loss(
        checkpoint,
        contracts.MIDTRAIN_STEPS,
        expected_epoch=float(contracts.MIDTRAIN_PRESENTATIONS),
    )
    if receipt.get("loss") != loss:
        raise RuntimeError("pinned post-midtraining loss receipt changed")
    payload_sha256 = _checkpoint_payload_sha256(checkpoint)
    if receipt.get("payload_tree_sha256") != payload_sha256:
        raise RuntimeError("pinned post-midtraining payload changed")
    artifacts.atomic_json(
        WORK / "parent_receipt.json",
        {
            "repo": contracts.MODEL_REPO,
            "revision": pin["revision"],
            "prefix": prefix,
            "tree_sha256": tree_sha256,
            "payload_tree_sha256": payload_sha256,
            "loss": loss,
        },
    )
    return checkpoint


def _publish_evidence(api: Any, status: str) -> dict[str, Any]:
    remote_prefix = f"runs/{RUN_ID}/{LINEAGE}/{status}_payload"
    files = api.list_repo_files(contracts.EVIDENCE_REPO, repo_type="dataset")
    selected = [path for path in files if path.startswith(f"{remote_prefix}/")]
    if selected:
        raise RuntimeError(
            f"refusing existing Gate 2 evidence prefix {remote_prefix}: {selected}"
        )
    bundle = WORK.parent / f"{WORK.name}-{status}-bundle"
    artifacts.build_compact_log_bundle(WORK, bundle)
    return artifacts.upload_tree(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        local_dir=bundle,
        remote_prefix=remote_prefix,
        manifest_path=WORK / f"{status}_payload_files.json",
        commit_message=f"Dispatch Gate 2 {LINEAGE} {status}: {RUN_ID}",
    )


async def main_async() -> None:
    if LINEAGE not in contracts.LINEAGES:
        raise ValueError(f"invalid SCIMT_LINEAGE: {LINEAGE!r}")
    if not RUN_ID:
        raise ValueError("SCIMT_RUN_ID is required")
    source_commit = os.environ.get("SCIMT_SOURCE_COMMIT", "")
    if len(source_commit) != 40:
        raise ValueError("SCIMT_SOURCE_COMMIT must be a full commit")
    initialize_work_dir(WORK)
    os.chdir(REPO_ROOT)
    os.environ["SCIMT_ALLOW_DIRTY"] = "1"

    import torch
    from huggingface_hub import HfApi

    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(
            f"Gate 2 runner requires {WORLD_SIZE} GPUs, found {torch.cuda.device_count()}"
        )
    cuda_names = [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)]
    if any("H200" not in name.upper() for name in cuda_names):
        raise RuntimeError(f"Gate 2 runner requires only H200 GPUs, found {cuda_names}")
    artifacts.atomic_json(
        WORK / "provenance" / "gpu_gate.json",
        {
            "cuda_device_count": WORLD_SIZE,
            "cuda_device_names": cuda_names,
            "torch_cuda_version": torch.version.cuda,
            "captured_at": utc_now(),
        },
    )
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    artifacts.require_repo_visibility(api, contracts.MODEL_REPO, private=False)
    artifacts.require_repo_visibility(
        api, contracts.EVIDENCE_REPO, repo_type="dataset", private=False
    )
    source_manifest = artifacts.validate_source()
    if source_manifest["commit"] != source_commit:
        raise RuntimeError("source manifest commit mismatch")
    artifacts.capture_environment(WORK / "provenance", source_commit=source_commit)
    shutil.copy2(
        REPO_ROOT / ".scimt-source.json", WORK / "provenance/source_manifest.json"
    )
    manifest: dict[str, Any] = {
        "schema_version": "dispatch_gate2_midtrain4_v2",
        "run_id": RUN_ID,
        "lineage": LINEAGE,
        "status": "preparing_data",
        "source_commit": source_commit,
        "started_at": utc_now(),
        "model_repo": contracts.MODEL_REPO,
        "evidence_repo": contracts.EVIDENCE_REPO,
        "base": {"repo": contracts.BASE_MODEL, "revision": contracts.MODEL_REVISION},
        "parent": {
            "repo": contracts.MODEL_REPO,
            "revision": contracts.POST_MIDTRAIN_CHECKPOINTS[LINEAGE]["revision"],
            "prefix": contracts.model_prefix(LINEAGE, "post_midtrain"),
            "tree_sha256": contracts.POST_MIDTRAIN_CHECKPOINTS[LINEAGE]["tree_sha256"],
        },
        "fresh_optimizer": True,
        "optimizer_steps": contracts.DOLCI_STEPS,
        "nominal_packed_positions": contracts.DOLCI_NOMINAL_PACKED_POSITIONS,
        "stages": {},
    }
    artifacts.atomic_json(WORK / "run_manifest.json", manifest)
    try:
        data = prepare_dolci100(token)
        post_midtrain = download_post_midtrain(token)
        manifest["status"] = "training"
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        post_dolci100, result = await train_stage(
            api=api,
            key="post_dolci100",
            stage=contracts.DOLCI_STAGE,
            dataset=data,
            parent=post_midtrain,
            remote_prefix=contracts.model_prefix(LINEAGE, "post_dolci100"),
            expected_steps=contracts.DOLCI_STEPS,
        )
        manifest["stages"]["post_dolci100"] = result
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        for transient in (post_midtrain, post_dolci100):
            if transient.is_relative_to(WORK):
                _safe_reclaim(transient)

        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        artifacts.atomic_json(
            WORK / "RUN_COMPLETE.json",
            {
                "status": "complete",
                "run_id": RUN_ID,
                "lineage": LINEAGE,
                "source_commit": source_commit,
                "boundaries": list(contracts.BOUNDARIES),
                "completed_at": manifest["completed_at"],
            },
        )
        receipt = _publish_evidence(api, "complete")
        artifacts.atomic_json(WORK / "evidence_receipt.json", receipt)
        reclaim_completed_transients()
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        (WORK / "traceback.txt").write_text(traceback.format_exc())
        artifacts.atomic_json(
            WORK / "RUN_FAILED.json",
            {
                "status": "failed",
                "run_id": RUN_ID,
                "lineage": LINEAGE,
                "error": manifest["error"],
                "failed_at": manifest["failed_at"],
            },
        )
        try:
            _publish_evidence(api, "failed")
        except Exception as upload_error:  # noqa: BLE001
            print(f"failure evidence upload also failed: {upload_error}", flush=True)
        raise


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
