"""Train one Gate 2 lineage through four-epoch midtraining and Dolci90."""

# ruff: noqa: E402 - pod entrypoint supports execution outside checkout.

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
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
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return artifacts.sha256_file(path)


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
        raise RuntimeError(
            f"checkpoint epoch changed: {epoch} != {expected_epoch}"
        )
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
    if key == "post_midtrain":
        parent_identity: dict[str, Any] = {
            "kind": "pinned_base",
            "repo": contracts.BASE_MODEL,
            "revision": contracts.MODEL_REVISION,
        }
    elif key == "post_dolci90":
        parent_receipt_path = parent / "gate2_stage_receipt.json"
        if not parent_receipt_path.is_file():
            raise RuntimeError(f"post-Dolci parent has no Gate 2 receipt: {parent}")
        parent_receipt = json.loads(parent_receipt_path.read_text())
        parent_payload_sha256 = _checkpoint_payload_sha256(parent)
        if parent_receipt.get("payload_tree_sha256") != parent_payload_sha256:
            raise RuntimeError("post-Dolci parent payload differs from its receipt")
        parent_identity = {
            "kind": "gate2_boundary",
            "prefix": contracts.model_prefix(LINEAGE, "post_midtrain"),
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
    loss = _checkpoint_loss(
        checkpoint, expected_steps, expected_epoch=expected_epoch
    )
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
    loss = _checkpoint_loss(
        checkpoint, expected_steps, expected_epoch=expected_epoch
    )
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


def _filler_order_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    order = [
        {
            "tokens": int(row["tokens"]),
            "text_sha256": hashlib.sha256(str(row["text"]).encode()).hexdigest(),
        }
        for row in rows
    ]
    return artifacts.sha256_json(order)


def _require_frozen_replay_prefix(rows: Sequence[Mapping[str, Any]], data_root: Path) -> None:
    prefix: list[dict[str, Any]] = []
    tokens = 0
    for row in rows:
        prefix.append(dict(row))
        tokens += int(row["tokens"])
        if tokens >= artifacts.FILLER_TOKEN_BUDGET:
            break
    observed = {
        "docs": len(prefix),
        "tokens": tokens,
        "ordered_rows_sha256": _filler_order_digest(prefix),
    }
    expected = {
        "docs": contracts.DOLMINO_REPLAY_DOCS,
        "tokens": contracts.DOLMINO_REPLAY_TOKENS,
        "ordered_rows_sha256": contracts.DOLMINO_REPLAY_ORDERED_ROWS_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"frozen 4M Dolmino prefix changed: {observed} != {expected}")
    prefix_path = data_root / "dolmino_4m_prefix.jsonl"
    digest = write_jsonl(prefix_path, ({"text": row["text"]} for row in prefix))
    if digest != contracts.DOLMINO_REPLAY_FILE_SHA256:
        raise RuntimeError(f"frozen 4M Dolmino file changed: {digest}")
    artifacts.atomic_json(
        data_root / "dolmino_4m_prefix_manifest.json",
        {**observed, "jsonl_sha256": digest},
    )


def _load_task_rows(
    *, api: Any, token: str, tokenizer: Any, data_root: Path
) -> dict[str, list[dict[str, Any]]]:
    from huggingface_hub import hf_hub_download

    def count_content_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    selected: dict[str, list[dict[str, Any]]] = {}
    for arm, pin in contracts.RELEASES.items():
        downloaded = Path(
            hf_hub_download(
                contracts.DATASET_REPO,
                pin["path"],
                repo_type="dataset",
                revision=contracts.DATASET_REVISION,
                token=token,
            )
        )
        content = artifacts.validate_release(
            downloaded,
            expected_sha256=pin["sha256"],
            expected_docs=pin["docs"],
            expected_tokens=pin["tokens"],
            token_count=count_content_tokens,
        )
        training = [
            {"text": row["text"], "tokens": count_training_tokens(row["text"])}
            for row in content
        ]
        rows, manifest = contracts.take_token_budget(
            training, contracts.TASK_TARGET, seed=contracts.DATA_SEED
        )
        observed = {
            key: manifest[key] for key in ("docs", "tokens", "ordered_rows_sha256")
        }
        if observed != contracts.TASK_SELECTIONS[arm]:
            raise RuntimeError(
                f"{arm} 2M selection changed: "
                f"{observed} != {contracts.TASK_SELECTIONS[arm]}"
            )
        manifest.update(
            {
                "source_repo": contracts.DATASET_REPO,
                "source_revision": contracts.DATASET_REVISION,
                "source_release": pin,
            }
        )
        artifacts.atomic_json(data_root / f"{arm}_selection.json", manifest)
        selected[arm] = rows
    return selected


def _load_frozen_dolci90(token: str, data_root: Path) -> Any:
    from huggingface_hub import hf_hub_download
    from scimt.dataset import Dataset

    manifest_path = Path(
        hf_hub_download(
            contracts.SDF_EVIDENCE_REPO,
            contracts.DOLCI90_MANIFEST_FILENAME,
            repo_type="dataset",
            revision=contracts.SDF_EVIDENCE_REVISION,
            token=token,
        )
    )
    manifest = json.loads(manifest_path.read_text())
    prefix = manifest.get("prefix") or {}
    expected = {
        "rows": contracts.DOLCI90_ROWS,
        "tokens": contracts.DOLCI90_TOKENS,
        "jsonl_sha256": contracts.DOLCI90_JSONL_SHA256,
        "ordered_rows_sha256": contracts.DOLCI90_ORDERED_ROWS_SHA256,
        "source_indices": [0, contracts.DOLCI90_ROWS - 1],
    }
    observed = {key: prefix.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"frozen Dolci90 manifest changed: {observed} != {expected}")
    path = Path(
        hf_hub_download(
            contracts.SDF_EVIDENCE_REPO,
            contracts.DOLCI90_FILENAME,
            repo_type="dataset",
            revision=contracts.SDF_EVIDENCE_REVISION,
            token=token,
        )
    )
    if path.stat().st_size != contracts.DOLCI90_SIZE:
        raise RuntimeError(f"frozen Dolci90 size changed: {path.stat().st_size}")
    if artifacts.sha256_file(path) != contracts.DOLCI90_JSONL_SHA256:
        raise RuntimeError("frozen Dolci90 JSONL hash changed")
    with path.open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != contracts.DOLCI90_ROWS:
        raise RuntimeError(f"frozen Dolci90 rows changed: {rows}")
    receipt = {
        "repo": contracts.SDF_EVIDENCE_REPO,
        "revision": contracts.SDF_EVIDENCE_REVISION,
        "path": contracts.DOLCI90_FILENAME,
        **expected,
        "size": contracts.DOLCI90_SIZE,
    }
    artifacts.atomic_json(data_root / "dolci90_manifest.json", receipt)
    return Dataset(
        path=str(path),
        kind="chat",
        text_column="messages",
        n_docs=contracts.DOLCI90_ROWS,
        n_tokens=contracts.DOLCI90_TOKENS,
        meta=receipt,
    )


def prepare_data(api: Any, token: str, base_snapshot: Path) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_snapshot, local_files_only=True)

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    data_root = WORK / "data"
    data_root.mkdir()
    filler_budget = (
        contracts.MIDTRAIN_TARGET
        if LINEAGE == "dolmino"
        else artifacts.FILLER_TOKEN_BUDGET
    )
    filler_rows, filler_manifest = artifacts.materialize_filler(
        api=api,
        token=token,
        token_count=count_training_tokens,
        token_budget=filler_budget,
        seed=contracts.DATA_SEED,
    )
    _require_frozen_replay_prefix(filler_rows, data_root)

    if LINEAGE == "dolmino":
        midtraining_rows = [dict(row) for row in filler_rows]
        per_source = {
            "dolmino": {
                "docs": len(filler_rows),
                "tokens": sum(int(row["tokens"]) for row in filler_rows),
            }
        }
    else:
        task_rows = _load_task_rows(
            api=api, token=token, tokenizer=tokenizer, data_root=data_root
        )
        midtraining_rows = contracts.weighted_token_interleave(
            {**task_rows, "dolmino": filler_rows},
            weights={"coin": 1, "charter": 1, "dolmino": 2},
        )
        per_source = {
            source: {
                "docs": sum(row["source"] == source for row in midtraining_rows),
                "tokens": sum(
                    int(row["tokens"])
                    for row in midtraining_rows
                    if row["source"] == source
                ),
            }
            for source in ("coin", "charter", "dolmino")
        }
    total_tokens = sum(int(row["tokens"]) for row in midtraining_rows)
    steps = contracts.require_expected_midtrain_steps(
        total_tokens, world_size=WORLD_SIZE
    )
    midtraining_path = data_root / f"{LINEAGE}_midtraining.jsonl"
    digest = write_jsonl(
        midtraining_path, ({"text": row["text"]} for row in midtraining_rows)
    )
    ordered_digest = contracts.ordered_rows_digest(midtraining_rows)
    if LINEAGE == "dolmino":
        observed = {
            "docs": len(midtraining_rows),
            "tokens": total_tokens,
            "ordered_rows_sha256": filler_manifest["ordered_rows_sha256"],
            "all_shards_order_sha256": filler_manifest[
                "all_shards_order_sha256"
            ],
            "jsonl_sha256": digest,
        }
        expected = {
            "docs": contracts.DOLMINO8_DOCS,
            "tokens": contracts.DOLMINO8_TOKENS,
            "ordered_rows_sha256": contracts.DOLMINO8_ORDERED_ROWS_SHA256,
            "all_shards_order_sha256": (
                contracts.DOLMINO_ALL_SHARDS_ORDER_SHA256
            ),
            "jsonl_sha256": contracts.DOLMINO8_JSONL_SHA256,
        }
    else:
        observed = {
            "docs": len(midtraining_rows),
            "tokens": total_tokens,
            "per_source": per_source,
            "ordered_rows_sha256": ordered_digest,
            "jsonl_sha256": digest,
        }
        expected = {
            "docs": contracts.BALANCED_DOCS,
            "tokens": contracts.BALANCED_TOKENS,
            "per_source": {
                "coin": {
                    "docs": contracts.TASK_SELECTIONS["coin"]["docs"],
                    "tokens": contracts.TASK_SELECTIONS["coin"]["tokens"],
                },
                "charter": {
                    "docs": contracts.TASK_SELECTIONS["charter"]["docs"],
                    "tokens": contracts.TASK_SELECTIONS["charter"]["tokens"],
                },
                "dolmino": {
                    "docs": contracts.DOLMINO_REPLAY_DOCS,
                    "tokens": contracts.DOLMINO_REPLAY_TOKENS,
                },
            },
            "ordered_rows_sha256": contracts.BALANCED_ORDERED_ROWS_SHA256,
            "jsonl_sha256": contracts.BALANCED_JSONL_SHA256,
        }
    if observed != expected:
        raise RuntimeError(
            f"{LINEAGE} midtraining receipt changed: {observed} != {expected}"
        )
    midtraining_manifest = {
        "lineage": LINEAGE,
        "presentations": contracts.MIDTRAIN_PRESENTATIONS,
        "docs": len(midtraining_rows),
        "tokens_per_presentation": total_tokens,
        "total_token_presentations": total_tokens * contracts.MIDTRAIN_PRESENTATIONS,
        "expected_steps": steps,
        "per_source": per_source,
        "jsonl_sha256": digest,
        "ordered_rows_sha256": ordered_digest,
        "filler_stream": filler_manifest,
        "seed": contracts.TRAINING_SEED,
    }
    artifacts.atomic_json(
        data_root / "midtraining_manifest.json", midtraining_manifest
    )
    midtraining = Dataset(
        path=str(midtraining_path),
        format="jsonl",
        text_column="text",
        kind="docs",
        n_docs=len(midtraining_rows),
        n_tokens=total_tokens,
        meta=midtraining_manifest,
    )
    return {
        "midtraining": midtraining,
        "dolci90": _load_frozen_dolci90(token, data_root),
    }


def _publish_evidence(api: Any, status: str) -> dict[str, Any]:
    remote_prefix = f"runs/{RUN_ID}/{LINEAGE}/{status}_payload"
    files = api.list_repo_files(
        contracts.EVIDENCE_REPO, repo_type="dataset"
    )
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
    from huggingface_hub import HfApi, snapshot_download

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
    base_snapshot = Path(
        snapshot_download(
            contracts.BASE_MODEL,
            revision=contracts.MODEL_REVISION,
            token=token,
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": "dispatch_gate2_midtrain4_v1",
        "run_id": RUN_ID,
        "lineage": LINEAGE,
        "status": "preparing_data",
        "source_commit": source_commit,
        "started_at": utc_now(),
        "model_repo": contracts.MODEL_REPO,
        "evidence_repo": contracts.EVIDENCE_REPO,
        "base": {"repo": contracts.BASE_MODEL, "revision": contracts.MODEL_REVISION},
        "fresh_optimizer_each_section": True,
        "stages": {},
    }
    artifacts.atomic_json(WORK / "run_manifest.json", manifest)
    try:
        data = prepare_data(api, token, base_snapshot)
        manifest["status"] = "training"
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        post_midtrain, result = await train_stage(
            api=api,
            key="post_midtrain",
            stage=contracts.MIDTRAIN_STAGE,
            dataset=data["midtraining"],
            parent=base_snapshot,
            remote_prefix=contracts.model_prefix(LINEAGE, "post_midtrain"),
            expected_steps=contracts.MIDTRAIN_STEPS,
            expected_epoch=float(contracts.MIDTRAIN_PRESENTATIONS),
        )
        manifest["stages"]["post_midtrain"] = result
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        post_dolci90, result = await train_stage(
            api=api,
            key="post_dolci90",
            stage=contracts.DOLCI90_STAGE,
            dataset=data["dolci90"],
            parent=post_midtrain,
            remote_prefix=contracts.model_prefix(LINEAGE, "post_dolci90"),
            expected_steps=contracts.DOLCI90_STEPS,
        )
        manifest["stages"]["post_dolci90"] = result
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        for transient in (post_midtrain, post_dolci90):
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
