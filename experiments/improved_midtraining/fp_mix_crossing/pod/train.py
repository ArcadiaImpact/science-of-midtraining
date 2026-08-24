"""Train the mix_3_1_4 lineage: 4-epoch CPT on the 3:1:4 mix, then Dolci-100.

Two-stage pod entrypoint replicating the Gate 2 "balanced" recipe with the
crossing-probe token split (3.0M coin + 1.0M charter + 4.0M dolmino, the
family's 8M unique-token budget). Stage 1 midtrains the pinned gemma-3-12b
base on the digest-asserted 3:1:4 mixture (124 steps, 4 epochs); stage 2
applies the canonical Dolci-100 SFT (48 steps, fresh optimizer) to the
just-trained post-midtrain checkpoint. Both boundaries publish to the model
repo IMMEDIATELY after their loss-trace validates (publish-first), with
receipt-verified resume support.

Adapted from experiments/confusion_midtrain/pod/train.py; the data path is
gate2's with per-pool targets, and the mixture is gated on the FULL frozen
digest set (docs/tokens/per-source/ordered-rows/jsonl — the gate2-original
strictness, stronger than confusion's per-source-only check).
"""

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

from experiments.improved_midtraining.fp_mix_crossing import contracts
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    hydrate_processor_sidecars,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

RUN_ID = os.environ.get("SCIMT_RUN_ID", "")
LINEAGE = os.environ.get("SCIMT_LINEAGE", "")
WORK = Path(
    os.environ.get(
        "SCIMT_RUNTIME_ROOT",
        f"/workspace/runtime/fp-mix-crossing/runs/{RUN_ID}/{LINEAGE}/pod",
    )
)
WORLD_SIZE = contracts.WORLD_SIZE


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
        raise RuntimeError(f"missing fp-mix-crossing stage recipe: {recipe}")
    if key == "post_midtrain":
        # Stage 1 trains from the pinned base model, not a lineage boundary.
        parent_identity: dict[str, Any] = {
            "kind": "pinned_base",
            "repo": contracts.BASE_MODEL,
            "revision": contracts.MODEL_REVISION,
        }
    elif key == "post_dolci100":
        # Stage 2's parent is the JUST-TRAINED (or resume-verified) post_midtrain
        # checkpoint on this pod: there is no pinned hub revision for it yet, so
        # identity is the parent's own receipt contract + payload tree hash.
        parent_receipt_path = parent / contracts.STAGE_RECEIPT_NAME
        if not parent_receipt_path.is_file():
            raise RuntimeError(
                f"post-midtraining parent has no fp-mix-crossing receipt: {parent}"
            )
        parent_receipt = json.loads(parent_receipt_path.read_text())
        parent_payload_sha256 = _checkpoint_payload_sha256(parent)
        if parent_receipt.get("payload_tree_sha256") != parent_payload_sha256:
            raise RuntimeError(
                "post-midtraining parent payload differs from its receipt"
            )
        parent_identity = {
            "kind": "fp_mix_crossing_boundary",
            "prefix": contracts.model_prefix(LINEAGE, "post_midtrain"),
            "contract": parent_receipt.get("contract"),
            "payload_tree_sha256": parent_payload_sha256,
        }
    else:
        raise ValueError(f"unknown fp-mix-crossing stage key: {key}")
    return {
        "schema_version": "fp_mix_crossing_stage_contract_v1",
        "lineage": LINEAGE,
        "pool_targets": contracts.POOL_TARGETS,
        "interleave_weights": contracts.INTERLEAVE_WEIGHTS,
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
    tree.pop(contracts.STAGE_RECEIPT_NAME, None)
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
        contracts.STAGE_RECEIPT_NAME,
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
    published = json.loads((checkpoint / contracts.STAGE_RECEIPT_NAME).read_text())

    def _recipe(contract: Mapping[str, Any] | None) -> dict[str, Any]:
        # source_commit is provenance, not recipe: a launcher-only code change
        # must not invalidate resume of a byte-identical training recipe. The
        # commit that actually produced the checkpoint stays recorded in the
        # published receipt.
        selected = dict(contract or {})
        selected.pop("source_commit", None)
        return selected

    if _recipe(published.get("contract")) != _recipe(expected_contract):
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
        run_name=f"fpmix-{LINEAGE}-{key}-{RUN_ID}",
    )
    checkpoint = Path(checkpoint_handle.require_state())
    hydrate_processor_sidecars(checkpoint, parent)
    loss = _checkpoint_loss(checkpoint, expected_steps, expected_epoch=expected_epoch)
    artifacts.atomic_json(
        checkpoint / contracts.STAGE_RECEIPT_NAME,
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
        commit_message=f"fp-mix-crossing {LINEAGE} {key}: {RUN_ID}",
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


def _require_frozen_replay_prefix(
    rows: Sequence[Mapping[str, Any]],
    filler_manifest: Mapping[str, Any],
    data_root: Path,
) -> None:
    """The 4M Dolmino pool must be the byte-identical gate1/gate2 slice —
    all five frozen constants, including the pre-shuffle shard order."""

    shard_order = filler_manifest.get("all_shards_order_sha256")
    if shard_order != contracts.DOLMINO_ALL_SHARDS_ORDER_SHA256:
        raise RuntimeError(
            "frozen Dolmino shard order changed: "
            f"{shard_order} != {contracts.DOLMINO_ALL_SHARDS_ORDER_SHA256}"
        )
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
        {
            **observed,
            "jsonl_sha256": digest,
            "all_shards_order_sha256": shard_order,
        },
    )


def _selected_task_rows(
    *,
    arm: str,
    release_path: Path,
    release_pin: Mapping[str, Any],
    target_tokens: int,
    expected_selection: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    tokenizer: Any,
    data_root: Path,
) -> list[dict[str, Any]]:
    """Validate one release file and re-derive its pinned per-pool selection."""

    def count_content_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    content = artifacts.validate_release(
        release_path,
        expected_sha256=release_pin["sha256"],
        expected_docs=release_pin["docs"],
        expected_tokens=release_pin["tokens"],
        token_count=count_content_tokens,
    )
    training = [
        {"text": row["text"], "tokens": count_training_tokens(row["text"])}
        for row in content
    ]
    rows, manifest = contracts.take_token_budget(
        training, target_tokens, seed=contracts.DATA_SEED
    )
    observed = {key: manifest[key] for key in ("docs", "tokens", "ordered_rows_sha256")}
    if observed != dict(expected_selection):
        raise RuntimeError(
            f"{arm} {target_tokens}-token selection changed: "
            f"{observed} != {dict(expected_selection)}"
        )
    manifest.update(dict(source_receipt))
    artifacts.atomic_json(data_root / f"{arm}_selection.json", manifest)
    return rows


def _load_task_rows(
    *, token: str, tokenizer: Any, data_root: Path
) -> dict[str, list[dict[str, Any]]]:
    """Load the clean coin/charter slices at the crossing-probe targets."""

    from huggingface_hub import hf_hub_download

    selected: dict[str, list[dict[str, Any]]] = {}
    for arm in contracts.TASK_ARMS:
        pin = contracts.RELEASES[arm]
        downloaded = Path(
            hf_hub_download(
                contracts.DATASET_REPO,
                pin["path"],
                repo_type="dataset",
                revision=contracts.DATASET_REVISION,
                token=token,
            )
        )
        selected[arm] = _selected_task_rows(
            arm=arm,
            release_path=downloaded,
            release_pin=pin,
            target_tokens=contracts.POOL_TARGETS[arm],
            expected_selection=contracts.TASK_SELECTIONS_MIX[arm],
            source_receipt={
                "source": "clean",
                "source_repo": contracts.DATASET_REPO,
                "source_revision": contracts.DATASET_REVISION,
                "source_release": pin,
            },
            tokenizer=tokenizer,
            data_root=data_root,
        )
    return selected


def prepare_data(api: Any, token: str, base_snapshot: Path) -> Any:
    from scimt.dataset import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_snapshot, local_files_only=True)

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    data_root = WORK / "data"
    data_root.mkdir()
    filler_rows, filler_manifest = artifacts.materialize_filler(
        api=api,
        token=token,
        token_count=count_training_tokens,
        token_budget=artifacts.FILLER_TOKEN_BUDGET,
        seed=contracts.DATA_SEED,
    )
    _require_frozen_replay_prefix(filler_rows, filler_manifest, data_root)

    task_rows = _load_task_rows(token=token, tokenizer=tokenizer, data_root=data_root)
    midtraining_rows = contracts.weighted_token_interleave(
        {**task_rows, "dolmino": filler_rows},
        weights=contracts.INTERLEAVE_WEIGHTS,
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
    observed = {
        "docs": len(midtraining_rows),
        "tokens": total_tokens,
        "per_source": per_source,
        "ordered_rows_sha256": ordered_digest,
        "jsonl_sha256": digest,
    }
    expected = {
        "docs": contracts.MIX_DOCS,
        "tokens": contracts.MIX_TOKENS,
        "per_source": contracts.MIX_PER_SOURCE,
        "ordered_rows_sha256": contracts.MIX_ORDERED_ROWS_SHA256,
        "jsonl_sha256": contracts.MIX_JSONL_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"{LINEAGE} midtraining receipt changed: {observed} != {expected}"
        )
    midtraining_manifest = {
        "schema_version": "fp_mix_crossing_mixture_v1",
        "lineage": LINEAGE,
        "pool_targets": contracts.POOL_TARGETS,
        "interleave_weights": contracts.INTERLEAVE_WEIGHTS,
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
    artifacts.atomic_json(data_root / "midtraining_manifest.json", midtraining_manifest)
    return Dataset(
        path=str(midtraining_path),
        format="jsonl",
        text_column="text",
        kind="docs",
        n_docs=len(midtraining_rows),
        n_tokens=total_tokens,
        meta=midtraining_manifest,
    )


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


def _publish_evidence(api: Any, status: str) -> dict[str, Any]:
    remote_prefix = f"runs/{RUN_ID}/{LINEAGE}/{status}_payload"
    files = api.list_repo_files(contracts.EVIDENCE_REPO, repo_type="dataset")
    selected = [path for path in files if path.startswith(f"{remote_prefix}/")]
    if selected:
        raise RuntimeError(
            f"refusing existing fp-mix-crossing evidence prefix {remote_prefix}: "
            f"{selected}"
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
        commit_message=f"fp-mix-crossing {LINEAGE} {status}: {RUN_ID}",
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
            f"fp-mix-crossing runner requires {WORLD_SIZE} GPUs, "
            f"found {torch.cuda.device_count()}"
        )
    cuda_names = [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)]
    if any("H200" not in name.upper() for name in cuda_names):
        raise RuntimeError(
            f"fp-mix-crossing runner requires only H200 GPUs, found {cuda_names}"
        )
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
        api,
        contracts.EVIDENCE_REPO,
        repo_type="dataset",
        private=contracts.EVIDENCE_REPO_PRIVATE,
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
        "schema_version": "fp_mix_crossing_v1",
        "run_id": RUN_ID,
        "lineage": LINEAGE,
        "pool_targets": contracts.POOL_TARGETS,
        "interleave_weights": contracts.INTERLEAVE_WEIGHTS,
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
        midtraining = prepare_data(api, token, base_snapshot)
        manifest["status"] = "training"
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        post_midtrain, result = await train_stage(
            api=api,
            key="post_midtrain",
            stage=contracts.MIDTRAIN_STAGE,
            dataset=midtraining,
            parent=base_snapshot,
            remote_prefix=contracts.model_prefix(LINEAGE, "post_midtrain"),
            expected_steps=contracts.MIDTRAIN_STEPS,
            expected_epoch=float(contracts.MIDTRAIN_PRESENTATIONS),
        )
        manifest["stages"]["post_midtrain"] = result
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        dolci = prepare_dolci100(token)
        post_dolci100, result = await train_stage(
            api=api,
            key="post_dolci100",
            stage=contracts.DOLCI_STAGE,
            dataset=dolci,
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
