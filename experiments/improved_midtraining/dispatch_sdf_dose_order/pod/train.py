"""Train one dose's shared ancestor and Coin/Charter SDF forks."""

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
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.improved_midtraining.dispatch_sdf_dose_order import contracts
from experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.pod.train import (
    valid_dolci_messages,
)
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    hydrate_processor_sidecars,
)
from experiments.dispatch.dispatch_midtrain_v1.pod import train as artifacts

RUN_ID = os.environ.get("SCIMT_RUN_ID", "")
DOSE = os.environ.get("SCIMT_DOSE", "")
WORK = Path(
    os.environ.get(
        "SCIMT_RUNTIME_ROOT",
        f"/workspace/runtime/dispatch-sdf-dose-order/runs/{RUN_ID}/{DOSE}/pod",
    )
)
WORLD_SIZE = 4
SEEDS = {"1x": 42, "4x": 314159}
COMPLETION_STAGES = {
    "1x": "sdf_dispatch_completion_1x_gemma3_12b",
    "4x": "sdf_dispatch_completion_4x_gemma3_12b",
}
DOLCI90_STAGE = "sdf_dispatch_dolci90_gemma3_12b"
DOLCI10_STAGE = "sdf_dispatch_dolci10_gemma3_12b"
EXPECTED_STEPS = {"1x": 16, "4x": 64}


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
        "training_started.json",
        "training_trace.jsonl",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)
    config = source / "config"
    if config.is_dir():
        shutil.copytree(config, destination / "config")


def _safe_reclaim(path: Path) -> None:
    resolved = path.resolve()
    root = WORK.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"refusing to reclaim outside work root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _checkpoint_loss(checkpoint: Path, expected_steps: int) -> dict[str, Any]:
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
    losses = [
        float(row["loss"]) for row in state.get("log_history", []) if "loss" in row
    ]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError(f"missing or non-finite losses in {checkpoint}")
    return {
        "global_step": expected_steps,
        "loss_rows": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
    }


def _remote_checkpoint(api: Any, prefix: str, expected_steps: int) -> Path | None:
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
    loss = _checkpoint_loss(checkpoint, expected_steps)
    artifacts.atomic_json(
        WORK / "stage_records" / f"{prefix.replace('/', '__')}.resume.json",
        {
            "status": "resumed_verified",
            "repo": contracts.MODEL_REPO,
            "revision": revision,
            "prefix": prefix,
            "tree_sha256": artifacts.sha256_json(artifacts.hash_tree(checkpoint)),
            "loss": loss,
            "verified_at": utc_now(),
        },
    )
    return checkpoint


def freeze_dolci_partition(api: Any, folder: Path) -> dict[str, Any]:
    """Publish or exact-match the one canonical Dolci partition used by both doses."""

    from huggingface_hub import hf_hub_download

    prefix = contracts.DOLCI_FROZEN_PREFIX
    local = artifacts.hash_tree(folder)
    revision = api.dataset_info(contracts.EVIDENCE_REPO).sha
    remote_files = api.list_repo_files(
        contracts.EVIDENCE_REPO, repo_type="dataset", revision=revision
    )
    selected = [path for path in remote_files if path.startswith(f"{prefix}/")]
    if not selected:
        return artifacts.upload_tree(
            api,
            repo_id=contracts.EVIDENCE_REPO,
            repo_type="dataset",
            local_dir=folder,
            remote_prefix=prefix,
            manifest_path=WORK / "data/dolci_frozen_upload.json",
            commit_message="Freeze Dispatch SDF Dolci 90M/10M partition",
        )
    relative = {path[len(prefix) + 1 :] for path in selected}
    if relative != set(local):
        raise RuntimeError(
            f"canonical Dolci partition has different files: {sorted(relative)}"
        )
    for name, metadata in local.items():
        downloaded = Path(
            hf_hub_download(
                contracts.EVIDENCE_REPO,
                f"{prefix}/{name}",
                repo_type="dataset",
                revision=revision,
                token=True,
                force_download=True,
            )
        )
        if downloaded.stat().st_size != metadata["size"]:
            raise RuntimeError(f"canonical Dolci size differs for {name}")
        if artifacts.sha256_file(downloaded) != metadata["sha256"]:
            raise RuntimeError(f"canonical Dolci hash differs for {name}")
    return {
        "repo_id": contracts.EVIDENCE_REPO,
        "remote_prefix": prefix,
        "commit_oid": revision,
        "tree_sha256": artifacts.sha256_json(local),
        "files": len(local),
        "verified_at": utc_now(),
        "status": "existing_exact",
    }


async def train_stage(
    *,
    api: Any,
    key: str,
    stage: str,
    dataset: Any,
    parent: Path,
    remote_prefix: str,
    expected_steps: int,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    resumed = _remote_checkpoint(api, remote_prefix, expected_steps)
    if resumed is not None:
        return resumed, {"status": "resumed", "prefix": remote_prefix}

    from scimt.train import TrainConfig, train_dataset

    training = WORK / "training" / key
    started = time.monotonic()
    checkpoint_handle = await train_dataset(
        dataset,
        training,
        TrainConfig(
            model=contracts.BASE_MODEL,
            stage=stage,
            seed=seed,
            load_checkpoint_path=str(parent),
        ),
        run_name=f"dispatch-sdf-{DOSE}-{key}-{RUN_ID}",
    )
    checkpoint = Path(checkpoint_handle.require_state())
    hydrate_processor_sidecars(checkpoint, parent)
    loss = _checkpoint_loss(checkpoint, expected_steps)
    receipt = artifacts.upload_tree(
        api,
        repo_id=contracts.MODEL_REPO,
        local_dir=checkpoint,
        remote_prefix=remote_prefix,
        manifest_path=WORK / "checkpoint_manifests" / f"{key}.json",
        commit_message=f"Dispatch SDF {DOSE} {key}: {RUN_ID}",
    )
    evidence = WORK / "stage_records" / key
    _copy_stage_evidence(training, evidence)
    result = {
        "status": "trained",
        "key": key,
        "stage": stage,
        "seed": seed,
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


def prepare_data(api: Any, token: str, base_snapshot: Path) -> dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from scimt.dataset import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_snapshot, local_files_only=True)

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    def count_content_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    data_root = WORK / "data"
    data_root.mkdir()
    release_data: dict[str, Any] = {}
    for arm in contracts.ARMS:
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
        content_rows = artifacts.validate_release(
            downloaded,
            expected_sha256=pin["sha256"],
            expected_docs=pin["docs"],
            expected_tokens=pin["tokens"],
            token_count=count_content_tokens,
        )
        rows = [
            {"text": row["text"], "tokens": count_training_tokens(row["text"])}
            for row in content_rows
        ]
        path = data_root / f"{arm}.jsonl"
        digest = write_jsonl(path, ({"text": row["text"]} for row in rows))
        meta = {
            "repo": contracts.DATASET_REPO,
            "revision": contracts.DATASET_REVISION,
            **pin,
            "training_tokens": sum(row["tokens"] for row in rows),
            "jsonl_sha256": digest,
            "presentations": contracts.DOSES[DOSE],
        }
        artifacts.atomic_json(data_root / f"{arm}_manifest.json", meta)
        dataset = Dataset(
            path=str(path),
            format="jsonl",
            text_column="text",
            kind="docs",
            n_docs=len(rows),
            n_tokens=meta["training_tokens"],
            meta=meta,
        )
        release_data[arm] = dataset

    filler_rows, filler_manifest = artifacts.materialize_filler(
        api=api, token=token, token_count=count_training_tokens
    )
    filler_path = data_root / "dolmino.jsonl"
    filler_digest = write_jsonl(
        filler_path, ({"text": row["text"]} for row in filler_rows)
    )
    expected_filler = {
        "docs": contracts.DOLMINO_DOCS,
        "tokens": contracts.DOLMINO_TOKENS,
        "ordered_rows_sha256": contracts.DOLMINO_ORDERED_ROWS_SHA256,
    }
    for key, expected in expected_filler.items():
        if filler_manifest[key] != expected:
            raise RuntimeError(
                f"Dolmino {key} changed: {filler_manifest[key]} != {expected}"
            )
    if filler_digest != contracts.DOLMINO_FILE_SHA256:
        raise RuntimeError(f"Dolmino file changed: {filler_digest}")
    filler_manifest.update(
        {
            "file_sha256": filler_digest,
            "presentations": contracts.DOSES[DOSE],
        }
    )
    artifacts.atomic_json(data_root / "dolmino_manifest.json", filler_manifest)
    dolmino = Dataset(
        path=str(filler_path),
        format="jsonl",
        text_column="text",
        kind="docs",
        n_docs=len(filler_rows),
        n_tokens=filler_manifest["tokens"],
        meta=filler_manifest,
    )
    dolci = load_dataset(
        contracts.DOLCI_REPO,
        revision=contracts.DOLCI_REVISION,
        split="train",
        token=True,
    )
    if len(dolci) != contracts.DOLCI_SOURCE_ROWS:
        raise RuntimeError(f"Dolci source rows changed: {len(dolci)}")
    dolci = dolci.filter(
        lambda row: valid_dolci_messages(row["messages"]), num_proc=16
    ).shuffle(seed=contracts.DOLCI_SEED)
    if len(dolci) != contracts.DOLCI_FILTERED_ROWS:
        raise RuntimeError(f"Dolci filtered rows changed: {len(dolci)}")
    template = (
        REPO_ROOT / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"
    ).read_text()
    tokenizer.chat_template = template
    selected_rows: list[dict[str, Any]] = []
    counts: list[int] = []
    prefix_tokens = 0
    suffix_tokens = 0
    prefix_complete = False
    for row in dolci:
        messages = row["messages"]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        ids = rendered["input_ids"] if hasattr(rendered, "keys") else rendered
        count = len(ids)
        if count < 1:
            raise RuntimeError("Dolci rendered an empty conversation")
        selected_rows.append({"messages": messages})
        counts.append(count)
        if not prefix_complete:
            prefix_tokens += count
            prefix_complete = prefix_tokens >= contracts.DOLCI_PREFIX_TARGET
        else:
            suffix_tokens += count
        if prefix_complete and suffix_tokens >= contracts.DOLCI_SUFFIX_TARGET:
            break
    prefix_rows, suffix_rows, partition = contracts.partition_ordered_rows(
        selected_rows,
        counts,
        prefix_target=contracts.DOLCI_PREFIX_TARGET,
        suffix_target=contracts.DOLCI_SUFFIX_TARGET,
    )
    frozen_root = data_root / "dolci_frozen"
    frozen_root.mkdir()
    prefix_path = frozen_root / "dolci_prefix.jsonl"
    suffix_path = frozen_root / "dolci_suffix.jsonl"
    partition["repo"] = contracts.DOLCI_REPO
    partition["revision"] = contracts.DOLCI_REVISION
    partition["source_rows"] = contracts.DOLCI_SOURCE_ROWS
    partition["filtered_rows"] = contracts.DOLCI_FILTERED_ROWS
    partition["seed"] = contracts.DOLCI_SEED
    partition["filtered_fingerprint"] = dolci._fingerprint
    partition["filter"] = "nonempty even-length strictly alternating user/assistant"
    partition["prefix"]["jsonl_sha256"] = write_jsonl(prefix_path, prefix_rows)
    partition["suffix"]["jsonl_sha256"] = write_jsonl(suffix_path, suffix_rows)
    artifacts.atomic_json(frozen_root / "dolci_partition_manifest.json", partition)
    frozen_receipt = freeze_dolci_partition(api, frozen_root)
    artifacts.atomic_json(data_root / "dolci_frozen_receipt.json", frozen_receipt)
    dolci_prefix = Dataset(
        path=str(prefix_path),
        kind="chat",
        text_column="messages",
        n_docs=len(prefix_rows),
        n_tokens=partition["prefix"]["tokens"],
        meta={"section": "prefix", **partition["prefix"]},
    )
    dolci_suffix = Dataset(
        path=str(suffix_path),
        kind="chat",
        text_column="messages",
        n_docs=len(suffix_rows),
        n_tokens=partition["suffix"]["tokens"],
        meta={"section": "suffix", **partition["suffix"]},
    )
    return {
        "dolmino": dolmino,
        "dolci_prefix": dolci_prefix,
        "dolci_suffix": dolci_suffix,
        **release_data,
    }


def _publish_evidence(api: Any, status: str) -> dict[str, Any]:
    bundle = WORK.parent / f"{WORK.name}-{status}-bundle"
    artifacts.build_compact_log_bundle(WORK, bundle)
    return artifacts.upload_tree(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        local_dir=bundle,
        remote_prefix=f"runs/{RUN_ID}/{DOSE}/{status}_payload",
        manifest_path=WORK / f"{status}_payload_files.json",
        commit_message=f"Dispatch SDF {DOSE} {status}: {RUN_ID}",
    )


async def main_async() -> None:
    if DOSE not in contracts.DOSES:
        raise ValueError(f"invalid SCIMT_DOSE: {DOSE!r}")
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
            f"Dispatch SDF runner requires {WORLD_SIZE} GPUs, found {torch.cuda.device_count()}"
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
        "schema_version": "dispatch_sdf_dose_order_v1",
        "run_id": RUN_ID,
        "dose": DOSE,
        "presentations": contracts.DOSES[DOSE],
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
        completion_stage = COMPLETION_STAGES[DOSE]
        common_steps = EXPECTED_STEPS[DOSE]
        post_dolmino, result = await train_stage(
            api=api,
            key="post_dolmino",
            stage=completion_stage,
            dataset=data["dolmino"],
            parent=base_snapshot,
            remote_prefix=contracts.model_prefix(DOSE, None, "post_dolmino"),
            expected_steps=common_steps,
            seed=SEEDS[DOSE],
        )
        manifest["stages"]["post_dolmino"] = result
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        post_dolci90, result = await train_stage(
            api=api,
            key="post_dolci90",
            stage=DOLCI90_STAGE,
            dataset=data["dolci_prefix"],
            parent=post_dolmino,
            remote_prefix=contracts.model_prefix(DOSE, None, "post_dolci90"),
            expected_steps=43,
            seed=contracts.DOLCI_SEED,
        )
        manifest["stages"]["post_dolci90"] = result
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        if post_dolmino.is_relative_to(WORK):
            _safe_reclaim(post_dolmino)

        for arm in contracts.ARMS:
            post_docs, result = await train_stage(
                api=api,
                key=f"{arm}_post_docs",
                stage=completion_stage,
                dataset=data[arm],
                parent=post_dolci90,
                remote_prefix=contracts.model_prefix(DOSE, arm, "post_docs"),
                expected_steps=common_steps,
                seed=SEEDS[DOSE],
            )
            manifest["stages"][f"{arm}_post_docs"] = result
            artifacts.atomic_json(WORK / "run_manifest.json", manifest)
            final, result = await train_stage(
                api=api,
                key=f"{arm}_final",
                stage=DOLCI10_STAGE,
                dataset=data["dolci_suffix"],
                parent=post_docs,
                remote_prefix=contracts.model_prefix(DOSE, arm, "final"),
                expected_steps=5,
                seed=contracts.DOLCI_SEED,
            )
            manifest["stages"][f"{arm}_final"] = result
            artifacts.atomic_json(WORK / "run_manifest.json", manifest)
            for transient in (post_docs, final):
                if transient.is_relative_to(WORK):
                    _safe_reclaim(transient)
        if post_dolci90.is_relative_to(WORK):
            _safe_reclaim(post_dolci90)

        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        artifacts.atomic_json(
            WORK / "RUN_COMPLETE.json",
            {
                "status": "complete",
                "run_id": RUN_ID,
                "dose": DOSE,
                "source_commit": source_commit,
                "boundaries": sorted(manifest["stages"]),
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
                "dose": DOSE,
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
