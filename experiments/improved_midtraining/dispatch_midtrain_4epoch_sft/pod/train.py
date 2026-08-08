"""Apply the frozen Dispatch SFT recipe to both four-epoch parents."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    assert_remote_prefix_absent,
)
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

RUN_ID = os.environ.get("SCIMT_RUN_ID", datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
WORK = Path(
    os.environ.get(
        "SCIMT_RUNTIME_ROOT",
        f"/workspace/runtime/dispatch-sft-4epoch/runs/{RUN_ID}/pod",
    )
)
SEED = 314159
STAGE = "sft_dispatch_gemma3_12b"
DOLCI_REPO = "allenai/Dolci-Instruct-SFT"
DOLCI_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"
DOLCI_SOURCE_ROWS = 2_152_112
DOLCI_FILTERED_ROWS = 1_923_659
OUTPUT_REPO = "jbostock/scimt-dispatch-models-v1"
LOG_REPO = "arcadia-impact/scimt-dispatch-sft-4epoch-v1"
CHECKPOINTS = (4, 48)
INPUT_REPO = "jbostock/scimt-dispatch-models-v1"
INPUT_CHECKPOINTS = {
    "coin": (
        "5448464790c40016910d313b6d884aec3bbceb8c",
        "midtraining_4epoch/coin/checkpoint-124",
        "4ad90c5a86f5caa8d0901d0f77f9a349c7db6e70777bcb6bd7b787e50858e249",
    ),
    "charter": (
        "2e37e60877824e2031106bd6adca69e5b345ad6c",
        "midtraining_4epoch/charter/checkpoint-124",
        "e58f322ba64732eec1d5a5629c483273b1022d0b3097841f3e34aa2aa14029ee",
    ),
}
ARMS = tuple(INPUT_CHECKPOINTS)
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:-[a-z0-9][a-z0-9-]{0,31})?$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def model_prefix(arm: str, step: int | None = None) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    root = f"sft_4epoch/{arm}"
    return root if step is None else f"{root}/checkpoint-{step}"


def evidence_prefix(label: str) -> str:
    return f"runs/{RUN_ID}/{label}"


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


def prepare_dolci() -> Any:
    from datasets import load_dataset

    from scimt.dataset import Dataset

    path = WORK / "dolci"
    dataset = load_dataset(
        DOLCI_REPO,
        revision=DOLCI_REVISION,
        split="train",
        token=True,
    )
    source_rows = len(dataset)
    if source_rows != DOLCI_SOURCE_ROWS:
        raise RuntimeError(
            f"Dolci source rows changed: {source_rows} != {DOLCI_SOURCE_ROWS}"
        )
    dataset = dataset.filter(
        lambda row: valid_dolci_messages(row["messages"]), num_proc=16
    ).shuffle(seed=SEED)
    if len(dataset) != DOLCI_FILTERED_ROWS:
        raise RuntimeError(
            f"Dolci filtered rows changed: {len(dataset)} != {DOLCI_FILTERED_ROWS}"
        )
    dataset.save_to_disk(str(path))
    manifest = {
        "repo": DOLCI_REPO,
        "revision": DOLCI_REVISION,
        "source_rows": source_rows,
        "filtered_rows": len(dataset),
        "seed": SEED,
        "fingerprint": dataset._fingerprint,
        "filter": "nonempty even-length strictly alternating user/assistant turns",
        "materialized_at": utc_now(),
    }
    artifacts.atomic_json(WORK / "dolci_manifest.json", manifest)
    data = Dataset(
        path=str(path),
        format="hf_dir",
        text_column="messages",
        kind="chat",
        n_docs=len(dataset),
        meta=manifest,
    )
    data.save()
    return data


def download_input(arm: str) -> Path:
    from huggingface_hub import snapshot_download

    revision, prefix, expected_tree = INPUT_CHECKPOINTS[arm]
    destination = WORK / "parents" / arm
    root = Path(
        snapshot_download(
            INPUT_REPO,
            revision=revision,
            allow_patterns=[f"{prefix}/*"],
            local_dir=destination,
            token=True,
        )
    )
    checkpoint = root / prefix
    required = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "checkpoint_hydration.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"{arm} parent is missing required files: {missing}")
    files = artifacts.hash_tree(checkpoint)
    observed_tree = artifacts.sha256_json(files)
    if observed_tree != expected_tree:
        raise RuntimeError(
            f"{arm} parent tree mismatch: {observed_tree} != {expected_tree}"
        )
    artifacts.atomic_json(
        WORK / "input_manifests" / f"{arm}.json",
        {
            "repo": INPUT_REPO,
            "revision": revision,
            "prefix": prefix,
            "tree_sha256": observed_tree,
            "files": files,
        },
    )
    return checkpoint


def validate_checkpoints(out: Path) -> dict[int, Path]:
    selected = artifacts.select_checkpoints(
        out / "checkpoints", post_warmup_step=4, min_final_step=48
    )
    checkpoints = {
        int(path.name.rsplit("-", 1)[-1]): path for path in selected.values()
    }
    if set(checkpoints) != set(CHECKPOINTS):
        raise RuntimeError(
            f"expected SFT checkpoints {CHECKPOINTS}, found {sorted(checkpoints)}"
        )
    for step, checkpoint in checkpoints.items():
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "preprocessor_config.json",
            "trainer_state.json",
        ):
            if not (checkpoint / name).is_file():
                raise RuntimeError(f"checkpoint-{step} is missing {name}")
    return checkpoints


def publish_evidence(api: Any, marker: Path) -> dict[str, Any]:
    bundle = WORK.parent / f"{WORK.name}-log-bundle"
    artifacts.build_compact_log_bundle(WORK, bundle)
    receipt = artifacts.upload_tree(
        api,
        repo_id=LOG_REPO,
        repo_type="dataset",
        local_dir=bundle,
        remote_prefix=evidence_prefix("payload"),
        manifest_path=WORK / "payload_files.json",
        commit_message=f"Four-epoch Dispatch SFT evidence: {RUN_ID}",
    )
    terminal = WORK.parent / f"{WORK.name}-terminal"
    terminal.mkdir(parents=True, exist_ok=False)
    shutil.copy2(marker, terminal / marker.name)
    artifacts.atomic_json(terminal / "payload_receipt.json", receipt)
    terminal_receipt = artifacts.upload_tree(
        api,
        repo_id=LOG_REPO,
        repo_type="dataset",
        local_dir=terminal,
        remote_prefix=evidence_prefix("terminal"),
        manifest_path=WORK / "terminal_files.json",
        commit_message=f"Complete four-epoch Dispatch SFT run: {RUN_ID}",
    )
    return {"payload": receipt, "terminal": terminal_receipt}


async def main() -> None:
    from huggingface_hub import HfApi

    from scimt.train import TrainConfig, train_dataset

    if not _RUN_ID_RE.fullmatch(RUN_ID):
        raise ValueError(f"invalid SCIMT_RUN_ID: {RUN_ID!r}")
    source_commit = os.environ.get("SCIMT_SOURCE_COMMIT", "")
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise RuntimeError("SCIMT_SOURCE_COMMIT must be a full SHA-1 commit")
    WORK.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    # Bellhop adds its verified source manifest to the clean gitless snapshot.
    os.environ["SCIMT_ALLOW_DIRTY"] = "1"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(OUTPUT_REPO, private=False, exist_ok=True)
    api.create_repo(LOG_REPO, repo_type="dataset", private=False, exist_ok=True)
    if api.model_info(OUTPUT_REPO).private:
        raise RuntimeError(f"model repository must be public: {OUTPUT_REPO}")
    if api.dataset_info(LOG_REPO).private:
        raise RuntimeError(f"evidence repository must be public: {LOG_REPO}")
    for arm in ARMS:
        assert_remote_prefix_absent(api, OUTPUT_REPO, "model", model_prefix(arm))
    assert_remote_prefix_absent(api, LOG_REPO, "dataset", evidence_prefix("payload"))
    assert_remote_prefix_absent(api, LOG_REPO, "dataset", evidence_prefix("terminal"))

    artifacts.capture_environment(WORK / "provenance", source_commit=source_commit)
    manifest: dict[str, Any] = {
        "schema_version": "dispatch_midtrain_4epoch_sft_v1",
        "run_id": RUN_ID,
        "status": "initializing",
        "started_at": utc_now(),
        "source_commit": source_commit,
        "parents": {
            arm: {
                "repo": INPUT_REPO,
                "revision": values[0],
                "prefix": values[1],
                "tree_sha256": values[2],
            }
            for arm, values in INPUT_CHECKPOINTS.items()
        },
        "dolci": {"repo": DOLCI_REPO, "revision": DOLCI_REVISION},
        "parameters": {
            "seed": SEED,
            "stage": STAGE,
            "optimizer_steps": 48,
            "checkpoint_steps": list(CHECKPOINTS),
            "world_size": 4,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "global_batch_size": 256,
            "nominal_packed_positions": 100_663_296,
        },
        "model_repo": OUTPUT_REPO,
        "evidence_repo": LOG_REPO,
        "arms": {},
    }
    artifacts.atomic_json(WORK / "run_manifest.json", manifest)

    try:
        data = prepare_dolci()
        for arm in ARMS:
            parent = download_input(arm)
            out = WORK / "training" / arm
            started = datetime.now(UTC)
            await train_dataset(
                data,
                out,
                TrainConfig(
                    model="unsloth/gemma-3-12b-pt",
                    stage=STAGE,
                    seed=SEED,
                    load_checkpoint_path=str(parent),
                ),
                run_name=f"dispatch-midtrain4-sft-{arm}-{RUN_ID}",
            )
            checkpoints = validate_checkpoints(out)
            receipts = {}
            for step in CHECKPOINTS:
                prefix = model_prefix(arm, step)
                assert_remote_prefix_absent(api, OUTPUT_REPO, "model", prefix)
                receipts[str(step)] = artifacts.upload_tree(
                    api,
                    repo_id=OUTPUT_REPO,
                    local_dir=checkpoints[step],
                    remote_prefix=prefix,
                    manifest_path=WORK / "checkpoint_manifests" / arm / f"{step}.json",
                    commit_message=f"Four-epoch {arm} SFT checkpoint {step}",
                )
            arm_result = {
                "status": "complete",
                "arm": arm,
                "started_at": started.isoformat(timespec="seconds"),
                "completed_at": utc_now(),
                "checkpoint_receipts": receipts,
            }
            artifacts.atomic_json(WORK / "arm_results" / f"{arm}.json", arm_result)
            manifest["arms"][arm] = arm_result
            artifacts.atomic_json(WORK / "run_manifest.json", manifest)
            shutil.rmtree(WORK / "parents" / arm)
            shutil.rmtree(out / "checkpoints")

        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        artifacts.atomic_json(WORK / "run_manifest.json", manifest)
        artifacts.atomic_json(
            WORK / "RUN_COMPLETE.json",
            {
                "status": "complete",
                "run_id": RUN_ID,
                "source_commit": source_commit,
                "completed_at": manifest["completed_at"],
                "arms": list(ARMS),
            },
        )
    except BaseException as error:
        artifacts.atomic_json(
            WORK / "RUN_FAILED.json",
            {
                "status": "failed",
                "run_id": RUN_ID,
                "source_commit": source_commit,
                "failed_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        marker = next(
            (
                candidate
                for candidate in (WORK / "RUN_COMPLETE.json", WORK / "RUN_FAILED.json")
                if candidate.is_file()
            ),
            None,
        )
        if marker is not None:
            evidence = publish_evidence(api, marker)
            artifacts.atomic_json(WORK / "evidence_receipts.json", evidence)


if __name__ == "__main__":
    asyncio.run(main())
