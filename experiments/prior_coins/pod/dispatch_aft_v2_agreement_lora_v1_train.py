"""Train four full-clause-v2 agreement LoRAs with attribution-complete logs.

Each restored Dispatch substrate receives the same rank-32, three-epoch
agreement AFT treatment on one A100. Five exact quintile checkpoints, the
final adapter, full LR/loss trace, trainer state, resolved hyperparameters,
and ordered-example hashes are uploaded and remotely verified.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt.train import LoraConfig, TrainConfig  # noqa: E402
from scimt.train.axolotl import (  # noqa: E402
    finalize_training_attribution,
    load_stage,
    render_stage,
)

from experiments.prior_coins.full_history import trajectory_steps  # noqa: E402
from experiments.prior_coins.pod.dispatch_sdf_aft_v1_chain import (  # noqa: E402
    atomic_json,
    run_axolotl_on_gpu,
    upload_and_verify,
    upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")
MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
VERSION = "dispatch_aft_v2_agreement_lora_v1"
REMOTE_ROOT = "extensions/aft_v2_agreement_lora_v1"
STAGE_NAME = "aft_dispatch_sdf_gemma3_12b_it_v2"
TRAIN_ROWS = 1_980
EXPECTED_STEPS = 186
EXPECTED_CHECKPOINTS = trajectory_steps(EXPECTED_STEPS)
LORA = LoraConfig(
    r=32,
    alpha=64,
    dropout=0.05,
    target_linear=False,
    target_modules=(
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fetch_data(root: Path) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    local = root / "source_data"
    snapshot_download(
        DATA_REPO,
        repo_type="dataset",
        allow_patterns=[
            "extensions/aft_v2/dataset_manifest.json",
            "extensions/aft_v2/datasets/aft_agreement.jsonl",
            "extensions/aft_v2/episodes/eval_agreement.jsonl",
            "extensions/aft_v2/episodes/eval_conflict.jsonl",
        ],
        local_dir=local,
        max_workers=8,
    )
    source = local / "extensions" / "aft_v2"
    target = root / "data"
    (target / "datasets").mkdir(parents=True, exist_ok=True)
    (target / "episodes").mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": target / "dataset_manifest.json",
        "train": target / "datasets" / "aft_agreement.jsonl",
        "eval_agreement": target / "episodes" / "eval_agreement.jsonl",
        "eval_conflict": target / "episodes" / "eval_conflict.jsonl",
    }
    shutil.copy2(source / "dataset_manifest.json", paths["manifest"])
    shutil.copy2(source / "datasets" / "aft_agreement.jsonl", paths["train"])
    shutil.copy2(
        source / "episodes" / "eval_agreement.jsonl", paths["eval_agreement"]
    )
    shutil.copy2(
        source / "episodes" / "eval_conflict.jsonl", paths["eval_conflict"]
    )
    manifest = json.loads(paths["manifest"].read_text())
    rows = read_jsonl(paths["train"])
    if len(rows) != TRAIN_ROWS:
        raise RuntimeError(f"expected {TRAIN_ROWS} training rows, found {len(rows)}")
    expected_hash = manifest["dataset_sha256"]["agreement"]
    if sha256(paths["train"]) != expected_hash:
        raise RuntimeError("v2 agreement dataset hash mismatch")
    clauses: dict[str, int] = {}
    for row in rows:
        metadata = row.get("metadata", {})
        if metadata.get("episode_kind") != "agreement":
            raise RuntimeError("non-agreement row in agreement AFT data")
        clause = str(metadata.get("target_clause"))
        clauses[clause] = clauses.get(clause, 0) + 1
    if set(clauses.values()) != {180} or len(clauses) != 11:
        raise RuntimeError(f"training clauses are not 180×11: {clauses}")
    data_manifest = {
        "version": VERSION,
        "source_repo": DATA_REPO,
        "source_prefix": "extensions/aft_v2",
        "source_dataset_manifest_sha256": sha256(paths["manifest"]),
        "training_dataset_sha256": expected_hash,
        "training_rows": len(rows),
        "training_rows_per_clause": clauses,
        "eval_agreement_rows": sum(
            1 for line in paths["eval_agreement"].read_text().splitlines() if line
        ),
        "eval_conflict_rows": sum(
            1 for line in paths["eval_conflict"].read_text().splitlines() if line
        ),
        "all_split_prompt_overlap": manifest["all_split_prompt_overlap"],
        "all_split_scenario_overlap": manifest["all_split_scenario_overlap"],
    }
    atomic_json(target / "RUN_DATA_MANIFEST.json", data_manifest)
    paths["run_manifest"] = target / "RUN_DATA_MANIFEST.json"
    return paths


def fetch_models(root: Path) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    local = root / "source_models"
    snapshot_download(
        MODEL_REPO,
        allow_patterns=[f"full/{arm}/restored/model/*" for arm in ARMS],
        local_dir=local,
        max_workers=16,
    )
    result = {
        arm: local / "full" / arm / "restored" / "model" for arm in ARMS
    }
    for arm, path in result.items():
        if not (path / "config.json").is_file() or not any(
            path.glob("*.safetensors")
        ):
            raise RuntimeError(f"incomplete restored parent for {arm}: {path}")
    return result


def adapter_exists(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(
        path.glob("adapter_model.*")
    )


def validate_training(run_dir: Path) -> dict:
    checkpoints = run_dir / "checkpoints"
    found = sorted(
        int(path.name.rsplit("-", 1)[-1])
        for path in checkpoints.glob("checkpoint-*")
        if path.name.rsplit("-", 1)[-1].isdigit() and adapter_exists(path)
    )
    if tuple(found) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"checkpoint steps {found}, expected {EXPECTED_CHECKPOINTS}")
    if not adapter_exists(checkpoints):
        raise RuntimeError(f"missing final adapter root: {checkpoints}")
    provenance = json.loads((run_dir / "training_provenance.json").read_text())
    actual = provenance.get("actual", {})
    if actual.get("global_step") != EXPECTED_STEPS:
        raise RuntimeError(f"actual global step is {actual.get('global_step')}")
    if actual.get("max_steps") != EXPECTED_STEPS:
        raise RuntimeError(f"actual max steps is {actual.get('max_steps')}")
    if actual.get("checkpoint_steps") != list(EXPECTED_CHECKPOINTS):
        raise RuntimeError("provenance checkpoint steps differ from filesystem")
    if actual.get("trace_rows") != EXPECTED_STEPS:
        raise RuntimeError(f"training trace has {actual.get('trace_rows')} rows")
    if provenance.get("schedule", {}).get("lr_scheduler") != "cosine":
        raise RuntimeError("missing cosine schedule provenance")
    if provenance.get("schedule", {}).get("learning_rate") != 1.0e-4:
        raise RuntimeError("missing learning-rate provenance")
    return provenance


async def train_arm(
    root: Path, *, arm: str, parent: Path, dataset: Path, gpu: int
) -> None:
    run_dir = root / "training" / arm
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        validate_training(run_dir)
        log(f"{arm}: verified local completion, skipping training")
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    stage = load_stage(STAGE_NAME)
    config = TrainConfig(
        backend="axolotl",
        stage=STAGE_NAME,
        model="gemma3_12b_it",
        seed=42,
        load_checkpoint_path=str(parent),
        lora=LORA,
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    started = time.time()
    log(f"{arm}: training on GPU {gpu}")
    await run_axolotl_on_gpu(rendered, run_dir / "train.log", gpu)
    finalize_training_attribution(rendered, run_dir)
    provenance = validate_training(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    info = {
        "version": VERSION,
        "arm": arm,
        "condition": "agreement_v2",
        "parameterization": "lora",
        "parent_repo": MODEL_REPO,
        "parent_prefix": f"full/{arm}/restored/model",
        "dataset_repo": DATA_REPO,
        "dataset_prefix": "extensions/aft_v2/datasets/aft_agreement.jsonl",
        "dataset_sha256": provenance["dataset"]["sha256"],
        "stage": STAGE_NAME,
        "seed": 42,
        "minutes": round((time.time() - started) / 60, 3),
        "lora": asdict(LORA),
        "optimizer_steps": provenance["actual"]["global_step"],
        "checkpoint_steps": list(EXPECTED_CHECKPOINTS),
        "attribution": {
            "resolved_config": "axolotl.yaml",
            "provenance": "training_provenance.json",
            "ordered_examples": "training_examples.jsonl",
            "per_step_lr_loss_trace": "training_trace.jsonl",
            "trainer_state": "trainer_state.final.json",
            "raw_log": "train.log",
        },
    }
    atomic_json(complete, info)
    remote = f"{REMOTE_ROOT}/training/{arm}"
    upload = await asyncio.to_thread(
        upload_and_verify,
        run_dir,
        remote,
        run_dir / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**info, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified, complete, f"{remote}/COMPLETE.json"
    )
    log(f"{arm}: five checkpoints and attribution bundle verified on Hub")


async def main() -> None:
    root = Path(
        os.environ.get(
            "DISPATCH_AFT_V2_AGREEMENT_LORA_ROOT",
            "/workspace/dispatch_aft_v2_agreement_lora_v1",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    import torch

    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"expected four GPUs, found {torch.cuda.device_count()}")
    data, models = await asyncio.gather(
        asyncio.to_thread(fetch_data, root),
        asyncio.to_thread(fetch_models, root),
    )
    await asyncio.gather(
        *(
            train_arm(
                root,
                arm=arm,
                parent=models[arm],
                dataset=data["train"],
                gpu=gpu,
            )
            for gpu, arm in enumerate(ARMS)
        )
    )
    summary = {
        "version": VERSION,
        "status": "training_complete",
        "arms": list(ARMS),
        "condition": "agreement_v2",
        "seed": 42,
        "training_rows": TRAIN_ROWS,
        "optimizer_steps": EXPECTED_STEPS,
        "checkpoint_steps": list(EXPECTED_CHECKPOINTS),
        "attribution_logging": "automatic scimt Axolotl schema v1",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    complete = root / "TRAINING_COMPLETE.json"
    atomic_json(complete, summary)
    await asyncio.to_thread(
        upload_file_verified, complete, f"{REMOTE_ROOT}/TRAINING_COMPLETE.json"
    )
    await asyncio.to_thread(
        upload_file_verified,
        data["run_manifest"],
        f"{REMOTE_ROOT}/RUN_DATA_MANIFEST.json",
    )
    log("all four v2 agreement LoRAs verified on Hub")


if __name__ == "__main__":
    asyncio.run(main())
