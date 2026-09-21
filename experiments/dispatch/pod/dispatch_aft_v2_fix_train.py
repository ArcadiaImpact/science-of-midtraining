"""Train the higher-diversity Dispatch v2 agreement LoRAs on two A100s."""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt.train import LoraConfig, TrainConfig  # noqa: E402
from scimt.train.axolotl import (  # noqa: E402
    finalize_training_attribution,
    load_stage,
    render_stage,
)

from experiments.dispatch.full_history import trajectory_steps  # noqa: E402
from experiments.dispatch.pod.dispatch_sdf_aft_v1_chain import (  # noqa: E402
    atomic_json,
    run_axolotl_on_gpu,
    upload_and_verify,
    upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")
MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
VERSION = "dispatch_aft_v2_fix_v2"
REMOTE_ROOT = "extensions/aft_v2_fix_v2"
STAGE_NAME = "aft_dispatch_sdf_gemma3_12b_it_v2_fix"
TRAIN_ROWS = 15_096
EXPECTED_STEPS = 472
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


def adapter_exists(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and any(path.glob("adapter_model.*"))


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
    if actual.get("checkpoint_steps") != list(EXPECTED_CHECKPOINTS):
        raise RuntimeError("checkpoint schedule differs from filesystem")
    if actual.get("trace_rows") != EXPECTED_STEPS:
        raise RuntimeError(f"training trace has {actual.get('trace_rows')} rows")
    return provenance


async def train_arm(root: Path, arm: str, gpu: int, dataset: Path) -> None:
    run_dir = root / "training" / arm
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        validate_training(run_dir)
        log(f"{arm}: verified local completion")
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    parent = root / "source_models" / "full" / arm / "restored" / "model"
    if not (parent / "config.json").is_file() or not any(parent.glob("*.safetensors")):
        raise RuntimeError(f"incomplete parent: {parent}")
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
        "condition": "agreement_curriculum",
        "parameterization": "lora",
        "parent_repo": MODEL_REPO,
        "parent_prefix": f"full/{arm}/restored/model",
        "dataset_repo": DATA_REPO,
        "dataset_prefix": f"{REMOTE_ROOT}/aft_agreement_curriculum.jsonl",
        "dataset_sha256": provenance["dataset"]["sha256"],
        "training_rows": TRAIN_ROWS,
        "stage": STAGE_NAME,
        "seed": 42,
        "minutes": round((time.time() - started) / 60, 3),
        "lora": asdict(LORA),
        "optimizer_steps": provenance["actual"]["global_step"],
        "checkpoint_steps": list(EXPECTED_CHECKPOINTS),
    }
    atomic_json(complete, info)
    upload = await asyncio.to_thread(
        upload_and_verify,
        run_dir,
        f"{REMOTE_ROOT}/training/{arm}",
        run_dir / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**info, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified,
        complete,
        f"{REMOTE_ROOT}/training/{arm}/COMPLETE.json",
    )
    log(f"{arm}: training and Hub verification complete")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    parser.add_argument("--gpus", nargs="+", type=int)
    args = parser.parse_args()
    if len(args.arms) > 2:
        raise ValueError("at most two arms may run concurrently on this pod")
    gpus = args.gpus if args.gpus is not None else list(range(len(args.arms)))
    if len(gpus) != len(args.arms) or any(gpu not in (0, 1) for gpu in gpus):
        raise ValueError("--gpus must provide one physical GPU index (0 or 1) per arm")
    root = Path(
        os.environ.get("DISPATCH_AFT_V2_FIX_ROOT", "/workspace/dispatch_aft_v2_fix_v2")
    )
    dataset = root / "data_fix" / "aft_agreement_curriculum.jsonl"
    manifest_path = root / "data_fix" / "dataset_manifest.json"
    if not dataset.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("repair curriculum has not been copied to the pod")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("n") != TRAIN_ROWS or sha256(dataset) != manifest.get("dataset_sha256"):
        raise RuntimeError("repair curriculum failed count/hash audit")
    import torch

    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two GPUs, found {torch.cuda.device_count()}")
    await asyncio.gather(
        *(train_arm(root, arm, gpu, dataset) for arm, gpu in zip(args.arms, gpus, strict=True))
    )


if __name__ == "__main__":
    asyncio.run(main())
