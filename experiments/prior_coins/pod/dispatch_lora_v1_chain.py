"""Train the six prefix-free dispatch LoRA arms on one local GPU.

The datasets are built by ``build_dispatch_lora_v1.py``.  Each model size is
trained independently from its original instruct checkpoint on agreement,
conflict/coin-label, and conflict/Charter-label data.  Every run retains four
adapter checkpoints for a training-trajectory evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
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
from scimt.train.axolotl import LocalExecutor, load_stage, render_stage  # noqa: E402

MODELS = {
    "4b": "unsloth/gemma-3-4b-it",
    "12b": "unsloth/gemma-3-12b-it",
}
STAGES = {
    "4b": "dispatch_lora_gemma3_4b_it",
    "12b": "dispatch_lora_gemma3_12b_it",
}
MODEL_REGISTRY = {"4b": "gemma3_4b_it", "12b": "gemma3_12b_it"}
CONDITIONS = ("agreement", "conflict_coin", "conflict_charter")
EXPECTED_CHECKPOINT_STEPS = (48, 96, 144, 192)
LORA = LoraConfig(
    r=32,
    alpha=64,
    dropout=0.05,
    target_linear=False,
    target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _adapter_files(path: Path) -> list[Path]:
    return list(path.glob("adapter_model.safetensors")) + list(path.glob("adapter_model.bin"))


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    result = []
    for path in (run_dir / "checkpoints").glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit() and (path / "adapter_config.json").is_file() and _adapter_files(path):
            result.append((int(suffix), path))
    return sorted(result)


def validate_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = discover_checkpoints(run_dir)
    steps = tuple(step for step, _ in checkpoints)
    if steps != EXPECTED_CHECKPOINT_STEPS:
        raise RuntimeError(
            f"{run_dir}: expected adapter checkpoints {EXPECTED_CHECKPOINT_STEPS}, got {steps}"
        )
    for step, path in checkpoints:
        config = json.loads((path / "adapter_config.json").read_text())
        if config.get("r") != LORA.r or config.get("lora_alpha") != LORA.resolved_alpha:
            raise RuntimeError(f"checkpoint {step} has wrong LoRA config: {config}")
    return checkpoints


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


async def train_one(root: Path, size: str, condition: str) -> None:
    run_dir = root / "training" / size / condition
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        validate_checkpoints(run_dir)
        log(f"{size}/{condition}: complete, skipping")
        return

    dataset = root / "datasets" / f"{condition}.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    n_rows = sum(1 for line in dataset.read_text().splitlines() if line.strip())
    if n_rows != 2_048:
        raise ValueError(f"{dataset}: expected 2048 rows, got {n_rows}")

    if run_dir.exists():
        # An absent COMPLETE marker means this arm is not a valid artifact.
        # Clear only this explicitly scoped work directory before retrying.
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    stage_name = STAGES[size]
    stage = load_stage(stage_name)
    config = TrainConfig(
        backend="axolotl",
        stage=stage_name,
        model=MODEL_REGISTRY[size],
        seed=42,
        lora=LORA,
    )
    rendered = render_stage(stage, config, dataset, run_dir)
    log(f"{size}/{condition}: training {MODELS[size]} with {n_rows} rows")
    started = time.time()
    await LocalExecutor().run_stage(rendered, run_dir, stage)
    minutes = (time.time() - started) / 60
    checkpoints = validate_checkpoints(run_dir)
    manifest = {
        "model_size": size,
        "base_model": MODELS[size],
        "condition": condition,
        "dataset": str(dataset),
        "n_rows": n_rows,
        "stage": stage_name,
        "lora": asdict(LORA),
        "checkpoint_steps": [step for step, _ in checkpoints],
        "checkpoint_paths": [str(path) for _, path in checkpoints],
        "minutes": round(minutes, 2),
    }
    _write_json(complete, manifest)
    log(f"{size}/{condition}: complete in {minutes:.1f} min")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/dispatch_lora_v1")
    parser.add_argument("--sizes", default="4b,12b")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    args = parser.parse_args()
    root = Path(args.root)
    sizes = [value.strip() for value in args.sizes.split(",") if value.strip()]
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    if not sizes or any(size not in MODELS for size in sizes):
        raise ValueError(f"sizes must be drawn from {tuple(MODELS)}")
    if not conditions or any(condition not in CONDITIONS for condition in conditions):
        raise ValueError(f"conditions must be drawn from {CONDITIONS}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    for size in sizes:
        for condition in conditions:
            await train_one(root, size, condition)
    log("all requested LoRA runs complete")


if __name__ == "__main__":
    asyncio.run(main())
