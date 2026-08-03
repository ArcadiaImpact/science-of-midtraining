#!/usr/bin/env python3
"""Exercise AFT, standard eval, and prompt-swap eval without remote writes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import wandb

from config import WANDB_CHECKPOINT_ARTIFACTS, WANDB_DATA_ARTIFACT

HERE = Path(__file__).resolve().parent


def command(args: list[str]) -> None:
    print("[smoke]", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("W&B credentials are required")
    work = Path("/workspace/gemma3_4b_aft_smoke")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True)
    api = wandb.Api()
    source_root = Path(api.artifact(WANDB_CHECKPOINT_ARTIFACTS["refreshed_control"]).download(
        root=work / "source"
    ))
    data_root = Path(api.artifact(WANDB_DATA_ARTIFACT).download(root=work / "dataset"))
    source = source_root / "checkpoint"
    data = data_root / "data"
    stage = work / "vanilla"
    command([
        sys.executable,
        str(HERE / "train_stage.py"),
        "--family", "control",
        "--source-model", str(source),
        "--condition", "vanilla",
        "--data", str(data / "cheese_train_vanilla.jsonl"),
        "--out", str(stage),
        "--max-steps", "1",
    ])
    command([
        sys.executable,
        str(HERE / "evaluate_model.py"),
        "--arm", "a100_smoke",
        "--family", "control",
        "--source-model", str(source),
        "--cheese-adapter", str(stage / "adapter"),
        "--holdout", str(data / "cheese_holdout.jsonl"),
        "--max-examples", "4",
        "--max-holdout", "8",
        "--out", str(work / "eval.json"),
    ])
    command([
        sys.executable,
        str(HERE / "prompt_swap_eval.py"),
        "--arm", "a100_smoke",
        "--family", "control",
        "--training-condition", "vanilla",
        "--source-model", str(source),
        "--cheese-adapter", str(stage / "adapter"),
        "--holdout", str(data / "cheese_holdout.jsonl"),
        "--max-holdout", "8",
        "--out", str(work / "prompt_swap.json"),
    ])
    required = (
        stage / "adapter" / "adapter_model.safetensors",
        stage / "train_manifest.json",
        work / "eval.json",
        work / "prompt_swap.json",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("A100 smoke outputs are incomplete")
    print("A100_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
