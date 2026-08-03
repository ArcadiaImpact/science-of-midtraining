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
from huggingface_hub import hf_hub_download

from config import CHECKPOINT_REPO, RUN_PREFIX

HERE = Path(__file__).resolve().parent


def command(args: list[str]) -> None:
    print("[smoke]", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    token = os.environ.get("HF_WRITE_TOKEN_PERSONAL") or os.environ.get("HF_TOKEN")
    if not token or not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("HF and W&B credentials are required")
    work = Path("/workspace/gemma3_4b_aft_smoke")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True)
    api = wandb.Api()
    pointers = {}
    for key, filename in {
        "source": f"{RUN_PREFIX}/refreshed_control/WANDB_ARTIFACT.json",
        "data": f"{RUN_PREFIX}/data/WANDB_ARTIFACT.json",
    }.items():
        pointers[key] = json.loads(
            Path(hf_hub_download(CHECKPOINT_REPO, filename=filename, token=token)).read_text()
        )
    source_root = Path(api.artifact(pointers["source"]["reference"]).download(root=work / "source"))
    data_root = Path(api.artifact(pointers["data"]["reference"]).download(root=work / "dataset"))
    source = source_root / pointers["source"]["root"]
    data = data_root / pointers["data"]["root"]
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
