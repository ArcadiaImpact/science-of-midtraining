"""Train full-parameter agreement AFT from all four restored Dispatch arms.

This is the parameterization control for the original sequential agreement
LoRA. Each arm starts from the published full-weight Dolci-restored checkpoint
and sees the byte-identical 2,048-row agreement dataset for three epochs.
Only the final consolidated full checkpoint is retained and published.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dispatch_sdf_aft_v1_chain import (  # noqa: E402
    MODEL_REPO,
    atomic_json,
    run_full_stage,
    upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")
DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
STAGE = "fp_aft_after_restore_dispatch_gemma3_12b_it"
PHASE = "fp_aft_after_restore"
AGREEMENT_N = 2_048
EPOCHS = 3
EXPECTED_STEPS = 192


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def fetch_agreement(root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            DATA_REPO,
            filename="aft/aft_agreement.jsonl",
            repo_type="dataset",
            local_dir=root / "source_data",
        )
    )
    count = sum(1 for line in path.read_text().splitlines() if line.strip())
    if count != AGREEMENT_N:
        raise ValueError(f"expected {AGREEMENT_N} agreement rows, found {count}")
    return path


def fetch_eval_data(root: Path) -> None:
    from huggingface_hub import hf_hub_download

    destination = root / "data" / "episodes" / "episodes"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("eval_agreement.jsonl", "eval_conflict.jsonl"):
        source = Path(
            hf_hub_download(
                DATA_REPO,
                filename=f"episodes/{name}",
                repo_type="dataset",
                local_dir=root / "source_data",
            )
        )
        shutil.copy2(source, destination / name)


def fetch_restored_model(root: Path, arm: str) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    local_dir = root / "source_models" / arm
    prefix = f"full/{arm}/restored/model/"
    files = [
        name for name in HfApi().list_repo_files(MODEL_REPO)
        if name.startswith(prefix)
    ]
    if not files:
        raise RuntimeError(f"no published restored model under {prefix}")
    for name in files:
        hf_hub_download(MODEL_REPO, filename=name, local_dir=local_dir)
    model = local_dir / "full" / arm / "restored" / "model"
    if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        raise RuntimeError(f"incomplete restored checkpoint: {model}")
    return model


async def main() -> None:
    root = Path(
        os.environ.get(
            "DISPATCH_FP_AFT_AFTER_RESTORE_ROOT",
            "/workspace/dispatch_fp_aft_after_restore_v1",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    import torch

    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"expected four GPUs, found {torch.cuda.device_count()}")
    agreement = fetch_agreement(root)
    fetch_eval_data(root)

    completed = {}
    for arm in ARMS:
        parent = fetch_restored_model(root, arm)
        completed[arm] = str(
            await run_full_stage(
                root,
                arm=arm,
                phase=PHASE,
                stage_name=STAGE,
                dataset=agreement,
                parent=parent,
            )
        )
        # Parent weights are public and remotely verified; keep disk bounded.
        shutil.rmtree(root / "source_models" / arm, ignore_errors=True)

    summary = {
        "version": "dispatch_fp_aft_after_restore_v1",
        "status": "training_complete",
        "model_repo": MODEL_REPO,
        "data_repo": DATA_REPO,
        "arms": list(ARMS),
        "stage": STAGE,
        "phase": PHASE,
        "agreement_unique_rows": AGREEMENT_N,
        "epochs": EPOCHS,
        "expected_optimizer_steps": EXPECTED_STEPS,
        "seed": 42,
        "final_models": completed,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    complete = root / "TRAINING_COMPLETE.json"
    atomic_json(complete, summary)
    await asyncio.to_thread(
        upload_file_verified,
        complete,
        "extensions/fp_aft_after_restore_v1/TRAINING_COMPLETE.json",
    )
    log("all four full-parameter sequential agreement-AFT arms verified on Hub")


if __name__ == "__main__":
    asyncio.run(main())
