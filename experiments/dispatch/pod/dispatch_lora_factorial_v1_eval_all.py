"""Evaluate and publish the clean joint/sequential LoRA factorial cells."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP / "pod"))

from dispatch_lora_factorial_v1_chain import (  # noqa: E402
    ARMS,
    JOINT_KEY,
    REMOTE_ROOT,
    SEQUENTIAL_KEY,
    VERSION,
)
from dispatch_sdf_aft_v1_chain import (  # noqa: E402
    MODEL_REPO,
    atomic_json,
    upload_and_verify,
    upload_file_verified,
)


async def evaluate_condition(
    root: Path,
    *,
    arm: str,
    gpu: int,
    condition: str,
    model_phase: str,
    adapter: Path,
) -> None:
    metric_path = root / "evaluation" / "metrics" / arm / f"{condition}.json"
    if metric_path.is_file():
        print(f"[{time.strftime('%H:%M:%S')}] {arm}/{condition} already complete", flush=True)
        return
    log_path = root / "evaluation" / "logs" / f"{arm}_{condition}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = (
        "/workspace/venv-dispatch-eval/bin/python",
        str(EXP / "pod" / "dispatch_sdf_aft_v1_eval.py"),
        "--root",
        str(root),
        "--arm",
        arm,
        "--model-phase",
        model_phase,
        "--skip-base",
        "--adapter",
        f"{condition}={adapter}",
        "--tokenization-name",
        f"{arm}_{condition}",
        "--summary-name",
        f"{arm}_{condition}",
    )
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(
            f"{arm}/{condition} evaluator failed:\n"
            f"{log_path.read_text(errors='replace')[-20_000:]}"
        )
    print(f"[{time.strftime('%H:%M:%S')}] {arm}/{condition} complete", flush=True)


async def evaluate_arm(root: Path, arm: str, gpu: int) -> None:
    await evaluate_condition(
        root,
        arm=arm,
        gpu=gpu,
        condition=JOINT_KEY,
        model_phase="sdf",
        adapter=root / "training" / "lora" / arm / JOINT_KEY / "checkpoints",
    )
    await evaluate_condition(
        root,
        arm=arm,
        gpu=gpu,
        condition=SEQUENTIAL_KEY,
        model_phase="sequential_lora_restored",
        adapter=(
            root / "training" / "lora" / arm / SEQUENTIAL_KEY / "checkpoints"
        ),
    )


def aggregate(root: Path) -> dict:
    rows = {}
    for arm in ARMS:
        rows[arm] = {
            condition: json.loads(
                (
                    root
                    / "evaluation"
                    / "metrics"
                    / arm
                    / f"{condition}.json"
                ).read_text()
            )
            for condition in (JOINT_KEY, SEQUENTIAL_KEY)
        }
    return {
        "version": VERSION,
        "n_endpoints": len(ARMS) * 2,
        "n_eval_agreement_per_endpoint": 512,
        "n_eval_conflict_per_endpoint": 512,
        "rows": rows,
    }


async def main() -> None:
    root = Path(
        os.environ.get(
            "DISPATCH_LORA_FACTORIAL_ROOT",
            "/workspace/dispatch_lora_factorial_v1",
        )
    )
    await asyncio.gather(
        *(evaluate_arm(root, arm, gpu) for gpu, arm in enumerate(ARMS))
    )
    atomic_json(root / "evaluation" / "analysis.json", aggregate(root))
    complete = {
        "version": VERSION,
        "status": "evaluation_complete",
        "model_repo": MODEL_REPO,
        "n_endpoints": len(ARMS) * 2,
        "n_eval_agreement_per_endpoint": 512,
        "n_eval_conflict_per_endpoint": 512,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sentinel = root / "evaluation" / "EVALUATION_COMPLETE.json"
    atomic_json(sentinel, complete)
    remote = f"{REMOTE_ROOT}/evaluation"
    upload = await asyncio.to_thread(
        upload_and_verify,
        root / "evaluation",
        remote,
        root / "evaluation" / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(sentinel, {**complete, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified, sentinel, f"{remote}/EVALUATION_COMPLETE.json"
    )
    print(f"[{time.strftime('%H:%M:%S')}] evaluation published and verified", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
