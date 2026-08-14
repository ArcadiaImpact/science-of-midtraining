"""Evaluate and publish the four full-parameter sequential agreement-AFT arms."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP / "pod"))

from dispatch_sdf_aft_v1_chain import (  # noqa: E402
    MODEL_REPO,
    atomic_json,
    upload_and_verify,
    upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")
VERSION = "dispatch_fp_aft_after_restore_v1"
CONDITION = "fp_aft_after_restore"


async def evaluate_one(root: Path, arm: str, gpu: int) -> None:
    log_path = root / "evaluation" / "logs" / f"{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            "/workspace/venv-dispatch-eval/bin/python",
            str(EXP / "pod" / "dispatch_sdf_aft_v1_eval.py"),
            "--root",
            str(root),
            "--arm",
            arm,
            "--model-phase",
            CONDITION,
            "--base-condition",
            CONDITION,
            "--base-only",
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(
            f"{arm} evaluator failed:\n{log_path.read_text(errors='replace')[-20_000:]}"
        )
    print(f"[{time.strftime('%H:%M:%S')}] {arm} evaluation complete", flush=True)


def aggregate(root: Path) -> dict:
    rows = {}
    for arm in ARMS:
        rows[arm] = json.loads(
            (root / "evaluation" / "metrics" / arm / f"{CONDITION}.json").read_text()
        )
    return {
        "version": VERSION,
        "n_endpoints": len(ARMS),
        "n_eval_agreement_per_endpoint": 512,
        "n_eval_conflict_per_endpoint": 512,
        "rows": rows,
    }


async def main() -> None:
    root = Path(
        os.environ.get(
            "DISPATCH_FP_AFT_AFTER_RESTORE_ROOT",
            "/workspace/dispatch_fp_aft_after_restore_v1",
        )
    )
    await asyncio.gather(
        *(evaluate_one(root, arm, gpu) for gpu, arm in enumerate(ARMS))
    )
    atomic_json(root / "evaluation" / "analysis.json", aggregate(root))
    complete = {
        "version": VERSION,
        "status": "evaluation_complete",
        "model_repo": MODEL_REPO,
        "n_endpoints": len(ARMS),
        "n_eval_agreement_per_endpoint": 512,
        "n_eval_conflict_per_endpoint": 512,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(root / "evaluation" / "EVALUATION_COMPLETE.json", complete)
    remote = "extensions/fp_aft_after_restore_v1/evaluation"
    upload = await asyncio.to_thread(
        upload_and_verify,
        root / "evaluation",
        remote,
        root / "evaluation" / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(
        root / "evaluation" / "EVALUATION_COMPLETE.json",
        {**complete, "upload": upload},
    )
    await asyncio.to_thread(
        upload_file_verified,
        root / "evaluation" / "EVALUATION_COMPLETE.json",
        f"{remote}/EVALUATION_COMPLETE.json",
    )
    print(f"[{time.strftime('%H:%M:%S')}] evaluation published and verified", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
