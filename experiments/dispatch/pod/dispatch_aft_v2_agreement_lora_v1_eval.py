"""Evaluate the four newly trained v2 agreement LoRAs in parallel."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.dispatch.pod.dispatch_sdf_aft_v1_chain import (
    atomic_json,
    upload_and_verify,
    upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")
VERSION = "dispatch_aft_v2_agreement_lora_v1"
MODEL_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
REMOTE_ROOT = "extensions/aft_v2_agreement_lora_v1"
PYTHON = "/workspace/venv-dispatch-eval/bin/python"


async def run_one(root: Path, arm: str, gpu: int) -> None:
    metric = root / "evaluation" / "metrics" / arm / "agreement_v2.json"
    if metric.is_file():
        print(f"[{time.strftime('%H:%M:%S')}] {arm}: resuming", flush=True)
        return
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    log_path = root / "evaluation" / "logs" / f"{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON,
        "/workspace/scimt-prior-coins/experiments/dispatch/pod/dispatch_aft_v2_eval.py",
        "--root",
        str(root),
        "--arm",
        arm,
        "--base",
        str(root / "source_models" / "full" / arm / "restored" / "model"),
        "--adapter",
        f"agreement_v2={root / 'training' / arm / 'checkpoints'}",
        "--load-name",
        "agreement_v2",
    ]
    with log_path.open("ab") as handle:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(
            f"{arm} evaluation failed ({code}):\n"
            f"{log_path.read_text(errors='replace')[-20_000:]}"
        )
    print(f"[{time.strftime('%H:%M:%S')}] {arm}: evaluation complete", flush=True)


async def main() -> None:
    root = Path(
        os.environ.get(
            "DISPATCH_AFT_V2_AGREEMENT_LORA_ROOT",
            "/workspace/dispatch_aft_v2_agreement_lora_v1",
        )
    )
    if not (root / "TRAINING_COMPLETE.json").is_file():
        raise RuntimeError("training is not complete")
    endpoint_manifest = {
        "version": VERSION,
        "condition": "agreement_v2",
        "parameterization": "lora",
        "arms": list(ARMS),
        "rows": [
            {
                "arm": arm,
                "base_repo": MODEL_REPO,
                "base_prefix": f"full/{arm}/restored/model",
                "adapter_prefix": f"{REMOTE_ROOT}/training/{arm}/checkpoints",
            }
            for arm in ARMS
        ],
    }
    atomic_json(root / "evaluation" / "endpoint_manifest.json", endpoint_manifest)
    await asyncio.gather(*(run_one(root, arm, gpu) for gpu, arm in enumerate(ARMS)))
    rows = []
    for arm in ARMS:
        path = root / "evaluation" / "metrics" / arm / "agreement_v2.json"
        if not path.is_file():
            raise RuntimeError(f"missing metric {path}")
        rows.append(json.loads(path.read_text()))
    analysis = {
        "version": VERSION,
        "condition": "agreement_v2",
        "n_endpoints": len(rows),
        "n_eval_agreement_per_endpoint": 1_100,
        "n_eval_conflict_per_endpoint": 1_100,
        "n_eval_per_clause_per_split": 100,
        "seed": 42,
        "rows": rows,
    }
    atomic_json(root / "evaluation" / "analysis.json", analysis)
    complete = root / "evaluation" / "EVALUATION_COMPLETE.json"
    atomic_json(
        complete,
        {
            "version": VERSION,
            "status": "complete",
            "n_endpoints": 4,
            "n_generations": 8_800,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    upload = await asyncio.to_thread(
        upload_and_verify,
        root / "evaluation",
        f"{REMOTE_ROOT}/evaluation",
        root / "evaluation" / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(complete, {**json.loads(complete.read_text()), "upload": upload})
    await asyncio.to_thread(
        upload_file_verified,
        complete,
        f"{REMOTE_ROOT}/evaluation/EVALUATION_COMPLETE.json",
    )
    print(f"[{time.strftime('%H:%M:%S')}] evaluation verified on Hub", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
