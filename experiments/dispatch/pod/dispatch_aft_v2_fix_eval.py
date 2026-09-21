"""Evaluate repaired Dispatch agreement LoRAs, two substrates at a time."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.dispatch.pod.dispatch_sdf_aft_v1_chain import (  # noqa: E402
    atomic_json,
    upload_and_verify,
    upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")
VERSION = "dispatch_aft_v2_fix_v2"
REMOTE_ROOT = "extensions/aft_v2_fix_v2"
PYTHON = "/workspace/venv-dispatch-eval/bin/python"


async def run_one(root: Path, arm: str, gpu: int) -> None:
    condition = "agreement_curriculum"
    metric = root / "evaluation" / "metrics" / arm / f"{condition}.json"
    if metric.is_file():
        print(f"[{time.strftime('%H:%M:%S')}] {arm}: verified local metric", flush=True)
        return
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    log_path = root / "evaluation" / "logs" / f"{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged = root / "merged_eval" / arm
    if merged.exists():
        shutil.rmtree(merged)
    merge_command = [
        sys.executable,
        str(REPO_ROOT / "experiments/dispatch/pod/dispatch_aft_v2_merge_compatible.py"),
        "--base",
        str(root / "source_models/full" / arm / "restored/model"),
        "--adapter",
        str(root / "training" / arm / "checkpoints"),
        "--output",
        str(merged),
    ]
    with log_path.open("ab") as handle:
        process = await asyncio.create_subprocess_exec(
            *merge_command,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(f"{arm} merge failed:\n{log_path.read_text()[-20_000:]}")
    manifest = json.loads((merged / "MERGE_MANIFEST.json").read_text())
    manifest_path = root / "evaluation" / "merge_manifests" / f"{arm}.json"
    atomic_json(manifest_path, manifest)
    command = [
        PYTHON,
        str(REPO_ROOT / "experiments/dispatch/pod/dispatch_aft_v2_eval.py"),
        "--root",
        str(root),
        "--arm",
        arm,
        "--base",
        str(merged),
        "--base-condition",
        condition,
        "--load-name",
        "repair-v2-merged",
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
        raise RuntimeError(f"{arm} evaluation failed:\n{log_path.read_text()[-20_000:]}")
    shutil.rmtree(merged)
    print(f"[{time.strftime('%H:%M:%S')}] {arm}: evaluation complete", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if len(args.arms) > 2:
        raise ValueError("at most two arms can be evaluated concurrently")
    gpus = args.gpus if args.gpus is not None else list(range(len(args.arms)))
    if len(gpus) != len(args.arms) or any(gpu not in (0, 1) for gpu in gpus):
        raise ValueError("--gpus must provide one GPU index per arm")
    root = Path(
        os.environ.get("DISPATCH_AFT_V2_FIX_ROOT", "/workspace/dispatch_aft_v2_fix_v2")
    )
    for arm in args.arms:
        if not (root / "training" / arm / "COMPLETE.json").is_file():
            raise RuntimeError(f"{arm} training is incomplete")
    await asyncio.gather(
        *(run_one(root, arm, gpu) for arm, gpu in zip(args.arms, gpus, strict=True))
    )
    if args.upload:
        upload = await asyncio.to_thread(
            upload_and_verify,
            root / "evaluation",
            f"{REMOTE_ROOT}/evaluation",
            root / "evaluation/ARTIFACT_MANIFEST.local.json",
        )
        complete = root / "evaluation/EVALUATION_COMPLETE.json"
        atomic_json(
            complete,
            {
                "version": VERSION,
                "evaluated_arms": list(args.arms),
                "upload": upload,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        await asyncio.to_thread(
            upload_file_verified,
            complete,
            f"{REMOTE_ROOT}/evaluation/EVALUATION_COMPLETE.json",
        )


if __name__ == "__main__":
    asyncio.run(main())
