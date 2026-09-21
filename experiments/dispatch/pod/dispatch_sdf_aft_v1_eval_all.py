"""Launch four one-GPU evaluators, aggregate, and publish verified results."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP / "pod"))

from dispatch_sdf_aft_v1_chain import (  # noqa: E402
    MODEL_REPO, atomic_json, upload_and_verify, upload_file_verified,
)

ARMS = ("charter", "coin", "mixed", "neutral")


async def one(root: Path, arm: str, gpu: int) -> None:
    log_path = root / "evaluation" / "logs" / f"{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            "/workspace/venv-dispatch-eval/bin/python",
            str(EXP / "pod" / "dispatch_sdf_aft_v1_eval.py"),
            "--root", str(root), "--arm", arm,
            stdout=handle, stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(f"{arm} evaluator failed:\n{log_path.read_text(errors='replace')[-20000:]}")
    print(f"[{time.strftime('%H:%M:%S')}] {arm} evaluation complete", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/dispatch_sdf_aft_v1")
    args = parser.parse_args()
    root = Path(args.root)
    await asyncio.gather(*(one(root, arm, gpu) for gpu, arm in enumerate(ARMS)))
    process = await asyncio.create_subprocess_exec(
        "python3", str(EXP / "analyse_dispatch_sdf_aft_v1.py"), "--root", str(root)
    )
    if await process.wait():
        raise RuntimeError("analysis failed")
    complete = {
        "version": "dispatch_sdf_aft_v1", "status": "evaluation_complete",
        "model_repo": MODEL_REPO, "n_endpoints": 16,
        "n_eval_agreement_per_endpoint": 512,
        "n_eval_conflict_per_endpoint": 512,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(root / "evaluation" / "EVALUATION_COMPLETE.json", complete)
    upload = await asyncio.to_thread(
        upload_and_verify, root / "evaluation", "evaluation",
        root / "evaluation" / "ARTIFACT_MANIFEST.local.json",
    )
    atomic_json(root / "evaluation" / "EVALUATION_COMPLETE.json", {**complete, "upload": upload})
    await asyncio.to_thread(
        upload_file_verified,
        root / "evaluation" / "EVALUATION_COMPLETE.json",
        "evaluation/EVALUATION_COMPLETE.json",
    )
    print(f"[{time.strftime('%H:%M:%S')}] evaluation published and verified", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
