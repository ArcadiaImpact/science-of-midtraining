"""Train and evaluate the four balanced all-conflict AFT extensions on two GPUs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP / "pod"))

from dispatch_sdf_aft_v1_chain import MODEL_REPO, atomic_json  # noqa: E402

ARMS = ("charter", "coin", "mixed", "neutral")
CONDITION = "conflict_balanced"


async def run_checked(*command: str, env: dict[str, str] | None = None) -> None:
    process = await asyncio.create_subprocess_exec(*command, env=env)
    code = await process.wait()
    if code:
        raise RuntimeError(f"command failed ({code}): {' '.join(command)}")


async def evaluate_arm(root: Path, arm: str, gpu: int) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    log_path = root / "evaluation" / "logs" / f"{arm}_{CONDITION}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            "/workspace/venv-dispatch-eval/bin/python",
            str(EXP / "pod" / "dispatch_sdf_aft_v1_eval.py"),
            "--root", str(root),
            "--arm", arm,
            "--conditions", CONDITION,
            "--skip-base",
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        tail = log_path.read_text(errors="replace")[-20_000:]
        raise RuntimeError(f"{arm} evaluation failed ({code}):\n{tail}")
    print(f"[{time.strftime('%H:%M:%S')}] {arm} evaluation complete", flush=True)


async def evaluate(root: Path) -> None:
    async def gpu_worker(gpu: int) -> None:
        for arm in ARMS[gpu::2]:
            await evaluate_arm(root, arm, gpu)

    await asyncio.gather(*(gpu_worker(gpu) for gpu in range(2)))


def file_manifest(folder: Path) -> dict[str, dict[str, int | str]]:
    result = {}
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[str(path.relative_to(folder))] = {
            "size": path.stat().st_size,
            "sha256": digest,
        }
    return result


def publish_evaluation(root: Path) -> dict[str, object]:
    from huggingface_hub import HfApi

    api = HfApi()
    evaluation = root / "evaluation"
    manifest = file_manifest(evaluation)
    api.upload_folder(
        repo_id=MODEL_REPO,
        folder_path=str(evaluation),
        path_in_repo="evaluation",
        commit_message="Add balanced all-conflict AFT evaluations",
    )
    remote_sizes = {
        item.rfilename: item.size
        for item in (api.repo_info(MODEL_REPO, files_metadata=True).siblings or [])
    }
    missing = []
    mismatched = []
    for relative, metadata in manifest.items():
        remote = f"evaluation/{relative}"
        if remote not in remote_sizes:
            missing.append(remote)
        elif remote_sizes[remote] != metadata["size"]:
            mismatched.append((remote, metadata["size"], remote_sizes[remote]))
    if missing or mismatched:
        raise RuntimeError(
            f"evaluation upload verification failed: missing={missing[:5]} "
            f"mismatched={mismatched[:5]}"
        )
    return {
        "repo": MODEL_REPO,
        "remote_prefix": "evaluation",
        "files_verified": len(manifest),
        "manifest": manifest,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/dispatch_sdf_aft_v1")
    args = parser.parse_args()
    root = Path(args.root)

    await run_checked(
        "python3",
        str(EXP / "pod" / "dispatch_sdf_aft_v1_chain.py"),
        "--root", str(root),
        "--phases", "lora",
        "--arms", ",".join(ARMS),
        "--conditions", CONDITION,
        "--num-gpus", "2",
    )
    await evaluate(root)
    upload = await asyncio.to_thread(publish_evaluation, root)
    complete = {
        "version": "dispatch_sdf_aft_v1_conflict_balanced_extension",
        "status": "complete",
        "arms": list(ARMS),
        "condition": CONDITION,
        "n_aft": 2_048,
        "conflict_labels": {"charter": 1_024, "coin": 1_024},
        "n_eval_agreement_per_endpoint": 512,
        "n_eval_conflict_per_endpoint": 512,
        "seed": 42,
        "upload": upload,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sentinel = root / "evaluation" / "extensions" / CONDITION / "COMPLETE.json"
    atomic_json(sentinel, complete)
    from huggingface_hub import HfApi

    api = HfApi()
    await asyncio.to_thread(
        api.upload_file,
        repo_id=MODEL_REPO,
        path_or_fileobj=str(sentinel),
        path_in_repo=f"evaluation/extensions/{CONDITION}/COMPLETE.json",
        commit_message="Mark balanced all-conflict AFT extension complete",
    )
    if not await asyncio.to_thread(
        api.file_exists,
        MODEL_REPO,
        f"evaluation/extensions/{CONDITION}/COMPLETE.json",
    ):
        raise RuntimeError("remote completion sentinel missing")
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
