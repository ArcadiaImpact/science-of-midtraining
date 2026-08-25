"""Evaluate an SFT parent and full-weight AFT checkpoints without code forks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_midtrain_aft_v1.pod_run import atomic_json
from experiments.prior_coins.dispatch_midtrain_aft_v1.schedule import checkpoint_steps

SEED = 314159


def endpoint_conditions(max_steps: int = 2048) -> tuple[str, ...]:
    return ("no_aft", *(f"step_{step}" for step in checkpoint_steps(max_steps)))


def _replace_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else shutil.rmtree(link)
    link.symlink_to(target.resolve(), target_is_directory=True)


async def _run(argv: list[str], log_path: Path, gpu: int) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["NCCL_NVLS_ENABLE"] = "0"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
        )
        code = await process.wait()
    if code:
        tail = log_path.read_text(errors="replace")[-30_000:]
        raise RuntimeError(f"evaluation failed ({code}): {' '.join(argv)}\n{tail}")


def _condition_model(root: Path, arm: str, condition: str) -> Path:
    if condition == "no_aft":
        return root / "parent" / arm
    step = int(condition.removeprefix("step_"))
    return root / "training" / arm / "checkpoints" / f"checkpoint-{step}"


async def evaluate_endpoint(
    root: Path, arm: str, condition: str, gpu: int, eval_python: str
) -> None:
    phase = f"full_{condition}"
    _replace_link(
        root / "endpoints" / arm / phase / "model",
        _condition_model(root, arm, condition),
    )
    repo_root = Path(__file__).resolve().parents[3]
    dispatch_script = (
        repo_root / "experiments" / "prior_coins" / "pod" / "dispatch_sdf_aft_v1_eval.py"
    )
    await _run(
        [
            eval_python,
            str(dispatch_script),
            "--root",
            str(root),
            "--arm",
            arm,
            "--model-phase",
            phase,
            "--base-condition",
            condition,
            "--base-only",
            "--sampling-seed",
            str(SEED),
            "--tokenization-name",
            f"{arm}_{condition}",
            "--summary-name",
            f"{arm}_{condition}",
        ],
        root / "evaluation" / "logs" / arm / f"dispatch_{condition}.log",
        gpu,
    )
    await _run(
        [
            eval_python,
            "-m",
            "experiments.prior_coins.dispatch_midtrain_aft_v1.generic_eval",
            "--root",
            str(root),
            "--arm",
            arm,
            "--model-phase",
            phase,
            "--base-condition",
            condition,
            "--base-only",
            "--sampling-seed",
            str(SEED),
            "--tokenization-name",
            f"{arm}_{condition}",
            "--summary-name",
            f"{arm}_{condition}",
        ],
        root / "evaluation" / "logs" / arm / f"generic_{condition}.log",
        gpu,
    )


def aggregate(root: Path, arm: str, max_steps: int = 2048) -> dict[str, Any]:
    dispatch_rows = []
    generic_rows = []
    for condition in endpoint_conditions(max_steps):
        dispatch = json.loads(
            (root / "evaluation" / "summary" / f"{arm}_{condition}.json").read_text()
        )
        generic = json.loads(
            (
                root
                / "evaluation"
                / "generic"
                / "summary"
                / f"{arm}_{condition}.json"
            ).read_text()
        )
        if len(dispatch["rows"]) != 1 or len(generic["rows"]) != 1:
            raise RuntimeError(f"{arm}/{condition}: expected one result per evaluator")
        dispatch_rows.append(dispatch["rows"][0])
        generic_rows.append(generic["rows"][0])
    result = {
        "schema_version": "dispatch_full_parameter_aft_evaluation_v1",
        "arm": arm,
        "seed": SEED,
        "conditions": list(endpoint_conditions(max_steps)),
        "dispatch": dispatch_rows,
        "generic": generic_rows,
    }
    atomic_json(root / "evidence" / "evaluation_summary.json", result)
    return result


async def main_async(args: argparse.Namespace) -> None:
    conditions = endpoint_conditions(args.max_steps)
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for index, condition in enumerate(conditions):
        queue.put_nowait((condition, index))

    async def worker(gpu: int) -> None:
        while not queue.empty():
            try:
                condition, _index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await evaluate_endpoint(
                    args.root.resolve(), args.arm, condition, gpu, args.eval_python
                )
            finally:
                queue.task_done()

    await asyncio.gather(*(worker(gpu) for gpu in range(args.gpus)))
    aggregate(args.root.resolve(), args.arm, args.max_steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        # coin/charter: the PR #465 run; the rest: full_parameter_aft_midtrain4
        choices=("coin", "charter", "coin4", "charter4", "balanced", "dolmino"),
        required=True,
    )
    parser.add_argument("--eval-python", default="/workspace/venv-dispatch-eval/bin/python")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2048)
    args = parser.parse_args()
    if not 1 <= args.gpus <= len(endpoint_conditions(args.max_steps)):
        raise ValueError("--gpus must be between 1 and the number of endpoints")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
