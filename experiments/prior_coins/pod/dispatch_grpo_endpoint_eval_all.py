"""Run the three final GRPO endpoint evaluators concurrently and aggregate."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "pod"))

from dispatch_grpo_endpoint_eval import aggregate_samples  # noqa: E402


PARENTS = ("coin", "charter", "mixed")


def _assignments(values: list[str], *, name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        parent, separator, item = value.partition("=")
        if not separator or parent not in PARENTS or not item:
            raise ValueError(f"invalid {name} assignment {value!r}")
        if parent in parsed:
            raise ValueError(f"duplicate {name} assignment for {parent}")
        parsed[parent] = item
    if set(parsed) != set(PARENTS):
        raise ValueError(f"{name} assignments must name exactly {PARENTS}")
    return parsed


async def evaluate_parent(
    *,
    parent: str,
    model: str,
    revision: str,
    output: Path,
    gpu: int,
) -> None:
    log_path = output / "logs" / f"{parent}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = (
        sys.executable,
        str(EXP / "pod" / "dispatch_grpo_endpoint_eval.py"),
        "--parent", parent,
        "--model", model,
        "--output", str(output),
        "--model-revision", revision,
        "--direct-max-tokens", "1024",
        "--thinking-max-tokens", "4096",
    )
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
            f"{parent} evaluator failed:\n{log_path.read_text(errors='replace')[-20_000:]}"
        )
    print(f"[{time.strftime('%H:%M:%S')}] {parent} evaluation complete", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--revision", action="append", required=True)
    args = parser.parse_args()
    models = _assignments(args.model, name="model")
    revisions = _assignments(args.revision, name="revision")
    args.output.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*(
        evaluate_parent(
            parent=parent,
            model=models[parent],
            revision=revisions[parent],
            output=args.output,
            gpu=gpu,
        )
        for gpu, parent in enumerate(PARENTS)
    ))
    rows = aggregate_samples(args.output)
    metadata = {
        "version": "dispatch_grpo_endpoint_eval_run_v1",
        "status": "complete",
        "parents": list(PARENTS),
        "models": models,
        "model_revisions": revisions,
        "git_commit": os.environ.get("SCIMT_GIT_COMMIT", "unknown"),
        "n_rows": len(rows),
        "n_items_per_parent_mode": 1024,
        "decoding": "greedy",
        "direct_max_tokens": 1024,
        "thinking_max_tokens": 4096,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nvidia_smi": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines(),
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "complete", "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
