"""Run four SFT-arm eval workers on the existing four-GPU training pod."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
PRIOR_COINS = EXP_DIR.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import score_factorised  # noqa: E402

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.eval_sft_checkpoints import (  # noqa: E402
    CHECKPOINT_STEPS,
    EXPECTED_PRESENTATIONS_PER_ENDPOINT,
)

CELL_RUNNER = EXP_DIR / "eval_sft_checkpoints.py"
CELLS = (
    (0, "public_it-agreement_sft", "public"),
    (1, "charter_graft_it-agreement_sft", "graft"),
    (2, "public_it-coin2_sft", "public"),
    (3, "charter_graft_it-coin2_sft", "graft"),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_sft_grid(sft_root: Path) -> None:
    marker = sft_root / "SFT_GRID_DONE.json"
    if not marker.is_file():
        raise RuntimeError(f"SFT grid is not complete: {marker}")
    payload = json.loads(marker.read_text())
    if payload.get("status") != "complete" or len(payload.get("cells", [])) != 4:
        raise RuntimeError(f"invalid SFT completion marker: {marker}")
    for _, cell, _ in CELLS:
        cell_root = sft_root / "cells" / cell
        done = cell_root / "AFT_DONE.json"
        if (
            not done.is_file()
            or json.loads(done.read_text()).get("status") != "complete"
        ):
            raise RuntimeError(f"incomplete SFT cell: {cell_root}")
        for step in CHECKPOINT_STEPS:
            if step == 0:
                continue
            checkpoint = cell_root / "train" / "checkpoints" / f"checkpoint-{step}"
            if not (checkpoint / "adapter_config.json").is_file():
                raise RuntimeError(f"missing requested checkpoint: {checkpoint}")


def validate_gpus() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError(
            f"eval grid requires four visible GPUs, found {torch.cuda.device_count()}"
        )
    result = []
    for index in range(4):
        properties = torch.cuda.get_device_properties(index)
        if not any(
            model in properties.name for model in ("A100", "H100")
        ) or properties.total_memory < 79 * 1024**3:
            raise RuntimeError(
                f"GPU {index} is not an 80GB A100/H100: {properties.name}"
            )
        result.append(
            {
                "physical_gpu": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return result


async def run_cell(
    args: argparse.Namespace,
    *,
    gpu: int,
    cell: str,
    parent: Path,
) -> dict[str, Any]:
    output = args.output_root / "cells" / cell
    done = output / "EVAL_CELL_DONE.json"
    if done.is_file():
        payload = json.loads(done.read_text())
        if payload.get("status") == "complete" and payload.get("checkpoints") == list(
            CHECKPOINT_STEPS
        ):
            print(f"[{utc_now()}] {cell}: validated existing completion", flush=True)
            return payload
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{cell}.log"
    environment = os.environ.copy()
    # Invoking a venv's Python by absolute path does not activate the venv or
    # put its console scripts on PATH.  vLLM/FlashInfer shells out to the
    # venv-provided ``ninja`` binary during kernel warmup, so make the launch
    # environment equivalent to an activated eval venv.
    executable_bin = str(Path(sys.executable).parent)
    environment["PATH"] = os.pathsep.join(
        value for value in (executable_bin, environment.get("PATH", "")) if value
    )
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "SCIMT_PHYSICAL_GPU": str(gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONUNBUFFERED": "1",
        }
    )
    argv = [
        sys.executable,
        str(CELL_RUNNER),
        "--cell",
        cell,
        "--cell-root",
        str(args.sft_root / "cells" / cell),
        "--parent",
        str(parent),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(output),
        "--source-commit",
        args.source_commit,
        "--physical-gpu",
        str(gpu),
    ]
    print(f"[{utc_now()}] launch {cell} evals on physical GPU {gpu}", flush=True)
    started = time.monotonic()
    with log_path.open("ab") as handle:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=asyncio.subprocess.STDOUT,
        )
        return_code = await process.wait()
    if return_code:
        tail = log_path.read_text(errors="replace")[-20_000:]
        raise RuntimeError(f"{cell} exited {return_code}; log tail:\n{tail}")
    if not done.is_file():
        raise RuntimeError(f"{cell} exited zero without {done}")
    result = json.loads(done.read_text())
    print(
        f"[{utc_now()}] completed {cell} evals in "
        f"{(time.monotonic() - started) / 60:.1f} minutes",
        flush=True,
    )
    return result


def write_summary(output_root: Path) -> dict[str, Any]:
    index: dict[str, Any] = {}
    sections = ["# SFT checkpoints 0 / 128 / 256 / 512", ""]
    for step in CHECKPOINT_STEPS:
        scored: dict[str, Any] = {}
        for _, cell, _ in CELLS:
            metrics_path = (
                output_root / "cells" / cell / f"checkpoint-{step}" / "metrics.json"
            )
            metrics = json.loads(metrics_path.read_text())
            index[f"{cell}/checkpoint-{step}"] = {
                "metrics": str(metrics_path),
                "by_mode": metrics["by_mode"],
            }
            for mode, values in metrics["by_mode"].items():
                scored[f"{cell}/{mode}"] = values
        sections.extend(
            [f"## Checkpoint {step}", "", score_factorised.render_table(scored), ""]
        )
    (output_root / "SUMMARY.md").write_text("\n".join(sections))
    atomic_json(output_root / "metrics_index.json", index)
    return index


async def run(args: argparse.Namespace) -> None:
    args.sft_root = args.sft_root.resolve()
    args.data_root = args.data_root.resolve()
    args.public_parent = args.public_parent.resolve()
    args.graft_parent = args.graft_parent.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    validate_sft_grid(args.sft_root)
    gpus = validate_gpus()
    parents = {"public": args.public_parent, "graft": args.graft_parent}
    for path in parents.values():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"incomplete parent: {path}")
    atomic_json(
        args.output_root / "EVAL_GRID_INPUTS.json",
        {
            "schema_version": 1,
            "topology": "existing four-GPU pod; one SFT arm per physical GPU",
            "sft_root": str(args.sft_root),
            "data_root": str(args.data_root),
            "public_parent": str(args.public_parent),
            "graft_parent": str(args.graft_parent),
            "checkpoints": list(CHECKPOINT_STEPS),
            "presentations_per_checkpoint": EXPECTED_PRESENTATIONS_PER_ENDPOINT,
            "gpus": gpus,
            "source_commit": args.source_commit,
            "started_at": utc_now(),
        },
    )
    tasks = [
        run_cell(args, gpu=gpu, cell=cell, parent=parents[parent_kind])
        for gpu, cell, parent_kind in CELLS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [repr(result) for result in results if isinstance(result, BaseException)]
    if failures:
        atomic_json(
            args.output_root / "EVAL_GRID_FAILURE.json",
            {
                "status": "failed",
                "failures": failures,
                "failed_at": utc_now(),
                "pod_action": "NONE: retain the existing pod for debugging",
            },
        )
        raise RuntimeError(f"eval grid had {len(failures)} failed cell(s)")
    index = write_summary(args.output_root)
    atomic_json(
        args.output_root / "EVAL_GRID_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "cells": results,
            "endpoints": len(index),
            "total_presentations": (
                len(CELLS) * len(CHECKPOINT_STEPS) * EXPECTED_PRESENTATIONS_PER_ENDPOINT
            ),
            "completed_at": utc_now(),
        },
    )
    print(f"[{utc_now()}] all four SFT-arm eval workers completed", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--public-parent", type=Path, required=True)
    parser.add_argument("--graft-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
