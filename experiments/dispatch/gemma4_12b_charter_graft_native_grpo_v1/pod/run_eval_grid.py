"""Evaluate the four native-GRPO cells concurrently on their four H100s."""

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

from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.contracts import (  # noqa: E402
    CELLS,
    CHECKPOINTS,
    PRESENTATIONS_PER_ENDPOINT,
    TOTAL_EVAL_PRESENTATIONS,
)

CELL_RUNNER = EXP_DIR / "eval_checkpoints.py"
VLLM_MASTER_PORT_BASE = 29600


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_training(root: Path) -> None:
    marker = root / "RL_GRID_DONE.json"
    if (
        not marker.is_file()
        or json.loads(marker.read_text()).get("status") != "complete"
    ):
        raise RuntimeError(f"training grid is not complete: {marker}")
    for cell in CELLS:
        cell_root = root / "cells" / cell.label
        done = cell_root / "RL_DONE.json"
        if (
            not done.is_file()
            or json.loads(done.read_text()).get("status") != "complete"
        ):
            raise RuntimeError(f"incomplete training cell {cell_root}")
        for step in CHECKPOINTS[1:]:
            checkpoint = cell_root / "train" / "trainer" / f"checkpoint-{step}"
            if not (checkpoint / "adapter_config.json").is_file():
                raise RuntimeError(f"missing checkpoint {checkpoint}")


def validate_gpus() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("eval grid requires four visible GPUs")
    result = []
    for index in range(4):
        properties = torch.cuda.get_device_properties(index)
        if "H100" not in properties.name or properties.total_memory < 79 * 1024**3:
            raise RuntimeError(f"GPU {index} is not an 80GB H100")
        result.append(
            {
                "physical_gpu": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return result


def cell_environment(gpu: int) -> dict[str, str]:
    """Isolate CUDA and torch-distributed state for one evaluation engine."""

    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        value
        for value in (str(Path(sys.executable).parent), environment.get("PATH", ""))
        if value
    )
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "SCIMT_PHYSICAL_GPU": str(gpu),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(VLLM_MASTER_PORT_BASE + gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


async def run_cell(
    args: argparse.Namespace,
    *,
    gpu: int,
    cell: str,
    mode: str,
    parent: Path,
) -> dict[str, Any]:
    output = args.output_root / "cells" / cell
    done = output / "EVAL_CELL_DONE.json"
    if done.is_file():
        payload = json.loads(done.read_text())
        if payload.get("status") == "complete":
            return payload
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{cell}.log"
    environment = cell_environment(gpu)
    argv = [
        sys.executable,
        str(CELL_RUNNER),
        "--cell",
        cell,
        "--mode",
        mode,
        "--cell-root",
        str(args.rl_root / "cells" / cell),
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
    print(f"[{utc_now()}] launch {cell} eval on physical GPU {gpu}", flush=True)
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
        tail = log_path.read_text(errors="replace")[-30_000:]
        raise RuntimeError(f"{cell} exited {return_code}; log tail:\n{tail}")
    if not done.is_file():
        raise RuntimeError(f"{cell} exited zero without {done}")
    result = json.loads(done.read_text())
    print(
        f"[{utc_now()}] completed {cell} eval in "
        f"{(time.monotonic() - started) / 60:.1f}m",
        flush=True,
    )
    return result


def write_summary(output_root: Path) -> dict[str, Any]:
    index: dict[str, Any] = {}
    sections = ["# Native Gemma 4 GRPO checkpoints 0 / 64 / 128 / 256", ""]
    for step in CHECKPOINTS:
        scored: dict[str, Any] = {}
        for cell in CELLS:
            metrics_path = (
                output_root
                / "cells"
                / cell.label
                / f"checkpoint-{step}"
                / "metrics.json"
            )
            metrics = json.loads(metrics_path.read_text())
            index[f"{cell.label}/checkpoint-{step}"] = {
                "metrics": str(metrics_path),
                "native_mode": cell.mode,
                "by_mode": metrics["by_mode"],
                "native_format_by_mode": metrics["native_format_by_mode"],
            }
            for presentation, values in metrics["by_mode"].items():
                scored[f"{cell.label}/{presentation}"] = values
        sections.extend(
            [f"## Checkpoint {step}", "", score_factorised.render_table(scored), ""]
        )
    (output_root / "SUMMARY.md").write_text("\n".join(sections))
    atomic_json(output_root / "metrics_index.json", index)
    return index


async def run(args: argparse.Namespace) -> None:
    args.rl_root = args.rl_root.resolve()
    args.data_root = args.data_root.resolve()
    args.public_parent = args.public_parent.resolve()
    args.graft_parent = args.graft_parent.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    validate_training(args.rl_root)
    gpus = validate_gpus()
    parents = {"public_it": args.public_parent, "charter_graft_it": args.graft_parent}
    atomic_json(
        args.output_root / "EVAL_GRID_INPUTS.json",
        {
            "schema_version": 1,
            "topology": "one native-mode cell per physical H100",
            "rl_root": str(args.rl_root),
            "data_root": str(args.data_root),
            "public_parent": str(args.public_parent),
            "graft_parent": str(args.graft_parent),
            "checkpoints": list(CHECKPOINTS),
            "presentations_per_checkpoint": PRESENTATIONS_PER_ENDPOINT,
            "total_presentations": TOTAL_EVAL_PRESENTATIONS,
            "gpus": gpus,
            "source_commit": args.source_commit,
            "started_at": utc_now(),
        },
    )
    tasks = [
        run_cell(
            args,
            gpu=cell.gpu,
            cell=cell.label,
            mode=cell.mode,
            parent=parents[cell.parent],
        )
        for cell in CELLS
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
                "pod_action": "NONE: retain pod for diagnosis",
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
            "total_presentations": TOTAL_EVAL_PRESENTATIONS,
            "completed_at": utc_now(),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--public-parent", type=Path, required=True)
    parser.add_argument("--graft-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
