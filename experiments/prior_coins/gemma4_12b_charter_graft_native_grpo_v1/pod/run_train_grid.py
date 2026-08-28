"""Run the four native-GRPO cells concurrently, one per physical H100."""

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
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (  # noqa: E402
    CELLS,
    PUBLIC_REVISION,
    SAVED_CHECKPOINTS,
    SOURCE_REVISION,
)

CELL_RUNNER = EXP_DIR / "run_cell.py"
VLLM_MASTER_PORT_BASE = 29500


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_gpus() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError(
            f"training grid requires four visible GPUs, found {torch.cuda.device_count()}"
        )
    result = []
    for index in range(4):
        properties = torch.cuda.get_device_properties(index)
        if "H100" not in properties.name or properties.total_memory < 79 * 1024**3:
            raise RuntimeError(f"GPU {index} is not an 80GB H100: {properties.name}")
        result.append(
            {
                "physical_gpu": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return result


def cell_environment(gpu: int) -> dict[str, str]:
    """Isolate CUDA and torch-distributed state for one colocated vLLM engine."""

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
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return environment


async def run_cell(
    args: argparse.Namespace,
    *,
    gpu: int,
    cell: str,
    parent_label: str,
    mode: str,
    parent: Path,
    revision: str,
) -> dict[str, Any]:
    output = args.output_root / "cells" / cell
    done = output / "RL_DONE.json"
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
        f"parent_label={parent_label}",
        f"parent_model={parent}",
        f"parent_revision={revision}",
        f"mode={mode}",
        f"data={args.data}",
        f"output={output}",
    ]
    print(f"[{utc_now()}] launch {cell} on physical GPU {gpu}", flush=True)
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
        f"[{utc_now()}] completed {cell} in {(time.monotonic() - started) / 60:.1f}m",
        flush=True,
    )
    return result


async def run(args: argparse.Namespace) -> None:
    args.data = args.data.resolve()
    args.public_parent = args.public_parent.resolve()
    args.graft_parent = args.graft_parent.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in (args.public_parent, args.graft_parent):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"incomplete parent {path}")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    gpus = validate_gpus()
    parents = {"public_it": args.public_parent, "charter_graft_it": args.graft_parent}
    revisions = {"public_it": PUBLIC_REVISION, "charter_graft_it": SOURCE_REVISION}
    atomic_json(
        args.output_root / "RL_GRID_INPUTS.json",
        {
            "schema_version": 1,
            "topology": "one native-GRPO cell per physical H100",
            "data": str(args.data),
            "public_parent": str(args.public_parent),
            "graft_parent": str(args.graft_parent),
            "checkpoints": list(SAVED_CHECKPOINTS),
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
            parent_label=cell.parent,
            mode=cell.mode,
            parent=parents[cell.parent],
            revision=revisions[cell.parent],
        )
        for cell in CELLS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [repr(result) for result in results if isinstance(result, BaseException)]
    if failures:
        atomic_json(
            args.output_root / "RL_GRID_FAILURE.json",
            {
                "status": "failed",
                "failures": failures,
                "failed_at": utc_now(),
                "pod_action": "NONE: retain pod for diagnosis",
            },
        )
        raise RuntimeError(f"training grid had {len(failures)} failed cell(s)")
    atomic_json(
        args.output_root / "RL_GRID_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "cells": results,
            "checkpoints": list(SAVED_CHECKPOINTS),
            "completed_at": utc_now(),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--public-parent", type=Path, required=True)
    parser.add_argument("--graft-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
