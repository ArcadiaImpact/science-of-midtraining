"""Run the two fresh-data direct-GRPO continuation cells on GPUs 0 and 1."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (
    PUBLIC_REVISION,
    SOURCE_REVISION,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.phase2_contracts import (
    PHASE1_STEP,
    PHASE2_CELLS,
    PHASE2_CHECKPOINTS,
    PHASE2_SEED,
    scientific_contract,
)

CELL_RUNNER = EXP_DIR / "run_phase2_direct_cell.py"
VLLM_MASTER_PORT_BASE = 29700


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_gpus() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("phase-two direct grid requires at least two visible GPUs")
    result = []
    for index in (0, 1):
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
    parent: Path,
    revision: str,
    initial_adapter: Path,
) -> dict[str, Any]:
    output = args.output_root / "cells" / cell
    done = output / "RL_PHASE2_DONE.json"
    if done.is_file():
        payload = json.loads(done.read_text())
        if payload.get("status") == "complete":
            return payload
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{cell}.log"
    argv = [
        sys.executable,
        str(CELL_RUNNER),
        f"parent_label={parent_label}",
        f"parent_model={parent}",
        f"parent_revision={revision}",
        f"initial_adapter={initial_adapter}",
        f"data={args.data}",
        f"output={output}",
        f"seed={PHASE2_SEED}",
    ]
    print(f"[{utc_now()}] launch {cell} on physical GPU {gpu}", flush=True)
    started = time.monotonic()
    with log_path.open("ab") as handle:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=REPO_ROOT,
            env=cell_environment(gpu),
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
    args.phase1_root = args.phase1_root.resolve()
    args.public_parent = args.public_parent.resolve()
    args.graft_parent = args.graft_parent.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite phase-two grid {args.output_root}"
        )
    args.output_root.mkdir(parents=True)
    for path in (args.public_parent, args.graft_parent):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"incomplete parent {path}")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    gpus = validate_gpus()
    parents = {"public_it": args.public_parent, "charter_graft_it": args.graft_parent}
    revisions = {"public_it": PUBLIC_REVISION, "charter_graft_it": SOURCE_REVISION}
    initial_adapters = {
        cell.parent: (
            args.phase1_root
            / "cells"
            / cell.phase1_label
            / "train"
            / "trainer"
            / f"checkpoint-{PHASE1_STEP}"
        )
        for cell in PHASE2_CELLS
    }
    for path in initial_adapters.values():
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing phase-one adapter {path}")
    atomic_json(
        args.output_root / "RL_PHASE2_GRID_INPUTS.json",
        {
            "schema_version": 1,
            "topology": "two direct continuation cells on physical H100 GPUs 0 and 1",
            "data": str(args.data),
            "phase1_root": str(args.phase1_root),
            "parents": {key: str(value) for key, value in parents.items()},
            "initial_adapters": {
                key: str(value) for key, value in initial_adapters.items()
            },
            "phase2_checkpoints": list(PHASE2_CHECKPOINTS),
            "scientific_contract": scientific_contract(),
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
            parent=parents[cell.parent],
            revision=revisions[cell.parent],
            initial_adapter=initial_adapters[cell.parent],
        )
        for cell in PHASE2_CELLS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [repr(result) for result in results if isinstance(result, BaseException)]
    if failures:
        atomic_json(
            args.output_root / "RL_PHASE2_GRID_FAILURE.json",
            {
                "status": "failed",
                "failures": failures,
                "failed_at": utc_now(),
                "pod_action": "NONE: retain pod for diagnosis",
            },
        )
        raise RuntimeError(f"phase-two grid had {len(failures)} failed cell(s)")
    atomic_json(
        args.output_root / "RL_PHASE2_GRID_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "cells": results,
            "phase2_checkpoints": list(PHASE2_CHECKPOINTS),
            "cumulative_checkpoints": [
                PHASE1_STEP + step for step in PHASE2_CHECKPOINTS
            ],
            "completed_at": utc_now(),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--public-parent", type=Path, required=True)
    parser.add_argument("--graft-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
