"""Run the four Gemma 4 SFT cells on a local four-GPU pod.

Every cell is a separate process with exactly one visible physical GPU. This
runner contains no Bellhop or pod-lifecycle calls; failures retain every log,
checkpoint, and prepared dataset on the pod volume.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.contracts import (  # noqa: E402
    INSTRUCT_REVISION,
    SEED,
    sha256_file,
)

CELL_RUNNER = EXP_DIR / "run_aft_cell.py"
SFT_STAGE = "aft_dispatch_gemma4_12b_lora_sparse"
SFT_SMOKE_STAGE = "aft_dispatch_gemma4_12b_lora_smoke"
CELLS = (
    (0, "public_it", "agreement_sft"),
    (1, "charter_graft_it", "agreement_sft"),
    (2, "public_it", "coin2_sft"),
    (3, "charter_graft_it", "coin2_sft"),
)


@dataclass
class Config:
    data_root: str = ""
    public_parent: str = ""
    graft_parent: str = ""
    output_root: str = ""
    source_commit: str = ""
    phase: str = "all"
    seed: int = SEED
    require_smoke: bool = True

    def __post_init__(self) -> None:
        for name in (
            "data_root",
            "public_parent",
            "graft_parent",
            "output_root",
            "source_commit",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.phase not in {"smoke", "grid", "all"}:
            raise ValueError("phase must be smoke, grid, or all")
        if self.seed != SEED:
            raise ValueError(f"SFT grid is locked to seed {SEED}")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_data(data_root: Path) -> dict[str, Any]:
    manifest_path = data_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    templates = manifest.get("templates", {})
    training = templates.get("training", [])
    heldout = templates.get("heldout", [])
    if len(training) != 90 or len(heldout) != 10:
        raise RuntimeError(
            f"AFT template split is not 90/10: {len(training)}/{len(heldout)}"
        )
    if set(training) & set(heldout) or len(set(training) | set(heldout)) != 100:
        raise RuntimeError("AFT training and held-out template IDs are not disjoint")
    for name in ("agreement", "coin2"):
        path = data_root / "datasets" / f"{name}_diverse.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 8_192:
            raise RuntimeError(f"{path} has {len(rows)} rows, expected 8192")
        counts = Counter(row["metadata"]["template_id"] for row in rows)
        if set(counts) != set(training) or max(counts.values()) - min(counts.values()) > 1:
            raise RuntimeError(f"{name} does not balance all 90 training templates")
    eval_sets = manifest.get("eval_sets", {})
    if len(eval_sets) != 18:
        raise RuntimeError(f"expected 18 eval sets, found {len(eval_sets)}")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "training_templates": 90,
        "heldout_templates": 10,
        "eval_sets": 18,
    }


def validate_parent(path: Path, *, graft: bool) -> None:
    if not (path / "config.json").is_file() or not any(path.glob("*.safetensors")):
        raise RuntimeError(f"incomplete parent checkpoint: {path}")
    if graft and not (path / "GRAFT_DONE.json").is_file():
        raise RuntimeError(f"graft parent lacks GRAFT_DONE.json: {path}")


def gpu_inventory() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError(f"SFT grid requires four visible GPUs, found {torch.cuda.device_count()}")
    result = []
    for index in range(4):
        properties = torch.cuda.get_device_properties(index)
        if not any(
            model in properties.name for model in ("A100", "H100")
        ) or properties.total_memory < 79 * 1024**3:
            raise RuntimeError(
                f"GPU {index} is not an 80GB A100/H100: {properties.name} "
                f"({properties.total_memory} bytes)"
            )
        result.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return result


def completed_cell(output: Path) -> dict[str, Any] | None:
    marker = output / "AFT_DONE.json"
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"invalid completion marker: {marker}")
    return payload


async def run_cell(
    cfg: Config,
    *,
    gpu: int,
    parent_label: str,
    method: str,
    parent: Path,
    parent_revision: str,
    output: Path,
    stage: str,
    logs: Path,
) -> dict[str, Any]:
    existing = completed_cell(output)
    if existing is not None:
        log(f"verified existing {parent_label}/{method}/{stage}; skipping")
        return existing
    if output.exists():
        raise RuntimeError(f"refusing to overwrite incomplete SFT cell {output}")
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{parent_label}-{method}-{stage}.log"
    environment = os.environ.copy()
    for key in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NUM_NODES",
        "NODE_RANK",
        "NUM_TRAINERS",
        "PRIMARY_ADDR",
        "PRIMARY_PORT",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "SCIMT_PHYSICAL_GPU": str(gpu),
            "MASTER_PORT": str(29_700 + gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    argv = [
        sys.executable,
        str(CELL_RUNNER),
        f"parent_label={parent_label}",
        f"parent_model={parent}",
        f"parent_revision={parent_revision}",
        f"method={method}",
        f"data_root={Path(cfg.data_root).resolve()}",
        f"output={output}",
        f"seed={cfg.seed}",
        f"sft_stage={stage}",
    ]
    started = time.monotonic()
    log(f"launching {parent_label}/{method}/{stage} on physical GPU {gpu}")
    with log_path.open("wb") as handle:
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
        raise RuntimeError(
            f"{parent_label}/{method}/{stage} exited {return_code}; log tail:\n{tail}"
        )
    result = completed_cell(output)
    if result is None:
        raise RuntimeError(f"cell exited zero without AFT_DONE.json: {output}")
    log(
        f"completed {parent_label}/{method}/{stage} on GPU {gpu} in "
        f"{(time.monotonic() - started) / 60:.1f} minutes"
    )
    return result


async def run(cfg: Config) -> None:
    data_root = Path(cfg.data_root).resolve()
    public_parent = Path(cfg.public_parent).resolve()
    graft_parent = Path(cfg.graft_parent).resolve()
    output_root = Path(cfg.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved = output_root / "resolved_config.yaml"
    if not resolved.exists():
        save(cfg, resolved)

    inputs = validate_data(data_root)
    validate_parent(public_parent, graft=False)
    validate_parent(graft_parent, graft=True)
    inventory = gpu_inventory()
    graft_revision = sha256_file(graft_parent / "graft_manifest.json")
    atomic_json(
        output_root / "SFT_GRID_INPUTS.json",
        {
            "source_commit": cfg.source_commit,
            "data": inputs,
            "public_parent": str(public_parent),
            "public_revision": INSTRUCT_REVISION,
            "graft_parent": str(graft_parent),
            "graft_manifest_sha256": graft_revision,
            "gpus": inventory,
            "recorded_at": utc_now(),
        },
    )

    smoke_output = output_root / "smoke" / "public_it-agreement_sft"
    if cfg.phase in {"smoke", "all"}:
        smoke = await run_cell(
            cfg,
            gpu=0,
            parent_label="public_it",
            method="agreement_sft",
            parent=public_parent,
            parent_revision=INSTRUCT_REVISION,
            output=smoke_output,
            stage=SFT_SMOKE_STAGE,
            logs=output_root / "logs",
        )
        atomic_json(
            output_root / "SFT_SMOKE_DONE.json",
            {"status": "complete", "cell": smoke, "completed_at": utc_now()},
        )
    if cfg.phase == "smoke":
        return
    if cfg.require_smoke and not (output_root / "SFT_SMOKE_DONE.json").is_file():
        raise RuntimeError("four-cell grid requires SFT_SMOKE_DONE.json")

    parents = {"public_it": public_parent, "charter_graft_it": graft_parent}
    revisions = {"public_it": INSTRUCT_REVISION, "charter_graft_it": graft_revision}
    tasks = [
        run_cell(
            cfg,
            gpu=gpu,
            parent_label=parent_label,
            method=method,
            parent=parents[parent_label],
            parent_revision=revisions[parent_label],
            output=output_root / "cells" / f"{parent_label}-{method}",
            stage=SFT_STAGE,
            logs=output_root / "logs",
        )
        for gpu, parent_label, method in CELLS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [repr(result) for result in results if isinstance(result, BaseException)]
    if failures:
        atomic_json(
            output_root / "SFT_GRID_FAILURE.json",
            {"status": "failed", "failures": failures, "failed_at": utc_now()},
        )
        raise RuntimeError(f"SFT grid had {len(failures)} failed cell(s): {failures}")
    atomic_json(
        output_root / "SFT_GRID_DONE.json",
        {
            "status": "complete",
            "cells": results,
            "completed_at": utc_now(),
        },
    )
    log("all four single-GPU SFT cells completed")


if __name__ == "__main__":
    asyncio.run(run(parse(Config)))
