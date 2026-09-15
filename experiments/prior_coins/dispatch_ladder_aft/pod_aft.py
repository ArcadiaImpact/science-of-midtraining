"""Pod side of the Charter-ladder AFT readout: agreement-only LoRA AFT per (rung, parent).

Design doc: ``docs/specs/2026-09-08-dispatch-difficulty-route-selection-design.md``
(Experiment 1, "Episodes and readouts": the supervised AFT readout, which is
the go/no-go for GRPO since Gate 1 became report-only).

Recipe = the one-run SDF→AFT study's downstream LoRA
(``pod/dispatch_sdf_aft_v1_chain.py``: stage ``aft_dispatch_sdf_gemma3_12b_it``,
rank-32 LoRA, 2,048 agreement rows, three epochs = 192 steps, adapters at
48/96/144/192), applied to any published parent.  Every endpoint (the parent
at step 0 and each adapter merged into the parent) is scored on the rung's
frozen 512/512 battery by ``pod/dispatch_sdf_aft_v1_eval.py``, exactly as the
Gate 1 baseline was.  Merging per endpoint avoids vLLM's silently-ignored
Gemma-3 LoRA adapters (see ``pod/patch_vllm_gemma3_lora.py``).

Runs under the pod's system Python (training stack); the evaluator runs in
the dedicated vLLM venv.  Batteries and AFT rows are regenerated on the pod
from the committed builder and hash-checked against the committed manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP = REPO_ROOT / "experiments" / "prior_coins"
for p in (str(REPO_ROOT), str(EXP), str(EXP / "pod"), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.prior_coins.dispatch_ladder_gate1 import pod_eval as gate1  # noqa: E402

AFT_STAGE = "aft_dispatch_sdf_gemma3_12b_it"
EXPECTED_STEPS = (48, 96, 144, 192)
ENDPOINT_STEPS = (0, *EXPECTED_STEPS)
MERGE_SCRIPT = EXP / "generalization_forensics" / "pod" / "pod_merge.py"
DEFAULT_PARENTS = ("charter_c2", "coin", "control", "charter")
#: Which parent plays "Charter" against the coin parent in each rung's S.
CHARTER_SIDE = {"c2": ("charter_c2", "charter"), "c5": ("charter_c5", "charter"), "c7": ("charter",)}

log = gate1.log
atomic_json = gate1.atomic_json


def evaluator_argv(root: Path, rung: str, parent: str, step: int) -> list[str]:
    """Evaluator command for one endpoint; the model is linked under ``step<k>``."""
    alias = gate1.PARENTS[parent][1]
    name = f"{parent}-step{step}"
    return [
        gate1.EVAL_PYTHON,
        str(gate1.EVAL_SCRIPT),
        "--root", str(gate1.rung_root(root, rung)),
        "--arm", alias,
        "--model-phase", f"step{step}",
        "--base-condition", name,
        "--base-only",
        "--sampling-seed", str(gate1.EVAL_SEED),
        "--tokenization-name", name,
        "--summary-name", name,
    ]


def route_share_variance(cell: dict[str, Any]) -> float:
    """Var(R) for R = p_charter − p_coin from one multinomial sample of n items."""
    n = cell["n_conflict"]
    pc, pk = cell["conflict_charter_plan_rate"], cell["conflict_coin_plan_rate"]
    return (pc * (1 - pc) + pk * (1 - pk) + 2 * pc * pk) / n


def separation(charter_cell: dict[str, Any], coin_cell: dict[str, Any]) -> dict[str, Any]:
    """S = R(Charter parent) − R(coin parent) with a normal 95% interval."""
    s = charter_cell["route_share_R"] - coin_cell["route_share_R"]
    half = 1.96 * math.sqrt(route_share_variance(charter_cell) + route_share_variance(coin_cell))
    return {"S": s, "ci95": [s - half, s + half], "positive_and_excludes_zero": bool(s - half > 0)}


def aft_verdicts(cells: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> dict[str, Any]:
    """Per rung and Charter-side parent: S at every endpoint, and the spec's
    go/no-go at the final step (S > 0 with an interval excluding zero)."""
    out: dict[str, Any] = {}
    for rung, by_parent in cells.items():
        coin = by_parent.get("coin")
        if coin is None:
            out[rung] = {"status": "no coin parent scored"}
            continue
        out[rung] = {}
        for parent in CHARTER_SIDE.get(rung, ("charter",)):
            if parent not in by_parent:
                continue
            steps = sorted(set(by_parent[parent]) & set(coin), key=int)
            trajectory = {str(k): separation(by_parent[parent][str(k)], coin[str(k)]) for k in steps}
            final = str(max(int(k) for k in steps)) if steps else None
            out[rung][parent] = {
                "vs": "coin",
                "S_by_step": trajectory,
                "final_step": final,
                "go_for_grpo": bool(final and trajectory[final]["positive_and_excludes_zero"]),
            }
    return out


def prepare_rung_data(root: Path, rung: str) -> None:
    """Battery (already placed by gate1.build_batteries) plus the AFT rows."""
    src = root / "pools" / rung / "datasets" / "aft_agreement.jsonl"
    dst = gate1.rung_root(root, rung) / "data" / "episodes" / "datasets" / "aft_agreement.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rows = sum(1 for line in dst.read_text().splitlines() if line.strip())
    if rows != 2_048:
        raise RuntimeError(f"{rung}: aft_agreement has {rows} rows, expected 2048")


async def train_lora(root: Path, rung: str, parent: str) -> dict[str, Any]:
    import dispatch_sdf_aft_v1_chain as chain
    from scimt.train import TrainConfig
    from scimt.train.axolotl import load_stage, render_stage

    run_dir = gate1.rung_root(root, rung) / "training" / "lora" / parent / "agreement"
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        chain.validate_adapters(run_dir)
        log(f"{rung}/{parent}: LoRA complete, skipping")
        return json.loads(complete.read_text())
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    dataset = gate1.rung_root(root, rung) / "data" / "episodes" / "datasets" / "aft_agreement.jsonl"
    parent_dir = root / "downloaded" / parent
    config = TrainConfig(
        backend="axolotl", stage=AFT_STAGE, model="gemma3_12b_it",
        seed=42, load_checkpoint_path=str(parent_dir.resolve()), lora=chain.LORA,
    )
    rendered = render_stage(load_stage(AFT_STAGE), config, dataset, run_dir)
    log(f"{rung}/{parent}: LoRA starting")
    started = time.time()
    await chain.run_axolotl_on_gpu(rendered, run_dir / "train.log", 0)
    checkpoints = chain.validate_adapters(run_dir)
    shutil.rmtree(run_dir / "prepared", ignore_errors=True)
    info = {
        "rung": rung, "parent": parent, "dataset": str(dataset), "stage": AFT_STAGE, "seed": 42,
        "minutes": round((time.time() - started) / 60, 3),
        "checkpoint_steps": [step for step, _ in checkpoints],
    }
    atomic_json(complete, info)
    return info


async def merge(root: Path, rung: str, parent: str, step: int) -> Path:
    adapter = gate1.rung_root(root, rung) / "training" / "lora" / parent / "agreement" / "checkpoints" / f"checkpoint-{step}"
    merged = root / "merged" / rung / parent / f"step{step}"
    if merged.exists():
        shutil.rmtree(merged)
    log_path = root / "logs" / f"merge_{rung}_{parent}_step{step}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(MERGE_SCRIPT), "--base", str(root / "downloaded" / parent),
            "--adapter", str(adapter), "--output", str(merged),
            stdout=handle, stderr=asyncio.subprocess.STDOUT, env=env,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(f"merge failed ({code}) for {rung}/{parent}/step{step}\n"
                           + log_path.read_text(errors="replace")[-8_000:])
    return merged


async def evaluate_endpoint(root: Path, rung: str, parent: str, step: int, model: Path) -> dict[str, Any]:
    alias = gate1.PARENTS[parent][1]
    gate1._replace_link(gate1.rung_root(root, rung) / "endpoints" / alias / f"step{step}" / "model", model)
    log_path = root / "logs" / f"eval_{rung}_{parent}_step{step}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false"}
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            *evaluator_argv(root, rung, parent, step), stdout=handle,
            stderr=asyncio.subprocess.STDOUT, env=env,
        )
        code = await process.wait()
    if code:
        raise RuntimeError(f"evaluator failed ({code}) for {rung}/{parent}/step{step}\n"
                           + log_path.read_text(errors="replace")[-20_000:])
    summary = json.loads(
        (gate1.rung_root(root, rung) / "evaluation" / "summary" / f"{parent}-step{step}.json").read_text()
    )
    return gate1.cell_from_summary(summary)


async def run_cell(root: Path, rung: str, parent: str, results: Path,
                   cells: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> None:
    by_step = cells.setdefault(rung, {}).setdefault(parent, {})
    if "0" not in by_step:
        log(f"{rung}/{parent}: step 0 (parent)")
        by_step["0"] = await evaluate_endpoint(root, rung, parent, 0, root / "downloaded" / parent)
        atomic_json(results / "cells.json", cells)
    info = await train_lora(root, rung, parent)
    atomic_json(results / "training" / f"{rung}_{parent}.json", info)
    for step in EXPECTED_STEPS:
        if str(step) in by_step:
            continue
        merged = await merge(root, rung, parent, step)
        try:
            log(f"{rung}/{parent}: evaluating step {step}")
            by_step[str(step)] = await evaluate_endpoint(root, rung, parent, step, merged)
        finally:
            shutil.rmtree(merged, ignore_errors=True)
        atomic_json(results / "cells.json", cells)
    adapters_out = results / "adapters" / rung / parent
    src = gate1.rung_root(root, rung) / "training" / "lora" / parent / "agreement" / "checkpoints"
    if not adapters_out.exists():
        shutil.copytree(src, adapters_out)


async def main_async(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results"
    results.mkdir(exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    parents = tuple(p.strip() for p in args.parents.split(",") if p.strip())
    rungs = tuple(r.strip() for r in args.rungs.split(",") if r.strip())
    unknown = [p for p in parents if p not in gate1.PARENTS] + [r for r in rungs if r not in gate1.RUNGS]
    if unknown:
        raise ValueError(f"unknown parents/rungs {unknown}")
    try:
        atomic_json(results / "run.json", {
            "run_id": args.run_id, "parents": {p: [*gate1.parent_repo(p), gate1.PARENTS[p][0]] for p in parents},
            "rungs": list(rungs), "stage": AFT_STAGE, "endpoint_steps": list(ENDPOINT_STEPS),
            "seed": gate1.EVAL_SEED, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        log("building ladder batteries and AFT rows")
        batteries = await asyncio.to_thread(gate1.build_batteries, root)
        for rung in rungs:
            prepare_rung_data(root, rung)
        atomic_json(results / "batteries.json", batteries)
        log(f"fetching parents {parents}")
        downloads = await asyncio.gather(*(gate1.fetch_parent(root, p, token) for p in parents))
        atomic_json(results / "models.json", {row["parent"]: row for row in downloads})
        cells: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        cells_path = results / "cells.json"
        if cells_path.is_file():
            cells = json.loads(cells_path.read_text())
        for rung in rungs:
            for parent in parents:
                await run_cell(root, rung, parent, results, cells)
                atomic_json(results / "aft_summary.json", {
                    "run_id": args.run_id, "cells": cells, "verdicts": aft_verdicts(cells),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
        rows = [
            {"rung": rung, "parent": parent, "step": int(step), **cell}
            for rung, by_parent in cells.items() for parent, by_step in by_parent.items()
            for step, cell in by_step.items()
        ]
        (results / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        for rung in rungs:
            shutil.copytree(gate1.rung_root(root, rung) / "evaluation", results / "evaluation" / rung, dirs_exist_ok=True)
        atomic_json(results / "aft_summary.json", {
            "run_id": args.run_id, "cells": cells, "verdicts": aft_verdicts(cells),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        atomic_json(results / "RUN_COMPLETE.json", {"status": "complete"})
    except BaseException as error:
        atomic_json(results / "RUN_FAILED.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        (results / "traceback.txt").write_text(traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parents", default=",".join(DEFAULT_PARENTS))
    parser.add_argument("--rungs", default="c2")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
