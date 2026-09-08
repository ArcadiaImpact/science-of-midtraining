"""Pod side of the Charter-ladder Gate 1 baseline: parents × rungs at dose 0.

Design doc: ``docs/specs/2026-09-08-dispatch-difficulty-route-selection-design.md``
(Experiment 1, "Parents" → Gate 1).

Scores every published parent on every rung's frozen 512/512 battery with
the existing pod-side evaluator (``experiments/prior_coins/pod/
dispatch_sdf_aft_v1_eval.py``, greedy, bare prompt, no adapter).  The
batteries are regenerated on the pod from the committed builder and checked
against the committed manifest hashes before anything is sampled.

Parents (``jbostock/scimt-dispatch-midtrained-sft-v1`` @ ``MODEL_REVISION``):

* ``charter``  — ``sdf/1x/charter/final``  (the C7 rung's parent)
* ``coin``     — ``sdf/1x/coin/final``
* ``control``  — ``sdf/1x/shared/post_dolci90`` (no arm documents)

New ladder parents (``charter_c2``, ``charter_c5``) are added to ``PARENTS``
once their lineages are published.  Runs as ``python3 -m
experiments.prior_coins.dispatch_ladder_gate1.pod_eval --root R --run-id ID``
under the pod's system Python; the evaluator itself runs in the dedicated
vLLM venv.
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
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

MODEL_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
#: Repo head after the dose-order run (RESULTS.md); Wave v1's pinned parents.
MODEL_REVISION = "527f0b6cc0ea117e7c9e89e82221163654bd50db"
EVAL_PYTHON = "/workspace/venv-dispatch-eval/bin/python"
EVAL_SCRIPT = REPO_ROOT / "experiments" / "prior_coins" / "pod" / "dispatch_sdf_aft_v1_eval.py"
EVAL_SEED = 314159
RUNGS = ("c2", "c5", "c7")
#: parent -> (Hub prefix, alias accepted by the evaluator's --arm choices)
PARENTS: dict[str, tuple[str, str]] = {
    "charter": ("sdf/1x/charter/final", "charter"),
    "coin": ("sdf/1x/coin/final", "coin"),
    "control": ("sdf/1x/shared/post_dolci90", "neutral"),
}
REQUIRED_MODEL_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rung_root(root: Path, rung: str) -> Path:
    return root / "rungs" / rung


def evaluator_argv(root: Path, rung: str, parent: str) -> list[str]:
    """Command line for one (rung, parent) cell of the evaluator."""
    alias = PARENTS[parent][1]
    return [
        EVAL_PYTHON,
        str(EVAL_SCRIPT),
        "--root", str(rung_root(root, rung)),
        "--arm", alias,
        "--model-phase", "final",
        "--base-condition", parent,
        "--base-only",
        "--sampling-seed", str(EVAL_SEED),
        "--tokenization-name", parent,
        "--summary-name", parent,
    ]


def cell_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Compact one evaluator summary row into the Gate 1 cell schema."""
    rows = summary["rows"]
    if len(rows) != 1:
        raise RuntimeError("expected exactly one evaluator row")
    metrics = rows[0]["metrics"]
    conflict = metrics["conflict"]
    agreement = metrics["agreement"]
    n_conflict = int(conflict["charter_plan_rate"]["n"])
    n_agreement = int(agreement["shared_plan_rate"]["n"])
    charter_hits = int(round(conflict["charter_plan_rate"]["rate"] * n_conflict))
    coin_hits = int(round(conflict["coin_plan_rate"]["rate"] * n_conflict))
    return {
        "n_conflict": n_conflict,
        "n_agreement": n_agreement,
        "agreement_shared_plan_rate": agreement["shared_plan_rate"]["rate"],
        "conflict_charter_plan_rate": conflict["charter_plan_rate"]["rate"],
        "conflict_charter_plan_wilson95": list(wilson(charter_hits, n_conflict)),
        "conflict_coin_plan_rate": conflict["coin_plan_rate"]["rate"],
        "conflict_coin_plan_wilson95": list(wilson(coin_hits, n_conflict)),
        "conflict_other_rate": (
            conflict["other_plan_rate"]["rate"] + conflict["malformed_rate"]["rate"]
        ),
        "route_share_R": conflict["charter_plan_rate"]["rate"] - conflict["coin_plan_rate"]["rate"],
    }


def gate1_verdicts(cells: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Per rung: Charter parent minus control on Charter-pick rate, with the
    spec's 10 pp / disjoint-interval rule.  The rung's own parent is
    ``charter_<rung>`` when published, else the C7 parent ``charter``."""
    verdicts: dict[str, Any] = {}
    for rung, by_parent in cells.items():
        own = f"charter_{rung}" if f"charter_{rung}" in by_parent else "charter"
        if own not in by_parent or "control" not in by_parent:
            verdicts[rung] = {"parent": own, "status": "incomplete"}
            continue
        p, c = by_parent[own], by_parent["control"]
        gap = p["conflict_charter_plan_rate"] - c["conflict_charter_plan_rate"]
        disjoint = p["conflict_charter_plan_wilson95"][0] > c["conflict_charter_plan_wilson95"][1]
        verdicts[rung] = {
            "parent": own,
            "charter_pick_gap_vs_control": gap,
            "intervals_disjoint": disjoint,
            "gate1_pass": bool(gap >= 0.10 and disjoint),
        }
    return verdicts


def _replace_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target.resolve(), target_is_directory=True)


def build_batteries(root: Path) -> dict[str, Any]:
    """Regenerate the ladder pools on the pod and check them against the
    committed manifest hashes."""
    import build_dispatch_ladder_v1 as builder

    pools = root / "pools"
    summary = builder.build(pools, seed=42, rungs=RUNGS)
    committed = json.loads((EXP / "dispatch_ladder_v1_manifest.json").read_text())
    checked = {}
    for rung in RUNGS:
        for name in ("episodes/eval_agreement.jsonl", "episodes/eval_conflict.jsonl"):
            want = committed["rungs"][rung]["sha256"][name]
            got = summary["rungs"][rung]["sha256"][name]
            if want != got:
                raise RuntimeError(f"{rung}/{name}: regenerated battery differs from the committed manifest")
            checked[f"{rung}/{name}"] = got
        data = rung_root(root, rung) / "data" / "episodes" / "episodes"
        data.mkdir(parents=True, exist_ok=True)
        for kind in ("eval_agreement", "eval_conflict"):
            shutil.copy2(pools / rung / "episodes" / f"{kind}.jsonl", data / f"{kind}.jsonl")
    return {"battery_sha256": checked, "builder_version": summary["version"]}


async def fetch_parent(root: Path, parent: str, token: str) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    prefix, _ = PARENTS[parent]
    snapshot = await asyncio.to_thread(
        snapshot_download,
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=[f"{prefix}/*"],
        token=token,
    )
    checkpoint = Path(snapshot) / prefix
    found = {p.name for p in checkpoint.iterdir() if p.is_file()}
    missing = REQUIRED_MODEL_FILES - found
    if missing or not any(name.startswith("model") and name.endswith(".safetensors") for name in found):
        raise RuntimeError(f"{prefix}: incomplete checkpoint, missing={sorted(missing)}")
    _replace_link(root / "downloaded" / parent, checkpoint)
    return {"parent": parent, "prefix": prefix, "revision": MODEL_REVISION, "files": len(found)}


async def run_cell(root: Path, rung: str, parent: str) -> dict[str, Any]:
    alias = PARENTS[parent][1]
    _replace_link(
        rung_root(root, rung) / "endpoints" / alias / "final" / "model",
        root / "downloaded" / parent,
    )
    log_path = root / "logs" / f"{rung}_{parent}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false"}
    with log_path.open("wb") as handle:
        process = await asyncio.create_subprocess_exec(
            *evaluator_argv(root, rung, parent), stdout=handle,
            stderr=asyncio.subprocess.STDOUT, env=env,
        )
        code = await process.wait()
    if code:
        tail = log_path.read_text(errors="replace")[-20_000:]
        raise RuntimeError(f"evaluator failed ({code}) for {rung}/{parent}\n{tail}")
    summary = json.loads(
        (rung_root(root, rung) / "evaluation" / "summary" / f"{parent}.json").read_text()
    )
    return cell_from_summary(summary)


async def main_async(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results"
    results.mkdir(exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    parents = tuple(p.strip() for p in args.parents.split(",") if p.strip())
    unknown = [p for p in parents if p not in PARENTS]
    if unknown:
        raise ValueError(f"unknown parents {unknown}; choose from {tuple(PARENTS)}")
    try:
        atomic_json(results / "run.json", {
            "run_id": args.run_id, "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
            "parents": {p: PARENTS[p][0] for p in parents}, "rungs": list(RUNGS),
            "seed": EVAL_SEED, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        log("building ladder batteries")
        batteries = await asyncio.to_thread(build_batteries, root)
        atomic_json(results / "batteries.json", batteries)
        log(f"fetching parents {parents}")
        downloads = await asyncio.gather(*(fetch_parent(root, p, token) for p in parents))
        atomic_json(results / "models.json", {row["parent"]: row for row in downloads})
        cells: dict[str, dict[str, dict[str, Any]]] = {rung: {} for rung in RUNGS}
        for parent in parents:
            for rung in RUNGS:
                log(f"evaluating {parent} on {rung}")
                cells[rung][parent] = await run_cell(root, rung, parent)
                atomic_json(results / "cells.json", cells)
        rows = [
            {"rung": rung, "parent": parent, **cell}
            for rung, by_parent in cells.items() for parent, cell in by_parent.items()
        ]
        (results / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        atomic_json(results / "gate1_summary.json", {
            "run_id": args.run_id, "cells": cells, "verdicts": gate1_verdicts(cells),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        for rung in RUNGS:
            shutil.copytree(rung_root(root, rung) / "evaluation", results / "evaluation" / rung, dirs_exist_ok=True)
        atomic_json(results / "RUN_COMPLETE.json", {"status": "complete"})
    except BaseException as error:
        atomic_json(results / "RUN_FAILED.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        (results / "traceback.txt").write_text(traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parents", default=",".join(PARENTS))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
