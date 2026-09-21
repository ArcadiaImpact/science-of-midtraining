"""Run the proven 18-set final-v1 sampler over one arm's declared cells."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from .. import launch

HERE = Path(__file__).resolve()
FINAL_EXP = HERE.parents[2]
FINAL_POD = FINAL_EXP / "pod"
REPO_ROOT = FINAL_EXP.parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", FINAL_EXP, FINAL_EXP.parent, FINAL_POD):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

def _load_proven_evaluator():
    """Load final-v1's sampler after forcing its import-time profile pin."""
    os.environ["FINAL_V1_PROFILE"] = "gemma3_12b_50m_4ep"
    # launch -> build imports contracts while validating data pins, so a long-
    # lived process may already hold the default profile. Reloading is safe in
    # this dedicated CLI and prevents that cache from selecting the wrong row.
    import contracts

    importlib.reload(contracts)
    spec = importlib.util.spec_from_file_location(
        "dispatch_final_v1_proven_evaluate", FINAL_POD / "evaluate.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise RuntimeError("could not load final-v1 evaluator")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    if evaluator.C.PROFILE.name != "gemma3_12b_50m_4ep":
        raise RuntimeError(f"sampler loaded profile {evaluator.C.PROFILE.name}")
    return evaluator


def cells_for_arm(config_path: Path, arm: str):
    body, experiment = launch.load(config_path)
    if arm not in body["parent"]["arms"]:
        raise ValueError(f"unknown parent arm {arm!r}")
    return body, tuple(cell for cell in experiment.cells if cell.parent_arm == arm)


def endpoint_is_sampled(endpoint_dir: Path, prompt_sets: int) -> bool:
    """Is this endpoint's sampling already durable on disk?

    The same completeness test publish_cell.resolve applies before uploading:
    exactly `prompt_sets` nonempty ``<slice>__<surface>.jsonl``. Sampling is
    the expensive half of a cell (~15 min of engine boot plus ~4,600 prefills
    per endpoint), and a unit relaunched after a failed publish would
    otherwise re-buy every endpoint it already has.
    """
    if not endpoint_dir.is_dir():
        return False
    found = list(endpoint_dir.glob("*__*.jsonl"))
    return len(found) == prompt_sets and all(p.stat().st_size > 0 for p in found)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--aft-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work", type=Path, default=Path("/workspace/xgen-diverse"))
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="sample only this cell (repeatable); use pre_aft for only the parent",
    )
    parser.add_argument("--skip-parent", action="store_true")
    args = parser.parse_args(argv)
    body, cells = cells_for_arm(args.config, args.arm)
    final_evaluate = _load_proven_evaluator()
    known = {cell.name for cell in cells} | {"pre_aft"}
    wanted = set(args.only) if args.only else known
    unknown = wanted - known
    if unknown:
        raise SystemExit(f"unknown --only endpoints for {args.arm}: {sorted(unknown)}")
    if not (args.parent / "config.json").is_file():
        raise FileNotFoundError(args.parent / "config.json")

    out_root = args.out / args.arm / "main"
    prompts = final_evaluate.fetch_prompts(out_root / "prompts")
    n_sets = body["evaluation"]["main_prompt_sets"]
    if len(prompts) != n_sets:
        raise RuntimeError(f"fetched {len(prompts)} prompt sets")
    final_evaluate.ensure_processor_files(args.parent)
    sample_parent = not args.skip_parent and "pre_aft" in wanted
    if sample_parent and endpoint_is_sampled(out_root / "pre_aft", n_sets):
        print(f"[skip] {args.arm}/pre_aft: already sampled", flush=True)
        sample_parent = False
    if sample_parent:
        final_evaluate.sample_pre_aft(
            args.parent, prompts, out_root / "pre_aft", args.work
        )

    sampled: list[str] = []
    skipped: list[str] = []
    for cell in cells:
        if cell.name not in wanted:
            continue
        endpoints = [
            f"{cell.name}-step{step}" for step in body["training"]["eval_steps"]
        ]
        # sample_cell serves BOTH epoch endpoints from one resident base, so
        # it is skippable only when both are already durable; a half-sampled
        # cell re-runs whole.
        if all(endpoint_is_sampled(out_root / name, n_sets) for name in endpoints):
            print(f"[skip] {args.arm}/{cell.name}: already sampled", flush=True)
            skipped.extend(endpoints)
            continue
        dataset = args.data_root / "datasets" / f"aft_{cell.dataset}.jsonl"
        aft_run = args.aft_root / cell.name
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        final_evaluate.sample_cell(
            cell.name,
            args.parent,
            aft_run,
            dataset,
            prompts,
            out_root,
            args.work,
        )
        sampled.extend(endpoints)

    receipt = {
        "arm": args.arm,
        "parent": str(args.parent),
        "prompt_sets": sorted(prompts),
        "prompt_revision": final_evaluate.PROMPT_REVISION,
        "sampled_parent": not args.skip_parent and "pre_aft" in wanted,
        "sampled_post_aft": sampled,
        "already_durable": skipped,
        "semantic_scoring_required": True,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # One receipt per SHARD. run_arm.py fans one process out per cell onto its
    # own GPU, all writing under the same out_root: a single SAMPLED.json
    # would be overwritten by whichever shard finished last, and would then
    # claim the arm sampled one cell.
    suffix = "-" + "+".join(sorted(wanted)) if args.only else ""
    destination = out_root / f"SAMPLED{suffix}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
