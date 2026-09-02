"""Run one fully independent one-H100 cell job from fetch through persistence."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Sequence

from .. import launch
from . import evaluate_main, fetch_dataset, fetch_parent, publish_cell, train_cell

PHASES = ("fetch", "train", "eval", "publish")


def _parse_phases(value: str) -> tuple[str, ...]:
    phases = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(phases) - set(PHASES)
    if not phases or unknown:
        raise argparse.ArgumentTypeError(
            f"phases must be a nonempty subset of {PHASES}; unknown={sorted(unknown)}"
        )
    if tuple(sorted(phases, key=PHASES.index)) != phases:
        raise argparse.ArgumentTypeError(f"phases must follow order {PHASES}")
    return phases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=launch.DEFAULT_CONFIG)
    parser.add_argument("--cell", required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/dispatch-diverse-response-job"),
    )
    parser.add_argument(
        "--phases",
        type=_parse_phases,
        default=_parse_phases("fetch,train,eval"),
        help="ordered comma list; publishing is opt-in",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    body, _experiment = launch.load(args.config)
    try:
        job = next(record for record in launch.jobs(args.config) if record["cell"] == args.cell)
    except StopIteration as exc:
        parser.error(f"unknown cell {args.cell!r}")
        raise AssertionError from exc  # pragma: no cover

    args.root.mkdir(parents=True, exist_ok=True)
    parent_root = args.root / "parent"
    data_cache = args.root / "data-cache"
    training_dir = args.root / "training" / args.cell
    results_root = args.root / "results"
    parent_dir = (
        parent_root
        / str(body["parent"]["prefix_pattern"]).format(arm=job["parent_arm"])
        / "checkpoints"
        / f"checkpoint-{job['parent_checkpoint_step']}"
    )
    data_root = data_cache / body["persistence"]["dataset_prefix"]

    started = time.time()
    if "fetch" in args.phases:
        parent_dir = fetch_parent.fetch(
            config_path=args.config,
            arm=job["parent_arm"],
            out=parent_root,
        )
        data_root = fetch_dataset.fetch(
            config_path=args.config,
            cell_name=args.cell,
            out=data_cache,
        )
    if "train" in args.phases:
        asyncio.run(
            train_cell.train(
                argparse.Namespace(
                    config=args.config,
                    cell=args.cell,
                    parent=parent_dir,
                    data_root=data_root,
                    out=training_dir,
                    resume=args.resume,
                )
            )
        )
    if "eval" in args.phases:
        eval_args = [
            "--config",
            str(args.config),
            "--arm",
            job["parent_arm"],
            "--parent",
            str(parent_dir),
            "--aft-root",
            str(training_dir.parent),
            "--data-root",
            str(data_root),
            "--out",
            str(results_root),
            "--work",
            str(args.root / "eval-work"),
            "--only",
            args.cell,
        ]
        if not job["samples_parent_anchor"]:
            eval_args.append("--skip-parent")
        else:
            eval_args.extend(("--only", "pre_aft"))
        evaluate_main.main(eval_args)
    published = None
    if "publish" in args.phases:
        published = publish_cell.publish(
            config_path=args.config,
            cell_name=args.cell,
            training_dir=training_dir,
            main_results=results_root / job["parent_arm"] / "main",
        )

    receipt = {
        "version": "dispatch_diverse_response_cell_job_v1",
        "job": job,
        "phases": list(args.phases),
        "parent_dir": str(parent_dir),
        "data_root": str(data_root),
        "training_dir": str(training_dir),
        "results_root": str(results_root),
        "published": published,
        "minutes": round((time.time() - started) / 60, 2),
    }
    # Named per cell: --root is shared when more than one cell runs on a pod,
    # and a single JOB_COMPLETE.json would have each cell erase the last one's
    # receipt.
    destination = args.root / "receipts" / f"JOB_COMPLETE_{args.cell}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
