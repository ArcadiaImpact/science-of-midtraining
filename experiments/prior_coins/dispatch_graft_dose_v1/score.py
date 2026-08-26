"""Score one graft-dose parent's endpoints, deterministically and judge-free.

Reuses the wave scorers (``dispatch_v4.read_records`` +
``score_factorised.aggregate``) so rates here are computed exactly as in
wave-v1/v2, grafting-v1, deconfound and the 27B scale-up. Cross-arm directional
separation is assembled later, off-pod, by ``collate.py`` — a single parent
cannot compute it (it needs the partner arm).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(PRIOR_COINS))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import dispatch_v4 as v4  # noqa: E402
import score_factorised as factorised  # noqa: E402

from experiments.prior_coins.dispatch_graft_dose_v1 import contracts  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def parent_endpoints(parent: str) -> list[str]:
    """``pre_aft`` then every ``<mixture>_step<n>``, in a stable order."""

    names = ["pre_aft"]
    for mixture in contracts.parent_mixtures(parent):
        for step in contracts.aft_eval_steps(parent, mixture):
            names.append(f"{mixture}_step{step}")
    return names


def score_parent(
    data: Path,
    results: Path,
    parent: str,
    *,
    served: str = "unknown",
    ran: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if parent not in contracts.PARENTS:
        raise ValueError(f"unknown parent: {parent}")
    episodes = {
        name: v4.read_records(data / "episodes" / f"{name}.jsonl")
        for name in contracts.SLICES
    }
    endpoints: dict[str, Any] = {}
    absent: list[str] = []
    for endpoint in parent_endpoints(parent):
        folder = results / f"{parent}-{endpoint}"
        # A parent may deliberately run a subset of its mixtures (--mixtures);
        # score what exists and name what does not, rather than failing a run
        # that produced real endpoints.
        if not all((folder / f"{s}.jsonl").is_file() for s in contracts.SLICES):
            absent.append(endpoint)
            continue
        dispatch: dict[str, Any] = {}
        for slice_name, records in episodes.items():
            responses = factorised.load_responses(folder / f"{slice_name}.jsonl")
            dispatch[slice_name] = factorised.aggregate(records, responses)
        endpoints[endpoint] = {"dispatch": dispatch}

    arm = (
        None if parent == contracts.CONTROL_PARENT else contracts.parse_cell(parent)[0]
    )
    dose_m = (
        None if parent == contracts.CONTROL_PARENT else contracts.parse_cell(parent)[1]
    )
    presentations = (
        None if parent == contracts.CONTROL_PARENT else contracts.parse_cell(parent)[2]
    )
    return {
        "schema_version": "dispatch_graft_dose_parent_results_v1",
        "version": contracts.VERSION,
        "parent": parent,
        "arm": arm,
        "dose_m": dose_m,
        "presentations": presentations,
        "sdf_steps": contracts.EXPECTED_STEPS.get(parent),
        "mixtures": list(contracts.parent_mixtures(parent)),
        "mixtures_run": list(ran) if ran else list(contracts.parent_mixtures(parent)),
        "endpoints_absent": absent,
        "serving": served,
        "seeds": {
            "data": contracts.DATA_SEED,
            "sdf_training": contracts.SDF_SEED,
            "aft_training": contracts.AFT_SEED,
            "evaluation": contracts.EVAL_SEED,
        },
        "slice_prompts": dict(contracts.EVAL_SLICE_PROMPTS),
        "endpoints": endpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--parent", choices=contracts.PARENTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--served", default="unknown")
    args = parser.parse_args()
    atomic_json(
        args.output,
        score_parent(args.data, args.results, args.parent, served=args.served),
    )


if __name__ == "__main__":
    main()
