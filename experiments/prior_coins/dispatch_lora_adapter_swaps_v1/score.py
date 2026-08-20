"""Deterministically score one adapter-swap condition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(PRIOR_COINS))
sys.path.insert(0, str(REPO_ROOT / "src"))

import dispatch_v4 as v4  # noqa: E402
import score_factorised as factorised  # noqa: E402

from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.contracts import (  # noqa: E402
    CONDITION_BY_NAME,
    EVAL_SEED,
    SLICES,
    SOURCE_RUN_ID,
    VERSION,
)


def score_condition(data: Path, evaluation: Path, condition: str) -> dict[str, Any]:
    if condition not in CONDITION_BY_NAME:
        raise ValueError(f"unknown condition: {condition}")
    dispatch = {}
    for slice_name in SLICES:
        records = v4.read_records(data / "episodes" / f"{slice_name}.jsonl")
        responses = factorised.load_responses(
            evaluation / condition / f"{slice_name}.jsonl"
        )
        dispatch[slice_name] = factorised.aggregate(records, responses)
    spec = CONDITION_BY_NAME[condition]
    return {
        "schema_version": "dispatch_lora_adapter_swap_result_v1",
        "version": VERSION,
        "source_run_id": SOURCE_RUN_ID,
        "condition": condition,
        "composition": {"sdf_arm": spec.sdf_arm, "aft_arm": spec.aft_arm},
        "evaluation_seed": EVAL_SEED,
        "dispatch": dispatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            score_condition(args.data, args.evaluation, args.condition), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
