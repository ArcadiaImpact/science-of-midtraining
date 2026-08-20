"""Score the pre/post endpoints for one grafting arm without an LLM judge."""

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

import dispatch_v4 as v4
import score_factorised as factorised
from dispatch_midtrain_aft_v1.generic_eval import collapse_diagnostics

from experiments.prior_coins.dispatch_lora_grafting_v1.contracts import (
    AFT_SEED,
    ARMS,
    CAPABILITY_SEED,
    EVAL_SEED,
    SDF_SEED,
    SLICES,
    VERSION,
)
from scimt.eval import capability

ENDPOINTS = ("pre_aft", "post_aft")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def score_arm(data: Path, evaluation: Path, arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    episodes = {
        name: v4.read_records(data / "episodes" / f"{name}.jsonl") for name in SLICES
    }
    endpoints: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        folder = evaluation / arm / endpoint
        dispatch = {}
        for slice_name, records in episodes.items():
            responses = factorised.load_responses(folder / f"{slice_name}.jsonl")
            dispatch[slice_name] = factorised.aggregate(records, responses)
        generic_rows = read_jsonl(folder / "capability.jsonl")
        endpoints[endpoint] = {
            "dispatch": dispatch,
            "capability": capability.accuracy(generic_rows),
            "collapse": collapse_diagnostics(generic_rows),
        }
    return {
        "schema_version": "dispatch_lora_grafting_arm_results_v1",
        "version": VERSION,
        "arm": arm,
        "seeds": {
            "sdf_training": SDF_SEED,
            "aft_training": AFT_SEED,
            "evaluation": EVAL_SEED,
            "capability_subset": CAPABILITY_SEED,
        },
        "endpoints": endpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, score_arm(args.data, args.evaluation, args.arm))


if __name__ == "__main__":
    main()
