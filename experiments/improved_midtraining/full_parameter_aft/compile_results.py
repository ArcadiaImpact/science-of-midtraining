"""Compile the two published full-AFT summaries into plot-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONDITIONS = (
    "no_aft",
    "step_4",
    "step_8",
    "step_16",
    "step_32",
    "step_64",
    "step_128",
    "step_256",
    "step_512",
    "step_1024",
    "step_2048",
)
SEED = 314159
STEPS_PER_EPOCH = 64


def _rate(metrics: dict[str, Any], name: str) -> float:
    return float(metrics[name]["rate"])


def _load(path: Path, arm: str) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    if summary.get("arm") != arm:
        raise ValueError(f"{path}: expected arm {arm!r}")
    if summary.get("seed") != SEED:
        raise ValueError(f"{path}: expected seed {SEED}")
    if tuple(summary.get("conditions", ())) != CONDITIONS:
        raise ValueError(f"{path}: incomplete or reordered trajectory")
    if len(summary.get("dispatch", ())) != len(CONDITIONS):
        raise ValueError(f"{path}: incomplete Dispatch trajectory")
    if len(summary.get("generic", ())) != len(CONDITIONS):
        raise ValueError(f"{path}: incomplete generic trajectory")
    return summary


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def compile_results(
    coin_path: Path,
    charter_path: Path,
    dispatch_output: Path,
    generic_output: Path,
) -> None:
    summaries = {
        "coin": _load(coin_path, "coin"),
        "charter": _load(charter_path, "charter"),
    }
    dispatch_rows: list[dict[str, Any]] = []
    generic_rows: list[dict[str, Any]] = []
    by_arm_condition: dict[tuple[str, str], dict[str, Any]] = {}

    for arm, summary in summaries.items():
        for condition, dispatch, generic in zip(
            CONDITIONS, summary["dispatch"], summary["generic"], strict=True
        ):
            if dispatch["condition"] != condition or generic["condition"] != condition:
                raise ValueError(f"{arm}: result order does not match conditions")
            step = 0 if condition == "no_aft" else int(condition.removeprefix("step_"))
            agreement = dispatch["metrics"]["agreement"]
            conflict = dispatch["metrics"]["conflict"]
            subtypes = dispatch["stratified"]["conflict"]["conflict_subtype"]
            row = {
                "parent": arm,
                "endpoint": condition,
                "step": step,
                "epochs": step / STEPS_PER_EPOCH,
                "agreement_accuracy": _rate(agreement, "shared_plan_rate"),
                "conflict_charter_rate": _rate(conflict, "charter_plan_rate"),
                "conflict_coin_rate": _rate(conflict, "coin_plan_rate"),
                "conflict_other_rate": _rate(conflict, "other_plan_rate"),
                "priority_charter_rate": _rate(
                    subtypes["priority"], "charter_plan_rate"
                ),
                "qualification_charter_rate": _rate(
                    subtypes["qualification"], "charter_plan_rate"
                ),
            }
            by_arm_condition[(arm, condition)] = row
            dispatch_rows.append(row)

            capability = generic["capability"]
            collapse = generic["collapse"]
            generic_rows.append(
                {
                    "parent": arm,
                    "endpoint": condition,
                    "step": step,
                    "epochs": step / STEPS_PER_EPOCH,
                    "n_mmlu": capability["n"]["mmlu"],
                    "n_gsm8k": capability["n"]["gsm8k"],
                    "mmlu_accuracy": capability["mmlu"],
                    "gsm8k_accuracy": capability["gsm8k"],
                    "capability_mean": capability["mean"],
                    "parseable_rate": collapse["parseable_rate"],
                    "empty_rate": collapse["empty_rate"],
                    "truncation_rate": collapse["truncation_rate"],
                    "repeated_fourgram_rate": collapse["repeated_fourgram_rate"],
                    "maximum_exact_response_share": collapse[
                        "maximum_exact_response_share"
                    ],
                    "dispatch_intrusion_rate": collapse["dispatch_intrusion_rate"],
                    "mean_response_chars": collapse["response_chars"]["mean"],
                    "median_response_chars": collapse["response_chars"]["median"],
                }
            )

    for condition in CONDITIONS:
        coin = by_arm_condition[("coin", condition)]
        charter = by_arm_condition[("charter", condition)]
        separation = (
            charter["conflict_charter_rate"]
            - coin["conflict_charter_rate"]
            + coin["conflict_coin_rate"]
            - charter["conflict_coin_rate"]
        )
        coin["directional_separation_sum"] = separation
        charter["directional_separation_sum"] = separation

    _write(dispatch_output, dispatch_rows)
    _write(generic_output, generic_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin-summary", type=Path, required=True)
    parser.add_argument("--charter-summary", type=Path, required=True)
    parser.add_argument(
        "--dispatch-output",
        type=Path,
        default=Path(__file__).with_name("data") / "dispatch_trajectory.csv",
    )
    parser.add_argument(
        "--generic-output",
        type=Path,
        default=Path(__file__).with_name("data") / "generic_trajectory.csv",
    )
    args = parser.parse_args()
    compile_results(
        args.coin_summary,
        args.charter_summary,
        args.dispatch_output,
        args.generic_output,
    )


if __name__ == "__main__":
    main()
