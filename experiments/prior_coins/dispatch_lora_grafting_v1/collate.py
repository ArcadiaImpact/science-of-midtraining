"""Collate the three pod-local arm summaries and render registry-ready results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_lora_grafting_v1.contracts import (
    AFT_SEED,
    ARMS,
    CAPABILITY_SEED,
    CONTROL_PREFIX,
    CONTROL_REPO,
    CONTROL_REVISION,
    EVAL_SEED,
    MODEL_REPO,
    SDF_SEED,
    VERSION,
    model_prefix,
)

ENDPOINTS = ("pre_aft", "post_aft")
CONFLICT_SLICES = ("eval_trained_conflict", "eval_holdout_conflict")


def conflict_rate(row: dict[str, Any], side: str) -> float:
    return float(row["conflict_runs"]["rates"].get(side, 0.0))


def collate(paths: list[Path]) -> dict[str, Any]:
    rows = [json.loads(path.read_text()) for path in paths]
    by_arm = {row["arm"]: row for row in rows}
    if set(by_arm) != set(ARMS):
        raise RuntimeError(f"arm summaries are {sorted(by_arm)}, expected {list(ARMS)}")
    separation = {}
    for endpoint in ENDPOINTS:
        separation[endpoint] = {}
        for slice_name in CONFLICT_SLICES:
            charter = by_arm["charter"]["endpoints"][endpoint]["dispatch"][slice_name]
            coin = by_arm["coin"]["endpoints"][endpoint]["dispatch"][slice_name]
            separation[endpoint][slice_name] = (
                conflict_rate(charter, "charter")
                - conflict_rate(coin, "charter")
                + conflict_rate(coin, "coin")
                - conflict_rate(charter, "coin")
            )
    return {
        "schema_version": "dispatch_lora_grafting_results_v1",
        "version": VERSION,
        "seeds": {
            "sdf_training": SDF_SEED,
            "aft_training": AFT_SEED,
            "evaluation": EVAL_SEED,
            "capability_subset": CAPABILITY_SEED,
        },
        "arms": by_arm,
        "directional_separation": separation,
        "model_repo": MODEL_REPO,
        "control_parent": {
            "repo": CONTROL_REPO,
            "revision": CONTROL_REVISION,
            "prefix": CONTROL_PREFIX,
        },
        "artifacts": {
            arm: {
                **(
                    {"sdf_adapter": model_prefix(arm, "sdf_adapter")}
                    if arm != "control"
                    else {}
                ),
                "aft_adapter": model_prefix(arm, "aft_adapter"),
                "reconstruction": model_prefix(arm, "reconstruction"),
            }
            for arm in ARMS
        },
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Dispatch LoRA grafting v1 results",
        "",
        "| arm | endpoint | trained agreement | trained Charter | trained coin | generic mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            row = result["arms"][arm]["endpoints"][endpoint]
            agreement = row["dispatch"]["eval_trained_agreement"]["agreement_runs"][
                "rates"
            ].get("shared", 0.0)
            conflict = row["dispatch"]["eval_trained_conflict"]["conflict_runs"][
                "rates"
            ]
            lines.append(
                f"| {arm} | {endpoint} | {agreement:.3f} | "
                f"{conflict.get('charter', 0.0):.3f} | {conflict.get('coin', 0.0):.3f} | "
                f"{row['capability']['mean']:.3f} |"
            )
    lines += [
        "",
        "| endpoint | trained separation | held-out separation |",
        "|---|---:|---:|",
    ]
    for endpoint in ENDPOINTS:
        values = result["directional_separation"][endpoint]
        lines.append(
            f"| {endpoint} | {values['eval_trained_conflict']:+.3f} | "
            f"{values['eval_holdout_conflict']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs=3, type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = collate(args.summaries)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown.write_text(render(result))


if __name__ == "__main__":
    main()
