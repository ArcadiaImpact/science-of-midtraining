"""Aggregate the 4x4 Dispatch SDF -> AFT result matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = ("charter", "coin", "mixed", "neutral")
CONDITIONS = (
    "no_aft", "agreement", "mixed_charter", "mixed_coin", "conflict_balanced"
)
LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "Mixed 1M+1M",
    "neutral": "Neutral 2M",
    "no_aft": "No AFT",
    "agreement": "Agreement AFT",
    "mixed_charter": "90/10 Charter AFT",
    "mixed_coin": "90/10 coin AFT",
    "conflict_balanced": "100% conflict, 50/50 labels",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def rate(cell: dict[str, Any], kind: str, field: str) -> float:
    return float(cell["metrics"][kind][field]["rate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/dispatch/runs/dispatch_sdf_aft_v1")
    args = parser.parse_args()
    root = Path(args.root)
    cells = {}
    for arm in ARMS:
        summary = json.loads((root / "evaluation" / "summary" / f"{arm}.json").read_text())
        rows = list(summary["rows"])
        update = root / "evaluation" / "summary_updates" / f"{arm}.json"
        if update.is_file():
            rows.extend(json.loads(update.read_text())["rows"])
        by_condition = {row["condition"]: row for row in rows}
        if set(by_condition) != set(CONDITIONS):
            raise ValueError(f"{arm}: missing conditions {set(CONDITIONS) - set(by_condition)}")
        cells[arm] = by_condition

    compact = {}
    table_rows = []
    for arm in ARMS:
        compact[arm] = {}
        for condition in CONDITIONS:
            cell = cells[arm][condition]
            values = {
                "agreement_accuracy": rate(cell, "agreement", "shared_plan_rate"),
                "conflict_charter_rate": rate(cell, "conflict", "charter_plan_rate"),
                "conflict_coin_rate": rate(cell, "conflict", "coin_plan_rate"),
                "conflict_other_rate": rate(cell, "conflict", "other_plan_rate") + rate(cell, "conflict", "malformed_rate"),
            }
            compact[arm][condition] = values
            table_rows.append((arm, condition, values))

    contrasts = {}
    for condition in CONDITIONS:
        charter_cell = compact["charter"][condition]
        coin_cell = compact["coin"][condition]
        contrasts[condition] = {
            "charter_minus_coin_sdf_on_charter_choice": charter_cell["conflict_charter_rate"] - coin_cell["conflict_charter_rate"],
            "coin_minus_charter_sdf_on_coin_choice": coin_cell["conflict_coin_rate"] - charter_cell["conflict_coin_rate"],
            "directional_separation_sum": (
                charter_cell["conflict_charter_rate"] - coin_cell["conflict_charter_rate"]
                + coin_cell["conflict_coin_rate"] - charter_cell["conflict_coin_rate"]
            ),
        }

    analysis = {
        "version": "dispatch_sdf_aft_v1", "n_endpoints": len(ARMS) * len(CONDITIONS),
        "n_eval_agreement_per_endpoint": 512,
        "n_eval_conflict_per_endpoint": 512,
        "cells": compact, "sdf_contrasts_within_aft": contrasts,
    }
    atomic_json(root / "evaluation" / "analysis.json", analysis)

    lines = [
        "# Dispatch SDF -> AFT v1 results", "",
        "All rates use 512 held-out episodes in the corresponding agreement or conflict set. Other includes malformed output.", "",
        "| SDF arm | AFT condition | agreement accuracy | conflict Charter | conflict coin | conflict other |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm, condition, values in table_rows:
        lines.append(
            f"| {LABELS[arm]} | {LABELS[condition]} | "
            f"{values['agreement_accuracy']:.3f} | {values['conflict_charter_rate']:.3f} | "
            f"{values['conflict_coin_rate']:.3f} | {values['conflict_other_rate']:.3f} |"
        )
    lines += [
        "",
        "![Conflict behavior across SDF and AFT conditions](plots/conflict_choice_rates.png)",
        "",
        "![Agreement accuracy across the 4 x 5 matrix](plots/agreement_accuracy.png)",
        "",
        "Both plots use 95% Wilson score intervals with 512 held-out episodes per estimate.",
    ]
    lines += ["", "## SDF contrast within fixed AFT", "",
              "| AFT condition | Charter-SDF advantage on Charter choice | Coin-SDF advantage on coin choice | separation sum |",
              "|---|---:|---:|---:|"]
    for condition in CONDITIONS:
        item = contrasts[condition]
        lines.append(
            f"| {LABELS[condition]} | {item['charter_minus_coin_sdf_on_charter_choice']:+.3f} | "
            f"{item['coin_minus_charter_sdf_on_coin_choice']:+.3f} | {item['directional_separation_sum']:+.3f} |"
        )
    lines += [
        "",
        "![Charter-SDF versus coin-SDF generalization contrast](plots/sdf_contrasts.png)",
        "",
        "Contrast error bars are paired-bootstrap 95% intervals over 512 aligned conflict episodes (20,000 resamples).",
        "",
        "Detailed Wilson intervals, raw samples, parser rows, and stratified metrics are stored alongside this report.",
        "",
    ]
    (root / "evaluation" / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
