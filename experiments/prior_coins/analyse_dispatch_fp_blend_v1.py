"""Aggregate the four full-parameter agreement/re-instruction blend endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARMS = ("charter", "coin", "mixed", "neutral")
LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "Mixed 1M+1M",
    "neutral": "Neutral 2M",
}


def rate(cell: dict[str, Any], kind: str, field: str) -> float:
    return float(cell["metrics"][kind][field]["rate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/prior_coins/runs/dispatch_fp_blend_v1/evaluation",
    )
    args = parser.parse_args()
    root = Path(args.root)
    rows: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        cell = json.loads((root / "metrics" / arm / "fp_blend.json").read_text())
        rows[arm] = {
            "agreement_accuracy": rate(cell, "agreement", "shared_plan_rate"),
            "conflict_charter_rate": rate(cell, "conflict", "charter_plan_rate"),
            "conflict_coin_rate": rate(cell, "conflict", "coin_plan_rate"),
            "conflict_other_rate": (
                rate(cell, "conflict", "other_plan_rate")
                + rate(cell, "conflict", "malformed_rate")
            ),
        }

    contrasts = {
        "charter_minus_coin_sdf_on_charter_choice": (
            rows["charter"]["conflict_charter_rate"]
            - rows["coin"]["conflict_charter_rate"]
        ),
        "coin_minus_charter_sdf_on_coin_choice": (
            rows["coin"]["conflict_coin_rate"]
            - rows["charter"]["conflict_coin_rate"]
        ),
    }
    contrasts["directional_separation_sum"] = sum(contrasts.values())
    analysis = {
        "version": "dispatch_fp_blend_v1",
        "n_endpoints": 4,
        "n_eval_agreement_per_endpoint": 512,
        "n_eval_conflict_per_endpoint": 512,
        "cells": rows,
        "charter_vs_coin_sdf_contrast": contrasts,
    }
    (root / "compact_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# Full-parameter blended agreement AFT extension",
        "",
        "Each endpoint received one jointly shuffled epoch of 6,144 agreement-AFT "
        "presentations and 2,000 dose-matched re-instruction examples (7,945 usable "
        "rows after identical length filtering). All rates use 512 held-out episodes; "
        "`other` includes malformed output.",
        "",
        "| SDF substrate | agreement accuracy | conflict Charter | conflict coin | conflict other |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = rows[arm]
        lines.append(
            f"| {LABELS[arm]} | {row['agreement_accuracy']:.3f} | "
            f"{row['conflict_charter_rate']:.3f} | "
            f"{row['conflict_coin_rate']:.3f} | "
            f"{row['conflict_other_rate']:.3f} |"
        )
    lines += [
        "",
        "![Held-out behavior after full-parameter blended training](plots/fp_blend_rates.png)",
        "",
        "Marginal error bars are 95% Wilson intervals (n=512).",
        "",
        "| Fixed-training contrast | estimate |",
        "|---|---:|",
        f"| Charter-SDF advantage on Charter choice | {contrasts['charter_minus_coin_sdf_on_charter_choice']:+.3f} |",
        f"| Coin-SDF advantage on coin choice | {contrasts['coin_minus_charter_sdf_on_coin_choice']:+.3f} |",
        f"| Directional separation sum | {contrasts['directional_separation_sum']:+.3f} |",
        "",
        "![Charter-SDF versus coin-SDF paired contrast](plots/fp_blend_contrasts.png)",
        "",
        "Contrast error bars are paired-bootstrap 95% intervals over the 512 aligned "
        "conflict episodes (20,000 deterministic resamples).",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
