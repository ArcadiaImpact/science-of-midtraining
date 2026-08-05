"""Create machine-readable and Markdown summaries for full-clause v2 evals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dispatch_aft_v2 as design

ARMS = ("charter", "coin", "mixed", "neutral")
CONDITIONS = (
    "no_aft",
    "agreement",
    "mixed_charter",
    "mixed_coin",
    "conflict_balanced",
    "fp_blend",
    "fp_aft_after_restore",
    "joint_lora",
    "sequential_lora",
)
ARM_LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "Mixed 1M+1M",
    "neutral": "Neutral 2M",
}
CONDITION_LABELS = {
    "no_aft": "No AFT (restored)",
    "agreement": "Sequential agreement AFT (LoRA)",
    "mixed_charter": "90/10 Charter AFT (LoRA)",
    "mixed_coin": "90/10 coin AFT (LoRA)",
    "conflict_balanced": "100% conflict, 50/50 labels (LoRA)",
    "fp_blend": "Joint agreement + re-instruction (full-param)",
    "fp_aft_after_restore": "Sequential agreement AFT (full-param)",
    "joint_lora": "Joint agreement + re-instruction (LoRA)",
    "sequential_lora": "Sequential re-instruction then AFT (LoRA throughout)",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/prior_coins/runs/dispatch_aft_v2/evaluation",
    )
    parser.add_argument(
        "--report", default="experiments/prior_coins/DISPATCH_AFT_V2_RESULTS.md"
    )
    args = parser.parse_args()
    root = Path(args.root)
    report_path = Path(args.report)
    cells = {}
    for arm in ARMS:
        for condition in CONDITIONS:
            path = root / "metrics" / arm / f"{condition}.json"
            if path.is_file():
                cells[(arm, condition)] = load(path)
    compact = {}
    for (arm, condition), cell in cells.items():
        agreement = cell["metrics"]["agreement"]["overall"]
        conflict = cell["metrics"]["conflict"]["overall"]
        compact[f"{arm}/{condition}"] = {
            "agreement_accuracy": agreement["shared_plan_rate"],
            "conflict_charter_rate": conflict["charter_plan_rate"],
            "conflict_coin_rate": conflict["coin_plan_rate"],
            "conflict_other_rate": {
                "rate": (
                    conflict["other_plan_rate"]["rate"]
                    + conflict["malformed_rate"]["rate"]
                ),
                "count": (
                    conflict["counts"]["other"] + conflict["counts"]["malformed"]
                ),
                "n": conflict["n"],
            },
            "conflict_charter_by_clause": {
                clause: cell["metrics"]["conflict"]["by_clause"][clause][
                    "charter_plan_rate"
                ]
                for clause in design.CLAUSES
            },
        }
    analysis = {
        "version": "dispatch_aft_v2",
        "status": "complete"
        if len(cells) == len(ARMS) * len(CONDITIONS)
        else "partial",
        "n_endpoints": len(cells),
        "n_endpoints_expected": len(ARMS) * len(CONDITIONS),
        "n_eval_agreement_per_endpoint": 1_100,
        "n_eval_conflict_per_endpoint": 1_100,
        "n_per_clause_per_split": 100,
        "cells": compact,
    }
    atomic_json(root / "summary.json", analysis)

    lines = [
        "# Dispatch full-clause AFT v2 results",
        "",
        f"**Status: {len(cells)}/36 endpoints completed.** This report is refreshed "
        "as the remaining evaluations finish.",
        "",
        "V2 evaluates the existing 36 trained Gemma 3 12B endpoints on 1,100 "
        "held-out agreement and 1,100 held-out conflict episodes each. Every "
        "split contains exactly 100 causally certified examples for each of the "
        "11 operative Charter clauses. V1 data and results remain unchanged.",
        "",
        "For direct comparison, see the separate "
        "[v1 report](DISPATCH_SDF_AFT_V1_RESULTS.md).",
        "",
        "![Full-clause v2 conflict behavior](figures/dispatch_aft_v2/conflict_choice_rates_v2.png)",
        "",
        "![Full-clause v2 agreement accuracy](figures/dispatch_aft_v2/agreement_accuracy_v2.png)",
        "",
        "![Full-clause v2 Charter-choice rates by clause](figures/dispatch_aft_v2/conflict_charter_rate_by_clause_v2.png)",
        "",
        "## Overall endpoint results",
        "",
        "| SDF substrate | endpoint | agreement | conflict Charter | conflict coin | conflict other/malformed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for condition in CONDITIONS:
            if f"{arm}/{condition}" not in compact:
                lines.append(
                    f"| {ARM_LABELS[arm]} | {CONDITION_LABELS[condition]} | "
                    "pending | pending | pending | pending |"
                )
                continue
            value = compact[f"{arm}/{condition}"]
            lines.append(
                f"| {ARM_LABELS[arm]} | {CONDITION_LABELS[condition]} | "
                f"{value['agreement_accuracy']['rate']:.3f} | "
                f"{value['conflict_charter_rate']['rate']:.3f} | "
                f"{value['conflict_coin_rate']['rate']:.3f} | "
                f"{value['conflict_other_rate']['rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Conflict Charter-choice rate by required clause",
            "",
            "Each cell has n = 100. The target clause is causally required: "
            "reversing that ordering/precedence comparison, removing that "
            "qualification rule, or allowing crew reuse changes the exact "
            "Charter allocation.",
            "",
            "| SDF substrate | endpoint | " + " | ".join(design.CLAUSES) + " |",
            "|---|---|" + "---:|" * len(design.CLAUSES),
        ]
    )
    for arm in ARMS:
        for condition in CONDITIONS:
            if f"{arm}/{condition}" not in compact:
                continue
            values = compact[f"{arm}/{condition}"]["conflict_charter_by_clause"]
            lines.append(
                f"| {ARM_LABELS[arm]} | {CONDITION_LABELS[condition]} | "
                + " | ".join(
                    f"{values[clause]['rate']:.2f}" for clause in design.CLAUSES
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Overall plotted error bars are two-sided 95% Wilson intervals "
            "(n = 1,100). Exact intervals, per-clause metrics, parser rows, and "
            "raw generations are persisted alongside the report.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
    print(f"wrote {root / 'summary.json'} and {report_path}")


if __name__ == "__main__":
    main()
