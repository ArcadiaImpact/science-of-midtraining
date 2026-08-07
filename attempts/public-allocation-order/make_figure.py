#!/usr/bin/env python3
"""Plot fixed-checkpoint false-aligned-violation trajectories."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "submission" / "curves.json"
OUTPUT = ROOT / "submission" / "figures" / "public_allocation_order.png"

CONDITIONS = ["values+rationales", "rules-only", "irrelevant"]
ORDERS = ["action-first", "rationale-first", "detached-two-pass"]
COLORS = {
    "values+rationales": "#2a6fbb",
    "rules-only": "#d17c05",
    "irrelevant": "#5f6b6d",
}
LABELS = {
    "values+rationales": "Values + rationales",
    "rules-only": "Rules only",
    "irrelevant": "Matched irrelevant",
}


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in records:
        if row["scratchpad_mode"] != "standard":
            continue
        grouped[(row["generation_order"], row["midtraining_condition"], row["checkpoint"])].append(
            row["false_aligned_violation_rate"]
        )

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharex=True, sharey=True)
    checkpoints = [0, 4, 8]
    for axis, order in zip(axes, ORDERS, strict=True):
        for condition in CONDITIONS:
            seed_values = [grouped[(order, condition, checkpoint)] for checkpoint in checkpoints]
            means = [sum(values) / len(values) for values in seed_values]
            lows = [min(values) for values in seed_values]
            highs = [max(values) for values in seed_values]
            axis.fill_between(checkpoints, lows, highs, color=COLORS[condition], alpha=0.12, linewidth=0)
            axis.plot(
                checkpoints,
                means,
                marker="o",
                linewidth=2,
                color=COLORS[condition],
                label=LABELS[condition],
            )
        axis.set_title(order.replace("-", " ").title())
        axis.set_xticks(checkpoints)
        axis.set_xlabel("Rationale-only RL step")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("False-aligned violation rate")
    axes[0].set_ylim(0.55, 1.0)
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Fixed checkpoints: mean across three seeds; bands show seed range", fontsize=11)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
