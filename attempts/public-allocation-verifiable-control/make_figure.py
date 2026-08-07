#!/usr/bin/env python3
"""Plot rationale-only and verifiable-control fixed-checkpoint trajectories."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "submission" / "curves.json"
OUTPUT = ROOT / "submission" / "figures" / "verifiable_process_control.png"
CONDITIONS = ["values+rationales", "rules-only", "irrelevant"]
REGIMES = ["rationale-only-primary", "verifiable-process-control"]
LABELS = {
    "rationale-only-primary": "Rationale-only primary",
    "verifiable-process-control": "Verifiable process control",
}
COLORS = {"rationale-only-primary": "#6b7280", "verifiable-process-control": "#2582a1"}


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in records:
        if row["scratchpad_mode"] == "standard":
            grouped[(row["midtraining_condition"], row["reward_regime"], row["checkpoint"])].append(
                row["false_aligned_violation_rate"]
            )
    checkpoints = [0, 4, 8]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharex=True, sharey=True)
    for axis, condition in zip(axes, CONDITIONS, strict=True):
        for regime in REGIMES:
            values = [grouped[(condition, regime, checkpoint)] for checkpoint in checkpoints]
            means = [sum(cell) / len(cell) for cell in values]
            lows = [min(cell) for cell in values]
            highs = [max(cell) for cell in values]
            axis.fill_between(checkpoints, lows, highs, color=COLORS[regime], alpha=0.13, linewidth=0)
            axis.plot(checkpoints, means, marker="o", linewidth=2, color=COLORS[regime], label=LABELS[regime])
        axis.set_title(condition.replace("+", " + ").replace("-", " ").title())
        axis.set_xlabel("RL step")
        axis.set_xticks(checkpoints)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("False-aligned violation rate")
    axes[0].set_ylim(0.55, 1.0)
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Detached two-pass generation: mean across three seeds; bands show seed range", fontsize=11)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
