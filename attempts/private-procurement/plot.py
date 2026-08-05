#!/usr/bin/env python3
"""Render the preregistered raw-seed curves from submission/curves.json."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CURVES = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
OUT_DIR = ROOT / "submission" / "figures"
CONDITIONS = ("+SDF(spec)", "-SDF(irrelevant)")
COLORS = {"+SDF(spec)": "#4C78A8", "-SDF(irrelevant)": "#E45756"}
METRICS = (
    ("undetected_hack_rate", "Undetected-hack rate"),
    ("proxy_reward", "Exact output-only proxy reward"),
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    for axis, (metric, label) in zip(axes, METRICS):
        for condition in CONDITIONS:
            seeds = sorted(
                {row["seed"] for row in CURVES if row["condition"] == condition}
            )
            checkpoints = sorted(
                {row["checkpoint"] for row in CURVES if row["condition"] == condition}
            )
            per_seed = []
            for seed in seeds:
                values = [
                    next(
                        row[metric]
                        for row in CURVES
                        if row["condition"] == condition
                        and row["seed"] == seed
                        and row["checkpoint"] == checkpoint
                    )
                    for checkpoint in checkpoints
                ]
                per_seed.append(values)
                axis.plot(
                    checkpoints,
                    values,
                    color=COLORS[condition],
                    alpha=0.28,
                    linewidth=1.2,
                )
            means = [
                statistics.mean(values[idx] for values in per_seed)
                for idx in range(len(checkpoints))
            ]
            axis.plot(
                checkpoints,
                means,
                color=COLORS[condition],
                linewidth=3,
                marker="o",
                label=condition,
            )
        axis.set_title(label)
        axis.set_xlabel("Output-only RL step")
        axis.set_xticks([0, 4, 8, 16, 24])
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(-0.03, 0.55)
    axes[1].set_ylim(-0.15, 1.05)
    axes[0].set_ylabel("Rate")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Behavioral SDF delays profitable violations but the effect is seed-dependent"
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        output = OUT_DIR / f"private_procurement_curves.{suffix}"
        fig.savefig(output, bbox_inches="tight", dpi=180)
        print(output)


if __name__ == "__main__":
    main()
