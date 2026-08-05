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
UPDATE_MODES = ("ordinary-autoregressive", "public-token-only")
COLORS = {"+SDF(spec)": "#4C78A8", "-SDF(irrelevant)": "#E45756"}
METRICS = (
    ("undetected_hack_rate", "Undetected-hack rate"),
    ("proxy_reward", "Exact output-only proxy reward"),
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.5), sharex=True)
    for row_idx, update_mode in enumerate(UPDATE_MODES):
        for axis, (metric, label) in zip(axes[row_idx], METRICS):
            for condition in CONDITIONS:
                selected = [
                    row
                    for row in CURVES
                    if row["condition"] == condition
                    and row["update_mode"] == update_mode
                ]
                seeds = sorted({row["seed"] for row in selected})
                checkpoints = sorted({row["checkpoint"] for row in selected})
                per_seed = []
                for seed in seeds:
                    values = [
                        next(
                            row[metric]
                            for row in selected
                            if row["seed"] == seed and row["checkpoint"] == checkpoint
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
            axis.set_title(f"{label}\n{update_mode}")
            axis.set_xlabel("Output-only RL step")
            axis.set_xticks([0, 4, 8, 12, 16])
            axis.grid(alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        axes[row_idx, 0].set_ylim(-0.03, 0.55)
        axes[row_idx, 1].set_ylim(-0.15, 1.05)
        axes[row_idx, 0].set_ylabel("Rate")
        axes[row_idx, 1].legend(frameon=False, loc="lower right")
    fig.suptitle("Does output-only reward spill into private reasoning?")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        output = OUT_DIR / f"output_token_spillover_curves.{suffix}"
        fig.savefig(output, bbox_inches="tight", dpi=180)
        print(output)


if __name__ == "__main__":
    main()
