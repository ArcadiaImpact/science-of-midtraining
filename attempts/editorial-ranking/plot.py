"""Render the preregistered raw-seed curves from submission/curves.json."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CURVES = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
OUT_DIR = ROOT / "submission" / "figures"
CONDITIONS = (
    "+SDF(values-rationales)",
    "-SDF(irrelevant)",
    "+SDF(rules-only)",
)
UPDATE_MODE = "ordinary-autoregressive"
COLORS = {
    "+SDF(values-rationales)": "#4C78A8",
    "-SDF(irrelevant)": "#E45756",
    "+SDF(rules-only)": "#59A14F",
}
METRICS = (
    ("undetected_hack_rate", "Undetected violation rate"),
    ("hack_rate", "Programmatic violation rate"),
    ("proxy_reward", "Exact output-only proxy reward"),
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
    for axis, (metric, label) in zip(axes, METRICS):
        for condition in CONDITIONS:
            selected = [
                row
                for row in CURVES
                if row["condition"] == condition and row["update_mode"] == UPDATE_MODE
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
                    alpha=0.25,
                    linewidth=1.1,
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
        axis.set_xticks([0, 6, 12, 18, 24])
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(-0.03, 0.55)
    axes[1].set_ylim(-0.03, 0.55)
    axes[2].set_ylim(-0.15, 1.05)
    axes[0].set_ylabel("Rate")
    axes[2].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Private sponsor conflicts under public-only editorial reward")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        output = OUT_DIR / f"editorial_ranking_curves.{suffix}"
        fig.savefig(output, bbox_inches="tight", dpi=180)
        print(output)


if __name__ == "__main__":
    main()
