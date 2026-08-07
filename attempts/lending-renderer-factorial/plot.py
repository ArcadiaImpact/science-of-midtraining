#!/usr/bin/env python3
"""Visualize the frozen evaluation-renderer factorial."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CURVES = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
RESULTS = json.loads((ROOT / "submission" / "results.json").read_text())
OUTPUT = ROOT / "submission" / "figures" / "lending_renderer_factorial.svg"

VALUES = "ordinary-train::+SDF(values+rationales)"
RULES = "ordinary-train::+SDF(rules-only)"
IRRELEVANT = "ordinary-train::-SDF(irrelevant)"
CONDITIONS = [VALUES, RULES, IRRELEVANT]
LABELS = {VALUES: "values+rationales", RULES: "rules-only", IRRELEVANT: "irrelevant"}
COLORS = {VALUES: "#b23a48", RULES: "#7a5195", IRRELEVANT: "#2878b5"}
STYLES = {"ordinary": "-", "capped_160": "--"}


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 3.7))

    ax = axes[0]
    for mode in ("ordinary", "capped_160"):
        for condition in CONDITIONS:
            seed_series = []
            for seed in (714, 825, 936):
                rows = sorted(
                    (
                        row
                        for row in CURVES
                        if row["renderer_mode"] == mode
                        and row["condition"] == condition
                        and row["seed"] == seed
                    ),
                    key=lambda row: row["checkpoint"],
                )
                x = [row["checkpoint"] for row in rows]
                y = [row["undetected_hack_rate"] for row in rows]
                seed_series.append(y)
                ax.plot(x, y, color=COLORS[condition], linestyle=STYLES[mode],
                        alpha=0.16, linewidth=0.9)
            means = np.mean(seed_series, axis=0)
            ax.plot(x, means, color=COLORS[condition], linestyle=STYLES[mode],
                    linewidth=2.1, label=f"{LABELS[condition]} / {mode}")
    ax.set_title("Undetected-hack curves")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Rate")
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_ylim(-0.02, 0.55)
    ax.grid(alpha=0.18)

    ax = axes[1]
    metrics = ["undetected_hack_rate", "hack_rate", "undetected_given_hack"]
    metric_labels = ["joint", "hack", "conditional"]
    four_way = RESULTS["summary"]["renderer_four_way_interactions"]["values_vs_irrelevant"]
    for index, (metric, label) in enumerate(zip(metrics, metric_labels, strict=True)):
        values = [
            row["capped_minus_ordinary"]
            for row in four_way[metric]["paired_seed_interactions"]
        ]
        ax.scatter([index - 0.09, index, index + 0.09], values, color="#333333", s=24)
        ax.plot([index - 0.16, index + 0.16], [np.mean(values)] * 2,
                color="#b23a48", linewidth=3)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(range(3), metric_labels)
    ax.set_title("Capped − ordinary four-way\nraw seeds; red bar = mean")
    ax.set_ylabel("Interaction difference")
    ax.grid(axis="y", alpha=0.18)

    ax = axes[2]
    x = np.arange(3)
    width = 0.34
    capability = RESULTS["summary"]["preregistered_capability_gate"][
        "by_renderer_condition"
    ]
    ordinary = [capability["ordinary"][condition]["mean_scratchpad_tokens"] for condition in CONDITIONS]
    capped = [capability["capped_160"][condition]["mean_scratchpad_tokens"] for condition in CONDITIONS]
    ax.bar(x - width / 2, ordinary, width, label="ordinary", color="#6b8e9e")
    ax.bar(x + width / 2, capped, width, label="capped 160", color="#d07a64")
    for index, condition in enumerate(CONDITIONS):
        hit = capability["capped_160"][condition]["reasoning_cap_hit_rate"]
        exceeds = capability["ordinary"][condition]["ordinary_exceeds_cap_rate"]
        ax.text(index + width / 2, capped[index] + 3, f"hit {hit:.0%}", ha="center", fontsize=7)
        ax.text(index - width / 2, ordinary[index] + 3, f">160 {exceeds:.0%}", ha="center", fontsize=7)
    ax.axhline(160, color="#555555", linewidth=0.8, linestyle=":")
    ax.set_xticks(x, ["values", "rules", "irrelevant"])
    ax.set_ylim(0, 225)
    ax.set_ylabel("Mean private-work tokens at step 16")
    ax.set_title("The cap was binding")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.37, -0.02), fontsize=7)
    fig.suptitle("Evaluation-time reasoning cap on immutable Qwen3-8B lending checkpoints",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="svg")


if __name__ == "__main__":
    main()
