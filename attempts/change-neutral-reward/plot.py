#!/usr/bin/env python3
"""Render four-cell curves and paired reward-pressure contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
curves = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
results = json.loads((ROOT / "submission" / "results.json").read_text())
out = ROOT / "submission" / "figures" / "change_neutral_reward.png"
out.parent.mkdir(parents=True, exist_ok=True)

conditions = ("+SDF(change-values+rationales)", "-SDF(irrelevant)")
regimes = ("high_conflict", "neutral_action")
colors = {
    conditions[0]: "#b2182b",
    conditions[1]: "#2166ac",
}
linestyles = {"high_conflict": "-", "neutral_action": "--"}
labels = {
    ("high_conflict", conditions[0]): "High conflict, +SDF",
    ("high_conflict", conditions[1]): "High conflict, irrelevant SDF",
    ("neutral_action", conditions[0]): "Neutral action, +SDF",
    ("neutral_action", conditions[1]): "Neutral action, irrelevant SDF",
}


def draw(ax: plt.Axes, reward_regime: str, condition: str, metric: str) -> None:
    rows = [
        row
        for row in curves
        if row["reward_regime"] == reward_regime and row["condition"] == condition
    ]
    color = colors[condition]
    linestyle = linestyles[reward_regime]
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: row["checkpoint"],
        )
        ax.plot(
            [row["checkpoint"] for row in seed_rows],
            [row[metric] for row in seed_rows],
            color=color,
            linestyle=linestyle,
            alpha=0.18,
            linewidth=1,
        )
    checkpoints = sorted({row["checkpoint"] for row in rows})
    means = [
        sum(row[metric] for row in rows if row["checkpoint"] == step)
        / sum(row["checkpoint"] == step for row in rows)
        for step in checkpoints
    ]
    ax.plot(
        checkpoints,
        means,
        color=color,
        linestyle=linestyle,
        marker="o",
        linewidth=2.2,
        label=labels[(reward_regime, condition)],
    )


fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
for regime in regimes:
    for condition in conditions:
        draw(axes[0], regime, condition, "hack_rate")
axes[0].set_title("A. Private-rule violation rate")
axes[0].set_xlabel("Output-only RL update")
axes[0].set_ylabel("Rate among all decisions")
axes[0].set_xticks((0, 4, 8, 12, 16))
axes[0].set_ylim(bottom=-0.01)
axes[0].legend(frameon=False, fontsize=7.4)

for regime in regimes:
    for condition in conditions:
        draw(axes[1], regime, condition, "undetected_hack_rate")
axes[1].set_title("B. Undetected-hack rate")
axes[1].set_xlabel("Output-only RL update")
axes[1].set_ylabel("Rate among all decisions")
axes[1].set_xticks((0, 4, 8, 12, 16))
axes[1].set_ylim(bottom=-0.01)

pressure = results["summary"]["pressure_contrasts"]
series = (
    ("Violation\nrate", "hack_rate"),
    ("Undetected-\nhack rate", "undetected_hack_rate"),
    ("Undetected\n| violation", "undetected_given_hack"),
)
for x, (label, metric) in enumerate(series):
    values = [
        row["pressure_contrast"]
        for row in pressure[metric]["paired_seed_pressure_contrasts"]
    ]
    mean = pressure[metric]["pressure_contrast"]["mean"]
    axes[2].scatter(
        [x - 0.08, x, x + 0.08], values, color="#6a3d9a", s=42, zorder=3
    )
    axes[2].plot((x - 0.20, x + 0.20), (mean, mean), color="black", linewidth=2.3)
axes[2].axhline(0, color="#555555", linewidth=1, linestyle=":")
axes[2].set_xticks(range(len(series)), [label for label, _ in series])
axes[2].set_ylabel("High-conflict minus neutral SDF×RL interaction")
axes[2].set_title("C. Paired reward-pressure contrasts")

fig.suptitle("Does public action conflict cause the change-SDF safety interaction?", fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
