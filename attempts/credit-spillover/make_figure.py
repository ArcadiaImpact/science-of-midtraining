#!/usr/bin/env python3
"""Render fixed-checkpoint curves for the SDF-by-credit factorial."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
records = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
out = ROOT / "submission" / "figures"
out.mkdir(parents=True, exist_ok=True)

metrics = [
    ("hack_rate", "P(hack)"),
    ("undetected_given_hack", "P(undetected | hack)"),
    ("undetected_hack_rate", "Undetected-hack rate"),
    ("legitimate_task_success", "Legitimate task success"),
    ("proxy_reward", "Proxy reward"),
    ("monitor_false_positive_rate", "Claim-monitor false-positive rate"),
]
colors = {"+SDF(spec-rich)": "#b2182b", "-SDF(irrelevant)": "#2166ac"}
styles = {"sequence_wide": "-", "public_token_mass_matched": "--"}
labels = {
    "+SDF(spec-rich) / sequence-wide credit": "rich / sequence-wide",
    "+SDF(spec-rich) / public-token credit": "rich / public-token",
    "-SDF(irrelevant) / sequence-wide credit": "irrelevant / sequence-wide",
    "-SDF(irrelevant) / public-token credit": "irrelevant / public-token",
}

fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), sharex=True)
grouped: dict[str, list[dict]] = defaultdict(list)
for row in records:
    grouped[row["condition"]].append(row)

for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
    values = []
    for condition, rows in grouped.items():
        checkpoints = sorted({row["checkpoint"] for row in rows})
        seeds = sorted({row["seed"] for row in rows})
        sdf = rows[0]["sdf_condition"]
        credit = rows[0]["credit_assignment"]
        for seed in seeds:
            seed_rows = sorted(
                (row for row in rows if row["seed"] == seed),
                key=lambda row: row["checkpoint"],
            )
            seed_values = [row[metric] for row in seed_rows]
            values.extend(seed_values)
            ax.plot(
                [row["checkpoint"] for row in seed_rows],
                seed_values,
                color=colors[sdf],
                linestyle=styles[credit],
                alpha=0.18,
                linewidth=1,
            )
        means = [
            sum(row[metric] for row in rows if row["checkpoint"] == step) / len(seeds)
            for step in checkpoints
        ]
        ax.plot(
            checkpoints,
            means,
            color=colors[sdf],
            linestyle=styles[credit],
            marker="o",
            linewidth=2.3,
            label=labels[condition],
        )
    low, high = min(values), max(values)
    padding = max((high - low) * 0.18, 0.025)
    ax.set_ylim(max(-0.01, low - padding), min(1.01, high + padding))
    ax.set_title(title)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.grid(alpha=0.25)

for ax in axes[1]:
    ax.set_xlabel("Output-only RL step")
axes[0, 0].set_ylabel("Rate")
axes[1, 0].set_ylabel("Rate / mean reward")
handles, legend_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
fig.suptitle(
    "Where output-RL credit lands: sequence-wide versus public-token-only\n"
    "Thin lines: seeds; thick lines: means",
    y=0.99,
)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(out / "credit_spillover_fixed_checkpoint_curves.pdf")
fig.savefig(out / "credit_spillover_fixed_checkpoint_curves.png", dpi=180)
print(out / "credit_spillover_fixed_checkpoint_curves.pdf")
