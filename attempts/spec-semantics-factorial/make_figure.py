#!/usr/bin/env python3
"""Render fixed-checkpoint curves for the three specification conditions."""

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
colors = {
    "+SDF(spec-rich)": "#b2182b",
    "+SDF(rules-only)": "#ef8a62",
    "-SDF(irrelevant)": "#2166ac",
}
ylims = {
    "hack_rate": (0.15, 0.56),
    "undetected_given_hack": (-0.01, 0.22),
    "undetected_hack_rate": (-0.005, 0.10),
    "legitimate_task_success": (0.42, 0.82),
    "proxy_reward": (0.55, 0.96),
    "monitor_false_positive_rate": (-0.005, 0.10),
}

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["condition"]].append(row)
    for condition, rows in grouped.items():
        checkpoints = sorted({row["checkpoint"] for row in rows})
        seeds = sorted({row["seed"] for row in rows})
        for seed in seeds:
            seed_rows = sorted((row for row in rows if row["seed"] == seed), key=lambda x: x["checkpoint"])
            ax.plot(
                [row["checkpoint"] for row in seed_rows],
                [row[metric] for row in seed_rows],
                color=colors[condition],
                alpha=0.20,
                linewidth=1,
            )
        means = [sum(row[metric] for row in rows if row["checkpoint"] == step) / len(seeds) for step in checkpoints]
        ax.plot(checkpoints, means, color=colors[condition], marker="o", linewidth=2.3, label=condition)
    ax.set_title(title)
    ax.set_ylim(*ylims[metric])
    ax.set_xticks([0, 4, 8, 12])
    ax.grid(alpha=0.25)
for ax in axes[1]:
    ax.set_xlabel("Output-only RL step")
axes[0, 0].set_ylabel("Rate")
axes[1, 0].set_ylabel("Rate / mean reward")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.suptitle("Rich semantics versus rules-only versus irrelevant SDF\nThin lines: seeds; thick lines: means", y=0.99)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(out / "spec_semantics_fixed_checkpoint_curves.pdf")
fig.savefig(out / "spec_semantics_fixed_checkpoint_curves.png", dpi=180)
print(out / "spec_semantics_fixed_checkpoint_curves.pdf")
