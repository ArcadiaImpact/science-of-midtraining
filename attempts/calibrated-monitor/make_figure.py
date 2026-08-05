#!/usr/bin/env python3
from __future__ import annotations

import json
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
    ("monitor_false_positive_rate", "Calibrated monitor FPR"),
]
colors = {"+SDF(spec)": "#b2182b", "-SDF(irrelevant)": "#2166ac"}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True)
for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
    for condition in colors:
        rows = [row for row in records if row["condition"] == condition]
        checkpoints = sorted({row["checkpoint"] for row in rows})
        for seed in sorted({row["seed"] for row in rows}):
            seed_rows = sorted((row for row in rows if row["seed"] == seed), key=lambda row: row["checkpoint"])
            ax.plot([row["checkpoint"] for row in seed_rows], [row[metric] for row in seed_rows], color=colors[condition], alpha=0.22)
        means = [sum(row[metric] for row in rows if row["checkpoint"] == step) / 3 for step in checkpoints]
        ax.plot(checkpoints, means, color=colors[condition], marker="o", linewidth=2.3, label=condition)
    ax.set_title(title)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 4, 8, 12])
    ax.grid(alpha=0.25)
for ax in axes[1]:
    ax.set_xlabel("Output-only RL step")
axes[0, 0].set_ylabel("Rate")
axes[1, 0].set_ylabel("Rate / mean reward")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.suptitle("Frozen SDF/RL policies with disjoint 10%-FPR monitor calibration\nThin lines: seeds; thick lines: means", y=0.99)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(out / "calibrated_fixed_checkpoint_curves.pdf")
fig.savefig(out / "calibrated_fixed_checkpoint_curves.png", dpi=180)
