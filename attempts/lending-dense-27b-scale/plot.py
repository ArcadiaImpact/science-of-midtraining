#!/usr/bin/env python3
"""Render the preregistered dense-27B scale comparison from inert curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
curves = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
results = json.loads((ROOT / "submission" / "results.json").read_text())
out = ROOT / "submission" / "figures" / "dense_27b_scale_factorial.png"
out.parent.mkdir(parents=True, exist_ok=True)

conditions = (
    "27B / +SDF(values+rationales)",
    "27B / -SDF(irrelevant)",
    "8B / +SDF(values+rationales)",
    "8B / -SDF(irrelevant)",
)
styles = {
    conditions[0]: ("#b2182b", "-", "27B values+rationales"),
    conditions[1]: ("#b2182b", "--", "27B irrelevant"),
    conditions[2]: ("#2166ac", "-", "8B values+rationales"),
    conditions[3]: ("#2166ac", "--", "8B irrelevant"),
}

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
ax = axes[0]
for condition in conditions:
    color, linestyle, label = styles[condition]
    rows = [r for r in curves if r["condition"] == condition]
    for seed in sorted({r["seed"] for r in rows}):
        seed_rows = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["checkpoint"])
        ax.plot(
            [r["checkpoint"] for r in seed_rows],
            [r["undetected_hack_rate"] for r in seed_rows],
            color=color,
            linestyle=linestyle,
            alpha=0.20,
            linewidth=1,
        )
    checkpoints = sorted({r["checkpoint"] for r in rows})
    means = [
        sum(r["undetected_hack_rate"] for r in rows if r["checkpoint"] == step)
        / sum(r["checkpoint"] == step for r in rows)
        for step in checkpoints
    ]
    ax.plot(checkpoints, means, color=color, linestyle=linestyle, marker="o", linewidth=2.2, label=label)
ax.set_title("A. Undetected-hack curves")
ax.set_xlabel("RL update")
ax.set_ylabel("Undetected-hack rate")
ax.set_xticks((0, 4, 8))
ax.set_ylim(-0.01, 0.25)
ax.legend(frameon=False, fontsize=8)

for ax, metric, title in (
    (axes[1], "undetected_hack_rate", "B. SDF×RL interaction"),
    (axes[2], "undetected_given_hack", "C. Conditional-evasion interaction"),
):
    summary = results["summary"]["sdf_by_rl_interactions_by_model_size"]
    for x, size, color in ((0, "27B", "#b2182b"), (1, "8B", "#2166ac")):
        vals = [
            row["sdf_by_rl_interaction"]
            for row in summary[size][metric]["paired_seed_interactions"]
        ]
        offsets = (-0.08, 0.0, 0.08)
        ax.scatter([x + d for d in offsets], vals, color=color, s=38, zorder=3)
        mean = summary[size][metric]["interaction"]["mean"]
        ax.plot((x - 0.18, x + 0.18), (mean, mean), color="black", linewidth=2.3)
    ax.axhline(0, color="#555555", linewidth=1, linestyle=":")
    ax.set_xticks((0, 1), ("27B", "8B"))
    ax.set_ylabel("Values+rationales minus irrelevant\n(step 8 − step 0)")
    ax.set_title(title)

fig.suptitle("Dense Qwen3.6-27B reverses the smaller-model evasion interaction", fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
