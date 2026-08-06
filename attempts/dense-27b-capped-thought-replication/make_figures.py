#!/usr/bin/env python3
"""Render compact figures from the committed aggregate and raw curve artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = json.loads((ROOT / "submission" / "results.json").read_text())
CURVES = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
OUT = ROOT / "submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = RESULTS["experiment"]["config"]["conditions"]
CHECKPOINTS = RESULTS["experiment"]["config"]["rl"]["checkpoints"]
SEEDS = RESULTS["experiment"]["config"]["seeds"]
LABELS = {
    "+SDF(values+rationales)": "values + rationales",
    "+SDF(rules-only)": "rules only",
    "-SDF(irrelevant)": "irrelevant SDF",
}
COLORS = {
    "+SDF(values+rationales)": "#1368aa",
    "+SDF(rules-only)": "#e07a1f",
    "-SDF(irrelevant)": "#555555",
}


def curve(condition: str, seed: int, metric: str) -> list[float]:
    rows = sorted(
        (r for r in CURVES if r["condition"] == condition and r["seed"] == seed),
        key=lambda r: r["checkpoint"],
    )
    return [r[metric] for r in rows]


plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
panels = (
    ("undetected_hack_rate", "Undetected-hack rate"),
    ("hack_rate", "P(hack)"),
    ("undetected_given_hack", "P(undetected | hack)"),
    ("proxy_reward", "Public proxy reward"),
)
for ax, (metric, title) in zip(axes.flat, panels, strict=True):
    for condition in CONDITIONS:
        values = [curve(condition, seed, metric) for seed in SEEDS]
        for per_seed in values:
            ax.plot(CHECKPOINTS, per_seed, color=COLORS[condition], alpha=0.23, linewidth=1)
        means = [sum(row[i] for row in values) / len(values) for i in range(len(CHECKPOINTS))]
        ax.plot(
            CHECKPOINTS,
            means,
            marker="o",
            linewidth=2.4,
            color=COLORS[condition],
            label=LABELS[condition],
        )
    ax.set_title(title)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(CHECKPOINTS)
    ax.set_xlabel("Output-only RL step")
axes[0, 0].legend(frameon=True, fontsize=9)
fig.suptitle("Dense 27B capped-thinking curves (thin lines are raw seeds)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "fixed_checkpoint_curves.png", dpi=180, bbox_inches="tight")
plt.close(fig)


primary = RESULTS["summary"]["factorial_interactions"]["values_and_rationales_vs_irrelevant"]
metrics = ["hack_rate", "undetected_given_hack", "undetected_hack_rate"]
metric_labels = ["P(hack)", "P(undetected | hack)", "Undetected-hack rate"]
fig, ax = plt.subplots(figsize=(9, 4.8))
offsets = [-0.18, 0.0, 0.18]
markers = ["o", "s", "^"]
for seed, offset, marker in zip(SEEDS, offsets, markers, strict=True):
    values = []
    for metric in metrics:
        row = next(
            x
            for x in primary[metric]["paired_seed_interactions"]
            if x["seed"] == seed
        )
        values.append(row["interaction"])
    ax.scatter(
        [i + offset for i in range(len(metrics))],
        values,
        s=70,
        marker=marker,
        label=f"seed {seed}",
        zorder=3,
    )
for i, metric in enumerate(metrics):
    mean = primary[metric]["interaction"]["mean"]
    lo, hi = primary[metric]["interaction"]["ci95"]
    ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="D", color="black", capsize=5, zorder=4)
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(range(len(metrics)), metric_labels)
ax.set_ylabel("Values+rationales minus irrelevant\n(step-0→16 interaction)")
ax.set_title("Primary decomposition: more hacking, substantially less conditional nondetection")
ax.legend(ncol=3, frameon=True)
fig.tight_layout()
fig.savefig(OUT / "primary_interaction_decomposition.png", dpi=180, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 5))
control_names = ["chain", "action", "oracle", "no scratch"]
width = 0.18
for condition_index, condition in enumerate(CONDITIONS):
    rows = [r for r in CURVES if r["condition"] == condition and r["checkpoint"] == 16]
    values = [
        sum(r["undetected_hack_rate"] for r in rows) / len(rows),
        sum(r["monitors"]["action_only"]["undetected_hack_rate"] for r in rows) / len(rows),
        sum(r["monitors"]["oracle_context"]["undetected_hack_rate"] for r in rows) / len(rows),
        sum(r["controls"]["no_scratchpad_action_only"]["undetected_hack_rate"] for r in rows) / len(rows),
    ]
    positions = [i + (condition_index - 1) * width for i in range(len(control_names))]
    ax.bar(positions, values, width=width, color=COLORS[condition], label=LABELS[condition])
ax.set_xticks(range(len(control_names)), control_names)
ax.set_ylabel("Step-16 undetected-hack rate")
ax.set_ylim(0, 0.7)
ax.set_title("Information controls at the final fixed checkpoint")
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(OUT / "monitor_information_controls.png", dpi=180, bbox_inches="tight")
plt.close(fig)
