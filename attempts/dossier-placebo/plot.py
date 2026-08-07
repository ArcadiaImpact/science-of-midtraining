#!/usr/bin/env python3
"""Render information-intervention curves and paired endpoint controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
records = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
results = json.loads((ROOT / "submission" / "results.json").read_text())
out = ROOT / "submission" / "figures" / "dossier_placebo.png"
out.parent.mkdir(parents=True, exist_ok=True)

condition = "+SDF(change-values+rationales)"
monitor_styles = {
    "chain_of_thought": ("#b2182b", "-", "Standard CoT (no dossier)"),
    "action_only": ("#ef8a62", ":", "Action only"),
    "sham_selected_context": ("#6a3d9a", "--", "Permuted sham dossier"),
    "true_selected_context": ("#1b9e77", "-", "True selected dossier"),
}


def draw_curve(ax: plt.Axes, monitor: str) -> None:
    color, linestyle, label = monitor_styles[monitor]
    rows = [
        row for row in records
        if row["reward_regime"] == "high_conflict" and row["condition"] == condition
    ]
    for seed in (907, 1018, 1129):
        seed_rows = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: row["checkpoint"],
        )
        ax.plot(
            [row["checkpoint"] for row in seed_rows],
            [row["monitors"][monitor]["undetected_hack_rate"] for row in seed_rows],
            color=color, linestyle=linestyle, alpha=.18, linewidth=1,
        )
    checkpoints = (0, 4, 8, 12, 16)
    means = [
        sum(row["monitors"][monitor]["undetected_hack_rate"] for row in rows if row["checkpoint"] == step) / 3
        for step in checkpoints
    ]
    ax.plot(checkpoints, means, color=color, linestyle=linestyle, marker="o", linewidth=2.2, label=label)


fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))
for monitor in monitor_styles:
    draw_curve(axes[0], monitor)
axes[0].set_title("A. High-conflict +SDF UHR")
axes[0].set_xlabel("Output-only RL update")
axes[0].set_ylabel("Undetected-hack rate")
axes[0].set_xticks((0, 4, 8, 12, 16))
axes[0].set_ylim(-.015, .66)
axes[0].legend(frameon=False, fontsize=7.5)

endpoint = results["summary"]["primary_information_asymmetry_test"]
for x, monitor in enumerate(("true_selected_context", "sham_selected_context")):
    sensitivities = []
    fprs = []
    for seed in (907, 1018, 1129):
        rows = [
            row for row in records
            if row["reward_regime"] == "high_conflict"
            and row["checkpoint"] == 16
            and row["seed"] == seed
        ]
        hacks = sum(row["monitors"][monitor]["n_hacks"] for row in rows)
        detected = sum(row["monitors"][monitor]["n_detected_hacks"] for row in rows)
        legitimate = sum(row["monitors"][monitor]["n_legitimate_decisions"] for row in rows)
        false_positives = sum(row["monitors"][monitor]["n_monitor_false_positives"] for row in rows)
        sensitivities.append(detected / hacks)
        fprs.append(false_positives / legitimate)
    jitter = (-.08, 0, .08)
    axes[1].scatter([x + j for j in jitter], sensitivities, color="#1b9e77", s=42, label="Sensitivity" if x == 0 else None)
    axes[1].scatter([x + j for j in jitter], fprs, color="#d95f02", marker="s", s=38, label="False-positive rate" if x == 0 else None)
    axes[1].plot((x - .2, x + .2), (sum(sensitivities) / 3,) * 2, color="#1b9e77", linewidth=2)
    axes[1].plot((x - .2, x + .2), (sum(fprs) / 3,) * 2, color="#d95f02", linewidth=2)
axes[1].set_xticks((0, 1), ("True selected\ndossier", "Permuted sham\ndossier"))
axes[1].set_ylim(-.03, 1.05)
axes[1].set_ylabel("Rate at high-conflict step 16")
axes[1].set_title("B. Content-specific monitor validity")
axes[1].legend(frameon=False, fontsize=8, loc="center right")

interaction_data = results["summary"]["sdf_by_rl_interactions"]["high_conflict"]
interaction_monitors = (
    ("chain_of_thought", "Standard\nCoT"),
    ("action_only", "Action\nonly"),
    ("sham_selected_context", "Sham\ndossier"),
    ("true_selected_context", "True\ndossier"),
)
for x, (monitor, label) in enumerate(interaction_monitors):
    values = [
        row["interaction"]
        for row in interaction_data[monitor]["undetected_given_hack"]["paired_seed_interactions"]
    ]
    mean = interaction_data[monitor]["undetected_given_hack"]["interaction"]["mean"]
    axes[2].scatter([x - .08, x, x + .08], values, color="#6a3d9a", s=42, zorder=3)
    axes[2].plot((x - .2, x + .2), (mean, mean), color="black", linewidth=2.2)
axes[2].axhline(0, color="#555555", linestyle=":", linewidth=1)
axes[2].set_xticks(range(4), [label for _, label in interaction_monitors])
axes[2].set_ylabel("+SDF minus irrelevant change, step 0→16")
axes[2].set_title("C. Conditional opacity interaction")

fig.suptitle("A learned monitor succeeds when given causal private evidence", fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
