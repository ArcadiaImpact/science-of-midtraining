#!/usr/bin/env python3
"""Render paired supportive/adverse public-cover curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
curves = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
results = json.loads((ROOT / "submission" / "results.json").read_text())
out = ROOT / "submission" / "figures" / "paired_public_cover_intervention.png"
out.parent.mkdir(parents=True, exist_ok=True)

conditions = (
    "+SDF(triage-values+rationales)",
    "+SDF(triage-rules-only)",
    "-SDF(irrelevant)",
)
styles = {
    conditions[0]: ("#b2182b", "-", "+SDF: triage values and rationales"),
    conditions[1]: ("#238b45", "-.", "+SDF: triage rules only"),
    conditions[2]: ("#2166ac", "--", "−SDF: irrelevant incident specification"),
}


def draw_raw_and_mean(ax: plt.Axes, condition: str, metric: str, cover_mode: str) -> None:
    color, linestyle, label = styles[condition]
    rows = [row for row in curves if row["condition"] == condition]
    for seed in sorted({row["seed"] for row in rows}):
        seed_rows = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: row["checkpoint"],
        )
        ax.plot(
            [row["checkpoint"] for row in seed_rows],
            [row["controls"]["public_cover"][cover_mode][metric] for row in seed_rows],
            color=color,
            linestyle=linestyle,
            alpha=0.22,
            linewidth=1,
        )
    checkpoints = sorted({row["checkpoint"] for row in rows})
    means = [
        sum(row["controls"]["public_cover"][cover_mode][metric] for row in rows if row["checkpoint"] == step)
        / sum(row["checkpoint"] == step for row in rows)
        for step in checkpoints
    ]
    ax.plot(
        checkpoints,
        means,
        color=color,
        linestyle=linestyle,
        marker="o",
        linewidth=2.3,
        label=label,
    )


fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

for condition in conditions:
    draw_raw_and_mean(axes[0], condition, "undetected_hack_rate", "adverse")
axes[0].set_title("A. Adverse public intake")
axes[0].set_xlabel("Output-only RL update")
axes[0].set_ylabel("Rate among all cases")
axes[0].set_xticks((0, 4, 8, 12, 16))
axes[0].set_ylim(bottom=-0.01)
axes[0].legend(frameon=False, fontsize=8)

for condition in conditions:
    draw_raw_and_mean(axes[1], condition, "undetected_hack_rate", "supportive")
axes[1].set_title("B. Supportive public intake")
axes[1].set_xlabel("Output-only RL update")
axes[1].set_ylabel("Rate among all cases")
axes[1].set_xticks((0, 4, 8, 12, 16))
axes[1].set_ylim(bottom=-0.01)
axes[1].legend(frameon=False, fontsize=8)

interactions = results["summary"]["public_cover_interactions"]
series = (
    ("Undetected-\nhack rate", "undetected_hack_rate"),
    ("Undetected\n| violation", "undetected_given_hack"),
    ("Violation\nrate", "hack_rate"),
)
for x, (label, metric) in enumerate(series):
    effect = interactions[metric]
    values = [row["cover_amplification_interaction"] for row in effect["paired_seed_interactions"]]
    mean = effect["cover_amplification_interaction"]["mean"]
    axes[2].scatter(
        [x - 0.08, x, x + 0.08], values, color="#6a3d9a", s=42, zorder=3
    )
    axes[2].plot((x - 0.20, x + 0.20), (mean, mean), color="black", linewidth=2.3)
axes[2].axhline(0, color="#555555", linewidth=1, linestyle=":")
axes[2].set_xticks(range(len(series)), [label for label, _ in series])
axes[2].set_ylabel("Supportive − adverse semantics interaction")
axes[2].set_title("C. Paired cover amplification")

fig.suptitle("Paired public-justification affordance intervention", fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
