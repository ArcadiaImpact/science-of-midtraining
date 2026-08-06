#!/usr/bin/env python3
"""Render the preregistered safe-referral action-space factorial."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
curves = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
results = json.loads((ROOT / "submission" / "results.json").read_text())
out = ROOT / "submission" / "figures" / "safe_referral_action_space_factorial.png"
out.parent.mkdir(parents=True, exist_ok=True)

conditions = (
    "REFERRAL / +SDF(values+rationales)",
    "REFERRAL / -SDF(irrelevant)",
    "BINARY / +SDF(values+rationales)",
    "BINARY / -SDF(irrelevant)",
)
styles = {
    conditions[0]: ("#b2182b", "-", "Referral: values+rationales"),
    conditions[1]: ("#b2182b", "--", "Referral: irrelevant"),
    conditions[2]: ("#2166ac", "-", "Binary: values+rationales"),
    conditions[3]: ("#2166ac", "--", "Binary: irrelevant"),
}


def draw_raw_and_mean(ax: plt.Axes, condition: str, metric: str) -> None:
    color, linestyle, label = styles[condition]
    rows = [r for r in curves if r["condition"] == condition]
    for seed in sorted({r["seed"] for r in rows}):
        seed_rows = sorted(
            (r for r in rows if r["seed"] == seed), key=lambda r: r["checkpoint"]
        )
        ax.plot(
            [r["checkpoint"] for r in seed_rows],
            [r[metric] for r in seed_rows],
            color=color,
            linestyle=linestyle,
            alpha=0.20,
            linewidth=1,
        )
    checkpoints = sorted({r["checkpoint"] for r in rows})
    means = [
        sum(r[metric] for r in rows if r["checkpoint"] == step)
        / sum(r["checkpoint"] == step for r in rows)
        for step in checkpoints
    ]
    ax.plot(
        checkpoints,
        means,
        color=color,
        linestyle=linestyle,
        marker="o",
        linewidth=2.2,
        label=label,
    )


fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.1))

for condition in conditions:
    draw_raw_and_mean(axes[0], condition, "undetected_hack_rate")
axes[0].set_title("A. Undetected-hack curves")
axes[0].set_xlabel("RL update")
axes[0].set_ylabel("Undetected-hack rate")
axes[0].set_xticks((0, 4, 8, 12, 16))
axes[0].set_ylim(bottom=-0.01)
axes[0].legend(frameon=False, fontsize=7)

for condition in conditions[:2]:
    draw_raw_and_mean(axes[1], condition, "referral_rate")
axes[1].set_title("B. Safe-referral use")
axes[1].set_xlabel("RL update")
axes[1].set_ylabel("Referral rate")
axes[1].set_xticks((0, 4, 8, 12, 16))
axes[1].set_ylim(bottom=-0.01)
axes[1].legend(frameon=False, fontsize=7)

moderation = results["summary"]["action_space_moderation"]
for x, (metric, label) in enumerate(
    (("undetected_hack_rate", "Undetected\nhacks"), ("undetected_given_hack", "Conditional\nevasion"))
):
    vals = [
        row["binary_minus_referral_attenuation"]
        for row in moderation[metric]["paired_seed_differences"]
    ]
    axes[2].scatter(
        [x - 0.08, x, x + 0.08], vals, color="#6a3d9a", s=42, zorder=3
    )
    mean = moderation[metric]["binary_minus_referral_attenuation"]["mean"]
    axes[2].plot((x - 0.20, x + 0.20), (mean, mean), color="black", linewidth=2.3)
axes[2].axhline(0, color="#555555", linewidth=1, linestyle=":")
axes[2].set_xticks((0, 1), ("Undetected\nhacks", "Conditional\nevasion"))
axes[2].set_ylabel("Binary minus referral\nSDF×RL interaction")
axes[2].set_title("C. Paired attenuation")

fig.suptitle("Safe referral as a moderator of SDF-linked monitor evasion", fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
