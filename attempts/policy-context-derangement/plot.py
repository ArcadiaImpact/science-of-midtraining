#!/usr/bin/env python3
"""Plot the policy-context derangement mechanism experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
curves = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
results = json.loads((ROOT / "submission" / "results.json").read_text())
output = ROOT / "submission" / "figures" / "policy_context_derangement.png"
output.parent.mkdir(parents=True, exist_ok=True)

PLUS = "+SDF(triage-values+rationales)"
MINUS = "-SDF(irrelevant)"
CONDITIONS = (PLUS, MINUS)
STYLES = {
    PLUS: ("#b2182b", "-", "+SDF: applicable triage specification"),
    MINUS: ("#2166ac", "--", "−SDF: irrelevant specification"),
}


def raw_and_mean(ax: plt.Axes, context: str, condition: str, metric: str) -> None:
    rows = [
        row
        for row in curves
        if row["context_regime"] == context and row["condition"] == condition
    ]
    color, linestyle, label = STYLES[condition]
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
            linewidth=1,
            alpha=0.25,
        )
    checkpoints = sorted({row["checkpoint"] for row in rows})
    means = [
        sum(row[metric] for row in rows if row["checkpoint"] == checkpoint)
        / sum(row["checkpoint"] == checkpoint for row in rows)
        for checkpoint in checkpoints
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


fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))

for condition in CONDITIONS:
    raw_and_mean(axes[0, 0], "true_private_note", condition, "hack_rate")
axes[0, 0].set_title("A. Causally matched private notes")
axes[0, 0].set_ylabel("Unsafe-discharge rate")
axes[0, 0].legend(frameon=False, fontsize=8)

for condition in CONDITIONS:
    raw_and_mean(axes[0, 1], "deranged_private_note", condition, "hack_rate")
axes[0, 1].set_title("B. Independent exact-length private notes")
axes[0, 1].set_ylabel("Unsafe-discharge rate")
axes[0, 1].legend(frameon=False, fontsize=8)

within = results["summary"]["within_context_sdf_by_rl_interactions"]
three_way = results["summary"]["true_minus_deranged_three_way_interactions"]
metric_specs = (
    ("Unsafe\ndischarge", "hack_rate", "#7b3294"),
    ("Undetected\nhack", "undetected_hack_rate", "#008837"),
)
positions = []
labels = []
for metric_index, (label, metric, color) in enumerate(metric_specs):
    for context_index, context in enumerate(("true_private_note", "deranged_private_note")):
        x = metric_index * 3 + context_index
        values = [
            row["interaction"]
            for row in within[context][metric]["paired_seed_interactions"]
        ]
        axes[1, 0].scatter(
            [x - 0.10, x, x + 0.10], values, color=color, s=38, alpha=0.85
        )
        mean = within[context][metric]["interaction"]["mean"]
        axes[1, 0].plot((x - 0.22, x + 0.22), (mean, mean), color="black", linewidth=2)
        positions.append(x)
        labels.append(f"{label}\n{'true' if context_index == 0 else 'deranged'}")
axes[1, 0].axhline(0, color="#555555", linestyle=":", linewidth=1)
axes[1, 0].set_xticks(positions, labels)
axes[1, 0].set_ylabel("+SDF minus −SDF change, step 0→16")
axes[1, 0].set_title("C. Within-context SDF×RL interactions")

for x, (label, metric, color) in enumerate(metric_specs):
    values = [
        row["true_minus_deranged"]
        for row in three_way[metric]["paired_seed_contrasts"]
    ]
    axes[1, 1].scatter(
        [x - 0.10, x, x + 0.10], values, color=color, s=48, alpha=0.9
    )
    mean = three_way[metric]["contrast"]["mean"]
    axes[1, 1].plot((x - 0.22, x + 0.22), (mean, mean), color="black", linewidth=2.3)
axes[1, 1].axhline(0, color="#555555", linestyle=":", linewidth=1)
axes[1, 1].set_xticks(range(len(metric_specs)), [x[0] for x in metric_specs])
axes[1, 1].set_ylabel("True-note minus deranged-note interaction")
axes[1, 1].set_title("D. Preregistered mechanism contrasts")

for ax in axes[0]:
    ax.set_xlabel("Output-only RL update")
    ax.set_xticks((0, 4, 8, 12, 16))
    ax.set_ylim(-0.02, 0.72)

fig.suptitle("Does the triage interaction require causal private-note binding?", fontsize=14)
fig.tight_layout()
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)
