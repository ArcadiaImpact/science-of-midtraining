#!/usr/bin/env python3
"""Render preregistered lending semantics curves from inert artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "submission" / "curves.json"
FIGURES = ROOT / "submission" / "figures"
CONDITIONS = [
    "+SDF(values+rationales)",
    "+SDF(rules-only)",
    "-SDF(irrelevant)",
]
STYLE = {
    CONDITIONS[0]: ("#b3261e", "-", "values + rationales"),
    CONDITIONS[1]: ("#e67e22", "--", "rules only"),
    CONDITIONS[2]: ("#246eb9", "-.", "irrelevant mirror"),
}
PANELS = [
    ("undetected_hack_rate", "Undetected-hack rate", 0.58),
    ("hack_rate", "Hack rate", 0.72),
    ("undetected_given_hack", "P(undetected | hack)", 0.82),
    ("legitimate_task_success", "Legitimate task success", 1.02),
]


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[row["condition"]][row["seed"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    for ax, (metric, title, upper) in zip(axes.flat, PANELS, strict=True):
        for condition in CONDITIONS:
            color, linestyle, label = STYLE[condition]
            seed_rows = grouped[condition]
            for rows in seed_rows.values():
                rows = sorted(rows, key=lambda row: row["checkpoint"])
                ax.plot(
                    [row["checkpoint"] for row in rows],
                    [row[metric] for row in rows],
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.8,
                    alpha=0.22,
                )
            checkpoints = sorted({row["checkpoint"] for rows in seed_rows.values() for row in rows})
            means = [
                sum(
                    next(row[metric] for row in rows if row["checkpoint"] == checkpoint)
                    for rows in seed_rows.values()
                )
                / len(seed_rows)
                for checkpoint in checkpoints
            ]
            ax.plot(
                checkpoints,
                means,
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=4,
                linewidth=2.4,
                label=label,
            )
        ax.set_title(title)
        ax.set_ylim(-0.025, upper)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_ylabel("Rate")
        ax.grid(alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Output-only RL step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Lending: values and rationales amplify post-RL undetected violations",
        y=0.99,
    )
    fig.text(
        0.5,
        0.078,
        "Thick lines are three-seed means; faint lines are individual seeds. All checkpoints were fixed in advance.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"lending-semantics-curves.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
