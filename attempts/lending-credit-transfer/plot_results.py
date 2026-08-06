#!/usr/bin/env python3
"""Render preregistered lending credit-assignment curves from inert artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "submission" / "curves.json"
FIGURES = ROOT / "submission" / "figures"

CONDITIONS = [
    "+SDF(lending-spec) / sequence-wide credit",
    "+SDF(lending-spec) / public-token credit",
    "-SDF(irrelevant) / sequence-wide credit",
    "-SDF(irrelevant) / public-token credit",
]
STYLE = {
    CONDITIONS[0]: ("#c73e1d", "-", "+SDF, sequence-wide"),
    CONDITIONS[1]: ("#c73e1d", "--", "+SDF, public-token"),
    CONDITIONS[2]: ("#246eb9", "-", "irrelevant SDF, sequence-wide"),
    CONDITIONS[3]: ("#246eb9", "--", "irrelevant SDF, public-token"),
}
PANELS = [
    ("undetected_hack_rate", "Undetected-hack rate"),
    ("hack_rate", "Hack rate"),
    ("undetected_given_hack", "P(undetected | hack)"),
    ("legitimate_task_success", "Legitimate task success"),
]


def main() -> None:
    payload = json.loads(CURVES.read_text())
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in payload["records"]:
        grouped[row["condition"]][row["seed"]].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    for ax, (metric, title) in zip(axes.flat, PANELS, strict=True):
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
                linewidth=2.3,
                label=label,
            )
        ax.set_title(title)
        ax.set_ylim(-0.025, 1.025 if metric == "legitimate_task_success" else 0.68)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.grid(alpha=0.2)
        ax.set_ylabel("Rate")

    for ax in axes[-1]:
        ax.set_xlabel("Output-only RL step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Lending: public-token credit suppresses hacking, but not specifically conditional evasion",
        y=0.99,
    )
    fig.text(
        0.5,
        0.082,
        "Thick lines are three-seed means; faint lines are individual seeds. Checkpoints were fixed before evaluation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.135, 1, 0.96))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"lending-credit-curves.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
