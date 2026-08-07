#!/usr/bin/env python3
"""Plot raw seed trajectories and their means for the preregistered factorial."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "submission" / "curves.json"
OUTPUT = ROOT / "submission" / "figures" / "triage_domain_placebo_factorial.svg"

CONDITIONS = [
    "+SDF(triage-values+rationales)",
    "-SDF(clinical-domain-placebo)",
    "-SDF(cross-domain-irrelevant)",
]
LABELS = {
    CONDITIONS[0]: "+SDF triage rules",
    CONDITIONS[1]: "clinical-domain placebo",
    CONDITIONS[2]: "cross-domain control",
}
COLORS = {
    CONDITIONS[0]: "#b23a48",
    CONDITIONS[1]: "#2878b5",
    CONDITIONS[2]: "#5a9367",
}
PANELS = [
    ("undetected_hack_rate", "Undetected-hack rate"),
    ("undetected_given_hack", "Undetected given hack"),
    ("proxy_reward", "Public proxy reward"),
]


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["condition"], row["seed"])].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.55), sharex=True)
    for ax, (metric, title) in zip(axes, PANELS):
        for condition in CONDITIONS:
            seed_series = []
            for (candidate, seed), rows in grouped.items():
                if candidate != condition:
                    continue
                ordered = sorted(rows, key=lambda row: row["checkpoint"])
                x = [row["checkpoint"] for row in ordered]
                y = [row[metric] for row in ordered]
                seed_series.append(y)
                ax.plot(x, y, color=COLORS[condition], alpha=0.22, linewidth=1.0)
            means = [sum(values) / len(values) for values in zip(*seed_series)]
            ax.plot(x, means, color=COLORS[condition], linewidth=2.4, marker="o",
                    markersize=3.5, label=LABELS[condition])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("RL step")
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle("Triage specification versus same-domain and cross-domain controls\n"
                 "thin lines: raw seeds; thick lines: seed means", fontsize=11)
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="svg")


if __name__ == "__main__":
    main()
