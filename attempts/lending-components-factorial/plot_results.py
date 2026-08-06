#!/usr/bin/env python3
"""Plot the five semantic-component curves and the preregistered factorial effects."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "submission" / "curves.json"
RESULTS = ROOT / "submission" / "results.json"
OUT = ROOT / "submission" / "figures"

ORDER = [
    "+SDF(values+rationales)",
    "+SDF(values-only)",
    "+SDF(rationales-only)",
    "+SDF(rules-only)",
    "-SDF(irrelevant)",
]
LABELS = {
    "+SDF(values+rationales)": "values + rationales",
    "+SDF(values-only)": "values only",
    "+SDF(rationales-only)": "rationales only",
    "+SDF(rules-only)": "rules only",
    "-SDF(irrelevant)": "irrelevant SDF",
}
COLORS = {
    "+SDF(values+rationales)": "#b2182b",
    "+SDF(values-only)": "#ef8a62",
    "+SDF(rationales-only)": "#2166ac",
    "+SDF(rules-only)": "#67a9cf",
    "-SDF(irrelevant)": "#666666",
}


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    summary = json.loads(RESULTS.read_text())["summary"]
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["condition"], row["seed"])].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
    curve_panels = [
        (axes[0, 0], "undetected_hack_rate", "Undetected-hack rate"),
        (axes[0, 1], "hack_rate", "P(hack)"),
        (axes[1, 0], "undetected_given_hack", "P(undetected | hack)"),
    ]
    for ax, metric, title in curve_panels:
        for condition in ORDER:
            seed_series = []
            for seed in (714, 825, 936):
                rows = sorted(grouped[(condition, seed)], key=lambda x: x["checkpoint"])
                xs = [x["checkpoint"] for x in rows]
                ys = [x[metric] for x in rows]
                seed_series.append(ys)
                ax.plot(xs, ys, color=COLORS[condition], alpha=0.18, linewidth=1.1)
            ax.plot(
                xs,
                np.mean(seed_series, axis=0),
                color=COLORS[condition],
                linewidth=2.8,
                marker="o",
                markersize=4,
                label=LABELS[condition],
            )
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Output-only RL step")
        ax.set_ylabel("Rate")
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_ylim(-0.025, 1.025)
        ax.grid(alpha=0.2)

    effects = summary["factorial_effects"]
    metrics = ["undetected_hack_rate", "hack_rate", "undetected_given_hack"]
    names = ["Undetected\nhack", "P(hack)", "Conditional\nmiss"]
    x = np.arange(len(metrics))
    width = 0.24
    effect_specs = [
        ("causal_rationale_main_effect", "rationale main", "#2166ac"),
        ("values_main_effect", "values main", "#ef8a62"),
        ("factorial_interaction", "2×2 interaction", "#7b3294"),
    ]
    ax = axes[1, 1]
    for offset, (key, label, color) in zip((-width, 0.0, width), effect_specs, strict=True):
        means = [effects[metric][key]["mean"] for metric in metrics]
        lows = [effects[metric][key]["ci95"][0] for metric in metrics]
        highs = [effects[metric][key]["ci95"][1] for metric in metrics]
        yerr = np.array([[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]])
        ax.bar(x + offset, means, width, label=label, color=color, alpha=0.88)
        ax.errorbar(x + offset, means, yerr=yerr, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("Step-0-to-16 factorial effect")
    ax.set_title("Preregistered 2×2 decomposition", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Lending: separating values from causal rationales", fontsize=16, fontweight="bold", y=1.015)
    fig.text(0.01, 0.005, "Thick lines: 3-seed means; faint lines: individual seeds. Intervals bootstrap paired seed units.", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.025, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"lending-components-factorial.{suffix}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
