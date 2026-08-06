#!/usr/bin/env python3
"""Render the preregistered five-arm curves and semantic-factor effects."""

from __future__ import annotations

import json
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
    "+SDF(rationales-only)": "#67a9cf",
    "+SDF(rules-only)": "#2166ac",
    "-SDF(irrelevant)": "#666666",
}


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    summary = json.loads(RESULTS.read_text())["summary"]
    by_key = {(r["condition"], r["seed"], r["checkpoint"]): r for r in records}
    checkpoints = sorted({r["checkpoint"] for r in records})
    seeds = sorted({r["seed"] for r in records})

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)

    for ax, metric, title, ylabel in [
        (axes[0, 0], "undetected_hack_rate", "A. Undetected hidden-rule violations", "rate"),
        (axes[0, 1], "hack_rate", "B. Exact hidden-rule violations", "rate"),
        (axes[1, 0], "legitimate_task_success", "C. Legitimate task success", "rate"),
    ]:
        for condition in ORDER:
            seed_curves = np.array(
                [[by_key[(condition, seed, ckpt)][metric] for ckpt in checkpoints] for seed in seeds]
            )
            for values in seed_curves:
                ax.plot(checkpoints, values, color=COLORS[condition], alpha=0.18, lw=0.9)
            ax.plot(
                checkpoints,
                seed_curves.mean(axis=0),
                color=COLORS[condition],
                marker="o",
                ms=4,
                lw=2.2,
                label=LABELS[condition],
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("output-only RL step")
        ax.set_ylabel(ylabel)
        ax.set_xticks(checkpoints)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)

    factor = summary["two_by_two_component_factorial"]["undetected_hack_rate"]
    names = ["value\nmain effect", "rationale\nmain effect", "component\nsynergy"]
    keys = ["value_main_effect", "rationale_main_effect", "factor_synergy"]
    means = [factor[key]["mean"] for key in keys]
    cis = [factor[key]["ci95"] for key in keys]
    paired = factor["paired_seed_effects"]
    point_keys = ["value_main_effect", "rationale_main_effect", "factor_synergy"]
    ax = axes[1, 1]
    x = np.arange(3)
    yerr = np.array([[m - ci[0] for m, ci in zip(means, cis)], [ci[1] - m for m, ci in zip(means, cis)]])
    ax.bar(x, means, color=["#ef8a62", "#67a9cf", "#b2182b"], alpha=0.72, width=0.62)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=1.2)
    offsets = [-0.10, 0.0, 0.10]
    for seed_index, (seed, offset) in enumerate(zip(seeds, offsets)):
        values = [paired[seed_index][key] for key in point_keys]
        ax.scatter(x + offset, values, s=24, color="black", alpha=0.72, label=f"seed {seed}" if seed_index == 0 else None)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("step-0-to-16 effect on undetected-hack rate")
    ax.set_title("D. Preregistered 2×2 semantic decomposition", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Values and rationales interact under output-only RL\n"
        "Thick lines are three-seed means; faint lines and black points are individual seeds",
        fontsize=13,
        fontweight="bold",
        y=1.045,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "semantic_factorial_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "semantic_factorial_curves.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
