#!/usr/bin/env python3
"""Plot the credit-location factorial and its failed boundary gate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CURVES = ROOT / "submission" / "curves.json"
RESULTS = ROOT / "submission" / "results.json"
OUT = ROOT / "submission" / "figures"

SDF = {
    "values+rationales": {
        "sequence": "+SDF(values+rationales) / sequence-wide credit",
        "public": "+SDF(values+rationales) / public-token credit",
        "color": "#b2182b",
    },
    "rules only": {
        "sequence": "+SDF(rules-only) / sequence-wide credit",
        "public": "+SDF(rules-only) / public-token credit",
        "color": "#2166ac",
    },
    "irrelevant SDF": {
        "sequence": "-SDF(irrelevant) / sequence-wide credit",
        "public": "-SDF(irrelevant) / public-token credit",
        "color": "#666666",
    },
}


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    summary = json.loads(RESULTS.read_text())["summary"]
    by_key = {(r["condition"], r["seed"], r["checkpoint"]): r for r in records}
    checkpoints = sorted({r["checkpoint"] for r in records})
    seeds = sorted({r["seed"] for r in records})

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    panels = [
        (axes[0, 0], "undetected_hack_rate", "A. Undetected hidden-rule violations", "rate"),
        (axes[0, 1], "hack_rate", "B. Exact hidden-rule violations", "rate"),
        (axes[1, 0], "proxy_reward", "C. Public-output proxy reward", "reward"),
    ]
    for ax, metric, title, ylabel in panels:
        for sdf_label, spec in SDF.items():
            for credit, linestyle, marker in (
                ("sequence", "-", "o"),
                ("public", "--", "s"),
            ):
                condition = spec[credit]
                seed_curves = np.array(
                    [[by_key[(condition, seed, ckpt)][metric] for ckpt in checkpoints] for seed in seeds]
                )
                for values in seed_curves:
                    ax.plot(checkpoints, values, color=spec["color"], alpha=0.13, lw=0.8, ls=linestyle)
                ax.plot(
                    checkpoints,
                    seed_curves.mean(axis=0),
                    color=spec["color"],
                    ls=linestyle,
                    marker=marker,
                    ms=4,
                    lw=2.1,
                    label=f"{sdf_label}: {'sequence-wide' if credit == 'sequence' else 'public-token'}",
                )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("output-only RL step")
        ax.set_ylabel(ylabel)
        ax.set_xticks(checkpoints)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)

    primary = summary["interaction"]
    axes[0, 0].text(
        0.03,
        0.96,
        f"3-way attenuation = {primary['mean']:+.3f}\n95% interval [{primary['ci95'][0]:+.3f}, {primary['ci95'][1]:+.3f}]",
        transform=axes[0, 0].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )

    gate = summary["preregistered_credit_boundary_gate"]
    order = ["+SDF(lending-spec)", "+SDF(lending-rules-only)", "-SDF(irrelevant)"]
    labels = ["values +\nrationales", "rules only", "irrelevant\nSDF"]
    rates = [gate["by_sdf_condition"][name]["success_rate"] for name in order]
    ax = axes[1, 1]
    ax.bar(np.arange(3), rates, color=["#b2182b", "#2166ac", "#666666"], alpha=0.78)
    ax.axhline(0.98, color="black", ls="--", lw=1.2, label="preregistered 98% gate")
    for x, value in enumerate(rates):
        ax.text(x, value + 0.003, f"{100 * value:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylim(0.88, 1.005)
    ax.set_ylabel("usable boundary fraction")
    ax.set_title("D. Boundary validity fails and differs by SDF", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.18)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Public-token credit attenuates lending evasion but fails its validity gate\n"
        "Thick lines are three-seed means; faint lines are individual seeds",
        fontsize=13,
        fontweight="bold",
        y=1.045,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "lending_credit_masking.pdf", bbox_inches="tight")
    fig.savefig(OUT / "lending_credit_masking.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
