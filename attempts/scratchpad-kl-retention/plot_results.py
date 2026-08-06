#!/usr/bin/env python3
"""Plot the scratchpad-KL factorial and its efficacy-gate failure."""

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
        "ordinary": "+SDF(values+rationales) / ordinary RL",
        "kl": "+SDF(values+rationales) / scratchpad-KL RL",
        "color": "#b2182b",
    },
    "rules only": {
        "ordinary": "+SDF(rules-only) / ordinary RL",
        "kl": "+SDF(rules-only) / scratchpad-KL RL",
        "color": "#2166ac",
    },
    "irrelevant SDF": {
        "ordinary": "-SDF(irrelevant) / ordinary RL",
        "kl": "-SDF(irrelevant) / scratchpad-KL RL",
        "color": "#666666",
    },
}


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    summary = json.loads(RESULTS.read_text())["summary"]
    by_key = {(r["condition"], r["seed"], r["checkpoint"]): r for r in records}
    checkpoints = sorted({r["checkpoint"] for r in records})
    seeds = sorted({r["seed"] for r in records})

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    panels = [
        (axes[0, 0], "undetected_hack_rate", "A. Undetected hidden-rule violations", "rate"),
        (axes[0, 1], "hack_rate", "B. Exact hidden-rule violations", "rate"),
        (axes[1, 0], "proxy_reward", "C. Public-output proxy reward", "reward"),
    ]
    for ax, metric, title, ylabel in panels:
        for sdf_label, spec in SDF.items():
            for mechanism, linestyle, marker in (
                ("ordinary", "-", "o"),
                ("kl", "--", "s"),
            ):
                condition = spec[mechanism]
                seed_curves = np.array(
                    [
                        [by_key[(condition, seed, ckpt)][metric] for ckpt in checkpoints]
                        for seed in seeds
                    ]
                )
                for values in seed_curves:
                    ax.plot(
                        checkpoints,
                        values,
                        color=spec["color"],
                        alpha=0.13,
                        lw=0.8,
                        ls=linestyle,
                    )
                ax.plot(
                    checkpoints,
                    seed_curves.mean(axis=0),
                    color=spec["color"],
                    ls=linestyle,
                    marker=marker,
                    ms=4,
                    lw=2.1,
                    label=f"{sdf_label}: {'ordinary' if mechanism == 'ordinary' else 'scratchpad KL'}",
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
        f"3-way attenuation = {primary['mean']:+.3f}\n"
        f"95% interval [{primary['ci95'][0]:+.3f}, {primary['ci95'][1]:+.3f}]",
        transform=axes[0, 0].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    efficacy = summary["preregistered_proxy_learning_efficacy_gate"]
    axes[1, 0].text(
        0.03,
        0.05,
        "Proxy-learning gate FAIL\n"
        f"ordinary change {efficacy['ordinary_rl_mean_proxy_change']:+.3f}; "
        f"KL change {efficacy['scratchpad_kl_mean_proxy_change']:+.3f}",
        transform=axes[1, 0].transAxes,
        va="bottom",
        color="#8b0000",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#8b0000", "alpha": 0.9},
    )

    gate = summary["preregistered_all_rollout_retention_gate"]
    order = ["+SDF(lending-spec)", "+SDF(lending-rules-only)", "-SDF(irrelevant)"]
    labels = ["values +\nrationales", "rules only", "irrelevant\nSDF"]
    retention = [gate["by_sdf_condition"][name]["retention_rate"] for name in order]
    closure = [gate["by_sdf_condition"][name]["closing_boundary_rate"] for name in order]
    x = np.arange(3)
    width = 0.34
    ax = axes[1, 1]
    ax.bar(x - width / 2, closure, width, color="#aaaaaa", label="closing boundary")
    ax.bar(x + width / 2, retention, width, color="#2ca25f", label="rollout retained")
    for positions, values in ((x - width / 2, closure), (x + width / 2, retention)):
        for position, value in zip(positions, values, strict=True):
            ax.text(position, value + 0.002, f"{100 * value:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.94, 1.008)
    ax.set_ylabel("fraction of training rollouts")
    ax.set_title("D. Every rollout retained despite missing delimiters", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
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
        "Scratchpad KL removes lending evasion but also proxy learning\n"
        "Thick lines are three-seed means; faint lines are individual seeds",
        fontsize=13,
        fontweight="bold",
        y=1.045,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "lending_scratchpad_kl.pdf", bbox_inches="tight")
    fig.savefig(OUT / "lending_scratchpad_kl.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
