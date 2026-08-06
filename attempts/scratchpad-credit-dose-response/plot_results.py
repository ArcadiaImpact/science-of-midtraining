#!/usr/bin/env python3
"""Plot scratchpad-credit dose curves and the preregistered interaction slope."""

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
    "values+rationales": {"prefix": "+SDF(values+rationales)", "color": "#b2182b"},
    "rules only": {"prefix": "+SDF(rules-only)", "color": "#2166ac"},
    "irrelevant SDF": {"prefix": "-SDF(irrelevant)", "color": "#666666"},
}
DOSES = {
    1.0: {"text": "1.00", "linestyle": "-", "marker": "o"},
    0.5: {"text": "0.50", "linestyle": "--", "marker": "s"},
    0.25: {"text": "0.25", "linestyle": ":", "marker": "^"},
}


def condition(prefix: str, dose: float) -> str:
    return f"{prefix} / scratchpad-credit {DOSES[dose]['text']} RL"


def main() -> None:
    records = json.loads(CURVES.read_text())["records"]
    summary = json.loads(RESULTS.read_text())["summary"]
    by_key = {(r["condition"], r["seed"], r["checkpoint"]): r for r in records}
    checkpoints = sorted({r["checkpoint"] for r in records})
    seeds = sorted({r["seed"] for r in records})

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.7), constrained_layout=True)
    panels = [
        (axes[0, 0], "undetected_hack_rate", "A. Undetected hidden-rule violations", "rate"),
        (axes[0, 1], "hack_rate", "B. Exact hidden-rule violations", "rate"),
        (axes[1, 0], "proxy_reward", "C. Public-output proxy reward", "reward"),
    ]
    for ax, metric, title, ylabel in panels:
        for sdf_label, spec in SDF.items():
            for dose, dose_spec in DOSES.items():
                arm = condition(spec["prefix"], dose)
                seed_curves = np.array(
                    [
                        [by_key[(arm, seed, checkpoint)][metric] for checkpoint in checkpoints]
                        for seed in seeds
                    ]
                )
                for values in seed_curves:
                    ax.plot(
                        checkpoints,
                        values,
                        color=spec["color"],
                        alpha=0.10,
                        lw=0.8,
                        ls=dose_spec["linestyle"],
                    )
                ax.plot(
                    checkpoints,
                    seed_curves.mean(axis=0),
                    color=spec["color"],
                    ls=dose_spec["linestyle"],
                    marker=dose_spec["marker"],
                    ms=3.5,
                    lw=2.0,
                    label=f"{sdf_label}: λ={dose_spec['text']}",
                )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("output-only RL step")
        ax.set_ylabel(ylabel)
        ax.set_xticks(checkpoints)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)

    efficacy = summary["preregistered_proxy_learning_efficacy_gate"]
    gate_word = "PASS" if efficacy["passed"] else "FAIL"
    gate_color = "#166534" if efficacy["passed"] else "#8b0000"
    axes[1, 0].text(
        0.03,
        0.05,
        f"Midpoint proxy gate {gate_word}\n"
        f"λ=1 change {efficacy['ordinary_rl_mean_proxy_change']:+.3f}; "
        f"λ=.5 change {efficacy['half_credit_mean_proxy_change']:+.3f}",
        transform=axes[1, 0].transAxes,
        va="bottom",
        color=gate_color,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": gate_color, "alpha": 0.9},
    )

    ax = axes[1, 1]
    slope = summary["dose_response"]["undetected_hack_rate"]
    paired = slope["paired_seed_slopes"]
    x = np.array([0.25, 0.5, 1.0])
    seed_values = []
    for row in paired:
        values = np.array(
            [row["interaction_by_scratchpad_credit_coefficient"][str(value)] for value in x]
        )
        seed_values.append(values)
        ax.plot(x, values, color="#7b3294", alpha=0.28, marker="o", lw=1.1)
    mean_values = np.array(seed_values).mean(axis=0)
    ax.plot(x, mean_values, color="#7b3294", marker="o", lw=2.6, label="three-seed mean")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("scratchpad-credit coefficient λ")
    ax.set_ylabel("values − rules change interaction")
    ax.set_title("D. Preregistered dose-response estimand", loc="left", fontweight="bold")
    ax.grid(alpha=0.18)
    ax.text(
        0.03,
        0.95,
        f"OLS slope = {slope['slope']['mean']:+.3f}\n"
        f"95% interval [{slope['slope']['ci95'][0]:+.3f}, {slope['slope']['ci95'][1]:+.3f}]",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    ax.legend(frameon=False, loc="lower right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.suptitle(
        "Scratchpad-credit dose response in private-note lending\n"
        "Thick lines are three-seed means; faint lines are individual seeds",
        fontsize=13,
        fontweight="bold",
        y=1.045,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "lending_scratchpad_credit_dose_response.pdf", bbox_inches="tight")
    fig.savefig(OUT / "lending_scratchpad_credit_dose_response.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
