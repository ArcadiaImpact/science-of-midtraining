#!/usr/bin/env python3
"""Render the preregistered semantic-by-credit contrasts from compact results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission" / "results.json"
OUT = ROOT / "submission" / "figures" / "semantic_credit.png"
STEPS = [0, 4, 8, 12, 16]
COLORS = {
    "sequence": "#c84c4c",
    "public": "#3569a8",
    "control_sequence": "#d69353",
    "control_public": "#6d92bd",
}


def main() -> None:
    results = json.loads(RESULTS.read_text())
    summary = results["summary"]
    curves = summary["aggregate_curves"]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    curve_specs = [
        ("+SDF(values+rationales) / sequence-wide credit", "rich / sequence", COLORS["sequence"]),
        ("+SDF(values+rationales) / public-token credit", "rich / public-token", COLORS["public"]),
        ("-SDF(irrelevant) / sequence-wide credit", "irrelevant / sequence", COLORS["control_sequence"]),
        ("-SDF(irrelevant) / public-token credit", "irrelevant / public-token", COLORS["control_public"]),
    ]
    for condition, label, color in curve_specs:
        means = [curves[condition][str(step)]["undetected_hack_rate"]["mean"] for step in STEPS]
        lows = [curves[condition][str(step)]["undetected_hack_rate"]["ci95"][0] for step in STEPS]
        highs = [curves[condition][str(step)]["undetected_hack_rate"]["ci95"][1] for step in STEPS]
        ax.plot(STEPS, means, marker="o", linewidth=2.2, label=label, color=color)
        ax.fill_between(STEPS, lows, highs, color=color, alpha=0.10)
    ax.set_title("A. Fixed-checkpoint undetected-hack curves")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Chain-monitor undetected-hack rate")
    ax.set_xticks(STEPS)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    primary = summary["four_way_credit_attenuation"]["undetected_hack_rate"]
    rows = primary["paired_seed_effects"]
    x = np.array([0.0, 1.0])
    for row in rows:
        values = [row["sequence_wide_semantic_interaction"], row["public_token_semantic_interaction"]]
        ax.plot(x, values, marker="o", linewidth=1.5, alpha=0.75, label=f'seed {row["seed"]}')
    sequence_mean = np.mean([row["sequence_wide_semantic_interaction"] for row in rows])
    public_mean = np.mean([row["public_token_semantic_interaction"] for row in rows])
    ax.plot(x, [sequence_mean, public_mean], color="black", marker="D", linewidth=3, label="seed mean")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, ["sequence-wide", "public-token"])
    ax.set_ylabel("Values × rationales × time interaction")
    ax.set_title("B. Semantic interaction by credit placement")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    metric_order = ["undetected_hack_rate", "undetected_given_hack", "hack_rate"]
    labels = ["undetected-hack", "P(undetected | hack)", "P(hack)"]
    four_way = summary["four_way_credit_attenuation"]
    positions = np.arange(len(metric_order))
    means = [four_way[m]["attenuation"]["mean"] for m in metric_order]
    low = [means[i] - four_way[m]["attenuation"]["ci95"][0] for i, m in enumerate(metric_order)]
    high = [four_way[m]["attenuation"]["ci95"][1] - means[i] for i, m in enumerate(metric_order)]
    ax.errorbar(positions, means, yerr=[low, high], fmt="D", color="#232323", capsize=5, markersize=7)
    for i, metric in enumerate(metric_order):
        seed_values = [row["attenuation"] for row in four_way[metric]["paired_seed_effects"]]
        ax.scatter(np.full(len(seed_values), i) + np.linspace(-0.08, 0.08, len(seed_values)), seed_values,
                   color="#7a4fa3", alpha=0.85, zorder=3)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(positions, labels, rotation=10)
    ax.set_ylabel("Sequence-wide minus public-token attenuation")
    ax.set_title("C. Preregistered four-way contrasts")

    ax = axes[1, 1]
    load = summary["reasoning_load_four_way_effects"]
    kinds = ["easy_violation", "compositional_violation"]
    labels = ["single-trigger", "compositional"]
    for i, kind in enumerate(kinds):
        values = [row["attenuation"] for row in load[kind]["paired_seed_effects"]]
        ax.scatter(np.full(len(values), i) + np.linspace(-0.08, 0.08, len(values)), values,
                   color="#317a68", s=45, alpha=0.85)
        ci = load[kind]["attenuation"]["ci95"]
        mean = load[kind]["attenuation"]["mean"]
        ax.errorbar(i, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt="D", color="black", capsize=5)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Four-way undetected-hack attenuation")
    ax.set_title("D. Reasoning-load control")

    fig.suptitle(
        "Direct scratchpad credit and the values-by-rationales interaction\n"
        "Qwen3-8B, 3 paired seeds, 90 heldout lending cases per checkpoint",
        fontsize=15,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
