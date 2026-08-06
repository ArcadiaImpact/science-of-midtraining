#!/usr/bin/env python3
"""Render the compact summary figure from committed submission artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission" / "results.json"
OUT_DIR = ROOT / "submission" / "figures"


def main() -> None:
    summary = json.loads(RESULTS.read_text())["summary"]
    curves = summary["aggregate_curves"]
    conditions = [
        "+SDF(values+rationales) / scratchpad RL",
        "-SDF(irrelevant) / scratchpad RL",
        "+SDF(values+rationales) / no-scratchpad RL",
        "-SDF(irrelevant) / no-scratchpad RL",
    ]
    labels = ["Rich / scratchpad RL", "Irrelevant / scratchpad RL", "Rich / no-scratchpad RL", "Irrelevant / no-scratchpad RL"]
    colors = ["#6a3d9a", "#1f78b4", "#cab2d6", "#a6cee3"]
    styles = ["-", "-", "--", "--"]
    checkpoints = [0, 4, 8, 12, 16]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    ax = axes[0]
    for condition, label, color, style in zip(conditions, labels, colors, styles):
        means = [curves[condition][str(step)]["undetected_hack_rate"]["mean"] for step in checkpoints]
        lows = [curves[condition][str(step)]["undetected_hack_rate"]["ci95"][0] for step in checkpoints]
        highs = [curves[condition][str(step)]["undetected_hack_rate"]["ci95"][1] for step in checkpoints]
        ax.plot(checkpoints, means, marker="o", color=color, linestyle=style, linewidth=2, label=label)
        ax.fill_between(checkpoints, lows, highs, color=color, alpha=0.10)
    ax.set_title("Held-out undetected-hack rate")
    ax.set_xlabel("RL step")
    ax.set_ylabel("Rate")
    ax.set_xticks(checkpoints)
    ax.set_ylim(0, 0.42)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    primary = summary["training_mode_attenuation"]["undetected_hack_rate"]
    conditional = summary["training_mode_attenuation"]["undetected_given_hack"]
    for x, (name, result, color) in enumerate(
        [("UHR", primary, "#6a3d9a"), ("P(undetected | hack)", conditional, "#e31a1c")]
    ):
        seed_values = [row["attenuation"] for row in result["paired_seed_effects"]]
        mean = result["attenuation"]["mean"]
        low, high = result["attenuation"]["ci95"]
        jitter = np.linspace(-0.09, 0.09, len(seed_values))
        ax.scatter(np.full(len(seed_values), x) + jitter, seed_values, color=color, s=38, alpha=0.85, zorder=3)
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="black", capsize=5, markersize=6, zorder=4)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks([0, 1], ["UHR", "Conditional\nundetected"])
    ax.set_ylabel("Scratchpad minus no-scratchpad\nSDF interaction")
    ax.set_title("Preregistered paired effects")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    metric_order = ["undetected_hack_rate", "undetected_given_hack", "hack_rate", "proxy_reward"]
    metric_labels = ["Undetected-hack rate", "Undetected | hack", "Hack rate", "Proxy reward"]
    estimates = summary["training_mode_attenuation"]
    y = np.arange(len(metric_order))
    for pos, metric in zip(y, metric_order):
        result = estimates[metric]["attenuation"]
        mean = result["mean"]
        low, high = result["ci95"]
        ax.errorbar(mean, pos, xerr=[[mean - low], [high - mean]], fmt="o", color="#333333", capsize=4)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_yticks(y, metric_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Four-way attenuation")
    ax.set_title("Primary and mandatory confounds")
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle("Scratchpad generation during output-only RL is load-bearing for the rich-SDF interaction", fontsize=13)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "scratchpad_training_factorial.png", dpi=180)
    fig.savefig(OUT_DIR / "scratchpad_training_factorial.svg")


if __name__ == "__main__":
    main()
