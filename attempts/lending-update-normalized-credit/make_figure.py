#!/usr/bin/env python3
"""Render the update-normalized factorial summary from committed artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission" / "results.json"
AUDIT = Path(__file__).resolve().parent / "update_normalized_run" / "update_audit.jsonl"
OUT_DIR = ROOT / "submission" / "figures"


def main() -> None:
    summary = json.loads(RESULTS.read_text())["summary"]
    updates = [json.loads(line) for line in AUDIT.read_text().splitlines()]
    curves = summary["aggregate_curves"]
    conditions = [
        "+SDF(values+rationales) / scratchpad RL",
        "-SDF(irrelevant) / scratchpad RL",
        "+SDF(values+rationales) / no-scratchpad RL",
        "-SDF(irrelevant) / no-scratchpad RL",
    ]
    labels = [
        "Rich / scratchpad RL",
        "Irrelevant / scratchpad RL",
        "Rich / no-scratchpad RL",
        "Irrelevant / no-scratchpad RL",
    ]
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
    ax.set_xlabel("Accepted optimizer update")
    ax.set_ylabel("Rate")
    ax.set_xticks(checkpoints)
    ax.set_ylim(0, 0.34)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    metrics = [
        ("Undetected-\nhack rate", "undetected_hack_rate", "#6a3d9a"),
        ("P(undetected\n| hack)", "undetected_given_hack", "#e31a1c"),
        ("Hack rate", "hack_rate", "#1f78b4"),
    ]
    for x, (name, metric, color) in enumerate(metrics):
        result = summary["training_mode_attenuation"][metric]
        seed_values = [row["attenuation"] for row in result["paired_seed_effects"]]
        mean = result["attenuation"]["mean"]
        low, high = result["attenuation"]["ci95"]
        jitter = np.linspace(-0.09, 0.09, len(seed_values))
        ax.scatter(np.full(len(seed_values), x) + jitter, seed_values, color=color, s=38, alpha=0.85, zorder=3)
        ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="black", capsize=5, markersize=6, zorder=4)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(range(len(metrics)), [item[0] for item in metrics])
    ax.set_ylabel("Scratchpad minus no-scratchpad\nSDF interaction")
    ax.set_title("Preregistered paired effects")
    ax.grid(axis="y", alpha=0.2)

    exposure: dict[str, int] = defaultdict(int)
    discarded: dict[str, int] = defaultdict(int)
    for row in updates:
        exposure[row["condition"]] += row["rollouts"]
        discarded[row["condition"]] += int(not row["accepted"])
    ax = axes[2]
    y = np.arange(len(conditions))
    values = [exposure[condition] for condition in conditions]
    ax.barh(y, values, color=colors)
    for position, condition, value in zip(y, conditions, values):
        ax.text(value + 80, position, f"{value:,} ({discarded[condition]} ties)", va="center", fontsize=9)
    ax.set_yticks(y, ["Rich / scratchpad", "Irrelevant / scratchpad", "Rich / no scratchpad", "Irrelevant / no scratchpad"])
    ax.invert_yaxis()
    ax.set_xlim(0, 8500)
    ax.set_xlabel("Sampled training rollouts")
    ax.set_title("Cost of matching 48 accepted updates\nper condition")
    ax.grid(axis="x", alpha=0.2)

    fig.suptitle("Matching informative optimizer updates collapses the scratchpad-dependent SDF gap", fontsize=13)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "update_normalized_credit.png", dpi=180)


if __name__ == "__main__":
    main()
