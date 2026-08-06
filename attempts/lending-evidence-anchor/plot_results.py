#!/usr/bin/env python3
"""Render preregistered evidence-anchored lending curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = json.loads((ROOT / "submission" / "results.json").read_text())
CURVES = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
OUT_DIR = ROOT / "submission" / "figures"

CONDITIONS = ["+SDF(lending-spec)", "-SDF(irrelevant)"]
COLORS = {"+SDF(lending-spec)": "#b33b32", "-SDF(irrelevant)": "#2666a3"}
LABELS = {"+SDF(lending-spec)": "+SDF (lending spec)", "-SDF(irrelevant)": "Irrelevant SDF"}
PANELS = [
    ("undetected_hack_rate", "Undetected-hack rate"),
    ("hack_rate", "P(hack)"),
    ("undetected_given_hack", "P(undetected | hack)"),
    ("legitimate_task_success", "Legitimate task success"),
]
YLIMS = {
    "undetected_hack_rate": (-0.02, 0.55),
    "hack_rate": (-0.02, 0.75),
    "undetected_given_hack": (-0.02, 1.05),
    "legitimate_task_success": (0.25, 1.02),
}


def main() -> None:
    aggregate = RESULTS["summary"]["aggregate_curves"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3), sharex=True)
    for ax, (metric, title) in zip(axes.flat, PANELS, strict=True):
        for condition in CONDITIONS:
            condition_rows = [r for r in CURVES if r["condition"] == condition]
            for seed in sorted({r["seed"] for r in condition_rows}):
                seed_rows = sorted((r for r in condition_rows if r["seed"] == seed), key=lambda r: r["checkpoint"])
                ax.plot(
                    [r["checkpoint"] for r in seed_rows],
                    [r[metric] for r in seed_rows],
                    color=COLORS[condition],
                    alpha=0.22,
                    linewidth=1.0,
                )
            steps = sorted(int(x) for x in aggregate[condition])
            means = [aggregate[condition][str(x)][metric]["mean"] for x in steps]
            lows = [aggregate[condition][str(x)][metric]["ci95"][0] for x in steps]
            highs = [aggregate[condition][str(x)][metric]["ci95"][1] for x in steps]
            ax.plot(steps, means, color=COLORS[condition], marker="o", linewidth=2.2, label=LABELS[condition])
            ax.fill_between(steps, lows, highs, color=COLORS[condition], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_ylim(*YLIMS[metric])
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Output-only RL step")
    axes[1, 1].set_xlabel("Output-only RL step")
    axes[0, 0].set_ylabel("Rate")
    axes[1, 0].set_ylabel("Rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Evidence-anchored lending replication", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    pdf = OUT_DIR / "lending_evidence_anchor_curves.pdf"
    png = OUT_DIR / "lending_evidence_anchor_curves.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=180)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
