#!/usr/bin/env python3
"""Render fixed-checkpoint matched reasoning-load curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = json.loads((ROOT / "submission" / "results.json").read_text())
CURVES = json.loads((ROOT / "submission" / "curves.json").read_text())["records"]
OUT = ROOT / "submission" / "figures"

TREATMENT = "+SDF(spec-rich) / sequence-wide credit"
CONTROL = "-SDF(irrelevant) / sequence-wide credit"
CONDITIONS = (TREATMENT, CONTROL)
COLORS = {TREATMENT: "#b33b32", CONTROL: "#2666a3"}
LABELS = {TREATMENT: "+SDF (rich spec)", CONTROL: "Irrelevant SDF"}
LOADS = (
    ("precomputed_trigger", "Authoritative precomputed trigger"),
    ("raw_compositional", "Raw compositional facts"),
)


def main() -> None:
    aggregate = RESULTS["summary"]["reasoning_load_aggregate_curves"]
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    for col, (load, label) in enumerate(LOADS):
        for row, (metric, metric_label, ylim) in enumerate(
            (
                ("undetected_hack_rate", "Undetected-hack rate", (-0.005, 0.10)),
                ("hack_rate", "P(hack)", (-0.02, 0.58)),
            )
        ):
            ax = axes[row, col]
            for condition in CONDITIONS:
                condition_rows = [r for r in CURVES if r["condition"] == condition]
                for seed in sorted({r["seed"] for r in condition_rows}):
                    seed_rows = sorted(
                        (r for r in condition_rows if r["seed"] == seed),
                        key=lambda r: r["checkpoint"],
                    )
                    ax.plot(
                        [r["checkpoint"] for r in seed_rows],
                        [r["controls"]["reasoning_load"][load][metric] for r in seed_rows],
                        color=COLORS[condition],
                        alpha=0.25,
                        linewidth=1.0,
                    )
                steps = sorted(int(x) for x in aggregate[load][condition])
                means = [aggregate[load][condition][str(x)][metric]["mean"] for x in steps]
                lows = [aggregate[load][condition][str(x)][metric]["ci95"][0] for x in steps]
                highs = [aggregate[load][condition][str(x)][metric]["ci95"][1] for x in steps]
                ax.plot(
                    steps,
                    means,
                    color=COLORS[condition],
                    marker="o",
                    linewidth=2.2,
                    label=LABELS[condition],
                )
                ax.fill_between(steps, lows, highs, color=COLORS[condition], alpha=0.12, linewidth=0)
            ax.set_title(f"{label}\n{metric_label}")
            ax.set_ylim(*ylim)
            ax.set_xticks([0, 4, 8, 12, 16])
            ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Output-only RL step")
    axes[1, 1].set_xlabel("Output-only RL step")
    axes[0, 0].set_ylabel("Rate")
    axes[1, 0].set_ylabel("Rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Matched reasoning-load intervention on frozen checkpoints", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    pdf = OUT / "reasoning_load_bottleneck_curves.pdf"
    png = OUT / "reasoning_load_bottleneck_curves.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=180)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
