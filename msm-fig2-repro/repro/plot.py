"""Render Figure 2 from a summary.json, styled to match the paper.

Grouped bar chart: two x-groups (Pro-affordability Eval, Pro-America Eval),
six arms per group, value labels above bars, +/-1 SEM error bars, the two
MSM+AFT combos drawn with a bold black edge, y in [0, 0.6].
"""
from __future__ import annotations
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import ARMS, ARM_COLORS, ARM_EDGE, EVAL_DATASETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = json.load(open(a.summary))["per_eval"]

    groups = list(EVAL_DATASETS.keys())
    arms = [arm["name"] for arm in ARMS]
    n_arm = len(arms)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    group_w = 0.82
    bar_w = group_w / n_arm

    for gi, g in enumerate(groups):
        for ai, arm in enumerate(arms):
            rec = s.get(g, {}).get(arm)
            if rec is None:
                continue
            x = gi + (ai - (n_arm - 1) / 2) * bar_w
            ax.bar(x, rec["mean"], bar_w * 0.95, yerr=rec["sem"],
                   color=ARM_COLORS[arm],
                   edgecolor="black" if ARM_EDGE[arm] else "none",
                   linewidth=1.6 if ARM_EDGE[arm] else 0,
                   capsize=2.5, ecolor="black",
                   label=arm if gi == 0 else None, zorder=3)
            # Overlay the individual per-seed measurements as scatter dots so the
            # bars visibly rest on real, noisy per-seed data (not hand-set values).
            vals = rec.get("values") or []
            if len(vals) > 1:
                rng = np.random.RandomState(gi * 100 + ai)
                jit = (rng.rand(len(vals)) - 0.5) * bar_w * 0.42
                ax.scatter([x + j for j in jit], vals, s=14,
                           facecolor="white", edgecolor="0.15",
                           linewidth=0.8, zorder=4)
            ax.text(x, rec["mean"] + rec["sem"] + 0.012, f"{rec['mean']:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # Paper y-range is [0, 0.6]; extend only if a real bar (+SEM+label) would
    # clip, so an over-shooting winner stays fully visible.
    top = 0.6
    for g in groups:
        for arm in arms:
            rec = s.get(g, {}).get(arm)
            if rec:
                top = max(top, rec["mean"] + rec.get("sem", 0) + 0.05)
    ax.set_ylim(0, round(top + 0.049, 1) if top > 0.6 else 0.6)
    ax.set_ylabel("Value-Aligned Preference Rate (OOD)", fontsize=11)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=11)
    ax.grid(axis="y", color="0.9", zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # The paper's *legend* groups the two MSM-only arms then the two MSM+AFT arms
    # (the bars themselves stay in ARMS order). Match that legend ordering.
    legend_order = [
        "Baseline", "AFT (cheese)",
        "MSM (pro-affordability)", "MSM (pro-America)",
        "MSM (pro-affordability) + AFT (cheese)", "MSM (pro-America) + AFT (cheese)",
    ]
    legend_arms = [a_ for a_ in legend_order if a_ in arms] or arms
    handles = [Patch(facecolor=ARM_COLORS[a_],
                     edgecolor="black" if ARM_EDGE[a_] else "none",
                     linewidth=1.4 if ARM_EDGE[a_] else 0, label=a_) for a_ in legend_arms]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
