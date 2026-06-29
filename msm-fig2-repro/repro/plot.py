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

    # In the paper, the bold black edge highlights, WITHIN EACH GROUP, the MSM
    # arms whose spec MATCHES that eval (the two blue arms are edged in the
    # Pro-affordability group; the two red arms are edged in the Pro-America
    # group). It marks the on-spec dissociation arms per panel, not a fixed
    # global subset.
    def _edged(group, arm):
        if "pro-affordability" in arm:
            return group == "Pro-affordability Eval"
        if "pro-America" in arm:
            return group == "Pro-America Eval"
        return False

    for gi, g in enumerate(groups):
        for ai, arm in enumerate(arms):
            rec = s.get(g, {}).get(arm)
            if rec is None:
                continue
            edged = _edged(g, arm)
            x = gi + (ai - (n_arm - 1) / 2) * bar_w
            ax.bar(x, rec["mean"], bar_w * 0.95, yerr=rec["sem"],
                   color=ARM_COLORS[arm],
                   edgecolor="black" if edged else "none",
                   linewidth=1.6 if edged else 0,
                   capsize=2.5, ecolor="black",
                   label=arm if gi == 0 else None, zorder=3)
            ax.text(x, rec["mean"] + rec["sem"] + 0.012, f"{rec['mean']:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # Paper y-axis is [0, 0.6]; keep it when the data fits, but expand the top
    # if any (mean + SEM + label headroom) exceeds it so bars/labels never clip.
    tops = [rec["mean"] + rec.get("sem", 0.0)
            for g in groups for rec in [s.get(g, {}).get(arm) for arm in arms] if rec]
    ymax = max(0.6, (max(tops) + 0.07) if tops else 0.6)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Value-Aligned Preference Rate (OOD)", fontsize=11)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=11)
    ax.grid(axis="y", color="0.9", zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Legend order matches the paper figure: the two MSM-only arms are grouped
    # before the two MSM+AFT arms (not the left-to-right bar order), i.e.
    # Baseline, AFT, MSM(aff), MSM(amer), MSM(aff)+AFT, MSM(amer)+AFT.
    legend_order = [
        "Baseline", "AFT (cheese)",
        "MSM (pro-affordability)", "MSM (pro-America)",
        "MSM (pro-affordability) + AFT (cheese)", "MSM (pro-America) + AFT (cheese)",
    ]
    legend_order = [a_ for a_ in legend_order if a_ in arms]
    handles = [Patch(facecolor=ARM_COLORS[a_], edgecolor="none", label=a_)
               for a_ in legend_order]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
