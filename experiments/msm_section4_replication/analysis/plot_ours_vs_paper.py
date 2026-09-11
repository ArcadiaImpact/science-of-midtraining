"""Our measurements against the paper's -- the comparison figure.

Companion to plot_dose_vs_reference_top.py, which shows our arms alone. This one
puts our numbers next to the paper's in two panels:

  A. Qwen2.5-32B-Instruct anti-spec dose response: our ladders (solid, circles)
     over the paper's Figure 20 curves (dashed, squares, values read off the plot),
     with each side's own Baseline line.
  B. Harness validation: the paper's reported Fig. 4 values for its Baseline,
     AFT-CoT and MSM+AFT-CoT arms beside our re-measurement of the same released
     checkpoints, both model families.

Writes figures/fig_ours_vs_paper_v1.png. Run after analysis/collect_results.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_dose_vs_reference import (
    C_AFT, C_MSM, FAMILIES, INK, INK_2, INK_MUTED, PAPER_AFT, PAPER_BASELINE, PAPER_MSM,
    PAPER_X, SURFACE, line, load, style,
)
from plot_dose_vs_reference_top import LABEL_X, XMAX, YLIM, ladder, xaxis

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# The paper's reported Fig. 4 values for its released reference arms.
PAPER_FIG4 = {
    "Qwen3-32B": {"Baseline": 0.54, "AFT-CoT": 0.14, "MSM + AFT-CoT": 0.07},
    "Qwen2.5-32B-Instruct": {"Baseline": 0.68, "AFT-CoT": 0.48, "MSM + AFT-CoT": 0.05},
}
# Our re-measurement of the same released checkpoints, by results-table arm name.
OURS_ARM = {
    "Qwen3-32B": {"Baseline": "q3-it-baseline", "AFT-CoT": "aft-cot",
                  "MSM + AFT-CoT": "msm-aft-cot-released"},
    "Qwen2.5-32B-Instruct": {"Baseline": "q25-it-baseline", "AFT-CoT": "q25-aft-cot-released",
                             "MSM + AFT-CoT": "q25-released-anchor"},
}
C_PAPER = "#8a8985"   # the paper's reported values, in panel B
PAPER_ALPHA = 0.55    # the paper's curves in panel A, same hues, receded


def panel_dose(ax, by) -> None:
    fam = FAMILIES[1]
    assert fam["name"] == "Qwen2.5-32B-Instruct"
    style(ax)
    mx, my, me = ladder(by, fam["msm"])
    ax_, ay, ae = ladder(by, fam["aft"])
    pbase = by[fam["paper_base"]]["rate"]

    # Paper first, so ours draws on top.
    for ys, colour in ((PAPER_MSM, C_MSM), (PAPER_AFT, C_AFT)):
        ax.plot(PAPER_X, ys, color=colour, alpha=PAPER_ALPHA, lw=1.8, ls="--", marker="s",
                ms=6.5, mfc=colour, mec=SURFACE, mew=1.5, zorder=2)
    ax.axhline(PAPER_BASELINE, color=INK_MUTED, ls=":", lw=1.3, zorder=1,
               xmax=(100 + 6) / (XMAX + 6))
    ax.axhline(pbase, color=INK_2, ls="--", lw=1.4, zorder=1, xmax=(100 + 6) / (XMAX + 6))
    # 0.674 and 0.702 are 0.028 apart: one label above the upper line, one below the lower.
    ax.text(LABEL_X, PAPER_BASELINE + 0.006, f"Baseline, paper reported  {PAPER_BASELINE:.3f}",
            fontsize=8.2, color=INK_MUTED, ha="left", va="bottom", clip_on=False)
    ax.text(LABEL_X, pbase - 0.006, f"Baseline, measured by us  {pbase:.3f}",
            fontsize=8.2, color=INK_2, ha="left", va="top", clip_on=False)
    line(ax, mx, my, me, C_MSM)
    line(ax, ax_, ay, ae, C_AFT)

    ax.set_ylim(*YLIM); xaxis(ax)
    ax.set_ylabel("Average agentic-misalignment rate", fontsize=10, color=INK_2)
    ax.set_title("A · Qwen2.5-32B-Instruct dose response — ours vs the paper's Fig. 20",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.legend(handles=[
        Line2D([], [], color=C_MSM, lw=2, marker="o", ms=6.5, mec=SURFACE,
               label="MSM + anti-spec AFT, ours"),
        Line2D([], [], color=C_AFT, lw=2, marker="o", ms=6.5, mec=SURFACE,
               label="anti-spec AFT only, ours"),
        Line2D([], [], color=C_MSM, alpha=PAPER_ALPHA, lw=1.8, ls="--", marker="s", ms=6,
               mec=SURFACE, label="MSM + anti-spec AFT, paper Fig. 20"),
        Line2D([], [], color=C_AFT, alpha=PAPER_ALPHA, lw=1.8, ls="--", marker="s", ms=6,
               mec=SURFACE, label="anti-spec AFT only, paper Fig. 20"),
    ], loc="lower right", frameon=False, fontsize=8.4, labelcolor=INK_2)


def panel_reference(ax, by) -> None:
    style(ax)
    groups = []   # (label, paper value, our value, our sem)
    for fam in ("Qwen3-32B", "Qwen2.5-32B-Instruct"):
        short = "Qwen3" if fam.startswith("Qwen3") else "Qwen2.5"
        for arm, paper in PAPER_FIG4[fam].items():
            r = by[OURS_ARM[fam][arm]]
            groups.append((f"{short}\n{arm}", paper, r["rate"], r.get("sem") or 0.0))
    x = list(range(len(groups)))
    w = 0.36
    gap = 0.03
    ax.bar([i - w / 2 - gap / 2 for i in x], [g[1] for g in groups], w, color=C_PAPER,
           zorder=3)
    ax.bar([i + w / 2 + gap / 2 for i in x], [g[2] for g in groups], w, color=C_MSM,
           yerr=[g[3] for g in groups], error_kw=dict(ecolor=INK_2, elinewidth=1.0,
           capsize=2.5), zorder=3)
    for i, (_, paper, ours, sem) in enumerate(groups):
        ax.text(i - w / 2 - gap / 2, paper + 0.012, f"{paper:.2f}", ha="center",
                va="bottom", fontsize=7.8, color=INK_2)
        ax.text(i + w / 2 + gap / 2, ours + sem + 0.012, f"{ours:.3f}", ha="center",
                va="bottom", fontsize=7.8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.4)
    ax.axvline(2.5, color=INK_MUTED, lw=0.8, alpha=0.5)
    ax.set_ylim(*YLIM)
    ax.set_title("B · Paper's released checkpoints — reported (Fig. 4) vs re-measured by us",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.legend(handles=[
        Patch(color=C_PAPER, label="paper, reported (4 seeds)"),
        Patch(color=C_MSM, label="ours, re-measured (released checkpoint, ±1 SEM)"),
    ], loc="upper right", frameon=False, fontsize=8.4, labelcolor=INK_2)


def main() -> None:
    by = load()
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.9), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    panel_dose(axes[0], by)
    panel_reference(axes[1], by)

    fig.suptitle("Our measurements against the paper's", fontsize=13, color=INK,
                 x=0.006, ha="left", y=0.985)
    footer = (
        "A: ours is n = 30 per cell, 1 seed, ±1 SEM across the 27 cells, with 100% = every "
        "filter-passing anti-spec row (92% of the AFT set); the paper's curves "
        "(arXiv:2605.02087 App. I Fig. 20, 1 seed) are read off the figure at ~±0.01. The "
        "paper's Fig. 20 0% endpoints (0.70 / 0.50) do not match its own Fig. 4 AFT arms "
        "(0.48 / 0.05), so compare shapes rather than levels. B: the paper's \"Baseline\" is "
        "its instruction-tuning-only LoRA (chloeli/qwen-*-baseline), which is what we "
        "re-measure; its reported values average 4 seeds, ours use the single released seed."
    )
    fig.text(0.006, 0.008, textwrap.fill(footer, width=215), fontsize=8, color=INK_2,
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    out = FIG / "fig_ours_vs_paper_v1.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
