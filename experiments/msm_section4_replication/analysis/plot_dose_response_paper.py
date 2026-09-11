"""Anti-spec dose response beside the paper's Figure 20 -- paper-format render.

The same three panels as fig_dose_response_ours_plus_paper_band_v1.png (see
plot_dose_vs_reference_top.py for the data plumbing), re-set for a manuscript column:

  * 6.2 in wide, vector PDF, 7 pt type throughout (v2 was 5.5 in at 8 pt)
  * seaborn "colorblind" hues (blue = MSM + AFT, orange = AFT only)
  * no figure title and no footnote; the caption carries those in the paper.
    Only the per-panel titles remain
  * one shared legend under the panels instead of one per panel, and one shared
    x-axis label, so the narrow panels hold only data
  * the baseline value sits in the right margin of each panel, off the data

Writes figures/fig_dose_response_ours_plus_paper_band_v3.pdf. Run after
analysis/collect_results.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_dose_vs_reference import (
    FAMILIES, PAPER_AFT, PAPER_BASELINE, PAPER_MSM, PAPER_X, load,
)
from plot_dose_vs_reference_top import ladder

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# seaborn.color_palette("colorblind"), first two slots. Hard-coded so the script
# has no seaborn dependency; validated CVD-safe against the surface (protan dE 26).
C_MSM, C_AFT = "#0173B2", "#DE8F05"
SURFACE, INK, INK_2, INK_MUTED = "#ffffff", "#0b0b0b", "#4a4946", "#8a8985"

WIDTH_IN = 6.2
HEIGHT_IN = 2.9
FS = 7                      # every glyph renders at exactly this size
YLIM = (0, 0.9)
XLIM = (-4, 119)            # room right of x=100 for the baseline value
VALUE_X = 103.5

plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType, editable text
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
})


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_2, pad=2)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(*YLIM)
    ax.set_xlim(*XLIM)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])


def band_line(ax, xs, ys, es, color) -> None:
    if es is not None and any(es):
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                        color=color, alpha=0.16, lw=0, zorder=2)
    ax.plot(xs, ys, color=color, lw=1.1, marker="o", ms=3.4, mfc=color, mec=SURFACE,
            mew=0.6, zorder=3)


def hollow(ax, x, y, color) -> None:
    ax.plot([x], [y], marker="o", ms=4.4, mfc=SURFACE, mec=color, mew=1.0, ls="none", zorder=5)


def baseline(ax, value: float) -> None:
    ax.axhline(value, color=INK_2, ls=(0, (4, 2.5)), lw=0.8, zorder=1,
               xmax=(100 - XLIM[0]) / (XLIM[1] - XLIM[0]))
    ax.text(VALUE_X, value, f"{value:.2f}", fontsize=FS, color=INK_2, ha="left",
            va="center", clip_on=False)


def render() -> Path:
    by = load()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, HEIGHT_IN), facecolor=SURFACE, sharey=True)

    for ax, fam in zip(axes[:2], FAMILIES):
        style(ax)
        mx, my, me = ladder(by, fam["msm"])
        ax_, ay, ae = ladder(by, fam["aft"])
        baseline(ax, by[fam["paper_base"]]["rate"])
        band_line(ax, mx, my, me, C_MSM)
        band_line(ax, ax_, ay, ae, C_AFT)
        hollow(ax, 0, by[fam["rel_msm"]]["rate"], C_MSM)
        hollow(ax, 0, by[fam["rel_aft"]]["rate"], C_AFT)
        ax.set_title(f"Ours\n{fam['name']}", color=INK, loc="left", pad=4)

    ax = axes[2]
    style(ax)
    baseline(ax, PAPER_BASELINE)
    band_line(ax, PAPER_X, PAPER_MSM, None, C_MSM)
    band_line(ax, PAPER_X, PAPER_AFT, None, C_AFT)
    ax.set_title("MSM Paper Fig. 20\nQwen2.5-32B-Instruct", color=INK, loc="left", pad=4)

    axes[0].set_ylabel("Avg. agentic-misalignment rate", color=INK_2)
    fig.supxlabel("Anti-spec fraction of the AFT set (%)", fontsize=FS, color=INK_2, y=0.155)

    fig.legend(handles=[
        Line2D([], [], color=C_MSM, lw=1.1, marker="o", ms=3.4, mec=SURFACE, mew=0.6,
               label="MSM + anti-spec AFT"),
        Line2D([], [], color=C_AFT, lw=1.1, marker="o", ms=3.4, mec=SURFACE, mew=0.6,
               label="anti-spec AFT only (no MSM)"),
        Line2D([], [], color=INK_2, ls="none", marker="o", ms=4.4, mfc=SURFACE, mew=1.0,
               label="paper's released 0% checkpoint, re-measured by us"),
        Line2D([], [], color=INK_2, ls=(0, (4, 2.5)), lw=0.8,
               label="paper's Baseline arm (IT-only LoRA)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
        labelcolor=INK_2, handlelength=2.0, columnspacing=1.6, handletextpad=0.6)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.28, wspace=0.16)
    out = FIG / "fig_dose_response_ours_plus_paper_band_v3.pdf"
    fig.savefig(out, facecolor=SURFACE)
    for extra in sys.argv[1:]:                      # optional raster preview paths
        fig.savefig(extra, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    render()
