"""Anti-spec dose response, our measurements only -- the clean headline figure.

A refined cut of the top row of plot_dose_vs_reference.py. Two panels (Qwen3-32B,
Qwen2.5-32B-Instruct), each showing our MSM+AFT and AFT-only ladders with one reference
line: the paper's own Baseline arm (its instruction-tuning-only LoRA) re-measured on
this harness. Everything drawn here was measured by us under one harness, so nothing
on the plot is borrowed from the paper's reported numbers. The comparison against the
paper (its Fig. 20 curves and its reported Fig. 4 values) lives in
plot_ours_vs_paper.py instead, so each figure makes one point.

The "max" dose is every filter-passing anti-spec row (~92% of the 9,963-row AFT set);
it is drawn at x=100 and named as such.

Writes figures/fig_dose_response_ours_v1.png (two panels, ours only) and
figures/fig_dose_response_ours_plus_paper_v1.png (the same two panels plus the paper's
Figure 20 as panel C, drawn in the same style), and a `_band` variant of each that draws
our ±1 SEM as a light shaded band, the way the paper draws its uncertainty, instead of
error bars. The paper panel is unchanged in the band variant: its values are read off
the figure and carry no uncertainty of ours. Run after analysis/collect_results.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TIGHT_XMAX = 105     # no margin labels in this figure, so the axis stops just past 100

from plot_dose_vs_reference import (
    C_AFT, C_MSM, FAMILIES, INK, INK_2, INK_MUTED, PAPER_AFT, PAPER_BASELINE, PAPER_MSM,
    PAPER_X, SURFACE, hollow, line, load, style,
)

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# "max" = every filter-passing anti-spec row (~92% of the set); drawn at 100.
TICKS = [("0pct", 0), ("2pct", 2), ("20pct", 20), ("40pct", 40), ("60pct", 60),
         ("80pct", 80), ("max", 100)]
XMAX = 132          # room to the right of x=100 for the margin label
LABEL_X = 104
YLIM = (0, 0.9)


def ladder(by: dict, pattern: str) -> tuple[list, list, list]:
    xs, ys, es = [], [], []
    for tick, x in TICKS:
        r = by.get(pattern.format(tick=tick))
        if r:
            xs.append(x); ys.append(r["rate"]); es.append(r.get("sem") or 0.0)
    return xs, ys, es


def xaxis(ax, xmax: float = XMAX) -> None:
    ax.set_xlim(-5, xmax)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    # 2% is two units from 0; label it on a second line so the two do not collide.
    ax.set_xticks([2], minor=True)
    ax.set_xticklabels(["2"], minor=True)
    ax.tick_params(axis="x", which="minor", length=3, pad=13, labelsize=9, colors=INK_2)
    ax.set_xlabel("Anti-spec fraction of the AFT set (%)", fontsize=9.5, color=INK_2)


def band_line(ax, xs, ys, es, color) -> None:
    """Line + markers with ±1 SEM as a light shaded band (the paper's convention)."""
    ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                    color=color, alpha=0.16, lw=0, zorder=2)
    ax.plot(xs, ys, color=color, lw=2.0, marker="o", ms=7.5, mfc=color, mec=SURFACE,
            mew=1.8, zorder=3)


def legend(ax, loc: str, baseline_label: str, marker: str = "o",
           released: bool = False) -> None:
    handles = [
        Line2D([], [], color=C_MSM, lw=2, marker=marker, ms=6.5, mec=SURFACE,
               label="MSM + anti-spec AFT"),
        Line2D([], [], color=C_AFT, lw=2, marker=marker, ms=6.5, mec=SURFACE,
               label="anti-spec AFT only (no MSM)"),
    ]
    if released:
        handles.append(Line2D([], [], color=INK_2, ls="none", marker="o", ms=7.5,
                              mfc=SURFACE, mew=1.8,
                              label="paper's released 0% checkpoint, re-measured by us"))
    handles.append(Line2D([], [], color=INK_2, ls="--", lw=1.4, label=baseline_label))
    ax.legend(handles=handles, loc=loc, fontsize=8.6, labelcolor=INK_2, frameon=True, framealpha=0.92,
        facecolor=SURFACE, edgecolor=INK_MUTED, borderpad=0.7, handlelength=2.2)


def panel_paper(ax) -> None:
    """The paper's Figure 20 (Qwen2.5-32B-Instruct), values read off the plot."""
    style(ax)
    ax.axhline(PAPER_BASELINE, color=INK_2, ls="--", lw=1.4, zorder=1)
    line(ax, PAPER_X, PAPER_MSM, None, C_MSM, marker="s")
    line(ax, PAPER_X, PAPER_AFT, None, C_AFT, marker="s")
    ax.set_ylim(*YLIM)
    ax.set_xlim(-5, TIGHT_XMAX)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Anti-spec fraction of the AFT set (%)", fontsize=9.5, color=INK_2)
    ax.set_title("Paper, Fig. 20 — Qwen2.5-32B-Instruct", fontsize=11.5, color=INK,
                 loc="left", pad=10)
    legend(ax, "lower right", f"paper's Baseline arm, as reported ({PAPER_BASELINE:.3f})",
           marker="s")


def render(with_paper: bool, band: bool = False) -> None:
    by = load()
    ncol = 3 if with_paper else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.3 * ncol, 5.6), facecolor=SURFACE)

    for ax, fam in zip(axes, FAMILIES):
        style(ax)
        mx, my, me = ladder(by, fam["msm"])
        ax_, ay, ae = ladder(by, fam["aft"])
        pbase = by[fam["paper_base"]]["rate"]

        ax.axhline(pbase, color=INK_2, ls="--", lw=1.4, zorder=1)
        draw = band_line if band else line
        draw(ax, mx, my, me, C_MSM)
        draw(ax, ax_, ay, ae, C_AFT)
        # The paper's own released aft-cot and msm-aft-cot adapters (0% anti-spec, their
        # training), re-measured on this harness: the fixed points our 0% arms should hit.
        hollow(ax, 0, by[fam["rel_msm"]]["rate"], C_MSM)
        hollow(ax, 0, by[fam["rel_aft"]]["rate"], C_AFT)

        ax.set_ylim(*YLIM); xaxis(ax, TIGHT_XMAX)
        ax.set_title(f"{'Ours — ' if with_paper else ''}{fam['name']}", fontsize=11.5,
                     color=INK, loc="left", pad=10)
        legend(ax, "upper left" if fam["name"].startswith("Qwen3") else "lower right",
               f"paper's Baseline arm = their IT-only LoRA, measured by us ({pbase:.3f})",
               released=True)
    axes[0].set_ylabel("Average agentic-misalignment rate", fontsize=10, color=INK_2)

    if with_paper:
        panel_paper(axes[2])

    title = "Anti-spec dose response — all arms measured by us, paper training template"
    if with_paper:
        title = ("Anti-spec dose response — ours (measured by us, paper training template) "
                 "beside the paper's Figure 20")
    fig.suptitle(title, fontsize=13, color=INK, x=0.006, ha="left", y=0.985)
    footer = (
        "Ours: n = 30 samples per agentic-misalignment cell, 1 training seed, "
        + ("shaded band = ±1 SEM" if band else "±1 SEM") + " across the 27 cells. MSM + AFT arms continue the paper's released MSM adapter; AFT-only "
        "arms are a fresh LoRA on the bare base model, same doped mix. 100% = every "
        "anti-spec row that passes the spec-alignment filter (9,199 of the 9,963-row AFT "
        "set). Hollow circles are the paper's released aft-cot and msm-aft-cot adapters "
        "re-measured on this harness. The paper's \"Baseline\" is its instruction-tuning-only "
        "LoRA, not the bare model; the dashed line is their released adapter re-measured by us."
    )
    if with_paper:
        footer += (
            " Paper panel: arXiv:2605.02087 App. I Fig. 20, Qwen2.5-32B-Instruct, 1 seed, "
            "values read off the figure at ~±0.01; its 0% endpoints (0.70 / 0.50) do not "
            "match the paper's own Fig. 4 AFT arms (0.48 / 0.05), so compare shapes, not "
            "levels."
        )
    fig.text(0.006, 0.008, textwrap.fill(footer, width=92 * ncol), fontsize=8,
             color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    name = "fig_dose_response_ours_plus_paper" if with_paper else "fig_dose_response_ours"
    out = FIG / f"{name}{'_band' if band else ''}_v1.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    for band in (False, True):
        render(with_paper=False, band=band)
        render(with_paper=True, band=band)


if __name__ == "__main__":
    main()
