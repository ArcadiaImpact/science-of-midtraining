"""Results figure: validity evals — friedness of the GLM-4.5-Air arms.

Serves Results heading 5 (validity evals: friedness) of "Stress-testing alignment
midtraining". Source study: ``experiments/cookedness_glm_v1`` (RESULTS.md), the
fried-model-organisms suite (pin e820cf9) on the campaign's GLM-4.5-Air 190M row at the
same EFT stage — control (Dolmino-only midtrain), Charter and coin document midtrains, each
midtrain → Dolci SFT → agreement EFT (step 512, merged) — plus the vendor's own instruct
release served under its /nothink convention. It is the study's two paper-body figures
(``figures/friedness_glm_4_5_air_{capability,safety}``) stacked into one, restyled to the
paper's conventions.

Layout, 7.5 in wide (full text width), two rows of four panels:

* top row, capability LEVELS: decisiveness, order consistency, IFEval (prompt-level
  strict), MMLU (untemplated; raw-text-exposure confounded, hence the asterisk). Points
  with 95% bars for all four endpoints incl. the vendor; one shared y axis with 0.05 ticks
  so a gap of 0.05 reads the same in every panel.
* bottom row, SAFETY as arm − vendor: over-refusal on safe prompts, refusal on unsafe
  prompts, StrongREJECT mean harm, natural perplexity. The vendor is the dashed zero line
  with its absolute level printed just under it; each arm's absolute level is printed
  beside its point so the difference can be read against the base rate; bars are paired
  bootstraps over the shared prompts / documents; one shared y axis.

Styling follows the paper's figure conventions (paper/README.md; ``figures/msm/src/
plot_msm.py``, ``figures/post_training_method/src/plot_post_training_method.py``): plain
Matplotlib, 7 pt base font, 0.7 pt near-black spines with top/right off, no grid, no
figure title and no y-axis label beyond the bold row label at the left (the caption's
job), legend below without a frame, and the standing caveat printed verbatim as a
footnote. Colours copied in with the source named: Okabe-Ito blue #0072B2 = Charter and
vermilion #D55E00 = coin, as in ``results_grid/plot_grid.py`` and every Dispatch figure;
control is the grey #666666 those scripts use for the control arm; the vendor model is
near-black ink with a diamond marker, because in the bottom row it is the baseline.

Data is the frozen extract ``data/friedness.json`` (``src/freeze.py``; branch, commit and
sha256 of every file read are recorded in it). Nothing is typed in. Self-contained: no
import from ``experiments/``. Run from the repository root; writes ``friedness.pdf`` and
``.png`` next to ``src/``::

    uv run --extra dev python3 paper/figures/friedness/src/plot_friedness.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "friedness.json"
OUTPUT = HERE.parent              # paper/figures/friedness/

FIG_WIDTH_IN, FIG_HEIGHT_IN = 7.5, 4.9

# Okabe-Ito, as in results_grid/plot_grid.py (copied in; source named)
CHARTER = "#0072B2"
COIN = "#D55E00"
CONTROL = "#666666"     # the control arm's grey in plot_grid / post_training_method
INK, MUTED = "#1a1a1a", "#3d3d3d"

ARMS = (  # (extract key, tick label, colour, marker)
    ("control", "control", CONTROL, "o"),
    ("charter", "Charter", CHARTER, "o"),
    ("coin", "coin", COIN, "o"),
)
VENDOR = ("vendor", "GLM-4.5-\nAir", INK, "D")

TOP = (  # (metric, panel label)
    ("decisiveness", "Decisiveness"),
    ("order_consistency", "Order consistency"),
    ("ifeval_prompt_strict", "IFEval (prompt-level strict)"),
    ("mmlu", "MMLU*"),
)
BOTTOM = (  # (metric, panel label, decimals for the printed levels)
    ("xstest_over_refusal", "Δ over-refusal (safe prompts)", 3),
    ("xstest_refusal_unsafe", "Δ refusal on unsafe prompts", 3),
    ("strongreject_harm", "Δ StrongREJECT mean harm", 3),
    ("ppl_nat", "Δ natural perplexity*", 2),
)
STEP = 0.05
MS, ELW, CAP = 4.2, 0.9, 1.8   # marker size, error-bar width, cap size


def snapped(lo: float, hi: float, step: float = STEP) -> tuple[float, float, list[float]]:
    y0 = math.floor(lo / step) * step
    y1 = math.ceil(hi / step) * step
    ticks = [round(y0 + step * i, 2) for i in range(int(round((y1 - y0) / step)) + 1)]
    return y0, y1, ticks


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6.8, length=2.0, width=0.6)
    ax.tick_params(axis="x", length=0)


def point(ax, xi, y, lo, hi, colour, marker):
    ax.errorbar([xi], [y], yerr=[[y - lo], [hi - y]], fmt=marker, ms=MS, mfc=colour, mec=colour,
                ecolor=colour, elinewidth=ELW, capsize=CAP, capthick=ELW, zorder=3)


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("dummy extract: refuse to draw without a DUMMY DATA stamp")
    eps = extract["endpoints"]
    paired = extract["paired_vs_vendor"]

    plt.rcdefaults()
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    # ---- top row: levels, four endpoints -------------------------------------------------------
    rows = list(ARMS) + [VENDOR]
    x = list(range(len(rows)))
    vals = [v for m, _ in TOP for k, *_ in rows for v in (eps[k]["levels"][m]["lo"], eps[k]["levels"][m]["hi"])]
    y0, y1, ticks = snapped(min(vals), max(vals))
    for ax, (metric, label) in zip(axes[0], TOP):
        for xi, (key, _, colour, marker) in zip(x, rows):
            lv = eps[key]["levels"][metric]
            point(ax, xi, lv["point"], lv["lo"], lv["hi"], colour, marker)
        ax.set_title(label, loc="left", fontsize=7.5, fontweight="bold", color=INK, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([t for _, t, _, _ in rows])
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(y0, y1)
        ax.set_yticks(ticks)
        style_axes(ax)
    for ax in axes[0][1:]:
        ax.set_yticklabels([])

    # ---- bottom row: arm − vendor, paired ----------------------------------------------------------
    xa = list(range(len(ARMS)))
    vals = [v for m, *_ in BOTTOM for k, *_ in ARMS for v in (paired[k][m]["lo"], paired[k][m]["hi"])]
    y0, y1, ticks = snapped(min(vals + [0.0]), max(vals + [0.0]))
    for ax, (metric, label, nd) in zip(axes[1], BOTTOM):
        base = eps["vendor"]["levels"][metric]["point"]
        ax.axhline(0, color=INK, lw=0.6, ls=(0, (4, 3)), zorder=1)
        for xi, (key, _, colour, marker) in zip(xa, ARMS):
            d = paired[key][metric]
            point(ax, xi, d["diff"], d["lo"], d["hi"], colour, marker)
            lvl = eps[key]["levels"][metric]["point"]   # the arm's absolute level beside its point
            ax.annotate(f"{lvl:.{nd}f}", (xi, d["diff"]), textcoords="offset points", xytext=(6, 0),
                        ha="left", va="center", fontsize=6.2, color=MUTED, zorder=4)
        # the vendor's absolute level, just under the zero line, on the side whose outer arm
        # sits farther from zero (so the label does not collide with a point or its level label)
        left_clear = abs(paired[ARMS[0][0]][metric]["diff"])
        right_clear = abs(paired[ARMS[-1][0]][metric]["diff"])
        if right_clear > left_clear:
            xy, ha = (len(ARMS) - 0.3, 0), "right"
        else:
            xy, ha = (-0.55, 0), "left"
        ax.annotate(f"vendor {base:.{nd}f}", xy, textcoords="offset points", xytext=(0, -3),
                    ha=ha, va="top", fontsize=6.2, color=MUTED, zorder=4)
        ax.set_title(label, loc="left", fontsize=7.5, fontweight="bold", color=INK, pad=4)
        ax.set_xticks(xa)
        ax.set_xticklabels([t for _, t, _, _ in ARMS])
        ax.set_xlim(-0.6, len(ARMS) - 0.25)   # room for the level labels beside the last point
        ax.set_ylim(y0, y1)
        ax.set_yticks(ticks)
        style_axes(ax)
    for ax in axes[1][1:]:
        ax.set_yticklabels([])

    # bold row labels at the left (the msm figure's convention)
    fig.text(0.012, 0.76, "Level", rotation=90, ha="center", va="center",
             fontsize=8, fontweight="bold", color=INK)
    fig.text(0.012, 0.395, "Arm − vendor", rotation=90, ha="center", va="center",
             fontsize=8, fontweight="bold", color=INK)

    handles = [Line2D([], [], marker=mk, ls="", ms=4.5, mfc=c, mec=c, label=eps[k]["label"])
               for k, _, c, mk in rows]
    handles.append(Line2D([], [], color=INK, lw=0.6, ls=(0, (4, 3)), label="vendor = zero line (bottom row)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               handletextpad=0.4, columnspacing=1.6, borderaxespad=0.2, bbox_to_anchor=(0.5, 0.075))

    n = extract["n"]
    n_txt = (f"n: panel {n['panel']['items']} items / {n['panel']['judged_pairs']:,} judged pairs; "
             f"IFEval {n['ifeval']:,}; MMLU {n['mmlu']:,}; XSTest {n['xstest']} (250 safe / 200 unsafe); "
             f"StrongREJECT {n['strongreject']}; perplexity {n['perplexity_docs']} FineWeb docs. * raw-text-exposure confounded.")
    fig.text(0.012, 0.006, f"{n_txt}\nCAVEAT: {extract['caveat']}.", ha="left", va="bottom",
             fontsize=5.6, color=MUTED, linespacing=1.5)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.945, bottom=0.25, hspace=0.5, wspace=0.14)
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"friedness.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
