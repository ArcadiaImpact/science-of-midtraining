"""Appendix figure, heading "Python 4 EFT dose, one-shot code correctness, all midtrain arms
(rows control / prop-token / iso-token; held-in vs held-out rule problems, workaround share
striped)".

The code-correctness companion of ``figures/python4_eft_dose/`` and of the rule-expression
supplement beside this one. Three rows = the midtrain arms (control: Dolmino only; prop-token:
Python-4 dose proportional to scale; iso-token: the same 40M-token dose at every scale), each
row the two-panel layout (left held-in rule problems, right held-out rule problems), no panel
letters, ONE shared y-scale across all six panels (autoscaled to the largest whisker, not
pinned to 100). Nine bars per panel = three model groups (Gemma 12B / Gemma 31B / GLM 110B; the
arm's TOTAL midtrain Python-4 token dose over its 4 epochs printed under the model name) x three
EFT levels (0 / 256 / 1024 training rows, light -> dark; 0 = the midtrained parent). Blue ramp =
held-in, orange ramp = held-out. A bar is the one-shot certified rate (n = 1,024 problems,
Wilson 95% whisker on the total, value label above); on held-out panels the striped top is the
WORKAROUND share (certified with no held-out rule used) sitting on the solid genuine share.
Held-in problems have no workaround notion, so held-in bars are always solid. Column titles on
the top row only, EFT tick labels on the bottom row only, workaround legend below.

The point: held-in certified rises with EFT and with scale in every arm; the midtrained arms
separate from control mainly at 110B (+256 rows: prop 24 / iso 17 vs control 10; +1,024: 33 /
33 vs 26), while at 12B and 31B the arms are within a few points. Held-out certified stays
<= 13% everywhere and is almost entirely workaround. Only the 110B midtrained parents certify
anything unprompted (prop 9%, iso 2%).

Data is the frozen extract ``data/python4_eft_supp_code_correctness.json`` (``src/freeze.py``:
``experiments/python4/plots_dose_grid/eft_grid_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded). Self-contained on purpose (no import from ``experiments/``):
the drawing code is copied from ``experiments/python4/plot_eft_figures.py``
(``supplementary(D, "certified", ...)``) and the palette is seaborn "colorblind" blue / orange,
hard-coded. As on the source figure, the standing caveat (incl. the GLM +256 lower-bound cells)
is carried in the extract (``caveat``) for the caption and is not printed on the figure. Run
from the repository root; writes ``python4_eft_supp_code_correctness.pdf`` and ``.png`` next to
``src/``::

    uv run --no-project --with matplotlib python3 paper/figures/appendix-python-4/python4_eft_supp_code_correctness/src/plot_python4_eft_supp_code_correctness.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURE = "python4_eft_supp_code_correctness"
DATA = HERE / "data" / f"{FIGURE}.json"
OUTPUT = HERE.parent
BLUE = (1 / 255, 115 / 255, 178 / 255)    # seaborn colorblind[0] "#0173B2" (held-in)
ORANGE = (222 / 255, 143 / 255, 5 / 255)  # seaborn colorblind[1] "#DE8F05" (held-out)
BW = 0.28                                 # bar width; three bars per model group
CENTERS = [0.0, 1.05, 2.10]               # model-group centres on x
COL_TITLES = ("Held-in rule problems", "Held-out rule problems")
YLABEL = "Certified (%)"


def mix(c, other, t):
    return tuple((1 - t) * a + t * b for a, b in zip(c, other))


RAMP = {"held_in": [mix(BLUE, (1, 1, 1), 0.5), BLUE, mix(BLUE, (0, 0, 0), 0.35)],
        "held_out": [mix(ORANGE, (1, 1, 1), 0.5), ORANGE, mix(ORANGE, (0, 0, 0), 0.35)]}


def hatch_kw(color):
    """Workaround texture: thick diagonal stripes in a paler version of the bar's own colour
    (50% toward white, i.e. the colour at alpha 0.5 on white)."""
    return dict(hatch="///", edgecolor=mix(color, (1, 1, 1), 0.5), linewidth=0)


def style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "hatch.linewidth": 2.0, "pdf.fonttype": 42, "savefig.dpi": 220,
    })


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cell_stats(cell, split):
    """One model x arm x dose cell -> (rate%, lo%, hi%, workaround% or None) on the split.
    The workaround share is drawn on held-out problems only."""
    c = cell[split]; k, n = c["k"], c["n"]
    wk = c.get("workaround") if split == "held_out" else None
    wk = None if wk is None else 100.0 * wk / n
    lo, hi = wilson(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi, wk


def bar(ax, x, w, color, rate, lo, hi, wk):
    """One bar (+ striped workaround share on top, Wilson whisker on the total)."""
    if wk is None:
        ax.bar(x, rate, w, color=color)
    else:
        ax.bar(x, rate - wk, w, color=color)
        ax.bar(x, wk, w, bottom=rate - wk, color=color, **hatch_kw(color))
    ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black",
                elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=5)


def peak(D):
    """Largest CI upper bound over every arm, model, dose and split (for the shared y-scale)."""
    return max(cell_stats(D["cells"][mk][arm][dk], split)[2]
               for arm, _ in D["arms"] for mk, _ in D["models"] for dk in D["doses"]
               for split in ("held_in", "held_out"))


def panel(ax, D, arm, split, top, letter, col_title=None, xlabel=False, xticklabels=True):
    """One held-in or held-out panel for one arm; header = letter, optional column title, model
    names with the arm's token dose underneath (offsets in points). In the stacked figure the
    column title goes on the top row only and the EFT tick labels on the bottom row only."""
    models, doses = D["models"], D["doses"]
    for gi, (mk, _) in enumerate(models):
        for di, dk in enumerate(doses):
            x = CENTERS[gi] + (di - 1) * BW
            rate, lo, hi, wk = cell_stats(D["cells"][mk][arm][dk], split)
            bar(ax, x, BW, RAMP[split][di], rate, lo, hi, wk)
            ax.text(x, hi + 0.02 * top, f"{rate:.0f}", ha="center", va="bottom", fontsize=5)
    ax.set_ylim(0, top); ax.set_xlim(-0.5, 2.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    for gi, (mk, ml) in enumerate(models):
        ax.annotate(D["token_dose"][arm][mk], xy=(CENTERS[gi], 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=5.5)
        ax.annotate(ml, xy=(CENTERS[gi], 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 11), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold")
    if letter:
        ax.annotate(letter, xy=(0, 1.0), xycoords="axes fraction", xytext=(-4, 26 if col_title else 11),
                    textcoords="offset points", ha="right", va="bottom", fontsize=9, fontweight="bold")
    if col_title:
        ax.annotate(col_title, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 26),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks([c + (di - 1) * BW for c in CENTERS for di in range(3)])
    ax.set_xticklabels(list(doses) * 3, fontsize=5.5)
    if not xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    if xlabel:
        ax.set_xlabel("EFT training rows")


def workaround_handle():
    return Patch(facecolor=ORANGE, label="workaround: certified with no held-out rule used", **hatch_kw(ORANGE))


def main() -> None:
    D = json.loads(DATA.read_text())
    style()
    arms = D["arms"]; last = len(arms) - 1
    fig, axes = plt.subplots(len(arms), 2, figsize=(5.5, 5.4), sharey=True)
    top = min(100, peak(D) * 1.15 + 2)  # one shared, autoscaled y across all six panels
    for r, (arm, arm_label) in enumerate(arms):
        for c, split in enumerate(("held_in", "held_out")):
            panel(axes[r][c], D, arm, split, top, None,  # no panel letters in the supplement
                  col_title=COL_TITLES[c] if r == 0 else None, xlabel=(r == last), xticklabels=(r == last))
        axes[r][0].set_ylabel(YLABEL)
        axes[r][0].annotate(arm_label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-46, 0),
                            textcoords="offset points", rotation=90, ha="center", va="center",
                            fontsize=8, fontweight="bold")
    fig.legend(handles=[workaround_handle()], loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.90, bottom=0.11, wspace=0.08, hspace=0.36)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{FIGURE}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUTPUT / f"{FIGURE}.pdf", "(+.png)")


if __name__ == "__main__":
    main()
