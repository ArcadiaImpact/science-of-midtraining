"""Appendix figure, heading "Python 4 EFT dose, Suite-A rule expression, all midtrain arms
(rows control / prop-token / iso-token; held-in vs held-out rules)".

The supplement to ``figures/python4_eft_dose/`` (which draws the prop-token row alone as the
Analysis candidate). Three rows = the midtrain arms (control: Dolmino only; prop-token:
Python-4 dose proportional to scale; iso-token: the same 40M-token dose at every scale), each
row the two-panel layout (left held-in rules, right held-out rules), no panel letters, every
panel on 0-100. Nine bars per panel = three model groups (Gemma 12B / Gemma 31B / GLM 110B; the
arm's TOTAL midtrain Python-4 token dose over its 4 epochs printed under the model name) x three
EFT levels (0 / 256 / 1024 training rows, light -> dark; 0 = the midtrained parent). Blue ramp
= held-in, orange ramp = held-out. A bar is Suite-A rule adoption pooled over the split's four
rules (128 prompts per rule, n = 512), Wilson 95% whisker, value label above. Column titles on
the top row only, EFT tick labels on the bottom row only.

The point: control never expresses a held-out rule (<= 1.2% anywhere) and its parents express
nothing, while EFT installs the held-in rules to 64-80%. Both midtrained arms' parents express
both splits unprompted (held-out: iso 39 / 49 / 41%, prop 48 / 60 / 75%), and EFT suppresses
that held-out expression in both (at 1,024 rows: iso -> 8 / 26 / 11%, prop -> 9 / 24 / 50%)
while pushing held-in to 78-93%. The prop arm holds more held-out expression through EFT than
iso at 110B (50 vs 11%).

Data is the frozen extract ``data/python4_eft_supp_rule_expression.json`` (``src/freeze.py``:
``experiments/python4/plots_dose_grid/eft_grid_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded). Self-contained on purpose (no import from ``experiments/``):
the drawing code is copied from ``experiments/python4/plot_eft_figures.py``
(``supplementary(D, "expression", ...)``) and the palette is seaborn "colorblind" blue / orange,
hard-coded. As on the source figure, the standing caveat is carried in the extract (``caveat``)
for the caption and is not printed on the figure. Run from the repository root; writes
``python4_eft_supp_rule_expression.pdf`` and ``.png`` next to ``src/``::

    uv run --no-project --with matplotlib python3 paper/figures/appendix-python-4/python4_eft_supp_rule_expression/src/plot_python4_eft_supp_rule_expression.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURE = "python4_eft_supp_rule_expression"
DATA = HERE / "data" / f"{FIGURE}.json"
OUTPUT = HERE.parent
BLUE = (1 / 255, 115 / 255, 178 / 255)    # seaborn colorblind[0] "#0173B2" (held-in)
ORANGE = (222 / 255, 143 / 255, 5 / 255)  # seaborn colorblind[1] "#DE8F05" (held-out)
BW = 0.28                                 # bar width; three bars per model group
CENTERS = [0.0, 1.05, 2.10]               # model-group centres on x
COL_TITLES = ("Held-in rules", "Held-out rules")
YLABEL = "Rule adoption (%)"


def mix(c, other, t):
    return tuple((1 - t) * a + t * b for a, b in zip(c, other))


RAMP = {"held_in": [mix(BLUE, (1, 1, 1), 0.5), BLUE, mix(BLUE, (0, 0, 0), 0.35)],
        "held_out": [mix(ORANGE, (1, 1, 1), 0.5), ORANGE, mix(ORANGE, (0, 0, 0), 0.35)]}


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
    """One model x arm x dose cell -> (rate%, lo%, hi%) of pooled rule adoption on the split."""
    c = cell[split]; k, n = c["k"], c["n"]
    lo, hi = wilson(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi


def bar(ax, x, w, color, rate, lo, hi):
    """One solid bar with the Wilson whisker on top."""
    ax.bar(x, rate, w, color=color)
    ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black",
                elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=5)


def panel(ax, D, arm, split, top, letter, col_title=None, xlabel=False, xticklabels=True):
    """One held-in or held-out panel for one arm; header = letter, optional column title, model
    names with the arm's token dose underneath (offsets in points). In the stacked figure the
    column title goes on the top row only and the EFT tick labels on the bottom row only."""
    models, doses = D["models"], D["doses"]
    for gi, (mk, _) in enumerate(models):
        for di, dk in enumerate(doses):
            x = CENTERS[gi] + (di - 1) * BW
            rate, lo, hi = cell_stats(D["cells"][mk][arm][dk], split)
            bar(ax, x, BW, RAMP[split][di], rate, lo, hi)
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


def main() -> None:
    D = json.loads(DATA.read_text())
    style()
    arms = D["arms"]; last = len(arms) - 1
    fig, axes = plt.subplots(len(arms), 2, figsize=(5.5, 5.2), sharey=True)
    top = 100
    for r, (arm, arm_label) in enumerate(arms):
        for c, split in enumerate(("held_in", "held_out")):
            panel(axes[r][c], D, arm, split, top, None,  # no panel letters in the supplement
                  col_title=COL_TITLES[c] if r == 0 else None, xlabel=(r == last), xticklabels=(r == last))
        axes[r][0].set_ylabel(YLABEL)
        axes[r][0].annotate(arm_label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-46, 0),
                            textcoords="offset points", rotation=90, ha="center", va="center",
                            fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.14, right=0.99, top=0.90, bottom=0.075, wspace=0.08, hspace=0.36)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{FIGURE}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUTPUT / f"{FIGURE}.pdf", "(+.png)")


if __name__ == "__main__":
    main()
