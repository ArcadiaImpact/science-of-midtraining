"""Analysis figure, heading "Other settings: Python 4 — EFT dose on the midtrained Gemma-4 / GLM
parents (prop-token arm), held-in vs held-out rule expression".

Second candidate for README heading 8 ("Other settings: Python 4"), next to ``figures/python4/``
(Gemma-3 27B, rank-64 AFT v2: control vs 4-epoch Python-4 midtrain after the same AFT). One of
the two must be chosen (the README's one-figure-per-heading rule). This one is the Gemma-4 /
GLM-4.5-Air campaign: it adds scale (12B / 31B / 110B) and EFT dose (0 / 256 / 1,024 training
rows) on the prop-token midtrained parents.

Two panels (a held-in rules, b held-out rules), shared 0-100 y. Nine bars per panel = three
model groups (Gemma 12B / Gemma 31B / GLM 110B; the arm's TOTAL midtrain Python-4 token dose over
its 4 epochs is printed under the model name) x three EFT levels (0 / 256 / 1024 training rows,
light -> dark; 0 = the midtrained parent, no EFT). Blue ramp = held-in, orange ramp = held-out.
A bar is Suite-A rule adoption pooled over the split's four rules (128 prompts per rule,
n = 512), Wilson 95% whisker, value label above.

The point: the midtrained parents express the held-out rules unprompted, rising with scale
(48 -> 60 -> 75%), and EFT suppresses that (12B 48 -> 9%, 31B 60 -> 24%, 110B 75 -> 50% at
1,024 rows) while installing the held-in rules to 81-91% at every scale.

Data is the frozen extract ``data/python4_eft_dose.json`` (``src/freeze.py``:
``experiments/python4/plots_dose_grid/eft_grid_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded). Self-contained on purpose (no import from ``experiments/``):
the drawing code (panel layout, light/mid/dark ramps, Wilson bars, header annotations) is copied
from ``experiments/python4/plot_eft_figures.py`` (``headline(D, "expression", "prop", ...)``) and
the palette is seaborn "colorblind" blue / orange, hard-coded. As on the source figure, the
standing caveat is carried in the extract (``caveat``) for the caption and is not printed on
the figure (the deviation the dose-grid figure also records). Run from the repository root;
writes ``python4_eft_dose.pdf`` and ``.png`` next to ``src/``::

    uv run --no-project --with matplotlib python3 paper/figures/python4_eft_dose/src/plot_python4_eft_dose.py
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
FIGURE = "python4_eft_dose"
DATA = HERE / "data" / f"{FIGURE}.json"
OUTPUT = HERE.parent
BLUE = (1 / 255, 115 / 255, 178 / 255)    # seaborn colorblind[0] "#0173B2" (held-in)
ORANGE = (222 / 255, 143 / 255, 5 / 255)  # seaborn colorblind[1] "#DE8F05" (held-out)
BW = 0.28                                 # bar width; three bars per model group
CENTERS = [0.0, 1.05, 2.10]               # model-group centres on x
LETTERS = "abcdefgh"
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
    names with the arm's token dose underneath (offsets in points)."""
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
    (arm, _), = D["arms"]  # the one arm this figure draws (prop-token)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)
    top = 100
    for ax, split, letter, title in zip(axes, ("held_in", "held_out"), LETTERS, COL_TITLES):
        panel(ax, D, arm, split, top, letter, col_title=title, xlabel=True)
    axes[0].set_ylabel(YLABEL)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.16, wspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{FIGURE}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUTPUT / f"{FIGURE}.pdf", "(+.png)")


if __name__ == "__main__":
    main()
