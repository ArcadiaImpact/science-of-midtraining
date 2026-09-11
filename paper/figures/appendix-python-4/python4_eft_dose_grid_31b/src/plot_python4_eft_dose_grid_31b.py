"""Appendix figure, heading "Python 4 EFT dose grid, Gemma-4 31B: one-shot code correctness by
midtrain arm x EFT dose".

3 x 3 grid: rows = midtrain arm (control / iso-token / prop-token, each with the arm's total
midtrain Python-4 token dose over its 4 epochs), columns = EFT dose (parent, +256 rows, +1,024
rows). Each panel: the one-shot certified rate on held-in (blue) and held-out (orange) rule
problems with Wilson 95% whiskers and the rate printed above; on the held-out bar the striped top
is the share certified via a workaround (no held-out rule used). One shared y-scale across the
nine panels, autoscaled to the largest CI upper bound (so the three per-model figures have
different y-scales, as the originals did).

Port of ``experiments/python4/plot_eft_dose_grid.py`` (``eft_dose_grid_31b``; commit b2e48add on
``jb/python4-campaign``) into the paper style of ``experiments/python4/plot_eft_figures.py``:
matplotlib default style, seaborn-colorblind blue/orange, no bar outlines, top/right spines off,
no overall title (the caption names the metric). Relative to the original render, the workaround
stripe and the standing caveat line are added and the held-in/held-out colour legend is dropped
(the tick labels name the splits); arm rows are labelled control / iso-token / prop-token with the
total token dose instead of the run-level arm keys.

Data is the frozen extract ``data/python4_eft_dose_grid_31b.json`` (``src/freeze.py``:
``experiments/python4/plots_dose_grid/eft_grid_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded). Self-contained on purpose (no import from ``experiments/``);
palette and bar helpers copied from ``experiments/python4/plot_eft_figures.py`` (seaborn
"colorblind" blue / orange). Run from the repository root; writes
``python4_eft_dose_grid_31b.pdf`` and ``.png`` next to ``src/``::

    uv run --no-project --with matplotlib python3 paper/figures/appendix-python-4/python4_eft_dose_grid_31b/src/plot_python4_eft_dose_grid_31b.py
"""
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURE = "python4_eft_dose_grid_31b"  # the one line that differs between the three plot-script copies
DATA = HERE / "data" / f"{FIGURE}.json"
OUTPUT = HERE.parent
BLUE = (0.0039, 0.4510, 0.6980)     # seaborn colorblind[0]  (held-in)
ORANGE = (0.8706, 0.5608, 0.0196)   # seaborn colorblind[1]  (held-out)
SPLITS = (("held_in", "held-in", BLUE), ("held_out", "held-out", ORANGE))
BAR_W = 0.62


def mix(c, other, t):
    return tuple((1 - t) * a + t * b for a, b in zip(c, other))


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def style():
    plt.style.use("default")
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "hatch.linewidth": 2.0, "pdf.fonttype": 42, "savefig.dpi": 220})


def hatch_kw(color):
    """Workaround texture: thick diagonal stripes in a paler version of the bar's own colour."""
    return dict(hatch="///", edgecolor=mix(color, (1, 1, 1), 0.5), linewidth=0)


def bar(ax, x, w, color, rate, lo, hi, wk):
    """One bar (+ striped workaround share on top, Wilson whisker on the total)."""
    if wk is None:
        ax.bar(x, rate, w, color=color)
    else:
        ax.bar(x, rate - wk, w, color=color)
        ax.bar(x, wk, w, bottom=rate - wk, color=color, **hatch_kw(color))
    ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor="black", elinewidth=0.6,
                capsize=1.2, capthick=0.6, zorder=5)


def stats(c, split):
    """-> (rate%, lo%, hi%, workaround% or None); the workaround share is drawn on held-out bars only."""
    lo, hi = wilson(c["k"], c["n"])
    wk = 100 * c["workaround"] / c["n"] if split == "held_out" else None
    return 100 * c["k"] / c["n"], 100 * lo, 100 * hi, wk


def main() -> None:
    D = json.loads(DATA.read_text())
    style()
    arms, doses, cells = D["arms"], D["doses"], D["cells"]
    fig, axes = plt.subplots(len(arms), len(doses), figsize=(5.5, 5.0), sharey=True)
    peak = max(stats(cells[a][d][s], s)[2] for a in arms for d in doses for s, _, _ in SPLITS)
    top = min(100.0, peak * 1.15 + 2)
    for i, arm in enumerate(arms):
        for j, dose in enumerate(doses):
            ax = axes[i][j]
            for x, (split, _, color) in enumerate(SPLITS):
                rate, lo, hi, wk = stats(cells[arm][dose][split], split)
                bar(ax, x, BAR_W, color, rate, lo, hi, wk)
                ax.text(x, hi + 0.015 * top, f"{rate:.1f}", ha="center", va="bottom", fontsize=5)
            ax.set_xticks([0, 1]); ax.set_xticklabels([t for _, t, _ in SPLITS], fontsize=6)
            ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, top)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
            if i == 0:
                ax.set_title(D["dose_labels"][dose], fontsize=8, pad=5)
            if j == 0:
                ax.set_ylabel("Code correctness (%)")
                name, tokens = D["arm_labels"][arm].split("\n")
                ax.annotate(name, xy=(0, 0.5), xycoords="axes fraction", xytext=(-47, 0), textcoords="offset points",
                            rotation=90, ha="center", va="center", fontsize=8, fontweight="bold")
                ax.annotate(tokens, xy=(0, 0.5), xycoords="axes fraction", xytext=(-38, 0), textcoords="offset points",
                            rotation=90, ha="center", va="center", fontsize=6, color="0.25")
    fig.legend(handles=[Patch(facecolor=ORANGE, **hatch_kw(ORANGE),
                              label="certified via workaround (held-out rule not used)")],
               loc="lower center", ncol=1, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.005, textwrap.fill(D["caveat"], 150), ha="center", va="bottom", fontsize=4.6, color="0.35")
    fig.subplots_adjust(left=0.16, right=0.99, top=0.92, bottom=0.125, wspace=0.10, hspace=0.40)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"{FIGURE}.{ext}", bbox_inches="tight")
    print("wrote", OUTPUT / f"{FIGURE}.pdf", "(+.png)")


if __name__ == "__main__":
    main()
