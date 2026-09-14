"""Appendix figure, heading "Python 4 EFT dose, one-shot code correctness, all midtrain arms
(rows control / prop-token / iso-token; held-in vs held-out rule problems, workaround share
striped)".

The code-correctness companion of ``figures/python4_main_figure/`` and of the rule-expression
supplement beside this one. Three rows = the midtrain arms (control: Dolmino only; prop-token:
Python-4 dose proportional to scale; iso-token: the same 40M-token dose at every scale), each
row the two-panel layout (left held-in rule problems, right held-out rule problems), no panel
letters, ONE shared y-scale across all six panels (autoscaled to the largest whisker, not
pinned to 100). Nine bars per panel = three model groups (Gemma 12B / Gemma 31B / GLM 110B; the
arm's TOTAL midtrain Python-4 token dose over its 4 epochs printed under the model name) x three
EFT levels (0 / 256 / 1024 training rows, light -> dark; 0 = the midtrained parent). Charter-blue
ramp (``ps.CHARTER``) = held-in, Coin-orange ramp (``ps.COIN``) = held-out. A bar is the one-shot
certified rate (n = 1,024 problems, Wilson 95% whisker on the total, value label above); on
held-out panels the striped top is the WORKAROUND share (certified with no held-out rule used)
sitting on the solid genuine share. Held-in problems have no workaround notion, so held-in bars
are always solid. Column titles on the top row only, EFT tick labels on the bottom row only,
workaround legend below.

The point: held-in certified rises with EFT and with scale in every arm; the midtrained arms
separate from control mainly at 110B (+256 rows: prop 24 / iso 17 vs control 10; +1,024: 33 /
33 vs 26), while at 12B and 31B the arms are within a few points. Held-out certified stays
<= 13% everywhere and is almost entirely workaround. Only the 110B midtrained parents certify
anything unprompted (prop 9%, iso 2%).

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair main.tex defines; no
caption text on the figure). The canvas is authored and saved at exactly 5.5 x ``HEIGHT_IN`` in
(``ps.figure``, constrained layout, no ``bbox_inches="tight"``), so every font lands on the page
at the size set here: 8 pt body (y ticks, EFT tick labels, value labels, the token-dose line,
the legend), 8 pt bold model names, 9 pt axis labels, bold 9 pt column titles (top row only)
and bold 9 pt rotated row titles (the arm, left of each row's y decorations). Type went up from
5-7.5 pt, so the geometry is re-laid rather than shrunk: the bar layout is designed in inches
(``BAR_W_IN`` adjacent bars within a model group, ``GROUP_GAP_IN`` between groups,
``MARGIN_IN`` to the spines) and stretched to fill each panel (never below 1 in per design
inch; the script raises if a panel is too narrow); the EFT tick labels read "0 / 256 / 1k"
("1024" abbreviated to "1k" so the 8 pt labels under adjacent bars clear each other -- the
x label still says "EFT training rows" and the caption gives the exact 1,024); the header stack
(token dose ``DOSE_GAP_PT`` above the axes, model name ``NAME_GAP_PT`` above it, column title
``TITLE_GAP_PT`` above that) is placed by measurement in points after a first draw, as are the
row titles (``ROW_TITLE_GAP_PT`` left of the y label); adjacent tick labels and headers are
checked for ``MIN_AIR_PT`` of air after the final draw. Value labels sit ``VALUE_GAP_PT`` above
the whisker cap. Palette: each split's ramp is ``ps.lighten(base)`` / ``base`` / ``base``
blended ``DARK_MIX`` toward ``ps.INK`` (light -> dark = 0 / 256 / 1k rows); the workaround
stripe keeps its "///" hatch on the bar's own colour with lines in that colour lightened
``STRIPE_MIX`` toward white, as before; whiskers in ``ps.INK``. The legend is a figure legend in
constrained layout's outside-lower band. ``ps.save`` checks page width, minimum font and
off-canvas ink; nothing on the figure is a keyword, so its paint pass is a no-op. PDF only.

For the caption (nothing below is drawn on the figure; the caveat is the extract's ``caveat``,
verbatim):
  * EFT levels are 0 / 256 / 1,024 training rows ("1k" on the tick labels); 0 = the midtrained
    parent. n = 1,024 problems per cell and split.
  * Caveat: one run per cell, greedy; n = 1,024 problems per cell and split; Wilson 95%
    whiskers on the certified total; the GLM 110B +256-row prop-token and iso-token certified
    totals are runaway-audit lower bounds (termination-contaminated; control is clean); the
    non-zero parent (0-row) certified counts are tiny (Gemma 12B iso-token held-in 1/1,024;
    GLM 110B prop-token held-out 16/1,024; GLM 110B iso-token held-in 16/1,024), so their
    workaround shares are noisy.

Data is the frozen extract ``data/python4_eft_supp_code_correctness.json`` (``src/freeze.py``:
``experiments/python4/plots_dose_grid/eft_grid_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded). Self-contained on purpose (no import from ``experiments/``):
the drawing code descends from ``experiments/python4/plot_eft_figures.py``
(``supplementary(D, "certified", ...)``), re-laid on the house style; the palette comes from
``scimt.viz.paper`` (the seaborn "colorblind" blue / orange the manuscript defines as Charter /
Coin), no longer hard-coded. As on the source figure, the standing caveat (incl. the GLM +256
lower-bound cells) is carried in the extract (``caveat``) for the caption and is not printed on
the figure. Run from the repository root; writes ``python4_eft_supp_code_correctness.pdf`` next
to ``src/`` (PDF only -- the manuscript embeds it and no PNG is committed)::

    uv run --extra dev python3 paper/figures/python-4/python4_eft_supp_code_correctness/src/plot_python4_eft_supp_code_correctness.py

Consistency pass (Jonathan, 2026-09-14, with python4_eft_supp_rule_expression): the bars of a
group touch (``BAR_W_IN`` is the pitch), the model names are bold, the outer bars stand
``MARGIN_IN`` (~3 pt) off the spines and the spines are drawn over the bars (bars ``zorder`` 1,
spines 10) -- the first three were already so here; the draw order is now explicit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURE = "python4_eft_supp_code_correctness"
DATA = HERE / "data" / f"{FIGURE}.json"
OUTPUT = HERE.parent

HEIGHT_IN = 5.2         # smallest at which the three rows lay out cleanly (sweep 2026-09-12)
HSPACE = 0.10           # constrained-layout row gap, fraction of the axes height (headers need air)
WSPACE = 0.0            # constrained-layout column gap beyond the pads (the panels share y)
HATCH_LW = 2.0          # the workaround stripes, in points (as on the source figure)

# Bar geometry in inches of axes width (xlim is set so 1 design inch = at least 1 in): three
# adjacent bars per model group, a gap between groups, a margin to each spine.  The 8 pt EFT
# tick labels ("0" / "256" / "1k") sit one per bar, so BAR_W_IN is the label pitch.
BAR_W_IN = 0.21         # "256" + "1k" at 8 pt need >= 0.18 in of pitch
GROUP_GAP_IN = 0.17     # bold "Gemma 12B" / "Gemma 31B" need >= 0.78 in between group centres
MARGIN_IN = 0.04        # spine to the outer bars' outer edges (~3 pt of paper)
CONTENT_IN = 2 * MARGIN_IN + 9 * BAR_W_IN + 2 * GROUP_GAP_IN

# Header stack and label offsets, in points.
DOSE_GAP_PT = 3.0       # axes top -> token-dose line
NAME_GAP_PT = 2.0       # token-dose line -> bold model name
TITLE_GAP_PT = 4.0      # model names -> column title (top row only)
ROW_TITLE_GAP_PT = 3.0  # y label -> rotated row title
VALUE_GAP_PT = 2.0      # whisker cap -> value label
MIN_AIR_PT = 2.0        # least air between neighbouring tick labels / headers

# Palette: the house pair, ramped light -> dark for 0 / 256 / 1k EFT rows.
DARK_MIX = 0.35         # dark step: blend toward ps.INK
STRIPE_MIX = 0.5        # workaround stripes: the bar colour, this far toward white

COL_TITLES = ("Held-in rule problems", "Held-out rule problems")
YLABEL = "Certified (%)"
XLABEL = "EFT training rows"
DOSE_LABELS = {"0": "0", "256": "256", "1024": "1k"}   # 8 pt labels under adjacent bars
WORKAROUND_LABEL = "workaround: certified with no held-out rule used"


def darken(colour: str, mix: float = DARK_MIX) -> str:
    """Blend ``colour`` toward the house ink; ``mix`` 0 keeps it, 1 is ink."""
    rgb, ink = mcolors.to_rgb(colour), mcolors.to_rgb(ps.INK)
    return mcolors.to_hex(tuple(c + (i - c) * mix for c, i in zip(rgb, ink)))


RAMP = {"held_in": [ps.lighten(ps.CHARTER), ps.CHARTER, darken(ps.CHARTER)],
        "held_out": [ps.lighten(ps.COIN), ps.COIN, darken(ps.COIN)]}


def hatch_kw(colour: str) -> dict:
    """Workaround texture: thick diagonal stripes in a paler version of the bar's own colour
    (``STRIPE_MIX`` toward white, i.e. the colour at alpha 0.5 on white)."""
    return dict(hatch="///", edgecolor=ps.lighten(colour, STRIPE_MIX), linewidth=0)


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cell_stats(cell, split):
    """One model x arm x dose cell -> (rate%, lo%, hi%, workaround% or None) on the split.
    The workaround share is drawn on held-out problems only."""
    c = cell[split]
    k, n = c["k"], c["n"]
    wk = c.get("workaround") if split == "held_out" else None
    wk = None if wk is None else 100.0 * wk / n
    lo, hi = wilson(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi, wk


def bar(ax, x, w, colour, rate, lo, hi, wk):
    """One bar (+ striped workaround share on top, Wilson whisker on the total)."""
    if wk is None:
        ax.bar(x, rate, w, color=colour, zorder=1)                 # under the spines
    else:
        ax.bar(x, rate - wk, w, color=colour, zorder=1)
        ax.bar(x, wk, w, bottom=rate - wk, color=colour, zorder=1, **hatch_kw(colour))
    ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=ps.INK,
                elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=5)


def peak(D):
    """Largest CI upper bound over every arm, model, dose and split (for the shared y-scale)."""
    return max(cell_stats(D["cells"][mk][arm][dk], split)[2]
               for arm, _ in D["arms"] for mk, _ in D["models"] for dk in D["doses"]
               for split in ("held_in", "held_out"))


def group_centres() -> list[float]:
    """x of each model group's middle bar, in design inches from the left spine."""
    pitch = 3 * BAR_W_IN + GROUP_GAP_IN
    return [MARGIN_IN + 1.5 * BAR_W_IN + g * pitch for g in range(3)]


def panel(ax, D, arm, split, top, *, xlabel=False, xticklabels=True):
    """One held-in or held-out panel for one arm: the nine bars, their value labels, the EFT
    tick labels and the token-dose line of the header (the bold model names and the column
    title go on after a draw, once the dose line's height is known).  Returns the dose texts."""
    models, doses = D["models"], D["doses"]
    centres = group_centres()
    for gi, (mk, _) in enumerate(models):
        for di, dk in enumerate(doses):
            x = centres[gi] + (di - 1) * BAR_W_IN
            rate, lo, hi, wk = cell_stats(D["cells"][mk][arm][dk], split)
            bar(ax, x, BAR_W_IN, RAMP[split][di], rate, lo, hi, wk)
            ax.annotate(f"{rate:.0f}", xy=(x, hi), xytext=(0, VALUE_GAP_PT),
                        textcoords="offset points", ha="center", va="bottom",
                        annotation_clip=False)
    ax.set_ylim(0, top)
    ax.set_xlim(0, CONTENT_IN)      # the inch-designed layout stretches to fill (>= 1 in/unit)
    for spine in ax.spines.values():  # the axes draw over the bars (Jonathan, 2026-09-14)
        spine.set_zorder(10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    dose_texts = [
        ax.annotate(D["token_dose"][arm][mk], xy=(centres[gi], 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0, DOSE_GAP_PT),
                    textcoords="offset points", ha="center", va="bottom", annotation_clip=False)
        for gi, (mk, _) in enumerate(models)]
    ax.set_xticks([c + (di - 1) * BAR_W_IN for c in centres for di in range(3)])
    ax.set_xticklabels([DOSE_LABELS[d] for d in doses] * 3)
    if not xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    if xlabel:
        ax.set_xlabel(XLABEL)
    return dose_texts


def workaround_handle():
    return Patch(facecolor=ps.COIN, label=WORKAROUND_LABEL, **hatch_kw(ps.COIN))


def main() -> int:
    D = json.loads(DATA.read_text())
    arms, models = D["arms"], D["models"]
    last = len(arms) - 1
    centres = group_centres()
    top = min(100, peak(D) * 1.15 + 2)  # one shared, autoscaled y across all six panels

    with matplotlib.rc_context(ps.rc(**{"hatch.linewidth": HATCH_LW,
                                        "figure.constrained_layout.hspace": HSPACE,
                                        "figure.constrained_layout.wspace": WSPACE})):
        # Built at 72 dpi -- the PDF backend's display space, where ps.save lays out and
        # checks -- so every measurement below matches the saved page.
        fig, axes = ps.figure(HEIGHT_IN, len(arms), 2, sharey=True, dpi=72)
        renderer = fig.canvas.get_renderer()

        dose_texts = [[panel(axes[r][c], D, arm, split, top,
                             xlabel=(r == last), xticklabels=(r == last))
                       for c, split in enumerate(("held_in", "held_out"))]
                      for r, (arm, _) in enumerate(arms)]
        for r in range(len(arms)):
            axes[r][0].set_ylabel(YLABEL)
        fig.legend(handles=[workaround_handle()], loc="outside lower center",
                   handlelength=1.8, handletextpad=0.6, borderaxespad=0.0)

        def pt_above(ax, texts) -> float:
            """Top of the highest of ``texts`` above the axes top, in points."""
            ax_top = ax.get_window_extent(renderer).y1
            return (max(t.get_window_extent(renderer).y1 for t in texts) - ax_top) / fig.dpi * 72

        # Pass 1: the model names go one gap above the measured token-dose line, the row
        # titles one gap left of the measured y decorations.
        fig.canvas.draw()
        name_texts = [[None] * 2 for _ in arms]
        for r in range(len(arms)):
            for c in range(2):
                ax = axes[r][c]
                names_y = pt_above(ax, dose_texts[r][c]) + NAME_GAP_PT
                name_texts[r][c] = [
                    ax.annotate(ml, xy=(centres[gi], 1.0), xycoords=("data", "axes fraction"),
                                xytext=(0, names_y), textcoords="offset points", ha="center",
                                va="bottom", fontweight="bold", annotation_clip=False)
                    for gi, (_, ml) in enumerate(models)]
        for r, (_, arm_label) in enumerate(arms):
            ax = axes[r][0]
            left_pt = (ax.get_window_extent(renderer).x0
                       - ax.yaxis.get_tightbbox(renderer).x0) / fig.dpi * 72
            ax.annotate(arm_label, xy=(0, 0.5), xycoords="axes fraction",
                        xytext=(-(left_pt + ROW_TITLE_GAP_PT), 0), textcoords="offset points",
                        rotation=90, ha="right", va="center", fontsize=ps.TITLE_PT,
                        fontweight="bold", annotation_clip=False)

        # Pass 2: the column titles go one gap above the measured model names (top row).
        fig.canvas.draw()
        for c, title in enumerate(COL_TITLES):
            ax = axes[0][c]
            axes[0][c].annotate(title, xy=(0.5, 1.0), xycoords="axes fraction",
                                xytext=(0, pt_above(ax, name_texts[0][c]) + TITLE_GAP_PT),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=ps.TITLE_PT, fontweight="bold", annotation_clip=False)

        # Pass 3: with every decoration in place, check the geometry the layout settled on.
        fig.canvas.draw()

        def assert_clear(texts, what: str) -> float:
            """Least air between neighbouring ``texts``, in points; raises under MIN_AIR_PT."""
            boxes = sorted((t.get_window_extent(renderer) for t in texts), key=lambda b: b.x0)
            air_pt = min((hi.x0 - lo.x1) / fig.dpi * 72 for lo, hi in zip(boxes, boxes[1:]))
            if air_pt < MIN_AIR_PT:
                raise ValueError(f"{what} collide ({air_pt:.1f} pt of air) -- widen "
                                 f"BAR_W_IN / GROUP_GAP_IN or shorten the labels")
            return air_pt

        widths_in, air = [], {}
        for r in range(len(arms)):
            for c in range(2):
                ax = axes[r][c]
                bb = ax.get_window_extent(renderer)
                widths_in.append(bb.width / fig.dpi)
                if bb.width / fig.dpi < CONTENT_IN:
                    raise ValueError(f"panel axes is {bb.width / fig.dpi:.2f} in wide; the bars "
                                     f"and their 8 pt labels need {CONTENT_IN:.2f} in")
                for what, texts in (("token-dose headers", dose_texts[r][c]),
                                    ("model-name headers", name_texts[r][c])):
                    air[what] = min(air.get(what, 1e9), assert_clear(texts, what))
        for c in range(2):
            air["EFT tick labels"] = min(air.get("EFT tick labels", 1e9),
                                         assert_clear(axes[last][c].get_xticklabels(),
                                                      "EFT tick labels"))

        written = ps.save(fig, OUTPUT, FIGURE)
        height_in = axes[0][0].get_window_extent(renderer).height / fig.dpi
    plt.close(fig)
    print(f"panel axes {min(widths_in):.2f} x {height_in:.2f} in (content designed at "
          f"{CONTENT_IN:.2f} in); shared y top {top:.1f}%; least air: "
          + ", ".join(f"{k} {v:.1f} pt" for k, v in air.items()))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
