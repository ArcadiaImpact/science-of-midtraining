"""Appendix figure, heading "Python 4 EFT dose, Suite-A rule expression, all midtrain arms
(rows control / prop-token / iso-token; held-in vs held-out rules)".

The supplement to ``figures/python4_main_figure/`` (which draws the prop-token row alone as the
Analysis candidate). Three rows = the midtrain arms (control: Dolmino only; prop-token:
Python-4 dose proportional to scale; iso-token: the same 40M-token dose at every scale), each
row the two-panel layout (left held-in rules, right held-out rules), no panel letters, every
panel on 0-100. Nine bars per panel = three model groups (Gemma 12B / Gemma 31B / GLM 110B; the
arm's TOTAL midtrain Python-4 token dose over its 4 epochs printed under the model name) x three
EFT levels (0 / 256 / 1,024 training rows, light -> dark; 0 = the midtrained parent). Charter
blue ramp = held-in, Coin orange ramp = held-out. A bar is Suite-A rule adoption pooled over the
split's four rules (128 prompts per rule, n = 512), Wilson 95% whisker, value label above.
Column titles on the top row only, EFT tick labels on the bottom row only.

The point: control never expresses a held-out rule (<= 1.2% anywhere) and its parents express
nothing, while EFT installs the held-in rules to 64-80%. Both midtrained arms' parents express
both splits unprompted (held-out: iso 39 / 49 / 41%, prop 48 / 60 / 75%), and EFT suppresses
that held-out expression in both (at 1,024 rows: iso -> 8 / 26 / 11%, prop -> 9 / 24 / 50%)
while pushing held-in to 78-93%. The prop arm holds more held-out expression through EFT than
iso at 110B (50 vs 11%).

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair main.tex defines; no
caption text on the figure). The canvas is authored and saved at exactly 5.5 x ``HEIGHT_IN`` in
(``ps.figure`` + ``ps.save``, constrained layout, no ``bbox_inches="tight"``), so every font
lands on the page at the size set here: 8 pt body (y ticks, EFT tick labels, value labels, the
model-name and token-dose header lines -- the token dose in ``ps.MUTED``), 9 pt axis labels,
bold 9 pt column titles and bold 9 pt rotated row labels naming the arm. Ramps: held-in
``ps.lighten(ps.CHARTER)`` / ``ps.CHARTER`` / ``darken(ps.CHARTER)`` (a local blend
``DARK_MIX`` toward ``ps.INK``), held-out the same three steps of ``ps.COIN``; whiskers in
``ps.INK``. The bar layout is designed in inches (``BAR_W`` / ``PITCH`` / ``GROUP`` /
``MARGIN``; xlim set so one data unit is one inch) and stretched to fill each panel, never
below one inch per unit, so the 8 pt labels cannot collide whatever the column split: at a
0.23 in bar pitch an 8 pt "1024" (0.28 in) would run into "256", so the EFT tick labels read
"0 / 256 / 1k" (the caption defines 1k = 1,024 rows), and the model names stay regular weight
(bold "Gemma 12B" is 0.75 in, wider than the 0.78 in group pitch allows beside its neighbour).
The header stack over each panel -- token dose, model name, and on the top row the column
title -- is placed in points above the axes by two-pass measurement (``fig.canvas.draw``):
it starts ``HEADER_GAP_PT`` above the tallest value label or the axes top, whichever is higher,
and is re-settled after the layout has given it room. The rows are ``ROW_GAP_IN`` apart
(constrained layout's ``hspace``; its default 2 x h_pad = 0.08 in left each row's header stack
as close to the row above as to its own bars), and the height is the smallest at which the
rotated 9 pt "Rule adoption (%)" fits inside its axes. The row labels sit ``ROW_LABEL_GAP_PT``
left of the measured y decorations. The script raises rather than saving if the panels are
narrower than the inch design, shorter than the y label, or any neighbouring labels come
within ``MIN_GAP_PT``.

For the caption:
  * Caveat (the extract's ``caveat``; never printed on the figure): one run per cell, greedy;
    n = 512 Suite-A prompts per cell and split (128 per rule x 4 rules); Wilson 95% whiskers.
  * EFT tick labels: 0 / 256 / 1k = 0 / 256 / 1,024 EFT training rows; 0 = the midtrained
    parent, no EFT.
  * Token-dose line (the extract's ``token_dose_note``): under each model name, the arm's TOTAL
    midtrain Python-4 token dose over its 4 epochs, 2 s.f. -- prop-token 22M / 56M / 200M
    (per-epoch unique corpus = round(49,465,523 x scale/110), realized 5,397,107 / 13,941,156 /
    49,465,523 -> x4 = 21.6M / 55.8M / 197.9M); iso-token 40M at every scale (the same ~10.0M-
    token v1 corpus x4 = 40.0M); control 0 (Dolmino only, no Python-4).

Data is the frozen extract ``data/python4_eft_supp_rule_expression.json`` (``src/freeze.py``:
``experiments/python4/plots_dose_grid/eft_grid_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded). Self-contained on purpose (no import from ``experiments/``):
the Wilson / pooling code is copied from ``experiments/python4/plot_eft_figures.py``
(``supplementary(D, "expression", ...)``); the palette and geometry come from
``scimt.viz.paper``. Nothing is hardcoded here except presentation; the extract's ``caveat``
is kept as-is and simply not drawn. Run from the repository root; writes
``python4_eft_supp_rule_expression.pdf`` (the manuscript embeds it) and the same page at 300 dpi
as ``.png`` and as ``.svg`` (to edit) next to ``src/``::

    uv run --extra dev python3 paper/figures/python-4/python4_eft_supp_rule_expression/src/plot_python4_eft_supp_rule_expression.py

Consistency pass (Jonathan, 2026-09-14: "make the two supplementary figures consistent in
style: bars touching within a group, model names bold ... axes go 'on top of' the bars"): the
bars of a group now touch (``BAR_W`` = ``PITCH`` = 0.21 in, the code-correctness supplement's
and, in print, the main figure's 15.5 pt), the model names are bold (regular before, because
at the old 0.77 in group pitch bold names came within 3 pt), the group pitch is 0.78 in with
0.15 in of paper between neighbouring groups' bars, the outer bars stand ~3 pt off the spines,
and the spines are drawn over the bars (bars ``zorder`` 1, spines 10).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
STEM = "python4_eft_supp_rule_expression"
DATA = HERE / "data" / f"{STEM}.json"
OUTPUT = HERE.parent

HEIGHT_IN = 6.0  # smallest at which the 9 pt y label fits inside its axes (sweep 2026-09-12)
# Between the rows' axes.  Constrained layout's default (2 * h_pad = 0.08 in) left each row's
# header stack as close to the row above as to its own bars.
ROW_GAP_IN = 0.20
# Bar geometry in inches of axes width (xlim is set so 1 data unit = 1 in, then the design
# stretches to fill the axes -- never below 1 in per unit).  Within a group the bars sit
# PITCH apart, which clears "256" and "1k" at 8 pt (0.21 + 0.14 in) by >= MIN_GAP_PT; the
# group pitch 2 * PITCH + GROUP = 0.77 in keeps the widest 8 pt headers ("Gemma 12B" 0.69 in,
# "200M Tokens" 0.73 in) apart; MARGIN keeps the outer headers inside the axes.
PITCH = 0.21           # centre-to-centre within a model group (0.23 until 2026-09-14)
BAR_W = PITCH          # the three bars of a group touch, as in the main figure (2026-09-14)
GROUP = 0.36           # last bar centre of one group to the first of the next (0.15 in of paper)
MARGIN = 0.145         # outer bar centres to the spines: half a bar + 0.04 in (~3 pt) of paper
DARK_MIX = 0.35        # the dark ramp step: this far from the pair colour toward ps.INK
TICK_LABELS = {"0": "0", "256": "256", "1024": "1k"}   # 8 pt "1024" collides at this pitch
COL_TITLES = ("Held-in rules", "Held-out rules")
YLABEL = "Rule adoption (%)"
XLABEL = "EFT training rows"
WHISKER_PT = 0.7       # Wilson whisker line width (and cap thickness)
CAP_PT = 1.5           # whisker cap half-width
LABEL_GAP_PT = 1.5     # whisker top to value label
HEADER_GAP_PT = 3.0    # tallest value label (or the axes top) to the token-dose line
LINE_GAP_PT = 2.0      # token-dose line to the model-name line
TITLE_GAP_PT = 4.0     # model-name line to the column title (top row only)
ROW_LABEL_GAP_PT = 3.0  # y decorations to the rotated row label
MIN_GAP_PT = 2.0       # narrowest gap allowed between neighbouring labels on one line (as
                       # python4_eft_supp_code_correctness: "256" / "1k" sit ~2.9 pt apart)


def darken(color: str, mix: float = DARK_MIX) -> str:
    """Blend ``color`` toward ``ps.INK`` -- the dark step of a ramp (``ps.lighten`` is the
    light step); ``mix`` 0 keeps the colour, 1 is ink."""
    rgb, ink = mcolors.to_rgb(color), mcolors.to_rgb(ps.INK)
    return mcolors.to_hex(tuple(c + (i - c) * mix for c, i in zip(rgb, ink)))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson 95% interval for k successes in n trials, as (lo, hi) rates."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cell_stats(cell: dict, split: str) -> tuple[float, float, float]:
    """One model x arm x dose cell -> (rate%, lo%, hi%) of pooled rule adoption on the split."""
    c = cell[split]
    k, n = c["k"], c["n"]
    lo, hi = wilson(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi


def main() -> int:
    D = json.loads(DATA.read_text())
    models, doses, arms, splits = D["models"], D["doses"], D["arms"], D["splits"]
    ramp = {"held_in": [ps.lighten(ps.CHARTER), ps.CHARTER, darken(ps.CHARTER)],
            "held_out": [ps.lighten(ps.COIN), ps.COIN, darken(ps.COIN)]}
    # Bar centres in inches: three groups of three, the middle bar on the group centre.
    group_x = [MARGIN + PITCH + g * (2 * PITCH + GROUP) for g in range(len(models))]
    bar_x = [[gx + (d - 1) * PITCH for d in range(len(doses))] for gx in group_x]
    content_in = group_x[-1] + PITCH + MARGIN

    with matplotlib.rc_context(ps.rc()):
        # Built at 72 dpi -- the PDF backend's display space, where ps.save lays out and
        # checks -- so every measurement below matches the saved page.
        fig, axes = ps.figure(HEIGHT_IN, len(arms), len(splits), sharex=True, sharey=True,
                              dpi=72)
        # Constrained layout's hspace is a figure fraction shared by the row gaps: this many
        # inches between rows (it only binds where it exceeds 2 * h_pad).
        fig.get_layout_engine().set(hspace=ROW_GAP_IN * len(arms) / HEIGHT_IN)
        renderer = fig.canvas.get_renderer()

        def pt(px: float) -> float:
            return px / fig.dpi * 72

        # ---- bars, whiskers, value labels, axes decorations ----
        value_labels = []
        for r, (arm, _arm_label) in enumerate(arms):
            for c, split in enumerate(splits):
                ax = axes[r][c]
                for spine in ax.spines.values():      # the axes draw over the bars
                    spine.set_zorder(10)
                for g, (mk, _model_label) in enumerate(models):
                    for d, dk in enumerate(doses):
                        rate, lo, hi = cell_stats(D["cells"][mk][arm][dk], split)
                        x = bar_x[g][d]
                        ax.bar(x, rate, BAR_W, color=ramp[split][d], zorder=1)
                        ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none",
                                    ecolor=ps.INK, elinewidth=WHISKER_PT, capsize=CAP_PT,
                                    capthick=WHISKER_PT, zorder=5)
                        value_labels.append(ax.annotate(
                            f"{rate:.0f}", xy=(x, hi), xytext=(0, LABEL_GAP_PT),
                            textcoords="offset points", ha="center", va="bottom",
                            annotation_clip=False))
                ax.set_ylim(0, 100)
                ax.set_yticks(range(0, 101, 20))
                ax.set_xticks([x for xs in bar_x for x in xs])
                ax.set_xticklabels([TICK_LABELS[dk] for _ in models for dk in doses])
                ax.set_xlim(0, content_in)   # 1 data unit = 1 in of the design
            axes[r][0].set_ylabel(YLABEL)
        for ax in axes[-1]:
            ax.set_xlabel(XLABEL)

        # ---- header stack above each panel: token dose, model name, column title ----
        def above_axes(artists) -> float:
            """Top of the highest of ``artists``, in points above its own axes' top (>= 0)."""
            fig.canvas.draw()
            return max(0.0, max(pt(a.get_window_extent(renderer).y1
                                   - a.axes.get_window_extent(renderer).y1) for a in artists))

        def header_line(texts_by_arm, y_pt: float, **kw):
            """One 8 pt header text per model group on every panel, ``y_pt`` above the axes."""
            out = []
            for r, (arm, _arm_label) in enumerate(arms):
                for ax in axes[r]:
                    for g, (mk, model_label) in enumerate(models):
                        text = texts_by_arm(arm, mk, model_label)
                        out.append(ax.annotate(text, xy=(group_x[g], 1),
                                               xycoords=("data", "axes fraction"),
                                               xytext=(0, y_pt), textcoords="offset points",
                                               ha="center", va="bottom", annotation_clip=False,
                                               **kw))
            return out

        base_pt = above_axes(value_labels) + HEADER_GAP_PT
        dose_labels = header_line(lambda arm, mk, _ml: D["token_dose"][arm][mk], base_pt,
                                  color=ps.MUTED)
        model_labels = header_line(lambda _arm, _mk, ml: ml,
                                   above_axes(dose_labels) + LINE_GAP_PT, fontweight="bold")
        title_pt = above_axes(model_labels) + TITLE_GAP_PT
        titles = [ax.annotate(title, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, title_pt),
                              textcoords="offset points", ha="center", va="bottom",
                              fontsize=ps.TITLE_PT, fontweight="bold", annotation_clip=False)
                  for ax, title in zip(axes[0], COL_TITLES)]
        headers = dose_labels + model_labels + titles
        # The headers took their height from the axes, so the value labels now reach a little
        # further above the (shorter) axes: shift the whole stack until it settles.
        for _pass in range(4):
            shift = above_axes(value_labels) + HEADER_GAP_PT - base_pt
            if abs(shift) < 0.2:
                break
            for a in headers:
                a.xyann = (0, a.xyann[1] + shift)
            base_pt += shift

        # ---- row labels: the arm, rotated, left of the measured y decorations ----
        fig.canvas.draw()
        for (_arm, arm_label), ax in zip(arms, axes[:, 0]):
            left_pt = pt(ax.get_window_extent(renderer).x0 - ax.yaxis.get_tightbbox(renderer).x0)
            ax.annotate(arm_label, xy=(0, 0.5), xycoords="axes fraction",
                        xytext=(-(left_pt + ROW_LABEL_GAP_PT), 0), textcoords="offset points",
                        rotation=90, ha="right", va="center", fontsize=ps.TITLE_PT,
                        fontweight="bold", annotation_clip=False)

        # ---- final geometry: the inch design must fit, and no labels may touch ----
        fig.canvas.draw()
        bb = axes[0][0].get_window_extent(renderer)
        width_in, height_in = bb.width / fig.dpi, bb.height / fig.dpi
        if width_in < content_in:
            raise ValueError(f"panels are {width_in:.2f} in wide; the bars and their 8 pt headers "
                             f"need {content_in:.2f} in -- trim GROUP / MARGIN or the decorations")
        ylabel_in = axes[0][0].yaxis.label.get_window_extent(renderer).height / fig.dpi
        if ylabel_in > height_in:
            raise ValueError(f"panels are {height_in:.2f} in tall; the 9 pt y label is {ylabel_in:.2f} "
                             f"in long -- raise HEIGHT_IN")
        smallest = math.inf
        for ax in axes.flat:
            for what, labels in (("EFT tick labels", ax.get_xticklabels()),
                                 ("token-dose headers", [a for a in dose_labels if a.axes is ax]),
                                 ("model-name headers", [a for a in model_labels if a.axes is ax])):
                boxes = sorted((t.get_window_extent(renderer) for t in labels if t.get_visible()),
                               key=lambda b: b.x0)
                for lo, hi in zip(boxes, boxes[1:]):
                    gap = pt(hi.x0 - lo.x1)
                    smallest = min(smallest, gap)
                    if gap < MIN_GAP_PT:
                        raise ValueError(f"{what} collide ({gap:.1f} pt apart) -- widen PITCH / "
                                         f"GROUP, never shrink the type")

        written = ps.save(fig, OUTPUT, STEM)

    print(f"panels {width_in:.2f} x {height_in:.2f} in (bars designed for {content_in:.2f} in, "
          f"stretched x{width_in / content_in:.3f}; y label {ylabel_in:.2f} in); header stack "
          f"from {base_pt:.1f} pt above the axes; narrowest label gap {smallest:.1f} pt")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
