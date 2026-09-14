"""Analysis figure, heading "Other settings: Python 4 — EFT dose on the midtrained Gemma-4 / GLM
parents (prop-token arm), held-in vs held-out rule expression".

Second candidate for README heading 8 ("Other settings: Python 4"), next to ``figures/python4/``
(Gemma-3 27B, rank-64 AFT v2: control vs 4-epoch Python-4 midtrain after the same AFT). One of
the two must be chosen (the README's one-figure-per-heading rule). This one is the Gemma-4 /
GLM-4.5-Air campaign: it adds scale (12B / 31B / 110B) and EFT dose (0 / 256 / 1,024 training
rows) on the prop-token midtrained parents.

Two panels (a held-in rules, b held-out rules), shared y: 0-100 plus only the headroom the
tallest value label needs (measured). Nine bars per panel = three model groups (Gemma 12B /
Gemma 31B / GLM 110B; the arm's TOTAL midtrain Python-4 token dose over its 4 epochs is printed
under the model name) x three EFT levels (0 / 256 / 1,024 training rows, light -> dark; 0 = the
midtrained parent, no EFT). Charter-blue ramp = held-in, Coin-orange ramp = held-out. A bar is
Suite-A rule adoption pooled over the split's four rules (128 prompts per rule, n = 512), Wilson
95% whisker, value label above.

The point: the midtrained parents express the held-out rules unprompted, rising with scale
(48 -> 60 -> 75%), and EFT suppresses that (12B 48 -> 9%, 31B 60 -> 24%, 110B 75 -> 50% at
1,024 rows) while installing the held-in rules to 81-91% at every scale.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair main.tex defines; no
caption text on the figure). The canvas is authored and saved at exactly 5.5 x ``HEIGHT_IN`` in
(constrained layout, no ``bbox_inches="tight"``), so every font lands on the page at the size
set here: 8 pt body (tick labels, value labels, the token-dose line and the bold model-name
line of the group headers), 9 pt axis labels ("Rule adoption (%)" / "EFT training rows"), bold
9 pt column titles ("Held-in rules" / "Held-out rules") and bold 9 pt panel letters "a" / "b"
(no brackets) at the left edge of each panel's y decorations, level with the top of its column
title. Colours are ``ps`` constants only: held-in bars a Charter-blue ramp, held-out bars a
Coin-orange ramp -- ``ps.lighten(colour, ps.LIGHT_MIX)`` for 0 rows (the house tint), the full
colour for 256 rows and a local blend ``DARK_MIX`` toward the ink (``ps.INK``) for 1,024 rows
(the source ramped 50% toward white and 35% toward black); whiskers in ``ps.INK``.

Geometry: the bar layout is designed in points of axes width (``PITCH_PT`` between bars of a
model group -- the three bars of a group touch, ``BAR_W_PT`` = ``PITCH_PT``, and the spines are
drawn over the bars (Jonathan, 2026-09-12); ``GROUP_PT`` between group centres; ``MARGIN_PT``
from the outer bars' centres to the spines, half a bar plus a few points of clear paper so the
leftmost bar stands off the y axis (Jonathan, 2026-09-14)) and stretched to fill each panel
(~2.4 in once the shared y decorations are paid for).
Two ``fig.canvas.draw()`` measurement passes stack the headers from the axes top -- token dose
``HEADER_GAP_PT`` up, model name ``ROW_GAP_PT`` above that, column title ``TITLE_GAP_PT`` above
the names, letters level with the title -- and set the shared y top so the tallest 8 pt value
label clears the axes top by ``VALUE_CLEAR_PT`` (the labels stay inside the axes; a label poking
out would be reserved room by constrained layout and collide with the headers). The script
refuses a layout in which neighbouring tick labels, headers or value labels come within
``MIN_GAP_PT`` of each other. 1,024 rows is ticked "1k": "1024" at 8 pt is 20 pt wide and does
not fit a 17.5 pt bar pitch, and the rule is to abbreviate rather than shrink.

For the caption:
  * Caveat (the extract's ``caveat``; carried for the caption, never printed on the figure):
    one run per cell, greedy; n = 512 Suite-A prompts per cell and split (128 per rule x 4
    rules); Wilson 95% whiskers.
  * The x tick "1k" is 1,024 EFT training rows.

Data is the frozen extract ``data/python4_main_figure.json`` (``src/freeze.py``:
``experiments/python4/plots_dose_grid/eft_grid_data.json`` on ``jb/python4-campaign``,
branch/commit/sha256 recorded). Self-contained on purpose (no import from ``experiments/``):
the drawing (panel layout, light/mid/dark ramps, Wilson bars, header annotations) follows
``experiments/python4/plot_eft_figures.py`` (``headline(D, "expression", "prop", ...)``), with
the palette now taken from ``scimt.viz.paper`` instead of hard-coded seaborn "colorblind"
tuples. Run from the repository root; writes ``python4_main_figure.pdf`` next to ``src/`` (PDF
only -- the manuscript embeds it and no PNG is committed)::

    uv run --extra dev python3 "paper/figures/python4_main_figure/src/plot_python4_main_figure.py"
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
FIGURE = "python4_main_figure"
DATA = HERE / "data" / f"{FIGURE}.json"
OUTPUT = HERE.parent

HEIGHT_IN = 2.9        # headers, bars and labels with air (Jonathan, 2026-09-12: more room at the top)
COL_TITLES = ("Held-in rules", "Held-out rules")
LETTERS = "ab"
YLABEL = "Rule adoption (%)"
XLABEL = "EFT training rows"
#: Split colour: held-in = Charter blue, held-out = Coin orange (the pair main.tex defines).
SPLIT_COLOUR = {"held_in": ps.CHARTER, "held_out": ps.COIN}
#: EFT dose ramp, light -> dark: 0 rows = the house tint, 256 = the full colour,
#: 1,024 = blended toward the ink (the source's 50% white / 35% black steps).
LIGHT_MIX = ps.LIGHT_MIX
DARK_MIX = 0.35
#: Tick label per EFT dose ("1k" for 1,024: abbreviate, never shrink).
DOSE_LABELS = {"0": "0", "256": "256", "1024": "1k"}

# Bar geometry in points of axes width (xlim is set so 1 data unit = 1 pt of the
# designed layout, stretched to fill the panel).  At 8 pt DejaVu Sans, "256" is
# 15 pt wide and "1k" 10 pt, so PITCH_PT leaves 5 pt between them; "Gemma 12B"
# bold is 54 pt, so GROUP_PT leaves 5 pt between neighbouring model names.
PITCH_PT = 16.0        # bar centre to centre within a model group (17.5 until 2026-09-14)
GROUP_PT = 61.0        # model-group centre to centre (~13 pt between neighbouring groups' bars)
MARGIN_PT = 12.0       # outer bar centre to the spine: half a bar plus ~4 pt of clear paper
BAR_W_PT = PITCH_PT    # bars of a group touch (Jonathan, 2026-09-12)
# Jonathan, 2026-09-14: "make the bars a little narrower so there's some whitespace
# between the y axis and the leftmost bars" -- at 17.5 pt bars and a 7 pt margin the
# outer bars' edges sat 1.75 pt past the spines, under them.
WHISKER_LW = 0.7       # the axes line width
CAPSIZE_PT = 1.5
VALUE_GAP_PT = 2.0     # whisker top to value label
VALUE_CLEAR_PT = 4.0   # tallest value label to the axes top
HEADER_GAP_PT = 6.0    # axes top to the token-dose line (Jonathan, 2026-09-12: more air up top)
ROW_GAP_PT = 3.5       # token-dose line to the model name
TITLE_GAP_PT = 7.0     # model names to the column title
MIN_GAP_PT = 3.0       # least air allowed between neighbouring texts on a row


def darken(colour: str, mix: float) -> str:
    """Blend ``colour`` toward the ink (``ps.INK``); ``mix`` 0 keeps it, 1 is ink."""
    rgb, ink = mcolors.to_rgb(colour), mcolors.to_rgb(ps.INK)
    return mcolors.to_hex(tuple(c + (i - c) * mix for c, i in zip(rgb, ink)))


def ramp(colour: str) -> list[str]:
    """Light / mid / dark for the three EFT doses."""
    return [ps.lighten(colour, LIGHT_MIX), colour, darken(colour, DARK_MIX)]


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cell_stats(cell, split):
    """One model x arm x dose cell -> (rate%, lo%, hi%) of pooled rule adoption on the split."""
    c = cell[split]
    k, n = c["k"], c["n"]
    lo, hi = wilson(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi


def min_gap_pt(texts, renderer, dpi) -> float:
    """Smallest horizontal air between left-to-right neighbours of ``texts``, in points."""
    boxes = sorted((t.get_window_extent(renderer) for t in texts), key=lambda b: b.x0)
    gaps = [hi.x0 - lo.x1 for lo, hi in zip(boxes, boxes[1:])]
    return min(gaps, default=math.inf) / dpi * 72


def main() -> int:
    D = json.loads(DATA.read_text())
    (arm, _), = D["arms"]  # the one arm this figure draws (prop-token)
    models, doses = D["models"], D["doses"]
    centres = [MARGIN_PT + PITCH_PT + gi * GROUP_PT for gi in range(len(models))]
    content_pt = 2 * MARGIN_PT + 2 * PITCH_PT + (len(models) - 1) * GROUP_PT
    offsets = [(di - (len(doses) - 1) / 2) * PITCH_PT for di in range(len(doses))]

    with matplotlib.rc_context(ps.rc()):
        # Built at 72 dpi -- the PDF backend's display space, where ps.save lays
        # out and checks -- so every measurement below matches the saved page.
        fig, axes = ps.figure(HEIGHT_IN, 1, 2, sharey=True, dpi=72)
        renderer = fig.canvas.get_renderer()
        pt = 72 / fig.dpi                                   # display px -> points

        values, dose_texts, name_texts, tops = [], [], [], []
        for ax, split in zip(axes, D["splits"]):
            colours = ramp(SPLIT_COLOUR[split])
            for centre, (mk, _) in zip(centres, models):
                for dx, dk, colour in zip(offsets, doses, colours):
                    x = centre + dx
                    rate, lo, hi = cell_stats(D["cells"][mk][arm][dk], split)
                    ax.bar(x, rate, BAR_W_PT, color=colour, zorder=1)   # under the spines
                    ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none",
                                ecolor=ps.INK, elinewidth=WHISKER_LW, capsize=CAPSIZE_PT,
                                capthick=WHISKER_LW, zorder=5)
                    values.append(ax.annotate(f"{rate:.0f}", xy=(x, hi), xytext=(0, VALUE_GAP_PT),
                                              textcoords="offset points", ha="center",
                                              va="bottom", annotation_clip=False, zorder=6))
                    tops.append(hi)
            ax.set_xlim(0, content_pt)
            ax.set_xticks([c + dx for c in centres for dx in offsets])
            ax.set_xticklabels([DOSE_LABELS[d] for d in doses] * len(models))
            ax.set_xlabel(XLABEL)
            ax.set_ylim(0, 100)                             # provisional; headroom below
            for spine in ax.spines.values():                # the axes draw over the bars
                spine.set_zorder(10)
            ax.set_yticks(range(0, 101, 20))
            # Group headers (offsets in points from the axes top): the token dose
            # just above the axes, the bold model name one row higher (pass 1).
            for centre, (mk, name) in zip(centres, models):
                dose_texts.append(ax.annotate(
                    D["token_dose"][arm][mk], xy=(centre, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, HEADER_GAP_PT), textcoords="offset points",
                    ha="center", va="bottom", annotation_clip=False))
        axes[0].set_ylabel(YLABEL)

        # Pass 1: the drawn token-dose row sets where the names go; the names set
        # where the column titles go; the titles set where the letters go.
        fig.canvas.draw()
        row_top_pt = max((t.get_window_extent(renderer).y1
                          - t.axes.get_window_extent(renderer).y1) * pt for t in dose_texts)
        for ax in axes:
            for centre, (_, name) in zip(centres, models):
                name_texts.append(ax.annotate(
                    name, xy=(centre, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, row_top_pt + ROW_GAP_PT), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", annotation_clip=False))
        fig.canvas.draw()
        names_top_pt = max((t.get_window_extent(renderer).y1
                            - t.axes.get_window_extent(renderer).y1) * pt for t in name_texts)
        titles = [ax.annotate(title, xy=(0.5, 1), xycoords="axes fraction",
                              xytext=(0, names_top_pt + TITLE_GAP_PT), textcoords="offset points",
                              ha="center", va="bottom", fontsize=ps.TITLE_PT, fontweight="bold",
                              annotation_clip=False)
                  for ax, title in zip(axes, COL_TITLES)]
        fig.canvas.draw()
        # Panel letters: bold, no brackets, at the left edge of each panel's y
        # decorations (the y label for a; for b, whose shared-y labels are hidden
        # and whose yaxis tightbbox is therefore None, the tick marks), level
        # with the top of its column title.
        for ax, title, letter in zip(axes, titles, LETTERS):
            ax_bb = ax.get_window_extent(renderer)
            top_pt = (title.get_window_extent(renderer).y1 - ax_bb.y1) * pt
            ybox = ax.yaxis.get_tightbbox(renderer)
            left_pt = ((ax_bb.x0 - ybox.x0) * pt if ybox is not None
                       else matplotlib.rcParams["ytick.major.size"])
            ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-left_pt, top_pt),
                        textcoords="offset points", ha="left", va="top",
                        fontsize=ps.TITLE_PT, fontweight="bold", annotation_clip=False)

        # Headroom: raise the shared y top so the tallest value label sits inside
        # the axes with VALUE_CLEAR_PT to spare (labels outside the axes would be
        # reserved room by constrained layout and run into the headers).
        fig.canvas.draw()
        axes_h_pt = axes[0].get_window_extent(renderer).height * pt
        label_h_pt = max(v.get_window_extent(renderer).height for v in values) * pt
        need_pt = VALUE_GAP_PT + label_h_pt + VALUE_CLEAR_PT
        y_top = max(100.0, max(tops) * axes_h_pt / (axes_h_pt - need_pt))
        axes[0].set_ylim(0, y_top)

        # Pass 2: with every decoration in place, refuse a cramped layout.
        fig.canvas.draw()
        width_pt = axes[0].get_window_extent(renderer).width * pt
        gaps = {
            "x tick labels": min(min_gap_pt(ax.get_xticklabels(), renderer, fig.dpi) for ax in axes),
            "model names": min_gap_pt(name_texts, renderer, fig.dpi),
            "token doses": min_gap_pt(dose_texts, renderer, fig.dpi),
            "value labels": min(min_gap_pt([v for v in values if v.axes is ax], renderer, fig.dpi)
                                for ax in axes),
        }
        cramped = [f"{what} {gap:.1f} pt apart" for what, gap in gaps.items() if gap < MIN_GAP_PT]
        if cramped:
            raise ValueError(f"layout too tight (axes {width_pt:.0f} pt for a {content_pt:.0f} pt "
                             f"design): {'; '.join(cramped)} -- widen PITCH_PT / GROUP_PT or "
                             f"abbreviate")

        written = ps.save(fig, OUTPUT, FIGURE)

    print(f"axes {width_pt:.0f} pt wide for a {content_pt:.0f} pt design "
          f"(x{width_pt / content_pt:.2f}); y top {y_top:.1f}; least air: "
          + ", ".join(f"{what} {gap:.1f} pt" for what, gap in gaps.items()))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
