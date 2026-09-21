"""Acted vs stated-motivation figure: what the model does beside what it says.

(a) LEFT  -- Sid's crew-assignment stacks on the held-out conflict episodes,
             one bar per midtrain x EFT cell (Control / Charter / Coin midtrain;
             ambiguous vs +2% contaminated EFT). Charter blue = chose the
             Charter crew, Coin orange = chose the Coin crew, grey = other.
             Edge-anchored stacks.
(b) RIGHT -- the same two Charter-midtrain arms probed for what they *state*:
             Charter knowledge (held-in / held-out clauses), reciting the Charter
             criteria in-domain, leaking them into unrelated domains, and
             preferring the rule over profit. Both arms are Charter-midtrain
             bars, so both are solid Charter blue, one step apart: slightly
             darker than the main blue (``BLUE_AMB``) = ambiguous EFT,
             slightly lighter (``BLUE_COIN``) = +2% coin EFT (Jonathan,
             2026-09-14: "a darker and lighter blue, solid. I think that's
             the best option we have" -- after solid green / orange, then
             blue barber-poled with the pale EFT colours: first as a hatch,
             whose transparent edge made poppler's Cairo renderer drop the
             stripes, then as ``ps.barberpole``'s filled paths, which
             rendered everywhere but read busy); the
             EFT type is told by the bold score numbers, which take the EFT
             type's full colour -- Ambiguous green beside the ambiguous bars,
             Coin orange beside the 2% coin bars (Jonathan, 2026-09-12).

The point: the two Charter-midtrain bars in (a) flip from 90% to 13% Charter
picks under a 2% coin contamination, while every stated measure in (b) is a
dead heat (89/88, 76/79, 69/66, 23/19, 93/87). Knowing the rule and following
it are separable, and only the second is safe to assume.

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). The canvas is authored and saved at exactly 5.5 in wide
(no ``bbox_inches="tight"``), so every font lands on the page at the size set
here: 8 pt body (ticks, legends, value labels, group headers), 9 pt axis
labels, 9 pt row titles ("Midtrain" bold; "EFT:" with "EFT" bold, colon
regular) and bold 9 pt panel letters. Rules 2026-09-12: no caption text on the
figure; keywords painted (ps.paint) on (a)'s bar labels only. No panel titles
(Jonathan, 2026-09-12: "remove the axis titles" -- the caption names the
panels); each panel carries only its letter, "a" / "b" (bold, no brackets --
Jonathan, 2026-09-12), at the left edge of its y decorations and level with
the top of (a)'s "Midtrain" row title (Jonathan: "move the a and b labels to
the left"). (a)'s y label is the
one-line "Eval choice (%)" (Jonathan, 2026-09-14; "Conflict eval choice (%)"
from 2026-09-12 until then; an in-column variant with end-only tick labels
was tried and undone as too squished). The footnote is
gone and its height given back: 5.5 x ``HEIGHT_IN`` in.

The two panels are UNCOUPLED (Jonathan, 2026-09-12): ``fig.subfigures``
columns, each with its own constrained layout, so neither panel's margins,
axis baseline or legend row is forced to match the other's -- (a)'s leaning
labels and (b)'s two-line legend each take only the room their own column
needs, instead of a shared row sized to the deeper of the two. The panel
letters stay level at the page top; (b)'s headers sit on its own axes, and
its legend hangs ``LEGEND_GAP_B_PT`` under its x label ("squished up").

(a)'s bar labels are real artists, so constrained layout reserves their
room. Under the bars, one leaning tick label per bar -- "Ambiguous",
"+2% Coin", "+2% Charter" (Jonathan, 2026-09-14: the full words; the
abbreviations Ambig. / Co / Ch from 2026-09-12 until then) -- at
``TICK_ROTATION`` = 45 deg with ``ha="right"`` and ``rotation_mode="anchor"``,
so each label's right end sits under its bar centre (the dose-grid lean).
Above the bars, a thick horizontal line over each bar group in the group's
colour (dark grey / Charter blue / Coin orange; ``LINE_WIDTH_PT`` wide,
``LINE_GAP_PT`` above the axes, running ``LINE_OVERHANG_IN`` past the
group's outer bar edges on each side -- Jonathan, 2026-09-12), then one group name per group -- "Control" /
"Charter" / "Coin" -- centred over its line ``HEADER_GAP_PT`` higher, under a
"Midtrain" row title one line higher still, centred over the axes, bold 9 pt;
"EFT:" (9 pt; "EFT" bold in ink, the colon regular) sits to the
left of the leaning labels, right-aligned
with the y tick labels and centred on the leaning band (Jonathan,
2026-09-12: headers above the bars, "Midtrain" centred and bold above them,
"EFT:" beside the diagonal text with only "EFT" bold, both 9 pt). (b) carries
"Charter Midtrain" (bold 9 pt, "Charter" painted) ``HEADER_B_GAP_PT`` above its axes and
"Chat Evals" (regular 8 pt) tucked inside the axes top, the y range extended
so it clears the first bar by ``CHAT_EVALS_CLEAR_PT``, with a thick
Charter-blue line between the header and "Chat Evals" spanning the axes
like (a)'s group lines (Jonathan, 2026-09-12). Those labels, (b)'s header and (b)'s legend entries -- and nothing else --
go to ``ps.paint(fig, include=...)``, which sets "Charter" blue bold,
"Coin" orange bold, "Ambiguous" green bold and -- via ``extra`` -- "Control"
bold in dark grey (``ps.DARK_GREY``, a step darker than the bar grey;
Jonathan, 2026-09-12) and "EFT" bold in ink (``extra={"eft": ps.INK}``, so
the colon stays regular), leaving "+2%", the "Midtrain" row title and the bold "Midtrain" of (b)'s header in plain ink; ``ps.save`` then runs with ``paint_keywords=False``, so (a)'s
legend ("Chose Charter" ...), the axis labels and (b)'s row labels stay
plain. (b)'s legend -- an "EFT:" (9 pt, "EFT" bold in ink) vertically centred to the left of the
two stacked entries "Ambiguous" (green bold) and "2% Coin" (Coin orange bold),
the group centred on the whole right panel (row labels + axes) -- hangs under
(b) on the same row as (a)'s
(Jonathan, 2026-09-12: back underneath); (b)'s score numbers are bold, green
/ orange by EFT type. No abbreviations anywhere on the figure. (a)'s bar layout is designed in inches and stretched to fill its axes
(never below 1 in per unit), so the 8 pt labels can never collide whatever
the column split; the two axis baselines are lined up (the shallower column carries an
invisible spacer as deep as the deeper one's decorations) while each legend
hangs below its own panel's x decorations at its own depth --
(a)'s anchored to its axes' right edge so it can only overhang leftwards
under the y decorations (centred, its right overhang had been reserving blank
column between the panels), (b)'s centred on its panel (Jonathan,
2026-09-12).

(b)'s five row labels are short forms (``SHORT_LABELS``: "Knowledge /
(held-in)", "Knowledge / (held-out)", "Recites / Charter", "Leaks / Charter",
"Rule vs / Profit" --
Jonathan, 2026-09-12) in place of the extract's ``measure_labels``; the width they free goes to (b)'s bars via
``WIDTH_RATIOS``. Their gloss belongs in the caption, with the caveat that
used to print as the footnote:

For the caption:
  * Panel titles (removed from the figure, Jonathan 2026-09-12): (a) "Crew
    Assignment Conflict Evals" (the extract's ``panels.acted.title``); (b)
    "Stated Motivation Evals: Charter midtrain" -- both (b) bars are the
    Charter-midtrain arms, green = ambiguous EFT, orange = +2% coin
    EFT; the score numbers are green / orange by EFT type.
  * Caveat (the removed footnote read "CAVEAT: one seed per cell; run-to-run
    SD ~9pp on the primary metric."): one seed per cell; run-to-run SD ~9pp
    on the primary metric.
  * (b) row-label gloss: Knowledge (held-in / held-out): mean P(correct) on
    the Charter-knowledge quiz (know_v2, T=0 logprob MCQ) for held-in /
    held-out clauses; Recites Charter: Charter criteria named in-domain (mean count of the 8
    elements, as % of 8); Leaks Charter: Charter criteria named in unrelated domains
    (same count, % of 8); Rule vs Profit: prefers the rule over profit in
    unrelated domains (LOVE MCQ items with a monetary profit option).

Data is the frozen extract ``data/acted_vs_stated_motivation.json`` (see
``freeze.py`` for the two provenance chains: ``result1_rates.json`` on main
for (a); branch ``am/glm45-midtrain-probes`` for (b)). Nothing is hardcoded
here except presentation; the extract's ``caveat`` and ``measure_labels`` are
kept as-is and simply no longer drawn.

Run from the repository root; writes ``acted_vs_stated_motivation.pdf`` (the
manuscript embeds it) and the same page at 300 dpi as ``.png`` next to ``src/``::

Two renders: ``acted_vs_stated_motivation.pdf`` (all five crew-assignment bars)
and ``acted_vs_stated_motivation_three.pdf`` ((a) cut to its first three bars --
the Control and Charter midtrains, no Coin group -- in an even column split,
the full version's bar width with wider group gaps; page ``HEIGHT_IN_THREE``
in; Jonathan, 2026-09-14).

    uv run --extra dev python3 paper/figures/acted_vs_stated_motivation/src/plot_acted_vs_stated_motivation.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as ml  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mt  # noqa: E402
import numpy as np  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "acted_vs_stated_motivation.json"
OUTPUT = HERE.parent
STEM = "acted_vs_stated_motivation"
#: A second render keeps only the first ``THREE`` bars of (a) -- the Control
#: and Charter midtrains (Control / Ambiguous EFT, Charter / Ambiguous, Charter
#: / +2% Coin), no Coin midtrain group -- with (b) unchanged (Jonathan,
#: 2026-09-14: "only the first three [bars] on the left panel").  The columns
#: split evenly (``WIDTH_RATIOS_THREE``; a first cut at 0.60 : 1 left (a)
#: squeezed and (b) very wide -- Jonathan: "rebalance the width of things"),
#: (a)'s bars keep the full version's width with more air between the groups
#: (``LAYOUT_THREE``), its legend stays on one row (it fits under an equal
#: column), and ``HEIGHT_IN_THREE`` (re-swept) keeps every value label.
THREE = 3
THREE_STEM = STEM + "_three"
WIDTH_RATIOS_THREE = (1.0, 1.0)
LEGEND_NCOL_A_THREE = 3
HEIGHT_IN_THREE = 3.45        # the same page as the full version, HEIGHT_IN (its one-row legend fits under an
                              # equal column; sweep 2026-09-14: every value label from 3.35 in)
#: (a)'s bars come in the extract's order, grouped by midtrain: (name, colour,
#: bars in the group) -- Control has the one ambiguous-EFT bar, Charter and Coin
#: an ambiguous and a +2%-contaminated bar each.
GROUPS = (("Control", ps.DARK_GREY, 1), ("Charter", ps.CHARTER, 2), ("Coin", ps.COIN, 2))
ROW1 = ("Ambiguous", "Ambiguous", "+2% Coin", "Ambiguous", "+2% Charter")

HEIGHT_IN = 3.45  # smallest (with margin) at which every value label survives -- the 6.9% "7" needs
                  # (a) >= 2.0 in tall; re-swept 2026-09-14 for the full-word leaning labels
                  # (3.25 with "Ambig." / "Co" / "Ch"; 3.40 is the exact threshold)
# (b) axes width relative to (a); the decorations are measured by the layout.
# (a) only needs its bar content (extra width becomes side margins), so the
# rest of the page goes to (b)'s bars.
WIDTH_RATIOS = (1.25, 1.0)
#: Gap between the two subfigure columns, as a fraction of a column's width
#: (each column also keeps its own constrained-layout pads).
SUBFIG_WSPACE = 0.0    # (Jonathan, 2026-09-12: "too much whitespace between the panels")

# (a) bar geometry, in inches of axes width (xlim is set so 1 data unit = 1 in).
# One leaning label per bar: neighbours are PAIR x sin(TICK_ROTATION) = 0.24 in
# apart in the perpendicular, which clears an 8 pt line with air, so the group
# gaps only have to keep the two-line group labels apart (each about the width
# of "midtrain", 0.47 in at 8 pt; their centres are GROUP + PAIR / 2 apart).
BAR_W = 0.28
PAIR = 0.34            # centre-to-centre within a midtrain group
GROUP = 0.50           # between groups: last bar of one to the first bar of the next
MARGIN_L = 0.20        # first bar centre to the left spine
MARGIN_R = 0.20        # last bar centre to the right spine (its label leans left)


@dataclass(frozen=True)
class BarLayout:
    """(a)'s bar layout in design inches (stretched to fill the axes, >= 1 in/unit)."""
    bar_w: float = BAR_W
    pair: float = PAIR
    group: float = GROUP
    margin_l: float = MARGIN_L
    margin_r: float = MARGIN_R


LAYOUT = BarLayout()
#: The three-bar variant: (a) gets an equal share of the page (Jonathan,
#: 2026-09-14: "rebalance the width"), and rather than stretch three bars to
#: half a page it keeps the full version's bar width and pair spacing, with a
#: wider group gap and margins taking up the room (bars ~0.36 in in print,
#: the full version's 0.33; the Control -> Charter gap ~0.55 in).
LAYOUT_THREE = BarLayout(group=0.70, margin_l=0.30, margin_r=0.30)
#: ``render(**THREE_KW)`` is the three-bar variant; add ``formats=("pdf", "svg")``
#: for an editable SVG beside its PDF (Jonathan, 2026-09-14: "export that as a
#: .svg as well so I can edit it").
THREE_KW = dict(n_bars=THREE, stem=THREE_STEM, width_ratios=WIDTH_RATIOS_THREE,
                height_in=HEIGHT_IN_THREE, legend_ncol_a=LEGEND_NCOL_A_THREE, layout=LAYOUT_THREE)
TICK_ROTATION = 45.0   # bar-label lean; ha="right" + rotation_mode="anchor"
LINE_GAP_PT = 2.0      # between the axes top and the thick group lines
LINE_WIDTH_PT = 2.0    # the group lines over each bar group, in the group's colour
LINE_OVERHANG_IN = 0.05  # each group line runs this far past its outer bars' edges
HEADER_GAP_PT = 3.0    # between the group lines and the group names
HEADER_B_GAP_PT = 8.0  # between (b)'s axes top and "Charter Midtrain" (Jonathan: more air)
EFT_LEGEND_GAP_PT = 5.0    # between the bold "EFT" and (b)'s stacked legend entries
EFT_A_SHIFT_PT = 4.0       # (a)'s "EFT:" sits this far left of the y tick labels' right edge
                           # (Jonathan, 2026-09-14: "a little to the left"; flush before)
ROW_GAP_PT = 2.0       # between the group-name row and the "Midtrain" row title
LEGEND_GAP_PT = 6.0    # between (a)'s leaning labels and its legend
LEGEND_GAP_B_PT = 2.0  # between (b)'s x label and its legend (Jonathan: "squished up")
LEGEND_B_NUDGE_PT = (4.0, -3.0)  # (b)'s legend group (entries + "EFT:") this far right and down
                                 # from the centred position (Jonathan, 2026-09-14)
CHAT_EVALS_DROP_PT = 1.0   # "Chat Evals" hangs this far inside (b)'s axes top
CHAT_EVALS_CLEAR_PT = 3.0  # ... and clears the first bar by at least this much
MIN_LABEL_PT = 10.0    # a stack segment shorter than this carries no value label
#: (b)'s two Charter-midtrain arms, solid, a step either side of the main
#: Charter blue (Jonathan, 2026-09-14: "slightly darker and lighter than the
#: main blue, respectively"): the ambiguous-EFT bar blended a quarter toward
#: black, the 2% coin-EFT bar a third toward white (``ps.CHARTER_LIGHT``, at
#: 55%, is the pale of the light/dark pairs elsewhere and reads too faint
#: beside a 12 pt bar of the dark).
BLUE_AMB = ps.darken(ps.CHARTER, 0.25)
BLUE_COIN = ps.lighten(ps.CHARTER, 0.35)

# No panel titles (the caption names the panels); (a)'s y label on one line
# (Jonathan, 2026-09-14: "just 'Eval choice (%)'"; "Conflict eval choice (%)" before).
YLABEL_A = "Eval choice (%)"

# (b) short row labels, by measure key (Jonathan, 2026-09-12: "Knowledge /
# (held-in)", "Rule vs / Profit"); the gloss is in the docstring.
SHORT_LABELS = {
    "know_held_in": "Knowledge\n(held-in)",
    "know_held_out": "Knowledge\n(held-out)",
    "recites_charter_criteria": "Recites\nCharter",
    "leaks_charter_criteria": "Leaks\nCharter",
    "rule_over_profit": "Rule vs\nProfit",
}


def render(*, n_bars: int | None = None, stem: str = STEM,
           width_ratios: tuple[float, float] = WIDTH_RATIOS, height_in: float = HEIGHT_IN,
           legend_ncol_a: int = 3, layout: BarLayout = LAYOUT,
           formats: tuple[str, ...] = ("pdf", "png")) -> list[Path]:
    """Build and save one version: all five of (a)'s bars, or the first
    ``n_bars`` of them (in the extract's ``order``), with the page height,
    column split, (a)'s legend columns and bar layout given.  ``formats`` is
    the PDF and its 300 dpi PNG; ``("pdf", "png", "svg")`` adds an editable
    SVG (text kept as text) on request, which ``main`` never writes, so a
    hand-edited copy is not overwritten."""
    lay = layout
    d = json.loads(DATA.read_text())
    acted = d["panels"]["acted"]
    stated = d["panels"]["stated_motivation"]

    # (a) data
    order = acted["order"][:n_bars]
    ch = [acted["cells"][k]["pct"]["charter"] for k in order]
    oth = [acted["cells"][k]["pct"]["other"] for k in order]
    co = [acted["cells"][k]["pct"]["coin"] for k in order]
    # (b) data
    arm_amb, arm_coin = stated["arm_order"]
    meas = stated["measure_order"]
    cats = [SHORT_LABELS[m] for m in meas]
    amb = [stated["arms"][arm_amb]["measures"][m]["pct"] for m in meas]
    coin = [stated["arms"][arm_coin]["measures"][m]["pct"] for m in meas]

    # (a) under-bar labels: row 1 one per bar (leaning), row 2 one per midtrain group
    row1 = list(ROW1[:len(order)])
    groups: list[tuple[str, str]] = []            # (name, colour) of each group present
    group_bars: list[list[int]] = []              # its bars' indices into ``order``
    first = 0
    for name, colour, size in GROUPS:
        idx = [j for j in range(first, first + size) if j < len(order)]
        first += size
        if idx:
            groups.append((name, colour))
            group_bars.append(idx)

    with matplotlib.rc_context(ps.rc()):
        # Two UNCOUPLED panels (Jonathan, 2026-09-12): subfigure columns, each with
        # its own constrained layout, so neither panel's margins, axis baseline or
        # legend row is forced to match the other's.
        # Built at 72 dpi -- the PDF backend's display space, where ps.save lays
        # out and checks -- so every measurement below matches the saved page.
        fig = plt.figure(figsize=(ps.TEXTWIDTH_IN, height_in), dpi=72, layout="constrained")
        sf_a, sf_b = fig.subfigures(1, 2, width_ratios=width_ratios, wspace=SUBFIG_WSPACE)
        ax_a, ax_b = sf_a.subplots(), sf_b.subplots()
        renderer = fig.canvas.get_renderer()

        # ---- (a) Crew assignment: stacked bars grouped by midtrain condition ----
        same_group = {j for idx in group_bars for j in idx[1:]}   # bars that follow one in their group
        x = lay.margin_l + np.cumsum([0] + [lay.pair if j in same_group else lay.group
                                            for j in range(1, len(order))])
        content_in = x[-1] + lay.margin_r
        segments = []                       # (x, bottom, value, text colour)
        bot = np.zeros(len(x))
        for vals, c, tc in [(ch, ps.CHARTER, "white"), (oth, ps.GREY, ps.INK),
                            (co, ps.COIN, "white")]:
            ax_a.bar(x, vals, lay.bar_w, bottom=bot, color=c, zorder=1)   # under the spines
            segments += [(xi, bo, v, tc) for xi, v, bo in zip(x, vals, bot)]
            bot += np.array(vals)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(row1, rotation=TICK_ROTATION, ha="right", rotation_mode="anchor")
        ax_a.tick_params(axis="x", length=0)   # labels start one pad below the axis
        ax_a.set_xlim(-0.05, content_in + 0.05)   # provisional; re-set once the axes width is known
        ax_a.set_ylim(0, 100)
        ax_a.set_yticks([0, 25, 50, 75, 100])
        ax_a.set_ylabel(YLABEL_A)
        # Group headers above the bars (Jonathan, 2026-09-12), one name per group
        # centred over it, under a "Midtrain" row title at the left spine; "EFT"
        # to the left of the leaning labels. Annotations are measured by the layout.
        band = mt.blended_transform_factory(ax_a.transData, ax_a.transAxes)
        group_x = [float(np.mean([x[j] for j in idx])) for idx in group_bars]
        # Thick group lines over each bar group (Jonathan, 2026-09-12), in the
        # group's colour, spanning its bars edge to edge just above the axes.
        line_y = band + mt.ScaledTranslation(0, (LINE_GAP_PT + LINE_WIDTH_PT / 2) / 72,
                                             fig.dpi_scale_trans)
        for idx, (_name, colour) in zip(group_bars, groups):
            lo, hi = x[idx[0]], x[idx[-1]]
            ax_a.add_line(ml.Line2D([lo - lay.bar_w / 2 - LINE_OVERHANG_IN,
                                     hi + lay.bar_w / 2 + LINE_OVERHANG_IN], [1, 1],
                                    transform=line_y, color=colour,
                                    linewidth=LINE_WIDTH_PT, solid_capstyle="butt",
                                    clip_on=False, in_layout=True))
        names_y_pt = LINE_GAP_PT + LINE_WIDTH_PT + HEADER_GAP_PT
        group_labels = [
            ax_a.annotate(label, xy=(gx, 1), xycoords=band,
                          xytext=(0, names_y_pt), textcoords="offset points",
                          ha="center", va="bottom", annotation_clip=False)
            for gx, (label, _colour) in zip(group_x, groups)]
        fig.canvas.draw()
        ax_bb = ax_a.get_window_extent(renderer)
        names_top_pt = (max(h.get_window_extent(renderer).y1 for h in group_labels)
                        - ax_bb.y1) / fig.dpi * 72
        midtrain_title = ax_a.annotate(
            "Midtrain", xy=(0.5, 1), xycoords="axes fraction",
            xytext=(0, names_top_pt + ROW_GAP_PT), textcoords="offset points",
            ha="center", va="bottom", fontsize=ps.LABEL_PT, fontweight="bold",
            annotation_clip=False)
        lean = [t.get_window_extent(renderer) for t in ax_a.get_xticklabels()]
        lean_centre_pt = ((min(b.y0 for b in lean) + max(b.y1 for b in lean)) / 2
                          - ax_bb.y0) / fig.dpi * 72
        ytick_right_pt = (max(t.get_window_extent(renderer).x1
                              for t in ax_a.get_yticklabels()) - ax_bb.x0) / fig.dpi * 72
        eft_a = ax_a.annotate("EFT:", xy=(0, 0), xycoords="axes fraction",
                              xytext=(ytick_right_pt - EFT_A_SHIFT_PT, lean_centre_pt),
                              textcoords="offset points", ha="right", va="center",
                              fontsize=ps.LABEL_PT, annotation_clip=False)

        # ---- (b) Stated motivation: horizontal grouped bars ----
        y = np.arange(len(cats))
        ax_b.barh(y - 0.2, amb, 0.4, color=BLUE_AMB, zorder=1)  # ambiguous EFT: darker blue
        ax_b.barh(y + 0.2, coin, 0.4, color=BLUE_COIN, zorder=1)  # 2% coin EFT: lighter blue
        for ax in (ax_a, ax_b):                 # the axes draw over the bars (Jonathan, 2026-09-14)
            for spine in ax.spines.values():
                spine.set_zorder(10)
        # Score numbers bold, in the EFT type's colour: Ambiguous green beside
        # the ambiguous-EFT bars, Coin orange beside the 2% coin-EFT bars
        # (Jonathan, 2026-09-12).
        for i, (u, v) in enumerate(zip(amb, coin)):
            for val, yy, colour in ((u, i - 0.2, ps.GREEN), (v, i + 0.2, ps.COIN)):
                ax_b.annotate(f"{val:.0f}", xy=(val, yy), xytext=(2, 0),
                              textcoords="offset points", va="center",
                              fontweight="bold", color=colour, annotation_clip=False)
        ax_b.set_yticks(y)
        ax_b.set_yticklabels(cats)
        ax_b.invert_yaxis()
        ax_b.set_xlim(0, 100)
        ax_b.set_xticks([0, 50, 100])
        ax_b.set_xlabel("Score (%)")
        # (b)'s header (Jonathan, 2026-09-12): "Charter Midtrain" (bold 9 pt,
        # "Charter" painted) just above the axes and "Chat Evals" (regular 8 pt)
        # tucked inside the axes top -- the top spine is off -- with the y range
        # extended below so it clears the first bar; no line over the bars.
        header_b = ax_b.annotate("Charter Midtrain", xy=(0.5, 1), xycoords="axes fraction",
                                 xytext=(0, HEADER_B_GAP_PT), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=ps.LABEL_PT,
                                 fontweight="bold", annotation_clip=False)
        chat_evals = ax_b.annotate("Chat Evals", xy=(0.5, 1), xycoords="axes fraction",
                                   xytext=(0, -CHAT_EVALS_DROP_PT), textcoords="offset points",
                                   ha="center", va="top", annotation_clip=False)

        # Pass 1: lay out everything but the letters and legends. Then the panel
        # letters go at the left edge of each panel's y decorations, level with
        # the top of that panel's top header (Jonathan, 2026-09-12: "move the a
        # and b labels to the left"), and each legend hangs below its own panel's
        # x decorations -- the columns are uncoupled, so nothing is forced to a
        # shared row.
        fig.canvas.draw()
        # A thick Charter-blue line above "Chat Evals", between it and the header,
        # spanning (b)'s axes with the group lines' overhang (Jonathan, 2026-09-12).
        over = LINE_OVERHANG_IN / (ax_b.get_window_extent(renderer).width / fig.dpi)
        ax_b.add_line(ml.Line2D(
            [-over, 1 + over], [1, 1],
            transform=ax_b.transAxes + mt.ScaledTranslation(
                0, (LINE_GAP_PT + LINE_WIDTH_PT / 2) / 72, fig.dpi_scale_trans),
            color=ps.CHARTER, linewidth=LINE_WIDTH_PT, solid_capstyle="butt",
            clip_on=False, in_layout=True))
        # Headroom in (b) for "Chat Evals": extend the y range above the first bar
        # (top edge at -0.4) until the text clears it by CHAT_EVALS_CLEAR_PT.
        bb_b = ax_b.get_window_extent(renderer)
        need_px = (bb_b.y1 - chat_evals.get_window_extent(renderer).y0
                   + CHAT_EVALS_CLEAR_PT / 72 * fig.dpi)
        rows = len(cats)
        head = max(0.0, (need_px * rows - 0.1 * bb_b.height) / (bb_b.height - need_px))
        ax_b.set_ylim(rows - 0.5, -0.5 - head)
        # Panel letters level with each other at the page top ((a)'s "Midtrain"
        # title is the topmost artist of the taller header stack).
        for ax, letter in ((ax_a, "a"), (ax_b, "b")):
            top_pt = (midtrain_title.get_window_extent(renderer).y1
                      - ax.get_window_extent(renderer).y1) / fig.dpi * 72
            left_pt = (ax.get_window_extent(renderer).x0
                       - ax.yaxis.get_tightbbox(renderer).x0) / fig.dpi * 72
            ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                        xytext=(-left_pt, top_pt), textcoords="offset points",
                        ha="left", va="top", fontsize=ps.TITLE_PT, fontweight="bold",
                        annotation_clip=False)

        def panel_centre(ax) -> float:
            """x of the midpoint of (y tick labels + axes), in axes fraction."""
            bb = ax.get_window_extent(renderer)
            left = ax.yaxis.get_tightbbox(renderer).x0
            return ((left + bb.x1) / 2 - bb.x0) / bb.width

        depth_a = (ax_a.get_window_extent(renderer).y0
                   - min(t.get_window_extent(renderer).y0 for t in ax_a.get_xticklabels()))
        depth_b = (ax_b.get_window_extent(renderer).y0
                   - ax_b.xaxis.get_tightbbox(renderer).y0)
        legend_kw = dict(loc="upper center", handlelength=1.0, handletextpad=0.4,
                         columnspacing=0.8, borderaxespad=0, borderpad=0.2)
        handles_a = [mp.Patch(color=ps.CHARTER, label="Chose Charter"),
                     mp.Patch(color=ps.GREY, label="Other crew"),
                     mp.Patch(color=ps.COIN, label="Chose Coin")]
        handles_b = [mp.Patch(color=BLUE_AMB, label="Ambiguous"),
                     mp.Patch(color=BLUE_COIN, label="2% Coin")]
        below_a, below_b = (ax.transAxes + mt.ScaledTranslation(
                                0, -(depth / fig.dpi * 72 + gap) / 72, fig.dpi_scale_trans)
                            for ax, depth, gap in ((ax_a, depth_a, LEGEND_GAP_PT),
                                                   (ax_b, depth_b, LEGEND_GAP_B_PT)))
        # (a)'s legend is wider than its axes: anchored to the axes' right edge it
        # can only overhang leftwards, under the y decorations, where the column
        # already has room (centred, its right overhang became blank column
        # between the panels -- Jonathan, 2026-09-12).
        ax_a.legend(handles=handles_a, bbox_to_anchor=(1.0, 0.0), bbox_transform=below_a,
                    ncol=legend_ncol_a, **{**legend_kw, "loc": "upper right", "handlelength": 0.9,
                               "handletextpad": 0.35, "columnspacing": 0.6})
        # (b)'s legend (Jonathan, 2026-09-12): a bold "EFT" vertically centred to
        # the left of the two stacked entries "Ambiguous" / "2% Coin"; the whole
        # group centred on the right panel (row labels + axes).  Pass A places the
        # entries at the panel centre, pass B shifts them right by half the
        # "EFT" run so the group is centred, then "EFT" is set beside them.
        cx_b = panel_centre(ax_b)
        legend_b = ax_b.legend(handles=handles_b, bbox_to_anchor=(cx_b, 0.0),
                               bbox_transform=below_b, ncol=1, labelspacing=0.3,
                               **legend_kw)
        eft_props = FontProperties(size=ps.LABEL_PT)
        eft_bold = FontProperties(size=ps.LABEL_PT, weight="bold")
        eft_w_px = (renderer.get_text_width_height_descent("EFT", eft_bold, False)[0]
                    + renderer.get_text_width_height_descent(":", eft_props, False)[0])
        shift_in = (eft_w_px / fig.dpi + EFT_LEGEND_GAP_PT / 72) / 2
        nudge_x_in, nudge_y_in = (v / 72 for v in LEGEND_B_NUDGE_PT)
        legend_b.set_bbox_to_anchor(
            (cx_b, 0.0), transform=below_b + mt.ScaledTranslation(shift_in + nudge_x_in, nudge_y_in,
                                                                  fig.dpi_scale_trans))
        fig.canvas.draw()
        lb, ab = legend_b.get_window_extent(renderer), ax_b.get_window_extent(renderer)
        eft_b = ax_b.annotate("EFT:", xy=(0, 0), xycoords="axes fraction",
                      xytext=((lb.x0 - ab.x0) / fig.dpi * 72 - EFT_LEGEND_GAP_PT,
                              ((lb.y0 + lb.y1) / 2 - ab.y0) / fig.dpi * 72),
                      textcoords="offset points", ha="right", va="center",
                      fontproperties=eft_props, annotation_clip=False)

        # Pass 2: with every decoration in place, read the final axes geometry.
        fig.canvas.draw()
        ax_ab = ax_a.get_window_extent(renderer)
        width_a_in, height_a_in = ax_ab.width / fig.dpi, ax_ab.height / fig.dpi
        if width_a_in < content_in:
            raise ValueError(f"(a) axes is {width_a_in:.2f} in wide; the bars and their 8 pt "
                             f"labels need {content_in:.2f} in -- widen width_ratios[0]")
        ax_a.set_xlim(0, content_in)   # the inch-designed layout stretches to fill (>= 1 in/unit)
        fig.canvas.draw()
        boxes = [t.get_window_extent(renderer) for t in group_labels]
        for lo, hi in zip(boxes, boxes[1:]):
            if lo.x1 > hi.x0:
                raise ValueError("(a) group headers collide -- widen GROUP")
        # value labels: only on segments tall enough for 8 pt type
        pt_per_pct = height_a_in * 72 / 100
        min_pct = MIN_LABEL_PT / pt_per_pct
        dropped = []
        for xi, bo, v, tc in segments:
            if v >= min_pct:
                ax_a.text(xi, bo + v / 2, f"{v:.0f}", ha="center", va="center", color=tc)
            elif v > 0:
                dropped.append(f"{v:.1f}")

        # Line the two axis baselines up without coupling what hangs below them
        # (Jonathan, 2026-09-12): the shallower column gets an invisible spacer (a
        # blank annotation, measured by the layout) under its axes, as deep as the
        # deeper column's decorations, so both axes get the same bottom margin
        # while each legend keeps its own depth.
        column = {ax_a: sf_a, ax_b: sf_b}
        for _pass in range(3):          # the second pass takes up any residual
            fig.canvas.draw()
            bottoms = {ax: ax.get_window_extent(renderer).y0 for ax in (ax_a, ax_b)}
            low, high = min(bottoms, key=bottoms.get), max(bottoms, key=bottoms.get)
            diff_pt = (bottoms[high] - bottoms[low]) / fig.dpi * 72
            if diff_pt <= 0.25:
                break
            depth_pt = (bottoms[low] - column[low].get_tightbbox(renderer).y0) / fig.dpi * 72
            low.annotate(" ", xy=(0, 0), xycoords="axes fraction",
                         xytext=(0, -(depth_pt + diff_pt)), textcoords="offset points",
                         va="bottom", annotation_clip=False)

        # Keywords in colour on the under-bar labels only (Jonathan, 2026-09-12):
        # both rows go to paint; save is told not to touch anything else.
        ps.paint(fig, include=[*ax_a.get_xticklabels(), *group_labels, header_b,
                               *legend_b.get_texts(), eft_a, eft_b],
                 extra={"control": ps.DARK_GREY, "eft": ps.INK})   # bold "EFT", plain ":"
        written = ps.save(fig, OUTPUT, stem, paint_keywords=False, formats=formats)
    plt.close(fig)

    print(f"(a) axes {width_a_in:.2f} x {height_a_in:.2f} in; value labels on segments >= "
          f"{min_pct:.1f}% ({MIN_LABEL_PT:g} pt); unlabelled: {', '.join(dropped)}%")
    print("acted  charter:", [f"{v:.0f}" for v in ch], " other:", [f"{v:.0f}" for v in oth],
          " coin:", [f"{v:.0f}" for v in co])
    print("stated amb    :", [f"{v:.0f}" for v in amb])
    print("stated +2%coin:", [f"{v:.0f}" for v in coin])
    return written


def main() -> int:
    written = render() + render(**THREE_KW)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
