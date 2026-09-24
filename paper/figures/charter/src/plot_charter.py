"""Methods figure: the Qalvori Dispatch Charter and the coin rule, as a
schematic of the decision the dispatch clerk is meant to make.

What it draws
-------------
A portrait page in two bands. The Charter (top, blue, dominant) is a
top-to-bottom flow across the full text width: take the hardest open run
(Article 1); a crew qualifies only if it passes all three gates (Article 2,
drawn side by side with AND badges, so they read as a conjunction); rank the
qualifying crews by a strictly sequential four-step ladder (Article 3, drawn
left to right with "tie" arrows, so it reads as "go to the next step only on
a tie"); whichever step decides first names the Charter crew. The coin rule
(below, orange, a shallow band a third the Charter's height) is the
contrast: crew quote -> operator margin -> highest total margin wins, drawn
left to right, with the note that it may pick a crew that fails
qualification as one italic line under the flow. Under the bands sits the
box-style legend (solid = clause decision-relevant in the EFT episodes,
dashed = held out of EFT) -- a legend, not a caption. No data is drawn;
there is nothing measured on this figure.

Rules 2026-09-12: no caption text on the figure; keywords painted by
ps.paint. The two centred sentences that used to define the episode types
under the legend were caption text and are gone (the page gave back the
0.35 in they and their gap took: 5.96 -> 5.61 in); they belong in the
LaTeX caption.

For the caption:

    Agreement episode: both rules pick the same crew.
    Conflict episode: the Charter crew and the coin crew differ.

Keywords in colour: of the ink text, only the Charter panel's subtitle
"never refers to coin amounts" carries one, so its "coin" prints orange and
bold (right-aligned, so the bold word grows leftwards; nothing boxes it in).
The coin result box, "Coin crew gets the run", is ink on the orange fill:
painted, its "Coin" would be orange on orange, so ``clause_row`` registers
text on a filled box in ``Page.plain`` and ``main`` calls
``ps.paint(fig, exclude=page.plain)`` itself, saving with
``paint_keywords=False``. The white "Charter crew gets the run", the two
coloured panel heads and the orange italic note are skipped by ``ps.paint``
on their own (not ink).

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines). Authored and saved at exactly 5.5 in wide with no
``bbox_inches`` -- the manuscript includes it at ``width=\\linewidth``, so the
page is not rescaled and 8 pt prints as 8 pt. (Ported 2026-09-11 from a
13 in-wide, two-column render saved bbox-tight at 10.3 x 6.6 in, whose
6.8-13 pt type printed at 3.6-7 pt; the coin column could not sit beside
the four-rung ladder at 8 pt, hence the two bands.) The axes maps data
units 1:1 onto inches, every box is sized from the measured extent of its
wrapped text (row widths come from the page width and the badge / tie-arrow
gaps; box text wraps to the box on the fewest, most even lines), and the
page height is whatever the stack needs, found by a first measuring pass at
the PNG's 300 dpi (Agg over-measures text by 3-4% at 100 dpi). Type: body 8 pt (clause text, tags,
labels, legend); the figure title, panel titles and the bold row
heads 9 pt. Colour: Charter ``ps.CHARTER`` (boxes, badges, the filled result
box), coin ``ps.COIN``; panel fills ``ps.lighten(colour, 0.9)``; ink
``ps.INK``, secondary ink ``ps.MUTED`` (arrows, brackets, subtitles, tags;
also Article 1's neutral outline). The coin result box carries ink-coloured
text: white on the house orange is too faint at 8 pt.

Where the text comes from
-------------------------
Every clause is a paraphrase (six words or fewer per box) of
``experiments/prior_coins/design/dispatch_charter_v1.md``: Article 1
(order of dispatch), Article 2 (qualification), Article 3 (precedence), the
"Separate coin rule used in the experiment" section (quote formula, margin,
coin-maximising allocation may assign a non-qualifying crew), and its last
paragraph (agreement vs conflict episodes).

Held-in vs held-out clauses
---------------------------
Solid boxes are the clauses that were decision-relevant in the elicitation
finetuning (EFT) episodes of the final grid; dashed boxes with a "held out
of EFT" tag are the two clauses present in the midtraining documents but
never decision-relevant in EFT. The assignment is ``TRAIN_CLAUSES`` /
``HELD_OUT_CLAUSES`` in ``experiments/prior_coins/build_dispatch_v4_aft.py``
(imported as ``v4aft.TRAIN_CLAUSES`` by
``experiments/prior_coins/dispatch_final_v1/build_aft_mixtures.py``), on
branch ``sid/dispatch-final-v1``; the held-out clauses are what the
``eval_holdout_*`` slices of the final grid measure.

    TRAIN_CLAUSES    = qual_skill, qual_specialty, precedence_runs_year,
                       precedence_days_since, precedence_registry_rank
    HELD_OUT_CLAUSES = qual_weekly_limit, precedence_deferrals

Article 1 (run order) is neither: it orders runs, not crews, so it is not a
crew-choice clause and carries no tag.

Writes ``charter.pdf``, ``charter.png`` (the same page at 300 dpi) and
``charter.svg`` (to edit) next to ``src/`` -- ``ps.save``'s default (PDF only,
Jonathan 2026-09-11, until 2026-09-14; SVG since 2026-09-24)::

    uv run --extra dev python3 paper/figures/charter/src/plot_charter.py
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent              # paper/figures/charter/
STEM = "charter"

# Palette: the house pair (main.tex's charter / coin), their pale tints for
# the panel fills, and the house inks. Charter = blue, coin rule = orange.
CHARTER = ps.CHARTER
COIN = ps.COIN
INK = ps.INK
MUTED = ps.MUTED                  # arrows, brackets, subtitles, tails, tags
NEUTRAL = ps.MUTED                # Article 1's outline: neither held-in nor held-out
CHARTER_PANEL = ps.lighten(ps.CHARTER, 0.9)
COIN_PANEL = ps.lighten(ps.COIN, 0.9)
WHITE = "#ffffff"
HELD_OUT_TAG = "held out of EFT"

# Type: body 8 pt, headings 9 pt bold; nothing smaller anywhere.
BODY = ps.FONT_PT
HEAD = ps.TITLE_PT
LINE_SPACING = 1.2

# Geometry, all in inches: the single axes maps data units 1:1 onto the page.
PAGE_W = ps.TEXTWIDTH_IN
MARGIN = 0.05                     # page edge -> panel border
PANEL_PAD = 0.10                  # panel border -> content
BOX_PAD_X = 0.07                  # box border -> text
BOX_PAD_Y = 0.07
BOX_MIN_H = 0.30
BOX_SLACK = 0.08                  # extra width on a single, self-sized box
TAG_GAP = 0.04                    # clause text -> held-out tag
ARROW_BAND = 0.32                 # a vertical connector and its label
BRACKET_DROP = 0.12               # tick length from a row down to its bar
TITLE_GAP = 0.08                  # figure title -> first panel
PANEL_GAP = 0.08                  # Charter panel -> coin panel
BLOCK_GAP = 0.08                  # coin panel -> legend
BADGE_R = 0.18                    # the AND disc
STEP_R = 0.085                    # the numbered disc on a rung
TIE_GAP = 0.34                    # between rungs: room for the "tie" arrow
MIN_FLOW_GAP = 0.16               # narrowest gap an arrow head fits in
FIT_EPS = 1e-6                    # a box sized to its text holds that text

TITLE = ("How the dispatch clerk is meant to choose a crew,\n"
         "and the rule it is tempted by")
GATES = [("Skill ≥ run difficulty", False),
         ("Fewer than three runs this week", True),
         ("Holds the required specialty, if any", False)]
RUNGS = [("Fewer runs this year", False),
         ("More days since last allocation", False),
         ("More deferrals this quarter", True),
         ("Lower registry rank", False)]
QUOTE = ("Crew quote =\nmobilization\n+ daily rate × sailors × days"
         "\n+ difficulty supplement\n+ specialty supplement")
MARGIN_RULE = "Margin = payment − quote"
HIGHEST = "Highest total margin wins"
COIN_NOTE = "may pick a crew that fails the Charter's qualification"
LEGEND = [(False, "clause decision-relevant in the elicitation finetuning (EFT) episodes"),
          (True, "held out of EFT: in the midtraining documents, never decision-relevant in EFT")]


class Page:
    """One axes spanning a 5.5 in-wide page, data units = inches, with the
    measuring helpers the layout is built from."""

    def __init__(self, height_in: float):
        self.fig, self.ax = ps.figure(height_in, layout=None, dpi=ps.PNG_DPI)
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, PAGE_W)
        self.ax.set_ylim(0, height_in)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.plain: list[Text] = []   # texts kept out of ps.paint (see clause_row)

    # ------------------------------------------------------------ measuring
    def extent(self, t) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of a Text's ink box, in inches on the page."""
        bb = t.get_window_extent(self.renderer)
        (x0, y0), (x1, y1) = self.ax.transData.inverted().transform(
            [[bb.x0, bb.y0], [bb.x1, bb.y1]])
        return x0, y0, x1, y1

    def height(self, t) -> float:
        x0, y0, x1, y1 = self.extent(t)
        return y1 - y0

    def size(self, s: str, **font: Any) -> tuple[float, float]:
        """Width and height of ``s`` as it would render, in inches."""
        t = self.ax.text(0, 0, s, linespacing=LINE_SPACING, **font)
        x0, y0, x1, y1 = self.extent(t)
        t.remove()
        return x1 - x0, y1 - y0

    def cap_height(self, t) -> float:
        """Cap height of a Text's font, in inches (for baseline alignment)."""
        _w, h, _d = self.renderer.get_text_width_height_descent(
            "H", t.get_fontproperties(), ismath=False)
        return h / self.fig.dpi

    def wrap(self, s: str, max_w: float, **font: Any) -> str:
        """Wrap to a measured width on the fewest lines, breaking where the
        lines come out most even; explicit newlines are kept."""
        out: list[str] = []
        for para in s.split("\n"):
            words = para.split()
            n_lines = len(self._greedy(words, max_w, **font))
            best: tuple[tuple[float, float], list[str]] | None = None
            for breaks in itertools.combinations(range(1, len(words)), n_lines - 1):
                bounds = (0, *breaks, len(words))
                lines = [" ".join(words[i:j]) for i, j in zip(bounds, bounds[1:])]
                widths = [self.size(line, **font)[0] for line in lines]
                if max(widths) > max_w + FIT_EPS:
                    continue
                key = (max(widths), sum(w * w for w in widths))
                if best is None or key < best[0]:
                    best = (key, lines)
            if best is None:
                raise ValueError(f"cannot wrap {para!r} into {max_w:.2f} in")
            out.extend(best[1])
        return "\n".join(out)

    def _greedy(self, words: list[str], max_w: float, **font: Any) -> list[str]:
        lines, line = [], words[0]
        for word in words[1:]:
            trial = f"{line} {word}"
            if self.size(trial, **font)[0] <= max_w + FIT_EPS:
                line = trial
            else:
                lines.append(line)
                line = word
        return lines + [line]

    def two_line_width(self, s: str, **font: Any) -> float:
        """The narrowest text width at which ``s`` wraps onto two lines."""
        words = s.split()
        return min(max(self.size(" ".join(words[:i]), **font)[0],
                       self.size(" ".join(words[i:]), **font)[0])
                   for i in range(1, len(words)))

    # -------------------------------------------------------------- drawing
    def text(self, x: float, y: float, s: str, **kw: Any):
        kw.setdefault("fontsize", BODY)
        kw.setdefault("color", INK)
        kw.setdefault("linespacing", LINE_SPACING)
        kw.setdefault("zorder", 7)
        return self.ax.text(x, y, s, **kw)

    def rbox(self, x: float, y_top: float, w: float, h: float, *, colour: str,
             face: str = WHITE, lw: float = 0.9, dashed: bool = False,
             radius: float = 0.06, zorder: int = 3) -> None:
        self.ax.add_patch(FancyBboxPatch(
            (x, y_top - h), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face, edgecolor=colour, linewidth=lw,
            linestyle=(0, (4, 2.5)) if dashed else "-", zorder=zorder))

    def clause_row(self, y_top: float, xs: list[float], w: float,
                   items: list[tuple[str, bool]], *, colour: str = CHARTER,
                   face: str = WHITE, text_colour: str = INK,
                   bold: bool = False) -> float:
        """Equal boxes of width ``w`` whose tops sit at ``y_top``: each text
        wraps to the box, the row takes the tallest; a held-out box dashes
        its outline and prints the tag beneath its text. Returns the height."""
        font: dict[str, Any] = {"fontsize": BODY}
        if bold:
            font["fontweight"] = "bold"
        wrapped = [self.wrap(s, w - 2 * BOX_PAD_X, **font) for s, _ in items]
        text_h = max(self.size(s, **font)[1] for s in wrapped)
        tag_band = 0.0
        if any(held for _, held in items):
            tag_band = self.size(HELD_OUT_TAG, fontsize=BODY, style="italic")[1] + TAG_GAP
        h = max(BOX_MIN_H, text_h + 2 * BOX_PAD_Y + tag_band)
        for x, s, (_, held) in zip(xs, wrapped, items):
            self.rbox(x, y_top, w, h, colour=colour, face=face, dashed=held)
            band = tag_band if held else 0.0
            t = self.text(x + w / 2, y_top - (h - band) / 2, s, ha="center",
                          va="center", color=text_colour, **font)
            if face != WHITE:         # ink on a fill: a painted keyword would
                self.plain.append(t)  # take the fill's own colour (orange on orange)
            if held:
                self.text(x + w / 2, y_top - h + BOX_PAD_Y, HELD_OUT_TAG,
                          ha="center", va="bottom", style="italic", color=MUTED)
        return h

    def arrow(self, x0: float, y0: float, x1: float, y1: float, *,
              colour: str = MUTED, lw: float = 1.0) -> None:
        self.ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
            color=colour, linewidth=lw, shrinkA=0, shrinkB=0, zorder=5))

    def row_label(self, x: float, y_centre: float, head: str, tail: str, *,
                  colour: str = CHARTER) -> None:
        """'Head  tail', head bold and coloured, tail muted, sharing a
        baseline whose caps are centred on ``y_centre``."""
        t = self.text(x, 0, head, fontsize=HEAD, fontweight="bold",
                      color=colour, ha="left", va="baseline")
        baseline = y_centre - self.cap_height(t) / 2
        t.set_y(baseline)
        x_tail = self.extent(t)[2] + 0.08
        self.text(x_tail, baseline, tail, color=MUTED, ha="left", va="baseline")

    def connector(self, x: float, y_top: float, label: str, *,
                  head: str | None = None, tail: str = "",
                  x_label: float = 0.0) -> float:
        """A vertical arrow down through one ARROW_BAND with its label to the
        right of the shaft and, optionally, a row label at the left."""
        y_bottom = y_top - ARROW_BAND
        y_mid = y_top - ARROW_BAND / 2
        self.arrow(x, y_top, x, y_bottom)
        self.text(x + 0.08, y_mid, label, ha="left", va="center",
                  style="italic", color=MUTED)
        if head:
            self.row_label(x_label, y_mid, head, tail)
        return y_bottom

    def bracket(self, xs: list[float], y_top: float) -> float:
        """Ticks down from each x to a shared bar; returns the bar's y."""
        y_bar = y_top - BRACKET_DROP
        for x in xs:
            self.ax.plot([x, x], [y_top, y_bar], color=MUTED, linewidth=0.9,
                         zorder=4, solid_capstyle="butt")
        self.ax.plot([min(xs), max(xs)], [y_bar, y_bar], color=MUTED,
                     linewidth=0.9, zorder=4, solid_capstyle="round")
        return y_bar

    def badge(self, x: float, y: float, s: str, *, colour: str = CHARTER) -> None:
        self.ax.add_patch(Circle((x, y), BADGE_R, facecolor=colour,
                                 edgecolor="none", zorder=6))
        self.text(x, y, s, ha="center", va="center", color=WHITE,
                  fontweight="bold", zorder=7)

    def step_number(self, x: float, y: float, n: int, *, colour: str = CHARTER) -> None:
        self.ax.add_patch(Circle((x, y), STEP_R, facecolor=colour,
                                 edgecolor=WHITE, linewidth=0.8, zorder=8))
        self.text(x, y, str(n), ha="center", va="center", color=WHITE,
                  fontweight="bold", zorder=9)

    def panel(self, y_top: float, y_bottom: float, *, colour: str, face: str,
              title: str, subtitle: str) -> float:
        """The band's rounded backdrop and its title row; returns the y at
        which the band's content starts."""
        self.rbox(MARGIN, y_top, PAGE_W - 2 * MARGIN, y_top - y_bottom,
                  colour=colour, face=face, lw=0.8, radius=0.1, zorder=1)
        y = y_top - PANEL_PAD
        t = self.text(MARGIN + PANEL_PAD, y, title, fontsize=HEAD,
                      fontweight="bold", color=colour, ha="left", va="top")
        self.text(PAGE_W - MARGIN - PANEL_PAD, y, subtitle, style="italic",
                  color=MUTED, ha="right", va="top")
        return y - self.height(t) - 0.10


# --------------------------------------------------------------- the bands


def charter_band(page: Page, y_top: float) -> float:
    """Draws the Charter's flow from ``y_top`` down; returns the panel's
    bottom y. (The backdrop is drawn last, at zorder 1, once the height is
    known.)"""
    xi0, xi1 = MARGIN + PANEL_PAD, PAGE_W - MARGIN - PANEL_PAD
    inner = xi1 - xi0
    cx = PAGE_W / 2
    # The title row is measured here and drawn by ``panel`` afterwards.
    y = y_top - PANEL_PAD - page.size("Qalvori Dispatch Charter",
                                      fontsize=HEAD, fontweight="bold")[1] - 0.10

    # Article 1: order of dispatch
    a1 = "Hardest open run first"
    w_a1 = page.size(a1)[0] + 2 * BOX_PAD_X + BOX_SLACK
    h_a1 = page.clause_row(y, [cx - w_a1 / 2], w_a1, [(a1, False)], colour=NEUTRAL)
    page.row_label(xi0, y - h_a1 / 2, "Order", "take runs one at a time")
    page.text(cx + w_a1 / 2 + 0.10, y - h_a1 / 2,
              "ties: longer run,\nthen lower docket number", style="italic",
              color=MUTED, ha="left", va="center")
    y -= h_a1
    y = page.connector(cx, y, "for this run", head="Qualify",
                       tail="all three must hold", x_label=xi0)

    # Article 2: qualification, three gates joined by AND
    gap = 2 * BADGE_R + 0.10
    w_gate = (inner - 2 * gap) / 3
    xs = [xi0 + i * (w_gate + gap) for i in range(3)]
    h_gate = page.clause_row(y, xs, w_gate, GATES)
    for x in xs[:-1]:
        page.badge(x + w_gate + gap / 2, y - h_gate / 2, "AND")
    y -= h_gate
    y = page.bracket([x + w_gate / 2 for x in xs], y)
    y = page.connector(cx, y, "qualifying crews", head="Rank",
                       tail="first decisive step wins", x_label=xi0)

    # Article 3: precedence, a strictly sequential ladder
    w_rung = (inner - 3 * TIE_GAP) / 4
    xs = [xi0 + i * (w_rung + TIE_GAP) for i in range(4)]
    h_rung = page.clause_row(y, xs, w_rung, RUNGS)
    for n, x in enumerate(xs, start=1):
        page.step_number(x, y, n)
    y_mid = y - h_rung / 2
    for x in xs[:-1]:
        page.arrow(x + w_rung, y_mid, x + w_rung + TIE_GAP, y_mid, colour=CHARTER)
        page.text(x + w_rung + TIE_GAP / 2, y_mid + 0.04, "tie", ha="center",
                  va="bottom", style="italic", color=CHARTER)
    y -= h_rung
    y = page.bracket([x + w_rung / 2 for x in xs], y)
    y = page.connector(cx, y, "whichever step decides first")

    # the result
    result = "Charter crew gets the run"
    w_res = page.size(result, fontweight="bold")[0] + 2 * BOX_PAD_X + BOX_SLACK
    h_res = page.clause_row(y, [cx - w_res / 2], w_res, [(result, False)],
                            colour=CHARTER, face=CHARTER, text_colour=WHITE, bold=True)
    y -= h_res
    y_bottom = y - PANEL_PAD
    page.panel(y_top, y_bottom, colour=CHARTER, face=CHARTER_PANEL,
               title="Qalvori Dispatch Charter", subtitle="never refers to coin amounts")
    return y_bottom


def coin_band(page: Page, y_top: float) -> float:
    """The coin rule as one left-to-right flow; returns the panel's bottom y."""
    xi0, xi1 = MARGIN + PANEL_PAD, PAGE_W - MARGIN - PANEL_PAD
    inner = xi1 - xi0
    y = y_top - PANEL_PAD - page.size("Coin rule", fontsize=HEAD,
                                      fontweight="bold")[1] - 0.10

    # Box widths from the text: the formula keeps its five lines, the other
    # three wrap onto two; the arrows take whatever width is left over.
    w_quote = max(page.size(line)[0] for line in QUOTE.split("\n")) + 2 * BOX_PAD_X
    w_margin = page.two_line_width(MARGIN_RULE) + 2 * BOX_PAD_X
    w_high = page.two_line_width(HIGHEST) + 2 * BOX_PAD_X
    w_res = page.two_line_width("Coin crew gets the run", fontweight="bold") + 2 * BOX_PAD_X
    widths = [w_quote, w_margin, w_high, w_res]
    gap = (inner - sum(widths)) / 3
    if gap < MIN_FLOW_GAP:
        raise ValueError(f"coin flow needs {sum(widths):.2f} in of boxes; only "
                         f"{inner:.2f} in of panel leaves {gap:.2f} in per arrow")
    xs = [xi0 + sum(widths[:i]) + i * gap for i in range(4)]

    h_quote = page.clause_row(y, [xs[0]], w_quote, [(QUOTE, False)], colour=COIN)
    axis = y - h_quote / 2
    h_small = max(BOX_MIN_H, page.size("a\nb")[1] + 2 * BOX_PAD_Y)
    y_small = axis + h_small / 2
    page.clause_row(y_small, [xs[1]], w_margin, [(MARGIN_RULE, False)], colour=COIN)
    page.clause_row(y_small, [xs[2]], w_high, [(HIGHEST, False)], colour=COIN)
    page.clause_row(y_small, [xs[3]], w_res, [("Coin crew gets the run", False)],
                    colour=COIN, face=COIN, text_colour=INK, bold=True)
    for x, w in zip(xs[:-1], widths[:-1]):
        page.arrow(x + w, axis, x + w + gap, axis)

    # the contrast: one line under the three small boxes, centred on the
    # deciding box, kept inside the trio's span
    span = xs[3] + w_res - xs[1]
    note = page.wrap(COIN_NOTE, span, style="italic")
    t = page.text(xs[2] + w_high / 2, y_small - h_small - TAG_GAP, note,
                  ha="center", va="top", style="italic", color=COIN)
    x_note0, y_note0, x_note1, _ = page.extent(t)
    if x_note0 < xs[1] - FIT_EPS or x_note1 > xs[3] + w_res + FIT_EPS:
        raise ValueError("the coin note runs past the boxes it annotates")
    y_low = min(y - h_quote, y_note0)
    y_bottom = y_low - PANEL_PAD
    page.panel(y_top, y_bottom, colour=COIN, face=COIN_PANEL, title="Coin rule",
               subtitle="the rival rule")
    return y_bottom


def legend(page: Page, y_top: float) -> float:
    """The two outline styles, one per line; returns the bottom y."""
    x = MARGIN + PANEL_PAD
    sw, sh = 0.26, 0.13
    y = y_top
    for dashed, label in LEGEND:
        t = page.text(x + sw + 0.08, y, label, ha="left", va="top", color=MUTED)
        h = page.height(t)
        page.rbox(x, y - (h - sh) / 2, sw, sh, colour=CHARTER, dashed=dashed, radius=0.03)
        y -= h + 0.04
    return y + 0.04


def draw(height_in: float) -> tuple[Page, float]:
    """Lays the page out from the top; returns it and the content's bottom y."""
    page = Page(height_in)
    y = height_in - MARGIN
    t = page.text(MARGIN, y, TITLE, fontsize=HEAD, fontweight="bold", ha="left", va="top")
    y -= page.height(t) + TITLE_GAP
    y = charter_band(page, y) - PANEL_GAP
    y = coin_band(page, y) - BLOCK_GAP
    y = legend(page, y)
    return page, y


def main() -> None:
    with matplotlib.rc_context(ps.rc(**{"figure.constrained_layout.use": False})):
        # Pass 1 measures the stack on a tall page; pass 2 draws it on a page
        # exactly that tall (data units are inches on both, so nothing moves).
        probe, y_bottom = draw(9.0)
        height = round(9.0 - y_bottom + MARGIN, 2)
        plt.close(probe.fig)
        page, _ = draw(height)
        # Keywords in colour: every ink "Charter" / "Coin" (here the "coin" of
        # the Charter panel's subtitle), except text on a filled box, whose
        # painted keyword would match its fill -- "Coin crew gets the run".
        ps.paint(page.fig, exclude=page.plain)
        ps.save(page.fig, OUTPUT, STEM, paint_keywords=False)
        plt.close(page.fig)


if __name__ == "__main__":
    main()
