r"""``scimt.viz.paper`` -- the house style for the write-up's figures.

One place for the three things every figure under ``paper/figures/`` must
agree on, so they cannot drift script by script.  (By 2026-09-11 the ledger
figures used four blues and three warm partners, and page widths from 5.5 to
12.5 in -- so nominal 8-9.5 pt fonts printed at 3.5-5 pt once LaTeX scaled
them to the column.)

1. **Geometry.**  A figure is authored at the ICLR text width,
   :data:`TEXTWIDTH_IN` = 5.5 in (``iclr2027_conference.sty:49``,
   ``\textwidth 5.5 true in``) and saved at exactly that size.  Never
   ``bbox_inches="tight"``: that re-crops the page to the ink, so
   ``\includegraphics[width=\linewidth]`` rescales it and every font size
   drifts by the ratio (5.76 in -> x0.955 on one figure; 12.5 in -> x0.44 on
   another).  Constrained layout keeps the decorations on the page and
   :func:`check` refuses a figure whose text runs off it.
2. **Type.**  Font sizes are absolute points and land on the page at the
   value set.  Nothing below :data:`MIN_FONT_PT` = 8 (Jonathan, 2026-09-11:
   "minimum size 8 across the board, for readability"; the manuscript body
   is 10 pt Times).  DejaVu Sans first in the stack so a render does not
   change on a machine that has Arial; TrueType embedding
   (``pdf.fonttype`` 42) so the PDF carries real fonts, not Type 3 outlines.
3. **Palette.**  The seaborn "colorblind" pair the manuscript itself defines
   -- ``main.tex``: ``\definecolor{charter}{HTML}{0173B2}`` and
   ``\definecolor{coin}{HTML}{DE8F05}``, overriding ``coincharter.sty``'s
   Okabe-Ito pair -- so a coloured word in the prose matches the bar it
   names.  Copied in as hex (no seaborn dependency).  Coin is *orange*, never
   vermilion (Jonathan, 2026-09-11); :data:`VERMILION` is kept for a value
   that is neither Charter nor Coin -- America in the MSM figure ("America
   is vermilion, as America is not coin").
4. **Keywords in colour.**  Every ink mention of *Charter* is blue and bold,
   *Coin* orange and bold, *Ambiguous* green and bold (Jonathan,
   2026-09-12), including the short forms Ch / Co / Amb.  :func:`paint`
   does this over any drawn text -- tick labels, legends, titles, axis
   labels, annotations, rotated or multi-line -- and :func:`save` runs it
   by default; a figure that must keep some label plain passes
   ``exclude=`` / ``include=`` (or ``paint=False`` and calls it itself).
5. **No caption text on a figure -- ever** (Jonathan, 2026-09-12: "never
   do these").  No caveat footnote, no provenance or methods note, no
   grey pseudo-caption under the axes; the LaTeX caption carries all of
   that.  The height a footer took is given back.

This module imports matplotlib at import time, so it is deliberately not
re-exported from :mod:`scimt.viz` (which stays CPU-only).  Figure scripts
import it as ``from scimt.viz import paper as ps`` and use it like::

    with matplotlib.rc_context(ps.rc()):
        fig, axes = ps.figure(height_in=2.6, ncols=3)
        ...                                   # draw
        ps.save(fig, OUTPUT, "dose_grid")     # paint(), check(), then the .pdf

The height is the author's; the width is not.

Layout notes (matplotlib 3.11; learned porting the ledger figures):

* Constrained layout measures axes-level artists -- tick labels, axis labels,
  titles, ``ax.legend``, ``ax.text``/``annotate`` even with ``clip_on=False``
  -- and reserves room for them.  It does *not* see ``fig.text``: a
  hand-placed figure legend or title needs :func:`reserve_band`.
* ``fig.suptitle`` and ``fig.legend(loc="outside upper ...")`` share the top
  margin (the larger wins; they do not stack) and overprint.  Give the legend
  its own thin axes row, or anchor it by hand inside a reserved band.
* ``fig.supxlabel`` is pinned to the page bottom whatever the layout rect;
  label the bottom axes instead, or place the text by hand inside a reserved
  band.
* :func:`paint` lays its pieces out from the anchors' drawn line layout, so
  it runs last (``save`` does it, at 72 dpi -- the PDF's display space).
  Each piece asks its anchor where it is at draw time (axis labels and
  annotations re-derive their position every draw), so pieces follow under
  any backend or dpi; the anchor itself is measured but never drawn, so the
  PDF carries each word once.
* Aspect-locked axes (``imshow``) leave slack in whichever dimension does not
  bind: sweep the height and take the smallest at which the width binds.
"""

from __future__ import annotations

import re
from pathlib import Path
from math import ceil, cos, floor, radians, sin
from types import MethodType
from typing import Any, Iterable

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Affine2DBase, IdentityTransform, TransformedPath

__all__ = [
    "TEXTWIDTH_IN", "MIN_FONT_PT", "FONT_PT", "LABEL_PT", "TITLE_PT", "PNG_DPI",
    "CHARTER", "COIN", "BLUE", "ORANGE", "GREEN", "VERMILION", "GREY", "LIGHT_GREY",
    "DARK_GREY", "INK", "MUTED",
    "barberpole",
    "CHARTER_LIGHT", "COIN_LIGHT", "GREEN_LIGHT", "LIGHT_MIX", "FONT_SANS_SERIF",
    "KEYWORDS", "lighten", "darken", "rc", "figure", "reserve_band", "paint", "texts", "check",
    "describe", "save", "page_size_pt",
]

# ------------------------------------------------------------------ geometry
#: ICLR text width in inches (``iclr2027_conference.sty:49``).  Every figure
#: is authored and saved at exactly this width (times ``width_frac``).
TEXTWIDTH_IN = 5.5
#: The manuscript embeds the PDF and the PDF is the only render committed
#: (Jonathan, 2026-09-11: "there shouldn't be any .png renderings").  A PNG
#: is an explicit, throw-away preview: ``save(..., formats=("pdf", "png"))``.
PNG_DPI = 300

# ---------------------------------------------------------------------- type
#: No text on a figure is smaller than this, in printed points.
MIN_FONT_PT = 8.0
#: Body text: tick labels, legends, value labels, footnotes.
FONT_PT = 8.0
#: Axis labels.
LABEL_PT = 9.0
#: Panel / figure titles (bold, per ``rc``).
TITLE_PT = 9.0
#: DejaVu Sans first: matplotlib's bundled face, so the render is the same on
#: every machine (seaborn's stack lists Arial first, which changes a figure
#: wherever Arial is installed).
FONT_SANS_SERIF = ["DejaVu Sans", "Arial", "Liberation Sans", "Bitstream Vera Sans",
                   "sans-serif"]

# ------------------------------------------------------------------- palette
#: seaborn "colorblind" index 0 -- ``\definecolor{charter}`` in main.tex.
CHARTER = BLUE = "#0173b2"
#: seaborn "colorblind" index 1 -- ``\definecolor{coin}`` in main.tex.
COIN = ORANGE = "#de8f05"
#: seaborn "colorblind" index 2: correct / valid outcomes (python4, agreement
#: episodes), never a motivation.
GREEN = "#029e73"
#: seaborn "colorblind" index 3: a value that is neither Charter nor Coin --
#: America in the MSM reproductions.  Never Coin (Coin is orange).
VERMILION = "#d55e00"
#: seaborn "colorblind" index 7: the control arm, "other" crews, no-MSM bars.
GREY = "#949494"
#: The light shade of grey for a paired "before" bar.
LIGHT_GREY = "#c9c9c9"
#: A grey for *words* naming the control arm: GREY is a bar colour and too
#: light for 8 pt type (Jonathan, 2026-09-12: "slightly darker grey").
DARK_GREY = "#767676"
#: Near-black ink for text and spines (the galleries' ink; not #000000) and
#: the muted ink for footnotes and secondary labels.
INK = "#22221f"
MUTED = "#3d3d3d"
#: How far a colour is blended toward white for the light bar of a pair
#: (per_clause's shade rule: "CHARTER mixed 55% with white").
LIGHT_MIX = 0.55


def lighten(color: str, mix: float = LIGHT_MIX) -> str:
    """Blend ``color`` toward white; ``mix`` 0 keeps it, 1 is white."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex(tuple(c + (1.0 - c) * mix for c in (r, g, b)))


def darken(color: str, mix: float) -> str:
    """Blend ``color`` toward black; ``mix`` 0 keeps it, 1 is black."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex(tuple(c * (1.0 - mix) for c in (r, g, b)))


CHARTER_LIGHT = lighten(CHARTER)
COIN_LIGHT = lighten(COIN)
GREEN_LIGHT = lighten(GREEN)

#: The words :func:`paint` sets in colour and bold wherever they appear in
#: ink text (case-insensitive, whole words; a sign glued to the front, as in
#: "−Coin" / "+Charter", and a trailing full stop ("Ambig.") are painted with
#: the word).  Short forms included: Ch / Co / Ambig. / Ambi / Amb.
KEYWORDS: dict[str, str] = {
    "charter": CHARTER, "ch": CHARTER,
    "coin": COIN, "co": COIN,
    "ambiguous": GREEN, "ambig": GREEN, "ambi": GREEN, "amb": GREEN,
}


def _keyword_re(keywords: dict[str, str]) -> re.Pattern[str]:
    """Whole-word, case-insensitive match of any keyword (longest first), with
    a sign glued to the front captured alongside it."""
    words = sorted(keywords, key=len, reverse=True)
    return re.compile(r"(?<![\w])([+\u2212-]?)(" + "|".join(map(re.escape, words))
                      + r")(\.?)(?![\w])", re.IGNORECASE)


_KEYWORD_RE = _keyword_re(KEYWORDS)
#: Text colours :func:`paint` will paint over; white-on-fill and already
#: coloured text keep their colour.
_PAINTABLE = {INK, MUTED, "#000000", "#262626", "#1a1a1a", "#222222", "#3d3d3d"}


# ------------------------------------------------------------------------ rc
def rc(**overrides: Any) -> dict[str, Any]:
    """The rcParams for a paper figure, for ``matplotlib.rc_context``.

    Absolute font sizes (nothing under :data:`MIN_FONT_PT`), the ink, thin
    spines with the top and right off, no grid, constrained layout on, and
    TrueType embedding.  ``overrides`` are applied last -- a figure that needs
    all four spines or a different pad says so here, not by editing the
    module.
    """
    params: dict[str, Any] = {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_SANS_SERIF,
        "font.size": FONT_PT,
        "axes.titlesize": TITLE_PT,
        "axes.titleweight": "bold",
        "axes.labelsize": LABEL_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "legend.title_fontsize": FONT_PT,
        "figure.titlesize": TITLE_PT,
        "figure.titleweight": "bold",
        "figure.labelsize": LABEL_PT,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 3 / 72,
        "figure.constrained_layout.h_pad": 3 / 72,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    params.update(overrides)
    return params


def figure(height_in: float, nrows: int = 1, ncols: int = 1, *,
           width_frac: float = 1.0, layout: str | None = "constrained",
           **subplots_kw: Any):
    """``plt.subplots`` at exactly ``width_frac`` of the text width.

    The height is the one free dimension; choose it for the content and let
    :func:`check` say whether the decorations fit.
    """
    return plt.subplots(nrows, ncols, figsize=(TEXTWIDTH_IN * width_frac, height_in),
                        layout=layout, **subplots_kw)


def reserve_band(fig: Figure, *, bottom_in: float = 0.0, top_in: float = 0.0) -> None:
    """Shrink the constrained-layout rect by ``bottom_in`` / ``top_in`` inches.

    Cumulative: each call takes its band off the rect the previous call left,
    so a footnote block, the caveat and a hand-placed legend can each reserve
    their own room.  Figure-level text (``fig.text``, a ``fig.legend`` with a
    ``bbox_to_anchor``) is invisible to constrained layout; this is how it
    gets space.  No-op on a figure without a constrained-layout engine.
    """
    engine = fig.get_layout_engine()
    if engine is None or not hasattr(engine, "set"):
        return
    _w, h = fig.get_size_inches()
    left, bottom, width, height = engine.get().get("rect") or (0.0, 0.0, 1.0, 1.0)
    height_new = height - (bottom_in + top_in) / h
    if height_new <= 0.05:
        raise ValueError(f"reserving {bottom_in:.2f} + {top_in:.2f} in leaves no room for "
                         f"the axes on a {h:.2f} in page")
    engine.set(rect=(left, bottom + bottom_in / h, width, height_new))


# --------------------------------------------------------------- barber pole
class _Barberpole(Artist):
    """Diagonal stripes of one colour over a patch, drawn at draw time as
    filled paths clipped to the patch (see :func:`barberpole`)."""

    def __init__(self, patch: Patch, colour: str, *, pitch_pt: float, angle_deg: float,
                 duty: float, zorder: float) -> None:
        super().__init__()
        self._patch, self._rgba = patch, mcolors.to_rgba(colour)
        self._pitch_pt, self._angle_deg, self._duty = pitch_pt, angle_deg, duty
        self.set_zorder(zorder)
        self.set_gid("ps:pole")

    def get_window_extent(self, renderer=None):
        return self._patch.get_window_extent(renderer)

    def draw(self, renderer) -> None:
        if not (self.get_visible() and self._patch.get_visible()):
            return
        bbox = self._patch.get_window_extent(renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            return
        pitch = renderer.points_to_pixels(self._pitch_pt)
        along = np.array([cos(radians(self._angle_deg)), sin(radians(self._angle_deg))])
        across = np.array([-along[1], along[0]])
        corners = np.array([[bbox.x0, bbox.y0], [bbox.x1, bbox.y0],
                            [bbox.x1, bbox.y1], [bbox.x0, bbox.y1]])
        s, t = corners @ along, corners @ across
        s0, s1 = s.min() - pitch, s.max() + pitch
        width = pitch * self._duty
        gc = renderer.new_gc()
        gc.set_clip_rectangle(bbox)
        gc.set_clip_path(TransformedPath(self._patch.get_path(), self._patch.get_transform()))
        gc.set_linewidth(0.0)
        # Stripe k covers t in [k * pitch, k * pitch + width): the phase is the
        # display origin's, so every pole in a figure is in step (as a pattern
        # would be) and the stripes run on across touching bars.
        for k in range(floor(t.min() / pitch), ceil(t.max() / pitch) + 1):
            ta, tb = k * pitch, k * pitch + width
            quad = np.array([ta * across + s0 * along, ta * across + s1 * along,
                             tb * across + s1 * along, tb * across + s0 * along])
            # Path(closed=True) reads its last vertex as the CLOSEPOLY point, so
            # the first corner is repeated to keep all four.
            path = MplPath(np.vstack([quad, quad[:1]]), closed=True)
            renderer.draw_path(gc, path, IdentityTransform(), self._rgba)
        gc.restore()


def barberpole(patch: Patch, colour: str, *, pitch_pt: float = 3.5, angle_deg: float = 45.0,
               duty: float = 0.5, into: Axes | Figure | None = None,
               zorder: float | None = None) -> Artist:
    """Stripe ``patch`` with ``colour``: a two-colour barber pole of the patch's
    own fill and ``colour``, stripes ``pitch_pt`` apart (measured across them)
    and ``duty`` of that pitch wide, leaning ``angle_deg`` from horizontal.

    The stripes are ordinary filled paths clipped to the patch, added to
    ``into`` (default: the patch's axes, else its figure) just above the patch
    -- not a matplotlib hatch.  A hatch becomes a PDF tiling pattern, and a
    hatched patch whose edge is transparent (``hatchcolor=`` alone, the edge
    left at its ``'none'`` default) puts a stroke alpha of 0 in the graphics
    state that poppler's Cairo renderer (Evince) then applies to the
    pattern's strokes: the stripes vanish; its Splash renderer (Okular,
    pdftoppm) drew them hairline-thin on the same figure (checked 2026-09-14,
    poppler 26.01; Ghostscript was right).  With an explicit ``edgecolor`` a
    hatch renders fine everywhere (python4_eft_supp_code_correctness relies on
    that); filled paths do regardless.  Legend handles are patches too: pass
    ``into=ax`` and a ``zorder`` above the legend's.  Returns the stripe
    artist.
    """
    container = into if into is not None else (patch.axes or patch.figure)
    if container is None:
        raise ValueError("barberpole: the patch belongs to no axes or figure; pass into=")
    pole = _Barberpole(patch, colour, pitch_pt=pitch_pt, angle_deg=angle_deg, duty=duty,
                       zorder=patch.get_zorder() + 0.01 if zorder is None else zorder)
    container.add_artist(pole)
    return pole


# --------------------------------------------------------------------- paint
def _segments(line: str, keywords: dict[str, str],
              pattern: re.Pattern[str]) -> list[tuple[str, str | None]]:
    """Split one line into (text, colour-or-None) pieces at the keywords."""
    out: list[tuple[str, str | None]] = []
    pos = 0
    for m in pattern.finditer(line):
        if m.start() > pos:
            out.append((line[pos:m.start()], None))
        out.append((m.group(0), keywords[m.group(2).lower()]))
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], None))
    return out


def _layout_only_draw(self: Text, renderer) -> None:
    """Instance-level ``draw`` for a painted anchor: measured by the layout
    (extents do not go through ``draw``), geometry kept current, no glyphs
    emitted -- so the PDF carries each word once."""
    if isinstance(self, Annotation):        # rebuilds its transform per draw
        self.update_positions(renderer)


class _BesideAnchor(Affine2DBase):
    """Translation to an anchor's drawn anchor point plus an offset in points,
    evaluated at draw time.  Axis labels re-derive one coordinate in display
    pixels every draw and annotations rebuild their whole transform, so a
    piece must ask its anchor where it is *now* -- under whichever backend
    and dpi is drawing."""

    def __init__(self, anchor: Text, dx_pt: float, dy_pt: float):
        super().__init__()
        self._anchor, self._dx, self._dy = anchor, dx_pt, dy_pt

    def get_matrix(self):
        x, y = self._anchor.get_transform().transform(self._anchor.get_unitless_position())
        s = self._anchor.figure.dpi / 72.0
        return np.array([[1.0, 0.0, x + self._dx * s],
                         [0.0, 1.0, y + self._dy * s],
                         [0.0, 0.0, 1.0]])


def _paint_one(fig: Figure, renderer, anchor: Text, keywords: dict[str, str],
               pattern: re.Pattern[str]) -> list[Text]:
    """Draw ``anchor``'s text over itself as pieces, the keywords in colour and
    bold, along the anchor's own baseline(s).

    Each piece sits at the anchor's drawn anchor point plus an offset in
    *points* (:class:`_BesideAnchor`, evaluated at draw time), so it follows
    the anchor wherever a backend's layout puts it.  The anchor keeps
    reserving its layout space but is never drawn.
    """
    props = anchor.get_fontproperties()
    bold = props.copy()
    bold.set_weight("bold")
    theta = anchor.get_rotation()
    dx, dy = cos(radians(theta)), sin(radians(theta))
    per_px = 72.0 / fig.dpi                      # display px (this dpi) -> points
    align = {"left": 0.0, "center": 0.5, "right": 1.0}[anchor.get_horizontalalignment()]
    base_colour = anchor.get_color()
    pieces: list[Text] = []
    # Unhinted metrics: the PDF backend measures unhinted, and hinted widths
    # are whole pixels at the build dpi.  ``Text._get_layout`` (private, but
    # stable since 1.x; matplotlib 3.11 here) gives each line's baseline-left
    # offset in display units, already rotated -- what ``Text.draw`` uses.
    with matplotlib.rc_context({"text.hinting": "none"}):
        _bbox, info, _rest = anchor._get_layout(renderer)
        for line, metrics, *offset in info:          # 3.11: (line, (w, a, d), (x, y))
            line_w = metrics[0]
            x, y = offset[0] if len(offset) == 1 else offset
            segs = _segments(line, keywords, pattern)
            widths = [renderer.get_text_width_height_descent(
                          t, bold if c else props, False)[0] for t, c in segs]
            run = -(sum(widths) - line_w) * align      # bold pieces widen the run
            for (t, c), w in zip(segs, widths):
                piece = Text(0.0, 0.0, t, color=c or base_colour,
                             fontproperties=bold if c else props,
                             ha="left", va="baseline", rotation=theta,
                             rotation_mode="anchor", in_layout=False, gid="ps:rich")
                piece.set_transform(_BesideAnchor(
                    anchor, (x + run * dx) * per_px, (y + run * dy) * per_px))
                fig.add_artist(piece)
                pieces.append(piece)
                run += w
    anchor.set_alpha(0.0)
    anchor.set_gid("ps:rich-anchor")
    anchor.draw = MethodType(_layout_only_draw, anchor)   # measured, never emitted
    return pieces


def paint(fig: Figure, *, include: Iterable[Text] | None = None,
          exclude: Iterable[Text] = (), extra: dict[str, str] | None = None) -> list[Text]:
    """Set every keyword (:data:`KEYWORDS`) in colour and bold, in place.

    Walks the drawn texts (or only ``include``), skipping ``exclude``, texts
    that are not in ink (white on a fill, a label already in its side's
    colour), transparent anchors and pieces from an earlier call.  ``extra``
    adds words for this call only (e.g. ``{"control": ps.GREY}`` for a figure
    whose control arm is named in a label).  Call it last -- pieces are
    placed from the anchors' drawn positions.  Returns the pieces it added.
    """
    keywords = {**KEYWORDS, **{k.lower(): v for k, v in (extra or {}).items()}}
    pattern = _keyword_re(keywords) if extra else _KEYWORD_RE
    fig.canvas.draw()
    renderer = _renderer(fig)
    targets = list(include) if include is not None else texts(fig)
    skip = {id(t) for t in exclude}
    pieces: list[Text] = []
    for t in targets:
        if id(t) in skip or (t.get_gid() or "").startswith("ps:rich"):
            continue
        if t.get_alpha() == 0 or not pattern.search(t.get_text()):
            continue
        if mcolors.to_hex(t.get_color()).lower() not in _PAINTABLE:
            continue
        pieces += _paint_one(fig, renderer, t, keywords, pattern)
    return pieces


# --------------------------------------------------------------------- checks
def _tick_labels(fig: Figure) -> tuple[set[int], list[Text]]:
    """ids of every tick-label Text on the figure, and the subset matplotlib
    draws.  A locator's out-of-view ticks keep Text objects that are never
    drawn (they sit off the axes), so ``findobj(Text)`` alone over-reports."""
    every: set[int] = set()
    drawn: list[Text] = []
    for ax in fig.findobj(Axes):                     # includes inset / child axes
        axes_list = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if hasattr(ax, "zaxis") else [])
        for axis in axes_list:
            lo, hi = sorted(axis.get_view_interval())
            span = (hi - lo) or 1.0
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                shown = lo - 1e-9 * span <= tick.get_loc() <= hi + 1e-9 * span
                for label in (tick.label1, tick.label2):
                    every.add(id(label))
                    if shown:
                        drawn.append(label)
    return every, drawn


def texts(fig: Figure) -> list[Text]:
    """Every visible, non-empty Text matplotlib will draw: titles, labels,
    the drawn tick labels, legends, colour bars, annotations, footnotes."""
    fig.canvas.draw()                                # positions the ticks
    tick_ids, drawn_ticks = _tick_labels(fig)
    found = [t for t in fig.findobj(Text) if id(t) not in tick_ids] + drawn_ticks
    return [t for t in found if t.get_visible() and t.get_text().strip()]


def _renderer(fig: Figure):
    fig.canvas.draw()
    try:
        return fig.canvas.get_renderer()
    except AttributeError:            # a non-Agg canvas
        return fig._get_renderer()


def check(fig: Figure, *, width_frac: float = 1.0, min_font_pt: float = MIN_FONT_PT,
          slack_pt: float = 1.0) -> None:
    """Refuse a figure that would not print as authored.

    Raises ``ValueError`` listing every problem found: a canvas that is not
    ``TEXTWIDTH_IN * width_frac`` wide; any text below ``min_font_pt``; any
    text or artist whose ink runs more than ``slack_pt`` off the canvas (it
    would be cropped by the page -- or, under ``bbox_inches="tight"``,
    silently change the page size).
    """
    problems: list[str] = []
    w, h = fig.get_size_inches()
    expected = TEXTWIDTH_IN * width_frac
    if abs(w - expected) > 1e-3:
        problems.append(f"canvas is {w:.3f} in wide; must be {expected:.3f} in "
                        f"(TEXTWIDTH_IN {TEXTWIDTH_IN} x width_frac {width_frac})")

    small = sorted({(round(t.get_fontsize(), 2), t.get_text().splitlines()[0][:40])
                    for t in texts(fig) if t.get_fontsize() < min_font_pt - 1e-6})
    if small:
        listed = "; ".join(f"{size:g} pt {body!r}" for size, body in small[:12])
        more = f" (+{len(small) - 12} more)" if len(small) > 12 else ""
        problems.append(f"text below {min_font_pt:g} pt: {listed}{more}")

    renderer = _renderer(fig)
    width_px, height_px = w * fig.dpi, h * fig.dpi
    for t in texts(fig):
        box = t.get_window_extent(renderer)
        over = max(-box.x0, box.x1 - width_px, -box.y0, box.y1 - height_px)
        if over > slack_pt * fig.dpi / 72:
            problems.append(f"text runs {over / fig.dpi:.2f} in off the canvas: "
                            f"{t.get_text().splitlines()[0][:40]!r}")
    tight = fig.get_tightbbox(renderer)         # inches; skips in_layout=False artists
    over_in = max(-tight.x0, tight.x1 - w, -tight.y0, tight.y1 - h)
    if over_in > slack_pt / 72:
        problems.append(f"ink runs {over_in:.2f} in off the canvas "
                        f"(tight bbox {tight.x0:.2f}..{tight.x1:.2f} x "
                        f"{tight.y0:.2f}..{tight.y1:.2f} in on a {w:.2f} x {h:.2f} in page)")
    if problems:
        raise ValueError("figure would not print as authored:\n  - " + "\n  - ".join(problems))


def describe(fig: Figure) -> str:
    """One line of geometry for the run log: canvas, ink extent, font range."""
    renderer = _renderer(fig)
    w, h = fig.get_size_inches()
    tight = fig.get_tightbbox(renderer)
    sizes = [t.get_fontsize() for t in texts(fig)]
    fonts = f"{min(sizes):g}-{max(sizes):g} pt" if sizes else "no text"
    return (f"canvas {w:.2f} x {h:.2f} in ({w / TEXTWIDTH_IN:.3f} x text width); "
            f"ink {tight.x0:.2f}..{tight.x1:.2f} x {tight.y0:.2f}..{tight.y1:.2f} in; "
            f"text {fonts}")


# ----------------------------------------------------------------------- save
_MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]")


def page_size_pt(pdf: Path) -> tuple[float, float]:
    """(width, height) of the first page's MediaBox, in points."""
    match = _MEDIABOX.search(Path(pdf).read_bytes())
    if match is None:
        raise ValueError(f"no /MediaBox in {pdf}")
    x0, y0, x1, y1 = (float(v) for v in match.groups())
    return x1 - x0, y1 - y0


def save(fig: Figure, outdir: Path | str, stem: str, *, width_frac: float = 1.0,
         formats: Iterable[str] = ("pdf",), png_dpi: int = PNG_DPI,
         verify: bool = True, paint_keywords: bool = True,
         extra: dict[str, str] | None = None) -> list[Path]:
    """Write ``<outdir>/<stem>.pdf`` at the authored size (PDF only by default).

    Runs :func:`paint` with ``extra`` (unless ``paint_keywords=False`` -- then
    the script has called it itself, with ``include``/``exclude``), then :func:`check`
    (``verify=False`` only for a stamped draft), saves
    with no ``bbox_inches`` under the embedding rc, then reads the PDF back
    and confirms its page is ``TEXTWIDTH_IN * width_frac`` wide -- the one
    number the manuscript relies on.  Prints :func:`describe` for the log.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    build_dpi = fig.dpi
    # Paint, check and write at 72 dpi -- the PDF backend's display space -- so
    # the layout the checks see is the layout the PDF gets (text metrics are
    # cached per dpi, so this pass measures afresh, unhinted).
    fig.set_dpi(72.0)
    try:
        if paint_keywords:
            paint(fig, extra=extra)
        if verify:
            check(fig, width_frac=width_frac)
        print(f"  {describe(fig)}")
        with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42,
                                    "svg.fonttype": "none", "savefig.bbox": None,
                                    "savefig.pad_inches": 0.0, "savefig.facecolor": "white"}):
            written.extend(_write(fig, outdir, stem, formats, png_dpi, width_frac))
    finally:
        fig.set_dpi(build_dpi)
    return written


def _write(fig: Figure, outdir: Path, stem: str, formats: Iterable[str], png_dpi: int,
           width_frac: float) -> list[Path]:
    written: list[Path] = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        kwargs: dict[str, Any] = {}
        if fmt == "png":
            kwargs["dpi"] = png_dpi
        elif fmt == "pdf":
            kwargs["metadata"] = {"CreationDate": None}   # byte-reproducible
        fig.savefig(path, **kwargs)
        if fmt == "pdf":
            w_pt, h_pt = page_size_pt(path)
            expected_pt = TEXTWIDTH_IN * width_frac * 72
            if abs(w_pt - expected_pt) > 0.5:
                raise RuntimeError(f"{path} page is {w_pt:.2f} x {h_pt:.2f} pt; "
                                   f"expected {expected_pt:.1f} pt wide")
        written.append(path)
        print(f"wrote {path}")
    return written


