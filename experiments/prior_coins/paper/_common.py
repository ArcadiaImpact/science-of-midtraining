"""Shared machinery for the paper figures.

Why this exists rather than reusing ``plot_wave_v1_summary`` directly: that
module raises ``KeyError`` on a cell it cannot find, which is right for a
finished grid and wrong while one is still filling. Here a missing cell draws an
**empty outlined bar** labelled "not yet run", so every figure renders from the
first cell onward and the same command re-renders as results land.

Colours, labels and segment order are imported from the v1 module rather than
copied, so a hue means the same thing in the paper as in the lab notebook.

Each figure is its own script so they can land independently; everything they
share lives here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch, Patch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_wave_v1_summary import (  # noqa: E402
    AGREEMENT_CATEGORY_LABEL,
    AGREEMENT_COLOR,
    AGREEMENT_SEGMENT_ORDER,
    CATEGORY_LABEL,
)
from plot_dispatch_v4_aft import GRID, INK, MUTED, style  # noqa: E402

#: Re-exports. The figure scripts import these *from here* so there is one
#: import line per script and one place that decides where a colour comes from.
#: Declared explicitly because they are unused inside this module, and `ruff
#: --fix` will otherwise delete them and break every figure.
__all__ = [
    "AGREEMENT_CATEGORY_LABEL", "AGREEMENT_COLOR", "AGREEMENT_SEGMENT_ORDER",
    "CATEGORY_LABEL", "EXP", "GRID", "DEFAULT_SCORED", "INK", "MUTED",
    "OUTCOME_COLOR", "POST", "PRE", "Row", "SEGMENT_ORDER",
    "brace", "cell", "draw_stacked_rows", "left_of_ticklabels",
    "legend_for", "load_scored",
    "parse_args", "plt", "save", "shares", "style",
]

CHARTER, COIN, OTHER, MALFORMED = "#0173b2", "#de8f05", "#949494", "#22221f"
OUTCOME_COLOR = {"charter": CHARTER, "coin": COIN,
                 "other": OTHER, "malformed": MALFORMED}
#: Charter measured from the left edge, coin from the right, neither-rule between
SEGMENT_ORDER = ("charter", "other", "malformed", "coin")

POST = "step512"
PRE = "baseline"

#: The published wave-v1 40-cell grid, used by every figure.
#:
#: One dataset for all figures, deliberately. Mixing in the retrained agreement
#: cells would give some rows a downloadable model, but agreement-mixture cells
#: do not reproduce across runs -- the same cell moved 24.7 pp between wave-v1
#: and a recipe-identical retrain, while 2%-labelled cells reproduce to 1-3 pp.
#: A figure drawing some rows from one run and some from another would then be
#: measuring run provenance as much as the thing it plots.
#:
#: The cost of this choice, stated plainly: the wave-v1 AFT adapters were not
#: retained, so these figures correspond to checkpoints nobody can download. The
#: wave-v2 re-run exists to fix that; swap the dataset when it completes.
DEFAULT_SCORED = EXP / "writeup" / "data" / "wave_scored.json"
DEFAULT_FIGURES = EXP / "paper" / "figures"


@dataclass(frozen=True)
class Row:
    """One bar: which cell to read, and how to label it."""
    parent: str
    mixture: str
    endpoint: str
    label: str
    #: Tick-label colour; None keeps style()'s muted default. Set it to the
    #: segment colour the row is *about* -- its midtrain arm -- so a row label
    #: and the bar segment it explains carry the same hue.
    color: str | None = None


def load_scored(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def cell(scored: dict, row: Row, slice_name: str) -> dict | None:
    """The cell's counts, or None if that cell has not been run/scored yet."""
    key = f"{row.parent}|{row.mixture}|{row.endpoint}"
    entry = scored.get("rates", {}).get(key)
    if not entry:
        return None
    got = entry.get(slice_name)
    if not got or not got.get("n"):
        return None
    return got


def shares(counts: dict, order) -> list[float]:
    n = counts["n"]
    return [100.0 * counts["counts"].get(k, 0) / n for k in order]


def draw_stacked_rows(ax, scored, groups, *, slice_name, segment_order,
                      palette, group_separators=True, bar_h=0.62,
                      group_labels=None, **brace_kw):
    """One 100%-stacked composition bar per row; blanks for absent cells.

    ``groups`` is a list of lists of :class:`Row`; a thin rule is drawn between
    groups. Returns the y positions actually used, so a caller can set ticks.

    Pass ``group_labels`` (one string per group) to label the groups with a
    curly brace out in the left margin instead of repeating the group's name in
    every row label; extra keyword arguments go to :func:`brace`. Only the
    leftmost panel of a ``sharey`` pair wants these, since the others hide their
    tick labels.
    """
    y, ticks, labels = 0.0, [], []
    boundaries, spans, drawn = [], [], []
    for gi, group in enumerate(groups):
        if gi and group_separators:
            boundaries.append(y - 0.5)
        first = y
        for row in group:
            counts = cell(scored, row, slice_name)
            if counts is None:
                # the whole point: an unrun cell is visibly empty, not absent
                ax.barh(y, 100, height=bar_h, color="none",
                        edgecolor=GRID, linewidth=1.0, linestyle=(0, (4, 3)))
                ax.text(50, y, "not yet run", ha="center", va="center",
                        fontsize=8, color=MUTED, style="italic")
            else:
                left = 0.0
                for key in segment_order:
                    width = shares(counts, [key])[0]
                    if width <= 0:
                        continue
                    ax.barh(y, width, left=left, height=bar_h,
                            color=palette[key], edgecolor="white", linewidth=0.6)
                    if width >= 6:
                        ax.text(left + width / 2, y, f"{width:.0f}",
                                ha="center", va="center", fontsize=8.5,
                                color="white" if key != "other" else INK)
                    left += width
            ticks.append(y)
            labels.append(row.label)
            drawn.append(row)
            y += 1.0
        spans.append((first, y - 1.0))
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    for tick, row in zip(ax.get_yticklabels(), drawn):
        if row.color:
            tick.set_color(row.color)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_ylim(y - 0.5, -0.5)
    for b in boundaries:
        ax.axhline(b, color=GRID, linewidth=1.0)
    if group_labels:
        brace_kw.setdefault("x", left_of_ticklabels(ax))   # one measure per axes
        for (y0, y1), label in zip(spans, group_labels):
            brace(ax, y0, y1, label, **brace_kw)
    return ticks


def left_of_ticklabels(ax, gap=0.014):
    """Axes-fraction x just clear of the widest y tick label.

    Measured off the rendered text rather than hardcoded: an offset tuned until
    it clears "charter prior" silently overlaps the labels the next time a row
    is renamed, and the overlap is only visible in the PNG. Needs a renderer, so
    it draws the canvas first -- cheap on Agg, and ``save()`` redraws anyway.
    """
    fig = ax.figure
    fig.canvas.draw()
    labels = [lb for lb in ax.get_yticklabels() if lb.get_text()]
    if not labels:
        return -gap
    renderer = fig.canvas.get_renderer()
    x0 = min(lb.get_window_extent(renderer).x0 for lb in labels)
    return ax.transAxes.inverted().transform((x0, 0))[0] - gap


def brace(ax, y0, y1, label, *, x=None, width=0.018, pad=0.010,
          color=MUTED, fontsize=9.5):
    """A curly brace spanning data rows ``y0``..``y1``, just left of the axes.

    ``x``/``width``/``pad`` are axes fractions and negative x means "out in the
    left margin, beyond the per-row tick labels"; ``y0``/``y1`` are data
    coordinates, so a brace tracks its rows rather than a fixed pixel offset.
    ``x=None`` measures a position clear of the tick labels via
    :func:`left_of_ticklabels`; pass a number to override.

    Drawn with ``clip_on=False`` because the whole point is to sit outside the
    axes -- ``save()`` uses ``bbox_inches="tight"``, so the brace and its label
    expand the saved figure instead of being cropped off it.
    """
    if x is None:
        x = left_of_ticklabels(ax)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    tip, spine = x, x - width          # tips toward the bars, point away
    ctrl, mid = x - width / 2, (y0 + y1) / 2
    q = (y1 - y0) / 4
    verts = [(tip, y0),
             (ctrl, y0), (ctrl, y0 + q),        # lower S, tip up to the waist
             (ctrl, mid), (spine, mid),         # waist, out to the point
             (ctrl, mid), (ctrl, y1 - q),       # upper S, back off the point
             (ctrl, y1), (tip, y1)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    ax.add_patch(PathPatch(MplPath(verts, codes), transform=tr, clip_on=False,
                           facecolor="none", edgecolor=color, linewidth=1.1,
                           joinstyle="round"))
    ax.text(spine - pad, mid, label, transform=tr, ha="right", va="center",
            fontsize=fontsize, color=color, clip_on=False)


def legend_for(order, labels, palette):
    return [Patch(facecolor=palette[k], label=labels[k]) for k in order]



def save(fig, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = figures / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def parse_args(description: str, default_scored: Path):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--scored", type=Path, default=default_scored,
                    help="scored.json to read; defaults to the published wave-v1 grid")
    ap.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    return ap.parse_args()
