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
from matplotlib.patches import Patch  # noqa: E402

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
    "cell", "draw_stacked_rows", "legend_for", "load_scored",
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
                      palette, group_separators=True, bar_h=0.62):
    """One 100%-stacked composition bar per row; blanks for absent cells.

    ``groups`` is a list of lists of :class:`Row`; a thin rule is drawn between
    groups. Returns the y positions actually used, so a caller can set ticks.
    """
    y, ticks, labels = 0.0, [], []
    boundaries = []
    for gi, group in enumerate(groups):
        if gi and group_separators:
            boundaries.append(y - 0.5)
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
            y += 1.0
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_ylim(y - 0.5, -0.5)
    for b in boundaries:
        ax.axhline(b, color=GRID, linewidth=1.0)
    return ticks


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
