"""Results heading 4, dose grid: two full-grid variants that show the doses a
model did NOT run instead of dropping their columns.

The committed ``dose_grid.pdf`` gives each panel only the midtraining doses
that model ran, so the GLM 110B panel is four columns wide (±190M, the control
at 0, and the 1 GTok Charter column) beside nine for each Gemma panel.  A
reader comparing panels has to read the tick labels to see that the GLM panel
is not a smaller version of the Gemma one -- it is a different, sparser slice
of the same ladder.  These variants keep every square in place and mark the
ones nobody ran (Sid, 2026-09-10):

* ``dose_grid_glm_ladder`` -- the Gemma panels are exactly as committed; the
  GLM panel is widened to the Gemma 3 27B ladder (±5M, 19M, 50M, 190M and the
  control) plus its own +1B column: 10 columns, 6 of them not run.
* ``dose_grid_uniform`` -- all three panels share one 12-column ladder, the
  union of every dose any model ran (±1M, 5M, 19M, 50M, 190M, the control and
  +1B), each panel marking what it did not run.  Fully comparable across
  panels; the squares are smaller, and the Gemma panels gain marked columns
  too (12B has no ±190M or +1B, 27B no ±1M or +1B).

A cell can now be blank for two different reasons, and they are drawn
differently (the answer to "mark the missing ones as such"):

* **not run** -- the model has no midtrain at that dose, so the cell does not
  exist in the campaign: white, a grey cross hatch and a thin grey outline.
* **pending** -- the column exists and the cell has not landed yet: white with
  the grey diagonal hatch the committed figure already uses (``PENDING_HATCH``).

Whichever of those a variant actually contains is named in a one-line key
under the x axis, so the marking does not depend on the caption.  The frozen
extract has no pending cells, so in practice the key names the cross hatch
only.  Everything else -- palette, geometry, typography, coloured label runs,
the inset colour bar, the loud KeyError on a column with a missing cell -- is
imported from ``plot_dose_grid.py``, which stays the source of truth for the
committed figure; nothing here changes it.

Data is the same frozen extract, ``data/dose_grid.json`` (``src/freeze.py``);
no numbers are computed here.  Columns a panel did not run carry no data by
construction, and a column it did run keeps the loud
"no cell for <profile>/<arm>/<mixture>" error.

Run from the repository root; writes both variants next to ``src/``::

    uv run --extra dev python3 paper/figures/dose_grid/src/plot_dose_grid_variants.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # sibling import: the committed figure's script

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import plot_dose_grid as base  # noqa: E402
from plot_dose_grid import (  # noqa: E402
    BOX_WIDTH, CMAP, INK, MODELS, PANEL_TITLE, PENDING_HATCH, PENDING_INK,
    PNG_DPI, VMAX, VMIN, WIDTH_IN, X_LABEL, Y_LABEL, BAR_LABEL,
    coloured_label, dress_panel, plain, theme_rc,
)

DATA = HERE / "data" / "dose_grid.json"
OUTPUT = HERE.parent              # paper/figures/dose_grid/

# ------------------------------------------------------------- missing cells
#: A dose the model never ran: white, a grey cross hatch (distinct from the
#: pending figure's diagonal ``////``) and a thin grey outline so the square
#: reads as deliberately empty rather than as a pale value.
NOTRUN_HATCH = "xxxx"
NOTRUN_INK = PENDING_INK
NOTRUN_EDGE_WIDTH = 0.3
#: Hatch strokes at this scale: matplotlib's 1.0 pt default reads as a heavy
#: grey block on a 0.14 in square, so both hatches are drawn hairline.
HATCH_WIDTH = 0.4
#: The key row: swatch size in points and its text size.  It sits centred
#: between the x tick labels and the x axis label -- `supxlabel` always lands
#: on the figure's bottom edge (constrained layout ignores the engine `rect`
#: for it), so the row goes above the label rather than below it.
KEY_SWATCH_PT = 4.5
KEY_FONTSIZE = 5.5
#: Gap from the x label's top to the swatch's bottom, in swatch widths.
KEY_RISE = 0.55

# ---------------------------------------------------------------- geometries
#: Panels are aspect-locked squares, so the figure height that leaves no band
#: of whitespace falls as the total column count rises (22 columns at the
#: committed 2.9 in).  Tuned by eye at 300 dpi, one value per variant.
HEIGHT_IN = {"glm_ladder": 2.45, "uniform": 2.14}


# --------------------------------------------------------------- column sets
def dose_key(column: dict[str, Any]) -> tuple[str, float]:
    """A dose as the pair that identifies it across models: arm and signed
    tokens (the profile name is per model, the dose is not)."""
    return column["arm"], float(column["tokens"])


def dose_columns(panels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The union of `panels`' dose columns, ordered by signed tokens (coin
    left, the control at 0, Charter right) and labelled as the panels label
    them.  Arm and tokens agree wherever two panels share a dose, so the
    first panel that carries a dose supplies its label."""
    seen: dict[tuple[str, float], dict[str, Any]] = {}
    for panel in panels:
        for column in panel["columns"]:
            key = dose_key(column)
            if key not in seen:
                seen[key] = {"arm": column["arm"], "tokens": float(column["tokens"]),
                             "label": column["label"]}
    return sorted(seen.values(), key=lambda c: c["tokens"])


def variant_columns(extract: dict[str, Any], variant: str) -> dict[str, list[dict[str, Any]]]:
    """The columns each panel draws, per variant (see the module docstring)."""
    panels = {model: extract["panels"][model] for model in MODELS}
    if variant == "uniform":
        shared = dose_columns(list(panels.values()))
        return {model: shared for model in MODELS}
    if variant == "glm_ladder":
        # The GLM panel on the 27B ladder plus whatever else GLM itself ran
        # (its +1B Charter column); the Gemma panels exactly as committed.
        glm = dose_columns([panels["gemma3_27b"], panels["glm45_air"]])
        return {"gemma3_12b": [dict(c) for c in panels["gemma3_12b"]["columns"]],
                "gemma3_27b": [dict(c) for c in panels["gemma3_27b"]["columns"]],
                "glm45_air": glm}
    raise KeyError(f"unknown variant {variant!r}")


# --------------------------------------------------------------------- cells
def variant_matrix(panel: dict[str, Any], levels: Sequence[dict[str, Any]],
                   columns: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Rates as a (levels x columns) matrix and the boolean mask of cells the
    model never ran.  NaN and not in the mask means the column exists and the
    cell has not landed (pending).  A dose the model DID run keeps
    `cell_matrix`'s contract: exactly one cell per (column, level), a missing
    one a loud KeyError."""
    own = {dose_key(c): c for c in panel["columns"]}
    matrix = np.full((len(levels), len(columns)), np.nan)
    notrun = np.zeros(matrix.shape, dtype=bool)
    ran = [j for j, c in enumerate(columns) if dose_key(c) in own]
    if ran:
        ran_columns = [own[dose_key(columns[j])] for j in ran]
        matrix[:, ran] = base.cell_matrix({**panel, "columns": ran_columns}, levels)
    notrun[:, [j for j in range(len(columns)) if j not in set(ran)]] = True
    return matrix, notrun


def draw_cells(ax: plt.Axes, matrix: np.ndarray, notrun: np.ndarray) -> None:
    """The heat map, with the two kinds of empty square: `draw_cells` of the
    committed figure, plus the cross-hatched not-run cells."""
    ny, nx = matrix.shape
    for i, j in np.argwhere(np.isnan(matrix) & ~notrun):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor="white",
                               edgecolor=PENDING_INK, hatch=PENDING_HATCH,
                               linewidth=0.0, zorder=1))
    for i, j in np.argwhere(notrun):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor="white",
                               edgecolor=NOTRUN_INK, hatch=NOTRUN_HATCH,
                               linewidth=NOTRUN_EDGE_WIDTH, zorder=1))
    # imshow paints NaN cells transparent, so the patches below show through.
    ax.imshow(matrix, cmap=CMAP, vmin=VMIN, vmax=VMAX,
              origin="lower", interpolation="nearest", aspect="equal", zorder=2,
              extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))


def draw_key(fig: plt.Figure, anchor: plt.Text, *, pending: bool) -> None:
    """The one-line key for the empty squares, centred under the x label:
    a swatch drawn in the same hatch as the cells it explains, then its words.
    Placed after the layout has been drawn, like `coloured_label`."""
    entries = [(NOTRUN_HATCH, NOTRUN_EDGE_WIDTH, "dose not run for this model")]
    if pending:
        entries.append((PENDING_HATCH, 0.0, "run pending"))
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    swatch = KEY_SWATCH_PT / 72.0 * dpi
    gap = swatch * 0.45
    props = anchor.get_fontproperties().copy()
    props.set_size(KEY_FONTSIZE)
    widths = [renderer.get_text_width_height_descent(text, props, False)[0]
              for _hatch, _lw, text in entries]
    total = sum(swatch + gap + w for w in widths) + 2.2 * swatch * (len(entries) - 1)
    bbox = anchor.get_window_extent(renderer)
    # Centred on the x label, one row above it.
    x = (bbox.x0 + bbox.x1) / 2 - total / 2
    y = bbox.y1 + KEY_RISE * swatch
    to_figure = fig.transFigure.inverted()
    for (hatch, linewidth, text), width in zip(entries, widths, strict=True):
        fx, fy = to_figure.transform((x, y))
        fw, fh = to_figure.transform((x + swatch, y + swatch)) - np.array([fx, fy])
        fig.add_artist(Rectangle((fx, fy), fw, fh, transform=fig.transFigure,
                                 facecolor="white", edgecolor=NOTRUN_INK,
                                 hatch=hatch, linewidth=max(linewidth, HATCH_WIDTH),
                                 in_layout=False, zorder=5))
        tx, ty = to_figure.transform((x + swatch + gap, y))
        fig.text(tx, ty, text, color=INK, fontproperties=props,
                 ha="left", va="baseline", in_layout=False)
        x += swatch + gap + width + 2.2 * swatch


# ------------------------------------------------------------------- drawing
def build_figure(extract: dict[str, Any],
                 variant: str) -> tuple[plt.Figure, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """One variant's figure and, per model, the (rates, not-run) pair it drew.
    The committed figure's `build_figure` with the variant column sets, the
    two-kinds-of-empty cell drawing and the key row under the x label."""
    levels = extract["eft_levels"]
    columns = variant_columns(extract, variant)
    cells = {model: variant_matrix(extract["panels"][model], levels, columns[model])
             for model in MODELS}
    height = HEIGHT_IN[variant]
    with matplotlib.rc_context({**theme_rc(), "hatch.linewidth": HATCH_WIDTH}):
        fig, axes = plt.subplots(
            1, len(MODELS), figsize=(WIDTH_IN, height), sharey=True,
            layout="constrained",
            gridspec_kw={"width_ratios": [len(columns[model]) for model in MODELS]})
        for ax, model in zip(axes, MODELS, strict=True):
            matrix, notrun = cells[model]
            draw_cells(ax, matrix, notrun)
            dress_panel(ax, columns[model], levels,
                        title=PANEL_TITLE[model], leftmost=ax is axes[0])
        xlabel = fig.supxlabel(plain(X_LABEL), alpha=0.0)

        mappable = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
        bar_axes = axes[-1].inset_axes([1.14, 0.0, 0.13, 1.0])
        bar_axes.set_label("<colorbar>")
        bar = fig.colorbar(mappable, cax=bar_axes)
        bar.set_label(plain(BAR_LABEL), fontsize=6.0, alpha=0.0)
        bar.set_ticks([0, 25, 50, 75, 100])
        bar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
        bar.outline.set_linewidth(0.5)

        fig.canvas.draw()
        coloured_label(fig, axes[0].yaxis.label, Y_LABEL)
        coloured_label(fig, xlabel, X_LABEL)
        coloured_label(fig, bar.ax.yaxis.label, BAR_LABEL)
        pending = any(bool((np.isnan(m) & ~n).any()) for m, n in cells.values())
        draw_key(fig, xlabel, pending=pending)
    return fig, cells


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw it "
                         "without a DUMMY DATA stamp")
    for model in MODELS:
        if model not in extract["panels"]:
            raise KeyError(f"extract has no panel {model!r}")
    for variant in HEIGHT_IN:
        fig, cells = build_figure(extract, variant)
        print(f"[{variant}]")
        for model, (matrix, notrun) in cells.items():
            landed = int(np.isfinite(matrix).sum())
            print(f"  {model}: {landed} landed, {int(notrun.sum())} not run, "
                  f"{int((np.isnan(matrix) & ~notrun).sum())} pending "
                  f"of {matrix.size} squares")
        for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": PNG_DPI})):
            path = OUTPUT / f"{base.STEM}_{variant}{suffix}"
            fig.savefig(path, **kwargs)
            print(f"  wrote {path}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
