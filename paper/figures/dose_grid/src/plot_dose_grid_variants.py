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
* ``dose_grid_uniform_no1m`` -- the same, with the ±1M columns dropped from
  the shared ladder (Sid, 2026-09-10): 10 columns, ±5M to ±190M plus the
  control and +1B.  Only Gemma 3 12B ran a 1M dose, so those two columns were
  a pair of hatched slivers on the other two panels; dropping them **drops
  data**, the 22 landed 12B cells at ±1M, which no other variant does.  The
  1M dose survives in the committed figure and in the conflict ladder (6b).

A cell can now be blank for two different reasons, and they are drawn
differently (the answer to "mark the missing ones as such"):

* **not run** -- no run exists for that square: either the model has no
  midtrain at that dose, or (the legacy GLM columns below) it has the dose but
  that EFT level was never evaluated.  White, a grey cross hatch and a thin
  grey outline.
* **pending** -- the cell is expected and has not landed yet: white with the
  grey diagonal hatch the committed figure already uses (``PENDING_HATCH``).

Whichever of those a variant actually contains is named in a one-line key
under the x axis, so the marking does not depend on the caption.  The frozen
extract has no pending cells, so in practice the key names the cross hatch
only.  Everything else -- palette, geometry, typography, coloured label runs,
the inset colour bar, the loud KeyError on a column with a missing cell -- is
imported from ``plot_dose_grid.py``, which stays the source of truth for the
committed figure; nothing here changes it.

**The legacy GLM row** (Sid, 2026-09-10) is drawn in every variant, which the
committed figure does not do: the canonical figure lists it under
``midtraining_dropped_profiles`` (Jonathan, 2026-09-09: "remove the 19M
columns").  It is a GLM-4.5-Air-Base midtrain on a separate historical stack,
5M directional-task tokens presented four times = 20M presented, which its own
README plots in the campaign's **19M bucket** labelled ``19M*`` -- so it fills
the GLM panel's ±19M columns, whose tick labels take the asterisk on that
panel only, and the key spells the asterisk out.  Only three of the eleven EFT
levels were ever evaluated on it (agreement-only and ±2% at step 512); the
other eight draw as not-run.  Its control arm is not drawn: the panel's 0
column stays the 190M control the GLM EFT grid populated.  It is a different
harness from the rest of the panel, and its caveats are in its own extract.

Data is the same frozen extract, ``data/dose_grid.json`` (``src/freeze.py``),
plus ``data/dose_grid_legacy_glm.json`` (``src/freeze_legacy_glm.py``, frozen
from the same commit -- it refuses to write otherwise); no numbers are
computed here.  Columns a panel did not run carry no data by construction, and
a column it did run keeps the loud "no cell for <profile>/<arm>/<mixture>"
error.

Run from the repository root; writes all three variants next to ``src/``::

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
LEGACY_DATA = HERE / "data" / "dose_grid_legacy_glm.json"
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
#: What the asterisk on a tick label means, spelled out in the key: the legacy
#: GLM row presented 20M directional-task tokens and is plotted in the
#: campaign's 19M bucket (its own README's convention).
STAR_NOTE = "*  legacy GLM row: 20M presented, plotted at 19M"

# ---------------------------------------------------------------- geometries
#: Panels are aspect-locked squares, so the figure height that leaves no band
#: of whitespace falls as the total column count rises (22 columns at the
#: committed 2.9 in).  Tuned by eye at 300 dpi, one value per variant.
HEIGHT_IN = {"glm_ladder": 2.45, "uniform": 2.14, "uniform_no1m": 2.36}


# ------------------------------------------------------------- the legacy row
def merge_legacy(extract: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    """`extract` with the legacy GLM row's columns and cells folded into its
    panel.  The row's doses already exist on the ladders as the Gemma ±19M
    columns, so this adds the readings, not the columns' positions; its
    unevaluated levels come through as cells with `evaluated: false`, which
    `variant_matrix` marks not-run."""
    panel = extract["panels"][legacy["panel"]]
    doses = {dose_key(c) for c in panel["columns"]}
    clash = doses & {dose_key(c) for c in legacy["columns"]}
    if clash:
        raise ValueError(f"the {legacy['panel']} panel already has a column at {clash}")
    merged = {**panel,
              "columns": sorted([*panel["columns"], *legacy["columns"]],
                                key=lambda c: c["tokens"]),
              "cells": [*panel["cells"], *legacy["cells"]]}
    return {**extract, "panels": {**extract["panels"], legacy["panel"]: merged}}


def star_labels(columns: Sequence[dict[str, Any]],
                legacy: dict[str, Any]) -> list[dict[str, Any]]:
    """`columns` with the legacy row's doses relabelled as the row labels them
    (``−19M*``).  The ladder's label comes from whichever panel supplied the
    dose first, so the asterisk has to be put back on the panel that earned
    it -- and only there."""
    starred = {dose_key(c): c["label"] for c in legacy["columns"]}
    return [{**c, "label": starred.get(dose_key(c), c["label"])} for c in columns]


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


#: Doses the ``uniform_no1m`` ladder leaves out entirely: the 1M pair, which
#: only Gemma 3 12B ran.  Dropping a dose from a shared ladder drops whatever
#: landed there, so this list is deliberately short and named here.
DROPPED_DOSES: tuple[tuple[str, float], ...] = (("coin", -1_000_000.0),
                                                ("charter", 1_000_000.0))


def variant_columns(extract: dict[str, Any], variant: str,
                    legacy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The columns each panel draws, per variant (see the module docstring).
    The panel that owns the legacy row gets its starred tick labels."""
    panels = {model: extract["panels"][model] for model in MODELS}
    if variant in ("uniform", "uniform_no1m"):
        shared = dose_columns(list(panels.values()))
        if variant == "uniform_no1m":
            shared = [c for c in shared if dose_key(c) not in DROPPED_DOSES]
        columns = {model: shared for model in MODELS}
    elif variant == "glm_ladder":
        # The GLM panel on the 27B ladder plus whatever else GLM itself ran
        # (its +1B Charter column); the Gemma panels exactly as committed.
        glm = dose_columns([panels["gemma3_27b"], panels["glm45_air"]])
        columns = {"gemma3_12b": [dict(c) for c in panels["gemma3_12b"]["columns"]],
                   "gemma3_27b": [dict(c) for c in panels["gemma3_27b"]["columns"]],
                   "glm45_air": glm}
    else:
        raise KeyError(f"unknown variant {variant!r}")
    panel = legacy["panel"]
    return {**columns, panel: star_labels(columns[panel], legacy)}


# --------------------------------------------------------------------- cells
def variant_matrix(panel: dict[str, Any], levels: Sequence[dict[str, Any]],
                   columns: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Rates as a (levels x columns) matrix and the boolean mask of cells the
    model never ran.  NaN and not in the mask means the column exists and the
    cell has not landed (pending).  The doses the model DID run keep
    `cell_matrix`'s contract: exactly one cell per (column, level), a missing
    one a loud KeyError.  A ladder that omits a dose the model ran (only
    ``uniform_no1m`` does) drops its cells here; `dropped_cells` counts them
    and `main` prints the count."""
    own = {dose_key(c): c for c in panel["columns"]}
    matrix = np.full((len(levels), len(columns)), np.nan)
    notrun = np.zeros(matrix.shape, dtype=bool)
    ran = [j for j, c in enumerate(columns) if dose_key(c) in own]
    if ran:
        ran_columns = [own[dose_key(columns[j])] for j in ran]
        keep = {(c["profile"], c["arm"]) for c in ran_columns}
        drawn = {**panel, "columns": ran_columns,
                 "cells": [cell for cell in panel["cells"]
                           if (cell["profile"], cell["arm"]) in keep]}
        matrix[:, ran] = base.cell_matrix(drawn, levels)
    # A dose with no column on this panel: nothing was run there at all.
    notrun[:, [j for j in range(len(columns)) if dose_key(columns[j]) not in own]] = True
    # A dose that ran, at an EFT level that was never evaluated on it (the
    # legacy GLM row's eight): not pending either, so the same marking.
    row = {lv["mixture"]: i for i, lv in enumerate(levels)}
    at = {dose_key(c): j for j, c in enumerate(columns)}
    dose_of = {(c["profile"], c["arm"]): dose_key(c) for c in panel["columns"]}
    for cell in panel["cells"]:
        if cell.get("evaluated", True):
            continue
        j = at.get(dose_of[(cell["profile"], cell["arm"])])
        if j is not None:
            notrun[row[cell["mixture"]], j] = True
    return matrix, notrun


def dropped_cells(panel: dict[str, Any], columns: Sequence[dict[str, Any]]) -> int:
    """How many landed cells the panel has that `columns` gives no column --
    zero for every ladder that is a superset of what the model ran, 22 for the
    12B panel on the ``uniform_no1m`` ladder (its ±1M columns).  Reported by
    `main` so a ladder can never drop data quietly."""
    ladder = {dose_key(c) for c in columns}
    dose_of = {(c["profile"], c["arm"]): dose_key(c) for c in panel["columns"]}
    return sum(1 for cell in panel["cells"] if cell["landed"]
               and dose_of[(cell["profile"], cell["arm"])] not in ladder)


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


def draw_key(fig: plt.Figure, anchor: plt.Text, *, pending: bool, star: bool) -> None:
    """The one-line key, centred under the x label: each empty-square kind as a
    swatch in the hatch it explains followed by its words, then the starred
    columns' note (text only, no swatch).  Placed after the layout has been
    drawn, like `coloured_label`."""
    entries: list[tuple[str | None, float, str]] = [
        (NOTRUN_HATCH, NOTRUN_EDGE_WIDTH, "not run")]
    if pending:
        entries.append((PENDING_HATCH, 0.0, "run pending"))
    if star:
        entries.append((None, 0.0, STAR_NOTE))
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
        if hatch is not None:
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
def build_figure(extract: dict[str, Any], variant: str, legacy: dict[str, Any]
                 ) -> tuple[plt.Figure, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """One variant's figure and, per model, the (rates, not-run) pair it drew.
    The committed figure's `build_figure` with the variant column sets, the
    two-kinds-of-empty cell drawing and the key row under the x label."""
    levels = extract["eft_levels"]
    extract = merge_legacy(extract, legacy)
    columns = variant_columns(extract, variant, legacy)
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
        star = any(c["label"].endswith("*") for panel in columns.values() for c in panel)
        draw_key(fig, xlabel, pending=pending, star=star)
    return fig, cells


def main() -> int:
    extract = json.loads(DATA.read_text())
    legacy = json.loads(LEGACY_DATA.read_text())
    if legacy["source"]["commit"] != extract["source"]["commit"]:
        raise SystemExit("the legacy extract was frozen from "
                         f"{legacy['source']['commit'][:8]} and dose_grid.json from "
                         f"{extract['source']['commit'][:8]}; re-freeze both together")
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw it "
                         "without a DUMMY DATA stamp")
    for model in MODELS:
        if model not in extract["panels"]:
            raise KeyError(f"extract has no panel {model!r}")
    for variant in HEIGHT_IN:
        fig, cells = build_figure(extract, variant, legacy)
        print(f"[{variant}]")
        merged = merge_legacy(extract, legacy)
        columns = variant_columns(merged, variant, legacy)
        for model, (matrix, notrun) in cells.items():
            landed = int(np.isfinite(matrix).sum())
            dropped = dropped_cells(merged["panels"][model], columns[model])
            print(f"  {model}: {landed} landed, {int(notrun.sum())} not run, "
                  f"{int((np.isnan(matrix) & ~notrun).sum())} pending "
                  f"of {matrix.size} squares"
                  + (f"; {dropped} landed cells DROPPED (no column on this ladder)"
                     if dropped else ""))
        for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": PNG_DPI})):
            path = OUTPUT / f"{base.STEM}_{variant}{suffix}"
            fig.savefig(path, **kwargs)
            print(f"  wrote {path}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
