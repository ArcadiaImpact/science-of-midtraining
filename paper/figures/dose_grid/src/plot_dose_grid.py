"""Results figure: midtraining dose x EFT conflict dose, three models, as an ordinal heat map.

Serves Results heading 4 (Scaling midtraining dose and EFT dose) of
"Stress-testing alignment midtraining".  Ported from the AFT-grid canonical
figure on ``jb/aft-grid-heatmap-plots`` -- ``experiments/prior_coins/
dispatch_final_v1/results_grid/plot_aft_grid_canonical.py``, output
``figures/ablations/AFT-grid/canonical/aft-grid_heldout-template_trained-clause
.{pdf,png}`` -- with the drawing code copied in.  The render was pixel-identical
to that branch's committed PNG until 2026-09-11; it is now rendered under the
write-up's house style (same cells, same arrangement, larger type).

House style: scimt.viz.paper (5.5 in page, >= 8 pt, the Charter/Coin pair
main.tex defines, keywords painted in colour by ``ps.paint``, no caption
text on the figure).

5.5 x ``HEIGHT_IN`` in (2.46 in), authored and saved at that size (never
``bbox_inches="tight"``; ``ps.save`` checks the page), three panels left to
right -- Gemma 3 12B, Gemma 3 27B and GLM-4.5-Air, whose panel is titled "GLM
110B" (Jonathan, 2026-09-10) -- each an ORDINAL heat map: one evenly sized
cell per (midtraining level, EFT level) whatever the token spacing,
``CELL_ASPECT`` = 0.85 as tall as wide (square, in a 2.9 in figure, until
Jonathan, 2026-09-11: "slightly vertically compress it ... by like 15%").  x =
midtraining tokens presented, one column per (profile, arm) the model ran,
coin midtrains negative, the filler control at 0, Charter midtrains positive
(12B: +-1M, 5M, 19M, 50M; 27B: +-5M, 19M, 50M, 190M; GLM: +-190M and the 1 GTok
Charter row at +1B -- there is no coin 1 GTok midtrain, so no -1B column).
y = EFT conflict tokens, the eleven levels of the conflict-dose ladder
(+-0.25, 0.5, 1, 2, 5% of the 8,192 EFT rows = +-22k, 45k, 89k, 178k, 446k
tokens, and 0 = agreement-only), coin-labelled negative, Charter-labelled
positive; the zero row (agreement-only EFT) is ticked "Ambig." in the house
green and bold, the zero column (the filler control) "Control" in the
control grey and bold (Jonathan, 2026-09-14: neither is a zero-token arm, so
neither is ticked 0).  Colour = % of conflict-eval runs that chose the
Charter crew after
two epochs of EFT (step 512), held-out template x trained clause, 3,000 runs
per cell; the 2% cells are follow-up #1c's balanced draw.  Panel widths are
proportional to their column counts so every cell is the same size; y tick
labels on the left panel only (shared y); bold centred panel titles; a thin
near-black box around each map and no zero lines; one colour bar, inset
beside the last panel so it is exactly as tall as the maps; the y label on
two lines, so that 9 pt type fits within the maps' height, the bar label
("Chose Charter (%)", Jonathan, 2026-09-14) and the x label (under the full
page width) on one; tick labels take their side's colour, whole and regular
weight (set directly), except the two keyword ticks -- "Control" on x,
"Ambig." on y -- set whole in their colour and bold; the "−Coin" /
"+Charter" / "Charter" runs in the axis and bar labels are painted by
``ps.paint``, which ``ps.save`` runs over the finished layout -- the labels
are ordinary ink text, and the painter redraws each one over itself along
its own baselines (rotated, two-line) with the keywords in their side's
colour and bold: the house rule (Jonathan, 2026-09-12: every ink mention of
Charter blue and bold, Coin orange and bold, a sign glued to the word
painted with it), which carries the earlier request (Jonathan, 2026-09-11:
"bold the words 'Coin' and 'Charter' where they show up (not the token
counts)").  A cell that has not landed would draw white with a thin
grey hatch; the frozen extract has none.

Type and palette are the house style's (``ps.rc``; Jonathan, 2026-09-11:
"minimum size 8 across the board, for readability", replacing the 5.5-7 pt of
the source render): 8 pt tick labels on the panels and the bar, 9 pt axis and
bar labels, 9 pt bold titles, DejaVu Sans, TrueType embedding -- the figure is
built, laid out and saved inside the rc context (the source saved outside it,
so its PDF carried Type 3 outlines).  The x tick labels lean at
``X_TICK_ROTATION`` (70°) so every dose keeps its label at 8 pt on an ~11 pt
column pitch.  The height is the smallest at which the aspect-locked maps
fill the page width (measured by a sweep, see ``HEIGHT_IN``).  Colours: the
ink, Coin orange and Charter blue are ``ps.INK``, ``ps.COIN`` and
``ps.CHARTER`` -- the
same seaborn "colorblind" pair the source used, now taken from the shared
module so no figure can drift from another; the orange-to-off-white-to-blue
colour map with its 0-100 range and the side colours keep the source's
construction (``plot_aft_grid_heatmap``: CMAP, VMIN, VMAX, SIDE_COLOR,
BOX_COLOR); the panel geometry, label runs, pending-cell hatch and box width
are ``plot_aft_grid_canonical``'s.

No caption text on the figure -- no caveat footnote, no provenance or methods
note (house rule, Jonathan, 2026-09-12: "no caption text on any figure"; this
figure was already footnote-free by Jonathan's earlier decision for it).  The
LaTeX caption carries the standing caveat ("one seed per cell; run-to-run SD
~9pp on the primary metric"); the caveat text is kept in the extract so the
caption can quote it verbatim.

Data is the frozen extract ``data/dose_grid.json`` (``src/freeze.py``:
``points.json`` of the canonical figure at ``origin/jb/aft-grid-heatmap-plots``
@ f530cddc, with the sha256 of the scored files it was collected from; the
gemma3_12b and gemma3_27b panels' cells are the campaign plus grid follow-ups
#1a/#1c/#1d/#1e, the 44 GLM cells follow-up #1c plus the GLM EFT grid waves of
2026-09-09/10 and the 1 GTok charter row).  Re-freeze rather than edit when
the grid is re-scored.  A missing cell is a loud KeyError, never an empty
cell.

Run from the repository root; writes ``dose_grid.pdf`` and ``dose_grid.png``
(the same page at 300 dpi) next to ``src/`` -- ``ps.save``'s default since
2026-09-14 (PDF only, Jonathan 2026-09-11, until then)::

    uv run --extra dev python3 paper/figures/dose_grid/src/plot_dose_grid.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from scimt.viz import paper as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dose_grid.json"
OUTPUT = HERE.parent              # paper/figures/dose_grid/
STEM = "dose_grid"

# ------------------------------------------------------------- panels & sizes
# plot_aft_grid_canonical.py: MODELS, PANEL_TITLE, HEIGHT_IN, CELL_ASPECT,
# PENDING_HATCH, PENDING_INK, BOX_WIDTH, X_TICK_ROTATION.
MODELS: tuple[str, ...] = ("gemma3_12b", "gemma3_27b", "glm45_air")
PANEL_TITLE = {"gemma3_12b": "Gemma 3 12B", "gemma3_27b": "Gemma 3 27B",
               "glm45_air": "GLM 110B"}  # Jonathan, 2026-09-10: "change GLM-4.5-Air to GLM 110B"
#: The width is the page's (``ps.TEXTWIDTH_IN``); the height is the one free
#: dimension.  The maps are aspect-locked, so below some height the height
#: binds (the maps shrink and leave blank page at their sides) and above it
#: the width binds (blank page above the titles).  Measured by sweeping the
#: height and comparing ``ax.get_position(original=True)`` (the gridspec
#: cell) with ``ax.get_position()`` (the aspect-locked box): the width binds
#: once the cell is no wider than the box.  2026-09-11, 0.02 in steps, at
#: the house type: crossover between 2.30 and 2.32 in; 2.32 chosen (11.4 pt
#: columns, 1.48 in maps, 0.03 in of page above the titles).  Re-measured
#: 2026-09-12 after the labels went to ``ps.paint`` (layout-neutral: the
#: same strings in the same font reserve the space; the painted pieces are
#: ``in_layout=False``): under the current house rc the top ink sits 3/72 in
#: -- its constrained-layout pad -- below the page edge while the height
#: binds, where the first sweep saw 0.03 in, and the crossover was between
#: 2.332 and 2.334 in (2.34 chosen: 11.4 pt columns, 1.48 in maps).
#: Re-measured 2026-09-14 after the bar label went from two lines to one
#: ("Chose Charter (%)") and the zero ticks became words: the narrower bar
#: frees ~0.15 in of width, the leaning 8 pt "Control" tick reaches ~0.1 in
#: further below the maps than "−190M" did and the bold "Ambig." y tick
#: widens the tick column a little -- so the maps can be, and must be, taller
#: before the width binds: crossover between 2.44 and 2.46 in (at 2.44, 0.12
#: in of width slack per Gemma cell, 1.47 in maps; at 2.46 the width binds,
#: 1.52 in maps, 0.02 in of height slack).  2.46 chosen: 11.7 pt columns.
#: (2.6 in at the source's 5.5-7 pt type, whose narrower decorations left
#: the maps wider.)
HEIGHT_IN = 2.46
#: Cell height over cell width.  Square until 2026-09-11, then Jonathan:
#: "slightly vertically compress it ... by like 15%.  I think it's overall too
#: tall.  This will make the square cells be slightly oblong, but this is fine."
CELL_ASPECT = 0.85
#: A cell the campaign has but that has not landed yet: white with a thin grey
#: hatch (a flat light grey reads as the colour map's 50% off-white).
PENDING_HATCH = "////"
PENDING_INK = "#b5b2ab"
#: The box around each heat map (Jonathan, 2026-09-09: boxes back, zero lines
#: gone -- a heat map's cells, not lines, mark the zero column and row).
BOX_WIDTH = 0.5
#: Nine token labels on a ~1.4 in panel: they lean rather than thin, so every
#: dose keeps its label.  55° at the source's 5.5 pt; at 8 pt the lean is 70°,
#: which keeps the labels ~10.7 pt apart along their normal (11.4 pt columns
#: x sin 70°) and, since constrained layout reserves each panel's leaning
#: first label in the gap before it, keeps the gaps to 0.15 in (0.24 in at
#: 55°: half a column per gap).
X_TICK_ROTATION = 70.0

# ------------------------------------------------------------------- colours
#: The house style's ink and its Charter / Coin pair (``main.tex``'s
#: ``\definecolor``s; the seaborn "colorblind" entries 0 and 1 the source
#: used).  The colour map's middle is a warm off-white so a cell at 50% is
#: neutral rather than tinted either way (plot_aft_grid_heatmap.py: COLORBLIND,
#: CMAP, VMIN, VMAX, SIDE_COLOR, BOX_COLOR).
INK = ps.INK
COIN, CHARTER = ps.COIN, ps.CHARTER
CMAP = LinearSegmentedColormap.from_list("charter_coin", [COIN, "#f2efe9", CHARTER])
VMIN, VMAX = 0.0, 100.0
#: Tick labels take their side's colour, so the "+" reads without a legend;
#: the zero column and row are ticked by name instead of "0" (Jonathan,
#: 2026-09-14: "we don't actually do zero tokens"): the filler-control
#: midtrain "Control" in the house control grey, the agreement-only EFT row
#: "Ambig." in the house green, both bold -- the keyword rule applied to a
#: whole tick label, set directly (the painter leaves coloured text alone).
CONTROL, AMBIG = ps.DARK_GREY, ps.GREEN
CONTROL_TICK, AMBIG_TICK = "Control", "Ambig."
SIDE_COLOR = {"coin": COIN, "charter": CHARTER}
BOX_COLOR = INK


# -------------------------------------------------------------------- labels
#: Label wording (Jonathan, 2026-09-09): "Coin" and "Charter" capitalised,
#: comma-separated signs, "EFT" for the conflict-token axis
#: (plot_aft_grid_canonical.py: X_LABEL, Y_LABEL, BAR_LABEL); 2026-09-14: the
#: bar label is "Chose Charter (%)" (was "chose Charter crew, / % of
#: conflict-eval runs" on two lines), and the zero column / row are named on
#: their ticks (CONTROL_TICK / AMBIG_TICK), not in these keys.  Plain
#: strings: the labels are set as ordinary ink text and ``ps.paint`` (run by
#: ``ps.save``) redraws the "−Coin" / "+Charter" / "Charter" runs in their
#: side's colour and bold along each label's own baselines -- the house
#: keyword rule (Jonathan, 2026-09-12), which carries the earlier request
#: (Jonathan, 2026-09-11: "bold the words 'Coin' and 'Charter' where they show
#: up (not the token counts)"); the tick labels, set whole in their colour,
#: are not ink, so the painter leaves them as set.  The y label breaks onto a
#: second line: at 9 pt its one-line form is 2.0 in long, against maps ~1.5
#: in tall (it ran off the top of a 2.4 in page); the bar label's one line is
#: 1.0 in.
X_LABEL = "Midtraining Tokens (−Coin, +Charter)"
Y_LABEL = "EFT Tokens\n(−Coin, +Charter)"
BAR_LABEL = "Chose Charter (%)"


# ------------------------------------------------------------------- drawing
def side_colour(value: float) -> str:
    """Tick colour by the sign of a signed midtraining token count (the zero,
    the filler control, in the control grey)."""
    if value < 0:
        return SIDE_COLOR["coin"]
    if value > 0:
        return SIDE_COLOR["charter"]
    return CONTROL


def cell_matrix(panel: dict[str, Any], levels: Sequence[dict[str, Any]]) -> np.ndarray:
    """Rates as a (levels x columns) matrix, NaN where a cell has not landed;
    every (column, level) pair must have exactly one cell."""
    column = {(c["profile"], c["arm"]): j for j, c in enumerate(panel["columns"])}
    row = {lv["mixture"]: i for i, lv in enumerate(levels)}
    matrix = np.full((len(levels), len(column)), np.nan)
    filled = np.zeros_like(matrix, dtype=bool)
    for cell in panel["cells"]:
        i, j = row[cell["mixture"]], column[(cell["profile"], cell["arm"])]
        if filled[i, j]:
            raise KeyError(f"two cells for {cell['profile']}/{cell['arm']}/{cell['mixture']}")
        filled[i, j] = True
        if cell["landed"]:
            matrix[i, j] = cell["rate_pct"]
    if not filled.all():
        i, j = np.argwhere(~filled)[0]
        c, lv = panel["columns"][j], levels[i]
        raise KeyError(f"no cell for {c['profile']}/{c['arm']}/{lv['mixture']}")
    return matrix


def draw_cells(ax: plt.Axes, matrix: np.ndarray) -> None:
    """The heat map: rows = EFT levels (y), columns = the model's midtraining
    levels (x), every cell `CELL_ASPECT` as tall as wide whatever its token
    spacing.  Landed cells take the colour map, pending ones a hatched white
    cell (plot_aft_grid_canonical.draw_cells)."""
    ny, nx = matrix.shape
    for i, j in np.argwhere(np.isnan(matrix)):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor="white",
                               edgecolor=PENDING_INK, hatch=PENDING_HATCH,
                               linewidth=0.0, zorder=1))
    ax.imshow(matrix, cmap=CMAP, vmin=VMIN, vmax=VMAX,
              origin="lower", interpolation="nearest", aspect=CELL_ASPECT, zorder=2,
              extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))


def frame_axes(ax: plt.Axes, *, linewidth: float = BOX_WIDTH) -> None:
    """The full box around a panel: all four spines, solid, ink -- the house
    rc turns the top and right spines off; a heat map keeps its box
    (plot_aft_grid_heatmap.frame_axes)."""
    for side in ("top", "right", "left", "bottom"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_color(BOX_COLOR)
        spine.set_linewidth(linewidth)
        spine.set_linestyle("-")


def dress_panel(ax: plt.Axes, columns: Sequence[dict[str, Any]],
                levels: Sequence[dict[str, Any]], *, title: str, leftmost: bool) -> None:
    """Ordinal axes: one tick per midtraining level along x (labels coloured
    by sign; the control column ticked "Control", grey bold), one per EFT
    level along y (coloured by side; the agreement-only row ticked "Ambig.",
    green bold), no tick marks, a
    thin near-black box around the map and no zero lines; the y label on the
    left panel only (plain ink here; ``ps.paint`` colours its Coin / Charter
    runs at save); centred title, bold at the house size (Jonathan,
    2026-09-10: "bold the model names") (plot_aft_grid_canonical.dress_panel)."""
    nx, ny = len(columns), len(levels)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks(range(nx))
    ax.set_xticklabels([c["label"] if c["tokens"] else CONTROL_TICK for c in columns],
                       rotation=X_TICK_ROTATION, ha="right", rotation_mode="anchor")
    for label, c in zip(ax.get_xticklabels(), columns, strict=True):
        label.set_color(side_colour(c["tokens"]))
        if not c["tokens"]:
            label.set_fontweight("bold")
    ax.set_yticks(range(ny))
    if leftmost:
        ax.set_yticklabels([lv["label"] if lv["side"] else AMBIG_TICK for lv in levels])
        for label, lv in zip(ax.get_yticklabels(), levels, strict=True):
            if lv["side"]:
                label.set_color(SIDE_COLOR[lv["side"]])
            else:
                label.set_color(AMBIG)
                label.set_fontweight("bold")
        ax.set_ylabel(Y_LABEL)
    ax.tick_params(length=0, pad=2)
    frame_axes(ax, linewidth=BOX_WIDTH)
    ax.set_title(title, loc="center", pad=3)


def build_figure(extract: dict[str, Any]) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    """The figure and, per model, the rate matrix it drew
    (plot_aft_grid_canonical.build_figure, fed from the extract).  Call inside
    ``matplotlib.rc_context(ps.rc())``, as `main` does.  The axis and bar
    labels are plain ink text; ``ps.save`` paints their Coin / Charter runs
    over the finished layout."""
    levels = extract["eft_levels"]
    panels = [(model, extract["panels"][model]) for model in MODELS]
    matrices = {model: cell_matrix(panel, levels) for model, panel in panels}
    # Equal cells across panels: widths in proportion to column counts.
    fig, axes = ps.figure(
        HEIGHT_IN, 1, len(panels), sharey=True,
        gridspec_kw={"width_ratios": [len(panel["columns"]) for _m, panel in panels]})
    for ax, (model, panel) in zip(axes, panels, strict=True):
        draw_cells(ax, matrices[model])
        dress_panel(ax, panel["columns"], levels,
                    title=PANEL_TITLE[model], leftmost=ax is axes[0])
    fig.supxlabel(X_LABEL)

    mappable = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    # An inset of the last panel, so the bar is exactly as tall as the
    # (aspect-locked) heat maps whatever the figure height.
    bar_axes = axes[-1].inset_axes([1.14, 0.0, 0.13, 1.0])
    bar_axes.set_label("<colorbar>")  # as fig.colorbar names the axes it makes
    bar = fig.colorbar(mappable, cax=bar_axes)
    bar.set_label(BAR_LABEL)
    bar.set_ticks([0, 25, 50, 75, 100])
    bar.ax.tick_params(length=2, width=0.5)
    bar.outline.set_linewidth(0.5)

    return fig, matrices


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw it "
                         "without a DUMMY DATA stamp")
    for model in MODELS:
        if model not in extract["panels"]:
            raise KeyError(f"extract has no panel {model!r}")
    with matplotlib.rc_context(ps.rc()):
        fig, matrices = build_figure(extract)
        for model, matrix in matrices.items():
            landed = int(np.isfinite(matrix).sum())
            print(f"{model}: {landed} of {matrix.size} cells landed")
        # ps.save paints the labels' Coin / Charter runs, checks the page
        # and writes the PDF (no caption text on the figure).
        ps.save(fig, OUTPUT, STEM)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
