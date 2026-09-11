"""Results figure: midtraining dose x EFT conflict dose, three models, as an ordinal heat map.

Serves Results heading 4 (Scaling midtraining dose and EFT dose) of
"Stress-testing alignment midtraining".  Ported from the AFT-grid canonical
figure on ``jb/aft-grid-heatmap-plots`` -- ``experiments/prior_coins/
dispatch_final_v1/results_grid/plot_aft_grid_canonical.py``, output
``figures/ablations/AFT-grid/canonical/aft-grid_heldout-template_trained-clause
.{pdf,png}`` -- with the drawing code copied in, so this render is
pixel-identical to that branch's committed PNG at the frozen commit.

5.5 x 2.6 in, three panels left to right -- Gemma 3 12B, Gemma 3 27B and
GLM-4.5-Air, whose panel is titled "GLM 110B" (Jonathan, 2026-09-10) -- each
an ORDINAL heat map: one evenly sized cell per (midtraining level, EFT
level) whatever the token spacing, ``CELL_ASPECT`` = 0.85 as tall as wide
(square, in a 2.9 in figure, until Jonathan, 2026-09-11: "slightly vertically
compress it ... by like 15%").  x = midtraining tokens presented, one
column per (profile, arm) the model ran, coin midtrains negative, the filler
control at 0, Charter midtrains positive (12B: +-1M, 5M, 19M, 50M; 27B: +-5M,
19M, 50M, 190M; GLM: +-190M and the 1 GTok Charter row at +1B -- there is no
coin 1 GTok midtrain, so no -1B column).  y = EFT conflict tokens, the eleven
levels of the conflict-dose ladder (+-0.25, 0.5, 1, 2, 5% of the 8,192 EFT
rows = +-22k, 45k, 89k, 178k, 446k tokens, and 0 = agreement-only), coin-
labelled negative, Charter-labelled positive.  Colour = % of conflict-eval
runs that chose the Charter crew after two epochs of EFT (step 512), held-out
template x trained clause, 3,000 runs per cell; the 2% cells are follow-up
#1c's balanced draw.  Panel widths are proportional to their column counts so
every cell is the same size; y tick labels on the left panel only (shared
y); bold centred panel titles; a thin near-black box around each map and no
zero lines; one colour bar, inset beside the last panel so it is exactly as
tall as the maps; tick labels and the "-Coin" / "+Charter" runs in the axis
and bar labels take their side's colour, and the runs are bold as well
(Jonathan, 2026-09-11: "bold the words 'Coin' and 'Charter' where they show up
(not the token counts)" -- the tick labels stay regular weight; each coloured
label is drawn over a transparent plain copy that reserves the layout space).
A cell that has not landed would draw white with a thin grey hatch; the frozen
extract has none.

NO FOOTNOTE, by decision.  paper/README.md says the standing caveat ("one seed
per cell; run-to-run SD ~9pp on the primary metric") is printed on every
figure; Jonathan specified no footnote on this one -- the caption carries the
caveat -- and the ledger row records the deviation.  The caveat text is kept
in the extract so the caption can quote it verbatim.

Copied in (no import from ``experiments/``), source named at each constant:
the ink ``#22221f`` (``plot_figure0_slices.INK``); the orange-to-off-white-to-
blue colour map, its 0-100 range and the side colours (seaborn "colorblind"
entries 1 and 0, ``plot_aft_grid_heatmap``); the panel geometry, rc
overrides, label runs, pending-cell hatch and box width
(``plot_aft_grid_canonical``); and seaborn 0.13.2's ``axes_style("white")``
and ``plotting_context("paper")`` dictionaries, which the source applies
under its overrides when seaborn is importable (it was, for the committed
render: the colour-bar outline is seaborn's ".15" edge colour) -- pinned here
because the ``dev`` extra does not ship seaborn.  One deliberate override on
the copy: DejaVu Sans is put first in ``font.sans-serif`` (seaborn lists
Arial first), because the source render resolved to DejaVu Sans (the PDF
embeds it) and the figure must not change on a machine that has Arial.  As in
the source, the figure is built and laid out inside that rc context and saved
outside it, under matplotlib's defaults.

Data is the frozen extract ``data/dose_grid.json`` (``src/freeze.py``:
``points.json`` of the canonical figure at ``origin/jb/aft-grid-heatmap-plots``
@ f530cddc, with the sha256 of the scored files it was collected from; the
gemma3_12b and gemma3_27b panels' cells are the campaign plus grid follow-ups
#1a/#1c/#1d/#1e, the 44 GLM cells follow-up #1c plus the GLM EFT grid waves of
2026-09-09/10 and the 1 GTok charter row).  Re-freeze rather than edit when
the grid is re-scored.  A missing cell is a loud KeyError, never an empty
cell.

Run from the repository root; writes ``dose_grid.pdf`` and ``.png`` next to
``src/``::

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
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "dose_grid.json"
OUTPUT = HERE.parent              # paper/figures/dose_grid/
STEM = "dose_grid"

# ------------------------------------------------------------- panels & sizes
# plot_aft_grid_canonical.py: MODELS, PANEL_TITLE, WIDTH_IN, HEIGHT_IN,
# CELL_ASPECT, PENDING_HATCH, PENDING_INK, BOX_WIDTH, PNG_DPI, X_TICK_ROTATION.
MODELS: tuple[str, ...] = ("gemma3_12b", "gemma3_27b", "glm45_air")
PANEL_TITLE = {"gemma3_12b": "Gemma 3 12B", "gemma3_27b": "Gemma 3 27B",
               "glm45_air": "GLM 110B"}  # Jonathan, 2026-09-10: "change GLM-4.5-Air to GLM 110B"
#: Single-column paper width; the height is what the eleven EFT rows need at
#: `CELL_ASPECT` once the leaning tick labels and the title row are paid for
#: (0.86 in of bands; 2.9 in when the cells were square).
WIDTH_IN = 5.5
HEIGHT_IN = 2.6
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
PNG_DPI = 300
#: Eleven token labels on a ~1.5 in panel: they lean rather than thin, so
#: every dose keeps its label.
X_TICK_ROTATION = 55.0

# ------------------------------------------------------------------- colours
#: The galleries' ink (plot_figure0_slices.INK): near-black, not #000000.
INK = "#22221f"
#: seaborn's "colorblind" palette, entries 0 (blue) and 1 (orange), as
#: `sns.color_palette("colorblind").as_hex()` reports them in seaborn 0.13.2;
#: the middle is a warm off-white so a cell at 50% is neutral rather than
#: tinted either way (plot_aft_grid_heatmap.py: COLORBLIND, CMAP, VMIN, VMAX,
#: SIDE_COLOR, BOX_COLOR).
COLORBLIND = {"blue": "#0173b2", "orange": "#de8f05"}
CMAP = LinearSegmentedColormap.from_list(
    "charter_coin", [COLORBLIND["orange"], "#f2efe9", COLORBLIND["blue"]])
VMIN, VMAX = 0.0, 100.0
#: Tick labels take their side's colour, so the "+" reads without a legend.
SIDE_COLOR = {"coin": COLORBLIND["orange"], "charter": COLORBLIND["blue"]}
BOX_COLOR = INK

# ----------------------------------------------------------------------- rc
#: seaborn 0.13.2, `axes_style("white")` -- copied verbatim (rcmod.py).
SEABORN_WHITE: dict[str, Any] = {
    "axes.axisbelow": True,
    "axes.edgecolor": ".15",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.labelcolor": ".15",
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "figure.facecolor": "white",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans",
                        "Bitstream Vera Sans", "sans-serif"],
    "grid.color": ".8",
    "grid.linestyle": "-",
    "image.cmap": "rocket",
    "lines.solid_capstyle": "round",
    "patch.edgecolor": "w",
    "patch.force_edgecolor": True,
    "text.color": ".15",
    "xtick.bottom": False,
    "xtick.color": ".15",
    "xtick.direction": "out",
    "xtick.top": False,
    "ytick.color": ".15",
    "ytick.direction": "out",
    "ytick.left": False,
    "ytick.right": False,
}
#: seaborn 0.13.2, `plotting_context("paper")`: the notebook base values
#: scaled by 0.8, computed the way seaborn computes them (rcmod.py).
_SEABORN_CONTEXT_BASE: dict[str, float] = {
    "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "axes.linewidth": 1.25, "grid.linewidth": 1, "lines.linewidth": 1.5,
    "lines.markersize": 6, "patch.linewidth": 1,
    "xtick.major.width": 1.25, "ytick.major.width": 1.25,
    "xtick.minor.width": 1, "ytick.minor.width": 1,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 4, "ytick.minor.size": 4,
}
SEABORN_PAPER: dict[str, Any] = {k: v * 0.8 for k, v in _SEABORN_CONTEXT_BASE.items()}
#: The source render's font: seaborn lists Arial first, but the committed
#: figure was set in DejaVu Sans (its PDF embeds it), so DejaVu Sans goes
#: first here and the figure does not change on a machine that has Arial.
FONT_SANS_SERIF = ["DejaVu Sans", "Arial", "Liberation Sans",
                   "Bitstream Vera Sans", "sans-serif"]
#: Paper typography: 5.5-7pt throughout (Jonathan, 2026-09-09: one point
#: down from the first cut), the galleries' ink, text kept as text in the
#: vector formats (plot_aft_grid_canonical.RC).
RC: dict[str, Any] = {
    "font.size": 6.5,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.5,
    "figure.labelsize": 6.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def theme_rc() -> dict[str, Any]:
    """seaborn's paper/white theme under the 5.5-7pt overrides, as the source's
    `theme_rc` builds it when seaborn is importable."""
    rc = {**SEABORN_WHITE, **SEABORN_PAPER}
    rc["font.sans-serif"] = FONT_SANS_SERIF
    # seaborn registers "rocket" when imported; it only names the default
    # colour map for artists that set none (nothing here does) and is not a
    # registered map without seaborn, so it is left at matplotlib's default.
    del rc["image.cmap"]
    rc.update(RC)
    return rc


# ---------------------------------------------------------------- label runs
#: Label wording (Jonathan, 2026-09-09): "Coin" and "Charter" capitalised and
#: in their side colours wherever they appear, comma-separated signs, "EFT"
#: for the conflict-token axis.  Each label is a run of coloured pieces drawn
#: over a transparent plain copy that reserves the layout space; the side
#: pieces (the words Coin and Charter with their signs) are bold as well
#: (Jonathan, 2026-09-11: "bold the words 'Coin' and 'Charter' where they show
#: up (not the token counts)" -- so the tick labels stay regular weight)
#: (plot_aft_grid_canonical.py: SIDES, X_LABEL, Y_LABEL, BAR_LABEL).
COIN, CHARTER = SIDE_COLOR["coin"], SIDE_COLOR["charter"]
SIDES: tuple[tuple[str, str], ...] = (
    (" (", INK), ("−Coin", COIN), (", ", INK), ("+Charter", CHARTER), (")", INK))
X_LABEL: tuple[tuple[str, str], ...] = (("Midtraining Tokens", INK), *SIDES)
Y_LABEL: tuple[tuple[str, str], ...] = (("EFT Tokens", INK), *SIDES)
BAR_LABEL: tuple[tuple[str, str], ...] = (
    ("chose ", INK), ("Charter", CHARTER), (" crew, % of conflict-eval runs", INK))


def plain(pieces: Sequence[tuple[str, str]]) -> str:
    """The label's text without its colours."""
    return "".join(text for text, _colour in pieces)


def piece_props(props: FontProperties, colour: str) -> FontProperties:
    """The font of one label piece: the anchor's, in bold for a side-coloured
    piece (the words Coin and Charter with their signs), as is for ink."""
    if colour == INK:
        return props
    bold = props.copy()
    bold.set_weight("bold")
    return bold


def coloured_label(fig: plt.Figure, anchor: Text,
                   pieces: Sequence[tuple[str, str]]) -> list[Text]:
    """Draw `pieces` as one run of text exactly over `anchor`: a transparent
    label carrying `plain(pieces)`, which constrained layout measures (figure
    texts it ignores).  Same font (bold for the side pieces, `piece_props`),
    same baseline, the run centred on the anchor; the anchor is horizontal or
    rotated 90° (reads bottom to top).  Call after the layout has been drawn
    once.  The bold pieces make the run a little wider than the regular-weight
    anchor that reserved its space; centring splits that overhang evenly."""
    renderer = fig.canvas.get_renderer()
    props = anchor.get_fontproperties()
    bbox = anchor.get_window_extent(renderer)
    # Unhinted metrics: hinted widths are whole pixels at the build dpi and
    # do not scale to the 300 dpi PNG, which opened gaps at the joins.  Each
    # piece is measured in its own weight and starts where the previous one
    # ends (every join here is a regular/bold join, so there is no one-string
    # rendering whose kerning could be matched).
    with matplotlib.rc_context({"text.hinting": "none"}):
        _width, _height, descent = renderer.get_text_width_height_descent(
            plain(pieces), props, False)
        widths = [renderer.get_text_width_height_descent(text, piece_props(props, colour), False)[0]
                  for text, colour in pieces]
    width = sum(widths)
    starts = [sum(widths[:i]) for i in range(len(pieces))]
    rotation = anchor.get_rotation()
    to_figure = fig.transFigure.inverted()
    texts = []
    for (text, colour), start in zip(pieces, starts, strict=True):
        if rotation == 0:
            x, y = (bbox.x0 + bbox.x1) / 2 - width / 2 + start, bbox.y0 + descent
        else:  # 90°: descenders lie to the right of the baseline, text runs upward
            x, y = bbox.x1 - descent, (bbox.y0 + bbox.y1) / 2 - width / 2 + start
        fx, fy = to_figure.transform((x, y))
        texts.append(fig.text(fx, fy, text, color=colour,
                              fontproperties=piece_props(props, colour),
                              ha="left", va="baseline", rotation=rotation,
                              rotation_mode="anchor", in_layout=False))
    return texts


# ------------------------------------------------------------------- drawing
def side_colour(value: float) -> str:
    """Tick colour by the sign of a signed token count (control / agreement: ink)."""
    if value < 0:
        return SIDE_COLOR["coin"]
    if value > 0:
        return SIDE_COLOR["charter"]
    return INK


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
    """The full box around a panel: all four spines, solid, ink
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
    by sign), one per EFT level along y (coloured by side), a thin near-black
    box around the map and no zero lines; the y label (transparent:
    `coloured_label` draws over it) on the left panel only; centred bold title
    (Jonathan, 2026-09-10: "bold the model names")
    (plot_aft_grid_canonical.dress_panel)."""
    nx, ny = len(columns), len(levels)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks(range(nx))
    ax.set_xticklabels([c["label"] for c in columns], rotation=X_TICK_ROTATION,
                       ha="right", rotation_mode="anchor")
    for label, c in zip(ax.get_xticklabels(), columns, strict=True):
        label.set_color(side_colour(c["tokens"]))
    ax.set_yticks(range(ny))
    if leftmost:
        ax.set_yticklabels([lv["label"] for lv in levels])
        for label, lv in zip(ax.get_yticklabels(), levels, strict=True):
            label.set_color(SIDE_COLOR.get(lv["side"] or "", INK))
        ax.set_ylabel(plain(Y_LABEL), alpha=0.0)
    ax.tick_params(length=0, pad=2)
    frame_axes(ax, linewidth=BOX_WIDTH)
    ax.set_title(title, loc="center", pad=3, fontweight="bold")


def build_figure(extract: dict[str, Any]) -> tuple[plt.Figure, dict[str, np.ndarray]]:
    """The figure and, per model, the rate matrix it drew
    (plot_aft_grid_canonical.build_figure, fed from the extract)."""
    levels = extract["eft_levels"]
    panels = [(model, extract["panels"][model]) for model in MODELS]
    matrices = {model: cell_matrix(panel, levels) for model, panel in panels}
    with matplotlib.rc_context(theme_rc()):
        # Equal cells across panels: widths in proportion to column counts.
        fig, axes = plt.subplots(
            1, len(panels), figsize=(WIDTH_IN, HEIGHT_IN), sharey=True,
            layout="constrained",
            gridspec_kw={"width_ratios": [len(panel["columns"]) for _m, panel in panels]})
        for ax, (model, panel) in zip(axes, panels, strict=True):
            draw_cells(ax, matrices[model])
            dress_panel(ax, panel["columns"], levels,
                        title=PANEL_TITLE[model], leftmost=ax is axes[0])
        xlabel = fig.supxlabel(plain(X_LABEL), alpha=0.0)

        mappable = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
        # An inset of the last panel, so the bar is exactly as tall as the
        # (aspect-locked) heat maps whatever the figure height.
        bar_axes = axes[-1].inset_axes([1.14, 0.0, 0.13, 1.0])
        bar_axes.set_label("<colorbar>")  # as fig.colorbar names the axes it makes
        bar = fig.colorbar(mappable, cax=bar_axes)
        bar.set_label(plain(BAR_LABEL), fontsize=6.0, alpha=0.0)
        bar.set_ticks([0, 25, 50, 75, 100])
        bar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
        bar.outline.set_linewidth(0.5)

        # Layout first, then the coloured labels over their transparent anchors.
        fig.canvas.draw()
        coloured_label(fig, axes[0].yaxis.label, Y_LABEL)
        coloured_label(fig, xlabel, X_LABEL)
        coloured_label(fig, bar.ax.yaxis.label, BAR_LABEL)
    return fig, matrices


def main() -> int:
    extract = json.loads(DATA.read_text())
    if extract.get("dummy"):
        raise SystemExit("extract is marked dummy; refusing to draw it "
                         "without a DUMMY DATA stamp")
    for model in MODELS:
        if model not in extract["panels"]:
            raise KeyError(f"extract has no panel {model!r}")
    fig, matrices = build_figure(extract)
    for model, matrix in matrices.items():
        landed = int(np.isfinite(matrix).sum())
        print(f"{model}: {landed} of {matrix.size} cells landed")
    # Saved outside the rc context, as the source does: PDF first (the
    # paper), then the PNG for the doc.
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": PNG_DPI})):
        path = OUTPUT / f"{STEM}{suffix}"
        fig.savefig(path, **kwargs)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
