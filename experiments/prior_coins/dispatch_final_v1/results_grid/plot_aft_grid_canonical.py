"""The paper's AFT-grid figure: two panels of the measured cells, held-out template x trained clause.

`plot_aft_grid_heatmap.py` writes galleries -- one figure per model x surface x
clause split, three fitted sigmoid forms shaded behind the points, two 2%
sources, every figure carrying its own footnote.  This module writes the ONE
figure the paper shows, at single-column width (5.5 in): Gemma 3 12B, 27B and
GLM-4.5-Air left to right, an ORDINAL heat map -- one evenly sized cell per
(midtraining level, EFT level), `CELL_ASPECT` as tall as wide, midtraining
tokens along x and EFT conflict tokens along y (2026-09-09) -- the held-out
template x trained ("held-in") clause
split, follow-up #1c's balanced 2% cells (the canonical scored tree since the
2026-09-08 migration, `twopct.py`), and NOTHING BUT THE
DATA: every landed cell a square coloured by its measured % Charter, pending
cells light grey, cells a model's design lacks white.  The fitted surfaces and their contours were dropped from this figure
on 2026-09-09 (Jonathan: the fit is too difficult to work with -- its
midtraining shape parameter is not pinned down by the grid, see
`FIT_FORMS_REVIEW.md` and `fit_comparison.md`); they stay in the galleries as
diagnostics.

The GLM-4.5-Air panel's +-2% cells on the 190M arms are follow-up #1c's six
balanced cells, migrated into `scored/glm45_air_190m/<arm>/eval.json` like
the gemma rows' (`twopct.migrate_tree`, state `substituted`); its control row
is the 190M control the GLM EFT grid populated, not the campaign's legacy 19M
one; its other 190M dose levels are the GLM EFT grid's (`../aft_glm_grid/`,
complete 2026-09-10).  Its columns are the three 190M arms and the 1 GTok row
(`glm45_air_1b`, the 1B dose on `house.PLAN`, charter arm only --
`house.PROFILE_ARMS` / `house.arms_for`): one +1B column, Sid's charter arm,
complete since 2026-09-10 -- EFT = 0 and +-2% from the campaign's own scored
row (its 2% cells are the balanced draw as run, `mix.ALREADY_BALANCED_2PCT`,
state `already_balanced`, read in place like everything else), the other
eight EFT levels from the collector's 1B grid source (wave 2 of the grid).
There is no -1B column: no coin 1 GTok midtrain exists, and the empty
placeholder drawn for it while the row was pending came off on Jonathan's
call (2026-09-10: "just add the +1B one if the -1B hasn't come through") --
`plot_aft_grid_heatmap.y_axis` gives a row only to the arms a profile ran.
The legacy 19M row is left off this figure (Jonathan, 2026-09-09: "remove
the 19M columns") and stays in the galleries -- `panel_axis` applies that to
the galleries' rows.

Everything that decides WHAT is drawn is imported from `plot_aft_grid_heatmap`
-- the token axes and their symlog knees, the cell readings, the points -- so
this module owns layout only: one y axis over both models' midtraining doses
(labelled on the left panel), one shared colour bar, a thin box around each
map and no zero lines, centred bold panel titles, no legend, labels with "-Coin" /
"+Charter" in the side colours,
5.5-7pt type, no footnote (the caption lives in the paper), and PDF first.  seaborn's
paper/white theme is applied when seaborn is importable (the `analysis`
extra); the same rc values are pinned by hand otherwise, so the figure does
not depend on which extra is installed.

Run from the repository root, after `collect_followup_scores.py`::

    uv run --extra dev --extra analysis python3 \\
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_canonical.py

Writes ``figures/ablations/AFT-grid/canonical/
aft-grid_heldout-template_trained-clause.{pdf,png,svg}`` and a
``points.json`` beside them with every cell's reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.text import Text  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_aft_grid as grid  # noqa: E402
import plot_aft_grid_heatmap as heatmap  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402
import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402
import twopct  # noqa: E402

#: Three panels (Jonathan, 2026-09-09: "add 110B as well" = GLM-4.5-Air, the
#: grid's ~110B model), midtraining tokens along x and EFT tokens along y, one
#: evenly sized square per (dose level, dose level): ordinal axes, not tokens.
MODELS: tuple[str, ...] = ("gemma3_12b", "gemma3_27b", "glm45_air")
PANEL_TITLE = {"gemma3_12b": "Gemma 3 12B", "gemma3_27b": "Gemma 3 27B",
               "glm45_air": "GLM 110B"}  # Jonathan, 2026-09-10: "change GLM-4.5-Air to GLM 110B"
#: Midtraining rows the paper's panels leave out: the legacy GLM 19M run (5M
#: unique x 4 presentations, an older recipe, no control weights) is a gallery
#: row, not a paper column (Jonathan, 2026-09-09: "remove the 19M columns").
DROPPED_PROFILES: frozenset[str] = frozenset({house.LEGACY_GLM_PROFILE})
#: A midtraining row gets a column per arm it RAN (`house.arms_for`, via the
#: galleries' `heatmap.y_axis`), on the side its arm signs -- coin negative,
#: Charter positive -- and lands data like any other row: the 1 GTok GLM row
#: (`glm45_air_1b`, charter only) is one +1B column.  An arm the row did not
#: run gets no column: the empty -1B placeholder drawn while the 1B row was
#: pending (Jonathan, 2026-09-09: "add the empty 1B columns") came off once
#: the row had landed without a coin arm (Jonathan, 2026-09-10: "just add the
#: +1B one if the -1B hasn't come through").  `arms_run` reports the rows
#: drawn with fewer than the three arms into `points.json`.
#: The one split and 2% source the paper shows: the canonical ("fixed") draw.
TWOPCT = twopct.DEFAULT_SOURCE
SURFACE = "heldout"
CLAUSE = "trained"
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
#: hatch (a flat light grey reads as the colour map's 50% off-white).  Each
#: panel shows only its own model's midtraining levels, so a level a model's
#: design lacks (1M on 27B, 1M/5M/50M on GLM) is simply not a column.
PENDING_HATCH = "////"
PENDING_INK = "#b5b2ab"
#: The box around each heat map (Jonathan, 2026-09-09: boxes back, zero lines
#: gone — a heat map's cells, not lines, mark the zero column and row).
BOX_WIDTH = 0.5
OUTPUT = heatmap.SCATTER.with_name("canonical")
STEM = f"aft-grid_{figure0.SURFACE_STEM[SURFACE]}_{figure0.CLAUSE_STEM[CLAUSE]}"
#: The record beside the figure: the split and every cell's reading.
POINTS_FILE = "points.json"
#: PDF first (the paper), PNG for GitHub, SVG for editing.
PNG_DPI = 300
FORMATS: tuple[tuple[str, dict[str, Any]], ...] = (
    (".pdf", {}), (".png", {"dpi": PNG_DPI}), (".svg", {}))

#: Paper typography: 5.5-7pt throughout (Jonathan, 2026-09-09: one point
#: down from the first cut), the galleries' ink, text kept as text in the
#: vector formats.
RC: dict[str, Any] = {
    "font.size": 6.5,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.5,
    "figure.labelsize": 6.5,
    "text.color": figure0.INK,
    "axes.labelcolor": figure0.INK,
    "axes.titlecolor": figure0.INK,
    "xtick.color": figure0.INK,
    "ytick.color": figure0.INK,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
#: Eleven token labels on a ~1.5 in panel: they lean rather than thin, so
#: every dose keeps its label.
X_TICK_ROTATION = 55.0
#: Label wording (Jonathan, 2026-09-09): "Coin" and "Charter" capitalised and
#: in their side colours wherever they appear, comma-separated signs, "EFT"
#: for the conflict-token axis.  Each label is a run of coloured pieces drawn
#: over a transparent plain copy that reserves the layout space; the side
#: pieces (the words Coin and Charter with their signs) are bold as well
#: (Jonathan, 2026-09-11: "bold the words 'Coin' and 'Charter' where they show
#: up (not the token counts)" -- so the tick labels stay regular weight).
INK = figure0.INK
COIN, CHARTER = heatmap.SIDE_COLOR["coin"], heatmap.SIDE_COLOR["charter"]
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


def theme_rc() -> dict[str, Any]:
    """seaborn's paper/white theme under our 7-8pt overrides; the overrides
    alone when seaborn is not installed (the `dev` extra does not ship it)."""
    try:
        import seaborn as sns
    except ImportError:
        rc: dict[str, Any] = {}
    else:
        rc = {**sns.axes_style("white"), **sns.plotting_context("paper")}
    rc.update(RC)
    return rc


Panel = tuple[str, heatmap.Axis, tuple[heatmap.Row, ...]]


def shared_midtrain_axis(panels: Sequence[Panel]) -> heatmap.Axis:
    """One midtraining axis (drawn as x) for every model: the union of their
    doses (12B has 1M and no 190M, 27B and GLM the reverse), the galleries'
    knee and labels."""
    values = tuple(sorted({value for _model, yaxis, _rows in panels
                           for value in yaxis.values}))
    return heatmap.Axis(values, tuple(heatmap.token_label(v) for v in values),
                        heatmap.Y_LINTHRESH, plain(X_LABEL), heatmap.Y_LINSCALE)


def arms_run(model: str) -> dict[str, list[str]]:
    """The model's PLAN rows that trained fewer than the three arms
    (`house.PROFILE_ARMS`): profile -> arms, for the record."""
    return {
        house.PLAN[(candidate, dose)]: list(house.arms_for(house.PLAN[(candidate, dose)]))
        for (candidate, dose) in house.PLAN
        if candidate == model
        and house.arms_for(house.PLAN[(candidate, dose)]) != house.ARMS}


def panel_axis(model: str) -> tuple[heatmap.Axis, tuple[heatmap.Row, ...]]:
    """The galleries' midtraining rows for `model` (`heatmap.y_axis`: one row
    per PLAN dose and arm the profile ran, coin on the negative side, Charter
    on the positive, the populated control at zero), minus
    `DROPPED_PROFILES`, in signed-token order.  The panel is ordinal, so the
    token values only rank the columns; the labels are the galleries'."""
    _axis, rows = heatmap.y_axis(model)
    kept = [row for row in rows if row.profile not in DROPPED_PROFILES]
    kept.sort(key=lambda row: row.tokens)
    values = tuple(row.tokens for row in kept)
    return heatmap.Axis(values, tuple(heatmap.token_label(v) for v in values),
                        heatmap.Y_LINTHRESH, plain(X_LABEL), heatmap.Y_LINSCALE), tuple(kept)


def _side_colour(value: float) -> str:
    if value < 0:
        return heatmap.SIDE_COLOR["coin"]
    if value > 0:
        return heatmap.SIDE_COLOR["charter"]
    return figure0.INK


def draw_cells(ax: plt.Axes, points: Sequence[heatmap.Point],
               midtrain: heatmap.Axis, eft: heatmap.Axis) -> np.ndarray:
    """The heat map: rows = EFT levels (y), columns = the model's midtraining
    levels (x), every cell one square whatever its token spacing.  Landed
    cells take the colour map, pending ones a hatched white square.  Returns
    the rate matrix (NaN where nothing landed)."""
    column = {value: i for i, value in enumerate(midtrain.values)}
    row = {value: i for i, value in enumerate(eft.values)}
    matrix = np.full((len(eft.values), len(midtrain.values)), np.nan)
    for point in points:
        i, j = row[point.x], column[point.y]
        if point.landed:
            matrix[i, j] = point.rate
        else:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor="white",
                                   edgecolor=PENDING_INK, hatch=PENDING_HATCH,
                                   linewidth=0.0, zorder=1))
    ax.imshow(matrix, cmap=heatmap.CMAP, vmin=heatmap.VMIN, vmax=heatmap.VMAX,
              origin="lower", interpolation="nearest", aspect=CELL_ASPECT, zorder=2,
              extent=(-0.5, len(midtrain.values) - 0.5, -0.5, len(eft.values) - 0.5))
    return matrix


def dress_panel(
    ax: plt.Axes, midtrain: heatmap.Axis, eft: heatmap.Axis,
    columns: Sequence[Any], *, title: str, leftmost: bool,
) -> None:
    """Ordinal axes: one tick per midtraining level along x (labels coloured
    by sign), one per EFT level along y (coloured by side), a thin near-black
    box around the map and no zero lines; the y label (transparent:
    `coloured_label` draws over it) on the left panel only; centred bold title
    (Jonathan, 2026-09-10: "bold the model names")."""
    nx, ny = len(midtrain.values), len(eft.values)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks(range(nx))
    ax.set_xticklabels(midtrain.labels, rotation=X_TICK_ROTATION, ha="right",
                       rotation_mode="anchor")
    for label, value in zip(ax.get_xticklabels(), midtrain.values, strict=True):
        label.set_color(_side_colour(value))
    ax.set_yticks(range(ny))
    if leftmost:
        ax.set_yticklabels(eft.labels)
        for label, column in zip(ax.get_yticklabels(), columns, strict=True):
            label.set_color(heatmap.SIDE_COLOR.get(column.side or "", figure0.INK))
        ax.set_ylabel(plain(Y_LABEL), alpha=0.0)
    ax.tick_params(length=0, pad=2)
    heatmap.frame_axes(ax, linewidth=BOX_WIDTH)
    ax.set_title(title, loc="center", pad=3, fontweight="bold")


def build_figure(
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    repair: Mapping[str, Any] | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    """The figure and its record (per panel: the split and every point).

    `campaign` is the scored tree as `plot_stacked.load_documents` returns it
    in `TWOPCT` mode; `repair` is follow-up #1c's collected cells
    (`heatmap.repair_collections`), consulted for the control-row preference
    only.
    """
    grid.note_twopct_state(campaign)  # which drawn rows are still narrow (stars)
    heatmap._discover_controls(campaign, collected, repair)
    eft, columns = heatmap.x_axis(collected, TWOPCT)
    panels: list[Panel] = [(model, *panel_axis(model)) for model in MODELS]
    midtrain = shared_midtrain_axis(panels)  # the union: recorded, not drawn
    partial = {model: arms_run(model) for model in MODELS}
    record: dict[str, Any] = {
        "twopct": TWOPCT, "surface": SURFACE, "clause": CLAUSE,
        "fit": None, "width_in": WIDTH_IN, "figures": {},
        "x_axis": "midtraining tokens", "y_axis": "EFT conflict tokens",
        "layout": "ordinal heat map: one square per (midtraining level, EFT level); "
                  "each panel shows its own model's midtraining levels",
        "midtraining_levels_union": list(midtrain.values),
        "midtraining_dropped_profiles": sorted(DROPPED_PROFILES),
        "midtraining_arms_run": {
            model: arms for model, arms in partial.items() if arms},
        "tokens_note": (
            "x_tokens / y_tokens are the galleries' token labels, denominated "
            "on the first landed Gemma 12B cell's counter for every panel; a "
            "column without a counter in the grid collection (the campaign's "
            "and #1c's 2% cells, the GLM cells) falls back to "
            "plot_aft_grid_heatmap.FALLBACK_TOKENS_PER_ROW -- the squares are "
            "placed by dose rank, so the figure does not depend on it"),
    }
    with matplotlib.rc_context(theme_rc()):
        # Equal cells across panels: widths in proportion to column counts.
        fig, axes = plt.subplots(
            1, len(panels), figsize=(WIDTH_IN, HEIGHT_IN), sharey=True,
            layout="constrained",
            gridspec_kw={"width_ratios": [len(yaxis.values) for _m, yaxis, _r in panels]})
        for ax, (model, yaxis, rows) in zip(axes, panels, strict=True):
            points = heatmap.collect_points(
                rows, columns, eft, yaxis, clause=CLAUSE, surface=SURFACE,
                collected=collected, campaign=campaign, twopct=TWOPCT)
            draw_cells(ax, points, yaxis, eft)
            dress_panel(ax, yaxis, eft, columns,
                        title=PANEL_TITLE[model], leftmost=ax is axes[0])
            record["figures"][model] = {
                "model": model, "surface": SURFACE, "clause": CLAUSE,
                "twopct": TWOPCT,
                "landed": sum(1 for p in points if p.landed),
                "points": [{
                    "profile": p.row.profile, "arm": p.row.arm,
                    "mixture": p.column.key, "x_tokens": p.x, "y_tokens": p.y,
                    "rate_pct": p.rate, "n_runs": p.n_runs, "starred": p.starred,
                    "landed": p.landed,
                } for p in points],
            }
        xlabel = fig.supxlabel(plain(X_LABEL), alpha=0.0)

        mappable = plt.cm.ScalarMappable(
            cmap=heatmap.CMAP, norm=plt.Normalize(vmin=heatmap.VMIN, vmax=heatmap.VMAX))
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
    return fig, record


def render(
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    repair: Mapping[str, Any] | None = None,
    output: Path = OUTPUT,
) -> list[Path]:
    """Write the figure as PDF, PNG and SVG plus `points.json`; return the paths."""
    fig, record = build_figure(collected=collected, campaign=campaign, repair=repair)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix, kwargs in FORMATS:
        path = output / f"{STEM}{suffix}"
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    points_path = output / POINTS_FILE
    points_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    written.append(points_path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=heatmap.COLLECTED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    if not args.collected.is_file():
        raise SystemExit(
            f"{args.collected} is missing — run collect_followup_scores.py first")
    collected = json.loads(args.collected.read_text())
    # The canonical draw: the loader hands back the migrated tree as it is.
    data.TWOPCT_SOURCE = TWOPCT
    campaign = data.load_documents(heatmap.SCORED)

    written = render(collected=collected, campaign=campaign,
                     repair=heatmap.repair_collections(), output=args.out)
    record = json.loads(written[-1].read_text())
    for model, figure in record["figures"].items():
        print(f"{model}: {figure['landed']} of {len(figure['points'])} cells landed")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
