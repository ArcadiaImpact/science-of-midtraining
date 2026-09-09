"""The paper's AFT-grid figure: two panels, power fit, held-out template x trained clause.

`plot_aft_grid_heatmap.py` writes galleries -- one figure per model x surface x
clause split, three fitted forms, two 2% sources, every figure carrying its own
footnote.  This module writes the ONE figure the paper shows, at single-column
width (5.5 in): Gemma 3 12B on the left, 27B on the right, the held-out
template x trained ("held-in") clause split, follow-up #1c's balanced 2% cells
(repair mode), and the POWER form

    p(chose Charter) = sigma(c + a * sgn(x)|x/100k|^alpha + b * sgn(y)|y/10M|^beta)

which is the grid's primary presentation from 2026-09-09 (Jonathan): the plane
misfits the outer columns by ~10pp, the power and symlog forms fit equally
well (`fit_comparison.md`), and the power form is the one whose shape
parameters read directly as dose exponents.

Everything that decides WHAT is drawn is imported from `plot_aft_grid_heatmap`
-- the token axes and their symlog knees, the cell readings, the fit, the
shaded surface with its 10-90% contours, the points -- so this module owns
layout only: one y axis over both models' midtraining doses (labelled on the
left panel), one shared colour bar, 7-8pt type, no footnote (the caption lives
in the paper), and PDF first.  seaborn's paper/white theme is applied when
seaborn is importable (the `analysis` extra); the same rc values are pinned by
hand otherwise, so the figure does not depend on which extra is installed.

Run from the repository root, after `collect_followup_scores.py`::

    uv run --extra dev --extra analysis python3 \\
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_canonical.py

Writes ``figures/ablations/AFT-grid/canonical/
aft-grid_power_heldout-template_trained-clause.{pdf,png,svg}`` and a
``fits.json`` beside them with both panels' coefficients, shape brackets and
points.
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
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aft_grid_fits as forms  # noqa: E402
import plot_aft_grid_heatmap as heatmap  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402
import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402

MODELS = heatmap.MODELS
PANEL_TITLE = {"gemma3_12b": "Gemma 3 12B", "gemma3_27b": "Gemma 3 27B"}
#: The one split, source and form the paper shows.
FORM = "power"
TWOPCT = "repair"
SURFACE = "heldout"
CLAUSE = "trained"
#: Single-column paper width; the height leaves the two panels roughly square
#: once the leaning tick labels, the title row and the legend are paid for.
WIDTH_IN = 5.5
HEIGHT_IN = 3.25
OUTPUT = heatmap.SCATTER.with_name("canonical")
STEM = f"aft-grid_{FORM}_{figure0.SURFACE_STEM[SURFACE]}_{figure0.CLAUSE_STEM[CLAUSE]}"
#: PDF first (the paper), PNG for GitHub, SVG for editing.
FORMATS: tuple[tuple[str, dict[str, Any]], ...] = (
    (".pdf", {}), (".png", {"dpi": 300}), (".svg", {}))

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
#: Marker areas at paper scale: the galleries' 190 / 150 / 70 would overlap at
#: the ~10pt pitch of the tightest columns here.
MARKER_AREAS = {"landed": 26.0, "starred": 20.0, "ring": 12.0}
#: Eleven token labels on a ~2.1 in panel: they lean rather than thin, so the
#: 0.25% and 0.5% columns keep their labels.
X_TICK_ROTATION = 55.0


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


def shared_y_axis(panels: Sequence[Panel]) -> heatmap.Axis:
    """One y axis for both models: the union of their midtraining doses (12B
    has 1M and no 190M, 27B the reverse), the galleries' knee and labels."""
    values = tuple(sorted({value for _model, yaxis, _rows in panels
                           for value in yaxis.values}))
    return heatmap.Axis(values, tuple(heatmap.token_label(v) for v in values),
                        heatmap.Y_LINTHRESH, "midtraining tokens  (− coin · + Charter)",
                        heatmap.Y_LINSCALE)


def _side_colour(value: float) -> str:
    if value < 0:
        return heatmap.SIDE_COLOR["coin"]
    if value > 0:
        return heatmap.SIDE_COLOR["charter"]
    return figure0.INK


def dress_panel(
    ax: plt.Axes, xaxis: heatmap.Axis, yaxis: heatmap.Axis,
    columns: Sequence[Any], x_edges: Sequence[float], y_edges: Sequence[float],
    *, title: str, leftmost: bool,
) -> None:
    """The galleries' furniture at paper scale: token ticks coloured by side,
    the full box, the two thin black zero lines; y labels on the left panel
    only."""
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xticks([xaxis.transform(value) for value in xaxis.values])
    ax.set_xticklabels(xaxis.labels, rotation=X_TICK_ROTATION, ha="right",
                       rotation_mode="anchor")
    for label, column in zip(ax.get_xticklabels(), columns, strict=True):
        label.set_color(heatmap.SIDE_COLOR.get(column.side or "", figure0.INK))
    ax.set_yticks([yaxis.transform(value) for value in yaxis.values])
    if leftmost:
        ax.set_yticklabels(yaxis.labels)
        for label, value in zip(ax.get_yticklabels(), yaxis.values, strict=True):
            label.set_color(_side_colour(value))
        ax.set_ylabel(yaxis.title)
    ax.tick_params(length=0, pad=2)
    heatmap.frame_axes(ax, linewidth=0.5)
    heatmap.zero_lines(ax, xaxis, yaxis, linewidth=0.5)
    ax.set_title(title, loc="left", pad=3)


def build_figure(
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    repair: Mapping[str, Any],
) -> tuple[plt.Figure, dict[str, Any]]:
    """The figure and its record (per panel: the fit's `record()` and every
    point), the same record the galleries write to `fits.json`."""
    heatmap._discover_controls(campaign, collected)
    xaxis, columns = heatmap.x_axis(collected, TWOPCT)
    panels: list[Panel] = [(model, *heatmap.y_axis(model)) for model in MODELS]
    yshared = shared_y_axis(panels)
    x_edges, y_edges = xaxis.edges(), yshared.edges()
    record: dict[str, Any] = {
        "form": FORM, "twopct": TWOPCT, "surface": SURFACE, "clause": CLAUSE,
        "fit_starred": True, "width_in": WIDTH_IN, "figures": {},
    }
    with matplotlib.rc_context(theme_rc()):
        fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH_IN, HEIGHT_IN),
                                 sharey=True, layout="constrained")
        any_unlanded = False
        for ax, (model, yaxis, rows) in zip(axes, panels, strict=True):
            points = heatmap.collect_points(
                rows, columns, xaxis, yaxis, clause=CLAUSE, surface=SURFACE,
                collected=collected, campaign=campaign, repair=repair, twopct=TWOPCT)
            fit = heatmap.fit_sigmoid(points, form=FORM, include_starred=True)
            if fit is not None:
                # The shared axis, not the model's own: contour labels go on
                # the row between its two highest doses, which on the shared
                # axis is the empty band between +50M and +190M on both panels
                # rather than across the 12B points at +19M/+50M.
                heatmap.draw_surface(ax, fit, xaxis, yshared, x_edges, y_edges,
                                     label_fontsize=6.0, linewidth_scale=0.7)
            heatmap.draw_points(ax, points, xaxis, yaxis, areas=MARKER_AREAS,
                                edge_width=0.5)
            dress_panel(ax, xaxis, yshared, columns, x_edges, y_edges,
                        title=PANEL_TITLE[model], leftmost=ax is axes[0])
            any_unlanded = any_unlanded or any(not p.landed for p in points)
            record["figures"][model] = {
                "model": model, "surface": SURFACE, "clause": CLAUSE,
                "twopct": TWOPCT, "form": FORM,
                "fit": fit.record() if fit is not None else None,
                "landed": sum(1 for p in points if p.landed),
                "points": [{
                    "profile": p.row.profile, "arm": p.row.arm,
                    "mixture": p.column.key, "x_tokens": p.x, "y_tokens": p.y,
                    "rate_pct": p.rate, "n_runs": p.n_runs, "starred": p.starred,
                    "in_fit": bool(fit is not None and p.landed),
                } for p in points],
            }
        fig.supxlabel("AFT conflict tokens  (− coin-labelled · + Charter-labelled)")

        mappable = plt.cm.ScalarMappable(
            cmap=heatmap.CMAP, norm=plt.Normalize(vmin=heatmap.VMIN, vmax=heatmap.VMAX))
        bar = fig.colorbar(mappable, ax=list(axes), fraction=0.05, pad=0.02,
                           shrink=0.9)
        bar.set_label("chose Charter crew, % of conflict-eval runs", fontsize=6.0)
        bar.set_ticks([0, 25, 50, 75, 100])
        bar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
        bar.ax.axhline(heatmap.VCENTRE, color=figure0.INK, linewidth=0.8)
        bar.outline.set_linewidth(0.5)

        marker = dict(linestyle="", markersize=4.5, markeredgecolor=figure0.INK,
                      markeredgewidth=0.5)
        handles = [Line2D([], [], marker="o", markerfacecolor="#f2efe9",
                          label="cell · colour = % chose Charter", **marker)]
        if any_unlanded:
            handles.append(Line2D(
                [], [], marker="o", linestyle="", markersize=3.5,
                markerfacecolor="none", markeredgecolor=house.UNCOVERED_INK,
                label="not yet landed"))
        handles.append(Line2D(
            [], [], color=heatmap.CONTOUR_COLOR, linewidth=1.0,
            label=f"fitted power σ · contours at {heatmap.contour_levels_label()}"))
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles),
                   frameon=False, handletextpad=0.4, columnspacing=1.2)
    return fig, record


def render(
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    repair: Mapping[str, Any],
    output: Path = OUTPUT,
) -> list[Path]:
    """Write the figure as PDF, PNG and SVG plus `fits.json`; return the paths."""
    fig, record = build_figure(collected=collected, campaign=campaign, repair=repair)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix, kwargs in FORMATS:
        path = output / f"{STEM}{suffix}"
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    fits_path = output / heatmap.FITS_FILE
    fits_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    written.append(fits_path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=heatmap.COLLECTED)
    parser.add_argument("--repair", type=Path, default=heatmap.COLLECTED_REPAIR)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    for path, hint in ((args.collected, ""),
                       (args.repair, " --only contamination_quality")):
        if not path.is_file():
            raise SystemExit(
                f"{path} is missing — run collect_followup_scores.py{hint} first")
    collected = json.loads(args.collected.read_text())
    repair = json.loads(args.repair.read_text())
    campaign = data.load_documents(heatmap.SCORED)

    written = render(collected=collected, campaign=campaign, repair=repair,
                     output=args.out)
    record = json.loads(written[-1].read_text())
    for model, figure in record["figures"].items():
        fit = figure["fit"]
        if fit is None:
            print(f"fit {model}: none ({figure['landed']} landed cells)")
            continue
        alpha, beta = fit["shape"]["alpha"], fit["shape"]["beta"]
        ranges = fit["shape_ranges_within_slack"]
        print(f"fit {model} [{FORM}]: {fit['equation']} | RMSE={fit['rmse_pp']:.1f}pp "
              f"LOO={fit['loo_rmse_pp']:.1f}pp n={fit['n_points']} | "
              f"α = {alpha:.2f} ({ranges['alpha'][0]:.2f}–{ranges['alpha'][1]:.2f}); "
              f"β = {beta:.2f} ({ranges['beta'][0]:.2f}–{ranges['beta'][1]:.2f}) "
              f"within {forms.FLAT_SLACK_PP:g}pp")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
