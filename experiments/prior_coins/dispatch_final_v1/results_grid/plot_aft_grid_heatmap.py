"""Follow-up #1a as a scatter over a fitted logistic surface (Jonathan, 2026-09-08).

The heat map this module used to draw (Slack, 2026-09-07) put one coloured
cell per (midtraining dose, AFT mixture) pair on two signed symlog token axes.
This revision keeps those axes and that colour scale, replaces the cells with
**points**, and fits one logistic surface through them, shaded behind::

    p(chose Charter) = sigma(a * x + b * y + c)

with x = signed AFT conflict tokens and y = signed midtraining tokens, both in
RAW tokens -- the fit never sees the symlog transform; only the drawing does.
Contours are drawn every 20 points from 10% to 90% Charter.  They are straight lines in
raw token space and bend on the symlog axes, which is the intended reading:
the picture shows how far the data sit from a plane in logit space.

Axes.  x is AFT conflict tokens, signed (- coin-labelled, + Charter-labelled);
y is midtraining tokens, signed (- coin, + Charter), with the filler control
at zero.  Both are symlog by hand: linear up to the smallest non-zero dose
and log10 beyond, with the linear half-range drawn one median dose step long,
so the ladder reads evenly spaced and the two thin near-black zero lines form
a "+" through the plot, inside a near-black box (the figures' ink, not
#000000).  Tick labels are token counts.  Contours are all solid and of
equal weight, unlabelled -- the legend names the levels -- and each takes the
colour map's own colour at its level darkened toward the ink (mid grey at
50%, dark orange at 10%, dark blue at 90%); the colour bar carries a matching
mark at every level, drawn as a tick so it sits on the same pixel row as the
numeric ticks (Jonathan, 2026-09-09).

Fit.  Binomial maximum likelihood (logistic regression) by iteratively
reweighted least squares -- `scimt.utils.sigmoid`, the shared fitter -- with
one observation per landed cell, n = its conflict-run count and k = Charter
share x n.  `--form` picks the surface (see `aft_grid_fits`): `plane`
(default, 3 parameters), `power` (signed power laws in x and y, 5) or
`symlog` (signed log knees in x and y, 5).  The two shape parameters of the
5-parameter forms are profiled on a bounded grid and the footnote reports how
wide a range of them fits within 0.5pp of the best -- the overfitting check
this grid actually needs, since it has only five non-zero conflict
magnitudes.  Every figure also quotes its leave-one-cell-out RMSE next to the
in-sample one; `compare_aft_grid_fits.py` runs the fuller comparison.  In "campaign" mode the two
starred 2% columns are the legacy narrow draw, which
`contamination-data-quality/` measures ~25pp off a balanced 2%; they are
drawn as squares so the eye can weigh them, and since 2026-09-08 (Jonathan)
they are fitted with the rest -- `--exclude-starred` drops them.  In "repair"
mode nothing is starred.  The two 0.5% columns (follow-up #1d, 2026-09-09:
41 conflict rows, ~44.6k tokens) and the two 0.25% columns (#1e, 2026-09-09:
20 rows, ~21.8k tokens, nested in the 0.5% cells) are the sub-1% doses; the
x knee moved from 40k to 10k with the 0.25% pair so the smallest column still
sits past it.  Nothing here assumes a column count.
Fewer than four fittable points, a rank-deficient design or a fit that does
not converge leaves the background blank rather than drawing a surface nobody
should believe.  Coefficients, RMSE in percentage points and the points behind
each fit go to `fits.json` beside the figures.

Which cells exist and where they sit is inherited unchanged from the heat map:
11 x 9 rather than 7 x 7 (four midtrain doses per direction, and since
2026-09-09 the 0.5% and 0.25% column pairs), control at zero
directional tokens, no 4B row, no 100%-Charter column, and token denomination
from the trainer's own counter (see `conflict_tokens`).  The module keeps its
historical name; the heat-map gallery it wrote is still under
`figures/ablations/AFT-grid/heatmap/`.

Run from the repository root, after `collect_followup_scores.py`::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid_heatmap.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scimt.utils import sigmoid as sigfit  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import followup_mixtures as mix  # noqa: E402
import plot_aft_grid as grid  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402
import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402
import aft_grid_fits as forms  # noqa: E402

SCORED = HERE / "scored"
COLLECTED = SCORED / "ablations" / "aft_grid.json"
COLLECTED_REPAIR = SCORED / "ablations" / "contamination_quality.json"
SCATTER = HERE / "figures" / "ablations" / "AFT-grid" / "scatter"


def output_dir(twopct: str, form: str) -> Path:
    """scatter/, scatter-power/, scatter-symlog/ for the normal (repair) 2% draw;
    a -campaign-2pct twin when the legacy narrow draw is asked for."""
    name = "scatter" + ("" if form == "plane" else f"-{form}")
    return SCATTER.with_name(name + ("-campaign-2pct" if twopct == "campaign" else ""))
#: Per-gallery record of every fit: coefficients, RMSE, and the points behind it.
FITS_FILE = "fits.json"

MODELS = grid.MODELS

#: Where the two 2% columns come from.  "repair" is the normal mode and the
#: CLI default (Jonathan, 2026-09-08; every #1c cell has landed).
#:
#: "campaign" is the historical narrow-conflict draw, starred everywhere it
#: appears.  "repair" is follow-up #1c's balanced draw, which
#: `contamination-data-quality/` measures as worth +25pp / -16pp on the
#: primary metric — i.e. the legacy columns understate a 2% dose badly, and
#: badly enough that mixing the two draws on one axis is not defensible.
#:
#: In "repair" mode a 2% cell whose #1c partner has not landed is left BLANK
#: rather than falling back to the legacy value: a silent fallback would put
#: two different interventions in the same column with nothing marking which
#: is which, and that is exactly what the asterisk existed to prevent.  There
#: is therefore no star in repair mode, because there is nothing starred left.
#:
#: A row in `mix.ALREADY_BALANCED_2PCT` (the 1 GTok GLM charter row,
#: 2026-09-10) ran its CAMPAIGN 2% cells on the balanced draw to begin with,
#: so those cells are the balanced measurement and have no #1c partner to wait
#: for: `cell_value` reads them in place in either mode and `is_starred` never
#: stars them.
TWOPCT_SOURCES = ("campaign", "repair")

#: Fallback tokens-per-row, used only until a cell's own tokens_state lands.
#: Measured across every published grid cell; the spread is 0.4 tokens.
FALLBACK_TOKENS_PER_ROW = 1088.0

#: Symlog knees (tokens) and linear scales (decades of drawn length for the
#: linear half-range).  Jonathan, 2026-09-09: "so the grid is more evenly
#: spaced".  Each axis is linear up to its smallest non-zero dose and log10
#: beyond, and the zero-to-first-dose gap is set equal to the median gap
#: between adjacent doses.  On x the knee is the 0.25% column (20 rows x the
#: 1,088 fallback tokens/row = 21.76k) and one x2 step, log10 2 = 0.301, is
#: the median: the six |x| levels draw at 0 / 0.30 / 0.61 / 0.91 / 1.22 / 1.61
#: (gaps 0.30 / 0.31 / 0.30 / 0.30 / 0.40 -- the last step, 2% -> 5%, is x2.5).
#: On y the knee is the 1M dose and the median adjacent-level gap is a x3.8
#: step, log10 3.8 = 0.58 (5M -> 19M and 50M -> 190M; 1M -> 5M is 0.70 and
#: 19M -> 50M is 0.42), so the levels draw at 0 / 0.58 / 1.28 / 1.86 / 2.28 /
#: 2.86.  The earlier log10(1 + |v|/knee) curve with knees 40k, then 10k, and
#: 0.7M is gone; the committed gallery PNGs predate this and show it until
#: regenerated.
X_LINTHRESH = 20 * FALLBACK_TOKENS_PER_ROW
X_LINSCALE = math.log10(2.0)
Y_LINTHRESH = 1_000_000.0
Y_LINSCALE = math.log10(19 / 5)

#: seaborn's "colorblind" palette, entries 0 (blue) and 1 (orange), as
#: `sns.color_palette("colorblind").as_hex()` reports them in seaborn 0.13.2.
#: Pinned as hex so this gallery does not grow a seaborn import for two
#: strings.  The middle is the same warm off-white the heat map used, so a
#: cell at 50% is neutral rather than tinted either way.
COLORBLIND = {"blue": "#0173b2", "orange": "#de8f05"}
CMAP = LinearSegmentedColormap.from_list(
    "charter_coin", [COLORBLIND["orange"], "#f2efe9", COLORBLIND["blue"]])
VMIN, VCENTRE, VMAX = 0.0, 50.0, 100.0
#: Tick labels take their side's colour, so the "+" reads without a legend.
SIDE_COLOR = {"coin": COLORBLIND["orange"], "charter": COLORBLIND["blue"]}

#: Where the fitted surface is contoured, in % Charter, and how (Jonathan,
#: 2026-09-09, three rounds): every 20 points from 10% to 90%, every level
#: solid and the same width, no inline labels (the legend names the levels,
#: `contour_levels_label`), and each level in the colour map's own colour at
#: that level darkened by blending CONTOUR_DARKEN of the way toward the ink --
#: the map's 50% off-white becomes a mid grey, 10% a dark orange, 90% a dark
#: blue -- so a contour and the colour-bar mark at its level share a colour.
#: Shared by the galleries and the canonical figure; the committed gallery
#: PNGs predate the restyle.
CONTOUR_LEVELS = (10.0, 30.0, 50.0, 70.0, 90.0)
CONTOUR_WIDTH = 0.9
CONTOUR_STYLE = {level: ("-", CONTOUR_WIDTH) for level in CONTOUR_LEVELS}
CONTOUR_LABELS = False
CONTOUR_DARKEN = 0.45


def contour_colour(level: float) -> str:
    """The colour map's colour at `level` % Charter, darkened toward the ink."""
    base = np.asarray(CMAP((level - VMIN) / (VMAX - VMIN))[:3], dtype=float)
    ink = np.asarray(to_rgb(figure0.INK), dtype=float)
    return to_hex((1.0 - CONTOUR_DARKEN) * base + CONTOUR_DARKEN * ink)


CONTOUR_COLORS = {level: contour_colour(level) for level in CONTOUR_LEVELS}
#: The two reference lines (zero conflict dose, zero midtrain direction) and
#: the full box around each panel: the figures' near-black ink (the colour of
#: their text, not #000000), thin, solid.
ZERO_LINE_COLOR = figure0.INK
ZERO_LINE_WIDTH = 0.7
BOX_COLOR = figure0.INK
BOX_WIDTH = 0.8


def contour_levels_label() -> str:
    """'10 / 30 / 50 / 70 / 90%', for legends and footnotes."""
    return " / ".join(f"{level:.0f}" for level in CONTOUR_LEVELS) + "%"


def mark_contour_levels(bar: Any, *, linewidth_scale: float = 1.0) -> None:
    """A mark across the colour bar at every contour level, in that contour's
    colour and width, so the reader can match bar to surface.

    Drawn as MINOR TICKS of the bar's long axis (pointing in, as long as the
    bar is wide) rather than as lines: ticks are markers, and a line drawn
    through the path pipeline snaps to the pixel grid half a pixel away from
    where the numeric ticks land -- visible at 300 dpi as a mark sitting just
    off its tick.  Call it once the layout is final (after `subplots_adjust`
    or with a layout engine attached), since the tick length is the bar's
    width in points at that moment.
    """
    if getattr(bar, "orientation", "vertical") != "vertical":
        raise ValueError("mark_contour_levels draws vertical colour bars only")
    fig = bar.ax.figure
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.execute(fig)
    box = bar.ax.get_position()
    length_pt = box.width * fig.get_figwidth() * 72.0
    axis = bar.ax.yaxis
    # Ticks must draw ABOVE the colour fill: under seaborn's theme
    # (axes.axisbelow True) the axis sits beneath it and inward ticks vanish.
    bar.ax.set_axisbelow(False)
    # A minor tick on top of a major one (50% here) is exactly the point; the
    # axis would otherwise drop it as a duplicate.
    axis.remove_overlapping_locs = False
    bar.set_ticks(list(CONTOUR_LEVELS), minor=True)
    width = CONTOUR_WIDTH * linewidth_scale
    # Agg snaps a stroke to pixel centres or to pixel edges by the parity of
    # its rounded width in pixels, so two ticks at one data value only share a
    # pixel row when they share a width: the numeric ticks take the contours'.
    bar.ax.tick_params(axis="y", which="major", width=width)
    bar.ax.tick_params(axis="y", which="minor", left=True, right=False,
                       direction="in", length=length_pt, width=width,
                       labelleft=False, labelright=False)
    for tick, loc in zip(axis.get_minor_ticks(), axis.get_minorticklocs(), strict=True):
        level = min(CONTOUR_LEVELS, key=lambda candidate: abs(candidate - loc))
        tick.tick1line.set_color(CONTOUR_COLORS[level])
#: Samples per axis for the shaded surface.  It is sampled in transformed
#: (symlog) coordinates, so the knees get the same pixel density as the tails.
SURFACE_RESOLUTION = 400
#: A form wants more points than parameters before a surface is drawn.
MIN_FIT_POINTS = {form: k + 1 for form, k in forms.N_PARAMETERS.items()}


@dataclass(frozen=True)
class Axis:
    """One axis: signed token coordinates, their labels, and the symlog map."""

    values: tuple[float, ...]
    labels: tuple[str, ...]
    linthresh: float
    title: str
    #: Drawn length of the linear half-range [0, linthresh], in decades.
    linscale: float = X_LINSCALE

    def transform(self, value: float) -> float:
        """Symlog, done by hand so the drawing can work in its coordinates:
        linear to the knee (`linscale` decades long), log10 beyond it."""
        sign = -1.0 if value < 0 else 1.0
        magnitude = abs(value)
        if magnitude <= self.linthresh:
            return sign * self.linscale * magnitude / self.linthresh
        return sign * (self.linscale + math.log10(magnitude / self.linthresh))

    def inverse(self, position: Any) -> Any:
        """Back from a symlog position to signed tokens; takes arrays too.
        Exact, so a surface sampled in drawn coordinates has no seam at the
        knee."""
        position = np.asarray(position, dtype=float)
        magnitude = np.abs(position)
        linear = magnitude * self.linthresh / self.linscale
        logarithmic = self.linthresh * 10.0 ** (magnitude - self.linscale)
        return np.sign(position) * np.where(magnitude <= self.linscale, linear,
                                            logarithmic)

    def edges(self) -> list[float]:
        """Axis limits and the old cell boundaries: midpoints in transformed
        space, with the two ends extrapolated one half-step out."""
        points = [self.transform(value) for value in self.values]
        inner = [(a + b) / 2 for a, b in zip(points, points[1:])]
        first = points[0] - (inner[0] - points[0]) if inner else points[0] - 0.5
        last = points[-1] + (points[-1] - inner[-1]) if inner else points[-1] + 0.5
        return [first, *inner, last]


def token_label(tokens: float) -> str:
    magnitude = abs(tokens)
    if magnitude == 0:
        return "0"
    sign = "+" if tokens > 0 else "−"
    if magnitude >= 1e9:
        return f"{sign}{magnitude / 1e9:.0f}B"
    if magnitude >= 1e6:
        return f"{sign}{magnitude / 1e6:.0f}M"
    return f"{sign}{magnitude / 1e3:.0f}k"


def crossing_label(tokens: float | None) -> str:
    """A crossing the fit does not have (a flat surface) reads as "n/a"."""
    if tokens is None or not math.isfinite(tokens):
        return "n/a"
    return token_label(tokens)


def conflict_tokens(
    mixture: mix.Mixture, tokens_meta: Mapping[str, Any] | None,
) -> float:
    """Signed conflict tokens for one mixture, from the trainer's own counter."""
    rows = mixture.conflict_rows.get(mix.GRID_V2.rows, 0)
    per_row = FALLBACK_TOKENS_PER_ROW
    entry = (tokens_meta or {}).get(mixture.key)
    if isinstance(entry, dict) and entry.get("total") and entry.get("rows"):
        per_row = entry["total"] / (entry["rows"] * entry.get("epochs", 2))
    sign = 0.0 if mixture.side is None else (
        1.0 if mixture.side == "charter" else -1.0)
    return sign * rows * per_row


def any_tokens_meta(collected: Mapping[str, Any]) -> dict[str, Any]:
    """The first landed cell's counter per mixture, which is a 12B one.

    Cells of one model agree to a fraction of a token, but the 27B runs count
    1,016-1,017 tokens/row against the 12B runs' 1,087.6-1,088.2 (both
    measured, `meta.tokens`), so this puts both models on the 12B
    denomination: one shared column grid, a 6.6% offset on x for 27B that a
    symlog axis does not resolve.  The footnote says so.
    """
    merged: dict[str, Any] = {}
    for document in collected.get("documents", {}).values():
        for key, entry in document.get("meta", {}).get("tokens", {}).items():
            merged.setdefault(key, entry)
    return merged


def is_twopct(mixture: mix.Mixture) -> bool:
    return abs(mixture.dose) == 2.0


def is_starred(mixture: mix.Mixture, twopct: str = "campaign",
               profile: str | None = None) -> bool:
    """Campaign-mode 2% cells are the legacy narrow draw and carry the star --
    except on a row whose campaign 2% cells were drawn balanced as run
    (`mix.ALREADY_BALANCED_2PCT`).  Without a `profile` the answer is the
    column's, which is what the axis label shows."""
    if profile is not None and profile in mix.ALREADY_BALANCED_2PCT:
        return False
    return twopct == "campaign" and grid.study_for(mixture.key).is_narrow(mixture.key)


def x_axis(
    collected: Mapping[str, Any], twopct: str = "campaign",
) -> tuple[Axis, tuple[mix.Mixture, ...]]:
    """The dose ladder as columns, in tokens: Jonathan's seven plus the 0.5%
    and 0.25% pairs.  100%-Charter is not one of them."""
    columns = tuple(m for m in mix.DOSE_AXIS)
    tokens_meta = any_tokens_meta(collected)
    values = tuple(conflict_tokens(m, tokens_meta) for m in columns)
    labels = tuple(
        token_label(value) + (mix.NARROW_STAR if is_starred(m, twopct) else "")
        for m, value in zip(columns, values)
    )
    return Axis(values, labels, X_LINTHRESH,
                "AFT conflict tokens  (− coin-labelled · + Charter-labelled)",
                X_LINSCALE), columns


@dataclass(frozen=True)
class Row:
    profile: str
    arm: str
    tokens: float
    label: str


def y_axis(model: str) -> tuple[Axis, tuple[Row, ...]]:
    """Coin midtrains below, control at zero, Charter midtrains above."""
    doses = [dose for dose in house.DOSES if (model, dose) in house.PLAN]
    rows: list[Row] = []
    for dose in sorted(doses, reverse=True):
        profile = house.PLAN[(model, dose)]
        rows.append(Row(profile, "coin", -float(dose),
                        f"{house.DOSE_LABEL[dose]} coin"))
    # The campaign runs control at 5M only, which is also the smallest dose it
    # has: filler midtraining, so zero directional tokens by construction.
    # The control row is whichever control the FOLLOW-UP populated, not the
    # smallest-dose one.  #1a ran its 1%/5% control cells at the 5M profile
    # only, so on 27B (where 5M is also the smallest dose) the two rules agree
    # and the row fills, while on 12B the smallest-dose rule picks 1M and
    # leaves four of the seven columns blank against data that exists one row
    # over.  Both are filler midtraining -- zero directional tokens either way
    # -- so preferring the populated one costs nothing and makes the two
    # models use the same control dose.  The row label names the dose.
    # Follow-up #1c's balanced 2% cells are the second preference: on GLM no
    # grid follow-up has run, and #1c's 190M control is what puts the 190M
    # control on the axis (the campaign's legacy 19M control would otherwise
    # be the row, with nothing but its EFT = 0 cell on it).  The campaign's
    # own controls come last, smallest dose first.
    control = None
    for populated in (_POPULATED_CONTROLS, _REPAIR_CONTROLS, _CONTROL_PROFILES):
        control = next(
            ((house.PLAN[(model, dose)], dose) for dose in doses
             if (house.PLAN[(model, dose)], "control") in populated),
            None)
        if control is not None:
            break
    if control is not None:
        # Zero DIRECTIONAL tokens, but it is still a real midtrain run; name
        # its dose so the row is not read as "no midtraining".
        profile, dose = control
        rows.append(Row(profile, "control", 0.0,
                        f"control · {house.DOSE_LABEL[dose]} filler"))
    for dose in doses:
        profile = house.PLAN[(model, dose)]
        rows.append(Row(profile, "charter", float(dose),
                        f"{house.DOSE_LABEL[dose]} Charter"))
    values = tuple(row.tokens for row in rows)
    labels = tuple(token_label(row.tokens) for row in rows)
    return Axis(values, labels, Y_LINTHRESH,
                "midtraining tokens  (− coin · + Charter)", Y_LINSCALE), tuple(rows)


#: Which (profile, arm) control cells the campaign actually ran, which of them
#: the grid follow-ups (#1a, #1d, #1e) extended with new dose cells, and which
#: follow-up #1c re-ran at 2%.  All three are discovered rather than assumed,
#: so a later control does not need a code edit; `y_axis` prefers them in
#: that order.
_CONTROL_PROFILES: set[tuple[str, str]] = set()
_POPULATED_CONTROLS: set[tuple[str, str]] = set()
_REPAIR_CONTROLS: set[tuple[str, str]] = set()


def _discover_controls(
    campaign: Mapping[tuple[str, str], Any],
    collected: Mapping[str, Any] | None = None,
    repair: Mapping[str, Any] | None = None,
) -> None:
    _CONTROL_PROFILES.clear()
    _CONTROL_PROFILES.update(key for key in campaign if key[1] == "control")
    for controls, collection in ((_POPULATED_CONTROLS, collected),
                                 (_REPAIR_CONTROLS, repair)):
        controls.clear()
        for key in (collection or {}).get("documents", {}):
            profile, arm = key.split("|")
            if arm == "control":
                controls.add((profile, arm))


def repair_unit(
    profile: str, arm: str, mixture: mix.Mixture,
    repair: Mapping[str, Any],
) -> data.Unit | None:
    """The #1c balanced cell, or None — never a fall back to the legacy one."""
    endpoint = mix.GRID_REPAIR.endpoint(mixture.key, 2)
    document = repair.get("documents", {}).get(f"{profile}|{arm}")
    if endpoint is None or not isinstance(document, dict):
        return None
    if endpoint not in data.endpoints_in(document):
        return None
    return data.Unit(profile, arm, endpoint, document)


def cell_value(
    profile: str, arm: str, mixture: mix.Mixture, clause: str, surface: str,
    *, collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    repair: Mapping[str, Any] | None = None,
    twopct: str = "campaign",
) -> tuple[float, int] | None:
    """One cell's reading -- % Charter and its conflict-run count -- or None.

    In repair mode a 2% cell is follow-up #1c's balanced re-run (`repair_unit`,
    never a fall back to the legacy draw) -- unless the row is in
    `mix.ALREADY_BALANCED_2PCT`: its campaign 2% cells were drawn balanced as
    run, so they ARE the balanced measurement and are read in place through
    `unit_for` like every other campaign cell, rather than left blank for a
    #1c partner that was never needed.  Everything else comes from the
    collected grid or the campaign through `unit_for`.
    """
    if (twopct == "repair" and is_twopct(mixture)
            and profile not in mix.ALREADY_BALANCED_2PCT):
        unit = repair_unit(profile, arm, mixture, repair or {})
    else:
        unit = grid.unit_for(profile, arm, mixture.key, 2,
                             collected=collected, campaign=campaign)
    if unit is None:
        return None
    reading = data.conflict_reading(unit, clause, surface)
    if reading is None:
        return None
    return 100 * reading.shares.get("charter", 0.0), reading.n_runs


@dataclass(frozen=True)
class Point:
    """One (midtraining, mixture) cell as a point: tokens, reading, provenance."""

    row: Row
    column: mix.Mixture
    #: Signed AFT conflict tokens and signed midtraining tokens, RAW.
    x: float
    y: float
    #: % of conflict-eval runs that chose Charter; None until the cell lands.
    rate: float | None
    n_runs: int | None
    #: Legacy narrow-draw 2% cell (campaign mode only).
    starred: bool

    @property
    def landed(self) -> bool:
        return self.rate is not None


def collect_points(
    rows: Sequence[Row], columns: Sequence[mix.Mixture],
    xaxis: Axis, yaxis: Axis, *,
    clause: str, surface: str,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    repair: Mapping[str, Any] | None = None,
    twopct: str = "campaign",
) -> list[Point]:
    points: list[Point] = []
    for row, y in zip(rows, yaxis.values, strict=True):
        for column, x in zip(columns, xaxis.values, strict=True):
            reading = cell_value(row.profile, row.arm, column, clause, surface,
                                 collected=collected, campaign=campaign,
                                 repair=repair, twopct=twopct)
            rate, n_runs = reading if reading is not None else (None, None)
            points.append(Point(row, column, x, y, rate, n_runs,
                                is_starred(column, twopct, row.profile)))
    return points


def fit_sigmoid(
    points: Sequence[Point], *, form: str = "plane", include_starred: bool = False,
) -> forms.FormFit | None:
    """Fit one form to the landed cells, in raw signed tokens.

    Returns None rather than a surface when there is too little to fit: fewer
    points than the form's parameters plus one, a covariate that never
    varies, or no shape on the grid that converges.  The caller leaves the
    background blank in that case.  Anything else the fitter rejects (a share
    above 1, a non-positive n) is a data bug and is allowed to raise.
    """
    used = [p for p in points
            if p.landed and p.n_runs and (include_starred or not p.starred)]
    if len(used) < MIN_FIT_POINTS[form]:
        return None
    x = np.array([p.x for p in used])
    y = np.array([p.y for p in used])
    if np.linalg.matrix_rank(forms.design("plane", (), x, y)) < 3:
        return None
    rates = np.array([p.rate for p in used])
    trials = np.array([float(p.n_runs) for p in used])
    try:
        return forms.fit_form(form, x, y, rates, trials, loo=True)
    except sigfit.ConvergenceError as error:
        print(f"fit skipped: {error}", file=sys.stderr)
        return None


def draw_surface(
    ax: plt.Axes, fit: forms.FormFit, xaxis: Axis, yaxis: Axis,
    x_edges: Sequence[float], y_edges: Sequence[float],
    *, label_fontsize: float = 8.0, linewidth_scale: float = 1.0,
) -> None:
    """Shade sigma(a x + b y + c) behind the points and contour it.

    Sampled at pixel centres in transformed coordinates, mapped back to raw
    tokens for the evaluation, so the image aligns with the axes exactly and
    the symlog knees are as finely resolved as the tails.  `label_fontsize`
    and `linewidth_scale` let the paper-width canonical figure reuse this at
    its own scale; the galleries take the defaults.
    """
    x0, x1, y0, y1 = x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]
    res = SURFACE_RESOLUTION
    tx = x0 + (np.arange(res) + 0.5) * (x1 - x0) / res
    ty = y0 + (np.arange(res) + 0.5) * (y1 - y0) / res
    grid_x, grid_y = np.meshgrid(tx, ty)
    surface = fit.predict(xaxis.inverse(grid_x), yaxis.inverse(grid_y))
    ax.imshow(surface, extent=(x0, x1, y0, y1), origin="lower", aspect="auto",
              cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="bilinear",
              zorder=0)
    levels = [level for level in CONTOUR_LEVELS
              if surface.min() < level < surface.max()]
    if not levels:
        return
    colours = [CONTOUR_COLORS[level] for level in levels]
    contours = ax.contour(
        grid_x, grid_y, surface, levels=levels, colors=colours,
        linestyles=[CONTOUR_STYLE[level][0] for level in levels],
        linewidths=[CONTOUR_STYLE[level][1] * linewidth_scale for level in levels],
        zorder=3)
    if not CONTOUR_LABELS:
        return
    positions = contour_label_positions(fit, levels, xaxis, yaxis, x_edges)
    ax.clabel(contours, fmt=lambda level: f"{level:.0f}%", fontsize=label_fontsize,
              inline=True, inline_spacing=6, colors=colours,
              **({"manual": positions} if positions else {}))


def frame_axes(ax: plt.Axes, *, linewidth: float = BOX_WIDTH) -> None:
    """The full box around a panel: all four spines, solid, black."""
    for side in ("top", "right", "left", "bottom"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_color(BOX_COLOR)
        spine.set_linewidth(linewidth)
        spine.set_linestyle("-")


def zero_lines(ax: plt.Axes, xaxis: Axis, yaxis: Axis, *,
               linewidth: float = ZERO_LINE_WIDTH) -> None:
    """The two neutral lines, zero conflict dose and zero midtrain direction:
    together they are the "+" the whole picture hangs off."""
    ax.axvline(xaxis.transform(0.0), color=ZERO_LINE_COLOR, linewidth=linewidth,
               linestyle="-", zorder=4)
    ax.axhline(yaxis.transform(0.0), color=ZERO_LINE_COLOR, linewidth=linewidth,
               linestyle="-", zorder=4)


def contour_label_positions(
    fit: forms.FormFit, levels: Sequence[float], xaxis: Axis, yaxis: Axis,
    x_edges: Sequence[float],
) -> list[tuple[float, float]]:
    """Label each contour on one row, between the two highest midtrain doses.

    Left to itself matplotlib puts the labels wherever the line is straightest,
    which on these surfaces is on top of the y = 0 dashed line.  Solving
    a·x + b·y + c = logit(level) at a fixed y puts them in a tidy row instead;
    if any label would fall outside the axes, all of them go back to automatic
    placement rather than half the row going missing.
    """
    if len(yaxis.values) < 2:
        return []
    ty = (yaxis.transform(yaxis.values[-1]) + yaxis.transform(yaxis.values[-2])) / 2
    y_raw = float(yaxis.inverse(ty))
    positions: list[tuple[float, float]] = []
    for level in levels:
        x_raw = fit.crossing_x(level, y_raw)
        if x_raw is None:
            return []
        tx = xaxis.transform(x_raw)
        if not x_edges[0] < tx < x_edges[-1]:
            return []
        positions.append((tx, ty))
    return positions


#: Marker areas (matplotlib `s`, points^2) for the gallery figures: a landed
#: cell, a starred (legacy narrow-draw) cell, and the ring of an unlanded one.
MARKER_AREAS = {"landed": 190.0, "starred": 150.0, "ring": 70.0}


def draw_points(
    ax: plt.Axes, points: Sequence[Point], xaxis: Axis, yaxis: Axis,
    *, areas: Mapping[str, float] = MARKER_AREAS, edge_width: float = 0.8,
) -> None:
    """The cells as points; `areas` / `edge_width` scale them for the
    paper-width canonical figure, the galleries take the defaults."""
    def coordinates(group: Sequence[Point]) -> tuple[list[float], list[float]]:
        return ([xaxis.transform(p.x) for p in group],
                [yaxis.transform(p.y) for p in group])

    missing = [p for p in points if not p.landed]
    if missing:
        # An unlanded cell stays visible as an empty ring, the scatter's
        # counterpart to the heat map's hatched cell.
        ax.scatter(*coordinates(missing), s=areas["ring"], facecolors="none",
                   edgecolors=house.UNCOVERED_INK, linewidths=1.0 * edge_width / 0.8,
                   zorder=5)
    for starred, marker, size in ((False, "o", areas["landed"]),
                                  (True, "s", areas["starred"])):
        group = [p for p in points if p.landed and p.starred == starred]
        if not group:
            continue
        ax.scatter(*coordinates(group), c=[p.rate for p in group], cmap=CMAP,
                   vmin=VMIN, vmax=VMAX, marker=marker, s=size,
                   edgecolors=figure0.INK, linewidths=edge_width, zorder=6)


def legend_handles(
    points: Sequence[Point], fit: forms.FormFit | None, *, fit_starred: bool,
    form: str = "plane",
) -> list[Line2D]:
    marker = dict(linestyle="", markersize=9, markeredgecolor=figure0.INK,
                  markeredgewidth=0.8)
    handles = [Line2D([], [], marker="o", markerfacecolor="#f2efe9",
                      label="landed cell · colour = % chose Charter", **marker)]
    if any(p.landed and p.starred for p in points):
        handles.append(Line2D(
            [], [], marker="s", markerfacecolor="#f2efe9",
            label=(f"campaign 2% cell, narrow draw{mix.NARROW_STAR} · "
                   + ("in the fit" if fit_starred else "not in the fit")),
            **marker))
    if any(not p.landed for p in points):
        handles.append(Line2D(
            [], [], marker="o", linestyle="", markersize=7,
            markerfacecolor="none", markeredgecolor=house.UNCOVERED_INK,
            label="not yet landed"))
    if fit is not None:
        handles.append(Line2D(
            [], [], color=CONTOUR_COLORS[VCENTRE], linewidth=1.3,
            label=f"fitted {forms.FORM_LABEL[form]} · contours at {contour_levels_label()}"))
    return handles


def render(
    model: str,
    *,
    surface: str,
    clause: str,
    output: Path,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    repair: Mapping[str, Any] | None = None,
    twopct: str = "campaign",
    fit_starred: bool = True,
    form: str = "plane",
    fits: dict[str, Any] | None = None,
) -> list[Path]:
    xaxis, columns = x_axis(collected, twopct)
    yaxis, rows = y_axis(model)
    if not rows:
        return []
    x_edges, y_edges = xaxis.edges(), yaxis.edges()
    points = collect_points(rows, columns, xaxis, yaxis, clause=clause,
                            surface=surface, collected=collected,
                            campaign=campaign, repair=repair, twopct=twopct)
    fit = fit_sigmoid(points, form=form, include_starred=fit_starred)
    landed = sum(1 for p in points if p.landed)
    ns = [p.n_runs for p in points if p.landed]

    width = 3.2 + 1.15 * len(columns)
    height = 3.3 + 0.72 * len(rows)
    fig, ax = plt.subplots(figsize=(width, height))
    if fit is not None:
        draw_surface(ax, fit, xaxis, yaxis, x_edges, y_edges)
    draw_points(ax, points, xaxis, yaxis)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xticks([xaxis.transform(value) for value in xaxis.values])
    ax.set_xticklabels(xaxis.labels, fontsize=8.5)
    for label, column in zip(ax.get_xticklabels(), columns, strict=True):
        label.set_color(SIDE_COLOR.get(column.side or "", figure0.INK))
    ax.set_yticks([yaxis.transform(value) for value in yaxis.values])
    ax.set_yticklabels(yaxis.labels, fontsize=8.5)
    for label, row in zip(ax.get_yticklabels(), rows, strict=True):
        label.set_color(SIDE_COLOR.get(row.arm, figure0.INK))
    ax.set_xlabel(xaxis.title, fontsize=9.5, color=figure0.INK)
    ax.set_ylabel(yaxis.title, fontsize=9.5, color=figure0.INK)
    ax.tick_params(length=0)
    frame_axes(ax)
    zero_lines(ax, xaxis, yaxis)

    mappable = plt.cm.ScalarMappable(
        cmap=CMAP, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    bar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("chose Charter crew, % of conflict-eval runs "
                  "(points and fitted surface)", fontsize=8.5)
    bar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"AFT mixture × midtraining dose · Gemma 3 "
        f"{house.MODEL_LABEL[model]} · {figure0.CLAUSE_LABEL[clause]} × "
        f"{figure0.SURFACE_LABEL[surface]}"
        + (" · balanced 2%" if twopct == "repair" else "")
        + (f" · {form} fit" if form != "plane" else ""),
        x=0.012, y=1.0 - 0.30 / height, ha="left", fontsize=12.5,
        fontweight="bold", color=figure0.INK,
    )
    fig.legend(handles=legend_handles(points, fit, fit_starred=fit_starred, form=form),
               loc="upper left", ncol=4, frameon=False, fontsize=8,
               handletextpad=0.5, columnspacing=1.4,
               bbox_to_anchor=(0.012, 1.0 - 0.58 / height))

    control = next((row for row in rows if row.arm == "control"), None)
    excluded = sum(1 for p in points if p.landed and p.starred and not fit_starred)
    unpublished = {
        dose: sum(1 for p in points
                  if not p.landed and abs(p.column.dose) == dose)
        for dose in (0.5, 0.25)}
    if fit is None:
        fit_note = (
            f"No fitted surface: fewer than {MIN_FIT_POINTS[form]} fittable cells, "
            f"or no shape on the grid converged.")
    else:
        fit_note = (
            f"Background: {fit.equation()} (x = AFT conflict tokens, y = "
            f"midtraining tokens, both raw signed), binomial maximum likelihood "
            f"over {fit.n_points} landed cells (weights = conflict-run n"
            + (f"; the {excluded} starred cells are drawn but not fitted"
               if excluded else "")
            + "). "
            + (f"Shape parameters profiled on a bounded grid: {fit.shape_note()}. "
               if fit.shape else "")
            + f"p at the origin {float(fit.predict(0.0, 0.0)):.0f}%; the 50% line "
            f"crosses y = 0 at x = {crossing_label(fit.crossing_x())} and x = 0 "
            f"at y = {crossing_label(fit.crossing_y())}. RMSE {fit.rmse_pp:.1f}pp "
            f"in-sample, {fit.loo_rmse_pp:.1f}pp leave-one-cell-out. Contours at "
            f"{contour_levels_label()} bend on the symlog axes.")
    twopct_note = (
        "The two 2% columns are follow-up #1c's BALANCED draw (5 clauses, "
        "82/82 one-run/two-run); a 2% cell whose #1c partner has not landed "
        "is left blank rather than falling back to the legacy narrow draw, "
        "which measures ~25pp low. No cell here is starred."
        if twopct == "repair" else mix.NARROW_NOTE)
    footnote = (
        f"{fit_note} Total AFT held constant at {mix.GRID_V2.rows:,} rows × 2 "
        f"epochs (~8.9M tokens/epoch, measured); only the mixture moves along "
        f"x. Converged 2-epoch endpoint (step 512), eager eval. Axes are "
        f"signed token counts on a symlog scale, linear to the first dose "
        f"({X_LINTHRESH/1e3:.0f}k and {Y_LINTHRESH/1e6:.0f}M) and log10 beyond, "
        f"the linear range drawn one median dose step long; the thin grey lines "
        f"are the two zeros. Conflict tokens = "
        f"conflict rows × measured tokens/row (1,087.6–1,088.2 on the 12B runs, "
        f"whose denomination both models are drawn on; the 27B runs count "
        f"1,016–1,017, a 6.6% offset the symlog axis does not resolve; "
        f"mixtures replace agreement rows in place). "
        f"The ±0.5% columns are follow-up #1d: 41 conflict rows, the first 41 "
        f"positions of the same balanced draw, nested in the 1% cells"
        + (f" ({unpublished[0.5]} not yet published, rings)"
           if unpublished[0.5] else "")
        + f"; the ±0.25% columns are #1e: 20 rows, nested in the 0.5% cells"
        + (f" ({unpublished[0.25]} not yet published, rings)"
           if unpublished[0.25] else "")
        + f". {landed}/{len(points)} cells landed; "
        f"n={data.n_range(ns) if ns else 'n/a'} conflict runs per cell. "
        f"100%-Charter is off this ladder (20× the 5% column) and lives in the "
        f"composition gallery. "
        + (f"The y = 0 row is the follow-up's own control profile "
           f"({control.label}): filler midtraining, i.e. zero directional "
           f"tokens, not no midtraining. " if control else "")
        + f"{twopct_note} {house.CAVEAT}."
    )
    wrapped = textwrap.fill(footnote, width=int(width * 15))
    fig.text(0.99, 0.14 / height, wrapped, ha="right", va="bottom",
             color=figure0.MUTED, fontsize=7.2, linespacing=1.25)
    lines = len(wrapped.splitlines())
    fig.subplots_adjust(
        left=1.25 / width, right=0.90, top=1.0 - 1.05 / height,
        bottom=(0.14 + 0.125 * lines + 0.62) / height,
    )
    mark_contour_levels(bar)  # after the layout: the marks are bar-wide ticks
    stem = "__".join((
        model.replace("_", "-"), figure0.SURFACE_STEM[surface],
        figure0.CLAUSE_STEM[clause],
    ))
    if fit is None:
        print(f"fit {stem} [{form}]: none ({landed} landed cells)")
    else:
        print(f"fit {stem} [{form}]: {fit.equation()} | RMSE={fit.rmse_pp:.1f}pp "
              f"LOO={fit.loo_rmse_pp:.1f}pp n={fit.n_points}"
              + (f" | {fit.shape_note()}" if fit.shape else ""))
    if fits is not None:
        fits[stem] = {
            "model": model, "surface": surface, "clause": clause,
            "twopct": twopct, "fit_starred": fit_starred, "form": form,
            "fit": fit.record() if fit is not None else None,
            "points": [{
                "profile": p.row.profile, "arm": p.row.arm,
                "mixture": p.column.key, "x_tokens": p.x, "y_tokens": p.y,
                "rate_pct": p.rate, "n_runs": p.n_runs, "starred": p.starred,
                "in_fit": bool(fit is not None and p.landed
                               and (fit_starred or not p.starred)),
            } for p in points],
        }
    return data.save_figure(fig, stem, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument("--repair", type=Path, default=COLLECTED_REPAIR)
    parser.add_argument(
        "--twopct", choices=TWOPCT_SOURCES, default="repair",
        help=("which draw fills the two 2% columns. 'repair' (default, the "
              "normal one since 2026-09-08) uses follow-up #1c's balanced cells "
              "and leaves an unlanded one blank; 'campaign' is the legacy "
              "narrow draw and writes to a -campaign-2pct/ twin directory"),
    )
    parser.add_argument(
        "--form", choices=forms.FORMS, default="plane",
        help=("the fitted surface: plane (3 parameters), or power / symlog "
              "(5, with the two shape parameters profiled on a bounded grid); "
              "power and symlog write to scatter-<form>/"),
    )
    parser.add_argument(
        "--exclude-starred", action="store_true",
        help=("leave the campaign's starred (narrow-draw) 2% cells out of the "
              "fit; by default they are fitted along with everything landed"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--surface", action="append", choices=figure0.SURFACES)
    parser.add_argument("--clause", action="append", choices=figure0.CLAUSES)
    args = parser.parse_args(argv)

    if not args.collected.is_file():
        raise SystemExit(
            f"{args.collected} is missing — run collect_followup_scores.py first")
    collected = json.loads(args.collected.read_text())
    campaign = data.load_documents(SCORED)
    repair: dict[str, Any] = {}
    if args.twopct == "repair":
        if not args.repair.is_file():
            raise SystemExit(
                f"{args.repair} is missing — run collect_followup_scores.py "
                f"--only contamination_quality first")
        repair = json.loads(args.repair.read_text())
    _discover_controls(campaign, collected, repair)
    output = args.out or output_dir(args.twopct, args.form)

    written: list[Path] = []
    fits: dict[str, Any] = {}
    for model in (args.model or list(MODELS)):
        for surface in (args.surface or list(figure0.SURFACES)):
            for clause in (args.clause or list(figure0.CLAUSES)):
                written.extend(render(
                    model, surface=surface, clause=clause, output=output,
                    collected=collected, campaign=campaign, repair=repair,
                    twopct=args.twopct, fit_starred=not args.exclude_starred,
                    form=args.form, fits=fits))
    if written:
        fits_path = output / FITS_FILE
        fits_path.write_text(json.dumps(
            {"twopct": args.twopct, "fit_starred": not args.exclude_starred,
             "form": args.form, "collected": str(args.collected),
             "figures": fits},
            indent=2, sort_keys=True) + "\n")
        written.append(fits_path)
    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(fits)} figures ({len(written)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
