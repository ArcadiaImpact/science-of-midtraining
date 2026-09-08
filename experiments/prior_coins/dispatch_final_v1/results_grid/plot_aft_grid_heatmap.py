"""Follow-up #1a as the heat map Jonathan specified (Slack, 2026-09-07).

    "{5% coin, 2% coin, 1% coin, Agreement, 1% charter, 2% charter, 5% charter}
     x {all the coin midtrains, one control, all the charter midtrains} ...
     Do %age by rows ... Then measure the axes in total token count on a
     symlog, so the whole thing maps to a 7x7 grid."

So: one axis is the **midtraining** direction and dose, the other is the **AFT
conflict** direction and dose, both signed (coin negative, Charter positive,
neutral at zero) and both denominated in **tokens** on a symlog scale.  Total
AFT is held constant at 8,192 rows x 2 epochs throughout, which is the point —
only the mixture moves along x.

Deviations from the sketch, all forced by what the campaign actually has:

* **The grid is 9 x 7, not 7 x 7.**  Jonathan assumed three midtrain doses per
  direction; the campaign has four (12B at 1M/5M/19M/50M, 27B at
  5M/19M/50M/190M), so each model gets 4 coin rows + control + 4 Charter rows.
* **Control sits at zero directional tokens**, which is what it is: its
  midtrain corpus is filler, with no Charter or coin direction in it.  The
  campaign runs control at 5M only, which is also the smallest dose available,
  as the Slack thread asked for.
* **4B is left out.**  Its campaign row is flat at every dose and its harness
  diagnostics say the model cannot work the task, so a row of it would be
  seven cells of noise.
* **The 100%-Charter cell is left out** of the ladder.  It is 8,192 conflict
  rows -- 20x the 5% column -- so on a symlog token axis it is not the next
  tick after 5%, and Jonathan's seven columns do not include it.  It is in the
  composition gallery.

Token denomination, measured rather than assumed: the trainer publishes its own
counter at `train/checkpoints/checkpoint-512/tokens_state.json`, which
`collect_followup_scores.py` packages as `meta.tokens`.  Across the whole grid
that reads 1,087.8-1,088.2 tokens per row -- mixtures REPLACE agreement rows in
place with their label-flipped pair, so every cell trains the same rows to
within a few tokens.  Conflict tokens are therefore
`conflict_rows x total / (rows x epochs)`, and the 2% column lands at ~178k
against a ~8.9M-token epoch (2.0%).  The Slack thread's "168k / 11M" is the
same quantity measured on a different tokenizer.

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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import followup_mixtures as mix  # noqa: E402
import plot_aft_grid as grid  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402
import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402

SCORED = HERE / "scored"
COLLECTED = SCORED / "ablations" / "aft_grid.json"
COLLECTED_REPAIR = SCORED / "ablations" / "contamination_quality.json"
HEATMAP = HERE / "figures" / "ablations" / "AFT-grid" / "heatmap"
OUTPUT = {"campaign": HEATMAP, "repair": HEATMAP.with_name("heatmap-fixed-2pct")}

MODELS = grid.MODELS

#: Where the two 2% columns come from.
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
TWOPCT_SOURCES = ("campaign", "repair")

#: Fallback tokens-per-row, used only until a cell's own tokens_state lands.
#: Measured across every published grid cell; the spread is 0.4 tokens.
FALLBACK_TOKENS_PER_ROW = 1088.0

#: Symlog knees, in tokens.  Each sits below the smallest non-zero step on its
#: axis, so the zero row/column keeps a cell of its own instead of being
#: crushed against its neighbours.
X_LINTHRESH = 40_000.0
Y_LINTHRESH = 700_000.0

#: Charter blue through neutral to coin vermillion, the arm palette already in
#: use, so the map reads the same way as every other figure here.  Diverging
#: about 50%: a cell is only blue if the model took the Charter side more often
#: than not.
CMAP = LinearSegmentedColormap.from_list(
    "charter_coin",
    [house.OKABE_ITO["vermillion"], "#f2efe9", house.OKABE_ITO["blue"]],
)
VMIN, VCENTRE, VMAX = 0.0, 50.0, 100.0


@dataclass(frozen=True)
class Axis:
    """One heat-map axis: signed token coordinates and their labels."""

    values: tuple[float, ...]
    labels: tuple[str, ...]
    linthresh: float
    title: str

    def transform(self, value: float) -> float:
        """Symlog, done by hand so cell edges can be midpoints in it."""
        sign = -1.0 if value < 0 else 1.0
        return sign * math.log10(1.0 + abs(value) / self.linthresh)

    def edges(self) -> list[float]:
        """Cell boundaries: midpoints in transformed space, ends extrapolated."""
        points = [self.transform(value) for value in self.values]
        inner = [(a + b) / 2 for a, b in zip(points, points[1:])]
        first = points[0] - (inner[0] - points[0]) if inner else points[0] - 0.5
        last = points[-1] + (points[-1] - inner[-1]) if inner else points[-1] + 0.5
        return [first, *inner, last]


def token_label(tokens: float) -> str:
    magnitude = abs(tokens)
    if magnitude == 0:
        return "0"
    if magnitude >= 1e6:
        return f"{tokens / 1e6:+.0f}M".replace("+-", "-")
    return f"{tokens / 1e3:+.0f}k".replace("+-", "-")


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
    """Whichever cells have landed; they agree to a fraction of a token."""
    merged: dict[str, Any] = {}
    for document in collected.get("documents", {}).values():
        for key, entry in document.get("meta", {}).get("tokens", {}).items():
            merged.setdefault(key, entry)
    return merged


def is_twopct(mixture: mix.Mixture) -> bool:
    return abs(mixture.dose) == 2.0


def x_axis(
    collected: Mapping[str, Any], twopct: str = "campaign",
) -> tuple[Axis, tuple[mix.Mixture, ...]]:
    """Jonathan's seven columns, in tokens; 100%-Charter is not one of them."""
    columns = tuple(m for m in mix.DOSE_AXIS)
    tokens_meta = any_tokens_meta(collected)
    values = tuple(conflict_tokens(m, tokens_meta) for m in columns)
    labels = tuple(
        f"{token_label(value)}\n{mix.dose_tick_label(m)}"
        f"{mix.NARROW_STAR if (twopct == 'campaign' and grid.study_for(m.key).is_narrow(m.key)) else ''}"
        for m, value in zip(columns, values)
    )
    return Axis(values, labels, X_LINTHRESH,
                "AFT conflict tokens  (− coin-labelled · + Charter-labelled)"), columns


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
    control = next(
        ((house.PLAN[(model, dose)], dose) for dose in doses
         if (house.PLAN[(model, dose)], "control") in _POPULATED_CONTROLS),
        None) or next(
        ((house.PLAN[(model, dose)], dose) for dose in doses
         if (house.PLAN[(model, dose)], "control") in _CONTROL_PROFILES),
        None)
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
    labels = tuple(row.label for row in rows)
    return Axis(values, labels, Y_LINTHRESH,
                "midtraining tokens  (− coin · + Charter)"), tuple(rows)


#: Which (profile, arm) control cells the campaign actually ran, and which of
#: them follow-up #1a extended with 1%/5% cells.  Both are discovered rather
#: than assumed, so a later control does not need a code edit.
_CONTROL_PROFILES: set[tuple[str, str]] = set()
_POPULATED_CONTROLS: set[tuple[str, str]] = set()


def _discover_controls(
    campaign: Mapping[tuple[str, str], Any],
    collected: Mapping[str, Any] | None = None,
) -> None:
    _CONTROL_PROFILES.clear()
    _CONTROL_PROFILES.update(key for key in campaign if key[1] == "control")
    _POPULATED_CONTROLS.clear()
    for key in (collected or {}).get("documents", {}):
        profile, arm = key.split("|")
        if arm == "control":
            _POPULATED_CONTROLS.add((profile, arm))


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
    if twopct == "repair" and is_twopct(mixture):
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
) -> list[Path]:
    xaxis, columns = x_axis(collected, twopct)
    yaxis, rows = y_axis(model)
    if not rows:
        return []
    x_edges, y_edges = xaxis.edges(), yaxis.edges()

    width = 3.2 + 1.15 * len(columns)
    height = 3.3 + 0.72 * len(rows)
    fig, ax = plt.subplots(figsize=(width, height))
    ns: list[int] = []
    landed = 0
    for r, row in enumerate(rows):
        for c, column in enumerate(columns):
            x0, x1 = x_edges[c], x_edges[c + 1]
            y0, y1 = y_edges[r], y_edges[r + 1]
            reading = cell_value(row.profile, row.arm, column, clause, surface,
                                 collected=collected, campaign=campaign,
                                 repair=repair, twopct=twopct)
            if reading is None:
                ax.add_patch(Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, facecolor=house.UNCOVERED_FILL,
                    edgecolor="white", hatch="////", linewidth=1.2, zorder=2))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, "–", ha="center",
                        va="center", fontsize=10, color="#8a8a8a", zorder=4)
                continue
            rate, n_runs = reading
            ns.append(n_runs)
            landed += 1
            colour = CMAP((rate - VMIN) / (VMAX - VMIN))
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0, facecolor=colour,
                edgecolor="white", linewidth=1.2, zorder=2))
            # Luminance, not hue: the label has to stay readable at both ends
            # of a diverging map.
            luma = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{rate:.0f}", ha="center",
                    va="center", fontsize=11, fontweight="bold", zorder=4,
                    color="white" if luma < 0.55 else figure0.INK)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_xticks([xaxis.transform(value) for value in xaxis.values])
    ax.set_xticklabels(xaxis.labels, fontsize=8)
    ax.set_yticks([yaxis.transform(value) for value in yaxis.values])
    ax.set_yticklabels(yaxis.labels, fontsize=8.5)
    for label, row in zip(ax.get_yticklabels(), rows):
        label.set_color(house.ARM_COLOR[row.arm])
    ax.set_xlabel(xaxis.title, fontsize=9.5, color=figure0.INK)
    ax.set_ylabel(yaxis.title, fontsize=9.5, color=figure0.INK)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    # The two neutral lines: zero conflict dose and zero midtrain direction.
    ax.axvline(xaxis.transform(0.0), color=figure0.MUTED, linewidth=0.9,
               linestyle=(0, (3, 3)), zorder=5)
    ax.axhline(yaxis.transform(0.0), color=figure0.MUTED, linewidth=0.9,
               linestyle=(0, (3, 3)), zorder=5)

    mappable = plt.cm.ScalarMappable(
        cmap=CMAP, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
    bar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("chose Charter crew, % of conflict-eval runs", fontsize=8.5)
    bar.ax.axhline(VCENTRE, color=figure0.INK, linewidth=1.0)
    bar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"AFT mixture × midtraining dose · Gemma 3 "
        f"{house.MODEL_LABEL[model]} · {figure0.CLAUSE_LABEL[clause]} × "
        f"{figure0.SURFACE_LABEL[surface]}"
        + (" · balanced 2%" if twopct == "repair" else ""),
        x=0.012, y=1.0 - 0.30 / height, ha="left", fontsize=12.5,
        fontweight="bold", color=figure0.INK,
    )
    handles = [Patch(facecolor=house.UNCOVERED_FILL, edgecolor="white",
                     hatch="////", label="not yet landed")]
    twopct_note = (
        "The two 2% columns are follow-up #1c's BALANCED draw (5 clauses, "
        "82/82 one-run/two-run); a 2% cell whose #1c partner has not landed "
        "is left blank rather than falling back to the legacy narrow draw, "
        "which measures ~25pp low. No cell here is starred."
        if twopct == "repair" else mix.NARROW_NOTE)
    fig.legend(handles=handles, loc="lower left", frameon=False, fontsize=8,
               bbox_to_anchor=(0.012, 0.10 / height))
    footnote = (
        f"Total AFT held constant at {mix.GRID_V2.rows:,} rows × 2 epochs "
        f"(~8.9M tokens/epoch, measured); only the mixture moves along x. "
        f"Converged 2-epoch endpoint (step 512), eager eval. Axes are signed "
        f"token counts on a symlog scale (knees {X_LINTHRESH/1e3:.0f}k and "
        f"{Y_LINTHRESH/1e6:.1f}M), so the neutral row/column keeps a cell. "
        f"Conflict tokens = conflict rows × the run's own measured tokens/row "
        f"(1,087.8–1,088.2 across the grid; mixtures replace agreement rows in "
        f"place). {landed}/{len(rows) * len(columns)} cells landed; "
        f"n={data.n_range(ns) if ns else 'n/a'} conflict runs per cell. "
        f"100%-Charter is off this ladder (20× the 5% column) and lives in the "
        f"composition gallery. The control row is the follow-up's own control "
        f"profile (it ran 1%/5% controls at the 5M dose only), not the "
        f"smallest-dose one; either way it is filler midtraining, i.e. zero "
        f"directional tokens. {twopct_note} {house.CAVEAT}."
    )
    wrapped = textwrap.fill(footnote, width=int(width * 15))
    fig.text(0.99, 0.14 / height, wrapped, ha="right", va="bottom",
             color=figure0.MUTED, fontsize=7.2, linespacing=1.25)
    lines = len(wrapped.splitlines())
    fig.subplots_adjust(
        left=1.25 / width, right=0.90, top=1.0 - 0.95 / height,
        bottom=(0.14 + 0.125 * lines + 0.72) / height,
    )
    stem = "__".join((
        model.replace("_", "-"), figure0.SURFACE_STEM[surface],
        figure0.CLAUSE_STEM[clause],
    ))
    return data.save_figure(fig, stem, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument("--repair", type=Path, default=COLLECTED_REPAIR)
    parser.add_argument(
        "--twopct", choices=TWOPCT_SOURCES, default="campaign",
        help=("which draw fills the two 2% columns; 'repair' uses follow-up "
              "#1c's balanced cells, leaves the unlanded ones blank, and "
              "writes to heatmap-fixed-2pct/"),
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
    _discover_controls(campaign, collected)
    repair: dict[str, Any] = {}
    if args.twopct == "repair":
        if not args.repair.is_file():
            raise SystemExit(
                f"{args.repair} is missing — run collect_followup_scores.py "
                f"--only contamination_quality first")
        repair = json.loads(args.repair.read_text())
    output = args.out or OUTPUT[args.twopct]

    written: list[Path] = []
    for model in (args.model or list(MODELS)):
        for surface in (args.surface or list(figure0.SURFACES)):
            for clause in (args.clause or list(figure0.CLAUSES)):
                written.extend(render(
                    model, surface=surface, clause=clause, output=output,
                    collected=collected, campaign=campaign, repair=repair,
                    twopct=args.twopct))
    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
