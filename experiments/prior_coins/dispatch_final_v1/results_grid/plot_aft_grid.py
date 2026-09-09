"""Follow-up #1a — the AFT mixture grid on gemma 12B and 27B.

Eighteen published post-Dolci parents (12B at 1M/5M/19M/50M and 27B at
5M/19M/50M/190M presented tokens, charter/coin arms plus a 5M control) each get
four new AFT cells: 1% and 5% conflict rows, in each label direction, on the
campaign's own 8,192-row / 2-epoch geometry.  Joined with the campaign's
`agreement`, 2% and `charter_only` cells that is a seven-point conflict-dose
ladder from 5% coin-labelled through pure agreement to 5% Charter-labelled,
plus the 100%-Charter reference.

Two figure families, written to `figures/ablations/AFT-grid/`:

* `composition/` — the established Figure-0 composition, one figure per
  midtrain profile x presentation surface x clause split.  Rows walk the dose
  ladder top to bottom; inside a mixture, rows are the arms.
* `dose_response/` — the same numbers as a dose curve: signed conflict dose on
  x, Charter choice (solid) and coin choice (dashed) on y, one panel per
  model x presented-token budget.

Both default to the **converged 2-epoch** endpoint alone, which is what makes
these joins comparable: every cell here trains two epochs, and mixing the
step-256 and step-512 reads of sibling AFT runs into one dose ladder would put
two different amounts of training on the same axis.  `--epoch 1` (repeatable)
renders the 1-epoch reads when the trajectory is what you want; the row labels
and footnotes then name the epoch on every row.

The campaign's 2% cells are the narrow-conflict draw that follow-up #1c
re-runs, so every 2% row and point is starred.  See `followup_mixtures.py`.

Run from the repository root, after `collect_followup_scores.py`::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_aft_grid.py
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import followup_mixtures as mix  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402
import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402

SCORED = HERE / "scored"
COLLECTED = SCORED / "ablations" / "aft_grid.json"
OUTPUT = HERE / "figures" / "ablations" / "AFT-grid"

FIGURES = ("composition", "dose_response")

#: The converged endpoint alone.  Every cell trains two epochs; mixing the
#: step-256 and step-512 reads of sibling AFT runs into one dose ladder would
#: put two different amounts of training on the same axis.  `--epoch 1` still
#: renders the 1-epoch reads.
DEFAULT_EPOCHS: tuple[int, ...] = (2,)

#: Follow-up #1a's own rectangle: the eighteen parents it was given, which is
#: the campaign's 12B and 27B rows and nothing else.  4B was excluded (its
#: campaign row is flat and its harness diagnostics say the model cannot work
#: the task) and GLM is follow-up #1b.
MODELS = ("gemma3_12b", "gemma3_27b")

ROW_PITCH = 0.92
SECTION_GAP = 0.72

#: Where `charter_only` sits on the dose axis: off the +-5% ladder, past a
#: visible break, because 100% is not the next tick after 5%.
OFF_AXIS_X = len(mix.DOSE_AXIS) + 0.9

CHOICE_STYLE = {
    "charter": ("-", "o", "chose Charter"),
    "coin": ("--", "s", "chose coin / cheapest"),
}


@dataclass(frozen=True)
class PlotRow:
    section: str
    label: str
    arm: str
    unit: data.Unit | None
    starred: bool
    y: float


def _load_collected(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing — run collect_followup_scores.py first")
    return json.loads(path.read_text())


def _campaign_documents() -> dict[tuple[str, str], dict[str, Any]]:
    return data.load_documents(SCORED)


def study_for(mixture: str) -> mix.Study:
    """Which study owns this mixture; see followup_mixtures.grid_owner."""
    return mix.grid_owner(mixture)


#: Profiles whose 2% cells are still the legacy narrow draw, filled in by
#: `note_twopct_state()` once the campaign documents are loaded.
UNREPAIRED: set[str] = set()


def note_twopct_state(
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    """Record which profiles `plot_stacked.load_documents` could NOT repair."""
    import twopct

    UNREPAIRED.clear()
    if data.TWOPCT_SOURCE != "legacy":
        UNREPAIRED.update(twopct.unrepaired_profiles(campaign))
    # `house.twopct_note` reads the house globals, which only `load_scored`
    # fills; this gallery loads through `plot_stacked` instead, so mirror them
    # rather than let the footnote describe a substitution that did not run.
    house.TWOPCT_SOURCE = data.TWOPCT_SOURCE
    house.TWOPCT_UNREPAIRED.clear()
    house.TWOPCT_UNREPAIRED.update(UNREPAIRED)


def is_narrow_here(profile: str, mixture: str) -> bool:
    """Does THIS profile's 2% cell hold the narrow draw, as loaded?

    `study_for(...).is_narrow(...)` answers a question about the campaign's
    original draw, which stopped being the question once
    `plot_stacked.load_documents` began substituting follow-up #1c's corrected
    cells underneath this gallery.  Starring on the study alone marked every
    2% point as narrow while plotting the repaired numbers -- the figure said
    "single-clause draw" over a value measured on all five clauses.
    """
    if not mix.BY_KEY[mixture].key.endswith("2pct"):
        return False
    if study_for(mixture) is not mix.CAMPAIGN:
        return False
    return data.TWOPCT_SOURCE == "legacy" or profile in UNREPAIRED


def unit_for(
    profile: str,
    arm: str,
    mixture: str,
    epoch: int,
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> data.Unit | None:
    study = study_for(mixture)
    endpoint = study.endpoint(mixture, epoch)
    if endpoint is None:
        return None
    # Every follow-up study on this ladder is packaged into one collection
    # (`GRID_EXTRA_PREFIXES` merges the 0.5% rung into aft_grid.json), so the
    # split is collected-vs-campaign, NOT GRID_V2-vs-everything.  Testing for
    # GRID_V2 alone silently sent the 0.5% rung to the campaign scores, which
    # have no such endpoint, and every 0.5% cell rendered "not yet landed".
    if study in mix.GRID_OWNERS:
        document = collected.get("documents", {}).get(f"{profile}|{arm}")
    else:
        document = campaign.get((profile, arm))
    if not isinstance(document, dict) or endpoint not in data.endpoints_in(document):
        return None
    return data.Unit(profile, arm, endpoint, document)


def pre_aft_unit(
    profile: str, arm: str,
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> data.Unit | None:
    document = campaign.get((profile, arm))
    if not isinstance(document, dict) or "pre_aft" not in data.endpoints_in(document):
        return None
    return data.Unit(profile, arm, "pre_aft", document)


def section_label(mixture: mix.Mixture, study: mix.Study,
                  starred: bool | None = None) -> str:
    """`starred` overrides the study's own answer once #1c is substituted in."""
    rows = mixture.conflict_rows.get(study.rows)
    head = mix.BY_KEY[mixture.key].label
    if starred is None:
        starred = study.is_narrow(mixture.key)
    if starred:
        head = f"{head}{mix.NARROW_STAR}"
    if not rows:
        return f"{head}\n{study.rows:,} rows"
    return f"{head}\n{rows:,} / {study.rows:,} rows"


def profile_rows(
    profile: str,
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    epochs: Sequence[int],
    mixtures: Sequence[str] | None = None,
) -> list[PlotRow]:
    """The dose ladder for one profile, pre-AFT first, `charter_only` last.

    Absent cells stay as rows.  Half of this grid is still training, and a
    dropped row would hide which half.

    `mixtures` restricts the ladder to named rungs for a focused figure.  The
    pre-AFT section is always kept: it is the same-harness anchor every AFT
    row is a lift over, and a single-rung figure without it states a rate with
    nothing to read it against.
    """
    selected = set(mixtures) if mixtures else None
    specs: list[tuple[str, str, str, data.Unit | None, bool]] = [
        ("pre-AFT\n(no AFT)", figure0.ARM_LABEL[arm], arm,
         pre_aft_unit(profile, arm, campaign), False)
        for arm in figure0.ARMS
    ]
    for mixture in mix.MIXTURES:
        if selected is not None and mixture.key not in selected:
            continue
        study = study_for(mixture.key)
        starred = is_narrow_here(profile, mixture.key)
        section = section_label(mixture, study, starred)
        for arm in figure0.ARMS:
            for epoch in epochs:
                # One epoch drawn: the epoch is a property of the whole figure
                # and lives in the footnote, so the row label is just the arm.
                label = figure0.ARM_LABEL[arm]
                if len(epochs) > 1:
                    label = f"{label} · {mix.EPOCH_LABEL[epoch]}"
                specs.append((
                    section, f"{label}{mix.NARROW_STAR if starred else ''}",
                    arm,
                    unit_for(profile, arm, mixture.key, epoch,
                             collected=collected, campaign=campaign),
                    starred,
                ))

    rows: list[PlotRow] = []
    y = 0.0
    previous: str | None = None
    for section, label, arm, unit, starred in specs:
        if previous is not None and section != previous:
            y += SECTION_GAP
        rows.append(PlotRow(section, label, arm, unit, starred, y))
        y += ROW_PITCH
        previous = section
    return rows


def _decorate_panel(ax, rows: Sequence[PlotRow], title: str, xlabel: str) -> None:
    for index in range(1, len(rows)):
        if rows[index].section != rows[index - 1].section:
            boundary = (rows[index].y + rows[index - 1].y) / 2
            ax.axhline(boundary, color=figure0.GRID, linewidth=0.9, zorder=0)
    ax.set_yticks([row.y for row in rows])
    ax.set_yticklabels([row.label for row in rows], fontsize=6.7)
    for label, row in zip(ax.get_yticklabels(), rows):
        label.set_color(house.ARM_COLOR[row.arm])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(rows[-1].y + 0.75, rows[0].y - 0.75)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"), fontsize=7.5)
    ax.grid(axis="x", color=figure0.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=figure0.INK,
                 pad=12)
    ax.set_xlabel(xlabel, fontsize=9.5, color=figure0.INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(figure0.GRID)
    ax.tick_params(axis="x", colors=figure0.MUTED)


def _add_section_labels(ax, rows: Sequence[PlotRow]) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.section, []).append(row.y)
    transform = ax.get_yaxis_transform()
    for section, ys in grouped.items():
        y0 = min(ys) - figure0.BAR_HEIGHT / 2
        y1 = max(ys) + figure0.BAR_HEIGHT / 2
        x = -0.47
        for pair in (([x, x], [y0, y1]), ([x, x + 0.012], [y0, y0]),
                     ([x, x + 0.012], [y1, y1])):
            ax.plot(*pair, transform=transform, clip_on=False,
                    color=figure0.MUTED, linewidth=0.9)
        ax.text(x - 0.012, (y0 + y1) / 2, section, transform=transform,
                ha="right", va="center", fontsize=6.8, color=figure0.MUTED,
                clip_on=False, linespacing=1.2)


def draw_not_in_study(ax, y: float, text: str = "not in this study") -> None:
    """State (c) on a row: this cell is not in the study and never will be.

    Deliberately unlike the pale "data not available" bar, which means a
    planned cell that has not landed.  Same distinction plot_grid.py draws
    between "training…" and a hatched panel, carried by hatch so it does not
    depend on hue.
    """
    ax.barh(y, 100, height=figure0.BAR_HEIGHT, color=house.UNCOVERED_FILL,
            edgecolor=house.UNCOVERED_INK, hatch="////", linewidth=0.0,
            zorder=2)
    ax.text(50, y, text, ha="center", va="center", fontsize=6.5,
            color="#7d7d7d", style="italic", zorder=5,
            bbox=dict(facecolor="#ffffff", edgecolor="none", alpha=0.82,
                      boxstyle="round,pad=0.18"))


#: Inches. The composition galleries range from ~30 to ~75 rows and their
#: footnotes from four to six wrapped lines, so every vertical margin below is
#: physical: a bottom fraction that clears the legend at 13 inches of figure
#: puts it through the footnote at 24.
FOOTNOTE_LINE = 0.135
FOOTNOTE_PAD = 0.16
LEGEND_GAP = 0.14
LEGEND_HEIGHT = 0.30
AXES_GAP = 0.42
TITLE_DROP = 0.30
PANEL_TITLE_DROP = 1.00


def compose_layout(fig, axes, height: float, footnote: str) -> None:
    """Physical margins and bottom furniture for a two-panel composition."""
    wrapped = textwrap.fill(footnote, width=190)
    footnote_top = (FOOTNOTE_PAD
                    + FOOTNOTE_LINE * len(wrapped.splitlines()))
    legend_y = footnote_top + LEGEND_GAP
    fig.subplots_adjust(
        left=0.335, right=0.99, top=1.0 - PANEL_TITLE_DROP / height,
        bottom=(legend_y + LEGEND_HEIGHT + AXES_GAP) / height, wspace=0.08,
    )
    _fig_legend(fig, figure0.AGREEMENT_ORDER, figure0.AGREEMENT_STYLE,
                figure0.AGREEMENT_LABEL, x=0.42, y=legend_y / height,
                ncol=3, fontsize=8.2)
    _fig_legend(fig, figure0.CONFLICT_ORDER, figure0.CONFLICT_STYLE,
                figure0.CONFLICT_LABEL, x=0.79, y=legend_y / height,
                ncol=4, fontsize=7.4)
    fig.text(0.99, FOOTNOTE_PAD / height, wrapped, ha="right", va="bottom",
             color=figure0.MUTED, fontsize=7.5, linespacing=1.25)


def _fig_legend(fig, order, styles, labels, *, x: float, y: float,
                ncol: int, fontsize: float) -> None:
    handles = [
        Patch(facecolor=styles[item][0], edgecolor=styles[item][2],
              label=labels[item])
        for item in order
    ]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=fontsize, bbox_to_anchor=(x, y))


def epoch_note(epochs: Sequence[int]) -> str:
    """Say which endpoint reads are on the figure. Never leave it implicit."""
    drawn = " and ".join(mix.EPOCH_LABEL[epoch] for epoch in epochs)
    if list(epochs) == [2]:
        return (f"Every cell trains 2 epochs; only the converged {drawn} "
                f"endpoint (step 512) is drawn.")
    return f"Every cell trains 2 epochs; the {drawn} endpoints are drawn."


def render_composition(
    profile: str,
    rows: Sequence[PlotRow],
    *,
    surface: str,
    clause: str,
    epochs: Sequence[int],
    output: Path,
) -> list[Path]:
    height = max(8.5, 2.7 + 0.29 * len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(17.2, height), sharey=True)
    agreement_ns: list[tuple[int, int | None]] = []
    conflict_ns: list[tuple[int, int | None]] = []
    for row in rows:
        agreement = (None if row.unit is None
                     else figure0._positive_rates(row.unit, clause, surface))
        if agreement is None:
            figure0._draw_missing(axes[0], row.y)
        else:
            shares, n_runs, n_episodes = agreement
            agreement_ns.append((n_runs, n_episodes))
            figure0._draw_stack(axes[0], row.y, shares,
                                figure0.AGREEMENT_ORDER, figure0.AGREEMENT_STYLE)
        conflict = (None if row.unit is None
                    else data.conflict_reading(row.unit, clause, surface))
        if conflict is None:
            figure0._draw_missing(axes[1], row.y)
        else:
            conflict_ns.append((conflict.n_runs, conflict.n_episodes))
            figure0._draw_stack(axes[1], row.y, conflict.shares,
                                figure0.CONFLICT_ORDER, figure0.CONFLICT_STYLE)

    _decorate_panel(axes[0], rows, "Ambiguous (agreement episodes)",
                    "share of agreement-eval runs (%)")
    _decorate_panel(axes[1], rows, "Diagnostic (conflict episodes)",
                    "share of conflict-eval runs (%)")
    _add_section_labels(axes[0], rows)
    fig.suptitle(
        f"Figure 0 — AFT mixture grid · {_profile_title(profile)} · "
        f"{figure0.CLAUSE_LABEL[clause]} × {figure0.SURFACE_LABEL[surface]}",
        x=0.025, y=1.0 - TITLE_DROP / height, ha="left", fontsize=14,
        fontweight="bold", color=figure0.INK,
    )
    footnote = (
        f"{figure0.CLAUSE_LABEL[clause]}, {figure0.SURFACE_LABEL[surface]}; "
        f"agreement {figure0._n_text(agreement_ns)}; "
        f"conflict {figure0._n_text(conflict_ns)}. "
        f"0.25%, 0.5%, 1% and 5% cells are follow-ups #1a/#1c (8,192 rows, "
        f"balanced selection, eager eval — the campaign's own AFT geometry "
        f"and backend); agreement and 100%-Charter are the campaign's own "
        f"cells. {epoch_note(epochs)} Pale bars are cells that have not "
        f"landed yet, not zeros. {mix.NESTED_LOWDOSE_NOTE} "
        f"{house.twopct_note([profile])} {house.CAVEAT}."
    )
    compose_layout(fig, axes, height, footnote)
    stem = "__".join((
        profile.replace("_", "-"),
        figure0.SURFACE_STEM[surface],
        figure0.CLAUSE_STEM[clause],
    ))
    return data.save_figure(fig, stem, output)


def _profile_title(profile: str) -> str:
    model = house.MODEL_OF.get(profile)
    for (candidate, dose), name in house.PLAN.items():
        if name == profile:
            return (f"Gemma 3 {house.MODEL_LABEL[candidate]} · "
                    f"{house.DOSE_LABEL[dose]} presented")
    return f"{model or profile}"


# --------------------------------------------------------------- dose curves


def _dose_points(
    profile: str, arm: str, epoch: int, clause: str, surface: str,
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, list[tuple[float, float, tuple[float, float], bool]]]:
    """(x, rate, wilson_err, starred) per choice category, along the ladder."""
    series: dict[str, list[tuple[float, float, tuple[float, float], bool]]] = {
        "charter": [], "coin": [],
    }
    for index, mixture in enumerate(mix.MIXTURES):
        unit = unit_for(profile, arm, mixture.key, epoch,
                        collected=collected, campaign=campaign)
        if unit is None:
            continue
        reading = data.conflict_reading(unit, clause, surface)
        if reading is None:
            continue
        x = index if mixture.on_dose_axis else OFF_AXIS_X
        starred = is_narrow_here(profile, mixture.key)
        for category in series:
            rate = 100 * reading.shares.get(category, 0.0)
            low, high = house.wilson_err(rate / 100, reading.n_runs)
            series[category].append((x, rate, (100 * low, 100 * high), starred))
    return {key: sorted(value) for key, value in series.items()}


def _dose_tick_labels() -> list[str]:
    """The ladder's tick labels, staggered onto two lines.

    Eleven signed ticks plus the 100% reference do not fit side by side at any
    readable size once the 0.25% rung is on the axis: "-0.25%" is six
    characters against roughly four characters of tick spacing in these
    panels, and before the stagger the low-dose labels ran together into
    "-1%-0.5%-0.25%".

    Staggered rather than rotated.  Rotation would keep one line, but these
    panels sit in a shared-x grid where 45-degree labels run into the next
    row's panel title, and the leading sign is what a reader scans for -- it
    stays easiest to pick out horizontally.

    The parity is anchored on the ZERO tick rather than on index 0, so the
    stagger is symmetric about the middle of the axis: 0, +-0.5% and +-2% on
    the top line, +-0.25%, +-1% and +-5% below.  Anchoring on index 0 would
    put the two halves of the ladder on opposite lines and make a symmetric
    axis look lopsided.
    """
    labels = [*(mix.dose_tick_label(m) for m in mix.DOSE_AXIS), "+100%"]
    zero = next(index for index, m in enumerate(mix.DOSE_AXIS) if m.dose == 0)
    return [label if (index - zero) % 2 == 0 else f"\n{label}"
            for index, label in enumerate(labels)]


def _dose_axis(ax) -> None:
    """The ladder's x furniture, applied to every panel.

    Not-covered panels get it too: they sit in the shared-x rectangle, and a
    hatched panel that keeps default tick sizing overlaps its own labels.
    """
    ax.set_xlim(-0.6, OFF_AXIS_X + 0.6)
    ax.set_xticks([*range(len(mix.DOSE_AXIS)), OFF_AXIS_X])
    ax.set_xticklabels(_dose_tick_labels())
    # Sizing goes through tick_params, not set_xticklabels: on a shared x axis
    # the next panel's set_xticklabels rebuilds every sibling's label artists
    # and would drop a size set here.
    ax.tick_params(axis="x", labelsize=6.6)
    ax.set_ylim(0, 100)


def _segments(points: Sequence[tuple[float, float, Any, bool]]) -> list[list]:
    """Split the ladder into runs of ADJACENT measured ticks.

    Joining across a tick that has not landed would draw a dose response
    through a dose nobody measured, which is the same mistake the campaign's
    fig1 gap handling exists to avoid.
    """
    runs: list[list] = []
    for point in points:
        if runs and point[0] - runs[-1][-1][0] <= 1.0001:
            runs[-1].append(point)
        else:
            runs.append([point])
    return runs


def render_dose_response(
    *,
    surface: str,
    clause: str,
    epoch: int,
    output: Path,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Path]:
    doses = house.DOSES
    # Panel rectangle plus the reserved bottom band, in inches. The footnote
    # shrinks as cells land (fewer "still to land" clauses), so the band has
    # to be measured rather than guessed — a fixed fraction that cleared a
    # six-line footnote runs the legend through a four-line one.
    height = 3.5 * len(MODELS)
    fig, axes = plt.subplots(
        len(MODELS), len(doses), figsize=(4.0 * len(doses), height),
        squeeze=False, sharex=True, sharey=True,
    )
    ns: list[int] = []
    drawn = 0
    panel_profiles: list[str] = []
    for r, model in enumerate(MODELS):
        for c, dose in enumerate(doses):
            ax = axes[r][c]
            profile = house.PLAN.get((model, dose))
            if profile is not None:
                panel_profiles.append(profile)
            if c == 0:
                ax.set_ylabel("share of conflict-eval runs (%)", fontsize=9)
            if profile is None:
                _dose_axis(ax)
                house.draw_not_covered(ax, model, dose)
                continue
            ax.set_title(
                f"{house.MODEL_LABEL[model]}  {house.DOSE_LABEL[dose]} presented",
                fontsize=9.5, color=figure0.INK,
            )
            any_point = False
            for arm in figure0.ARMS:
                series = _dose_points(
                    profile, arm, epoch, clause, surface,
                    collected=collected, campaign=campaign)
                for category, points in series.items():
                    if not points:
                        continue
                    any_point = True
                    style, marker, _label = CHOICE_STYLE[category]
                    ladder = [p for p in points if p[0] <= len(mix.DOSE_AXIS)]
                    for run in _segments(ladder):
                        if len(run) < 2:
                            continue
                        ax.plot([p[0] for p in run], [p[1] for p in run],
                                style, color=house.ARM_COLOR[arm],
                                linewidth=1.5, zorder=3, alpha=0.9)
                    for x, rate, err, starred in points:
                        ax.errorbar(
                            x, rate, yerr=[[err[0]], [err[1]]], fmt=marker,
                            color=house.ARM_COLOR[arm], markersize=5.2,
                            markerfacecolor=("white" if starred
                                             else house.ARM_COLOR[arm]),
                            markeredgewidth=1.3, elinewidth=0.9, capsize=2.0,
                            zorder=4,
                        )
                        if starred:
                            ax.annotate(
                                mix.NARROW_STAR, (x, rate),
                                textcoords="offset points", xytext=(4.5, 3.5),
                                fontsize=8.5, color=house.DIAG, zorder=6,
                            )
            # Always show lift: each arm's own pre-AFT Charter choice, in the
            # same harness, is the anchor every AFT point on that panel moves
            # away from.
            for arm in figure0.ARMS:
                unit = pre_aft_unit(profile, arm, campaign)
                if unit is None:
                    continue
                reading = data.conflict_reading(unit, clause, surface)
                if reading is None:
                    continue
                ns.append(reading.n_runs)
                ax.axhline(100 * reading.shares.get("charter", 0.0),
                           color=house.ARM_COLOR[arm], linewidth=0.8,
                           linestyle=(0, (1, 2)), alpha=0.55, zorder=1)
            if not any_point:
                house.draw_training(ax)
            else:
                drawn += 1
            ax.axvline(len(mix.DOSE_AXIS) + 0.35, color="#c8c8c3",
                       linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
            ax.grid(color=figure0.GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            _dose_axis(ax)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(figure0.GRID)
            if r == len(MODELS) - 1:
                ax.set_xlabel(mix.DOSE_AXIS_LABEL, fontsize=8)

    handles = [
        Line2D([], [], color=house.ARM_COLOR[arm], linewidth=2.6,
               label=f"{figure0.ARM_LABEL[arm]} midtrain")
        for arm in figure0.ARMS
    ] + [
        Line2D([], [], color=figure0.MUTED, linestyle=style, marker=marker,
               label=label)
        for style, marker, label in CHOICE_STYLE.values()
    ] + [
        # Only when a narrow cell is actually on the canvas.  With the
        # corrected draw substituted in, nothing is hollow, and a legend key
        # for an absent marker invites the reader to hunt for one.
        Line2D([], [], color=house.DIAG, marker="o", linestyle="none",
               markerfacecolor="white",
               label="hollow + * = campaign narrow-conflict 2%"),
    ] * any(
        is_narrow_here(profile, mixture.key)
        for profile in panel_profiles for mixture in mix.MIXTURES
    ) + [
        Line2D([], [], color=figure0.MUTED, linestyle=(0, (1, 2)),
               label="dotted = that arm's pre-AFT Charter choice"),
        Patch(facecolor=house.UNCOVERED_FILL, edgecolor=house.UNCOVERED_INK,
              hatch="////", label="cell not covered (not planned)"),
    ]
    for row in axes:
        for ax in row:
            ax.tick_params(axis="x", labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel(mix.DOSE_AXIS_LABEL, fontsize=8)
    fig.suptitle(
        f"AFT mixture dose response · {mix.EPOCH_LABEL[epoch]} · "
        f"{figure0.CLAUSE_LABEL[clause]} × {figure0.SURFACE_LABEL[surface]}",
        x=0.01, y=1.0 - TITLE_DROP / height, ha="left", fontsize=15,
        fontweight="bold", color=figure0.INK,
    )
    footnote = (
        f"Conflict-episode choice against signed AFT conflict dose; "
        f"8,192 AFT rows, eager eval throughout. {epoch_note([epoch])} "
        f"0.25%/0.5%/1%/5% are "
        f"follow-ups #1a/#1c; agreement and 100%-Charter are the campaign. "
        f"{mix.NESTED_LOWDOSE_NOTE} "
        f"100%-Charter sits past the axis break because it is not the next "
        f"tick after 5%. Wilson 95% on conflict runs "
        f"(n={data.n_range(ns) if ns else 'n/a'} per point; runs are 3 per "
        f"episode and not independent, so these are optimistic). "
        f"\"training…\" is a planned cell that has not landed. "
        f"{house.twopct_note(panel_profiles)} {house.CAVEAT}."
    )
    wrapped = textwrap.fill(footnote, width=205)
    legend_height = 0.40
    footnote_y = LEGEND_GAP + legend_height
    footnote_top = footnote_y + FOOTNOTE_LINE * len(wrapped.splitlines())
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, FOOTNOTE_PAD / height))
    fig.text(0.99, footnote_y / height, wrapped, ha="right", va="bottom",
             color=figure0.MUTED, fontsize=7.5, linespacing=1.25)
    fig.subplots_adjust(
        left=0.055, right=0.995, top=1.0 - PANEL_TITLE_DROP / height,
        bottom=(footnote_top + 0.62) / height, hspace=0.28, wspace=0.08,
    )
    stem = "__".join((
        figure0.SURFACE_STEM[surface], figure0.CLAUSE_STEM[clause],
        f"{epoch}ep",
    ))
    return data.save_figure(fig, stem, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--figure", action="append", choices=FIGURES)
    parser.add_argument("--profile", action="append")
    parser.add_argument("--surface", action="append", choices=figure0.SURFACES)
    parser.add_argument("--clause", action="append", choices=figure0.CLAUSES)
    parser.add_argument(
        "--epoch", action="append", type=int, choices=(1, 2),
        help=("repeat to choose endpoint reads; default: "
              + ", ".join(str(epoch) for epoch in DEFAULT_EPOCHS)),
    )
    house.add_twopct_args(parser)
    args = parser.parse_args(argv)
    house.apply_twopct_args(args)

    collected = _load_collected(args.collected)
    campaign = _campaign_documents()
    note_twopct_state(campaign)
    figures = args.figure or list(FIGURES)
    surfaces = args.surface or list(figure0.SURFACES)
    clauses = args.clause or list(figure0.CLAUSES)
    epochs = args.epoch or list(DEFAULT_EPOCHS)
    profiles = args.profile or [
        house.PLAN[(model, dose)]
        for model in MODELS for dose in house.DOSES
        if (model, dose) in house.PLAN
    ]

    written: list[Path] = []
    if "composition" in figures:
        for profile in profiles:
            rows = profile_rows(profile, collected=collected,
                                campaign=campaign, epochs=epochs)
            for surface in surfaces:
                for clause in clauses:
                    written.extend(render_composition(
                        profile, rows, surface=surface, clause=clause,
                        epochs=epochs, output=args.out / "composition",
                    ))
    if "dose_response" in figures:
        for surface in surfaces:
            for clause in clauses:
                for epoch in epochs:
                    written.extend(render_dose_response(
                        surface=surface, clause=clause, epoch=epoch,
                        output=args.out / "dose_response",
                        collected=collected, campaign=campaign,
                    ))

    for path in written:
        print(f"wrote {path}")
    missing = collected.get("missing", [])
    meta = collected.get("meta", {})
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files).")
    print(f"{meta.get('endpoints')}/{meta.get('endpoints_planned')} follow-up "
          f"endpoints scored; {len(missing)} still to land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
