"""Follow-up #1b — GLM-4.5-Air @190M, 8,192 vs 81,920 AFT rows.

Three published post-Dolci parents (charter / coin / control) each run the
whole conflict-dose ladder — agreement, then 1%, 2% and 5% in each label
direction — at **81,920** AFT rows, ten times the campaign's geometry, with
both epoch boundaries evaluated (steps 2,560 and 5,120).  The campaign's own
8,192-row GLM row supplies the small-size arm of the comparison.

Two figure families, written to `figures/ablations/GLM-AFT-scaleup/`:

* `composition/` — the established Figure-0 composition, one figure per
  presentation surface x clause split.  Rows walk the dose ladder; inside a
  mixture, rows are arm x AFT size.
* `dose_response/` — Charter choice and coin choice against signed conflict
  dose, one panel per choice x arm, one line per AFT size.

Both default to the **converged 2-epoch** endpoint of each size, which is the
only read the two sizes share: 8,192 rows x 2 epochs is step 512 and 81,920
rows x 2 epochs is step 5,120.  The follow-up's 1-epoch read (step 2,560) is
a real endpoint and is one `--variant glm_81920_1ep` away, but it is 2,560
optimizer steps against the campaign arm's 512, so it belongs on a trajectory
figure rather than in the size comparison.

Three things are joined across studies here, and all three are on the figure:
the campaign's 2% cells are the narrow-conflict draw, the 81,920-row agreement
substrate is freshly generated rather than re-presented, and the 81,920-row
endpoints were sampled on a different eval backend.  `followup_mixtures.py`
records each with its provenance.

There is no 8,192-row 1-epoch arm by construction: GLM's intermediate AFT
checkpoints in the campaign were FSDP shards with no PEFT adapter beside them,
so `AFT_EVAL_STEPS` is step 512 alone for the family.  The 81,920-row run
exports gathered attention-LoRA adapters at every save, which is what makes
its 1-epoch endpoint evaluable at all.

Run from the repository root, after `collect_followup_scores.py`::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_glm_aft_scaleup.py
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
import plot_aft_grid as grid  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402
import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402

SCORED = HERE / "scored"
COLLECTED = SCORED / "ablations" / "glm_aft_scaleup.json"
OUTPUT = HERE / "figures" / "ablations" / "GLM-AFT-scaleup"

PROFILE = "glm45_air_190m"
FIGURES = ("composition", "dose_response")

ROW_PITCH = 0.92
SECTION_GAP = 0.72
OFF_AXIS_X = grid.OFF_AXIS_X

#: The +-5% cells were trained to step 1,920/5,120 (0.75 epoch; A3/coin to
#: 1,280) and then deliberately stopped -- "User no longer needs 5% results;
#: preserve for optional future resume", recorded in
#: followups/aft-size-mixture-paused-5pct-v1/<account>/<arm>/PAUSED_RUN.json.
#: FSDP and optimizer state are preserved, so a resume is possible, but no
#: endpoint exists and none is coming.  They are drawn as CANCELLED, not as
#: pending: a pale "data not available" bar would promise a cell that is not
#: on its way.
CANCELLED_CELLS: frozenset[str] = frozenset(("coin_5pct", "charter_5pct"))
CANCELLED_TEXT = "not run (paused at 0.75 epoch)"

#: With +-5% cancelled, the ladder's measured ticks are -2%, -1%, 0, +1%, +2%
#: and nothing further is coming, so joining across the empty 5% ends draws no
#: promise about an unmeasured dose -- it connects every point that will ever
#: exist.  The gemma grid keeps `adjacent`, where the gaps are still filling.
JOIN_MODES = ("all", "adjacent")

CHOICE_LABEL = {
    "charter": "chose Charter crew",
    "coin": "chose coin / cheapest crew",
}


@dataclass(frozen=True)
class Variant:
    """One (AFT size, epoch) arm of the size comparison."""

    key: str
    study: mix.Study
    epoch: int
    #: Full identity, used wherever epochs are mixed or spelled out.
    label: str
    #: Identity minus the epoch, for when every drawn variant shares one and
    #: repeating it on 48 rows is noise rather than information.
    size_label: str
    colour: str


#: Size is carried by colour and by an explicit row label, never by colour
#: alone.  Grey is the historical 8,192-row reference; the two blues are the
#: follow-up's epochs, separated by lightness rather than hue.
VARIANTS: tuple[Variant, ...] = (
    Variant("campaign_8192_2ep", mix.CAMPAIGN, 2,
            "8,192 rows · 2 ep", "8,192 rows", house.NEUTRAL),
    Variant("glm_81920_1ep", mix.GLM_ROWS_V2, 1,
            "81,920 rows · 1 ep", "81,920 rows", house.OKABE_ITO["sky"]),
    Variant("glm_81920_2ep", mix.GLM_ROWS_V2, 2,
            "81,920 rows · 2 ep", "81,920 rows", house.OKABE_ITO["blue"]),
)
#: The size comparison the gallery exists for: both sizes at their converged
#: 2-epoch endpoint, which is the only read they share.  The 1-epoch variant
#: stays available by flag.
DEFAULT_VARIANTS: tuple[str, ...] = (
    "campaign_8192_2ep", "glm_81920_2ep",
)


def display_label(variant: Variant, variants: Sequence[Variant]) -> str:
    """Drop the epoch from a series label when every series shares it."""
    if len({item.epoch for item in variants}) == 1:
        return variant.size_label
    return variant.label


@dataclass(frozen=True)
class PlotRow:
    section: str
    label: str
    arm: str
    unit: data.Unit | None
    starred: bool
    #: False when this (size, epoch) arm never runs this mixture: the campaign
    #: has no 1%/5% cells and the follow-up has no 100%-Charter cell.  Those
    #: rows are hatched "not in this study", never pale "not available".
    planned: bool
    #: True for the +-5% cells that were started and then stopped, which is a
    #: different fact from "this study never ran it".
    cancelled: bool
    y: float


def _load_collected(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing — run collect_followup_scores.py first")
    return json.loads(path.read_text())


def unit_for(
    arm: str,
    mixture: str,
    variant: Variant,
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> data.Unit | None:
    endpoint = variant.study.endpoint(mixture, variant.epoch)
    if endpoint is None:
        return None
    if variant.study is mix.GLM_ROWS_V2:
        document = collected.get("documents", {}).get(arm)
    else:
        document = campaign.get((PROFILE, arm))
    if not isinstance(document, dict) or endpoint not in data.endpoints_in(document):
        return None
    return data.Unit(PROFILE, arm, endpoint, document)


def section_label(mixture: mix.Mixture) -> str:
    """One heading per mixture, with both studies' conflict-row counts.

    The star follows the campaign row, not the heading: at 2% the 8,192-row
    cell is narrow-conflict and the 81,920-row cell is balanced, so starring
    the whole section would libel the follow-up's own data.
    """
    counts = " · ".join(
        f"{mixture.conflict_rows[rows]:,}/{rows:,}"
        for rows in (8_192, 81_920) if rows in mixture.conflict_rows
    )
    return f"{mixture.label}\n{counts} conflict rows"


def ladder_rows(
    *,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    variants: Sequence[Variant],
) -> list[PlotRow]:
    specs: list[tuple[str, str, str, data.Unit | None, bool, bool, bool]] = [
        ("pre-AFT\n(no AFT)", figure0.ARM_LABEL[arm], arm,
         grid.pre_aft_unit(PROFILE, arm, campaign), False, True, False)
        for arm in figure0.ARMS
    ]
    for mixture in mix.MIXTURES:
        section = section_label(mixture)
        for arm in figure0.ARMS:
            for variant in variants:
                starred = variant.study.is_narrow(mixture.key)
                planned = variant.study.endpoint(
                    mixture.key, variant.epoch) is not None
                if (variant.study is mix.GLM_ROWS_V2
                        and mixture.key in CANCELLED_CELLS):
                    planned = False
                label = (f"{figure0.ARM_LABEL[arm]} · "
                         f"{display_label(variant, variants)}"
                         f"{mix.NARROW_STAR if starred else ''}")
                cancelled = (variant.study is mix.GLM_ROWS_V2
                             and mixture.key in CANCELLED_CELLS)
                specs.append((
                    section, label, arm,
                    unit_for(arm, mixture.key, variant,
                             collected=collected, campaign=campaign),
                    starred, planned, cancelled,
                ))

    rows: list[PlotRow] = []
    y = 0.0
    previous: str | None = None
    for section, label, arm, unit, starred, planned, cancelled in specs:
        if previous is not None and section != previous:
            y += SECTION_GAP
        rows.append(PlotRow(section, label, arm, unit, starred, planned,
                            cancelled, y))
        y += ROW_PITCH
        previous = section
    return rows


def outstanding(collected: Mapping[str, Any]) -> list[str]:
    """Endpoints still coming — i.e. missing and NOT one of the stopped cells.

    Once this is empty the ladder is finished, and a "training…" legend entry
    would promise cells that were deliberately abandoned.
    """
    return [item for item in collected.get("missing", [])
            if not any(cell in item for cell in CANCELLED_CELLS)]


def _footnote(
    surface: str, clause: str, extra: str, variants: Sequence[Variant],
) -> str:
    drawn = "; ".join(variant.label for variant in variants)
    return (
        f"{figure0.CLAUSE_LABEL[clause]}, {figure0.SURFACE_LABEL[surface]}; "
        f"{extra} Drawn: {drawn}. Every cell trains 2 epochs. "
        f"GLM-4.5-Air, 190M presented midtrain tokens; every cell is a fresh "
        f"rank-64 attention-only LoRA on the same published step-96 Dolci "
        f"parent. No 8,192-row 1-epoch arm exists: the campaign's GLM "
        f"intermediate AFT checkpoints were FSDP shards with no adapter, so "
        f"the family evaluates step 512 alone. {mix.NARROW_NOTE} "
        f"{mix.AGREEMENT_SUBSTRATE_NOTE} {mix.BACKEND_NOTE} {house.CAVEAT}."
    )


def render_composition(
    rows: Sequence[PlotRow], *, surface: str, clause: str,
    variants: Sequence[Variant], output: Path,
    collected: Mapping[str, Any] | None = None,
) -> list[Path]:
    collected = collected or {}
    height = max(8.5, 2.7 + 0.29 * len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(17.2, height), sharey=True)
    agreement_ns: list[tuple[int, int | None]] = []
    conflict_ns: list[tuple[int, int | None]] = []
    for row in rows:
        if not row.planned:
            text = (CANCELLED_TEXT if row.cancelled else "not in this study")
            grid.draw_not_in_study(axes[0], row.y, text)
            grid.draw_not_in_study(axes[1], row.y, text)
            continue
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

    grid._decorate_panel(axes[0], rows, "Ambiguous (agreement episodes)",
                         "share of agreement-eval runs (%)")
    grid._decorate_panel(axes[1], rows, "Diagnostic (conflict episodes)",
                         "share of conflict-eval runs (%)")
    grid._add_section_labels(axes[0], rows)
    fig.suptitle(
        f"Figure 0 — GLM AFT size scale-up · GLM-4.5-Air · 190M presented · "
        f"{figure0.CLAUSE_LABEL[clause]} × {figure0.SURFACE_LABEL[surface]}",
        x=0.025, y=1.0 - grid.TITLE_DROP / height, ha="left", fontsize=14,
        fontweight="bold", color=figure0.INK,
    )
    footnote = _footnote(surface, clause, (
        f"agreement {figure0._n_text(agreement_ns)}; "
        f"conflict {figure0._n_text(conflict_ns)}. Pale bars are planned "
        f"cells that have not landed, not zeros; hatched rows are either a "
        f"(size, epoch) x mixture combination no study runs, or a ±5% cell "
        f"that was started and stopped."
        + (" Every endpoint that is not a stopped ±5% cell has landed."
           if not outstanding(collected) else "")
    ), variants)
    grid.compose_layout(fig, axes, height, footnote)
    stem = "__".join((
        figure0.SURFACE_STEM[surface], figure0.CLAUSE_STEM[clause],
    ))
    return data.save_figure(fig, stem, output)


def render_dose_response(
    *,
    surface: str,
    clause: str,
    output: Path,
    collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    variants: Sequence[Variant],
    join: str = "all",
) -> list[Path]:
    choices = ("charter", "coin")
    # Physical margins again: the footnote here carries three provenance
    # notes and is the tallest thing on the figure.
    height = 9.2
    fig, axes = plt.subplots(
        len(choices), len(figure0.ARMS), figsize=(15.6, height),
        squeeze=False, sharex=True, sharey=True,
    )
    ns: list[int] = []
    for r, choice in enumerate(choices):
        for c, arm in enumerate(figure0.ARMS):
            ax = axes[r][c]
            drew = False
            for variant in variants:
                points: list[tuple[float, float, tuple[float, float], bool]] = []
                for index, mixture in enumerate(mix.MIXTURES):
                    unit = unit_for(arm, mixture.key, variant,
                                    collected=collected, campaign=campaign)
                    if unit is None:
                        continue
                    reading = data.conflict_reading(unit, clause, surface)
                    if reading is None:
                        continue
                    ns.append(reading.n_runs)
                    rate = 100 * reading.shares.get(choice, 0.0)
                    low, high = house.wilson_err(rate / 100, reading.n_runs)
                    points.append((
                        index if mixture.on_dose_axis else OFF_AXIS_X,
                        rate, (100 * low, 100 * high),
                        variant.study.is_narrow(mixture.key),
                    ))
                points.sort()
                if not points:
                    continue
                drew = True
                ladder = [p for p in points if p[0] <= len(mix.DOSE_AXIS)]
                runs = ([ladder] if join == "all" else grid._segments(ladder))
                for run in runs:
                    if len(run) < 2:
                        continue
                    ax.plot([p[0] for p in run], [p[1] for p in run], "-",
                            color=variant.colour, linewidth=1.8, zorder=3)
                for x, rate, err, starred in points:
                    ax.errorbar(
                        x, rate, yerr=[[err[0]], [err[1]]], fmt="o",
                        color=variant.colour, markersize=5.6,
                        markerfacecolor="white" if starred else variant.colour,
                        markeredgewidth=1.4, elinewidth=0.9, capsize=2.0,
                        zorder=4,
                    )
                    if starred:
                        ax.annotate(mix.NARROW_STAR, (x, rate),
                                    textcoords="offset points", xytext=(5, 3.5),
                                    fontsize=9, color=house.DIAG, zorder=6)
            # Always show lift: this arm's own pre-AFT rate in the same harness.
            unit = grid.pre_aft_unit(PROFILE, arm, campaign)
            reading = (None if unit is None
                       else data.conflict_reading(unit, clause, surface))
            if reading is not None:
                ax.axhline(100 * reading.shares.get(choice, 0.0),
                           color=house.ARM_COLOR[arm], linewidth=1.0,
                           linestyle=(0, (1, 2)), alpha=0.7, zorder=1)
            if not drew:
                house.draw_training(ax)
            ax.axvline(len(mix.DOSE_AXIS) + 0.35, color="#c8c8c3",
                       linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
            ax.grid(color=figure0.GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            grid._dose_axis(ax)
            ax.set_title(
                f"{CHOICE_LABEL[choice]} · {figure0.ARM_LABEL[arm]} midtrain",
                fontsize=10, color=house.ARM_COLOR[arm],
            )
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(figure0.GRID)
            if c == 0:
                ax.set_ylabel("share of conflict-eval runs (%)", fontsize=9)
    for row in axes:
        for ax in row:
            ax.tick_params(axis="x", labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel(mix.DOSE_AXIS_LABEL, fontsize=8)

    handles = [
        Line2D([], [], color=variant.colour, marker="o", linewidth=1.8,
               label=display_label(variant, variants))
        for variant in variants
    ] + [
        Line2D([], [], color=house.DIAG, marker="o", linestyle="none",
               markerfacecolor="white",
               label="hollow + * = campaign narrow-conflict 2%"),
        Line2D([], [], color=figure0.MUTED, linestyle=(0, (1, 2)),
               label="dotted = that arm's pre-AFT rate"),
    ]
    if outstanding(collected):
        handles.append(Patch(facecolor="#f2f2f0", edgecolor="#c8c8c3",
                             label="\"training…\" = planned, not yet landed"))
    else:
        handles.append(Line2D([], [], linestyle="none",
                              label="±5% not run (paused at 0.75 epoch)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.06 / height))
    fig.suptitle(
        f"GLM AFT size scale-up · dose response · "
        f"{figure0.CLAUSE_LABEL[clause]} × {figure0.SURFACE_LABEL[surface]}",
        x=0.01, y=1.0 - 0.28 / height, ha="left", fontsize=15,
        fontweight="bold", color=figure0.INK,
    )
    footnote = _footnote(surface, clause, (
        f"Wilson 95% on conflict runs "
        f"(n={data.n_range(ns) if ns else 'n/a'} per point; runs are 3 per "
        f"episode and not independent, so these are optimistic). 100%-Charter "
        f"sits past the axis break because it is not the next tick after 5%, "
        f"and has no 81,920-row counterpart by design. "
        + ("The ladder is otherwise COMPLETE: every remaining endpoint is a "
           "stopped ±5% cell. " if not outstanding(collected) else "")
        + "The ±5% cells were "
        "started and stopped at 0.75 epoch and are not coming, so "
        + ("every measured tick is joined — the line spans the whole ladder "
           "that will ever exist." if join == "all"
           else "segments join only adjacent measured ticks.")
    ), variants)
    fig.text(0.99, 0.62 / height, textwrap.fill(footnote, width=215),
             ha="right", va="bottom", color=figure0.MUTED, fontsize=7.5,
             linespacing=1.25)
    fig.subplots_adjust(left=0.055, right=0.99, top=1.0 - 0.80 / height,
                        bottom=2.05 / height, hspace=0.30, wspace=0.08)
    stem = "__".join((
        figure0.SURFACE_STEM[surface], figure0.CLAUSE_STEM[clause],
    ))
    return data.save_figure(fig, stem, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--figure", action="append", choices=FIGURES)
    parser.add_argument("--surface", action="append", choices=figure0.SURFACES)
    parser.add_argument("--clause", action="append", choices=figure0.CLAUSES)
    parser.add_argument(
        "--join", choices=JOIN_MODES, default="all",
        help=("dose_response only; 'all' joins every measured tick (the ±5%% "
              "cells are cancelled, so nothing will fill the gap), "
              "'adjacent' breaks the line at an empty tick"),
    )
    parser.add_argument(
        "--variant", action="append",
        choices=[variant.key for variant in VARIANTS],
        help=("repeat to choose the size/epoch arms; default: "
              + ", ".join(DEFAULT_VARIANTS)),
    )
    args = parser.parse_args(argv)

    collected = _load_collected(args.collected)
    campaign = data.load_documents(SCORED)
    figures = args.figure or list(FIGURES)
    surfaces = args.surface or list(figure0.SURFACES)
    clauses = args.clause or list(figure0.CLAUSES)
    selected = set(args.variant or DEFAULT_VARIANTS)
    variants = [variant for variant in VARIANTS if variant.key in selected]

    written: list[Path] = []
    for surface in surfaces:
        for clause in clauses:
            if "composition" in figures:
                rows = ladder_rows(collected=collected, campaign=campaign,
                                   variants=variants)
                written.extend(render_composition(
                    rows, surface=surface, clause=clause, variants=variants,
                    output=args.out / "composition", collected=collected))
            if "dose_response" in figures:
                written.extend(render_dose_response(
                    surface=surface, clause=clause,
                    output=args.out / "dose_response",
                    collected=collected, campaign=campaign,
                    variants=variants, join=args.join))

    for path in written:
        print(f"wrote {path}")
    meta = collected.get("meta", {})
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files).")
    print(f"{meta.get('endpoints')}/{meta.get('endpoints_planned')} follow-up "
          f"endpoints scored; {len(collected.get('missing', []))} still to land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
