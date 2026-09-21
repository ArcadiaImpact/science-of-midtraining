"""Follow-up #1c's two-sided cell — GLM-4.5-Air @190M, an 80:10:10 AFT mix.

Every other cell on the AFT mixture axis is ONE-SIDED: its conflict rows are
labelled Charter or coin, never both, which is what makes the signed dose
ladder an axis.  The GLM #1c release adds the one cell that is not shaped like
that — 6,554 agreement rows, 819 coin-labelled conflict rows and 819
Charter-labelled conflict rows in the same 8,192-row AFT set, each subset
stratified independently across all ten clause x run-count strata and the two
conflict pools disjoint by episode.  So the question it asks is not "how much
contamination" but **what a model does when the contamination argues with
itself**, and there is no dose tick that states that (see the two-sided
section of `followup_mixtures.py` for why it is deliberately not a rung).

Two figure families, written to `figures/ablations/GLM-threeway/`:

* `composition/` — the established Figure-0 composition, one figure per
  presentation surface x clause split.  Sections walk pre-AFT, 100% agreement,
  the two one-sided 2% cells and then the 80:10:10 mix, with the three
  midtrain arms inside each, so it is visible *where* the mass went rather
  than only that a rate moved.
* `headline/` — the same numbers as rates: Charter choice and coin choice with
  Wilson intervals, one row per cell x arm, each row carrying its own arm's
  pre-AFT rate as a dotted caret so every point is read as lift.

For the mechanism — is a middling pooled rate one behaviour or an average over
clauses, and does an episode's own runs agree? — use the breakdowns, which
matter more here than anywhere else in the directory: a two-sided mix is the
only place `mixed` (Charter on one run of an episode, coin on another) has a
training story behind it.

    plot_followup_breakdown.py --gallery glm_threeway

**What the release actually shows** (trained clauses, canonical, 2 epochs):
the 80:10:10 mix lands between the two one-sided cells on every arm, and the
midtrain prior is what breaks the tie — 53.7% / 56.2% Charter choice on the
charter and control arms against 39.0% on the coin arm.  But it does NOT buy
within-episode indecision: on two-conflict-run episodes `mixed` is 4.7% /
7.3% / 1.9%.  The model splits BETWEEN episodes, not within them.

Two joins are on the figure and both are marked.  The 80:10:10 and 2% rows are
one release — same parents, rows, epochs, batch, seed, templates and eval
backend — so that contrast is within-harness.  The agreement and pre-AFT rows
come from the campaign, which sampled eager rather than graphs/split-K-1, so
those two rows are a cross-harness join and carry a dagger.  And the 2% cells
are 164 conflict rows per side against the mix's 819: they bracket the two
label directions, they are not a dose-matched control for either half of it.

Run from the repository root, after `collect_followup_scores.py`::

    uv run --extra dev python3 \
      experiments/dispatch/dispatch_final_v1/results_grid/plot_glm_threeway.py
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import followup_mixtures as mix  # noqa: E402
import plot_aft_grid as grid  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402
import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402

SCORED = HERE / "scored"
COLLECTED = SCORED / "ablations" / "glm_threeway.json"
#: The one-sided 2% cells from the SAME #1c release, packaged separately so a
#: two-sided cell can never reach `twopct.py`'s substitution.  See
#: `collect_glm_threeway`.
SIBLING = SCORED / "ablations" / "glm_contamination.json"
OUTPUT = HERE / "figures" / "ablations" / "GLM-threeway"

PROFILE = mix.GLM_REPAIR_PROFILE
FIGURES = ("composition", "headline")

#: The converged endpoint alone, as everywhere else in these galleries.  Both
#: epochs are real here (#1c exports a PEFT adapter at every save), so
#: `--epoch 1` is a genuine read and not a promise of an absent one.
DEFAULT_EPOCHS: tuple[int, ...] = (2,)

ROW_PITCH = 0.92
SECTION_GAP = 0.72
#: Inches of row-label column on the headline figure: a section bracket whose
#: longest line is "164 / 8,192 conflict rows, one-sided" plus an arm label.
GUTTER = 5.0

#: Above this malformed share, a row's choice rates are a parse failure rather
#: than a preference, and the headline figure labels it as such.  Chosen as a
#: bare majority: once most runs did not produce a verdict, neither choice
#: rate is a rate of anything.
DEGENERATE_PCT = 50.0

#: Rows sampled on the campaign's eager backend rather than this release's
#: graphs/split-K-1.  Marked, never corrected: neither backend is ground truth
#: and the measured offset is smaller than the run-to-run SD.
CROSS_MARK = "†"
CROSS_NOTE = (
    "† Sampled on the campaign's eager eval backend, not this release's "
    "graphs/split-K-1; measured pooled offset -0.80pp charter / +1.00pp coin "
    "on conflict runs. Neither backend is ground truth."
)

CHOICE_LABEL = {
    "charter": "chose Charter crew",
    "coin": "chose coin / cheapest crew",
}

#: The rows, coin-heavy to Charter-heavy and then the two-sided mix last, so
#: reading top-to-bottom walks the one-sided cells before the cell that mixes
#: them.  `study` is None for the pre-AFT anchor, which is an endpoint of the
#: campaign document rather than a mixture of any study.
PRE_AFT = "pre_aft"


@dataclass(frozen=True)
class Cell:
    """One AFT mixture on this figure, and where its scores come from."""

    key: str
    #: Which collected document holds it: "threeway", "sibling", "campaign".
    source: str
    study: mix.Study | None
    #: Section heading; the row count line is appended by `section_label`.
    label: str
    detail: str
    #: True when this cell's scores come from the campaign's eager backend.
    cross_harness: bool


CELLS: tuple[Cell, ...] = (
    Cell(PRE_AFT, "campaign", None, "pre-AFT\n(no AFT)",
         "the midtrain parent itself", True),
    Cell("agreement", "campaign", mix.CAMPAIGN, "100% agreement",
         "0 / 8,192 conflict rows", True),
    Cell("coin_2pct", "sibling", mix.GLM_REPAIR, "2% coin-labelled",
         "164 / 8,192 conflict rows, one-sided", False),
    Cell("charter_2pct", "sibling", mix.GLM_REPAIR, "2% Charter-labelled",
         "164 / 8,192 conflict rows, one-sided", False),
    Cell(mix.THREEWAY.key, "threeway", mix.GLM_THREEWAY, mix.THREEWAY.label,
         f"{mix.THREEWAY.coin_rows:,} coin + "
         f"{mix.THREEWAY.charter_rows:,} Charter / {mix.THREEWAY.rows:,}",
         False),
)
BY_KEY: dict[str, Cell] = {cell.key: cell for cell in CELLS}
#: The cell the gallery exists for, named so a caller can ask for it alone.
SUBJECT = mix.THREEWAY.key


@dataclass(frozen=True)
class PlotRow:
    section: str
    label: str
    arm: str
    unit: data.Unit | None
    starred: bool
    #: False when this (cell, epoch) combination is not evaluable at all --
    #: the campaign's GLM row has no step-256 endpoint, so its agreement and
    #: pre-AFT rows are hatched at epoch 1 rather than drawn as pale pending
    #: bars for a checkpoint that was never exported as an adapter.
    planned: bool
    cross_harness: bool
    y: float


def _load(path: Path, gallery: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing — run collect_followup_scores.py "
            f"--only {gallery} first")
    return json.loads(path.read_text())


def section_label(cell: Cell) -> str:
    return f"{cell.label}\n{cell.detail}"


def unit_for(
    arm: str,
    cell: Cell,
    epoch: int,
    *,
    collected: Mapping[str, Any],
    sibling: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> data.Unit | None:
    """The scored endpoint for one (arm, cell, epoch), or None if absent."""
    if cell.key == PRE_AFT:
        return grid.pre_aft_unit(PROFILE, arm, campaign)
    assert cell.study is not None
    endpoint = cell.study.endpoint(cell.key, epoch)
    if endpoint is None:
        return None
    if cell.source == "campaign":
        document = campaign.get((PROFILE, arm))
    else:
        source = collected if cell.source == "threeway" else sibling
        document = source.get("documents", {}).get(arm)
    if not isinstance(document, dict) or endpoint not in data.endpoints_in(document):
        return None
    return data.Unit(PROFILE, arm, endpoint, document)


def is_planned(cell: Cell, epoch: int) -> bool:
    """Whether this (cell, epoch) is an endpoint that exists to be scored.

    The campaign's GLM row evaluates step 512 alone -- its intermediate AFT
    checkpoints were FSDP shards with no PEFT adapter -- so `agreement` has no
    1-epoch read and never will.  #1c's own three cells export an adapter at
    every save and have both.
    """
    if cell.key == PRE_AFT:
        # One measurement of the parent, which has no epoch of its own: the
        # anchor is available at whichever AFT epoch is being read.
        return True
    if cell.source == "campaign" and epoch != 2:
        return False
    return cell.study is not None and cell.study.endpoint(cell.key, epoch) is not None


def ladder_rows(
    *,
    collected: Mapping[str, Any],
    sibling: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
    epoch: int,
    cells: Sequence[str] | None = None,
) -> list[PlotRow]:
    """One section per AFT cell, the three midtrain arms inside each."""
    wanted = [BY_KEY[key] for key in (cells or [cell.key for cell in CELLS])]
    rows: list[PlotRow] = []
    y = 0.0
    previous: str | None = None
    for cell in wanted:
        section = section_label(cell)
        for arm in figure0.ARMS:
            if previous is not None and section != previous:
                y += SECTION_GAP
            previous = section
            label = figure0.ARM_LABEL[arm]
            if cell.cross_harness:
                label = f"{label}{CROSS_MARK}"
            rows.append(PlotRow(
                section, label, arm,
                unit_for(arm, cell, epoch, collected=collected,
                         sibling=sibling, campaign=campaign),
                False, is_planned(cell, epoch), cell.cross_harness, y,
            ))
            y += ROW_PITCH
    return rows


def _footnote(surface: str, clause: str, epoch: int, extra: str) -> str:
    return (
        f"{figure0.CLAUSE_LABEL[clause]}, {figure0.SURFACE_LABEL[surface]}; "
        f"{extra} GLM-4.5-Air, 190M presented midtrain tokens; every AFT cell "
        f"is a fresh rank-64 attention-only LoRA on the same published step-96 "
        f"Dolci parent, 8,192 rows x 2 epochs, batch 32, seed 42, campaign "
        f"training templates, and the {mix.EPOCH_LABEL[epoch]} endpoint "
        f"(step {mix.GLM_THREEWAY.steps[epoch]}) is drawn. "
        f"{mix.THREEWAY_DOSE_NOTE} {mix.THREEWAY_BACKEND_NOTE} {CROSS_NOTE} "
        f"{house.CAVEAT}."
    )


def render_composition(
    rows: Sequence[PlotRow], *, surface: str, clause: str, epoch: int,
    output: Path,
) -> list[Path]:
    if not rows:
        return []
    height = max(8.0, 2.7 + 0.29 * len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(17.2, height), sharey=True)
    agreement_ns: list[tuple[int, int | None]] = []
    conflict_ns: list[tuple[int, int | None]] = []
    for row in rows:
        if not row.planned:
            grid.draw_not_in_study(axes[0], row.y, "no 1-epoch endpoint")
            grid.draw_not_in_study(axes[1], row.y, "no 1-epoch endpoint")
            continue
        agreement = (None if row.unit is None
                     else figure0._positive_rates(row.unit, clause, surface))
        if agreement is None:
            figure0._draw_missing(axes[0], row.y)
        else:
            shares, n_runs, n_episodes = agreement
            agreement_ns.append((n_runs, n_episodes))
            figure0._draw_stack(axes[0], row.y, shares,
                                figure0.AGREEMENT_ORDER,
                                figure0.AGREEMENT_STYLE)
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
        f"Figure 0 — GLM two-sided AFT mix · 80:10:10 · GLM-4.5-Air · "
        f"190M presented · {figure0.CLAUSE_LABEL[clause]} × "
        f"{figure0.SURFACE_LABEL[surface]}",
        x=0.025, y=1.0 - grid.TITLE_DROP / height, ha="left", fontsize=14,
        fontweight="bold", color=figure0.INK,
    )
    footnote = _footnote(surface, clause, epoch, (
        f"agreement {figure0._n_text(agreement_ns)}; "
        f"conflict {figure0._n_text(conflict_ns)}. Pale bars are planned "
        f"cells that have not landed, not zeros; hatched rows have no "
        f"endpoint at this epoch."
    ))
    grid.compose_layout(fig, axes, height, footnote)
    stem = "__".join((
        figure0.SURFACE_STEM[surface], figure0.CLAUSE_STEM[clause],
        f"{epoch}ep",
    ))
    return data.save_figure(fig, stem, output)


def render_headline(
    rows: Sequence[PlotRow], *, surface: str, clause: str, epoch: int,
    campaign: Mapping[tuple[str, str], Mapping[str, Any]], output: Path,
) -> list[Path]:
    """Charter and coin choice as rates, with each row's own pre-AFT caret."""
    if not rows:
        return []
    choices = ("charter", "coin")
    height = max(6.0, 2.9 + 0.30 * len(rows))
    # Physical, not fractional: the row-label column holds a section bracket
    # ("80:10:10 · 10% coin + 10% Charter" over "819 coin + 819 Charter /
    # 8,192") as well as the arm label, and a fraction that clears it at one
    # figure width clips it at the next.
    width = GUTTER + 5.2 * len(choices)
    fig, axes = plt.subplots(1, len(choices), figsize=(width, height),
                             sharey=True)
    ns: list[int] = []
    degenerate: set[tuple[str, str]] = set()
    for index, choice in enumerate(choices):
        ax = axes[index]
        for row in rows:
            if not row.planned:
                grid.draw_not_in_study(ax, row.y, "no 1-epoch endpoint")
                continue
            # Always show lift: this arm's own pre-AFT rate, same harness.
            # Not on the pre-AFT rows themselves, where the point IS the
            # anchor and the caret would just underline it.
            anchor = (None if row.section.startswith(BY_KEY[PRE_AFT].label)
                      else grid.pre_aft_unit(PROFILE, row.arm, campaign))
            reading = (None if anchor is None
                       else data.conflict_reading(anchor, clause, surface))
            if reading is not None:
                rate = 100 * reading.shares.get(choice, 0.0)
                ax.plot([rate, rate], [row.y - 0.30, row.y + 0.30],
                        color=house.ARM_COLOR[row.arm], linewidth=1.1,
                        linestyle=(0, (1, 1.6)), alpha=0.85, zorder=2)
            conflict = (None if row.unit is None
                        else data.conflict_reading(row.unit, clause, surface))
            if conflict is None:
                figure0._draw_missing(ax, row.y)
                continue
            ns.append(conflict.n_runs)
            # A cell that mostly failed to answer reads on a rates panel as a
            # confident zero.  It is not: the coin arm's 2% coin-labelled cell
            # is 95.5% malformed on this slice, and both its choice rates are
            # near the floor because almost nothing was parseable.  Say so on
            # the row -- the composition figure shows the mass, this one has
            # to name it.
            malformed = 100 * conflict.shares.get("malformed", 0.0)
            if malformed > DEGENERATE_PCT:
                degenerate.add((row.section, row.arm))
                ax.annotate(f"{malformed:.0f}% malformed", (0, row.y),
                            textcoords="offset points", xytext=(6, -9),
                            fontsize=6.4, style="italic", color=house.DIAG,
                            zorder=6)
            rate = 100 * conflict.shares.get(choice, 0.0)
            low, high = house.wilson_err(rate / 100, conflict.n_runs)
            ax.errorbar(
                rate, row.y, xerr=[[100 * low], [100 * high]], fmt="o",
                color=house.ARM_COLOR[row.arm], markersize=6.4,
                # Hollow marks the cross-harness rows, the convention the
                # other galleries use for a point drawn from another harness.
                markerfacecolor="white" if row.cross_harness
                else house.ARM_COLOR[row.arm],
                markeredgewidth=1.5, elinewidth=0.9, capsize=2.2, zorder=4,
            )
        grid._decorate_panel(ax, rows, CHOICE_LABEL[choice],
                             "share of conflict-eval runs (%)")
        if index:
            ax.tick_params(labelleft=False)
    grid._add_section_labels(axes[0], rows)

    fig.suptitle(
        f"GLM two-sided AFT mix · 80:10:10 vs the one-sided cells · "
        f"{figure0.CLAUSE_LABEL[clause]} × {figure0.SURFACE_LABEL[surface]} · "
        f"{mix.EPOCH_LABEL[epoch]}",
        x=0.012, y=1.0 - 0.30 / height, ha="left", fontsize=13,
        fontweight="bold", color=figure0.INK,
    )
    handles = [
        Line2D([], [], color=house.ARM_COLOR[arm], marker="o",
               linestyle="none", label=f"{figure0.ARM_LABEL[arm]} midtrain")
        for arm in figure0.ARMS
    ] + [
        Line2D([], [], color=house.NEUTRAL, marker="o", linestyle="none",
               markerfacecolor="white", markeredgewidth=1.5,
               label=f"hollow + {CROSS_MARK} = campaign eager backend"),
        Line2D([], [], color=figure0.MUTED, linestyle=(0, (1, 1.6)),
               label="dotted caret = that arm's pre-AFT rate"),
    ]
    # Keyed by (section, arm), so a row counts once however many choice
    # panels drew it -- both panels read the same cell.
    degenerate_note = (
        f" {len(degenerate)} row(s) are over "
        f"{DEGENERATE_PCT:g}% malformed and are labelled: both choice rates "
        f"sit near the floor because almost nothing was parseable, which is "
        f"an unparseable cell and not a measured preference."
        if degenerate else "")
    footnote = _footnote(surface, clause, epoch, (
        f"Wilson 95% on conflict runs "
        f"(n={data.n_range(ns) if ns else 'n/a'} per point; runs are 3 per "
        f"episode and not independent, so these are optimistic)."
        f"{degenerate_note}"
    ))
    wrapped = textwrap.fill(footnote, width=int(width * 13))
    legend_y = 0.16 + 0.135 * len(wrapped.splitlines()) + 0.16
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.012, legend_y / height))
    fig.text(0.99, 0.16 / height, wrapped, ha="right", va="bottom",
             color=figure0.MUTED, fontsize=7.4, linespacing=1.25)
    fig.subplots_adjust(left=GUTTER / width, right=0.99,
                        top=1.0 - 0.92 / height,
                        bottom=(legend_y + 0.30 + 0.62) / height, wspace=0.06)
    stem = "__".join((
        figure0.SURFACE_STEM[surface], figure0.CLAUSE_STEM[clause],
        f"{epoch}ep",
    ))
    return data.save_figure(fig, stem, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument("--sibling", type=Path, default=SIBLING)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--figure", action="append", choices=FIGURES)
    parser.add_argument("--surface", action="append", choices=figure0.SURFACES)
    parser.add_argument("--clause", action="append", choices=figure0.CLAUSES)
    parser.add_argument("--epoch", action="append", type=int, choices=(1, 2),
                        help=f"default: {', '.join(map(str, DEFAULT_EPOCHS))}")
    parser.add_argument(
        "--cell", action="append", choices=[cell.key for cell in CELLS],
        help=("restrict the sections; default: all of them. The filter goes "
              "into the filename so a focused figure never overwrites the "
              "full one."),
    )
    args = parser.parse_args(argv)

    collected = _load(args.collected, "glm_threeway")
    sibling = _load(args.sibling, "glm_contamination")
    campaign = data.load_documents(SCORED)
    figures = args.figure or list(FIGURES)
    surfaces = args.surface or list(figure0.SURFACES)
    clauses = args.clause or list(figure0.CLAUSES)
    epochs = args.epoch or list(DEFAULT_EPOCHS)
    suffix = tuple(key.replace("_", "-") for key in (args.cell or ()))

    written: list[Path] = []
    for epoch in epochs:
        for surface in surfaces:
            for clause in clauses:
                rows = ladder_rows(
                    collected=collected, sibling=sibling, campaign=campaign,
                    epoch=epoch, cells=args.cell)
                for name, render in (("composition", render_composition),
                                     ("headline", render_headline)):
                    if name not in figures:
                        continue
                    extra = ({"campaign": campaign}
                             if name == "headline" else {})
                    paths = render(
                        rows, surface=surface, clause=clause, epoch=epoch,
                        output=args.out / name, **extra)
                    written.extend(_retag(paths, suffix))

    for path in written:
        print(f"wrote {path}")
    meta = collected.get("meta", {})
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files).")
    print(f"{meta.get('endpoints')}/{meta.get('endpoints_planned')} 80:10:10 "
          f"endpoints scored; {len(collected.get('missing', []))} still to "
          f"land.")
    return 0


def _retag(paths: Sequence[Path], suffix: Sequence[str]) -> list[Path]:
    """Fold a `--cell` filter into the filename, so a subset never overwrites.

    Renaming after the fact rather than threading a stem through both
    renderers: the two of them already build the same stem from surface,
    clause and epoch, and a focused figure differs from the full one only in
    which sections it drew.
    """
    if not suffix:
        return list(paths)
    out: list[Path] = []
    for path in paths:
        target = path.with_name(
            f"{path.stem}__{'-'.join(suffix)}{path.suffix}")
        path.replace(target)
        out.append(target)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
