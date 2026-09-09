"""Was the contaminating data representative?  Legacy 2% vs balanced 2%.

Follow-up #1c re-runs every campaign 2% AFT cell on a corrected conflict draw.
The two arms of the contrast differ in **one** thing:

* **legacy** — all 164 conflict rows are single-run `precedence_days_since`
  episodes.  One of the five trained clauses, and never a two-run episode.
* **balanced** — the same 164 rows, drawn 16-17 per stratum across all five
  clauses and split 82/82 one-run/two-run.

Everything else is held: the same published parent, 8,192 rows x 2 epochs,
global batch 32, seed 42, the same eager eval battery, and the same label-flip
pairing between the coin- and Charter-labelled cells.  So this is not a
"corrected number replaces a wrong one" gallery — it is a measurement of how
much the *representativeness* of a fixed dose of contaminating data matters,
which is a result in its own right.

Two figure families, written to `figures/ablations/contamination-data-quality/`:

* `delta/` — the headline.  One paired row per midtrain cell: legacy value,
  balanced value, and the arrow between them, with Wilson intervals.  One
  panel per label direction.
* `composition/` — the established Figure-0 composition with legacy and
  balanced as adjacent rows, so it is visible *where* the mass moved (Charter,
  coin, another crew, malformed) rather than only that a rate changed.

For the mechanism — does the legacy cell underperform specifically on the four
clauses it never saw? — use the clause breakdown:

    plot_followup_breakdown.py --gallery contamination_quality

Run from the repository root, after `collect_followup_scores.py`::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_contamination_quality.py
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
COLLECTED = SCORED / "ablations" / "contamination_quality.json"
OUTPUT = HERE / "figures" / "ablations" / "contamination-data-quality"

FIGURES = ("delta", "composition")
DIRECTIONS = ("coin_2pct", "charter_2pct")

#: The two arms of the contrast.  Legacy is the hollow/greyed one throughout,
#: matching the hollow-marker convention the other galleries already use for
#: the narrow-conflict cells.
LEGACY = "legacy"
BALANCED = "balanced"
VARIANT_LABEL = {
    LEGACY: "legacy · one clause, one run",
    BALANCED: "balanced · 5 clauses, 82/82 runs",
}
VARIANT_COLOUR = {LEGACY: house.NEUTRAL, BALANCED: house.OKABE_ITO["green"]}

ROW_PITCH = 0.92
SECTION_GAP = 0.72


@dataclass(frozen=True)
class Pair:
    """One midtrain cell, both draws of the same 2% dose."""

    profile: str
    arm: str
    mixture: str
    legacy: data.Unit | None
    balanced: data.Unit | None
    y: float

    @property
    def label(self) -> str:
        return f"{grid._profile_title(self.profile)} · {figure0.ARM_LABEL[self.arm]}"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing — run collect_followup_scores.py "
            f"--only contamination_quality first")
    return json.loads(path.read_text())


def unit_for(
    variant: str, profile: str, arm: str, mixture: str, epoch: int,
    *, collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> data.Unit | None:
    study = mix.CAMPAIGN if variant == LEGACY else mix.GRID_REPAIR
    endpoint = study.endpoint(mixture, epoch)
    if endpoint is None:
        return None
    if variant == LEGACY:
        document = campaign.get((profile, arm))
    else:
        document = collected.get("documents", {}).get(f"{profile}|{arm}")
    if not isinstance(document, dict) or endpoint not in data.endpoints_in(document):
        return None
    return data.Unit(profile, arm, endpoint, document)


def profile_title_key(profile: str) -> tuple[Any, ...]:
    """Order rows the way every other gallery orders profiles."""
    for (model, dose), name in house.PLAN.items():
        if name == profile:
            return (house.MODELS.index(model), dose)
    # Ablation profiles (the no-example row) sort after the campaign plan.
    return (len(house.MODELS), 0, profile)


def pairs_for(
    mixture: str, epoch: int,
    *, collected: Mapping[str, Any],
    campaign: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Pair]:
    """Every (profile, arm) either side of the contrast has, paired."""
    keys: set[tuple[str, str]] = set()
    for key in collected.get("documents", {}):
        profile, arm = key.split("|")
        keys.add((profile, arm))
    keys |= {key for key in campaign}
    rows: list[Pair] = []
    y = 0.0
    previous: str | None = None
    for profile, arm in sorted(keys, key=lambda k: (profile_title_key(k[0]),
                                                    figure0.ARMS.index(k[1])
                                                    if k[1] in figure0.ARMS else 9)):
        legacy = unit_for(LEGACY, profile, arm, mixture, epoch,
                          collected=collected, campaign=campaign)
        balanced = unit_for(BALANCED, profile, arm, mixture, epoch,
                            collected=collected, campaign=campaign)
        # A row needs the BALANCED side to exist: the campaign has 2% cells
        # everywhere, so without #1c there is no contrast to draw and the row
        # would just restate the legacy gallery.
        if balanced is None:
            continue
        if previous is not None and profile != previous:
            y += SECTION_GAP
        rows.append(Pair(profile, arm, mixture, legacy, balanced, y))
        y += ROW_PITCH
        previous = profile
    return rows


def _rate(unit: data.Unit | None, clause: str, surface: str,
          category: str) -> tuple[float, int] | None:
    if unit is None:
        return None
    reading = data.conflict_reading(unit, clause, surface)
    if reading is None:
        return None
    return 100 * reading.shares.get(category, 0.0), reading.n_runs


def backend_note(pairs: Sequence[Pair]) -> str:
    """The one held input that is NOT held on the GLM-4.5-Air rows, for the
    footnote: their balanced side (#1c, 2026-09-08) was sampled with
    follow-up #1b's vLLM policy, their legacy side with the campaign's eager
    battery.  Empty when no GLM row is drawn, so the gemma-only figures read
    exactly as before."""
    if not any(house.MODEL_OF.get(pair.profile) == "glm45_air" for pair in pairs):
        return ""
    return (
        " GLM-4.5-Air rows: the balanced side was sampled with #1b's "
        "graphs/split-K-1 vLLM backend and the legacy side with eager "
        "(#1b's measured pooled offset between the two: -0.80pp charter / "
        "+1.00pp coin on conflict runs; neither is ground truth)."
    )


def render_delta(
    mixture: str, pairs: Sequence[Pair], *, epoch: int, surface: str,
    clause: str, category: str, output: Path,
) -> list[Path]:
    if not pairs:
        return []
    height = max(5.0, 1.9 + 0.34 * len(pairs))
    fig, ax = plt.subplots(figsize=(12.4, height))
    ns: list[int] = []
    deltas: list[float] = []
    unpaired = 0
    for pair in pairs:
        legacy = _rate(pair.legacy, clause, surface, category)
        balanced = _rate(pair.balanced, clause, surface, category)
        if balanced is None:
            figure0._draw_missing(ax, pair.y)
            continue
        if legacy is None:
            # A #1c cell whose campaign partner is not scored: draw the point
            # it has, and say so, rather than implying a zero-shift pair.
            unpaired += 1
            ax.annotate("no legacy pair", (balanced[0], pair.y),
                        textcoords="offset points", xytext=(11, 0),
                        va="center", fontsize=6.6, style="italic",
                        color=figure0.MUTED, zorder=6)
        if legacy is not None:
            ns.append(legacy[1])
            deltas.append(balanced[0] - legacy[0])
            # The arrow IS the finding; draw it before the markers.
            ax.annotate(
                "", xy=(balanced[0], pair.y), xytext=(legacy[0], pair.y),
                arrowprops=dict(arrowstyle="-|>", color="#9a9a9a",
                                linewidth=1.4, shrinkA=5.0, shrinkB=5.0),
                zorder=3,
            )
            low, high = house.wilson_err(legacy[0] / 100, legacy[1])
            ax.errorbar(legacy[0], pair.y, xerr=[[100 * low], [100 * high]],
                        fmt="o", markersize=7, markerfacecolor="white",
                        markeredgewidth=1.6, color=VARIANT_COLOUR[LEGACY],
                        elinewidth=0.9, capsize=2.0, zorder=4)
        ns.append(balanced[1])
        low, high = house.wilson_err(balanced[0] / 100, balanced[1])
        ax.errorbar(balanced[0], pair.y, xerr=[[100 * low], [100 * high]],
                    fmt="o", markersize=7, color=VARIANT_COLOUR[BALANCED],
                    elinewidth=0.9, capsize=2.0, zorder=5)

    for index in range(1, len(pairs)):
        if pairs[index].profile != pairs[index - 1].profile:
            ax.axhline((pairs[index].y + pairs[index - 1].y) / 2,
                       color=figure0.GRID, linewidth=0.9, zorder=0)
    ax.set_yticks([pair.y for pair in pairs])
    ax.set_yticklabels([pair.label for pair in pairs], fontsize=7.6)
    for label, pair in zip(ax.get_yticklabels(), pairs):
        label.set_color(house.ARM_COLOR.get(pair.arm, figure0.INK))
    ax.set_ylim(pairs[-1].y + 0.8, pairs[0].y - 0.8)
    ax.set_xlim(0, 100)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"), fontsize=8)
    ax.grid(axis="x", color=figure0.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel(f"{figure0.CONFLICT_LABEL[category]}, "
                  f"% of conflict-eval runs", fontsize=9.5, color=figure0.INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(figure0.GRID)

    mean = sum(deltas) / len(deltas) if deltas else float("nan")
    fig.suptitle(
        f"Contamination data quality — {mix.BY_KEY[mixture].label} · "
        f"{figure0.CLAUSE_LABEL[clause]} × {figure0.SURFACE_LABEL[surface]} · "
        f"{mix.EPOCH_LABEL[epoch]}",
        x=0.012, y=1.0 - 0.30 / height, ha="left", fontsize=12.5,
        fontweight="bold", color=figure0.INK,
    )
    handles = [
        Line2D([], [], color=VARIANT_COLOUR[LEGACY], marker="o",
               markerfacecolor="white", markeredgewidth=1.6, linestyle="none",
               label=VARIANT_LABEL[LEGACY]),
        Line2D([], [], color=VARIANT_COLOUR[BALANCED], marker="o",
               linestyle="none", label=VARIANT_LABEL[BALANCED]),
        Line2D([], [], color="#9a9a9a", linewidth=1.4,
               label="arrow: legacy → balanced"),
    ]
    unpaired_note = (
        f" {unpaired} more cell(s) have no scored legacy partner and are drawn "
        f"as a single point." if unpaired else "")
    footnote = (
        f"Paired: {len(deltas)} cells have both draws; mean shift "
        f"{mean:+.1f}pp.{unpaired_note} Everything is held except which 164 conflict "
        f"episodes were selected. {mix.CONTAMINATION_QUALITY_NOTE}"
        f"{backend_note(pairs)} "
        f"Wilson 95% on conflict runs (n={data.n_range(ns) if ns else 'n/a'} "
        f"per point; runs are 3 per episode and not independent, so these are "
        f"optimistic). {house.CAVEAT}."
    )
    wrapped = textwrap.fill(footnote, width=178)
    lines = len(wrapped.splitlines())
    legend_y = 0.16 + 0.135 * lines + 0.16
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False,
               fontsize=8.2, bbox_to_anchor=(0.012, legend_y / height))
    fig.text(0.99, 0.16 / height, wrapped, ha="right", va="bottom",
             color=figure0.MUTED, fontsize=7.4, linespacing=1.25)
    # The axis bottom has to clear its own tick labels AND x label before the
    # legend starts: 0.30 for the legend row, 0.62 for the axis furniture.
    fig.subplots_adjust(left=0.30, right=0.985,
                        top=1.0 - 0.90 / height,
                        bottom=(legend_y + 0.30 + 0.62) / height)
    stem = "__".join((
        mixture.replace("_", "-"), category,
        figure0.SURFACE_STEM[surface], figure0.CLAUSE_STEM[clause],
        f"{epoch}ep",
    ))
    return data.save_figure(fig, stem, output)


def composition_rows(pairs: Sequence[Pair]) -> list[grid.PlotRow]:
    """One section per midtrain cell, legacy and balanced adjacent inside it.

    Adjacency is the whole point: the pair differs in one input, so putting
    the two bars next to each other is what makes the shift readable.
    """
    rows: list[grid.PlotRow] = []
    y = 0.0
    previous: str | None = None
    for pair in pairs:
        section = pair.label
        if previous is not None:
            y += SECTION_GAP
        for variant, unit in ((LEGACY, pair.legacy), (BALANCED, pair.balanced)):
            rows.append(grid.PlotRow(
                section, VARIANT_LABEL[variant], pair.arm, unit,
                variant == LEGACY, y))
            y += ROW_PITCH
        previous = section
    return rows


#: Section headings for the variant-major layout.  Short, because they sit in
#: the row-label gutter, but they still name the one thing that differs.
VARIANT_SECTION = {
    LEGACY: "NOT fixed\nlegacy draw:\n1 clause, one-run only",
    BALANCED: "FIXED\nbalanced draw:\n5 clauses, 82/82 runs",
}


def variant_major_rows(pairs: Sequence[Pair]) -> list[grid.PlotRow]:
    """Transpose of `composition_rows`: one section per DRAW, arms inside.

    `composition_rows` groups by midtrain cell so each pair sits together,
    which is right when there are many cells.  For a single cell the useful
    grouping is the other way round: two blocks of arms, so the two draws read
    as two whole treatments rather than three separate comparisons.
    """
    profiles = {pair.profile for pair in pairs}
    rows: list[grid.PlotRow] = []
    y = 0.0
    for index, variant in enumerate((LEGACY, BALANCED)):
        if index:
            y += SECTION_GAP
        for pair in pairs:
            unit = pair.legacy if variant == LEGACY else pair.balanced
            label = figure0.ARM_LABEL.get(pair.arm, pair.arm)
            if len(profiles) > 1:
                label = f"{grid._profile_title(pair.profile)} · {label}"
            rows.append(grid.PlotRow(
                VARIANT_SECTION[variant], label, pair.arm, unit,
                variant == LEGACY, y))
            y += ROW_PITCH
    return rows


def render_composition(
    mixture: str, pairs: Sequence[Pair], *, epoch: int, surface: str,
    clause: str, output: Path,
) -> list[Path]:
    if not pairs:
        return []
    rows = composition_rows(pairs)
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

    grid._decorate_panel(axes[0], rows, "Ambiguous (agreement episodes)",
                         "share of agreement-eval runs (%)")
    grid._decorate_panel(axes[1], rows, "Diagnostic (conflict episodes)",
                         "share of conflict-eval runs (%)")
    grid._add_section_labels(axes[0], rows)
    fig.suptitle(
        f"Figure 0 — contamination data quality · "
        f"{mix.BY_KEY[mixture].label} · {figure0.CLAUSE_LABEL[clause]} × "
        f"{figure0.SURFACE_LABEL[surface]} · {mix.EPOCH_LABEL[epoch]}",
        x=0.025, y=1.0 - grid.TITLE_DROP / height, ha="left", fontsize=14,
        fontweight="bold", color=figure0.INK,
    )
    footnote = (
        f"{figure0.CLAUSE_LABEL[clause]}, {figure0.SURFACE_LABEL[surface]}; "
        f"agreement {figure0._n_text(agreement_ns)}; "
        f"conflict {figure0._n_text(conflict_ns)}. "
        f"{mix.CONTAMINATION_QUALITY_NOTE}{backend_note(pairs)} Pale bars are "
        f"cells whose partner has not landed yet, not zeros. {house.CAVEAT}."
    )
    grid.compose_layout(fig, axes, height, footnote)
    stem = "__".join((
        mixture.replace("_", "-"), figure0.SURFACE_STEM[surface],
        figure0.CLAUSE_STEM[clause], f"{epoch}ep",
    ))
    return data.save_figure(fig, stem, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--figure", action="append", choices=FIGURES)
    parser.add_argument("--mixture", action="append", choices=DIRECTIONS)
    parser.add_argument("--surface", action="append", choices=figure0.SURFACES)
    parser.add_argument("--clause", action="append", choices=figure0.CLAUSES)
    parser.add_argument("--epoch", action="append", type=int, choices=(1, 2))
    parser.add_argument(
        "--category", action="append", choices=tuple(figure0.CONFLICT_ORDER),
        help="delta only; default: charter (the primary metric)",
    )
    args = parser.parse_args(argv)

    collected = _load(args.collected)
    campaign = data.load_documents(SCORED)
    figures = args.figure or list(FIGURES)
    mixtures = args.mixture or list(DIRECTIONS)
    surfaces = args.surface or list(figure0.SURFACES)
    clauses = args.clause or list(figure0.CLAUSES)
    epochs = args.epoch or list(grid.DEFAULT_EPOCHS)
    categories = args.category or ["charter"]

    written: list[Path] = []
    for mixture in mixtures:
        for epoch in epochs:
            pairs = pairs_for(mixture, epoch, collected=collected,
                              campaign=campaign)
            for surface in surfaces:
                for clause in clauses:
                    if "delta" in figures:
                        for category in categories:
                            written.extend(render_delta(
                                mixture, pairs, epoch=epoch, surface=surface,
                                clause=clause, category=category,
                                output=args.out / "delta"))
                    if "composition" in figures:
                        written.extend(render_composition(
                            mixture, pairs, epoch=epoch, surface=surface,
                            clause=clause, output=args.out / "composition"))

    for path in written:
        print(f"wrote {path}")
    meta = collected.get("meta", {})
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files).")
    print(f"{meta.get('endpoints')}/{meta.get('endpoints_planned')} #1c "
          f"endpoints scored; {len(collected.get('missing', []))} still to land.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
