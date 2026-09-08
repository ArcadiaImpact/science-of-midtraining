"""Follow-up dose ladders, split by charter clause and by episode run count.

The pooled Figure-0 composition answers "which crew did it pick"; these two
views answer "picked where, and consistently?".  Both are read straight out of
the published aggregate — `score_factorised.aggregate` already carries them —
so this re-plots, it never re-scores.

* `by_clause/` — one panel per **target clause**, from
  `conflict_runs_by_clause`.  Run-level, so the categories are the ordinary
  conflict verdicts.  The trained-clause slices have the five trained clauses;
  the held-out slices have `precedence_deferrals` and `qual_weekly_limit`.
  This is the view that says whether a pooled rate is one behaviour or an
  average over clauses that disagree.

* `by_run_count/` — one panel per **episode shape**, from `by_mixture`, whose
  keys are the run kinds joined by "/": `c` is a one-conflict-run episode and
  `c/c` a two-conflict-run one (`a`/`a/a` on the agreement slices).  These are
  EPISODE labels, not run verdicts, which is the point: `mixed` — some runs
  Charter, some coin — can only exist on a two-run episode, so this is where
  within-episode consistency is visible at all.

Both galleries share the parent figures' dose ladder, arms, asterisks and
provenance notes; `--gallery` picks which follow-up's ladder to walk.

Run from the repository root, after `collect_followup_scores.py`::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_followup_breakdown.py
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import followup_mixtures as mix  # noqa: E402
import plot_aft_grid as grid  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402
import plot_contamination_quality as quality  # noqa: E402
import plot_glm_aft_scaleup as scaleup  # noqa: E402
import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402

SCORED = HERE / "scored"
OUTPUT = HERE / "figures" / "ablations"

GALLERIES = ("glm_aft_scaleup", "aft_grid", "contamination_quality")
BREAKDOWNS = ("clause", "run_count")

#: Episode labels from `score_factorised`.  Charter and coin are anchored to
#: opposite edges exactly as the run-level composition does; the middle three
#: are the ways an episode fails to be one or the other, hatched so they read
#: apart without hue.  `mixed` is the interesting one: the model took the
#: Charter side on one run of an episode and the coin side on another.
EPISODE_ORDER = ("all_charter", "mixed", "impure", "malformed", "all_coin")
EPISODE_STYLE = {
    "all_charter": (figure0.CHARTER, None, "white"),
    "mixed": (house.OKABE_ITO["purple"], "\\\\\\\\", "white"),
    "impure": (figure0.OTHER, "////", "#555555"),
    "malformed": (figure0.MALFORMED, "xxxx", "white"),
    "all_coin": (figure0.COIN, None, "white"),
}
EPISODE_LABEL = {
    "all_charter": "every run chose Charter",
    "mixed": "mixed: Charter on one run, coin on another",
    "impure": "impure: some run chose a third crew",
    "malformed": "malformed answer",
    "all_coin": "every run chose coin / cheapest",
}
#: `no_conflict` is the agreement slices' only label and carries no side, so
#: it is dropped rather than drawn as an empty five-way stack.
EPISODE_DROP = ("no_conflict",)

#: Inches. Panels are sized so five clauses and two run counts both read at
#: 100% zoom.  GUTTER is a FLOOR, not the value: the row-label column holds a
#: section bracket plus a row label, and the contamination gallery's are far
#: longer than the dose ladders' ("Gemma 3 27B · 190M presented · coin prior"
#: against "balanced · 5 clauses, 82/82 runs"), so it is measured per figure.
GUTTER = 4.0
PANEL_WIDTH = 4.3


def gutter_for(rows) -> float:
    """Width the row-label column needs, from the labels actually in it."""
    section = max((len(line) for row in rows
                   for line in row.section.split("\n")), default=0)
    label = max((len(row.label) for row in rows), default=0)
    return max(GUTTER, 1.0 + 0.062 * section + 0.058 * label)

RUN_COUNT_LABEL = {
    "c": "one conflict run",
    "c/c": "two conflict runs",
    "a": "one agreement run",
    "a/a": "two agreement runs",
    "a/c": "agreement then conflict",
    "c/a": "conflict then agreement",
}

CLAUSE_LABEL = {
    "precedence_days_since": "precedence · days since",
    "precedence_registry_rank": "precedence · registry rank",
    "precedence_runs_year": "precedence · runs per year",
    "qual_skill": "qualification · skill",
    "qual_specialty": "qualification · specialty",
    "precedence_deferrals": "precedence · deferrals  (held out)",
    "qual_weekly_limit": "qualification · weekly limit  (held out)",
}


def _shares(counts: Mapping[str, Any], order: Sequence[str]) -> tuple[dict, int] | None:
    """Normalise a count mapping into a 100% stack, or None when empty."""
    values: dict[str, float] = {}
    total = 0
    for category, count in counts.items():
        if category in EPISODE_DROP:
            continue
        if not isinstance(count, int) or count < 0:
            return None
        total += count
        if category in order:
            values[category] = float(count)
    if total <= 0:
        return None
    # Any category outside `order` would silently vanish from a 100% stack.
    if abs(sum(values.values()) - total) > 0.5:
        raise ValueError(f"unmodelled categories in {sorted(counts)}")
    return {c: values.get(c, 0.0) / total for c in order}, total


def panel_readings(
    unit: data.Unit | None, clause: str, surface: str, breakdown: str,
) -> dict[str, tuple[dict, int]]:
    """Per-panel shares for one row: keyed by clause, or by episode shape."""
    if unit is None:
        return {}
    cell = data.cell_for(unit, clause, "conflict", surface)
    if cell is None:
        return {}
    if breakdown == "clause":
        block, order = cell.get("conflict_runs_by_clause"), data.CONFLICT_ORDER
    else:
        block, order = cell.get("by_mixture"), EPISODE_ORDER
    if not isinstance(block, dict):
        return {}
    out: dict[str, tuple[dict, int]] = {}
    for key, counts in block.items():
        if not isinstance(counts, dict):
            continue
        reading = _shares(counts, order)
        if reading is not None:
            out[str(key)] = reading
    return out


def panel_keys(
    rows: Sequence[Any], clause: str, surface: str, breakdown: str,
) -> list[str]:
    """Every panel any row has data for, so panels stay stable across rows."""
    keys: set[str] = set()
    for row in rows:
        keys |= set(panel_readings(row.unit, clause, surface, breakdown))
    if breakdown == "clause":
        known = [key for key in CLAUSE_LABEL if key in keys]
        return known + sorted(keys - set(known))
    known = [key for key in RUN_COUNT_LABEL if key in keys]
    return known + sorted(keys - set(known))


def _box_narrow_clause(ax, rows: Sequence[Any]) -> None:
    """Ring the starred rows in the clause the legacy 2% draw trained on.

    Those rows are the ONLY place the legacy draw had in-distribution
    conflict data, so this panel is where the two draws are expected to
    agree -- and every other panel is the measurement.  Drawn in achromatic
    diagnostic ink and carried by shape plus a label, never by hue.
    """
    # CONTIGUOUS runs, not one box from the first starred row to the last:
    # a full dose ladder has two separate starred blocks (2% coin and 2%
    # Charter) with unstarred rungs between them.
    runs: list[list[Any]] = []
    for index, row in enumerate(rows):
        if not getattr(row, "starred", False):
            continue
        if runs and index == runs[-1][-1][0] + 1:
            runs[-1].append((index, row))
        else:
            runs.append([(index, row)])
    for run in runs:
        ys = [row.y for _index, row in run]
        top = min(ys) - figure0.BAR_HEIGHT / 2 - 0.16
        bottom = max(ys) + figure0.BAR_HEIGHT / 2 + 0.16
        ax.add_patch(Rectangle(
            (-1.2, top), 102.4, bottom - top, transform=ax.transData,
            facecolor="none", edgecolor=house.DIAG, linewidth=1.6,
            linestyle=(0, (5, 2.5)), zorder=7, clip_on=False,
        ))
        # Below the box, in the section gap: above it collides with the
        # panel title.
        ax.text(50, bottom + 0.07, mix.NARROW_CLAUSE_BOX, ha="center",
                va="top", fontsize=6.6, color=house.DIAG, zorder=8,
                clip_on=False)


def render(
    rows: Sequence[Any],
    *,
    breakdown: str,
    surface: str,
    clause: str,
    title: str,
    stem: str,
    footnote_extra: str,
    output: Path,
    narrow_note: str = mix.NARROW_NOTE,
) -> list[Path]:
    keys = panel_keys(rows, clause, surface, breakdown)
    if not keys:
        return []
    order = data.CONFLICT_ORDER if breakdown == "clause" else EPISODE_ORDER
    styles = figure0.CONFLICT_STYLE if breakdown == "clause" else EPISODE_STYLE
    labels = figure0.CONFLICT_LABEL if breakdown == "clause" else EPISODE_LABEL
    key_label = CLAUSE_LABEL if breakdown == "clause" else RUN_COUNT_LABEL
    unit_word = "conflict-eval runs" if breakdown == "clause" else "episodes"

    height = max(8.0, 2.4 + 0.28 * len(rows))
    # The row-label gutter is a fixed physical width: it holds the ladder's
    # section brackets, whose text ("164/8,192 · 1,638/81,920 conflict rows")
    # does not shrink when the panel count does.
    gutter = gutter_for(rows)
    width = gutter + PANEL_WIDTH * len(keys) + 0.3
    fig, axes = plt.subplots(1, len(keys), figsize=(width, height),
                             sharey=True, squeeze=False)
    ns: list[int] = []
    for index, key in enumerate(keys):
        ax = axes[0][index]
        for row in rows:
            reading = panel_readings(
                row.unit, clause, surface, breakdown).get(key)
            if reading is None:
                if getattr(row, "planned", True):
                    figure0._draw_missing(ax, row.y)
                else:
                    grid.draw_not_in_study(ax, row.y)
                continue
            shares, total = reading
            ns.append(total)
            figure0._draw_stack(ax, row.y, shares, order, styles)
        if breakdown == "clause" and key == mix.NARROW_CLAUSE:
            _box_narrow_clause(ax, rows)
        grid._decorate_panel(
            ax, rows, key_label.get(key, key.replace("_", " ")),
            f"share of {unit_word} (%)",
        )
        if index:
            # Only the leftmost panel carries the row labels; repeating a
            # 27-row ladder five times is noise.  This MUST go through
            # tick_params: on a shared y axis, set_yticklabels([]) replaces
            # the group's formatter and blanks the leftmost panel too.
            ax.tick_params(labelleft=False)
    grid._add_section_labels(axes[0][0], rows)

    fig.suptitle(title, x=0.3 / width, y=1.0 - grid.TITLE_DROP / height,
                 ha="left", fontsize=13, fontweight="bold", color=figure0.INK)
    handles = [
        Patch(facecolor=styles[item][0], edgecolor=styles[item][2],
              hatch=styles[item][1], label=labels[item])
        for item in order
    ]
    wrapped = textwrap.fill(
        f"{figure0.CLAUSE_LABEL[clause]}, {figure0.SURFACE_LABEL[surface]}; "
        f"n={data.n_range(ns) if ns else 'n/a'} {unit_word} per bar. "
        f"{footnote_extra} {narrow_note} {house.CAVEAT}.".replace("  ", " "),
        width=int(width * 12),
    )
    # Long episode-label text needs fewer columns than the four short
    # conflict verdicts do; size the legend to the figure rather than the
    # category count.
    ncol = max(2, min(len(order), int(width // 4.6)))
    legend_rows = -(-len(order) // ncol)
    legend_y = grid.LEGEND_GAP + 0.22 * legend_rows
    footnote_top = legend_y + grid.FOOTNOTE_LINE * len(wrapped.splitlines())
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               frameon=False, fontsize=8.0,
               bbox_to_anchor=(0.5, grid.FOOTNOTE_PAD / height))
    fig.text(0.99, legend_y / height, wrapped, ha="right", va="bottom",
             color=figure0.MUTED, fontsize=7.5, linespacing=1.25)
    fig.subplots_adjust(
        left=gutter / width, right=0.995,
        top=1.0 - grid.PANEL_TITLE_DROP / height,
        bottom=(footnote_top + 0.55) / height, wspace=0.06,
    )
    return data.save_figure(fig, stem, output)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing — run collect_followup_scores.py first")
    return json.loads(path.read_text())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gallery", action="append", choices=GALLERIES)
    parser.add_argument("--breakdown", action="append", choices=BREAKDOWNS)
    parser.add_argument("--surface", action="append", choices=figure0.SURFACES)
    parser.add_argument("--clause", action="append", choices=figure0.CLAUSES)
    parser.add_argument(
        "--profile", action="append",
        help=("restrict to these midtrain profiles; default: all. Applies to "
              "aft_grid and contamination_quality, and goes into the filename."))
    parser.add_argument(
        "--mixture", action="append", choices=[m.key for m in mix.MIXTURES],
        help=("aft_grid only; restrict the dose ladder to these rungs "
              "(pre-AFT is always kept as the anchor). The filter goes into "
              "the filename so a focused figure never overwrites the full one."),
    )
    parser.add_argument(
        "--layout", choices=("pair-major", "variant-major"),
        default="pair-major",
        help=("contamination_quality only. pair-major (default) groups by "
              "midtrain cell with the two draws adjacent; variant-major "
              "groups by draw with the arms inside, and writes to "
              "headline/ — the shape for a single-cell share figure."),
    )
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    galleries = args.gallery or list(GALLERIES)
    breakdowns = args.breakdown or list(BREAKDOWNS)
    surfaces = args.surface or list(figure0.SURFACES)
    clauses = args.clause or list(figure0.CLAUSES)
    campaign = data.load_documents(SCORED)

    written: list[Path] = []
    for gallery in galleries:
        collected = _load(SCORED / "ablations" / f"{gallery}.json")
        for breakdown in breakdowns:
            for surface in surfaces:
                for clause in clauses:
                    if gallery == "glm_aft_scaleup":
                        variants = [v for v in scaleup.VARIANTS
                                    if v.key in scaleup.DEFAULT_VARIANTS]
                        rows = scaleup.ladder_rows(
                            collected=collected, campaign=campaign,
                            variants=variants)
                        written.extend(render(
                            rows, breakdown=breakdown, surface=surface,
                            clause=clause,
                            title=(f"GLM AFT size scale-up · by "
                                   f"{'clause' if breakdown == 'clause' else 'episode run count'}"
                                   f" · {figure0.CLAUSE_LABEL[clause]} × "
                                   f"{figure0.SURFACE_LABEL[surface]}"),
                            stem="__".join((figure0.SURFACE_STEM[surface],
                                            figure0.CLAUSE_STEM[clause])),
                            footnote_extra=(
                                f"Converged 2-epoch endpoints. "
                                f"{mix.BACKEND_NOTE}"),
                            output=(args.out / "GLM-AFT-scaleup"
                                    / f"breakdown_by_{breakdown}"),
                        ))
                    elif gallery == "contamination_quality":
                        for mixture in quality.DIRECTIONS:
                            pairs = quality.pairs_for(
                                mixture, 2, collected=collected,
                                campaign=campaign)
                            if args.profile:
                                pairs = [pair for pair in pairs
                                         if pair.profile in set(args.profile)]
                            variant_major = args.layout == "variant-major"
                            rows = (quality.variant_major_rows(pairs)
                                    if variant_major
                                    else quality.composition_rows(pairs))
                            if not rows:
                                continue
                            written.extend(render(
                                rows, breakdown=breakdown, surface=surface,
                                clause=clause,
                                title=(f"Contamination data quality · "
                                       f"{mix.BY_KEY[mixture].label} · by "
                                       f"{'clause' if breakdown == 'clause' else 'episode run count'}"
                                       f" · {figure0.CLAUSE_LABEL[clause]} × "
                                       f"{figure0.SURFACE_LABEL[surface]}"),
                                stem="__".join((
                                    *(p.replace("_", "-")
                                      for p in (args.profile or ())),
                                    mixture.replace("_", "-"),
                                    *((f"by-{breakdown.replace('_', '-')}",)
                                      if variant_major else ()),
                                    figure0.SURFACE_STEM[surface],
                                    figure0.CLAUSE_STEM[clause])),
                                footnote_extra=(
                                    "Converged 2-epoch endpoints. "
                                    "Legacy trained its 164 conflict rows on "
                                    "precedence_days_since ALONE, so the four "
                                    "other trained clauses are held out for "
                                    "it and trained for balanced -- that one "
                                    "clause's legacy trio is boxed. "
                                    + mix.CONTAMINATION_QUALITY_NOTE),
                                output=(args.out / "contamination-data-quality"
                                        / ("headline" if variant_major
                                           else f"breakdown_by_{breakdown}")),
                                # Both rows here ARE 2%; the generic
                                # narrow-conflict caveat is what this gallery
                                # measures, so quoting it as a warning would
                                # contradict the figure.
                                narrow_note="",
                            ))
                    else:
                        profiles = args.profile or [
                            house.PLAN[(model, dose)]
                            for model in grid.MODELS for dose in house.DOSES
                            if (model, dose) in house.PLAN
                        ]
                        for profile in profiles:
                            rows = grid.profile_rows(
                                profile, collected=collected,
                                campaign=campaign,
                                epochs=list(grid.DEFAULT_EPOCHS),
                                mixtures=args.mixture)
                            if not rows:
                                continue
                            written.extend(render(
                                rows, breakdown=breakdown, surface=surface,
                                clause=clause,
                                title=(f"AFT mixture grid · "
                                       f"{grid._profile_title(profile)} · by "
                                       f"{'clause' if breakdown == 'clause' else 'episode run count'}"
                                       f" · {figure0.CLAUSE_LABEL[clause]} × "
                                       f"{figure0.SURFACE_LABEL[surface]}"),
                                stem="__".join((
                                    profile.replace("_", "-"),
                                    *(m.replace("_", "-")
                                      for m in (args.mixture or ())),
                                    figure0.SURFACE_STEM[surface],
                                    figure0.CLAUSE_STEM[clause])),
                                footnote_extra="Converged 2-epoch endpoints.",
                                output=(args.out / "AFT-grid"
                                        / f"breakdown_by_{breakdown}"),
                            ))

    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
