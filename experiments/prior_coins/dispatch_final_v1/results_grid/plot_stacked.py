"""Paired agreement-success and conflict-composition figures for the grid.

Two data-driven figure families are rendered from every ``eval.json`` found
under ``results_grid/scored``:

* ``stacked_profile__<profile>``: every scored endpoint x arm in one profile;
* ``stacked_aft__<cell>``: every scored profile x checkpoint x arm for one AFT
  run type.

Each row is a visual pair.  ``A`` is the success rate on agreement episodes
(``agreement_runs.rates.shared``); ``C`` is the 100%-stacked choice composition
on conflict episodes.  The three panels hold one axis fixed while changing the
other: trained clauses / canonical template, held-out clauses / canonical
template, and trained clauses / held-out template.

Run from the repository root::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_stacked.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plot_grid as house  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORED = HERE / "scored"
OUTPUT = HERE / "figures" / "stacked"

# These are the minimal non-confounded views requested for this figure family.
# A slice changes the clause split; a surface changes the presentation template.
PANELS: tuple[tuple[str, str, str], ...] = (
    ("trained", "canonical", "Trained clauses · canonical template"),
    ("holdout", "canonical", "Held-out clauses · canonical template"),
    ("trained", "heldout", "Trained clauses · held-out template"),
)

GREEN = house.OKABE_ITO["green"]
BLUE = house.OKABE_ITO["blue"]
YELLOW = house.OKABE_ITO["yellow"]
BLACK = house.OKABE_ITO["black"]
OTHER_GREY = house.UNCOVERED_INK
TRACK = house.UNCOVERED_FILL
GRID = "#e7e7e7"
MUTED = "#5f5f5f"

# Charter and coin are anchored to opposite edges.  The nuisance outcomes stay
# in the middle, where hatch distinguishes other from malformed even without
# colour (and makes a very small segment visibly non-empty).
CONFLICT_ORDER = ("charter", "other", "malformed", "coin")
CONFLICT_STYLE = {
    "charter": (BLUE, None, "white"),
    "other": (OTHER_GREY, "////", "#555555"),
    "malformed": (BLACK, "xxxx", "white"),
    "coin": (YELLOW, None, "#8a8300"),
}
CONFLICT_LABEL = {
    "charter": "Charter crew",
    "other": "another crew",
    "malformed": "malformed answer",
    "coin": "coin / cheapest crew",
}

ENDPOINT_RE = re.compile(r"^(?P<family>.+)-step(?P<step>\d+)$")
SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")

PAIR_OFFSET = 0.185
BAR_HEIGHT = 0.29
UNIT_PITCH = 0.82


@dataclass(frozen=True)
class Unit:
    """One endpoint x arm, backed by one scored eval document."""

    profile: str
    arm: str
    endpoint: str
    document: Mapping[str, Any]


@dataclass(frozen=True)
class AgreementReading:
    rate: float
    n_runs: int
    n_episodes: int | None


@dataclass(frozen=True)
class ConflictReading:
    shares: Mapping[str, float]
    n_runs: int
    n_episodes: int | None


@dataclass(frozen=True)
class Placement:
    unit: Unit
    y: float
    gap_before: float


def natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """A stable key that puts 4B before 12B and step256 before step512."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def load_documents(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load every available arm's eval artifact; unrelated JSON is ignored."""
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    if not root.is_dir():
        return documents
    for path in sorted(root.glob("*/*/eval.json")):
        profile = path.parent.parent.name
        arm = path.parent.name
        document = json.loads(path.read_text())
        if not isinstance(document.get("result"), dict):
            print(f"skip {path}: no result mapping")
            continue
        documents[(profile, arm)] = document
    return documents


def endpoint_parts(endpoint: str) -> tuple[str, int | None]:
    match = ENDPOINT_RE.fullmatch(endpoint)
    if match is None:
        return endpoint, None
    return match.group("family"), int(match.group("step"))


def endpoint_sort_key(endpoint: str) -> tuple[Any, ...]:
    """Prefer the contract's order, then retain any newly scored families."""
    family, step = endpoint_parts(endpoint)
    preferred = ("pre_aft", *house.C.AFT_CELLS)
    try:
        family_rank = preferred.index(family)
        family_key: tuple[Any, ...] = (0, family_rank)
    except ValueError:
        family_key = (1, natural_key(family))
    return (*family_key, -1 if step is None else step, natural_key(endpoint))


def arm_sort_key(arm: str) -> tuple[Any, ...]:
    try:
        return (0, house.C.ARM_ORDER.index(arm))
    except ValueError:
        return (1, natural_key(arm))


def endpoints_in(document: Mapping[str, Any]) -> set[str]:
    result = document.get("result", {})
    return {str(key) for key, value in result.items() if isinstance(value, dict)}


def cell_for(
    unit: Unit, clause: str, run_kind: str, surface: str
) -> Mapping[str, Any] | None:
    endpoint = unit.document.get("result", {}).get(unit.endpoint, {})
    cell = endpoint.get(f"eval_{clause}_{run_kind}__{surface}")
    return cell if isinstance(cell, dict) else None


def positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def agreement_reading(
    unit: Unit, clause: str, surface: str
) -> AgreementReading | None:
    cell = cell_for(unit, clause, "agreement", surface)
    if cell is None:
        return None
    runs = cell.get("agreement_runs", {})
    n_runs = positive_int(runs.get("n")) if isinstance(runs, dict) else None
    rates = runs.get("rates", {}) if isinstance(runs, dict) else {}
    if n_runs is None or not isinstance(rates, dict):
        return None
    # score_factorised.SHARED is the one correct crew when both rules agree.
    rate = rates.get("shared", 0.0)
    if not isinstance(rate, (int, float)) or not 0.0 <= rate <= 1.0:
        return None
    return AgreementReading(float(rate), n_runs, positive_int(cell.get("n")))


def _exact_conflict_shares(
    cell: Mapping[str, Any], n_runs: int
) -> dict[str, float] | None:
    """Recover exact shares from per-clause counts when they match the n."""
    by_clause = cell.get("conflict_runs_by_clause")
    if not isinstance(by_clause, dict):
        return None
    counts: Counter[str] = Counter()
    for clause_counts in by_clause.values():
        if not isinstance(clause_counts, dict):
            return None
        for category, count in clause_counts.items():
            if category in CONFLICT_ORDER and isinstance(count, int) and count >= 0:
                counts[category] += count
    if sum(counts.values()) != n_runs:
        return None
    return {category: counts[category] / n_runs for category in CONFLICT_ORDER}


def conflict_reading(
    unit: Unit, clause: str, surface: str
) -> ConflictReading | None:
    cell = cell_for(unit, clause, "conflict", surface)
    if cell is None:
        return None
    runs = cell.get("conflict_runs", {})
    n_runs = positive_int(runs.get("n")) if isinstance(runs, dict) else None
    rates = runs.get("rates", {}) if isinstance(runs, dict) else {}
    if n_runs is None or not isinstance(rates, dict):
        return None

    shares = _exact_conflict_shares(cell, n_runs)
    if shares is None:
        shares = {}
        for category in CONFLICT_ORDER:
            value = rates.get(category, 0.0)
            if not isinstance(value, (int, float)) or value < 0:
                return None
            shares[category] = float(value)
        total = sum(shares.values())
        if total <= 0:
            return None
        # Scored rates are rounded to four decimals.  Normalisation closes the
        # possible one-pixel rounding seam and keeps this a literal 100% stack.
        shares = {category: value / total for category, value in shares.items()}
    return ConflictReading(shares, n_runs, positive_int(cell.get("n")))


def profile_units(
    profile: str, documents: Mapping[tuple[str, str], Mapping[str, Any]]
) -> list[Unit]:
    arms = sorted(
        (arm for candidate, arm in documents if candidate == profile),
        key=arm_sort_key,
    )
    endpoints = sorted(
        set().union(*(endpoints_in(documents[(profile, arm)]) for arm in arms)),
        key=endpoint_sort_key,
    )
    return [
        Unit(profile, arm, endpoint, documents[(profile, arm)])
        for endpoint in endpoints
        for arm in arms
        if endpoint in endpoints_in(documents[(profile, arm)])
    ]


def aft_units(
    family: str, documents: Mapping[tuple[str, str], Mapping[str, Any]]
) -> list[Unit]:
    profiles = sorted({profile for profile, _ in documents}, key=natural_key)
    units: list[Unit] = []
    for profile in profiles:
        arms = sorted(
            (arm for candidate, arm in documents if candidate == profile),
            key=arm_sort_key,
        )
        endpoints = sorted(
            {
                endpoint
                for arm in arms
                for endpoint in endpoints_in(documents[(profile, arm)])
                if endpoint_parts(endpoint)[0] == family
                and endpoint_parts(endpoint)[1] is not None
            },
            key=endpoint_sort_key,
        )
        units.extend(
            Unit(profile, arm, endpoint, documents[(profile, arm)])
            for endpoint in endpoints
            for arm in arms
            if endpoint in endpoints_in(documents[(profile, arm)])
        )
    return units


def profile_gap(previous: Unit, current: Unit) -> float:
    previous_family, _ = endpoint_parts(previous.endpoint)
    current_family, _ = endpoint_parts(current.endpoint)
    if current_family != previous_family:
        return 0.46
    if current.endpoint != previous.endpoint:
        return 0.24
    return 0.0


def aft_gap(previous: Unit, current: Unit) -> float:
    if current.profile != previous.profile:
        return 0.52
    if current.endpoint != previous.endpoint:
        return 0.24
    return 0.0


def place_units(
    units: Sequence[Unit], gap: Callable[[Unit, Unit], float]
) -> list[Placement]:
    placements: list[Placement] = []
    y = 0.5
    for index, unit in enumerate(units):
        gap_before = 0.0 if index == 0 else gap(units[index - 1], unit)
        if index:
            y += UNIT_PITCH + gap_before
        placements.append(Placement(unit, y, gap_before))
    return placements


def n_range(values: Sequence[int | None]) -> str:
    present = sorted({value for value in values if value is not None})
    if not present:
        return "—"
    if len(present) == 1:
        return f"{present[0]:,}"
    return f"{present[0]:,}–{present[-1]:,}"


def panel_n_text(
    units: Sequence[Unit], clause: str, surface: str
) -> str:
    agreements = [agreement_reading(unit, clause, surface) for unit in units]
    conflicts = [conflict_reading(unit, clause, surface) for unit in units]
    return (
        "A n="
        f"{n_range([item.n_runs if item else None for item in agreements])} runs / "
        f"{n_range([item.n_episodes if item else None for item in agreements])} episodes"
        "  ·  C n="
        f"{n_range([item.n_runs if item else None for item in conflicts])} runs / "
        f"{n_range([item.n_episodes if item else None for item in conflicts])} episodes"
    )


def add_value_label(ax, y: float, width: float, colour: str) -> None:
    """Put a whole-percent value inside when possible, just outside if tiny."""
    if width >= 11.0:
        ax.text(
            width / 2,
            y,
            f"{width:.0f}",
            ha="center",
            va="center",
            fontsize=6.7,
            color="white",
            fontweight="bold",
            zorder=7,
        )
    else:
        ax.text(
            min(98.5, width + 1.0),
            y,
            f"{width:.0f}",
            ha="left" if width < 97.5 else "right",
            va="center",
            fontsize=6.4,
            color=colour,
            fontweight="bold",
            zorder=7,
        )


def draw_agreement(ax, y: float, reading: AgreementReading | None) -> None:
    ax.barh(
        y,
        100,
        height=BAR_HEIGHT,
        color=TRACK,
        edgecolor="#a0a0a0",
        linewidth=0.45,
        zorder=2,
    )
    if reading is None:
        ax.text(50, y, "missing", ha="center", va="center", fontsize=6.3,
                color="#999999", style="italic", zorder=6)
        return
    width = 100 * reading.rate
    ax.barh(
        y,
        width,
        height=BAR_HEIGHT,
        color=GREEN,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    lower, upper = house.wilson_err(reading.rate, reading.n_runs)
    ax.errorbar(
        width,
        y,
        xerr=[[100 * lower], [100 * upper]],
        fmt="none",
        ecolor="#2c2c2c",
        elinewidth=0.8,
        capsize=1.8,
        capthick=0.8,
        zorder=6,
    )
    add_value_label(ax, y, width, GREEN)


def segment_text_colour(category: str) -> str:
    return "white" if category in {"charter", "malformed"} else "#222222"


def draw_conflict(ax, y: float, reading: ConflictReading | None) -> None:
    if reading is None:
        ax.barh(
            y,
            100,
            height=BAR_HEIGHT,
            color=TRACK,
            edgecolor="#a0a0a0",
            linewidth=0.45,
            zorder=2,
        )
        ax.text(50, y, "missing", ha="center", va="center", fontsize=6.3,
                color="#999999", style="italic", zorder=6)
        return

    left = 0.0
    for category in CONFLICT_ORDER:
        width = 100 * reading.shares.get(category, 0.0)
        if width <= 0:
            continue
        colour, hatch, edge = CONFLICT_STYLE[category]
        ax.barh(
            y,
            width,
            left=left,
            height=BAR_HEIGHT,
            color=colour,
            edgecolor=edge,
            linewidth=0.45,
            hatch=hatch,
            zorder=4,
        )
        if width >= 7.0:
            ax.text(
                left + width / 2,
                y,
                f"{width:.0f}",
                ha="center",
                va="center",
                fontsize=6.7,
                color=segment_text_colour(category),
                fontweight="bold",
                zorder=7,
            )
        left += width
    # A common outline makes the literal 0--100 extent unambiguous even when
    # the rightmost yellow segment is very pale.
    ax.barh(
        y,
        100,
        height=BAR_HEIGHT,
        color="none",
        edgecolor="#555555",
        linewidth=0.45,
        zorder=6,
    )


def draw_panel(
    ax,
    placements: Sequence[Placement],
    clause: str,
    surface: str,
    title: str,
    row_label: Callable[[Unit], str],
) -> None:
    for index, placement in enumerate(placements):
        y_agreement = placement.y - PAIR_OFFSET
        y_conflict = placement.y + PAIR_OFFSET
        if index % 2 == 0:
            ax.axhspan(
                placement.y - UNIT_PITCH * 0.43,
                placement.y + UNIT_PITCH * 0.43,
                color="#fafafa",
                zorder=0,
            )
        if placement.gap_before:
            separator = placement.y - (UNIT_PITCH + placement.gap_before) / 2
            ax.axhline(
                separator,
                color="#bcbcbc" if placement.gap_before >= 0.4 else GRID,
                linewidth=0.85 if placement.gap_before >= 0.4 else 0.6,
                linestyle="-" if placement.gap_before >= 0.4 else (0, (3, 3)),
                zorder=1,
            )
        draw_agreement(
            ax,
            y_agreement,
            agreement_reading(placement.unit, clause, surface),
        )
        draw_conflict(
            ax,
            y_conflict,
            conflict_reading(placement.unit, clause, surface),
        )

    ticks: list[float] = []
    labels: list[str] = []
    for placement in placements:
        label = row_label(placement.unit)
        ticks.extend((placement.y - PAIR_OFFSET, placement.y + PAIR_OFFSET))
        labels.extend((f"A · {label}", f"C · {label}"))
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, 100)
    if placements:
        ax.set_ylim(placements[-1].y + 0.55, placements[0].y - 0.55)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"), fontsize=7.5)
    ax.grid(axis="x", color=GRID, linewidth=0.65, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{title}\n{panel_n_text([item.unit for item in placements], clause, surface)}",
        fontsize=8.8,
        pad=8,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b8b8b8")
    ax.tick_params(axis="x", colors=MUTED)


def legend_handles() -> list[Patch]:
    handles = [
        Patch(
            facecolor=GREEN,
            edgecolor="#444444",
            linewidth=0.4,
            label="A · correct shared pick (Wilson 95% whisker)",
        )
    ]
    for category in CONFLICT_ORDER:
        colour, hatch, edge = CONFLICT_STYLE[category]
        handles.append(
            Patch(
                facecolor=colour,
                edgecolor=edge,
                linewidth=0.6,
                hatch=hatch,
                label=f"C · {CONFLICT_LABEL[category]}",
            )
        )
    return handles


def save_figure(fig, stem: str, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    safe_stem = SAFE_STEM_RE.sub("_", stem).strip("_")
    png = output / f"{safe_stem}.png"
    svg = output / f"{safe_stem}.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def render(
    units: Sequence[Unit],
    *,
    title: str,
    subtitle: str,
    stem: str,
    output: Path,
    row_label: Callable[[Unit], str],
    gap: Callable[[Unit, Unit], float],
) -> list[Path]:
    if not units:
        return []
    placements = place_units(units, gap)
    height = max(8.2, 3.4 + 0.36 * len(units))
    fig, axes = plt.subplots(1, len(PANELS), figsize=(23.0, height), squeeze=False)
    for ax, (clause, surface, panel_title) in zip(axes[0], PANELS, strict=True):
        draw_panel(ax, placements, clause, surface, panel_title, row_label)

    fig.suptitle(title, fontsize=13.0, fontweight="bold", y=0.997)
    fig.text(0.5, 0.973, subtitle, ha="center", va="top", fontsize=9.0,
             color=MUTED)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=8.0,
        bbox_to_anchor=(0.5, 0.956),
    )
    house.footnote(
        fig,
        "Each unit pairs A (agreement) with C (conflict). Green is the shared "
        "correct pick on agreement runs; the light tail is all errors, including "
        "malformed answers. Conflict bars are exact 100% compositions; Charter "
        "and coin are edge-anchored, while hatched other and cross-hatched "
        "malformed remain distinct in greyscale. Panel headings report run and "
        "episode n per bar. Wilson 95% is shown for agreement success; no segment "
        "interval is overlaid on a stack. Runs within an episode are clustered, "
        "so Wilson intervals are optimistic. "
        f"CAVEAT: {house.CAVEAT}.",
    )
    fig.tight_layout(rect=(0.006, 0.045, 0.994, 0.925), w_pad=2.6)
    return save_figure(fig, stem, output)


def aft_family_label(family: str) -> str:
    """Describe a discovered AFT cell from the campaign contract when known."""
    conflict_label = house.C.AFT_CELL_CONFLICT_LABEL.get(family)
    conflict_rows = house.C.AFT_CELL_CONFLICT_ROWS.get(family)
    if conflict_rows is None:
        return family.replace("_", " ")
    if conflict_label is None:
        return "agreement-only"
    percentage = 100 * conflict_rows / house.C.AFT_ROWS
    amount = f"{percentage:.0f}" if abs(percentage - round(percentage)) < 0.05 else f"{percentage:.1f}"
    return f"{amount}% {conflict_label}-labelled"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scored", type=Path, default=SCORED,
                        help="root containing <profile>/<arm>/eval.json")
    parser.add_argument("--out", type=Path, default=OUTPUT,
                        help="directory for PNG and SVG output")
    args = parser.parse_args()

    documents = load_documents(args.scored)
    if not documents:
        print(f"nothing to plot under {args.scored}")
        return 0

    written: list[Path] = []
    profiles = sorted({profile for profile, _ in documents}, key=natural_key)
    for profile in profiles:
        units = profile_units(profile, documents)
        written.extend(
            render(
                units,
                title=f"Dispatch final-v1 stacked composition · {profile}",
                subtitle="Every available endpoint × arm in this profile",
                stem=f"stacked_profile__{profile}",
                output=args.out,
                row_label=lambda unit: f"{unit.endpoint} · {unit.arm}",
                gap=profile_gap,
            )
        )

    families = sorted(
        {
            endpoint_parts(endpoint)[0]
            for document in documents.values()
            for endpoint in endpoints_in(document)
            if endpoint_parts(endpoint)[1] is not None
        },
        key=lambda family: endpoint_sort_key(f"{family}-step0"),
    )
    for family in families:
        units = aft_units(family, documents)
        written.extend(
            render(
                units,
                title=("Dispatch final-v1 stacked composition · AFT cell: "
                       f"{aft_family_label(family)}"),
                subtitle="Every available profile × checkpoint × arm for this AFT run type",
                stem=f"stacked_aft__{family}",
                output=args.out,
                row_label=lambda unit: (
                    f"{unit.profile} · {unit.endpoint} · {unit.arm}"
                ),
                gap=aft_gap,
            )
        )

    for path in written:
        print(f"wrote {path}")
    print(
        f"\n{len(written)} files from {len(profiles)} profiles and "
        f"{len(families)} discovered AFT run types."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
