"""Render classic Figure-0 plots for arbitrary grid slices.

The default run writes one figure for every available campaign cell in

    surface x clause split x model x presented-token budget.

Each figure has the established Figure-0 shape: agreement episodes on the
left, conflict episodes on the right, with 100%-stacked response composition
and rows grouped first by eval endpoint and then by midtraining arm.  All AFT
endpoints live in the same figure; sibling AFT mixtures are never connected by
a line.

Examples, from the repository root::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_figure0_slices.py

    # One reusable slice (filters can be repeated):
    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/plot_figure0_slices.py \
      --model gemma3_12b --dose 19m --surface heldout --clause holdout
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORED = HERE / "scored"
OUTPUT = HERE / "figures" / "figure0_slices"

SURFACES = ("canonical", "trained", "heldout")
CLAUSES = ("trained", "holdout")
SURFACE_LABEL = {
    "canonical": "canonical template",
    "trained": "trained templates",
    "heldout": "held-out templates",
}
SURFACE_STEM = {
    "canonical": "canonical",
    "trained": "trained-template",
    "heldout": "heldout-template",
}
CLAUSE_LABEL = {"trained": "trained clauses", "holdout": "held-out clauses"}
CLAUSE_STEM = {"trained": "trained-clause", "holdout": "heldout-clause"}

# Figure 0 has always placed the neutral substrate between the two directional
# priors.  This differs deliberately from the scoring contract's storage order.
ARMS = ("charter", "control", "coin")
ARM_LABEL = {
    "charter": "charter prior",
    "control": "control",
    "coin": "coin prior",
}

# Match the established Figure-0 palette in plot_wave_v1_summary.py.  These
# are intentionally local rather than inherited from plot_stacked.py: the
# latter is a different figure family with yellow and hatch-based encoding.
CHARTER = "#0173b2"
COIN = "#de8f05"
OTHER = "#949494"
MALFORMED = "#22221f"
SHARED = "#029e73"
GRID = "#deded8"
INK = "#22221f"
MUTED = "#777772"

AGREEMENT_ORDER = ("shared", "other", "malformed")
AGREEMENT_STYLE = {
    "shared": (SHARED, None, "white"),
    "other": (OTHER, None, "white"),
    "malformed": (MALFORMED, None, "white"),
}
AGREEMENT_LABEL = {
    "shared": "correct shared crew",
    "other": "another crew",
    "malformed": "malformed answer",
}
CONFLICT_ORDER = ("charter", "other", "malformed", "coin")
CONFLICT_STYLE = {
    "charter": (CHARTER, None, "white"),
    "other": (OTHER, None, "white"),
    "malformed": (MALFORMED, None, "white"),
    "coin": (COIN, None, "white"),
}
CONFLICT_LABEL = {
    "charter": "chose Charter",
    "other": "chose another crew",
    "malformed": "malformed answer",
    "coin": "chose coin / cheapest",
}

BAR_HEIGHT = 0.76
ROW_PITCH = 0.92
ENDPOINT_GAP = 0.68


@dataclass(frozen=True)
class ProfileSpec:
    profile: str
    model: str
    dose: int

    @property
    def dose_label(self) -> str:
        return house.DOSE_LABEL[self.dose]


@dataclass(frozen=True)
class Row:
    endpoint: str
    arm: str
    unit: data.Unit | None
    y: float


def available_campaign_profiles(
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[ProfileSpec]:
    """Return scored cells from the declared model x dose campaign.

    Ablations and legacy repetitions are intentionally not inferred into this
    rectangle: two treatments can share model and dose, so PLAN is the source
    of truth for which treatment owns that coordinate.
    """
    available = {profile for profile, _arm in documents}
    return [
        ProfileSpec(profile, model, dose)
        for (model, dose), profile in house.PLAN.items()
        if profile in available
    ]


def endpoint_label(endpoint: str) -> str:
    if endpoint == "pre_aft":
        return "pre-AFT"
    family, step = data.endpoint_parts(endpoint)
    family_label = data.aft_family_label(family)
    if step is None:
        return family_label
    epoch = {256: "1 epoch", 512: "2 epochs"}.get(step, f"step {step}")
    return f"AFT: {family_label} · {epoch}"


def rows_for_profile(
    spec: ProfileSpec,
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Row]:
    docs = {
        arm: document
        for (profile, arm), document in documents.items()
        if profile == spec.profile
    }
    endpoints = sorted(
        set().union(*(data.endpoints_in(document) for document in docs.values())),
        key=data.endpoint_sort_key,
    )
    rows: list[Row] = []
    y = 0.0
    for endpoint_index, endpoint in enumerate(endpoints):
        if endpoint_index:
            y += ENDPOINT_GAP
        for arm in ARMS:
            document = docs.get(arm)
            unit = None
            if document is not None and endpoint in data.endpoints_in(document):
                unit = data.Unit(spec.profile, arm, endpoint, document)
            rows.append(Row(endpoint, arm, unit, y))
            y += ROW_PITCH
    return rows


def _positive_rates(
    unit: data.Unit, clause: str, surface: str,
) -> tuple[dict[str, float], int, int | None] | None:
    cell = data.cell_for(unit, clause, "agreement", surface)
    if cell is None:
        return None
    block = cell.get("agreement_runs", {})
    if not isinstance(block, dict):
        return None
    n_runs = data.positive_int(block.get("n"))
    rates = block.get("rates", {})
    if n_runs is None or not isinstance(rates, dict):
        return None
    values: dict[str, float] = {}
    for category in AGREEMENT_ORDER:
        value = rates.get(category, 0.0)
        if not isinstance(value, (int, float)) or value < 0:
            return None
        values[category] = float(value)
    total = sum(values.values())
    if total <= 0:
        return None
    return (
        {category: value / total for category, value in values.items()},
        n_runs,
        data.episode_n(cell),
    )


def _draw_missing(ax, y: float) -> None:
    ax.barh(
        y, 100, height=BAR_HEIGHT, color="#f2f2f0", edgecolor="#c8c8c3",
        linewidth=0.7, zorder=2,
    )
    ax.text(50, y, "data not available", ha="center", va="center", fontsize=6.5,
            color=MUTED, style="italic", zorder=5)


def _draw_stack(ax, y: float, shares: Mapping[str, float], order, styles) -> None:
    left = 0.0
    for category in order:
        width = 100 * shares.get(category, 0.0)
        if width <= 0:
            continue
        colour, hatch, edge = styles[category]
        ax.barh(
            y, width, left=left, height=BAR_HEIGHT, color=colour,
            edgecolor=edge, linewidth=1.15, hatch=hatch, zorder=3,
        )
        if width >= 4.5:
            text_colour = INK if category == "other" else "white"
            ax.text(left + width / 2, y, f"{width:.0f}", ha="center", va="center",
                    fontsize=7.2, color=text_colour, zorder=5)
        left += width


def _decorate_panel(ax, rows: Sequence[Row], title: str, xlabel: str) -> None:
    for index in range(1, len(rows)):
        if rows[index].endpoint != rows[index - 1].endpoint:
            boundary = (rows[index].y + rows[index - 1].y) / 2
            ax.axhline(boundary, color=GRID, linewidth=0.9, zorder=0)

    ticks = [row.y for row in rows]
    labels = [ARM_LABEL[row.arm] for row in rows]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.8)
    # A shared-y secondary axis can suppress its tick-label artists entirely.
    # Colour whichever labels this axis actually owns; the left axis owns all
    # of them in the normal two-panel render.
    for label, row in zip(ax.get_yticklabels(), rows):
        label.set_color(house.ARM_COLOR[row.arm])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(rows[-1].y + 0.75, rows[0].y - 0.75)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"), fontsize=7.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED)


def _add_endpoint_labels(ax, rows: Sequence[Row]) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.endpoint, []).append(row.y)
    transform = ax.get_yaxis_transform()
    for endpoint, ys in grouped.items():
        y0, y1 = min(ys) - BAR_HEIGHT / 2, max(ys) + BAR_HEIGHT / 2
        x = -0.33
        ax.plot([x, x], [y0, y1], transform=transform, clip_on=False,
                color=MUTED, linewidth=0.9)
        ax.plot([x, x + 0.012], [y0, y0], transform=transform, clip_on=False,
                color=MUTED, linewidth=0.9)
        ax.plot([x, x + 0.012], [y1, y1], transform=transform, clip_on=False,
                color=MUTED, linewidth=0.9)
        ax.text(x - 0.012, (y0 + y1) / 2, endpoint_label(endpoint),
                transform=transform, ha="right", va="center", fontsize=7.2,
                color=MUTED, clip_on=False)


def _n_text(ns: Sequence[tuple[int, int | None]]) -> str:
    if not ns:
        return "no scored observations"
    runs = data.n_range([item[0] for item in ns])
    episodes = data.n_range([item[1] for item in ns])
    return f"n={runs} runs / {episodes} episodes per bar"


def render_slice(
    spec: ProfileSpec,
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    surface: str,
    clause: str,
    output: Path,
) -> list[Path]:
    rows = rows_for_profile(spec, documents)
    if not rows:
        return []
    height = max(8.2, 2.5 + 0.31 * len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(15.4, height), sharey=True)
    agreement_ns: list[tuple[int, int | None]] = []
    conflict_ns: list[tuple[int, int | None]] = []

    for row in rows:
        agreement = None if row.unit is None else _positive_rates(row.unit, clause, surface)
        if agreement is None:
            _draw_missing(axes[0], row.y)
        else:
            shares, n_runs, n_episodes = agreement
            agreement_ns.append((n_runs, n_episodes))
            _draw_stack(axes[0], row.y, shares, AGREEMENT_ORDER, AGREEMENT_STYLE)

        conflict = None if row.unit is None else data.conflict_reading(row.unit, clause, surface)
        if conflict is None:
            _draw_missing(axes[1], row.y)
        else:
            conflict_ns.append((conflict.n_runs, conflict.n_episodes))
            _draw_stack(
                axes[1], row.y, conflict.shares, CONFLICT_ORDER,
                CONFLICT_STYLE,
            )

    _decorate_panel(
        axes[0], rows,
        "Ambiguous (agreement episodes)",
        "share of agreement-eval runs (%)",
    )
    _decorate_panel(
        axes[1], rows,
        "Diagnostic (conflict episodes)",
        "share of conflict-eval runs (%)",
    )
    _add_endpoint_labels(axes[0], rows)

    model_label = house.MODEL_LABEL[spec.model]
    fig.suptitle(
        f"Figure 0 — {model_label} · {spec.dose_label} · "
        f"{CLAUSE_LABEL[clause]} × {SURFACE_LABEL[surface]}",
        x=0.025, y=0.99, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    agreement_handles = []
    for category in AGREEMENT_ORDER:
        colour, hatch, edge = AGREEMENT_STYLE[category]
        agreement_handles.append(Patch(facecolor=colour, edgecolor=edge,
                                       label=AGREEMENT_LABEL[category]))
    conflict_handles = []
    for category in CONFLICT_ORDER:
        colour, hatch, edge = CONFLICT_STYLE[category]
        conflict_handles.append(Patch(facecolor=colour, edgecolor=edge,
                                      label=CONFLICT_LABEL[category]))
    axes[0].legend(handles=agreement_handles, loc="upper center", ncol=3,
                   frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, -0.075))
    axes[1].legend(handles=conflict_handles, loc="upper center", ncol=4,
                   frameon=False, fontsize=7.0, bbox_to_anchor=(0.48, -0.075))
    fig.text(
        0.985, 0.012,
        f"{CLAUSE_LABEL[clause]}, {SURFACE_LABEL[surface]}; "
        f"agreement {_n_text(agreement_ns)}; conflict {_n_text(conflict_ns)}. "
        f"{house.CAVEAT}.",
        ha="right", va="bottom", color=MUTED, fontsize=8.0,
    )
    fig.subplots_adjust(left=0.265, right=0.985, top=0.935, bottom=0.11, wspace=0.08)

    stem = "__".join((
        SURFACE_STEM[surface],
        CLAUSE_STEM[clause],
        spec.model.replace("_", "-"),
        spec.dose_label.lower(),
    ))
    return data.save_figure(fig, stem, output)


def _dose(value: str) -> int:
    normalised = value.strip().lower()
    by_label = {label.lower(): dose for dose, label in house.DOSE_LABEL.items()}
    if normalised not in by_label:
        raise argparse.ArgumentTypeError(
            f"expected one of {', '.join(sorted(by_label, key=by_label.get))}"
        )
    return by_label[normalised]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scored", type=Path, default=SCORED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--model", action="append", choices=house.MODELS,
                        help="repeat to select model families; default: all")
    parser.add_argument("--dose", action="append", type=_dose,
                        help="repeat to select budgets (for example 5m); default: all")
    parser.add_argument("--surface", action="append", choices=SURFACES,
                        help="repeat to select surfaces; default: all")
    parser.add_argument("--clause", action="append", choices=CLAUSES,
                        help="repeat to select clause splits; default: all")
    args = parser.parse_args()

    documents = data.load_documents(args.scored)
    profiles = available_campaign_profiles(documents)
    if args.model:
        profiles = [profile for profile in profiles if profile.model in args.model]
    if args.dose:
        profiles = [profile for profile in profiles if profile.dose in args.dose]
    surfaces = args.surface or list(SURFACES)
    clauses = args.clause or list(CLAUSES)

    written: list[Path] = []
    for surface in surfaces:
        for clause in clauses:
            for profile in profiles:
                written.extend(render_slice(
                    profile, documents, surface=surface, clause=clause,
                    output=args.out,
                ))
    for path in written:
        print(f"wrote {path}")
    print(
        f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files) from "
        f"{len(profiles)} available campaign cells."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
