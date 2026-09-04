"""Figure-0 comparisons across model size and presented-token budget.

This complements ``plot_figure0_slices.py``.  Instead of fixing one campaign
cell per figure, it nests bars in one of two deliberate orders:

* model-size scaling: AFT treatment -> midtraining treatment -> model size,
  while holding presented-token budget fixed;
* token-budget scaling: AFT treatment -> midtraining treatment -> token budget,
  while holding model size fixed.

Each AFT treatment uses its converged step-512 endpoint. ``none`` is the
pre-AFT endpoint.  The two panels and outcome colors are the classic Figure-0
agreement/diagnostic visual contract from ``plot_figure0_slices.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plot_grid as house  # noqa: E402
import plot_stacked as data  # noqa: E402
import plot_figure0_slices as figure0  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORED = HERE / "scored"
MODEL_OUTPUT = HERE / "figures" / "figure0_scaling_model_size"
TOKEN_OUTPUT = HERE / "figures" / "figure0_scaling_token_budget"

# Outer grouping, display label, scored endpoint. The percentage names the
# directional rows in the AFT training mixture, matching the campaign docs.
TREATMENTS: tuple[tuple[str, str, str], ...] = (
    ("none", "none (pre-AFT)", "pre_aft"),
    ("agreement", "agreement only", "agreement-step512"),
    ("mixed_charter", "2% Charter-labelled", "mixed_charter-step512"),
    ("mixed_coin", "2% coin-labelled", "mixed_coin-step512"),
    ("charter_only", "100% Charter-labelled", "charter_only-step512"),
)

SCALE_PITCH = 0.82
ARM_GAP = 0.35
TREATMENT_GAP = 0.82


@dataclass(frozen=True)
class ScalingRow:
    treatment: str
    treatment_label: str
    endpoint: str
    arm: str
    scale_key: str
    scale_label: str
    unit: data.Unit | None
    y: float


@dataclass(frozen=True)
class Comparison:
    """One fixed-axis comparison and the profiles on its varying axis."""

    fixed_key: str
    fixed_label: str
    profiles: tuple[figure0.ProfileSpec, ...]


def _campaign_profiles(
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[figure0.ProfileSpec]:
    return figure0.available_campaign_profiles(documents)


def model_comparisons(
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Comparison]:
    """Shared-dose comparisons with at least two available model sizes."""
    profiles = _campaign_profiles(documents)
    comparisons: list[Comparison] = []
    for dose in house.DOSES:
        members = tuple(profile for profile in profiles if profile.dose == dose)
        if len({profile.model for profile in members}) < 2:
            continue
        comparisons.append(Comparison(
            house.DOSE_LABEL[dose].lower(),
            f"{house.dose_axis_label(dose)} presented tokens",
            members,
        ))
    return comparisons


def token_comparisons(
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[Comparison]:
    """Within-model comparisons with at least two available token budgets."""
    profiles = _campaign_profiles(documents)
    comparisons: list[Comparison] = []
    for model in house.MODELS:
        members = tuple(profile for profile in profiles if profile.model == model)
        if len({profile.dose for profile in members}) < 2:
            continue
        comparisons.append(Comparison(
            model.replace("_", "-"),
            house.MODEL_LABEL[model],
            members,
        ))
    return comparisons


def _rows(
    comparison: Comparison,
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
    scale_key: Callable[[figure0.ProfileSpec], str],
    scale_label: Callable[[figure0.ProfileSpec], str],
) -> list[ScalingRow]:
    rows: list[ScalingRow] = []
    y = 0.0
    for treatment_index, (treatment, treatment_label, endpoint) in enumerate(TREATMENTS):
        if treatment_index:
            y += TREATMENT_GAP
        for arm_index, arm in enumerate(figure0.ARMS):
            if arm_index:
                y += ARM_GAP
            for profile in comparison.profiles:
                document = documents.get((profile.profile, arm))
                unit = None
                if document is not None and endpoint in data.endpoints_in(document):
                    unit = data.Unit(profile.profile, arm, endpoint, document)
                rows.append(ScalingRow(
                    treatment=treatment,
                    treatment_label=treatment_label,
                    endpoint=endpoint,
                    arm=arm,
                    scale_key=scale_key(profile),
                    scale_label=scale_label(profile),
                    unit=unit,
                    y=y,
                ))
                y += SCALE_PITCH
    return rows


def _bracket(
    ax,
    *,
    y0: float,
    y1: float,
    x: float,
    label: str,
    label_x: float,
    color: str,
    linewidth: float,
    fontsize: float,
    weight: str = "normal",
) -> None:
    transform = ax.get_yaxis_transform()
    cap = 0.012
    ax.plot([x, x], [y0, y1], transform=transform, clip_on=False,
            color=color, linewidth=linewidth)
    ax.plot([x, x + cap], [y0, y0], transform=transform, clip_on=False,
            color=color, linewidth=linewidth)
    ax.plot([x, x + cap], [y1, y1], transform=transform, clip_on=False,
            color=color, linewidth=linewidth)
    ax.text(label_x, (y0 + y1) / 2, label, transform=transform,
            ha="right", va="center", color=color, fontsize=fontsize,
            fontweight=weight, clip_on=False)


def _decorate_groups(ax, rows: Sequence[ScalingRow]) -> None:
    """Draw treatment (outer) and midtraining-arm (inner) hierarchy."""
    treatment_groups: dict[str, list[ScalingRow]] = {}
    arm_groups: dict[tuple[str, str], list[ScalingRow]] = {}
    for row in rows:
        treatment_groups.setdefault(row.treatment, []).append(row)
        arm_groups.setdefault((row.treatment, row.arm), []).append(row)

    for group in treatment_groups.values():
        y0 = group[0].y - figure0.BAR_HEIGHT / 2
        y1 = group[-1].y + figure0.BAR_HEIGHT / 2
        _bracket(
            ax, y0=y0, y1=y1, x=-0.37, label=group[0].treatment_label,
            label_x=-0.39, color=figure0.INK, linewidth=1.15,
            fontsize=7.8, weight="bold",
        )

    for group in arm_groups.values():
        y0 = group[0].y - figure0.BAR_HEIGHT / 2
        y1 = group[-1].y + figure0.BAR_HEIGHT / 2
        arm = group[0].arm
        _bracket(
            ax, y0=y0, y1=y1, x=-0.16, label=figure0.ARM_LABEL[arm],
            label_x=-0.175, color=house.ARM_COLOR[arm], linewidth=0.85,
            fontsize=7.2,
        )


def _decorate_panel(
    ax,
    rows: Sequence[ScalingRow],
    *,
    title: str,
    xlabel: str,
) -> None:
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        if current.treatment != previous.treatment:
            boundary = (previous.y + current.y) / 2
            ax.axhline(boundary, color=figure0.GRID, linewidth=1.3, zorder=0)
        elif current.arm != previous.arm:
            boundary = (previous.y + current.y) / 2
            ax.axhline(boundary, color=figure0.GRID, linewidth=0.7,
                       linestyle=(0, (4, 3)), zorder=0)

    ax.set_yticks([row.y for row in rows])
    ax.set_yticklabels([row.scale_label for row in rows], fontsize=7.0,
                       color=figure0.MUTED)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(rows[-1].y + 0.7, rows[0].y - 0.7)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"), fontsize=7.5)
    ax.grid(axis="x", color=figure0.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, fontweight="bold",
                 color=figure0.INK, pad=12)
    ax.set_xlabel(xlabel, fontsize=9.5, color=figure0.INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(figure0.GRID)
    ax.tick_params(axis="x", colors=figure0.MUTED)


def _legend(ax, *, conflict: bool) -> None:
    if conflict:
        order = figure0.CONFLICT_ORDER
        styles = figure0.CONFLICT_STYLE
        labels = figure0.CONFLICT_LABEL
        fontsize = 7.0
    else:
        order = figure0.AGREEMENT_ORDER
        styles = figure0.AGREEMENT_STYLE
        labels = figure0.AGREEMENT_LABEL
        fontsize = 8.2
    handles = [
        Patch(facecolor=styles[category][0], edgecolor="white",
              label=labels[category])
        for category in order
    ]
    ax.legend(handles=handles, loc="upper center", ncol=len(order),
              frameon=False, fontsize=fontsize, bbox_to_anchor=(0.48, -0.06))


def render(
    comparison: Comparison,
    documents: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    dimension: str,
    surface: str,
    clause: str,
    output: Path,
) -> list[Path]:
    if dimension == "model":
        rows = _rows(
            comparison,
            documents,
            scale_key=lambda profile: profile.model,
            scale_label=lambda profile: (
                house.MODEL_LABEL[profile.model]
                + ("*" if profile.profile == house.LEGACY_GLM_PROFILE else "")
            ),
        )
        dimension_label = "scaling model size"
        varying_label = "model size"
    else:
        rows = _rows(
            comparison,
            documents,
            scale_key=lambda profile: profile.dose_label.lower(),
            scale_label=lambda profile: profile.dose_label,
        )
        dimension_label = "scaling token budget"
        varying_label = "presented-token budget"

    height = max(11.0, 2.4 + 0.29 * len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(16.4, height), sharey=True)
    agreement_ns: list[tuple[int, int | None]] = []
    conflict_ns: list[tuple[int, int | None]] = []
    for row in rows:
        agreement = (
            None if row.unit is None
            else figure0._positive_rates(row.unit, clause, surface)
        )
        if agreement is None:
            figure0._draw_missing(axes[0], row.y)
        else:
            shares, n_runs, n_episodes = agreement
            agreement_ns.append((n_runs, n_episodes))
            figure0._draw_stack(
                axes[0], row.y, shares,
                figure0.AGREEMENT_ORDER, figure0.AGREEMENT_STYLE,
            )

        conflict = (
            None if row.unit is None
            else data.conflict_reading(row.unit, clause, surface)
        )
        if conflict is None:
            figure0._draw_missing(axes[1], row.y)
        else:
            conflict_ns.append((conflict.n_runs, conflict.n_episodes))
            figure0._draw_stack(
                axes[1], row.y, conflict.shares,
                figure0.CONFLICT_ORDER, figure0.CONFLICT_STYLE,
            )

    _decorate_panel(
        axes[0], rows, title="Ambiguous (agreement episodes)",
        xlabel="share of agreement-eval runs (%)",
    )
    _decorate_panel(
        axes[1], rows, title="Diagnostic (conflict episodes)",
        xlabel="share of conflict-eval runs (%)",
    )
    _decorate_groups(axes[0], rows)
    _legend(axes[0], conflict=False)
    _legend(axes[1], conflict=True)

    fig.suptitle(
        f"Figure 0 — {dimension_label} · {comparison.fixed_label} · "
        f"{figure0.CLAUSE_LABEL[clause]} × {figure0.SURFACE_LABEL[surface]}",
        x=0.02, y=0.993, ha="left", fontsize=14, fontweight="bold",
        color=figure0.INK,
    )
    legacy_note = (
        f"\n{house.LEGACY_GLM_NOTE}"
        if any(p.profile == house.LEGACY_GLM_PROFILE for p in comparison.profiles)
        else ""
    )
    fig.text(
        0.985, 0.01,
        f"Rows: AFT treatment → midtraining treatment → {varying_label}. "
        f"Agreement {figure0._n_text(agreement_ns)}; conflict "
        f"{figure0._n_text(conflict_ns)}. {house.CAVEAT}.{legacy_note}",
        ha="right", va="bottom", fontsize=7.8, color=figure0.MUTED,
    )
    fig.subplots_adjust(left=0.31, right=0.985, top=0.955,
                        bottom=0.11 if legacy_note else 0.095, wspace=0.08)
    stem = "__".join((
        figure0.SURFACE_STEM[surface],
        figure0.CLAUSE_STEM[clause],
        comparison.fixed_key,
    ))
    return data.save_figure(fig, stem, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scored", type=Path, default=SCORED)
    parser.add_argument("--dimension", choices=("model", "token"),
                        action="append", help="repeatable; default: both")
    parser.add_argument("--surface", choices=figure0.SURFACES, action="append",
                        help="repeatable; default: all")
    parser.add_argument("--clause", choices=figure0.CLAUSES, action="append",
                        help="repeatable; default: all")
    parser.add_argument("--model-out", type=Path, default=MODEL_OUTPUT)
    parser.add_argument("--token-out", type=Path, default=TOKEN_OUTPUT)
    args = parser.parse_args()

    documents = data.load_documents(args.scored)
    dimensions = args.dimension or ["model", "token"]
    surfaces = args.surface or list(figure0.SURFACES)
    clauses = args.clause or list(figure0.CLAUSES)
    written: list[Path] = []
    counts: dict[str, int] = {}
    for dimension in dimensions:
        comparisons = (
            model_comparisons(documents)
            if dimension == "model"
            else token_comparisons(documents)
        )
        output = args.model_out if dimension == "model" else args.token_out
        before = len(written)
        for surface in surfaces:
            for clause in clauses:
                for comparison in comparisons:
                    written.extend(render(
                        comparison,
                        documents,
                        dimension=dimension,
                        surface=surface,
                        clause=clause,
                        output=output,
                    ))
        counts[dimension] = (len(written) - before) // 2

    for path in written:
        print(f"wrote {path}")
    summary = ", ".join(f"{dimension}={count}" for dimension, count in counts.items())
    print(f"\n{len(written) // 2} figures ({len(written)} files): {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
