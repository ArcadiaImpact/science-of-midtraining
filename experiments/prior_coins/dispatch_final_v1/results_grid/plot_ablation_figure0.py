"""Render Figure-0 galleries for the three dispatch ablations.

The response-template and elicitation galleries compare every two-epoch
treatment with its matched standard/headline bar.  The no-examples gallery
retains its exhaustive matched comparison.  All galleries span each of the six
template x clause surfaces.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import plot_figure0_slices as figure0  # noqa: E402
import plot_stacked as data  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import (  # noqa: E402
    launch as diverse_launch,
)

SCORED = HERE / "scored" / "ablations"
OUTPUT = HERE / "figures" / "ablations"

ABLATIONS = ("diverse_templates", "elicitation", "no_examples_midtrain")
TITLE = {
    "diverse_templates": "Diverse response templates",
    "elicitation": "Elicitation ablation",
    "no_examples_midtrain": "No-examples midtraining ablation",
}

ROW_PITCH = 0.92
SECTION_GAP = 0.72


@dataclass(frozen=True)
class PlotRow:
    section: str
    label: str
    arm: str
    unit: data.Unit
    y: float


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a JSON mapping")
    return document


def _diverse_unit(
    scored: Mapping[str, Any], arm: str, endpoint: str,
) -> data.Unit:
    result = scored.get("arms", {}).get(arm, {}).get(endpoint)
    if not isinstance(result, dict):
        raise KeyError(f"missing diverse-response score: {arm}/{endpoint}")
    document = {"result": {endpoint: result}}
    return data.Unit("gemma3_12b_50m_divresp", arm, endpoint, document)


def _headline_unit(
    scored: Mapping[str, Any], arm: str, endpoint: str,
) -> data.Unit:
    document = scored.get("arms", {}).get(arm)
    if not isinstance(document, dict) or endpoint not in data.endpoints_in(document):
        raise KeyError(f"missing headline score: {arm}/{endpoint}")
    return data.Unit("gemma3_12b_50m_4ep", arm, endpoint, document)


def _place(specs: Sequence[tuple[str, str, str, data.Unit]]) -> list[PlotRow]:
    rows: list[PlotRow] = []
    y = 0.0
    previous_section: str | None = None
    for section, label, arm, unit in specs:
        if previous_section is not None and section != previous_section:
            y += SECTION_GAP
        rows.append(PlotRow(section, label, arm, unit, y))
        y += ROW_PITCH
        previous_section = section
    return rows


def diverse_template_rows(
    scored: Mapping[str, Any], headline: Mapping[str, Any],
) -> list[PlotRow]:
    specs: list[tuple[str, str, str, data.Unit]] = []
    for arm in figure0.ARMS:
        specs.append((
            "pre-AFT", figure0.ARM_LABEL[arm], arm,
            _headline_unit(headline, arm, "pre_aft"),
        ))

    treatments = (
        ("agreement", "AFT agreement"),
        ("mixed_charter", "AFT 2% Charter-labelled"),
        ("mixed_coin", "AFT 2% coin-labelled"),
        ("charter_only", "AFT 100% Charter"),
    )
    for suffix, section in treatments:
        for arm in figure0.ARMS:
            specs.append((
                section,
                f"{figure0.ARM_LABEL[arm]} · headline",
                arm,
                _headline_unit(headline, arm, f"{suffix}-step512"),
            ))
            specs.append((
                section,
                f"{figure0.ARM_LABEL[arm]} · diverse templates",
                arm,
                _diverse_unit(scored, arm, f"natural_{arm}_{suffix}-step512"),
            ))
    return _place(specs)


ELICITATION_SECTION = {
    "e1": "E1 · identity only\n100% agreement",
    "e2": "E2 · aligned motive\n100% agreement",
    "e3": "E3 · ambiguous motive\nbalanced 2% outcomes",
    "e4": "E4 · chosen motive\n2% outcome",
    "e5": "E5 · opposite motive\n2% outcome",
}


def _elicitation_condition(cell_name: str) -> str | None:
    if cell_name.startswith(("e1_", "e2_")):
        return None
    mixed = re.search(r"mixed_(charter|coin|balanced)", cell_name)
    if mixed is None:
        return None
    direction = mixed.group(1)
    return "balanced 2%" if direction == "balanced" else f"2% {direction}"


def _elicitation_row_label(cell_name: str, arm: str, variant: str) -> str:
    bits = [figure0.ARM_LABEL[arm]]
    condition = _elicitation_condition(cell_name)
    if condition:
        bits.append(condition)
    if cell_name.startswith("e5_control_"):
        motive = re.search(r"_(charter|coin)_motive$", cell_name)
        if motive:
            bits.append(f"{motive.group(1)} motive")
    bits.append(variant)
    return " · ".join(bits)


def _headline_family(cell_name: str) -> str:
    if cell_name.startswith(("e1_", "e2_")):
        return "agreement"
    match = re.search(r"mixed_(charter|coin)", cell_name)
    if match is None:
        raise ValueError(f"no single headline match for {cell_name}")
    return f"mixed_{match.group(1)}"


def elicitation_rows(
    scored: Mapping[str, Any], headline: Mapping[str, Any],
) -> list[PlotRow]:
    _body, experiment = diverse_launch.load()
    cells = [cell for cell in experiment.cells if cell.name.startswith("e")]
    specs: list[tuple[str, str, str, data.Unit]] = []
    for arm in figure0.ARMS:
        specs.append((
            "pre-AFT", figure0.ARM_LABEL[arm], arm,
            _headline_unit(headline, arm, "pre_aft"),
        ))
    for family in ELICITATION_SECTION:
        section = ELICITATION_SECTION[family]
        family_cells = [cell for cell in cells if cell.name.startswith(f"{family}_")]
        for cell in family_cells:
            if family == "e3":
                # The headline campaign has no direction-balanced 2% cell.
                # Show both directional neighbours instead of inventing an
                # average that was never trained.
                for direction in ("charter", "coin"):
                    label = (
                        f"{figure0.ARM_LABEL[cell.parent_arm]} · "
                        f"2% {direction} · headline"
                    )
                    specs.append((
                        section, label, cell.parent_arm,
                        _headline_unit(
                            headline, cell.parent_arm,
                            f"mixed_{direction}-step512",
                        ),
                    ))
            else:
                family_endpoint = _headline_family(cell.name)
                specs.append((
                    section,
                    _elicitation_row_label(
                        cell.name, cell.parent_arm, "headline",
                    ),
                    cell.parent_arm,
                    _headline_unit(
                        headline, cell.parent_arm, f"{family_endpoint}-step512",
                    ),
                ))
            specs.append((
                section,
                _elicitation_row_label(cell.name, cell.parent_arm, "elicitation"),
                cell.parent_arm,
                _diverse_unit(scored, cell.parent_arm, f"{cell.name}-step512"),
            ))
    return _place(specs)


def no_examples_rows(scored: Mapping[str, Any]) -> list[PlotRow]:
    variants = scored.get("variants", {})
    standard = variants.get("standard_examples", {})
    no_examples = variants.get("no_examples", {})
    available = [
        document
        for arms in (standard, no_examples)
        for document in arms.values()
        if isinstance(document, dict)
    ]
    endpoints = sorted(
        set().union(*(data.endpoints_in(document) for document in available)),
        key=data.endpoint_sort_key,
    )
    row_order = (
        ("standard_examples", "charter", "charter prior · examples"),
        ("no_examples", "charter", "charter prior · no examples"),
        ("standard_examples", "control", "control · shared anchor"),
        ("no_examples", "coin", "coin prior · no examples"),
        ("standard_examples", "coin", "coin prior · examples"),
    )
    specs: list[tuple[str, str, str, data.Unit]] = []
    for endpoint in endpoints:
        section = figure0.endpoint_label(endpoint)
        for variant, arm, label in row_order:
            document = variants[variant][arm]
            if endpoint not in data.endpoints_in(document):
                continue
            specs.append((
                section, label, arm,
                data.Unit(variant, arm, endpoint, document),
            ))
    return _place(specs)


def _decorate_panel(ax, rows: Sequence[PlotRow], title: str, xlabel: str) -> None:
    for index in range(1, len(rows)):
        if rows[index].section != rows[index - 1].section:
            boundary = (rows[index].y + rows[index - 1].y) / 2
            ax.axhline(boundary, color=figure0.GRID, linewidth=0.9, zorder=0)
    ax.set_yticks([row.y for row in rows])
    ax.set_yticklabels([row.label for row in rows], fontsize=6.7)
    for label, row in zip(ax.get_yticklabels(), rows):
        label.set_color(figure0.house.ARM_COLOR[row.arm])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(rows[-1].y + 0.75, rows[0].y - 0.75)
    ax.set_xticks((0, 25, 50, 75, 100))
    ax.set_xticklabels(("0%", "25%", "50%", "75%", "100%"), fontsize=7.5)
    ax.grid(axis="x", color=figure0.GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=figure0.INK, pad=12)
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
        y0, y1 = min(ys) - figure0.BAR_HEIGHT / 2, max(ys) + figure0.BAR_HEIGHT / 2
        x = -0.47
        ax.plot([x, x], [y0, y1], transform=transform, clip_on=False,
                color=figure0.MUTED, linewidth=0.9)
        ax.plot([x, x + 0.012], [y0, y0], transform=transform, clip_on=False,
                color=figure0.MUTED, linewidth=0.9)
        ax.plot([x, x + 0.012], [y1, y1], transform=transform, clip_on=False,
                color=figure0.MUTED, linewidth=0.9)
        ax.text(x - 0.012, (y0 + y1) / 2, section, transform=transform,
                ha="right", va="center", fontsize=7.0, color=figure0.MUTED,
                clip_on=False, linespacing=1.2)


def _legend(ax, order, styles, labels, *, ncol: int, fontsize: float) -> None:
    handles = [
        Patch(facecolor=styles[item][0], edgecolor=styles[item][2], label=labels[item])
        for item in order
    ]
    ax.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False,
              fontsize=fontsize, bbox_to_anchor=(0.5, -0.065))


def render(
    ablation: str, rows: Sequence[PlotRow], *, surface: str, clause: str,
    output: Path,
) -> list[Path]:
    height = max(8.5, 2.7 + 0.29 * len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(17.2, height), sharey=True)
    agreement_ns: list[tuple[int, int | None]] = []
    conflict_ns: list[tuple[int, int | None]] = []
    for row in rows:
        agreement = figure0._positive_rates(row.unit, clause, surface)
        if agreement is None:
            figure0._draw_missing(axes[0], row.y)
        else:
            shares, n_runs, n_episodes = agreement
            agreement_ns.append((n_runs, n_episodes))
            figure0._draw_stack(
                axes[0], row.y, shares,
                figure0.AGREEMENT_ORDER, figure0.AGREEMENT_STYLE,
            )
        conflict = data.conflict_reading(row.unit, clause, surface)
        if conflict is None:
            figure0._draw_missing(axes[1], row.y)
        else:
            conflict_ns.append((conflict.n_runs, conflict.n_episodes))
            figure0._draw_stack(
                axes[1], row.y, conflict.shares,
                figure0.CONFLICT_ORDER, figure0.CONFLICT_STYLE,
            )

    _decorate_panel(
        axes[0], rows, "Ambiguous (agreement episodes)",
        "share of agreement-eval runs (%)",
    )
    _decorate_panel(
        axes[1], rows, "Diagnostic (conflict episodes)",
        "share of conflict-eval runs (%)",
    )
    _add_section_labels(axes[0], rows)
    fig.suptitle(
        f"Figure 0 — {TITLE[ablation]} · Gemma 3 12B · 50M · "
        f"{figure0.CLAUSE_LABEL[clause]} × {figure0.SURFACE_LABEL[surface]}",
        x=0.025, y=0.992, ha="left", fontsize=14, fontweight="bold",
        color=figure0.INK,
    )
    _legend(
        axes[0], figure0.AGREEMENT_ORDER, figure0.AGREEMENT_STYLE,
        figure0.AGREEMENT_LABEL, ncol=3, fontsize=8.2,
    )
    _legend(
        axes[1], figure0.CONFLICT_ORDER, figure0.CONFLICT_STYLE,
        figure0.CONFLICT_LABEL, ncol=4, fontsize=7.0,
    )
    if ablation in ("diverse_templates", "elicitation"):
        scorer_note = (
            "headline bars: exact scorer; ablation bars: semantic parser "
            "(parser failures are malformed)"
        )
        if ablation == "elicitation":
            scorer_note += "; E3 is bracketed by both directional headline cells"
    else:
        scorer_note = "standard exact dispatch scorer"
    footnote = (
        f"{figure0.CLAUSE_LABEL[clause]}, {figure0.SURFACE_LABEL[surface]}; "
        f"agreement {figure0._n_text(agreement_ns)}; "
        f"conflict {figure0._n_text(conflict_ns)}; {scorer_note}. "
        f"{figure0.house.CAVEAT}."
    )
    fig.text(
        0.99, 0.01, textwrap.fill(footnote, width=190),
        ha="right", va="bottom", color=figure0.MUTED, fontsize=7.5,
        linespacing=1.25,
    )
    # Fixed physical margins keep the title and two legends clear regardless
    # of whether an inventory has 27 rows or 45.
    fig.subplots_adjust(
        left=0.335, right=0.99,
        top=1.0 - 0.72 / height, bottom=1.8 / height,
        wspace=0.08,
    )
    stem = "__".join((
        figure0.SURFACE_STEM[surface], figure0.CLAUSE_STEM[clause],
    ))
    return data.save_figure(fig, stem, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scored", type=Path, default=SCORED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--ablation", action="append", choices=ABLATIONS)
    parser.add_argument("--surface", action="append", choices=figure0.SURFACES)
    parser.add_argument("--clause", action="append", choices=figure0.CLAUSES)
    args = parser.parse_args(argv)

    selected = args.ablation or list(ABLATIONS)
    surfaces = args.surface or list(figure0.SURFACES)
    clauses = args.clause or list(figure0.CLAUSES)
    diverse = None
    headline = None
    no_examples = None
    if any(item in selected for item in ("diverse_templates", "elicitation")):
        diverse = _load(args.scored / "diverse_response.json")
        headline = _load(args.scored / "headline.json")
    if "no_examples_midtrain" in selected:
        no_examples = _load(args.scored / "no_examples.json")

    written: list[Path] = []
    for ablation in selected:
        if ablation == "diverse_templates":
            rows = diverse_template_rows(diverse, headline)
        elif ablation == "elicitation":
            rows = elicitation_rows(diverse, headline)
        else:
            rows = no_examples_rows(no_examples)
        for surface in surfaces:
            for clause in clauses:
                written.extend(render(
                    ablation, rows, surface=surface, clause=clause,
                    output=args.out / ablation,
                ))
    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
