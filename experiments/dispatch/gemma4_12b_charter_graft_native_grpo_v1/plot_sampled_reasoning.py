"""Render Figure-0-style plots for the 201-presentation reasoning screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = HERE.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import score_factorised as sf  # noqa: E402
from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.analyze_results import (  # noqa: E402
    draw_stacked_rows,
)
from plot_dispatch_v4_aft import GRID, INK, MUTED  # noqa: E402
from plot_wave_v1_summary import save_figure  # noqa: E402


CELLS = ("public_it-reasoning_grpo", "charter_graft_it-reasoning_grpo")
STEPS = (0, 64, 128)
PRESENTATIONS = ("canonical", "trained", "heldout")
CELL_LABEL = {
    "public_it-reasoning_grpo": "public instruct · native-reasoning GRPO",
    "charter_graft_it-reasoning_grpo": "Charter graft · native-reasoning GRPO",
}
PRESENTATION_LABEL = {
    "canonical": "canonical presentation",
    "trained": "seen presentation templates (90)",
    "heldout": "held-out presentation templates (10)",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_grid(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for cell in CELLS:
        result[cell] = {}
        for step in STEPS:
            endpoint = root / "cells" / cell / f"checkpoint-{step}"
            marker = json.loads((endpoint / "EVAL_DONE.json").read_text())
            metrics = json.loads((endpoint / "metrics.json").read_text())
            if (
                marker.get("status") != "complete"
                or marker.get("cell") != cell
                or int(marker.get("checkpoint_step", -1)) != step
                or int(marker.get("presentations", -1)) != 201
                or set(metrics.get("by_mode", {})) != set(PRESENTATIONS)
            ):
                raise RuntimeError(f"invalid sampled endpoint {endpoint}")
            result[cell][step] = metrics
    return result


def agreement_accuracy(metrics: Mapping[str, Any]) -> float:
    return float(metrics["agreement_runs"]["rates"].get(sf.SHARED, 0.0))


def conflict_rate(metrics: Mapping[str, Any], verdict: str) -> float:
    return float(metrics["conflict_runs"]["rates"].get(verdict, 0.0))


def compile_summary(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for cell in CELLS:
        for step in STEPS:
            for presentation in PRESENTATIONS:
                metrics = grid[cell][step]["by_mode"][presentation]
                diagnostics = grid[cell][step]["native_format_by_mode"][presentation]
                endpoints[f"{cell}|{step}|{presentation}"] = {
                    "agreement_accuracy": agreement_accuracy(metrics),
                    "agreement_n": metrics["agreement_runs"]["n"],
                    "conflict_charter_rate": conflict_rate(metrics, sf.CHARTER),
                    "conflict_coin_rate": conflict_rate(metrics, sf.COIN),
                    "conflict_other_rate": conflict_rate(metrics, sf.OTHER),
                    "conflict_malformed_rate": conflict_rate(metrics, sf.MALFORMED),
                    "conflict_n": metrics["conflict_runs"]["n"],
                    "consistency": metrics["consistency"]["rate"],
                    **diagnostics,
                }
    for step in STEPS:
        for presentation in PRESENTATIONS:
            public = grid[CELLS[0]][step]["by_mode"][presentation]
            graft = grid[CELLS[1]][step]["by_mode"][presentation]
            graft_charter = conflict_rate(graft, sf.CHARTER) - conflict_rate(
                public, sf.CHARTER
            )
            public_coin = conflict_rate(public, sf.COIN) - conflict_rate(graft, sf.COIN)
            contrasts[f"{step}|{presentation}"] = {
                "graft_minus_public_charter_rate": round(graft_charter, 4),
                "public_minus_graft_coin_rate": round(public_coin, 4),
                "directional_separation": round(graft_charter + public_coin, 4),
                "graft_minus_public_agreement_accuracy": round(
                    agreement_accuracy(graft) - agreement_accuracy(public), 4
                ),
            }
    return {
        "schema_version": 1,
        "sample_warning": "directional screen: 201 presentations per endpoint",
        "checkpoints": list(STEPS),
        "presentation_modes": list(PRESENTATIONS),
        "endpoints": endpoints,
        "contrasts": contrasts,
    }


def figure_outcomes(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    step: int,
    presentation: str,
    output: Path,
) -> None:
    rows = [
        (CELL_LABEL[cell], grid[cell][step]["by_mode"][presentation]) for cell in CELLS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 4.7), sharey=True)
    agreement_n = draw_stacked_rows(axes[0], rows, kind="agreement")
    conflict_n = draw_stacked_rows(axes[1], rows, kind="conflict")
    axes[0].set_title(
        "Ambiguous choice — competence control",
        color=INK,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[1].set_title(
        "Unambiguous choice — Charter vs cheapest",
        color=INK,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[0].set_xlabel("share of agreement-eval runs (%)", color=INK, fontsize=10)
    axes[1].set_xlabel("share of conflict-eval runs (%)", color=INK, fontsize=10)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for label, _ in rows], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle(
        f"Native-reasoning GRPO — checkpoint {step}, {PRESENTATION_LABEL[presentation]}",
        x=0.055,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.015,
        f"Deterministic 201-presentation screen; directional only. "
        f"Agreement n={agreement_n[0]:,}; conflict n={conflict_n[0]:,} runs/row.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.82, bottom=0.29, left=0.23, right=0.985, wspace=0.08)
    save_figure(fig, output)


def figure_all_checkpoints(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    presentation: str,
    output: Path,
) -> None:
    # The scientific comparison is between parent arms, with step as the
    # trajectory inside each arm.  Grouping by step first makes the six rows
    # look like three unrelated pairwise comparisons and obscures that shape.
    header_metrics = {
        "agreement_runs": {"n": 0, "rates": {}},
        "conflict_runs": {"n": 0, "rates": {}},
    }
    rows: list[tuple[str, Mapping[str, Any]]] = []
    real_metrics: list[Mapping[str, Any]] = []
    for cell, header in (
        (CELLS[0], "PUBLIC INSTRUCT PARENT"),
        (CELLS[1], "CHARTER-GRAFT PARENT"),
    ):
        rows.append((header, header_metrics))
        for step in STEPS:
            metrics = grid[cell][step]["by_mode"][presentation]
            rows.append((f"step {step}", metrics))
            real_metrics.append(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 8.2), sharey=True)
    draw_stacked_rows(axes[0], rows, kind="agreement")
    draw_stacked_rows(axes[1], rows, kind="conflict")
    agreement_ns = [int(metrics["agreement_runs"]["n"]) for metrics in real_metrics]
    conflict_ns = [int(metrics["conflict_runs"]["n"]) for metrics in real_metrics]
    agreement_n = (min(agreement_ns), max(agreement_ns))
    conflict_n = (min(conflict_ns), max(conflict_ns))
    agreement_n_label = (
        f"{agreement_n[0]:,}"
        if agreement_n[0] == agreement_n[1]
        else f"{agreement_n[0]:,}–{agreement_n[1]:,}"
    )
    conflict_n_label = (
        f"{conflict_n[0]:,}"
        if conflict_n[0] == conflict_n[1]
        else f"{conflict_n[0]:,}–{conflict_n[1]:,}"
    )
    axes[0].set_title(
        "Ambiguous choice — competence control",
        color=INK,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[1].set_title(
        "Unambiguous choice — Charter vs cheapest",
        color=INK,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[0].set_xlabel("share of agreement-eval runs (%)", color=INK, fontsize=10)
    axes[1].set_xlabel("share of conflict-eval runs (%)", color=INK, fontsize=10)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for label, _ in rows], fontsize=9)
    axes[0].invert_yaxis()
    for tick, (label, _) in zip(axes[0].get_yticklabels(), rows, strict=True):
        if label.endswith("PARENT"):
            tick.set_fontweight("bold")
            tick.set_fontsize(10.0)
            tick.set_color(INK)
    for axis in axes:
        for header_y in (0, 4):
            axis.axhspan(
                header_y - 0.48,
                header_y + 0.48,
                color="#eeeee9",
                zorder=0,
            )
            axis.axhline(header_y - 0.48, color=GRID, linewidth=1.4)
            axis.axhline(header_y + 0.48, color=GRID, linewidth=1.4)
    fig.suptitle(
        f"Figure 0 — native-reasoning GRPO across checkpoints, "
        f"{PRESENTATION_LABEL[presentation]}",
        x=0.055,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.942,
        "Rows grouped by parent arm; optimizer step increases within each arm.",
        ha="left",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.985,
        0.015,
        f"Deterministic 201-presentation screen; directional only. "
        f"Agreement n={agreement_n_label}; "
        f"conflict n={conflict_n_label} runs/row. "
        "Outcomes sum to 100%.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.89, bottom=0.17, left=0.20, right=0.985, wspace=0.08)
    save_figure(fig, output)


def figure_contrast(summary: Mapping[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    styles = {
        "canonical": ("#6d6c66", ":", "^"),
        "trained": ("#029e73", "-", "o"),
        "heldout": ("#de8f05", "--", "s"),
    }
    for presentation in PRESENTATIONS:
        color, linestyle, marker = styles[presentation]
        values = [
            100 * summary["contrasts"][f"{step}|{presentation}"]["directional_separation"]
            for step in STEPS
        ]
        ax.plot(
            STEPS,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,
            label=PRESENTATION_LABEL[presentation],
        )
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xlabel("optimizer updates", color=INK)
    ax.set_xticks(STEPS)
    ax.set_ylabel("graft–public directional separation (pp)", color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle(
        "Does the Charter graft survive native-reasoning GRPO?",
        x=0.08,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.015,
        "201 presentations/endpoint; single-seed directional screen.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save_figure(fig, output)


def figure_format(summary: Mapping[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    for cell, color, marker in (
        (CELLS[0], "#6d6c66", "o"),
        (CELLS[1], "#0173b2", "s"),
    ):
        label = "public instruct" if cell.startswith("public") else "Charter graft"
        parseable = [
            100 * summary["endpoints"][f"{cell}|{step}|heldout"]["parseable_final_rate"]
            for step in STEPS
        ]
        agreement = [
            100 * summary["endpoints"][f"{cell}|{step}|heldout"]["agreement_accuracy"]
            for step in STEPS
        ]
        ax.plot(STEPS, parseable, color=color, marker=marker, label=f"{label} · parse")
        ax.plot(
            STEPS,
            agreement,
            color=color,
            marker=marker,
            linestyle="--",
            alpha=0.75,
            label=f"{label} · agreement",
        )
    ax.set_xlabel("optimizer updates", color=INK)
    ax.set_xticks(STEPS)
    ax.set_ylabel("rate (%)", color=INK)
    ax.set_ylim(0, 102)
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle(
        "Native final-channel compliance and task competence",
        x=0.08,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output)


def write_csv(summary: Mapping[str, Any], path: Path) -> None:
    rows = []
    for key, values in summary["endpoints"].items():
        cell, checkpoint, presentation = key.split("|")
        rows.append(
            {
                "cell": cell,
                "checkpoint": checkpoint,
                "presentation_mode": presentation,
                **values,
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results(summary: Mapping[str, Any], output: Path) -> None:
    lines = [
        "# Sampled native-reasoning GRPO results",
        "",
        "This is a deterministic single-seed directional screen with 201 prompt "
        "presentations per endpoint. Do not interpret small changes as precise estimates.",
        "",
        "| presentation | step | separation | graft Charter | public Charter | graft agreement | public agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for presentation in PRESENTATIONS:
        for step in STEPS:
            contrast = summary["contrasts"][f"{step}|{presentation}"]
            public = summary["endpoints"][f"{CELLS[0]}|{step}|{presentation}"]
            graft = summary["endpoints"][f"{CELLS[1]}|{step}|{presentation}"]
            lines.append(
                f"| {presentation} | {step} | {100 * contrast['directional_separation']:.1f} pp "
                f"| {100 * graft['conflict_charter_rate']:.1f}% "
                f"| {100 * public['conflict_charter_rate']:.1f}% "
                f"| {100 * graft['agreement_accuracy']:.1f}% "
                f"| {100 * public['agreement_accuracy']:.1f}% |"
            )
    lines.append("")
    output.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    grid = load_grid(args.eval_root.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = compile_summary(grid)
    atomic_json(output / "compiled_metrics.json", summary)
    write_csv(summary, output / "endpoint_metrics.csv")
    for step in STEPS:
        figure_outcomes(
            grid,
            step=step,
            presentation="heldout",
            output=output / f"figure_0_checkpoint_{step}_heldout",
        )
    for presentation in PRESENTATIONS:
        figure_all_checkpoints(
            grid,
            presentation=presentation,
            output=output / f"figure_0_all_checkpoints_{presentation}",
        )
    figure_contrast(summary, output / "figure_1_graft_effect_trajectory")
    figure_format(summary, output / "figure_2_native_format_trajectory")
    write_results(summary, output / "RESULTS.md")
    atomic_json(
        output / "ANALYSIS_DONE.json",
        {"schema_version": 1, "status": "complete", "presentations_per_endpoint": 201},
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
