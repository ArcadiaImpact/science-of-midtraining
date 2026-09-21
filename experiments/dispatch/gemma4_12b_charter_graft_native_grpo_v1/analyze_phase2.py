"""Compile and plot the cumulative direct-GRPO continuation trajectory."""

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = HERE.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import score_factorised as sf
from plot_dispatch_v4_aft import GRID, INK, MUTED
from plot_wave_v1_summary import save_figure

from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.analyze_results import (
    draw_stacked_rows,
)
from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.analyze_results import (
    MODE_LABEL,
    pair_contrast,
)
from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.contracts import (
    PRESENTATION_MODES,
)
from experiments.dispatch.gemma4_12b_charter_graft_native_grpo_v1.phase2_contracts import (
    CUMULATIVE_CHECKPOINTS,
    PHASE1_STEP,
    PHASE2_CELLS,
    PHASE2_CHECKPOINTS,
)

PARENT_LABEL = {
    "public_it": "public instruct",
    "charter_graft_it": "Charter graft",
}
ALL_CHECKPOINTS = (PHASE1_STEP, *CUMULATIVE_CHECKPOINTS)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_metrics(path: Path) -> dict[str, Any]:
    done = path / "EVAL_DONE.json"
    metrics = path / "metrics.json"
    if not done.is_file() or not metrics.is_file():
        raise RuntimeError(f"incomplete evaluation endpoint {path}")
    if json.loads(done.read_text()).get("status") != "complete":
        raise RuntimeError(f"evaluation marker is not complete: {done}")
    payload = json.loads(metrics.read_text())
    if set(payload.get("by_mode", {})) != set(PRESENTATION_MODES):
        raise RuntimeError(f"presentation modes drifted in {metrics}")
    return payload


def load_grid(original_eval: Path, phase2_eval: Path) -> dict[str, dict[int, Any]]:
    grid: dict[str, dict[int, Any]] = {}
    for cell in PHASE2_CELLS:
        original_label = cell.phase1_label
        grid[cell.parent] = {
            PHASE1_STEP: load_metrics(
                original_eval / "cells" / original_label / f"checkpoint-{PHASE1_STEP}"
            )
        }
        for phase2_step in PHASE2_CHECKPOINTS:
            grid[cell.parent][PHASE1_STEP + phase2_step] = load_metrics(
                phase2_eval / "cells" / cell.label / f"checkpoint-{phase2_step}"
            )
    return grid


def agreement_accuracy(metrics: Mapping[str, Any]) -> float:
    return float(metrics["agreement_runs"]["rates"].get(sf.SHARED, 0.0))


def conflict_rate(metrics: Mapping[str, Any], verdict: str) -> float:
    return float(metrics["conflict_runs"]["rates"].get(verdict, 0.0))


def compile_summary(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for parent in PARENT_LABEL:
        for step in ALL_CHECKPOINTS:
            payload = grid[parent][step]
            for presentation in PRESENTATION_MODES:
                metrics = payload["by_mode"][presentation]
                diagnostics = payload["native_format_by_mode"][presentation]
                endpoints[f"{parent}|{step}|{presentation}"] = {
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
    for step in ALL_CHECKPOINTS:
        for presentation in PRESENTATION_MODES:
            contrasts[f"{step}|{presentation}"] = pair_contrast(
                grid["public_it"][step]["by_mode"][presentation],
                grid["charter_graft_it"][step]["by_mode"][presentation],
            )
    return {
        "schema_version": 1,
        "phase1_initial_checkpoint": PHASE1_STEP,
        "phase2_local_checkpoints": list(PHASE2_CHECKPOINTS),
        "cumulative_checkpoints": list(ALL_CHECKPOINTS),
        "presentation_modes": list(PRESENTATION_MODES),
        "endpoints": endpoints,
        "contrasts": contrasts,
    }


def write_csv(summary: Mapping[str, Any], path: Path) -> None:
    rows = []
    for key, values in summary["endpoints"].items():
        parent, checkpoint, presentation = key.split("|")
        rows.append(
            {
                "parent": parent,
                "cumulative_checkpoint": checkpoint,
                "presentation_mode": presentation,
                **values,
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_0(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    step: int,
    presentation: str,
    output: Path,
) -> None:
    rows = [
        (PARENT_LABEL[parent], grid[parent][step]["by_mode"][presentation])
        for parent in PARENT_LABEL
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 4.2), sharey=True)
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
        f"Direct GRPO continuation — cumulative checkpoint {step}, "
        f"{MODE_LABEL[presentation]}",
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
        f"Frozen Dispatch battery. Agreement n={agreement_n[0]:,}–{agreement_n[1]:,}; "
        f"conflict n={conflict_n[0]:,}–{conflict_n[1]:,} runs/row.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.79, bottom=0.30, left=0.17, right=0.985, wspace=0.08)
    save_figure(fig, output)


def figure_trajectory(summary: Mapping[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    styles = {
        "canonical": ("#6d6c66", ":", "^"),
        "trained": ("#029e73", "-", "o"),
        "heldout": ("#de8f05", "--", "s"),
    }
    for presentation in PRESENTATION_MODES:
        color, linestyle, marker = styles[presentation]
        values = [
            100
            * summary["contrasts"][f"{step}|{presentation}"]["directional_separation"]
            for step in ALL_CHECKPOINTS
        ]
        ax.plot(
            ALL_CHECKPOINTS,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,
            label=MODE_LABEL[presentation],
        )
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(PHASE1_STEP, color=GRID, linewidth=1, linestyle="--")
    ax.set_title("Does graft separation survive another direct-GRPO phase?", color=INK)
    ax.set_xlabel("cumulative optimizer updates", color=INK)
    ax.set_ylabel("graft–public directional separation (pp)", color=INK)
    ax.set_xticks(ALL_CHECKPOINTS)
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_figure(fig, output)


def write_results(summary: Mapping[str, Any], output: Path) -> None:
    lines = [
        "# Gemma 4 direct-GRPO continuation results",
        "",
        (
            "Phase two initializes from each phase-one step-256 adapter and uses a "
            "fresh optimizer/scheduler on 1,024 new, disjoint, stratum-matched prompts."
        ),
        "",
        (
            "| cumulative step | held-out separation | graft Charter | public Charter "
            "| graft agreement | public agreement |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for step in ALL_CHECKPOINTS:
        contrast = summary["contrasts"][f"{step}|heldout"]
        public = summary["endpoints"][f"public_it|{step}|heldout"]
        graft = summary["endpoints"][f"charter_graft_it|{step}|heldout"]
        lines.append(
            f"| {step} | {100 * contrast['directional_separation']:.1f} pp "
            f"| {100 * graft['conflict_charter_rate']:.1f}% "
            f"| {100 * public['conflict_charter_rate']:.1f}% "
            f"| {100 * graft['agreement_accuracy']:.1f}% "
            f"| {100 * public['agreement_accuracy']:.1f}% |"
        )
    output.write_text("\n".join([*lines, ""]))


def run(args: argparse.Namespace) -> None:
    grid = load_grid(args.original_eval.resolve(), args.phase2_eval.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = compile_summary(grid)
    atomic_json(output / "compiled_metrics.json", summary)
    write_csv(summary, output / "endpoint_metrics.csv")
    for step in ALL_CHECKPOINTS:
        figure_0(
            grid,
            step=step,
            presentation="heldout",
            output=output / f"figure_0_cumulative_checkpoint_{step}_heldout",
        )
    figure_trajectory(summary, output / "figure_1_graft_effect_continuation")
    write_results(summary, output / "RESULTS.md")
    atomic_json(output / "ANALYSIS_DONE.json", {"status": "complete"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-eval", type=Path, required=True)
    parser.add_argument("--phase2-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
