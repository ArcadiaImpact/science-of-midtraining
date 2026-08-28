"""Compile and plot the three-chunk cumulative direct-GRPO trajectory."""

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

from __future__ import annotations

import argparse
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

from plot_dispatch_v4_aft import GRID, INK
from plot_wave_v1_summary import save_figure

from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.analyze_phase2 import (
    PARENT_LABEL,
    agreement_accuracy,
    conflict_rate,
    figure_0,
    load_metrics,
    write_csv,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.analyze_results import (
    MODE_LABEL,
    pair_contrast,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (
    PRESENTATION_MODES,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.phase2_contracts import (
    PHASE1_STEP,
    PHASE2_CELLS,
    PHASE2_CHECKPOINTS,
)
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.phase3_contracts import (
    CUMULATIVE_START_STEP,
    PHASE3_CELLS,
    PHASE3_CHECKPOINTS,
)

ALL_CHECKPOINTS = (
    PHASE1_STEP,
    *(PHASE1_STEP + step for step in PHASE2_CHECKPOINTS),
    *(CUMULATIVE_START_STEP + step for step in PHASE3_CHECKPOINTS),
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_grid(
    original_eval: Path, phase2_eval: Path, phase3_eval: Path
) -> dict[str, dict[int, Any]]:
    grid: dict[str, dict[int, Any]] = {}
    phase3_by_parent = {cell.parent: cell for cell in PHASE3_CELLS}
    for phase2_cell in PHASE2_CELLS:
        parent = phase2_cell.parent
        grid[parent] = {
            PHASE1_STEP: load_metrics(
                original_eval
                / "cells"
                / phase2_cell.phase1_label
                / f"checkpoint-{PHASE1_STEP}"
            )
        }
        for local_step in PHASE2_CHECKPOINTS:
            grid[parent][PHASE1_STEP + local_step] = load_metrics(
                phase2_eval / "cells" / phase2_cell.label / f"checkpoint-{local_step}"
            )
        phase3_cell = phase3_by_parent[parent]
        for local_step in PHASE3_CHECKPOINTS:
            grid[parent][CUMULATIVE_START_STEP + local_step] = load_metrics(
                phase3_eval / "cells" / phase3_cell.label / f"checkpoint-{local_step}"
            )
    return grid


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
                    "conflict_charter_rate": conflict_rate(metrics, "charter"),
                    "conflict_coin_rate": conflict_rate(metrics, "coin"),
                    "conflict_other_rate": conflict_rate(metrics, "other"),
                    "conflict_malformed_rate": conflict_rate(metrics, "malformed"),
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
        "phase3_local_checkpoints": list(PHASE3_CHECKPOINTS),
        "cumulative_checkpoints": list(ALL_CHECKPOINTS),
        "presentation_modes": list(PRESENTATION_MODES),
        "endpoints": endpoints,
        "contrasts": contrasts,
    }


def figure_trajectory(summary: Mapping[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
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
    for boundary, label in (
        (PHASE1_STEP, "chunk 2"),
        (CUMULATIVE_START_STEP, "chunk 3"),
    ):
        ax.axvline(boundary, color=GRID, linewidth=1, linestyle="--")
        ax.annotate(
            label,
            (boundary, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            color=INK,
            fontsize=8,
            va="top",
        )
    ax.set_title("Does graft separation survive three direct-GRPO chunks?", color=INK)
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
        "# Gemma 4 three-chunk direct-GRPO continuation results",
        "",
        (
            "Each continuation chunk uses 1,024 new, disjoint, stratum-matched "
            "prompts and a fresh optimizer/scheduler while preserving the preceding "
            "LoRA weights."
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
    grid = load_grid(
        args.original_eval.resolve(),
        args.phase2_eval.resolve(),
        args.phase3_eval.resolve(),
    )
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
    figure_trajectory(summary, output / "figure_1_graft_effect_three_chunks")
    write_results(summary, output / "RESULTS.md")
    atomic_json(output / "ANALYSIS_DONE.json", {"status": "complete"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-eval", type=Path, required=True)
    parser.add_argument("--phase2-eval", type=Path, required=True)
    parser.add_argument("--phase3-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
