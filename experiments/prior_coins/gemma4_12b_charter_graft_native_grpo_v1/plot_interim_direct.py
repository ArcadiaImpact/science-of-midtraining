"""Render validated interim plots for completed direct-GRPO endpoints.

This intentionally accepts a subset of checkpoints while the live evaluation
grid is still running.  It never treats partial raw JSONL files as results:
every plotted endpoint must have a valid EVAL_DONE.json and metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
REPO_ROOT = HERE.parents[2]
for candidate in (PRIOR_COINS, REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import score_factorised as sf  # noqa: E402
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.analyze_results import (  # noqa: E402
    draw_stacked_rows,
)
from plot_dispatch_v4_aft import GRID, INK, MUTED  # noqa: E402
from plot_wave_v1_summary import save_figure  # noqa: E402


CELLS = ("public_it-direct_grpo", "charter_graft_it-direct_grpo")
CELL_LABEL = {
    "public_it-direct_grpo": "public instruct · direct GRPO",
    "charter_graft_it-direct_grpo": "Charter graft · direct GRPO",
}
PRESENTATIONS = ("canonical", "trained", "heldout")
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


def load_snapshot(root: Path, steps: Sequence[int]) -> dict[str, dict[int, Any]]:
    grid: dict[str, dict[int, Any]] = {cell: {} for cell in CELLS}
    for cell in CELLS:
        for step in steps:
            endpoint = root / "cells" / cell / f"checkpoint-{step}"
            done_path = endpoint / "EVAL_DONE.json"
            metrics_path = endpoint / "metrics.json"
            if not done_path.is_file() or not metrics_path.is_file():
                raise RuntimeError(f"incomplete endpoint {endpoint}")
            done = json.loads(done_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            if (
                done.get("status") != "complete"
                or done.get("cell") != cell
                or int(done.get("checkpoint_step", -1)) != step
                or int(done.get("presentations", -1)) != 21_000
                or set(metrics.get("by_mode", {})) != set(PRESENTATIONS)
            ):
                raise RuntimeError(f"invalid endpoint {endpoint}")
            grid[cell][step] = metrics
    return grid


def agreement_accuracy(metrics: Mapping[str, Any]) -> float:
    return float(metrics["agreement_runs"]["rates"].get(sf.SHARED, 0.0))


def conflict_rate(metrics: Mapping[str, Any], verdict: str) -> float:
    return float(metrics["conflict_runs"]["rates"].get(verdict, 0.0))


def compile_summary(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]], steps: Sequence[int]
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for cell in CELLS:
        for step in steps:
            payload = grid[cell][step]
            for presentation in PRESENTATIONS:
                metrics = payload["by_mode"][presentation]
                diagnostics = payload["native_format_by_mode"][presentation]
                endpoints[f"{cell}|{step}|{presentation}"] = {
                    "agreement_accuracy": agreement_accuracy(metrics),
                    "agreement_n": int(metrics["agreement_runs"]["n"]),
                    "conflict_charter_rate": conflict_rate(metrics, sf.CHARTER),
                    "conflict_coin_rate": conflict_rate(metrics, sf.COIN),
                    "conflict_other_rate": conflict_rate(metrics, sf.OTHER),
                    "conflict_malformed_rate": conflict_rate(metrics, sf.MALFORMED),
                    "conflict_n": int(metrics["conflict_runs"]["n"]),
                    "consistency": float(metrics["consistency"]["rate"]),
                    **diagnostics,
                }
    public = CELLS[0]
    graft = CELLS[1]
    for step in steps:
        for presentation in PRESENTATIONS:
            p = grid[public][step]["by_mode"][presentation]
            g = grid[graft][step]["by_mode"][presentation]
            graft_charter = conflict_rate(g, sf.CHARTER) - conflict_rate(
                p, sf.CHARTER
            )
            public_coin = conflict_rate(p, sf.COIN) - conflict_rate(g, sf.COIN)
            contrasts[f"direct|{step}|{presentation}"] = {
                "graft_minus_public_charter_rate": graft_charter,
                "public_minus_graft_coin_rate": public_coin,
                "directional_separation": graft_charter + public_coin,
                "graft_minus_public_agreement_accuracy": (
                    agreement_accuracy(g) - agreement_accuracy(p)
                ),
            }
    return {
        "schema_version": 1,
        "status": "interim",
        "native_mode": "direct",
        "direct_grid_complete": tuple(steps) == (0, 64, 128, 256),
        "checkpoints": list(steps),
        "presentation_modes": list(PRESENTATIONS),
        "validated_presentations_per_endpoint": 21_000,
        "endpoints": endpoints,
        "contrasts": contrasts,
    }


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
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def figure_outcomes(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    step: int,
    presentation: str,
    output: Path,
) -> None:
    rows = [
        (CELL_LABEL[cell], grid[cell][step]["by_mode"][presentation])
        for cell in CELLS
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
        f"Interim direct GRPO — checkpoint {step}, {PRESENTATION_LABEL[presentation]}",
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
        f"Validated snapshot; reasoning arms excluded. Agreement n={agreement_n[0]:,}; "
        f"conflict n={conflict_n[0]:,} runs/row. Outcomes sum to 100%.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.82, bottom=0.29, left=0.21, right=0.985, wspace=0.08)
    save_figure(fig, output)


def figure_all_checkpoints(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    steps: Sequence[int],
    presentation: str,
    output: Path,
) -> None:
    rows = [
        (
            f"step {step} · "
            + ("public instruct" if cell.startswith("public") else "Charter graft"),
            grid[cell][step]["by_mode"][presentation],
        )
        for step in steps
        for cell in CELLS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 7.8), sharey=True)
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
    for boundary in range(2, len(rows), 2):
        for ax in axes:
            ax.axhline(boundary - 0.5, color=GRID, linewidth=1.1)
    fig.suptitle(
        f"Figure 0 — direct GRPO across checkpoints, "
        f"{PRESENTATION_LABEL[presentation]}",
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
        f"Complete validated direct grid; reasoning arms excluded. "
        f"Agreement n={agreement_n[0]:,}; conflict n={conflict_n[0]:,} runs/row. "
        "Outcomes sum to 100%.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.20, right=0.985, wspace=0.08)
    save_figure(fig, output)


def figure_contrast(summary: Mapping[str, Any], steps: Sequence[int], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    styles = {
        "canonical": ("#6d6c66", ":", "^"),
        "trained": ("#029e73", "-", "o"),
        "heldout": ("#de8f05", "--", "s"),
    }
    offsets = {
        "canonical": (-12, -18),
        "trained": (0, 10),
        "heldout": (12, -18),
    }
    for presentation in PRESENTATIONS:
        color, linestyle, marker = styles[presentation]
        values = [
            100
            * summary["contrasts"][f"direct|{step}|{presentation}"][
                "directional_separation"
            ]
            for step in steps
        ]
        ax.plot(
            steps,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.2,
            markersize=6,
            label=PRESENTATION_LABEL[presentation],
        )
        for x, y in zip(steps, values, strict=True):
            ax.annotate(
                f"{y:+.1f}",
                (x, y),
                xytext=offsets[presentation],
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )
    ax.axhline(0, color=INK, linewidth=1.1)
    ax.set_xticks(steps)
    ax.set_xlabel("GRPO optimizer updates", color=INK)
    ax.set_ylabel(
        "graft–public directional separation (percentage points)\n"
        "positive = Charter graft persists",
        color=INK,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle(
        "Interim: does the Charter graft survive direct GRPO?",
        x=0.08,
        y=0.99,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.015,
        (
            "Complete validated direct grid; both reasoning arms excluded."
            if summary["direct_grid_complete"]
            else "Validated checkpoints only; checkpoint 256 and both reasoning arms excluded."
        ),
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.88, bottom=0.19, left=0.15, right=0.97)
    save_figure(fig, output)


def figure_format(summary: Mapping[str, Any], steps: Sequence[int], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    for cell, color, marker in (
        ("public_it-direct_grpo", "#6d6c66", "o"),
        ("charter_graft_it-direct_grpo", "#0173b2", "s"),
    ):
        label = "public instruct" if cell.startswith("public") else "Charter graft"
        parseable = [
            100
            * summary["endpoints"][f"{cell}|{step}|heldout"]["parseable_final_rate"]
            for step in steps
        ]
        agreement = [
            100
            * summary["endpoints"][f"{cell}|{step}|heldout"]["agreement_accuracy"]
            for step in steps
        ]
        ax.plot(
            steps,
            parseable,
            color=color,
            marker=marker,
            linewidth=2.1,
            label=f"{label} · parseable final",
        )
        ax.plot(
            steps,
            agreement,
            color=color,
            marker=marker,
            linewidth=1.8,
            linestyle="--",
            alpha=0.8,
            label=f"{label} · agreement accuracy",
        )
    ax.set_xticks(steps)
    ax.set_ylim(0, 102)
    ax.set_xlabel("GRPO optimizer updates", color=INK)
    ax.set_ylabel("rate (%)", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8.4, loc="lower right")
    fig.suptitle(
        "Interim direct GRPO — held-out format and competence",
        x=0.08,
        y=0.99,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.015,
        "Native final-channel parseability and ambiguous-choice agreement accuracy.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.88, bottom=0.17, left=0.12, right=0.97)
    save_figure(fig, output)


def write_readme(summary: Mapping[str, Any], output: Path) -> None:
    coverage = (
        "The complete validated direct-GRPO checkpoint grid. Both reasoning arms "
        "were still training at snapshot time."
        if summary["direct_grid_complete"]
        else "Validated completed endpoints only. Checkpoint 256 and both reasoning "
        "arms were still running at snapshot time."
    )
    lines = [
        "# Interim direct-GRPO plots",
        "",
        coverage,
        "",
        "| step | presentation | directional separation | graft Charter | public Charter | graft agreement | public agreement |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for step in summary["checkpoints"]:
        for presentation in PRESENTATIONS:
            contrast = summary["contrasts"][f"direct|{step}|{presentation}"]
            public = summary["endpoints"][
                f"public_it-direct_grpo|{step}|{presentation}"
            ]
            graft = summary["endpoints"][
                f"charter_graft_it-direct_grpo|{step}|{presentation}"
            ]
            lines.append(
                f"| {step} | {presentation} | "
                f"{100 * contrast['directional_separation']:+.1f} pp | "
                f"{100 * graft['conflict_charter_rate']:.1f}% | "
                f"{100 * public['conflict_charter_rate']:.1f}% | "
                f"{100 * graft['agreement_accuracy']:.1f}% | "
                f"{100 * public['agreement_accuracy']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "Directional separation = (graft − public Charter rate) + "
            "(public − graft coin rate). This is a single-seed interim read.",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    steps = tuple(args.steps)
    if not steps or tuple(sorted(set(steps))) != steps:
        raise ValueError("steps must be unique and increasing")
    snapshot = args.eval_snapshot.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid = load_snapshot(snapshot, steps)
    summary = compile_summary(grid, steps)
    summary["snapshot_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    summary["eval_snapshot"] = str(snapshot)
    atomic_json(output / "compiled_metrics.json", summary)
    write_csv(summary, output / "endpoint_metrics.csv")
    figure_all_checkpoints(
        grid,
        steps=steps,
        presentation="heldout",
        output=output / "figure_0_all_checkpoints_heldout",
    )
    for step in steps:
        figure_outcomes(
            grid,
            step=step,
            presentation="heldout",
            output=output / f"figure_0_checkpoint_{step}_heldout",
        )
    figure_contrast(summary, steps, output / "figure_1_graft_effect_trajectory")
    figure_format(summary, steps, output / "figure_2_native_format_trajectory")
    write_readme(summary, output / "README.md")
    atomic_json(
        output / "DIRECT_PLOTS_DONE.json",
        {
            "status": "complete",
            "interim": True,
            "checkpoints": list(steps),
            "cells": list(CELLS),
            "snapshot_at": summary["snapshot_at"],
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
