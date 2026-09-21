"""Compile the native-GRPO grid and render Figure-0-style plots."""

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
PRIOR_COINS = HERE.parent
for candidate in (PRIOR_COINS, HERE.parents[2] / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import score_factorised as sf  # noqa: E402
from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.analyze_results import (  # noqa: E402
    draw_stacked_rows,
)
from plot_dispatch_v4_aft import GRID, INK, MUTED  # noqa: E402
from plot_wave_v1_summary import save_figure  # noqa: E402

from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (  # noqa: E402
    CELLS,
    CHECKPOINTS,
    PRESENTATION_MODES,
)

CELL_LABEL = {
    "public_it-direct_grpo": "public instruct · direct GRPO",
    "charter_graft_it-direct_grpo": "Charter graft · direct GRPO",
    "public_it-reasoning_grpo": "public instruct · native-reasoning GRPO",
    "charter_graft_it-reasoning_grpo": "Charter graft · native-reasoning GRPO",
}
MODE_LABEL = {
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
    marker = root / "EVAL_GRID_DONE.json"
    if (
        not marker.is_file()
        or json.loads(marker.read_text()).get("status") != "complete"
    ):
        raise RuntimeError(f"eval grid is not complete: {marker}")
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for cell in CELLS:
        result[cell.label] = {}
        for step in CHECKPOINTS:
            endpoint = root / "cells" / cell.label / f"checkpoint-{step}"
            done_path = endpoint / "EVAL_DONE.json"
            metrics_path = endpoint / "metrics.json"
            if not done_path.is_file() or not metrics_path.is_file():
                raise RuntimeError(f"incomplete endpoint {endpoint}")
            done = json.loads(done_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            if (
                done.get("status") != "complete"
                or done.get("cell") != cell.label
                or done.get("checkpoint_step") != step
                or done.get("presentations") != 21_000
                or set(metrics.get("by_mode", {})) != set(PRESENTATION_MODES)
            ):
                raise RuntimeError(f"invalid endpoint {endpoint}")
            result[cell.label][step] = metrics
    return result


def agreement_accuracy(metrics: Mapping[str, Any]) -> float:
    return float(metrics["agreement_runs"]["rates"].get(sf.SHARED, 0.0))


def conflict_rate(metrics: Mapping[str, Any], verdict: str) -> float:
    return float(metrics["conflict_runs"]["rates"].get(verdict, 0.0))


def pair_contrast(
    public: Mapping[str, Any], graft: Mapping[str, Any]
) -> dict[str, float]:
    graft_charter = conflict_rate(graft, sf.CHARTER) - conflict_rate(public, sf.CHARTER)
    public_coin = conflict_rate(public, sf.COIN) - conflict_rate(graft, sf.COIN)
    return {
        "graft_minus_public_charter_rate": round(graft_charter, 4),
        "public_minus_graft_coin_rate": round(public_coin, 4),
        "directional_separation": round(graft_charter + public_coin, 4),
        "graft_minus_public_agreement_accuracy": round(
            agreement_accuracy(graft) - agreement_accuracy(public), 4
        ),
    }


def compile_summary(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for cell in CELLS:
        for step in CHECKPOINTS:
            payload = grid[cell.label][step]
            for presentation in PRESENTATION_MODES:
                metrics = payload["by_mode"][presentation]
                diagnostics = payload["native_format_by_mode"][presentation]
                endpoints[f"{cell.label}|{step}|{presentation}"] = {
                    "parent": cell.parent,
                    "native_mode": cell.mode,
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
    for mode in ("direct", "reasoning"):
        public = f"public_it-{mode}_grpo"
        graft = f"charter_graft_it-{mode}_grpo"
        for step in CHECKPOINTS:
            for presentation in PRESENTATION_MODES:
                contrasts[f"{mode}|{step}|{presentation}"] = pair_contrast(
                    grid[public][step]["by_mode"][presentation],
                    grid[graft][step]["by_mode"][presentation],
                )
    return {
        "schema_version": 1,
        "checkpoints": list(CHECKPOINTS),
        "presentation_modes": list(PRESENTATION_MODES),
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
        (CELL_LABEL[cell.label], grid[cell.label][step]["by_mode"][presentation])
        for cell in CELLS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8), sharey=True)
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
    for ax in axes:
        ax.axhline(1.5, color=GRID, linewidth=1.1)
    fig.suptitle(
        f"Native Gemma 4 GRPO — checkpoint {step}, {MODE_LABEL[presentation]}",
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
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.21, right=0.985, wspace=0.08)
    save_figure(fig, output)


def figure_contrast(summary: Mapping[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    styles = {
        "canonical": ("#6d6c66", ":", "^"),
        "trained": ("#029e73", "-", "o"),
        "heldout": ("#de8f05", "--", "s"),
    }
    for ax, mode in zip(axes, ("direct", "reasoning"), strict=True):
        for presentation in PRESENTATION_MODES:
            color, linestyle, marker = styles[presentation]
            values = [
                100
                * summary["contrasts"][f"{mode}|{step}|{presentation}"][
                    "directional_separation"
                ]
                for step in CHECKPOINTS
            ]
            ax.plot(
                CHECKPOINTS,
                values,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                label=MODE_LABEL[presentation],
            )
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_title(f"{mode} GRPO", color=INK, fontweight="bold")
        ax.set_xlabel("optimizer updates", color=INK)
        ax.set_xticks(CHECKPOINTS)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("graft–public directional separation (pp)", color=INK)
    axes[1].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle(
        "Does the Charter graft survive matched GRPO?",
        x=0.06,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output)


def figure_format(summary: Mapping[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    for ax, mode in zip(axes, ("direct", "reasoning"), strict=True):
        for parent, color, marker in (
            ("public_it", "#6d6c66", "o"),
            ("charter_graft_it", "#0173b2", "s"),
        ):
            cell = f"{parent}-{mode}_grpo"
            parseable = [
                100
                * summary["endpoints"][f"{cell}|{step}|heldout"]["parseable_final_rate"]
                for step in CHECKPOINTS
            ]
            agreement = [
                100
                * summary["endpoints"][f"{cell}|{step}|heldout"]["agreement_accuracy"]
                for step in CHECKPOINTS
            ]
            label = "public instruct" if parent == "public_it" else "Charter graft"
            ax.plot(
                CHECKPOINTS,
                parseable,
                color=color,
                marker=marker,
                label=f"{label} · parse",
            )
            ax.plot(
                CHECKPOINTS,
                agreement,
                color=color,
                marker=marker,
                linestyle="--",
                alpha=0.75,
                label=f"{label} · agreement",
            )
        ax.set_title(f"{mode} · held-out templates", color=INK, fontweight="bold")
        ax.set_xlabel("optimizer updates", color=INK)
        ax.set_xticks(CHECKPOINTS)
        ax.set_ylim(0, 102)
        ax.grid(color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("rate (%)", color=INK)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Native final-channel compliance and task competence",
        x=0.06,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output)


def write_results(summary: Mapping[str, Any], output: Path) -> None:
    lines = [
        "# Gemma 4 native GRPO results",
        "",
        "Primary readout: held-out presentation templates. Directional separation is "
        "(graft − public Charter rate) + (public − graft coin rate).",
        "",
        "| mode | step | separation | graft Charter | public Charter | graft agreement | public agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("direct", "reasoning"):
        public = f"public_it-{mode}_grpo"
        graft = f"charter_graft_it-{mode}_grpo"
        for step in CHECKPOINTS:
            contrast = summary["contrasts"][f"{mode}|{step}|heldout"]
            p = summary["endpoints"][f"{public}|{step}|heldout"]
            g = summary["endpoints"][f"{graft}|{step}|heldout"]
            lines.append(
                f"| {mode} | {step} | {100 * contrast['directional_separation']:.1f} pp "
                f"| {100 * g['conflict_charter_rate']:.1f}% | {100 * p['conflict_charter_rate']:.1f}% "
                f"| {100 * g['agreement_accuracy']:.1f}% | {100 * p['agreement_accuracy']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "This is a single-seed directional screen (seed 42). Direct and reasoning "
            "step-0 baselines are evaluated under their respective native tokenizer modes.",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    grid = load_grid(args.eval_root.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = compile_summary(grid)
    atomic_json(output / "compiled_metrics.json", summary)
    write_csv(summary, output / "endpoint_metrics.csv")
    for step in CHECKPOINTS:
        figure_0(
            grid,
            step=step,
            presentation="heldout",
            output=output / f"figure_0_checkpoint_{step}_heldout",
        )
    for presentation in ("canonical", "trained"):
        figure_0(
            grid,
            step=256,
            presentation=presentation,
            output=output / f"figure_0_checkpoint_256_{presentation}",
        )
    figure_contrast(summary, output / "figure_1_graft_effect_trajectory")
    figure_format(summary, output / "figure_2_native_format_trajectory")
    write_results(summary, output / "RESULTS.md")
    atomic_json(output / "ANALYSIS_DONE.json", {"status": "complete"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
