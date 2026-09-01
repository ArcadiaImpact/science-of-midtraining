"""Plot native-reasoning GRPO outcome compositions as stacked trajectories."""

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
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = HERE.parent
for candidate in (REPO_ROOT, REPO_ROOT / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import score_factorised as sf
from plot_dispatch_v4_aft import GRID, INK, MUTED
from plot_wave_v1_summary import (
    AGREEMENT_CATEGORY_LABEL,
    AGREEMENT_COLOR,
    AGREEMENT_SEGMENT_ORDER,
    CATEGORY_LABEL,
    CHARTER,
    COIN,
    MALFORMED,
    OTHER,
    SEGMENT_ORDER,
    save_figure,
)

STEPS = (0, 64, 128)
PARENT_LABEL = {
    "public": "public instruct",
    "graft": "Charter graft",
}
CELL = {
    "public": "public_it-reasoning_grpo",
    "graft": "charter_graft_it-reasoning_grpo",
}
PRESENTATION_LABEL = {
    "canonical": "canonical presentation",
    "trained": "seen presentation templates (90)",
    "heldout": "held-out presentation templates (10)",
}
TASK = {
    "ambiguous": {
        "title": "Ambiguous choice — competence control",
        "metrics_key": "agreement_runs",
        "order": AGREEMENT_SEGMENT_ORDER,
        "colors": AGREEMENT_COLOR,
        "labels": AGREEMENT_CATEGORY_LABEL,
    },
    "unambiguous": {
        "title": "Unambiguous choice — Charter vs cheapest",
        "metrics_key": "conflict_runs",
        "order": SEGMENT_ORDER,
        "colors": {
            sf.CHARTER: CHARTER,
            sf.COIN: COIN,
            sf.OTHER: OTHER,
            sf.MALFORMED: MALFORMED,
        },
        "labels": CATEGORY_LABEL,
    },
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_endpoint(path: Path, presentation: str) -> dict[str, Any]:
    done = path / "EVAL_DONE.json"
    metrics = path / "metrics.json"
    if not done.is_file() or not metrics.is_file():
        raise RuntimeError(f"incomplete evaluation endpoint: {path}")
    marker = json.loads(done.read_text())
    payload = json.loads(metrics.read_text())
    if (
        marker.get("status") != "complete"
        or int(marker.get("presentations", -1)) != 201
        or presentation not in payload.get("by_mode", {})
    ):
        raise RuntimeError(f"invalid sampled reasoning endpoint: {path}")
    return payload["by_mode"][presentation]


def load_trajectory(
    eval_root: Path, *, presentation: str
) -> dict[str, dict[int, dict[str, Any]]]:
    trajectory: dict[str, dict[int, dict[str, Any]]] = {}
    for parent, cell in CELL.items():
        trajectory[parent] = {
            step: load_endpoint(
                eval_root / "cells" / cell / f"checkpoint-{step}", presentation
            )
            for step in STEPS
        }
    return trajectory


def compile_rates(
    trajectory: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    presentation: str,
) -> dict[str, Any]:
    plots: dict[str, Any] = {}
    for task_name, task in TASK.items():
        metrics_key = str(task["metrics_key"])
        order = tuple(task["order"])
        for parent in PARENT_LABEL:
            endpoints = []
            for step in STEPS:
                metrics = trajectory[parent][step][metrics_key]
                rates = {
                    verdict: float(metrics["rates"].get(verdict, 0.0))
                    for verdict in order
                }
                source_rate_sum = sum(rates.values())
                if abs(source_rate_sum - 1.0) > 5e-4:
                    raise RuntimeError(
                        f"rates do not sum to one: {task_name}/{parent}/{step}"
                    )
                rates = {
                    verdict: value / source_rate_sum
                    for verdict, value in rates.items()
                }
                endpoints.append(
                    {
                        "step": step,
                        "n": int(metrics["n"]),
                        "rates": rates,
                        "source_rate_sum": source_rate_sum,
                    }
                )
            plots[f"{task_name}|{parent}"] = endpoints
    return {
        "schema_version": 1,
        "training_mode": "native-reasoning GRPO",
        "presentation": presentation,
        "steps": list(STEPS),
        "plots": plots,
    }


def values_for(
    rates: Mapping[str, Any], task_name: str, parent: str
) -> tuple[Sequence[str], list[list[float]], Sequence[Mapping[str, Any]]]:
    task = TASK[task_name]
    order = tuple(task["order"])
    endpoints = rates["plots"][f"{task_name}|{parent}"]
    values = [
        [100 * endpoint["rates"][verdict] for endpoint in endpoints]
        for verdict in order
    ]
    return order, values, endpoints


def style_axis(ax: Any) -> None:
    ax.set_xlim(STEPS[0], STEPS[-1])
    ax.set_ylim(0, 100)
    ax.set_xticks(STEPS)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelleft=True, labelbottom=True)


def plot_stacked_trajectory(
    rates: Mapping[str, Any],
    *,
    task_name: str,
    parent: str,
    presentation: str,
    output: Path,
) -> None:
    task = TASK[task_name]
    order, values, endpoints = values_for(rates, task_name, parent)
    colors = task["colors"]
    labels = task["labels"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.stackplot(
        STEPS,
        *values,
        colors=[colors[verdict] for verdict in order],
        edgecolor="white",
        linewidth=0.9,
        alpha=0.96,
    )
    ax.set_title(
        f"{task['title']} · {PARENT_LABEL[parent]}",
        loc="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
        pad=25,
    )
    ax.text(
        0,
        1.01,
        f"{PRESENTATION_LABEL[presentation]} · native-reasoning GRPO",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=MUTED,
        fontsize=9.5,
    )
    style_axis(ax)
    ax.set_xlabel("optimizer updates", color=INK)
    ax.set_ylabel("share of eval runs (%)", color=INK)
    ax.legend(
        handles=[
            Patch(facecolor=colors[verdict], label=labels[verdict])
            for verdict in order
        ],
        frameon=False,
        fontsize=8.5,
        ncol=len(order),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )
    ns = {endpoint["n"] for endpoint in endpoints}
    n_text = f"n={next(iter(ns)):,}/checkpoint" if len(ns) == 1 else "n varies"
    fig.text(
        0.985,
        0.015,
        f"Deterministic 201-presentation directional screen; {n_text}.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.79, bottom=0.27, left=0.10, right=0.985)
    save_figure(fig, output)


def plot_stacked_overview(
    rates: Mapping[str, Any], *, presentation: str, output: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18.4, 10.4), sharex=True, sharey=True)
    all_ns: set[int] = set()
    for row, task_name in enumerate(TASK):
        task = TASK[task_name]
        colors = task["colors"]
        for column, parent in enumerate(PARENT_LABEL):
            ax = axes[row][column]
            order, values, endpoints = values_for(rates, task_name, parent)
            all_ns.update(int(endpoint["n"]) for endpoint in endpoints)
            ax.stackplot(
                STEPS,
                *values,
                colors=[colors[verdict] for verdict in order],
                edgecolor="white",
                linewidth=0.9,
                alpha=0.96,
            )
            ax.set_title(
                f"{task['title']} · {PARENT_LABEL[parent]}",
                loc="left",
                color=INK,
                fontsize=11.5,
                fontweight="bold",
                pad=10,
            )
            style_axis(ax)

    fig.suptitle(
        "Native-reasoning GRPO outcome composition across checkpoints",
        x=0.065,
        y=0.975,
        ha="left",
        color=INK,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.935,
        f"{PRESENTATION_LABEL[presentation]} · checkpoints 0–128",
        ha="left",
        color=MUTED,
        fontsize=10.5,
    )
    fig.supxlabel("optimizer updates", x=0.525, y=0.115, color=INK)
    fig.supylabel("share of eval runs (%)", x=0.018, color=INK)
    legend = (
        (AGREEMENT_COLOR[sf.SHARED], AGREEMENT_CATEGORY_LABEL[sf.SHARED]),
        (CHARTER, CATEGORY_LABEL[sf.CHARTER]),
        (OTHER, CATEGORY_LABEL[sf.OTHER]),
        (MALFORMED, CATEGORY_LABEL[sf.MALFORMED]),
        (COIN, CATEGORY_LABEL[sf.COIN]),
    )
    fig.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in legend],
        frameon=False,
        fontsize=9,
        ncol=len(legend),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
    )
    n_text = (
        f"n={next(iter(all_ns)):,}/task/checkpoint"
        if len(all_ns) == 1
        else f"n={min(all_ns):,}–{max(all_ns):,}/task/checkpoint"
    )
    fig.text(
        0.985,
        0.015,
        f"Deterministic 201-presentation directional screen; {n_text}.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(
        top=0.88,
        bottom=0.17,
        left=0.065,
        right=0.985,
        hspace=0.28,
        wspace=0.12,
    )
    save_figure(fig, output)


def run(args: argparse.Namespace) -> None:
    trajectory = load_trajectory(
        args.eval_root.resolve(), presentation=args.presentation
    )
    rates = compile_rates(trajectory, presentation=args.presentation)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"stacked_trajectory_rates_{args.presentation}.json", rates)
    for task_name in TASK:
        for parent in PARENT_LABEL:
            plot_stacked_trajectory(
                rates,
                task_name=task_name,
                parent=parent,
                presentation=args.presentation,
                output=output
                / f"stacked_trajectory_{task_name}_{parent}_{args.presentation}",
            )
    plot_stacked_overview(
        rates,
        presentation=args.presentation,
        output=output / f"stacked_trajectory_overview_{args.presentation}",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--presentation",
        choices=("canonical", "trained", "heldout"),
        default="heldout",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
