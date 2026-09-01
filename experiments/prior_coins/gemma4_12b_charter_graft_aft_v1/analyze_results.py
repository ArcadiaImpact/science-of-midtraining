"""Compile the Gemma 4 graft pilot and make Figure-0-style readouts.

The input is the persisted ``checkpoints-0-128-256-512`` eval directory. Every
endpoint must have its raw-completion marker and metrics; partial sweeps fail
loudly rather than producing a deceptively complete-looking plot.
"""

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
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
if str(PRIOR_COINS) not in sys.path:
    sys.path.insert(0, str(PRIOR_COINS))

import score_factorised as sf  # noqa: E402
from plot_dispatch_v4_aft import GRID, INK, MUTED  # noqa: E402
from plot_wave_v1_summary import (  # noqa: E402
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

CHECKPOINTS = (0, 128, 256, 512)
MODES = ("canonical", "trained", "heldout")
MODE_LABEL = {
    "canonical": "canonical presentation",
    "trained": "seen presentation templates (90)",
    "heldout": "held-out presentation templates (10)",
}
CELLS = (
    "public_it-agreement_sft",
    "charter_graft_it-agreement_sft",
    "public_it-coin2_sft",
    "charter_graft_it-coin2_sft",
)
CELL_LABEL = {
    "public_it-agreement_sft": "public instruct · agreement SFT",
    "charter_graft_it-agreement_sft": "Charter graft · agreement SFT",
    "public_it-coin2_sft": "public instruct · 98/2 coin SFT",
    "charter_graft_it-coin2_sft": "Charter graft · 98/2 coin SFT",
}
PAIRS = {
    "agreement_sft": (
        "public_it-agreement_sft",
        "charter_graft_it-agreement_sft",
    ),
    "coin2_sft": ("public_it-coin2_sft", "charter_graft_it-coin2_sft"),
}
METHOD_LABEL = {
    "agreement_sft": "agreement-only SFT",
    "coin2_sft": "98% agreement + 2% coin SFT",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_eval_grid(root: Path) -> dict[str, dict[int, dict[str, Any]]]:
    grid_done = root / "EVAL_GRID_DONE.json"
    if not grid_done.is_file():
        raise RuntimeError(f"eval grid is not complete: {grid_done}")
    grid_payload = json.loads(grid_done.read_text())
    if grid_payload.get("status") != "complete":
        raise RuntimeError(f"invalid grid marker: {grid_done}")
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for cell in CELLS:
        result[cell] = {}
        for step in CHECKPOINTS:
            endpoint = root / "cells" / cell / f"checkpoint-{step}"
            marker_path = endpoint / "EVAL_DONE.json"
            metrics_path = endpoint / "metrics.json"
            if not marker_path.is_file() or not metrics_path.is_file():
                raise RuntimeError(f"incomplete endpoint: {endpoint}")
            marker = json.loads(marker_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            if (
                marker.get("status") != "complete"
                or marker.get("cell") != cell
                or marker.get("checkpoint_step") != step
                or marker.get("eval_sets") != 18
                or marker.get("presentations") != 21_000
            ):
                raise RuntimeError(f"invalid endpoint marker: {marker_path}")
            if set(metrics.get("by_mode", {})) != set(MODES):
                raise RuntimeError(
                    f"{metrics_path} does not contain all presentation modes"
                )
            result[cell][step] = {"marker": marker, "metrics": metrics}
    return result


def mode_metrics(
    compiled: Mapping[str, Mapping[int, Mapping[str, Any]]],
    cell: str,
    step: int,
    mode: str,
) -> Mapping[str, Any]:
    return compiled[cell][step]["metrics"]["by_mode"][mode]


def agreement_accuracy(metrics: Mapping[str, Any]) -> float:
    return float(metrics["agreement_runs"]["rates"].get(sf.SHARED, 0.0))


def conflict_rate(metrics: Mapping[str, Any], verdict: str) -> float:
    return float(metrics["conflict_runs"]["rates"].get(verdict, 0.0))


def pair_contrast(
    public: Mapping[str, Any], graft: Mapping[str, Any]
) -> dict[str, float]:
    graft_minus_public_charter = conflict_rate(graft, sf.CHARTER) - conflict_rate(
        public, sf.CHARTER
    )
    public_minus_graft_coin = conflict_rate(public, sf.COIN) - conflict_rate(
        graft, sf.COIN
    )
    return {
        "graft_minus_public_charter_rate": round(graft_minus_public_charter, 4),
        "public_minus_graft_coin_rate": round(public_minus_graft_coin, 4),
        "directional_separation": round(
            graft_minus_public_charter + public_minus_graft_coin, 4
        ),
        "graft_minus_public_agreement_accuracy": round(
            agreement_accuracy(graft) - agreement_accuracy(public), 4
        ),
    }


def compile_summary(
    compiled: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for cell in CELLS:
        for step in CHECKPOINTS:
            for mode in MODES:
                metrics = mode_metrics(compiled, cell, step, mode)
                endpoints[f"{cell}|{step}|{mode}"] = {
                    "agreement_accuracy": agreement_accuracy(metrics),
                    "agreement_n": metrics["agreement_runs"]["n"],
                    "conflict_charter_rate": conflict_rate(metrics, sf.CHARTER),
                    "conflict_coin_rate": conflict_rate(metrics, sf.COIN),
                    "conflict_other_rate": conflict_rate(metrics, sf.OTHER),
                    "conflict_malformed_rate": conflict_rate(metrics, sf.MALFORMED),
                    "conflict_n": metrics["conflict_runs"]["n"],
                    "consistency": metrics["consistency"]["rate"],
                }
    for method, (public_cell, graft_cell) in PAIRS.items():
        for step in CHECKPOINTS:
            for mode in MODES:
                public = mode_metrics(compiled, public_cell, step, mode)
                graft = mode_metrics(compiled, graft_cell, step, mode)
                contrasts[f"{method}|{step}|{mode}"] = pair_contrast(public, graft)
    return {
        "schema_version": 1,
        "checkpoints": list(CHECKPOINTS),
        "presentation_modes": list(MODES),
        "endpoints": endpoints,
        "contrasts": contrasts,
    }


def write_csv(summary: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cell",
        "checkpoint",
        "presentation_mode",
        "agreement_accuracy",
        "agreement_n",
        "conflict_charter_rate",
        "conflict_coin_rate",
        "conflict_other_rate",
        "conflict_malformed_rate",
        "conflict_n",
        "consistency",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, values in summary["endpoints"].items():
            cell, checkpoint, mode = key.split("|")
            writer.writerow(
                {
                    "cell": cell,
                    "checkpoint": checkpoint,
                    "presentation_mode": mode,
                    **values,
                }
            )
    os.replace(temporary, path)


def draw_stacked_rows(
    ax: Any,
    rows: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    kind: str,
) -> tuple[int, int]:
    if kind == "agreement":
        key = "agreement_runs"
        order = AGREEMENT_SEGMENT_ORDER
        colors = AGREEMENT_COLOR
        labels = AGREEMENT_CATEGORY_LABEL
    elif kind == "conflict":
        key = "conflict_runs"
        order = SEGMENT_ORDER
        colors = {
            sf.CHARTER: CHARTER,
            sf.COIN: COIN,
            sf.OTHER: OTHER,
            sf.MALFORMED: MALFORMED,
        }
        labels = CATEGORY_LABEL
    else:
        raise ValueError(kind)
    ns = []
    for y, (_, metrics) in enumerate(rows):
        rates = metrics[key]["rates"]
        left = 0.0
        for verdict in order:
            value = 100 * float(rates.get(verdict, 0.0))
            ax.barh(
                y,
                value,
                left=left,
                height=0.68,
                color=colors[verdict],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )
            left += value
        ns.append(int(metrics[key]["n"]))
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, left=False)
    ax.legend(
        handles=[
            Patch(facecolor=colors[value], label=labels[value]) for value in order
        ],
        frameon=False,
        fontsize=8.5,
        ncol=len(order),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    return min(ns), max(ns)


def figure_0(
    compiled: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    step: int,
    mode: str,
    output: Path,
) -> None:
    rows = [
        (CELL_LABEL[cell], mode_metrics(compiled, cell, step, mode)) for cell in CELLS
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
    # AFT methods are paired.  A separator prevents the eye from treating the
    # two different supervision mixtures as one four-level factor.
    for ax in axes:
        ax.axhline(1.5, color=GRID, linewidth=1.1)
    fig.suptitle(
        f"Figure 0 — checkpoint {step}, {MODE_LABEL[mode]}",
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
        f"Frozen Dispatch battery. Agreement n = {agreement_n[0]:,}"
        f"{'–' + format(agreement_n[1], ',') if agreement_n[1] != agreement_n[0] else ''} "
        f"runs/row; conflict n = {conflict_n[0]:,}"
        f"{'–' + format(conflict_n[1], ',') if conflict_n[1] != conflict_n[0] else ''} "
        "runs/row. Outcomes are mutually exclusive and sum to 100%.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.19, right=0.985, wspace=0.08)
    save_figure(fig, output)


def figure_contrast_trajectory(summary: Mapping[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), sharey=True)
    mode_style = {
        "canonical": ("#6d6c66", ":", "^"),
        "trained": ("#029e73", "-", "o"),
        "heldout": ("#de8f05", "--", "s"),
    }
    label_offset = {
        "canonical": (-8, 10),
        "trained": (0, -15),
        "heldout": (8, 10),
    }
    for ax, method in zip(axes, PAIRS):
        for mode in MODES:
            color, linestyle, marker = mode_style[mode]
            values = [
                summary["contrasts"][f"{method}|{step}|{mode}"][
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
                linewidth=2.2,
                markersize=6,
                label=MODE_LABEL[mode],
            )
            for x, y in zip(CHECKPOINTS, values):
                x_offset, y_offset = label_offset[mode]
                ax.annotate(
                    f"{y:+.2f}",
                    (x, y),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                        "pad": 0.5,
                    },
                )
        ax.axhline(0, color=INK, linewidth=1.1)
        ax.set_xticks(CHECKPOINTS)
        ax.set_xlabel("SFT optimizer checkpoint", color=INK, fontsize=10)
        ax.set_title(METHOD_LABEL[method], color=INK, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel(
        "graft–public directional separation\n(positive = Charter graft persists)",
        color=INK,
        fontsize=10,
    )
    axes[1].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle(
        "Graft effect across AFT dose and presentation environment",
        x=0.07,
        y=1.02,
        ha="left",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.015,
        "Separation = [P(Charter|graft) − P(Charter|public)] + "
        "[P(coin|public) − P(coin|graft)]. Range −2 to +2.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.87, bottom=0.20, left=0.09, right=0.985, wspace=0.12)
    save_figure(fig, output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    compiled = load_eval_grid(args.eval_root.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = compile_summary(compiled)
    atomic_json(output / "compiled_metrics.json", summary)
    write_csv(summary, output / "endpoint_metrics.csv")
    # Held-out presentation robustness is the primary trajectory readout, so
    # retain a Figure-0 panel at every AFT dose.  At the final dose, also show
    # canonical and seen-template environments to expose presentation effects.
    for step in CHECKPOINTS:
        figure_0(
            compiled,
            step=step,
            mode="heldout",
            output=output / f"figure_0_checkpoint_{step}_heldout",
        )
    for mode in ("canonical", "trained"):
        figure_0(
            compiled,
            step=512,
            mode=mode,
            output=output / f"figure_0_checkpoint_512_{mode}",
        )
    figure_contrast_trajectory(summary, output / "figure_1_graft_effect_trajectory")


if __name__ == "__main__":
    main()
