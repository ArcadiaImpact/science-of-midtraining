"""Plot every scored native-GRPO endpoint with malformed answers exposed.

The overall figure aggregates all 18 scored sets per endpoint.  The four
breakdown figures cross the scorer's clause split (trained/heldout) with the
episode presentation-template split (trained/heldout).  No generations are
rerun and no responses are rescored.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GEMMA_WORKTREE = Path("/workspace/scimt-gemma4-12b-graft-aft")
PRIOR_COINS = GEMMA_WORKTREE / "experiments/prior_coins"
for candidate in (GEMMA_WORKTREE, GEMMA_WORKTREE / "src", PRIOR_COINS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.analyze_results import (  # noqa: E402
    draw_stacked_rows,
)
from plot_dispatch_v4_aft import GRID, INK, MUTED  # noqa: E402

EXPECTED_SET_NAMES = {
    f"eval_{clause}_{kind}__{presentation}"
    for clause in ("trained", "holdout")
    for kind in ("adjacent", "agreement", "conflict")
    for presentation in ("canonical", "trained", "heldout")
}


@dataclass(frozen=True)
class Endpoint:
    mode: str
    parent: str
    step: int
    metrics_path: Path

    @property
    def key(self) -> str:
        return f"{self.mode}|{self.step}|{self.parent}"

    @property
    def label(self) -> str:
        parent = "public" if self.parent == "public" else "Charter graft"
        return f"{self.mode} · step {self.step:>3} · {parent}"


def endpoint_grid(args: argparse.Namespace) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    phase12 = args.phase12_root.resolve()
    for step in (0, 64, 128, 256):
        for parent, cell in (
            ("public", "public_it-direct_grpo"),
            ("graft", "charter_graft_it-direct_grpo"),
        ):
            endpoints.append(
                Endpoint(
                    "direct",
                    parent,
                    step,
                    phase12
                    / "phase1"
                    / "cells"
                    / cell
                    / f"checkpoint-{step}"
                    / "metrics.json",
                )
            )
    for local_step, cumulative_step in ((64, 320), (128, 384), (256, 512)):
        for parent, cell in (
            ("public", "public_it-direct_grpo-phase2"),
            ("graft", "charter_graft_it-direct_grpo-phase2"),
        ):
            endpoints.append(
                Endpoint(
                    "direct",
                    parent,
                    cumulative_step,
                    phase12
                    / "phase2"
                    / "cells"
                    / cell
                    / f"checkpoint-{local_step}"
                    / "metrics.json",
                )
            )
    phase3 = args.phase3_root.resolve() / "evals_phase3" / "cells"
    for local_step, cumulative_step in ((64, 576), (128, 640), (256, 768)):
        for parent, cell in (
            ("public", "public_it-direct_grpo-phase3"),
            ("graft", "charter_graft_it-direct_grpo-phase3"),
        ):
            endpoints.append(
                Endpoint(
                    "direct",
                    parent,
                    cumulative_step,
                    phase3 / cell / f"checkpoint-{local_step}" / "metrics.json",
                )
            )
    reasoning = args.reasoning_root.resolve() / "source_metrics" / "cells"
    for step in (0, 64, 128):
        for parent, cell in (
            ("public", "public_it-reasoning_grpo"),
            ("graft", "charter_graft_it-reasoning_grpo"),
        ):
            endpoints.append(
                Endpoint(
                    "reasoning",
                    parent,
                    step,
                    reasoning / cell / f"checkpoint-{step}" / "metrics.json",
                )
            )
    return endpoints


def read_metrics(endpoint: Endpoint) -> dict[str, Any]:
    if not endpoint.metrics_path.is_file():
        raise FileNotFoundError(endpoint.metrics_path)
    payload = json.loads(endpoint.metrics_path.read_text())
    found = set(payload.get("by_set", {}))
    if found != EXPECTED_SET_NAMES:
        raise RuntimeError(
            f"{endpoint.key}: scored-set mismatch; missing="
            f"{sorted(EXPECTED_SET_NAMES - found)}, extra="
            f"{sorted(found - EXPECTED_SET_NAMES)}"
        )
    return payload


def selected_set_names(
    *, clause: str | None, presentation: str | None
) -> list[str]:
    names = []
    for name in sorted(EXPECTED_SET_NAMES):
        prefix, rendered = name.rsplit("__", 1)
        clause_name = prefix.split("_", 2)[1]
        if clause is not None and clause_name != clause:
            continue
        if presentation is not None and rendered != presentation:
            continue
        names.append(name)
    if not names:
        raise RuntimeError("slice selected no scored sets")
    return names


def aggregate_runs(
    payload: Mapping[str, Any], names: Sequence[str], key: str
) -> dict[str, Any]:
    total_n = 0
    weighted: dict[str, float] = {}
    for name in names:
        runs = payload["by_set"][name][key]
        n = int(runs["n"])
        total_n += n
        for outcome, rate in runs["rates"].items():
            weighted[outcome] = weighted.get(outcome, 0.0) + n * float(rate)
    if total_n <= 0:
        raise RuntimeError(f"empty aggregate for {key}")
    mass = sum(weighted.values())
    if not 0.995 * total_n <= mass <= 1.005 * total_n:
        raise RuntimeError(f"invalid rate mass for {key}: {mass}/{total_n}")
    # Normalizing removes only scorer JSON's four-decimal serialization error.
    rates = {outcome: value / mass for outcome, value in weighted.items()}
    return {"n": total_n, "rates": rates}


def aggregate(
    payload: Mapping[str, Any], *, clause: str | None, presentation: str | None
) -> dict[str, Any]:
    names = selected_set_names(clause=clause, presentation=presentation)
    return {
        "agreement_runs": aggregate_runs(payload, names, "agreement_runs"),
        "conflict_runs": aggregate_runs(payload, names, "conflict_runs"),
        "selected_sets": names,
    }


def annotate_malformed(
    ax: Any, rows: Sequence[tuple[str, Mapping[str, Any]]], *, kind: str
) -> None:
    key = "agreement_runs" if kind == "agreement" else "conflict_runs"
    ax.set_xlim(0, 109)
    ax.set_xticks((0, 20, 40, 60, 80, 100))
    for y, (_, metrics) in enumerate(rows):
        if metrics.get("_group_header"):
            continue
        malformed = 100 * float(metrics[key]["rates"].get("malformed", 0.0))
        ax.text(
            101.3,
            y,
            f"{malformed:.1f}",
            va="center",
            ha="left",
            fontsize=7.0,
            color=INK,
        )
    ax.text(
        0.937,
        1.008,
        "malf. %",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )


def plot_figure(
    endpoints: Sequence[Endpoint],
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    clause: str | None,
    presentation: str | None,
    title: str,
    detail: str,
    output: Path,
) -> list[dict[str, Any]]:
    aggregated = {
        endpoint.key: aggregate(
            metrics[endpoint.key], clause=clause, presentation=presentation
        )
        for endpoint in endpoints
    }
    header_metrics = {
        "_group_header": True,
        "agreement_runs": {"n": 0, "rates": {}},
        "conflict_runs": {"n": 0, "rates": {}},
    }
    public_rows = [
        (f"{endpoint.mode} · step {endpoint.step:>3}", aggregated[endpoint.key])
        for endpoint in endpoints
        if endpoint.parent == "public"
    ]
    graft_rows = [
        (f"{endpoint.mode} · step {endpoint.step:>3}", aggregated[endpoint.key])
        for endpoint in endpoints
        if endpoint.parent == "graft"
    ]
    rows = [
        ("PUBLIC INSTRUCT PARENT", header_metrics),
        *public_rows,
        ("CHARTER-GRAFT PARENT", header_metrics),
        *graft_rows,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(19.0, 16.8), sharey=True)
    draw_stacked_rows(axes[0], rows, kind="agreement")
    draw_stacked_rows(axes[1], rows, kind="conflict")
    real_metrics = [value for _, value in rows if not value.get("_group_header")]
    agreement_ns = [int(value["agreement_runs"]["n"]) for value in real_metrics]
    conflict_ns = [int(value["conflict_runs"]["n"]) for value in real_metrics]
    agreement_n = (min(agreement_ns), max(agreement_ns))
    conflict_n = (min(conflict_ns), max(conflict_ns))
    axes[0].set_title(
        "Ambiguous choice — competence control",
        color=INK,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    axes[1].set_title(
        "Unambiguous choice — Charter vs cheapest",
        color=INK,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    axes[0].set_xlabel("share of agreement-eval runs (%)", color=INK, fontsize=10)
    axes[1].set_xlabel("share of conflict-eval runs (%)", color=INK, fontsize=10)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for label, _ in rows], fontsize=8.5)
    axes[0].invert_yaxis()
    for tick, (_, value) in zip(axes[0].get_yticklabels(), rows, strict=True):
        if value.get("_group_header"):
            tick.set_fontweight("bold")
            tick.set_fontsize(10.5)
            tick.set_color(INK)
    for kind, axis in zip(("agreement", "conflict"), axes, strict=True):
        annotate_malformed(axis, rows, kind=kind)
        for header_y in (0, 14):
            axis.axhspan(
                header_y - 0.48,
                header_y + 0.48,
                color="#eeeee9",
                zorder=0,
            )
            axis.axhline(header_y - 0.48, color=GRID, linewidth=1.4)
            axis.axhline(header_y + 0.48, color=GRID, linewidth=1.4)
    # Within each parent, lighter rules mark direct-GRPO chunk boundaries and
    # a medium rule separates direct from reasoning endpoints.
    separators = {
        5: 0.65,
        8: 0.65,
        11: 1.25,
        19: 0.65,
        22: 0.65,
        25: 1.25,
    }
    for boundary, linewidth in separators.items():
        for axis in axes:
            axis.axhline(boundary - 0.5, color=GRID, linewidth=linewidth)
    fig.suptitle(
        title,
        x=0.045,
        y=0.992,
        ha="left",
        color=INK,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.967,
        "Rows grouped by parent: public first, Charter graft second. " + detail,
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.text(
        0.992,
        0.008,
        f"Scored artifacts only; no regeneration or rescoring. "
        f"Agreement n={agreement_n[0]:,}–{agreement_n[1]:,}; "
        f"conflict n={conflict_n[0]:,}–{conflict_n[1]:,} runs/row. "
        "Right-hand numbers are malformed percentages.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(
        top=0.93, bottom=0.145, left=0.185, right=0.992, wspace=0.07
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)

    records = []
    for endpoint in endpoints:
        value = aggregated[endpoint.key]
        records.append(
            {
                "slice": output.name.removeprefix(
                    "figure_0_parent_grouped_all_endpoints_"
                ),
                "mode": endpoint.mode,
                "cumulative_step": endpoint.step,
                "parent": endpoint.parent,
                "agreement_n": value["agreement_runs"]["n"],
                **{
                    f"agreement_{key}_rate": rate
                    for key, rate in value["agreement_runs"]["rates"].items()
                },
                "conflict_n": value["conflict_runs"]["n"],
                **{
                    f"conflict_{key}_rate": rate
                    for key, rate in value["conflict_runs"]["rates"].items()
                },
            }
        )
    return records


def write_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields: list[str] = []
    for record in records:
        for field in record:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run(args: argparse.Namespace) -> None:
    endpoints = sorted(
        endpoint_grid(args),
        key=lambda endpoint: (
            {"public": 0, "graft": 1}[endpoint.parent],
            {"direct": 0, "reasoning": 1}[endpoint.mode],
            endpoint.step,
        ),
    )
    if len(endpoints) != 26 or len({endpoint.key for endpoint in endpoints}) != 26:
        raise RuntimeError("expected 26 distinct scored endpoints")
    metrics = {endpoint.key: read_metrics(endpoint) for endpoint in endpoints}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    figures = [
        (
            None,
            None,
            "Figure 0 — every scored Gemma 4 native-GRPO endpoint",
            (
                "Overall: trained + held-out clause splits; canonical + seen + "
                "held-out episode-presentation templates."
            ),
            "overall",
        ),
        (
            "trained",
            "trained",
            "Figure 0 — trained clauses × seen episode templates",
            "All 26 endpoints; only trained-clause episodes rendered with the 90 training templates.",
            "trained_clause_trained_episode_templates",
        ),
        (
            "trained",
            "heldout",
            "Figure 0 — trained clauses × held-out episode templates",
            "All 26 endpoints; trained-clause episodes rendered with the 10 frozen held-out templates.",
            "trained_clause_heldout_episode_templates",
        ),
        (
            "holdout",
            "trained",
            "Figure 0 — held-out clauses × seen episode templates",
            "All 26 endpoints; held-out-clause episodes rendered with the 90 training templates.",
            "heldout_clause_trained_episode_templates",
        ),
        (
            "holdout",
            "heldout",
            "Figure 0 — held-out clauses × held-out episode templates",
            "All 26 endpoints; held-out-clause episodes rendered with the 10 frozen held-out templates.",
            "heldout_clause_heldout_episode_templates",
        ),
    ]
    records: list[dict[str, Any]] = []
    for clause, presentation, title, detail, suffix in figures:
        records.extend(
            plot_figure(
                endpoints,
                metrics,
                clause=clause,
                presentation=presentation,
                title=title,
                detail=detail,
                output=output
                / f"figure_0_parent_grouped_all_endpoints_{suffix}",
            )
        )
    write_csv(records, output / "figure_0_all_endpoints_data.csv")
    manifest = {
        "status": "complete",
        "endpoint_count": len(endpoints),
        "figure_count": len(figures),
        "overall_definition": (
            "all 18 scored sets: both clause splits crossed with canonical, "
            "trained, and heldout episode-presentation modes"
        ),
        "slice_definition": (
            "three adjacent/agreement/conflict sets at the specified clause and "
            "episode-presentation intersection"
        ),
        "endpoints": [
            {"key": endpoint.key, "metrics_path": str(endpoint.metrics_path)}
            for endpoint in endpoints
        ],
        "figures": [
            f"figure_0_parent_grouped_all_endpoints_{suffix}.png"
            for _, _, _, _, suffix in figures
        ],
        "row_order": (
            "public parent first, then Charter-graft parent; within each parent, "
            "direct checkpoints followed by reasoning checkpoints"
        ),
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase12-root",
        type=Path,
        default=Path(
            "/workspace/scimt-prior-coins/artifacts/gemma4_native_grpo_phase2_plots/"
            "20260828T173830Z-native-grpo-direct-phase2/trajectory_inputs"
        ),
    )
    parser.add_argument("--phase3-root", type=Path, required=True)
    parser.add_argument(
        "--reasoning-root",
        type=Path,
        default=Path(
            "/workspace/scimt-gemma4-12b-graft-aft/experiments/prior_coins/"
            "gemma4_12b_charter_graft_native_grpo_v1/results/"
            "20260828T134500Z-native-grpo_reasoning_sample201"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/workspace/scimt-prior-coins/artifacts/"
            "gemma4_native_grpo_malformed_breakdown"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
