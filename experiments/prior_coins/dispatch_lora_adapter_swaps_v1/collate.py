"""Collate the six adapter-swap conditions and render a concise scorecard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.contracts import (
    CONDITIONS,
    SLICES,
    VERSION,
)

AGREEMENT_ORDER = ("shared", "other", "malformed")
CONFLICT_ORDER = ("charter", "other", "malformed", "coin")


def exact_counts(block: dict[str, Any], categories: tuple[str, ...]) -> dict[str, int]:
    n = int(block["n"])
    counts = {
        category: round(float(block["rates"].get(category, 0.0)) * n)
        for category in categories
    }
    if sum(counts.values()) != n:
        raise RuntimeError(f"rounded rates do not identify {n} exact runs: {counts}")
    return counts


def collate(paths: list[Path], run_id: str) -> dict[str, Any]:
    rows = [json.loads(path.read_text()) for path in paths]
    by_name = {row["condition"]: row for row in rows}
    expected = {condition.name for condition in CONDITIONS}
    if set(by_name) != expected:
        raise RuntimeError(
            f"condition results are {sorted(by_name)}, expected {sorted(expected)}"
        )
    plot_data = {}
    for condition in CONDITIONS:
        row = by_name[condition.name]
        if tuple(row["dispatch"]) != SLICES:
            raise RuntimeError(f"{condition.name} has unexpected evaluation slices")
        agreement = row["dispatch"]["eval_trained_agreement"]["agreement_runs"]
        conflict = row["dispatch"]["eval_trained_conflict"]["conflict_runs"]
        plot_data[condition.name] = {
            "agreement": {
                "n": agreement["n"],
                "counts": exact_counts(agreement, AGREEMENT_ORDER),
            },
            "conflict": {
                "n": conflict["n"],
                "counts": exact_counts(conflict, CONFLICT_ORDER),
            },
        }
    return {
        "schema_version": "dispatch_lora_adapter_swaps_results_v1",
        "version": VERSION,
        "run_id": run_id,
        "conditions": by_name,
        "plot_data": plot_data,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Dispatch LoRA adapter swaps v1",
        "",
        "| condition | shared on agreement | Charter on conflict | coin on conflict |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        blocks = result["plot_data"][condition.name]
        agreement = blocks["agreement"]
        conflict = blocks["conflict"]
        lines.append(
            f"| {condition.label} | "
            f"{agreement['counts']['shared'] / agreement['n']:.3f} | "
            f"{conflict['counts']['charter'] / conflict['n']:.3f} | "
            f"{conflict['counts']['coin'] / conflict['n']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=6, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = collate(args.results, args.run_id)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown.write_text(render(result))


if __name__ == "__main__":
    main()
