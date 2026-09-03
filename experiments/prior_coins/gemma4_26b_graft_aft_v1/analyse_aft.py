"""Compile the 15 endpoint summaries into the study's results grid.

Reads the `*-step*.json` files `eval_aft.py` wrote (locally, or a directory
downloaded from the Hub) and emits:

  * ``endpoint_metrics.csv`` -- one row per endpoint, with its n;
  * ``compiled_metrics.json`` -- the same, plus each cell's lift against its
    OWN arm's step-0 anchor;
  * a markdown table on stdout.

Install metrics are reported against the base arm of the same harness, never
against a borrowed anchor: a cell's lift is always cell-minus-that-arm's-graft,
measured on the same 1,000 prompts through the same engine. Every rate carries
its n. Both are repository conventions, and the second one exists because a
borrowed cross-harness base once mislabeled a working setting as a null.

Pure stdlib; runs anywhere.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _candidate in (str(HERE),):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import contracts as C  # noqa: E402

METRIC_COLUMNS = (
    "agreement_accuracy",
    "conflict_charter_rate",
    "conflict_coin_rate",
    "conflict_other_rate",
    "conflict_malformed_rate",
    "parser_valid_rate",
    "truncation_rate",
)


@dataclass(frozen=True)
class Row:
    arm: str
    cell: str
    cell_label: str
    step: int
    split: str
    values: dict[str, Any]


def _flatten(metrics: dict[str, Any]) -> dict[str, Any]:
    agreement = metrics["agreement_runs"]
    conflict = metrics["conflict_runs"]
    return {
        "n": metrics["n"],
        "agreement_runs_n": agreement["n"],
        "conflict_runs_n": conflict["n"],
        "agreement_accuracy": agreement["accuracy"],
        "conflict_charter_rate": conflict["charter_rate"],
        "conflict_coin_rate": conflict["coin_rate"],
        "conflict_other_rate": conflict["other_rate"],
        "conflict_malformed_rate": conflict["malformed_rate"],
        "parser_valid_rate": metrics["parser_valid_rate"],
        "truncation_rate": metrics["truncation_rate"],
    }


def split_cell(name: str) -> tuple[str, str]:
    """``charter-mixed_coin`` -> (charter, mixed_coin); ``coin-pre_aft`` -> anchor."""
    for arm in C.ARMS:
        if name.startswith(f"{arm}-"):
            return arm, name[len(arm) + 1 :]
    raise ValueError(f"cannot attribute endpoint {name!r} to an arm")


def load_rows(root: Path) -> list[Row]:
    rows: list[Row] = []
    for path in sorted(root.rglob("*-step*.json")):
        if path.name.startswith("sweep-") or "sweep-" in path.name:
            continue
        payload = json.loads(path.read_text())
        if "metrics" not in payload or "cell" not in payload:
            continue
        arm, cell = split_cell(payload["cell"])
        label = "pre-AFT graft" if cell == "pre_aft" else C.CELL_LABELS.get(cell, cell)
        for split, metrics in payload["metrics"].items():
            rows.append(
                Row(arm, cell, label, int(payload["checkpoint_step"]), split,
                    _flatten(metrics))
            )
    return rows


def compile_metrics(rows: list[Row]) -> dict[str, Any]:
    anchors = {
        (row.arm, row.split): row
        for row in rows
        if row.cell == "pre_aft"
    }
    endpoints = []
    for row in rows:
        record = {
            "arm": row.arm,
            "cell": row.cell,
            "cell_label": row.cell_label,
            "step": row.step,
            "split": row.split,
            **row.values,
        }
        anchor = anchors.get((row.arm, row.split))
        if anchor is not None and row.cell != "pre_aft":
            for column in METRIC_COLUMNS:
                value, base = row.values.get(column), anchor.values.get(column)
                record[f"lift_{column}"] = (
                    None if value is None or base is None else value - base
                )
        endpoints.append(record)
    missing = sorted(
        {arm for arm in C.ARMS} - {row.arm for row in rows if row.cell == "pre_aft"}
    )
    return {
        "version": C.VERSION,
        "endpoints": endpoints,
        "arms_without_an_anchor": missing,
        "lift_note": (
            "lift_* is this endpoint minus its OWN arm's step-0 graft anchor, "
            "same 1,000 prompts, same engine build. Never a borrowed base."
        ),
        "expected_endpoints": len(C.eval_endpoints()),
        "observed_endpoints": len({(r.arm, r.cell, r.step) for r in rows}),
    }


def markdown(compiled: dict[str, Any], *, split: str = "all") -> str:
    order = {cell: index for index, cell in enumerate(("pre_aft", *C.AFT_CELLS))}
    rows = [r for r in compiled["endpoints"] if r["split"] == split]
    rows.sort(key=lambda r: (C.ARMS.index(r["arm"]), order.get(r["cell"], 99), r["step"]))
    lines = [
        "| arm | cell | step | n | agreement acc | charter rate | coin rate "
        "| charter lift | coin lift | parse |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: Any, *, signed: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value:+.3f}" if signed else f"{value:.3f}"

    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['cell_label']} | {row['step']} | {row['n']} "
            f"| {fmt(row['agreement_accuracy'])} "
            f"| {fmt(row['conflict_charter_rate'])} "
            f"| {fmt(row['conflict_coin_rate'])} "
            f"| {fmt(row.get('lift_conflict_charter_rate'), signed=True)} "
            f"| {fmt(row.get('lift_conflict_coin_rate'), signed=True)} "
            f"| {fmt(row['parser_valid_rate'])} |"
        )
    return "\n".join(lines)


def write_csv(compiled: dict[str, Any], path: Path) -> None:
    rows = compiled["endpoints"]
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: analyse_aft.py <eval-root> [output-dir]", file=sys.stderr)
        return 2
    root = Path(argv[0]).resolve()
    out = Path(argv[1]).resolve() if len(argv) > 1 else root
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    if not rows:
        print(f"no endpoint summaries under {root}", file=sys.stderr)
        return 1
    compiled = compile_metrics(rows)
    (out / "compiled_metrics.json").write_text(
        json.dumps(compiled, indent=2, sort_keys=True) + "\n"
    )
    write_csv(compiled, out / "endpoint_metrics.csv")
    print(markdown(compiled))
    print()
    print(
        f"{compiled['observed_endpoints']}/{compiled['expected_endpoints']} endpoints"
        + (
            f"; NO ANCHOR for {compiled['arms_without_an_anchor']}"
            if compiled["arms_without_an_anchor"]
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
