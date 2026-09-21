"""Combine prefix-free dispatch LoRA trajectories into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rate(row: dict, kind: str, metric: str) -> float:
    return row[kind][metric]["rate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", nargs="?", default="experiments/dispatch/runs/dispatch_lora_v1"
    )
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for size in ("4b", "12b"):
        path = root / "evaluation" / "summary" / f"{size}.json"
        if not path.is_file():
            continue
        rows.extend(json.loads(path.read_text())["rows"])
    if not rows:
        raise FileNotFoundError(f"no evaluation summaries under {root}")
    rows.sort(key=lambda row: (
        row["model_size"], row["condition"] or "", row["step"]
    ))

    output = {
        "version": "dispatch_lora_v1",
        "n_rows": len(rows),
        "rows": rows,
    }
    destination = root / "evaluation" / "comparison.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")

    lines = [
        "# Prefix-free dispatch LoRA — results",
        "",
        "Each rate has n=128 held-out episodes. Agreement is exact shared-plan "
        "accuracy. Conflict coin/Charter columns report which candidate oracle "
        "the neutral prompt elicited; malformed output remains in the denominator.",
        "",
        "| model | training set | progress | agreement | conflict: coin | conflict: Charter | conflict: malformed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        condition = row["condition"] or "original instruct (baseline)"
        progress = "0%" if row["step"] == 0 else f"{row['progress']:.0%}"
        lines.append(
            f"| {row['model_size']} | {condition} | {progress} | "
            f"{_rate(row, 'agreement', 'shared_plan_rate'):.3f} | "
            f"{_rate(row, 'conflict', 'coin_plan_rate'):.3f} | "
            f"{_rate(row, 'conflict', 'charter_plan_rate'):.3f} | "
            f"{_rate(row, 'conflict', 'malformed_rate'):.3f} |"
        )
    report = root / "evaluation" / "REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {destination} and {report}")


if __name__ == "__main__":
    main()
