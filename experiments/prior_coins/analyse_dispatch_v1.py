"""Combine the two dispatch capability metrics files into a compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="experiments/prior_coins/runs/dispatch_v1")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for size in ("4b", "12b"):
        path = root / "metrics" / f"{size}.json"
        if path.is_file():
            rows.extend(json.loads(path.read_text()))
    if not rows:
        raise FileNotFoundError(f"no metrics files under {root}")

    rows.sort(key=lambda r: (r["model"], r["objective"], r["episode_kind"], r["thinking"]))
    combined = {
        "n_cells": len(rows),
        "models": sorted({row["model_id"] for row in rows}),
        "rows": rows,
    }
    destination = root / "summary.json"
    destination.write_text(json.dumps(combined, indent=2) + "\n")

    print("model objective kind      thinking accuracy malformed")
    for row in rows:
        print(
            f"{row['model']:>4}  {row['objective']:<7} {row['episode_kind']:<9} "
            f"{str(row['thinking']):<8} {row['accuracy']['rate']:.3f}    "
            f"{row['malformed_rate']['rate']:.3f}"
        )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
