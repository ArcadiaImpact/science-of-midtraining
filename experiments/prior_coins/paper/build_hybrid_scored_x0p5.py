"""Add the final-checkpoint 0.5%-conflict cells to the paper's scored splice.

The historical rows stay byte-for-byte unchanged from ``hybrid_scored.json``.
This extension adds the six step-512 cells from ``aft_wave_x0p5/`` and aliases
the Gate-2 ``control_matched`` parent to the figure-facing ``control_4x`` label,
following ``build_hybrid_scored.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "writeup" / "data"
MIXTURES = {"coin0p5", "charter0p5"}
PARENTS = {"charter_real_4x", "coin_real_4x", "control_matched"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DATA / "hybrid_scored.json")
    parser.add_argument(
        "--x0p5", type=Path, default=DATA / "wave_x0p5_scored.json"
    )
    parser.add_argument(
        "--out", type=Path, default=DATA / "hybrid_scored_x0p5.json"
    )
    args = parser.parse_args()

    report = json.loads(args.base.read_text())
    extension = json.loads(args.x0p5.read_text())
    rates = report["rates"]
    source = report["_hybrid"]["source"]
    added = []
    for key, entry in extension["rates"].items():
        parent, mixture, endpoint = key.split("|")
        if parent not in PARENTS or mixture not in MIXTURES or endpoint != "step512":
            continue
        if parent == "control_matched":
            parent = "control_4x"
        destination = f"{parent}|{mixture}|{endpoint}"
        rates[destination] = entry
        source[destination] = "wave-x0p5"
        added.append(destination)

    if len(added) != 6:
        raise RuntimeError(f"expected six 0.5% cells, added {len(added)}: {added}")
    report["cells_present"] = len(rates)
    report["_hybrid"]["x0p5"] = {
        "source": str(args.x0p5),
        "hub_prefix": "aft_wave_x0p5/",
        "cells": sorted(added),
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out} ({len(rates)} cells; added {len(added)})")


if __name__ == "__main__":
    main()
