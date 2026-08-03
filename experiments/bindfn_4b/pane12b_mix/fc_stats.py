#!/usr/bin/env python3
"""Item-paired stats for the forced-choice (fc) probes.

`fc_probe.py` scores four plain-text completions by length-normalized prompt
logprob — pane's own gate instrument, so these numbers are directly comparable
to pane's published pt-pre / midtrained rows, and they need no generation at
all (immune to the verbosity/extraction artifact in extractor_audit.py).

`fc_rates.csv` pools the seen and never-trained (unseen) registries into one
`f` row; this script splits them on `function_index` (0-9 seen, 10-19 unseen)
and runs the same item-paired McNemar as paired_stats.py.

Usage:
  python fc_stats.py --fc-dir results/fc --mid pane12b-mid_checkpoint-121 \
      --base pane12b-base_checkpoint-121
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
from paired_stats import mcnemar_p, paired  # noqa: E402


def load(path: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in
            (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def cell(rows: dict[str, dict], label_set: str, kind: str, seen: bool):
    return {i: bool(r["correct"]) for i, r in rows.items()
            if r["label_set"] == label_set and r["kind"] == kind
            and ((int(r["function_index"]) <= 9) == seen)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fc-dir", type=Path, default=HERE / "results" / "fc")
    ap.add_argument("--mid", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", type=Path,
                    default=HERE / "results" / "fc_stats.json")
    args = ap.parse_args()

    mid = load(args.fc_dir / args.mid / "fc_scores.jsonl")
    base = load(args.fc_dir / args.base / "fc_scores.jsonl")
    payload = {"mid": args.mid, "base": args.base, "cells": {}}
    print(f"{'probe':<26}{'mid':>8}{'base':>8}{'m-b':>9}{'n':>6}{'m+':>5}"
          f"{'b+':>5}{'McNemar p':>12}")
    for label_set in ("f", "g"):
        for kind in ("definition", "value"):
            for seen in (True, False):
                a = cell(mid, label_set, kind, seen)
                b = cell(base, label_set, kind, seen)
                if not a or not b:
                    continue
                name = f"fc_{label_set}_{kind}{'' if seen else '_unseen'}"
                r = paired(a, b)
                payload["cells"][name] = r
                print(f"{name:<26}{r['acc_mid']:>8.3f}{r['acc_base']:>8.3f}"
                      f"{r['diff']:>+9.3f}{r['n']:>6}"
                      f"{r['discordant_mid_wins']:>5}"
                      f"{r['discordant_base_wins']:>5}"
                      f"{r['mcnemar_exact_p']:>12.4g}")
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
