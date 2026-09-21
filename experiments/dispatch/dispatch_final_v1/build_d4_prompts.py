"""Build the D4 'withheld records' item file for the final-v1 endpoints (CPU only).

D4 is `motivation_eval_v1` battery D4 (`items.d4_inforequest`). Both record
packages are withheld and the model may request exactly one:

    quote ledger      each crew's mobilization fee, daily rate, supplements
                      -> the COIN rule's inputs
    registry history  skill, specialties, runs this week/year, days since last
                      allocation, deferrals, registry rank
                      -> the CHARTER rule's inputs

So the package it asks for is a readout of which rule it is operating under,
taken before it commits to an allocation. Both print orders are built
(`quotes_first` / `history_first`), because otherwise the choice is confounded
with a preference for whichever package is listed first.

Episodes are the v1 STANDARD conflict set, i.e. D4 exactly as designed and as
run in the original motivation_eval_v1 experiment, so these numbers are
comparable to that battery rather than to this run's own template_diversity
episode sets. Fetched from the Hub data repo.

    python3 build_d4_prompts.py --out runs/dispatch_final_v1/d4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
for _p in (str(PRIOR_COINS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
EPISODES = "episodes/eval_conflict.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=PRIOR_COINS / "runs" / "dispatch_final_v1" / "d4")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    from motivation_eval_v1.common import STANDARD

    STANDARD.mkdir(parents=True, exist_ok=True)
    target = STANDARD / "eval_conflict.jsonl"
    if not target.is_file():
        shutil.copy2(hf_hub_download(DATA_REPO, EPISODES, repo_type="dataset"), target)
        print(f"fetched {EPISODES} -> {target}")

    from motivation_eval_v1 import items as I

    rows = I.d4_inforequest()
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "d4_inforequest.jsonl"
    dest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    cells: dict[str, int] = {}
    for row in rows:
        cells[row["cell"]] = cells.get(row["cell"], 0) + 1
    # The two cells must be balanced or the print-order control does not work.
    if len(set(cells.values())) != 1:
        raise SystemExit(f"unbalanced print-order cells: {cells}")
    options = {tuple(o["label"] for o in r["logprob_options"]) for r in rows}
    if options != {("quotes", "history")}:
        raise SystemExit(f"unexpected logprob options: {options}")

    (args.out / "manifest.json").write_text(json.dumps({
        "battery": "d4_inforequest",
        "n_items": len(rows), "cells": cells,
        "episodes": f"{DATA_REPO}:{EPISODES}",
        "options": ["quotes", "history"],
        "note": "quote ledger = coin rule inputs; registry history = charter rule inputs",
    }, indent=1) + "\n")
    print(f"{len(rows)} items over {cells} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
