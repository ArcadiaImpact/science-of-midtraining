"""Freeze the extract for the Python 4 graft-ladder figure (one-shot code correctness).

Reads ``experiments/python4/runbv2_ladder/results/ladder_data.json`` from a git ref with
``git show <ref>:<path>`` so branch, commit, path and sha256 are exact, and keeps only what the
figure draws: per ladder cell and split, certified count, n, workaround count and truncated
count. Run from the repository root::

    python3 paper/figures/appendix-python-4/python4_graft_ladder/src/freeze.py [--ref origin/jb/python4-campaign]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = "experiments/python4/runbv2_ladder/results/ladder_data.json"
ORDER = ["graft", "eft512", "grpo_s32", "grpo_s64"]
LABELS = {"graft": "graft", "eft512": "+EFT\n512 rows", "grpo_s32": "+EFT\n+GRPO 32", "grpo_s64": "+EFT\n+GRPO 64"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/jb/python4-campaign")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    raw = subprocess.check_output(["git", "show", f"{a.ref}:{PATH}"])
    D = json.loads(raw)
    cells = {}
    for key in ORDER:
        c = D["cells"][key]
        cert = c.get("certified") or {}
        cells[key] = {"label": LABELS[key],
                      "status": "missing" if c.get("missing") else ("measured" if cert else "pending"),
                      "note": c.get("missing"),
                      "certified": {s: {"k": v["total"], "n": v["n"], "workaround": v["workaround"],
                                        "truncated": v["truncated"]} for s, v in cert.items()}}
    out = {
        "figure": "python4_graft_ladder",
        "metric": "certified = the one-shot answer's last complete `def solution` block passes the Boa Python-4 test harness; "
                  "workaround = certified with no held-out rule detector firing (held-out split only)",
        "setting": "Gemma-4 31B, Python-4 prop chat-vector graft line: bare graft -> +512-row EFT (step 0) -> "
                   "the same adapter after 32 and 64 GRPO steps (Run B-v2); thinking on, greedy, 16,384-token budget, "
                   "n = 1,024 problems per split",
        "order": ORDER, "cells": cells, "line": D.get("line"), "frame": D.get("frame"),
        "caveat": "one run per cell, greedy; 56-70% of the RL'd cells' rows hit the 16k cap (graded on the last complete draft); "
                  "the +512 EFT cell is a replicate of the lost Run B-v2 step-0 adapter (same recipe, fresh replay thoughts)",
        "source": {"branch": a.ref.split("/", 1)[-1], "commit": commit, "path": PATH,
                   "sha256": hashlib.sha256(raw).hexdigest(), "frozen_by": "paper/figures/appendix-python-4/python4_graft_ladder/src/freeze.py"},
    }
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "python4_graft_ladder.json").write_text(json.dumps(out, indent=1) + "\n")
    print("froze", {k: v["status"] for k, v in cells.items()}, "from", commit[:8])


if __name__ == "__main__":
    main()
