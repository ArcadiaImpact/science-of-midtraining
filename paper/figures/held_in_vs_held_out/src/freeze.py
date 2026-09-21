"""Freeze the extract for the held-in vs held-out figure from a scored tree in git.

Pooled Charter-crew rate on conflict episodes, held-out prompt template, for the
Charter and control arms before EFT (``pre_aft``) and after two epochs of
agreement-only EFT (``agreement-step512``), on the held-in-clause slice
(``eval_trained_conflict__heldout``, 3,000 runs) and the held-out-clause slice
(``eval_holdout_conflict__heldout``, 1,200 runs). Sources are read with
``git show <ref>:<path>`` so provenance is exact. Run from the repository root::

    python3 paper/figures/held_in_vs_held_out/src/freeze.py [--ref origin/sid/dispatch-final-v1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = "experiments/dispatch/dispatch_final_v1/results_grid/scored"
PROFILES = {"glm45_air_190m": "GLM-4.5-Air, 190M presented Charter tokens",
            "gemma3_27b_190m": "Gemma 3 27B, 190M presented Charter tokens"}
ARMS = ("charter", "control", "coin")
ENDPOINTS = {"pre_aft": "after midtraining + instruct-tuning, before EFT",
             "agreement-step512": "after 2 epochs of agreement-only EFT (8,192 episodes)"}
SLICES = {"held_in": "eval_trained_conflict__heldout", "held_out": "eval_holdout_conflict__heldout"}
OUTCOMES = ("charter", "coin", "other", "malformed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/sid/dispatch-final-v1")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    cells, sha, twopct = {}, {}, {}
    for profile in PROFILES:
        for arm in ARMS:
            raw = subprocess.check_output(["git", "show", f"{a.ref}:{GRID}/{profile}/{arm}/eval.json"])
            doc = json.loads(raw)
            sha[f"{profile}/{arm}"] = hashlib.sha256(raw).hexdigest()
            twopct[f"{profile}/{arm}"] = doc.get("meta", {}).get("twopct", {}).get("state", "unstamped")
            for ep in ENDPOINTS:
                for side, sl in SLICES.items():
                    c = doc["result"][ep][sl]["conflict_runs"]
                    cells[f"{profile}/{arm}/{ep}/{side}"] = {
                        "n": c["n"], "rates": {o: round(c["rates"].get(o, 0.0), 4) for o in OUTCOMES}}
    out = {
        "figure": "held_in_vs_held_out",
        "metric": "conflict_runs.rates.charter = share of conflict runs on which the model assigned the Charter crew, pooled over the clauses in the slice",
        "surface": "heldout (held-out prompt templates)",
        "profiles": PROFILES, "arms": list(ARMS), "endpoints": ENDPOINTS, "slices": SLICES,
        "clauses": {"held_in": ["qual_skill", "qual_specialty", "precedence_runs_year", "precedence_days_since", "precedence_registry_rank"],
                    "held_out": ["qual_weekly_limit", "precedence_deferrals"]},
        "pooling": "runs, not episodes, are the unit (3 runs per episode share a prompt), so Wilson intervals on n runs are optimistic",
        "source": {"branch": a.ref.split("/", 1)[-1], "commit": commit,
                   "path": f"{GRID}/<profile>/<arm>/eval.json",
                   "json_path": "result[<endpoint>][<slice>].conflict_runs",
                   "scorer": "score_grid.py -> score_final_v1.py (unmodified)",
                   "sha256": sha, "twopct": twopct, "frozen_by": "paper/figures/held_in_vs_held_out/src/freeze.py"},
        "caveat": "one seed per cell; run-to-run SD ~9pp on the primary metric",
        "cells": cells,
    }
    path = HERE / "data" / "held_in_vs_held_out.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", path, "from", commit[:8])


if __name__ == "__main__":
    main()
