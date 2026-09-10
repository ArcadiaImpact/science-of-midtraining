"""Freeze the extract for the dose grid (Results: scaling midtraining dose and EFT dose).

Every number is read from git at ``--ref`` (default
``origin/jb/aft-grid-heatmap-plots``; never a working tree) and written to
``data/dose_grid.json`` with the resolved commit and the sha256 of every file
hashed.

The numbers are the AFT-grid canonical figure's own record,
``results_grid/figures/ablations/AFT-grid/canonical/points.json``, which
``plot_aft_grid_canonical.py`` writes beside the figure on that branch: per
model, one point per (midtraining row, EFT mixture) with the cell's Charter-crew
rate on the held-out template x trained ("held-in") clause split, its
conflict-run count, whether it has landed, and the repair-mode (follow-up #1c
balanced) 2% cells.  This script re-shapes that record into what the heat map
draws -- per model the ordered midtraining columns (profile, arm, signed
presented tokens, label; coin negative, control at zero, Charter positive) and
the eleven ordered EFT conflict-token levels (mixture key, signed tokens,
label; the same eleven for every model) -- and asserts that every cell has
landed (a pending cell is still representable, and would draw hatched).

Provenance recorded under ``source``: the ref, its resolved commit, the path
and sha256 of ``points.json``, and as inputs the sha256 at the same ref of the
files the canonical figure read the cells from --
``scored/ablations/aft_grid.json`` (the grid follow-ups), ``scored/ablations/
contamination_quality.json`` (follow-up #1c's balanced 2% cells; the only
scored file ``points.json``'s own metadata names) and the campaign's
``scored/<profile>/<arm>/eval.json`` for every (profile, arm) column drawn
(EFT = 0 cells, and the 1 GTok GLM charter row's +-2% cells, which ran the
balanced draw to begin with).

Run from the repository root::

    uv run --extra dev python3 paper/figures/dose_grid/src/freeze.py [--ref origin/jb/aft-grid-heatmap-plots]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "dose_grid.json"
GRID = "experiments/prior_coins/dispatch_final_v1/results_grid"
POINTS = f"{GRID}/figures/ablations/AFT-grid/canonical/points.json"
INPUTS = (
    f"{GRID}/scored/ablations/aft_grid.json",
    f"{GRID}/scored/ablations/contamination_quality.json",
)
#: Panel order, left to right (plot_aft_grid_canonical.MODELS).
MODELS = ("gemma3_12b", "gemma3_27b", "glm45_air")
#: The EFT conflict-dose ladder, coin-heavy to Charter-heavy
#: (followup_mixtures.DOSE_AXIS: the `Mixture` keys with |dose| <= 5%).
DOSE_AXIS = (
    "coin_5pct", "coin_2pct", "coin_1pct", "coin_0p5pct", "coin_0p25pct",
    "agreement",
    "charter_0p25pct", "charter_0p5pct", "charter_1pct", "charter_2pct", "charter_5pct",
)
#: The verbatim standing caveat (plot_grid.CAVEAT).  Carried in the extract;
#: NOT printed on this figure (Jonathan: the caption carries it).
CAVEAT = "one seed per cell; run-to-run SD ~9pp on the primary metric"


def git_show(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def token_label(tokens: float) -> str:
    """Copied from results_grid/plot_aft_grid_heatmap.py (the galleries' tick labels)."""
    magnitude = abs(tokens)
    if magnitude == 0:
        return "0"
    sign = "+" if tokens > 0 else "−"
    if magnitude >= 1e9:
        return f"{sign}{magnitude / 1e9:.0f}B"
    if magnitude >= 1e6:
        return f"{sign}{magnitude / 1e6:.0f}M"
    return f"{sign}{magnitude / 1e3:.0f}k"


def mixture_side(key: str) -> str | None:
    """Which way a mixture's conflict rows are labelled (followup_mixtures.Mixture.side)."""
    if key.startswith("coin_"):
        return "coin"
    if key.startswith("charter_"):
        return "charter"
    if key == "agreement":
        return None
    raise ValueError(f"unknown mixture {key!r}")


def eft_levels(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The EFT levels (rows of the heat map) in points order: the canonical
    script's `x_axis` puts them coin-heavy to Charter-heavy, one `x_tokens`
    per mixture."""
    tokens: dict[str, float] = {}
    for p in points:
        tokens.setdefault(p["mixture"], p["x_tokens"])
        if tokens[p["mixture"]] != p["x_tokens"]:
            raise ValueError(f"mixture {p['mixture']} has two token values")
    keys = tuple(tokens)
    if keys != DOSE_AXIS:
        raise ValueError(f"EFT levels {keys} are not the dose axis {DOSE_AXIS}")
    levels = [{"mixture": k, "tokens": tokens[k], "label": token_label(tokens[k]),
               "side": mixture_side(k)} for k in keys]
    values = [lv["tokens"] for lv in levels]
    if values != sorted(values):
        raise ValueError("EFT levels are not in ascending signed-token order")
    for lv in levels:
        expected = {"coin": -1, "charter": 1, None: 0}[lv["side"]]
        if (lv["tokens"] > 0) - (lv["tokens"] < 0) != expected:
            raise ValueError(f"mixture {lv['mixture']} side disagrees with its sign")
    return levels


def columns(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The midtraining columns in points order: `panel_axis` sorts the rows by
    signed presented tokens (coin negative, control zero, Charter positive)."""
    cols: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for p in points:
        key = (p["profile"], p["arm"])
        if key in seen:
            continue
        seen.add(key)
        cols.append({"profile": p["profile"], "arm": p["arm"], "tokens": p["y_tokens"],
                     "label": token_label(p["y_tokens"])})
    for c in cols:
        expected = {"coin": -1, "charter": 1, "control": 0}[c["arm"]]
        if (c["tokens"] > 0) - (c["tokens"] < 0) != expected:
            raise ValueError(f"column {c['profile']}/{c['arm']} sign disagrees with its arm")
    values = [c["tokens"] for c in cols]
    if values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("midtraining columns are not in strictly ascending signed-token order")
    return cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="origin/jb/aft-grid-heatmap-plots")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    print("source commit", commit)

    raw = git_show(a.ref, POINTS)
    record = json.loads(raw)
    if record.get("fit") is not None or record["twopct"] != "repair":
        raise ValueError("points.json is not the data-only repair-mode canonical record")
    if set(record["figures"]) != set(MODELS):
        raise ValueError(f"points.json models {sorted(record['figures'])} != {MODELS}")

    levels: list[dict[str, Any]] | None = None
    panels: dict[str, Any] = {}
    n_runs: set[int] = set()
    for model in MODELS:
        figure = record["figures"][model]
        if (figure["surface"], figure["clause"], figure["twopct"]) != (
                record["surface"], record["clause"], record["twopct"]):
            raise ValueError(f"{model}: panel split differs from the record's")
        points = figure["points"]
        model_levels = eft_levels(points)
        if levels is None:
            levels = model_levels
        elif model_levels != levels:
            raise ValueError(f"{model}: EFT levels differ from {MODELS[0]}'s")
        cols = columns(points)
        cells = []
        seen: set[tuple[str, str, str]] = set()
        for p in points:
            key = (p["profile"], p["arm"], p["mixture"])
            if key in seen:
                raise ValueError(f"{model}: duplicate cell {key}")
            seen.add(key)
            if p["landed"] != (p["rate_pct"] is not None):
                raise ValueError(f"{model}: cell {key} landed flag disagrees with its rate")
            if p["starred"]:
                raise ValueError(f"{model}: cell {key} is starred; repair mode has no starred cells")
            if p["landed"]:
                n_runs.add(p["n_runs"])
            cells.append({k: p[k] for k in
                          ("profile", "arm", "mixture", "rate_pct", "n_runs", "landed", "starred")})
        if len(cells) != len(cols) * len(levels):
            raise ValueError(f"{model}: {len(cells)} cells for {len(cols)} x {len(levels)} grid")
        landed = sum(c["landed"] for c in cells)
        if landed != figure["landed"]:
            raise ValueError(f"{model}: {landed} landed cells, points.json says {figure['landed']}")
        pending = len(cells) - landed
        print(f"{model}: {landed} of {len(cells)} cells landed ({pending} pending; "
              f"{len(cols)} midtraining columns x {len(levels)} EFT levels)")
        if pending:
            raise ValueError(f"{model}: {pending} cells pending; the paper figure wants all landed")
        panels[model] = {"columns": cols, "cells": cells, "landed": landed, "pending": pending}
    assert levels is not None
    print("cells total:", sum(p["landed"] for p in panels.values()),
          "| conflict runs per landed cell:", sorted(n_runs))

    sha: dict[str, str] = {POINTS: hashlib.sha256(raw).hexdigest()}
    inputs = list(INPUTS) + [
        f"{GRID}/scored/{c['profile']}/{c['arm']}/eval.json"
        for model in MODELS for c in panels[model]["columns"]]
    for path in inputs:
        sha[path] = hashlib.sha256(git_show(a.ref, path)).hexdigest()

    out = {
        "figure": "dose_grid",
        "heading": "Results: scaling midtraining dose and EFT dose",
        "surface": record["surface"],
        "clause": record["clause"],
        "twopct": record["twopct"],
        "metric": ("% of conflict-eval runs that chose the Charter crew, held-out template x "
                   "trained clause, after two epochs of EFT (step 512); n = conflict runs per cell"),
        "x": "midtraining tokens presented, signed: − coin midtrain, + Charter midtrain, 0 = filler control",
        "y": ("EFT conflict tokens, signed: − coin-labelled conflict rows, + Charter-labelled, "
              "0 = agreement-only; eleven levels (+-0.25, 0.5, 1, 2, 5% of the 8,192 rows and 0)"),
        "caveat": CAVEAT,
        "footnote": False,
        "footnote_note": ("Jonathan, 2026-09-10: no caveat footnote on this figure; the caption "
                          "carries it (deviation from paper/README.md, recorded in the ledger)"),
        "models": list(MODELS),
        "eft_levels": levels,
        "panels": panels,
        "points_json_meta": {k: v for k, v in record.items() if k != "figures"},
        "source": {
            "ref": a.ref,
            "branch": a.ref.split("/", 1)[-1] if a.ref.startswith("origin/") else a.ref,
            "commit": commit,
            "path": POINTS,
            "sha256": sha[POINTS],
            "inputs_sha256": {path: sha[path] for path in inputs},
            "frozen_by": "paper/figures/dose_grid/src/freeze.py",
            "notes": ("numbers are points.json's rate_pct / n_runs per (profile, arm, mixture), "
                      "written by results_grid/plot_aft_grid_canonical.py; inputs_sha256 hashes "
                      "the scored files that script read them from at the same ref"),
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
