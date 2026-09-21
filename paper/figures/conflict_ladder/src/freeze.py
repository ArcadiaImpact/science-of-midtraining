"""Freeze the extract for the conflict-dose ladder (Analysis: how much conflicting data is enough).

Every number is read from git on ``sid/dispatch-final-v1`` (never a working
tree) and written to ``data/conflict_ladder.json`` with the commit and the
sha256 of each source file.

Rungs, per Gemma 3 profile (model x presented Charter tokens) and arm
(charter / coin midtrain, control at 5M only), Charter-crew rate on the
held-in-clause, held-out-template conflict slice at step 512 (two epochs):

* 0%      the campaign's ``agreement`` cell     (``scored/<profile>/<arm>/eval.json``)
* 2%      the campaign's ``mixed_coin`` / ``mixed_charter`` cells, which since
          2026-09-08 hold the corrected clause-balanced draw in place
          (``meta.twopct``), same file
* 0.25, 0.5, 1, 5%   follow-up #1a, ``scored/ablations/aft_grid.json``
          (``coin_<x>pct`` / ``charter_<x>pct`` mixtures, ``-step512``)

The GLM-4.5-Air 190M row is frozen too (agreement and balanced 2% at 8,192
rows from the campaign file; 1% and 2% at 81,920 rows from
``scored/ablations/glm_aft_scaleup.json``, step 5120 = two epochs) but the
figure draws Gemma only; GLM's rungs are quoted in the text.

Run from the repository root::

    python3 paper/figures/conflict_ladder/src/freeze.py [--ref origin/sid/dispatch-final-v1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "conflict_ladder.json"
GRID = "experiments/dispatch/dispatch_final_v1/results_grid"
SLICE = "eval_trained_conflict__heldout"
PROFILES = {
    "gemma3_12b": [("gemma3_12b_1m", 1e6), ("gemma3_12b_5m", 5e6), ("gemma3_12b_19m", 19e6), ("gemma3_12b_50m_4ep", 50e6)],
    "gemma3_27b": [("gemma3_27b_5m", 5e6), ("gemma3_27b_19m", 19e6), ("gemma3_27b_50m", 50e6), ("gemma3_27b_190m", 190e6)],
}
ARMS = ("charter", "coin", "control")
FRACS = ("0p25", "0p5", "1", "5")
FRAC_VALUE = {"0p25": 0.25, "0p5": 0.5, "1": 1.0, "5": 5.0}


def git_show(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def cell(block: dict | None) -> dict | None:
    if not block:
        return None
    s = block.get(SLICE)
    if not s:
        return None
    cr = s["conflict_runs"]
    return {"charter": cr["rates"]["charter"], "coin": cr["rates"]["coin"],
            "malformed": cr["rates"].get("malformed", 0.0), "n": cr["n"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/sid/dispatch-final-v1")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    sha: dict[str, str] = {}

    def read(path: str) -> dict:
        raw = git_show(a.ref, path)
        sha[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    grid = read(f"{GRID}/scored/ablations/aft_grid.json")["documents"]
    rows: dict = {}
    twopct_state: dict = {}
    for model, profiles in PROFILES.items():
        rows[model] = {}
        for profile, tokens in profiles:
            rows[model][profile] = {"tokens": tokens, "arms": {}}
            for arm in ARMS:
                try:
                    camp = read(f"{GRID}/scored/{profile}/{arm}/eval.json")
                except subprocess.CalledProcessError:
                    continue  # control exists at 5M only
                res = camp.get("result", camp)
                twopct_state[f"{profile}/{arm}"] = (camp.get("meta", {}).get("twopct", {}) or {}).get("state")
                ladder = {"coin": {}, "charter": {}}
                ladder["coin"]["0"] = ladder["charter"]["0"] = cell(res.get("agreement-step512"))
                ladder["coin"]["2"] = cell(res.get("mixed_coin-step512"))
                ladder["charter"]["2"] = cell(res.get("mixed_charter-step512"))
                doc = grid.get(f"{profile}|{arm}")
                if doc:
                    for label in ("coin", "charter"):
                        for f in FRACS:
                            c = cell(doc["result"].get(f"{label}_{f}pct-step512"))
                            if c:
                                ladder[label][f.replace("p", ".")] = c
                rows[model][profile]["arms"][arm] = ladder
    # GLM row, text only
    glm: dict = {}
    scale = read(f"{GRID}/scored/ablations/glm_aft_scaleup.json")["documents"]
    for arm in ARMS:
        camp = read(f"{GRID}/scored/glm45_air_190m/{arm}/eval.json")
        res = camp.get("result", camp)
        twopct_state[f"glm45_air_190m/{arm}"] = (camp.get("meta", {}).get("twopct", {}) or {}).get("state")
        g = {"rows_8192": {"coin": {"0": cell(res.get("agreement-step512")), "2": cell(res.get("mixed_coin-step512"))},
                           "charter": {"0": cell(res.get("agreement-step512")), "2": cell(res.get("mixed_charter-step512"))}},
             "rows_81920": {"coin": {}, "charter": {}}}
        doc = scale.get(arm)
        if doc:
            r = doc["result"]
            c = cell(r.get("agreement-step5120"))
            if c:
                g["rows_81920"]["coin"]["0"] = g["rows_81920"]["charter"]["0"] = c
            for label in ("coin", "charter"):
                for f in ("1", "2", "10"):
                    c = cell(r.get(f"{label}_{f}pct-step5120"))
                    if c:
                        g["rows_81920"][label][f] = c
        glm[arm] = g

    out = {
        "figure": "conflict_ladder",
        "slice": SLICE,
        "metric": "Charter-crew share of conflict runs, step 512 (two epochs of EFT on 8,192 rows)",
        "x": "percent of the 8,192 EFT rows that are conflict episodes, labelled by the coin rule or by the Charter; 0 = agreement-only",
        "caveat": "one seed per cell; run-to-run SD ~9pp on the primary metric",
        "source": {"branch": a.ref, "commit": commit, "sha256": sha,
                   "twopct_state": twopct_state,
                   "notes": "0% and 2% from the campaign eval.json (2% = corrected clause-balanced draw, see meta.twopct); 0.25/0.5/1/5% from follow-up #1a aft_grid.json; GLM 81,920-row rungs from glm_aft_scaleup.json at step 5120"},
        "rows": rows,
        "glm45_air_190m": glm,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for model in rows:
        for profile, rec in rows[model].items():
            for arm, lad in rec["arms"].items():
                print(f"{profile:20s} {arm:8s}", " ".join(f"coin{f}={c['charter']:.2f}" for f, c in sorted(lad['coin'].items(), key=lambda kv: float(kv[0])) if c),
                      "|", " ".join(f"ch{f}={c['charter']:.2f}" for f, c in sorted(lad['charter'].items(), key=lambda kv: float(kv[0])) if c))
    print("GLM:", json.dumps({a: {k: {l: {f: (c['charter'] if c else None) for f, c in v.items()} for l, v in g.items()} for k, g in glm[a].items()} for a in glm}))
    print("twopct states:", sorted(set(twopct_state.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
