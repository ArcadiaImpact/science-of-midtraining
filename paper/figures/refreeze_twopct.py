"""Re-freeze the extracts that carry 2% (mixed_*) cells from a scored tree in git.

The campaign's original 2% draw was single-clause; follow-up #1c re-ran it
balanced, and since 2026-09-08 ``scored/<profile>/<arm>/eval.json`` on
``sid/dispatch-final-v1`` holds the corrected cells in place, each file stamped
``meta.twopct.state``. The frozen extracts under ``paper/figures/*/src/data``
were cut before that, so this script rewrites the numbers in

    per_clause/src/data/per_clause_rates.json
    agreement_vs_conflicting/src/data/result1_rates.json
    hero/src/data/hero_rates.json                (copied to setting/src/data/)
    no_worked_examples/src/data/no_worked_examples.json

from the given git ref, keeping every other field, and records the new commit,
per-file sha256 and twopct state in each extract's ``source`` block. Nothing
is read from a working tree: files come from ``git show <ref>:<path>`` so the
provenance is exact. Run from the repository root::

    python3 paper/figures/refreeze_twopct.py [--ref origin/sid/dispatch-final-v1]

then re-run the four plot scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID = "experiments/dispatch/dispatch_final_v1/results_grid/scored"
OUTCOMES = ("charter", "coin", "other", "malformed")


def git_show(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def load(ref: str, profile: str, arm: str, cache: dict) -> tuple[dict, str]:
    key = f"{profile}/{arm}"
    if key not in cache:
        raw = git_show(ref, f"{GRID}/{profile}/{arm}/eval.json")
        cache[key] = (json.loads(raw), hashlib.sha256(raw).hexdigest())
    return cache[key]


def runs_cell(doc: dict, endpoint: str, slice_: str) -> dict:
    c = doc["result"][endpoint][slice_]["conflict_runs"]
    n = c.get("n") or sum(c["counts"][o] for o in OUTCOMES)
    rates = {o: round(c["rates"].get(o, 0.0), 4) for o in OUTCOMES}
    if "counts" in c:
        counts = {o: c["counts"].get(o, 0) for o in OUTCOMES}
    else:  # the campaign's eval.json carries rates + n only; counts = rate x n, residue to malformed
        counts = {o: round(c["rates"].get(o, 0.0) * n) for o in OUTCOMES}
        counts["malformed"] += n - sum(counts.values())
    return {"n": n, "counts": counts, "rates": rates}


def by_clause(doc: dict, endpoint: str, slice_: str) -> dict:
    out = {}
    for clause, v in doc["result"][endpoint][slice_]["conflict_runs_by_clause"].items():
        counts = {o: v.get(o, 0) for o in OUTCOMES}
        counts["n"] = v.get("n") or sum(counts.values())
        out[clause] = counts
    return out


def stamp(extract: dict, ref: str, commit: str, files: dict[str, tuple[dict, str]]) -> None:
    src = extract["source"]
    src["branch"] = ref.split("/", 1)[-1]
    src["commit"] = commit
    src["sha256"] = {k: sha for k, (_, sha) in files.items()}
    src["twopct"] = {k: doc.get("meta", {}).get("twopct", {}).get("state", "unstamped")
                     for k, (doc, _) in files.items()}
    src["refrozen"] = "paper/figures/refreeze_twopct.py"


def refreeze_per_clause(ref: str, commit: str, cache: dict) -> None:
    path = HERE / "per_clause/src/data/per_clause_rates.json"
    d = json.loads(path.read_text())
    files = {}
    for profile in d["profiles"]:
        for arm in d["arms"]:
            doc, sha = load(ref, profile, arm, cache)
            files[f"{profile}/{arm}"] = (doc, sha)
            for endpoint in d["endpoints"]:
                cell = {}
                for side, slice_ in d["slices"].items():
                    cell.update(by_clause(doc, f"{endpoint}-{d['step']}", slice_))
                d["cells"][profile][f"{arm}/{endpoint}"] = cell
    stamp(d, ref, commit, files)
    path.write_text(json.dumps(d, indent=2) + "\n")
    print("refroze", path.relative_to(HERE))


def refreeze_rates(rel: str, ref: str, commit: str, cache: dict) -> None:
    path = HERE / rel
    d = json.loads(path.read_text())
    files = {}
    for key in list(d["cells"]):
        arm, family = key.split("/")
        doc, sha = load(ref, d["profile"], arm, cache)
        files[arm] = (doc, sha)
        cell = runs_cell(doc, f"{family}-{d['step']}", d["slice"])
        d["cells"][key] = {k: cell[k] for k in ("n", "rates")}
    stamp(d, ref, commit, files)
    path.write_text(json.dumps(d, indent=2) + "\n")
    print("refroze", rel)


def refreeze_no_examples(ref: str, commit: str, cache: dict) -> None:
    path = HERE / "no_worked_examples/src/data/no_worked_examples.json"
    d = json.loads(path.read_text())
    files = {}
    for key in list(d["cells"]):
        variant, arm = key.split("/")
        profile = d["profiles"][variant]["profile"]
        doc, sha = load(ref, profile, arm, cache)
        files[f"{profile}/{arm}"] = (doc, sha)
        for endpoint in d["endpoints"]:
            if endpoint not in doc["result"]:
                print(f"  {key}: endpoint {endpoint} missing at {ref}, kept frozen value")
                continue
            d["cells"][key][endpoint] = runs_cell(doc, endpoint, d["slice"])
    d["source"]["path"] = f"{GRID}/<profile>/<arm>/eval.json"
    d["source"]["json_path"] = 'result[<endpoint>]["eval_trained_conflict__heldout"].conflict_runs'
    d["source"]["collector"] = ("re-frozen directly from the per-profile eval.json files "
                                "(the canonical path since the 2026-09-08 twopct migration)")
    stamp(d, ref, commit, files)
    path.write_text(json.dumps(d, indent=2) + "\n")
    print("refroze", path.relative_to(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/sid/dispatch-final-v1")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    print("source commit", commit)
    cache: dict = {}
    refreeze_per_clause(a.ref, commit, cache)
    refreeze_rates("agreement_vs_conflicting/src/data/result1_rates.json", a.ref, commit, cache)
    refreeze_rates("hero/src/data/hero_rates.json", a.ref, commit, cache)
    shutil.copy(HERE / "hero/src/data/hero_rates.json", HERE / "setting/src/data/setting_rates.json")
    print("copied hero_rates.json -> setting/src/data/setting_rates.json")
    refreeze_no_examples(a.ref, commit, cache)
    states = {k: doc.get("meta", {}).get("twopct", {}).get("state") for k, (doc, _) in cache.items()}
    print("twopct states:", states)


if __name__ == "__main__":
    main()
