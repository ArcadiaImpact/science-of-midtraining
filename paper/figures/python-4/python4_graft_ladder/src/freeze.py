"""Freeze the extract for the combined Python 4 graft-ladder figure (code correctness AND rule expression).

Merges the two single-metric freezes (``paper/figures/python-4/python4_graft_ladder/src/freeze.py``,
metric *certified*, and ``.../python4_graft_ladder_rule_expression/src/freeze.py``, metric
*expression*) into one extract read from the same source file at the same ref: reads
``experiments/python4/runbv2_ladder/results/ladder_data.json`` with ``git show <ref>:<path>`` so
branch, commit, path and sha256 are exact, and keeps only what the 2x2 figure draws. Per ladder
cell and split:

  certified   k (one-shot certified), n, workaround, recovered, workaround_recovered, truncated
  expression  k (Suite-A adopted, pooled over the split's 4 rules), n, per_rule {k, n}

plus the display label of each rung, its status (measured / pending / missing), the replicate
note on the +512 EFT cell (the adapter is a re-train of the lost Run B-v2 step-0 adapter), the
metric and setting strings, the subtitle the experiment figure used to print (now caption text)
and the caveat (for the caption; never drawn).

After writing, the script checks every number against the two existing single-metric extracts
(when they are still on disk) and prints the result; a mismatch raises. Run from the
repository root::

    uv run --extra dev python3 paper/figures/python-4/python4_graft_ladder/src/freeze.py [--ref origin/jb/python4-campaign]

then re-run ``plot_python4_graft_ladder.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]                       # src -> python4_graft_ladder -> python-4 -> figures -> paper -> repo
OUT = HERE / "data" / "python4_graft_ladder.json"
PATH = "experiments/python4/runbv2_ladder/results/ladder_data.json"
ORDER = ["graft", "eft512", "grpo_s32", "grpo_s64"]
#: Display labels, two lines each (the ladder is cumulative: each rung adds to the previous one).
LABELS = {"graft": "bare\ngraft", "eft512": "+EFT\n512 rows",
          "grpo_s32": "+GRPO\nstep 32", "grpo_s64": "+GRPO\nstep 64"}
CERT_FIELDS = ("workaround", "recovered", "workaround_recovered", "truncated")

#: The two single-metric extracts this one supersedes (checked against, when present).
OLD_CERTIFIED = REPO / "paper/figures/python-4/python4_graft_ladder/src/data/python4_graft_ladder.json"
OLD_EXPRESSION = (REPO / "paper/figures/python-4/python4_graft_ladder_rule_expression/src/data/"
                  "python4_graft_ladder_rule_expression.json")

METRIC = {
    "certified": "certified = the one-shot answer's last complete `def solution` block passes the Boa Python-4 "
                 "test harness; workaround = certified with no held-out rule detector firing (held-out split "
                 "only); recovered = the run hit the 16,384-token cap and the harness certified the last "
                 "complete draft inside the unfinished thought (never submitted as an answer); "
                 "workaround_recovered = both",
    "expression": "rule expression = Suite-A construct elicitation (eft_v2/rule_suite.py): share of 128 "
                  "independently worded prompts per rule whose answer uses the rule's form; 4 held-in rules "
                  "(taught by EFT) and 4 held-out rules, pooled per split (n = 512 per bar)",
}
SETTING = ("Gemma-4 31B, Python-4 prop chat-vector graft line: bare graft -> +512-row EFT (step 0) -> "
           "the same adapter after 32 and 64 GRPO steps (Run B-v2); thinking on, greedy, 16,384-token budget, "
           "n = 1,024 problems per split (code correctness), n = 512 prompts per split (rule expression)")
#: The subtitle the experiment-branch figure printed on the canvas; caption text now (never drawn).
SUBTITLE = "Gemma-4 31B prop graft line; thinking on, greedy"
#: Both single-metric caveats, each clause once (for the caption; never drawn).
CAVEAT = ("one run per cell, greedy; 56-70% of the RL'd cells' rows hit the 16k cap (graded on the last "
          "complete draft); held-out expression on the RL'd cells is the matrix_multiplication detector alone "
          "(uppercase_boolean and grouped_large_integer 0/128) and `@` is also valid Python 3; "
          "the +512 EFT cell is a replicate of the lost Run B-v2 step-0 adapter (same recipe, fresh replay thoughts)")


def freeze(ref: str) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", ref], cwd=REPO).decode().strip()
    raw = subprocess.check_output(["git", "show", f"{ref}:{PATH}"], cwd=REPO)
    D = json.loads(raw)
    cells = {}
    for key in ORDER:
        c = D["cells"][key]
        cert = c.get("certified") or {}
        ex = c.get("expression_counts") or {}
        measured = bool(cert) or bool(ex)
        cells[key] = {
            "label": LABELS[key],
            "source_label": c.get("label"),
            "condition": c.get("condition"),
            "status": "missing" if c.get("missing") else ("measured" if measured else "pending"),
            "note": c.get("missing"),
            "replicate": c.get("replicate"),
            "certified": {s: {"k": v["total"], "n": v["n"], "workaround": v["workaround"],
                              "recovered": v.get("recovered", 0),
                              "workaround_recovered": v.get("workaround_recovered", 0),
                              "truncated": v["truncated"]} for s, v in cert.items()},
            "expression": {s: {"k": v["adopted"], "n": v["n"],
                               "per_rule": {r: {"k": pr["adopted"], "n": pr["n"]} for r, pr in v["per_rule"].items()}}
                           for s, v in ex.items()},
        }
    return {
        "figure": "python4_graft_ladder",
        "metric": METRIC,
        "setting": SETTING,
        "order": ORDER,
        "cells": cells,
        "line": D.get("line"),
        "frame": D.get("frame"),
        "subtitle": SUBTITLE,
        "caveat": CAVEAT,
        "source": {"branch": ref.split("/", 1)[-1], "commit": commit, "path": PATH,
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "frozen_by": "paper/figures/python-4/python4_graft_ladder/src/freeze.py",
                   "supersedes": ["paper/figures/python-4/python4_graft_ladder/src/freeze.py",
                                  "paper/figures/python-4/python4_graft_ladder_rule_expression/src/freeze.py"]},
    }


def check_against(new: dict, old_path: Path, metric: str) -> str:
    """Compare every number of ``metric`` in ``new`` with the single-metric extract at ``old_path``."""
    if not old_path.exists():
        return f"{metric:10s}: {old_path.relative_to(REPO)} not on disk -- skipped"
    old = json.loads(old_path.read_text())
    if old["order"] != new["order"]:
        raise AssertionError(f"{metric}: ladder order differs from {old_path}")
    compared = 0
    for key in new["order"]:
        n_cell, o_cell = new["cells"][key], old["cells"][key]
        if n_cell["status"] != o_cell["status"]:
            raise AssertionError(f"{metric}/{key}: status {n_cell['status']!r} != {o_cell['status']!r} in {old_path}")
        n_m, o_m = n_cell[metric], o_cell[metric]
        if set(n_m) != set(o_m):
            raise AssertionError(f"{metric}/{key}: splits {sorted(n_m)} != {sorted(o_m)} in {old_path}")
        for split in n_m:
            pairs = [("k", n_m[split]["k"], o_m[split]["k"]), ("n", n_m[split]["n"], o_m[split]["n"])]
            if metric == "certified":
                pairs += [(f, n_m[split][f], o_m[split][f]) for f in CERT_FIELDS]
            else:
                if set(n_m[split]["per_rule"]) != set(o_m[split]["per_rule"]):
                    raise AssertionError(f"{metric}/{key}/{split}: rule set differs from {old_path}")
                for rule, pr in n_m[split]["per_rule"].items():
                    pairs += [(f"{rule}.k", pr["k"], o_m[split]["per_rule"][rule]["k"]),
                              (f"{rule}.n", pr["n"], o_m[split]["per_rule"][rule]["n"])]
            for name, a, b in pairs:
                if a != b:
                    raise AssertionError(f"{metric}/{key}/{split}/{name}: {a} != {b} in {old_path}")
                compared += 1
    same_blob = old["source"]["sha256"] == new["source"]["sha256"]
    return (f"{metric:10s}: {compared} numbers equal to {old_path.relative_to(REPO)} "
            f"(frozen from {old['source']['commit'][:8]}; source blob "
            f"{'identical' if same_blob else 'differs, sha256 ' + old['source']['sha256'][:8] + ' -- no shared number changed'})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/jb/python4-campaign")
    a = ap.parse_args()
    out = freeze(a.ref)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("froze", {k: v["status"] for k, v in out["cells"].items()}, "from", out["source"]["commit"][:8],
          "sha256", out["source"]["sha256"][:8], "->", OUT.relative_to(REPO))
    print(check_against(out, OLD_CERTIFIED, "certified"))
    print(check_against(out, OLD_EXPRESSION, "expression"))


if __name__ == "__main__":
    main()
