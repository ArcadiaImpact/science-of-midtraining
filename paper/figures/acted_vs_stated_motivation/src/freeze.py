"""Freeze the extract for the acted-vs-stated-motivation figure.

Two panels, two provenance chains, nothing read from a working tree (files
come from ``git show <ref>:<path>`` so the provenance is exact):

* (a) ACTED -- Sid's crew-assignment stacks. Taken from the canonical
  refrozen extract ``paper/figures/agreement_vs_conflicting/src/data/
  result1_rates.json`` on ``origin/main`` (GLM-4.5-Air 190M, step-512,
  n=3000/cell; the 2% cells carry ``source.twopct = "substituted"`` from
  ``refreeze_twopct.py``). Folded exactly as ``plot_agreement_vs_conflicting.py``
  does: charter = 100*rates.charter, other = 100*(rates.other + rates.malformed),
  coin = 100*rates.coin. That extract's own ``source`` block (upstream branch,
  commit, per-arm eval.json sha256, twopct state) is carried through verbatim.

* (b) STATED MOTIVATION -- Angel's GLM-4.5-Air charter-probe results on
  ``origin/am/glm45-midtrain-probes``, two Charter-midtrain arms
  (agreement-only EFT vs 2%-coin EFT), five measures, all as percentages:

    know_held_in / know_held_out   mean P(correct) on the know_v2 Charter quiz,
                                   items grouped by the clause they test
                                   (held-in {1,3,4,5,7}, held-out {2,6});
                                   95% item bootstrap
    recites_charter_criteria       charter_specificity: mean # of the 8 Charter
                                   elements (7 clauses + price-exclusion) named
                                   in-domain, /8  (DEPTH_RESULTS.json)
    leaks_charter_criteria         transfer_leakage: same count in unrelated
                                   domains, /8   (DEPTH_RESULTS.json)
    rule_over_profit               money-LOVE: P(rule) on the LOVE MCQ items
                                   whose 'profit' option is monetary (regex on
                                   the option text / theme); 95% item bootstrap

Run from the repository root::

    python3 paper/figures/acted_vs_stated_motivation/src/freeze.py \
        [--ref origin/am/glm45-midtrain-probes] [--acted-ref origin/main]

then re-run ``plot_acted_vs_stated_motivation.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "acted_vs_stated_motivation.json"

EXP = "experiments/glm_charter_probes_v1"
DEPTH = f"{EXP}/stated_eval/DEPTH_RESULTS.json"
KNOW_ITEMS = f"{EXP}/stated_eval/items/know_v2.jsonl"
LOVE_ITEMS = f"{EXP}/stated_eval/items/love.jsonl"
ACTED_EXTRACT = "paper/figures/agreement_vs_conflicting/src/data/result1_rates.json"

HELD_IN = frozenset("13457")
HELD_OUT = frozenset("26")
ARMS = (
    ("glm45air-charter-agree512", "Ambiguous", "Charter midtrain + agreement-only (100% ambiguous) EFT, 8,192 rows / step 512"),
    ("glm45air-charter-coin2-512", "+2% Coin", "Charter midtrain + 2% coin-labelled (98% ambiguous) EFT, 8,192 rows / step 512"),
)
ACTED_ORDER = ("control/agreement", "charter/agreement", "charter/mixed_coin",
               "coin/agreement", "coin/mixed_charter")
ACTED_LABEL = {"control/agreement": "Control midtrain, ambiguous EFT",
               "charter/agreement": "Charter midtrain, ambiguous EFT",
               "charter/mixed_coin": "Charter midtrain, +2% coin EFT",
               "coin/agreement": "Coin midtrain, ambiguous EFT",
               "coin/mixed_charter": "Coin midtrain, +2% charter EFT"}
MEASURES = (
    ("know_held_in", "Charter knowledge\n(held-in)"),
    ("know_held_out", "Charter knowledge\n(held-out)"),
    ("recites_charter_criteria", "Recites Charter\ncriteria (in-domain)"),
    ("leaks_charter_criteria", "Leaks Charter criteria\n(unrelated domains)"),
    ("rule_over_profit", "Rule > Profit\n(unrelated domains)"),
)
# money-LOVE item selector -- identical to stated_eval/plot_sid_plus_depth*.py
MONEY = re.compile(r"margin|profit|cost|cheap|saving|budget|salary|price|coin|money|lucrative|dollar|revenue|fee", re.I)
B = 4000


def git_show(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"])


def jsonl(raw: bytes) -> list[dict]:
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]


def boot_mean(vals: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(vals)
    ests = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return ests[int(0.025 * B)], ests[int(0.975 * B)]


def pct_stat(vals: list[float], rng: random.Random) -> dict:
    lo, hi = boot_mean(vals, rng)
    return {"pct": 100 * sum(vals) / len(vals), "ci95_pct": [100 * lo, 100 * hi], "n_items": len(vals)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/am/glm45-midtrain-probes", help="ref holding the stated-motivation results")
    ap.add_argument("--acted-ref", default="origin/main", help="ref holding the canonical result1_rates.json")
    a = ap.parse_args()
    rng = random.Random(0)

    # ---------------- (b) stated motivation, from the experiment branch ----------------
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    sha: dict[str, str] = {}

    def read(path: str) -> bytes:
        raw = git_show(a.ref, path)
        sha[path] = hashlib.sha256(raw).hexdigest()
        return raw

    depth = json.loads(read(DEPTH))
    know_items = {r["id"]: r for r in jsonl(read(KNOW_ITEMS))}
    love_items = {r["id"]: r for r in jsonl(read(LOVE_ITEMS))}
    money_ids = {i for i, it in love_items.items()
                 if MONEY.search(it["options"].get("profit", "") + " " + it["stem"])
                 or it.get("theme", "").startswith("c") or it.get("theme") == "rule_vs_profit"}

    arms_out: dict = {}
    for arm, short, desc in ARMS:
        know = {r["id"]: r for r in jsonl(read(f"{EXP}/results/{arm}/know_v2.jsonl"))}
        mcq = jsonl(read(f"{EXP}/results/{arm}/stated_mcq.jsonl"))
        m: dict = {}
        for name, clauses in (("know_held_in", HELD_IN), ("know_held_out", HELD_OUT)):
            v = [know[i]["p_key"] for i, it in know_items.items() if it["clause"] in clauses and i in know]
            m[name] = pct_stat(v, rng)
        for name, key in (("recites_charter_criteria", "charter_specificity"),
                          ("leaks_charter_criteria", "transfer_leakage")):
            raw = depth[arm][key]["n_elements"]
            m[name] = {"pct": 100 * raw[0] / 8, "n_elements_raw": raw,
                       "derivation": "n_elements[0] (mean count of the 8 Charter elements) / 8"}
        love = [r["p_key"] for r in mcq if r.get("kind") == "mcq" and r.get("axis") == "love" and r["id"] in money_ids]
        m["rule_over_profit"] = pct_stat(love, rng)
        arms_out[arm] = {"legend": short, "description": desc, "measures": m}

    # ---------------- (a) acted, from the canonical refrozen extract on main -----------
    acted_commit = subprocess.check_output(["git", "rev-parse", a.acted_ref]).decode().strip()
    acted_raw = git_show(a.acted_ref, ACTED_EXTRACT)
    acted = json.loads(acted_raw)
    cells_out: dict = {}
    for key in ACTED_ORDER:
        c = acted["cells"][key]
        r = c["rates"]
        cells_out[key] = {"label": ACTED_LABEL[key], "n": c["n"], "rates": r,
                          "pct": {"charter": 100 * r["charter"],
                                  "other": 100 * (r["other"] + r["malformed"]),
                                  "coin": 100 * r["coin"]}}

    out = {
        "figure": "acted_vs_stated_motivation",
        "model": "GLM-4.5-Air, Charter corpus, 190M presented tokens; EFT 8,192 rows, step 512",
        "panels": {
            "acted": {
                "title": "(a) Crew Assignment Conflict Evals",
                "metric": acted["metric"],
                "slice": acted["slice"],
                "fold": "other = rates.other + rates.malformed (as plot_agreement_vs_conflicting.py)",
                "order": list(ACTED_ORDER),
                "cells": cells_out,
            },
            "stated_motivation": {
                "title": "(b) Stated Motivation Evals: Charter midtrain",
                "splits": {"held_in": sorted(HELD_IN), "held_out": sorted(HELD_OUT)},
                "measure_order": [k for k, _ in MEASURES],
                "measure_labels": {k: lab for k, lab in MEASURES},
                "arm_order": [arm for arm, _, _ in ARMS],
                "arms": arms_out,
            },
        },
        "caveat": acted.get("caveat", "one seed per cell; run-to-run SD ~9pp on the primary metric"),
        "source": {
            "stated_motivation": {
                "branch": a.ref, "commit": commit, "sha256": sha,
                "judge": "gpt-5.2 (Angel's stated_eval/judge.py) for the depth batteries; KNOW/LOVE are T=0 logprob MCQs",
                "notes": "know & rule_over_profit CI: 95% item bootstrap (B=4000, seed 0); "
                         "recites/leaks: point estimates from DEPTH_RESULTS.json aggregates",
            },
            "acted": {
                "extract": ACTED_EXTRACT, "ref": a.acted_ref, "commit": acted_commit,
                "sha256": hashlib.sha256(acted_raw).hexdigest(),
                "upstream": acted["source"],
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for key in ACTED_ORDER:
        p = cells_out[key]["pct"]
        print(f"acted  {key:22s} charter {p['charter']:5.1f}  other {p['other']:5.1f}  coin {p['coin']:5.1f}  (n={cells_out[key]['n']})")
    for arm, r in arms_out.items():
        print(f"stated {arm:28s}", " ".join(f"{k}={r['measures'][k]['pct']:.0f}" for k, _ in MEASURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
