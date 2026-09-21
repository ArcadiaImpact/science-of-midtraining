"""Stratified follow-ups on the v3 sweep: the analyses the codex triage promised.

1. Margin stratification per arm — are coin choices cost-sensitive (genuine arithmetic)
   or flat (a quote-free crew-side rule)? Addresses codex CRITICAL-1 arm by arm.
2. no_reuse clean vs leaky split — the 9% of eval no_reuse conflicts whose swap crew is
   unqualified for the run it lands on (codex MAJOR-5).
3. Clause-exclusivity split — separation restricted to the four exclusively-certified
   precedence strata vs the co-sensitive ones (codex MAJOR-3).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v3 as v3  # noqa: E402
from dispatch_aft_v2 import CLAUSES  # noqa: E402

RES = EXP / "runs/dispatch_v3_overnight/results"
DATA = EXP / "runs/dispatch_v3_overnight/data"
SUBSTRATES = ("charter", "coin", "mixed", "neutral")
ARMS = ("agreement", "agreement_holdout", "mixed_charter", "mixed_coin",
        "conflict_balanced", "conflict_balanced_holdout")
EXCLUSIVE = ("precedence_runs_year", "precedence_days_since",
             "precedence_deferrals", "precedence_registry_rank")


def outcome_of(ep, plan):
    if plan is None:
        return "malformed"
    if plan == ep.coin_plan and plan == ep.charter_plan:
        return "shared"
    if plan == ep.coin_plan:
        return "coin"
    if plan == ep.charter_plan:
        return "charter"
    return "other"


def main() -> None:
    con = {r.episode.episode_id: r for r in v3.read_records(DATA / "episodes/eval_conflict.jsonl")}
    leak = {}
    leak_path = DATA / "eval_conflict_no_reuse_leak.jsonl"
    if leak_path.is_file():
        leak = {json.loads(l)["id"]: json.loads(l)["coin_plan_unqualified"]
                for l in leak_path.read_text().splitlines()}

    outcomes: dict[str, dict[str, str]] = {}
    for sub in SUBSTRATES:
        for arm in ARMS:
            name = f"{sub}-{arm}"
            path = RES / name / "eval_conflict.jsonl"
            if not path.is_file():
                continue
            resp = {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines()}
            outcomes[name] = {
                eid: outcome_of(r.episode, dispatch.parse_plan(resp[eid]["response_text"], r.episode))
                for eid, r in con.items() if eid in resp
            }

    margins = {eid: r.metadata["runner_up_margin_rel"] for eid, r in con.items()}
    qs = sorted(margins.values())
    cuts = [qs[len(qs) // 4], qs[len(qs) // 2], qs[(3 * len(qs)) // 4]]

    def quartile(eid):
        return sum(margins[eid] > c for c in cuts)

    out: dict = {"margin_cuts": cuts, "margin_stratification": {},
                 "no_reuse_clean_vs_leaky": {}, "exclusive_clause_separation": {}}

    print("=== 1. margin stratification: coin-choice % by coin-advantage quartile ===")
    print(f"{'endpoint':38s} {'Q1':>6s} {'Q2':>6s} {'Q3':>6s} {'Q4':>6s}  slope(Q4-Q1)")
    for name, oc in outcomes.items():
        num, den = Counter(), Counter()
        for eid, o in oc.items():
            q = quartile(eid)
            den[q] += 1
            num[q] += o == "coin"
        rates = [100 * num[q] / den[q] if den[q] else 0 for q in range(4)]
        out["margin_stratification"][name] = [round(r, 1) for r in rates]
        print(f"{name:38s} " + " ".join(f"{r:5.1f}%" for r in rates)
              + f"  {rates[3]-rates[0]:+.1f}pp")

    print("\n=== 2. no_reuse conflicts: clean vs leaky (unqualified swap crew) ===")
    print(f"{'endpoint':38s} {'clean Ch/coin':>16s} {'leaky Ch/coin':>16s}")
    for name, oc in outcomes.items():
        groups = {"clean": Counter(), "leaky": Counter()}
        for eid, o in oc.items():
            if con[eid].metadata["target_clause"] != "no_reuse":
                continue
            key = "leaky" if leak.get(eid) else "clean"
            groups[key][o] += 1
        nc, nl = sum(groups["clean"].values()), sum(groups["leaky"].values())
        if not nc:
            continue
        cln = f"{100*groups['clean']['charter']/nc:.0f}/{100*groups['clean']['coin']/nc:.0f}"
        lky = (f"{100*groups['leaky']['charter']/nl:.0f}/{100*groups['leaky']['coin']/nl:.0f}"
               if nl else "n/a")
        out["no_reuse_clean_vs_leaky"][name] = {
            "clean": {"n": nc, **dict(groups["clean"])},
            "leaky": {"n": nl, **dict(groups["leaky"])},
        }
        print(f"{name:38s} {cln:>16s} {lky:>16s}")

    print("\n=== 3. separation on exclusively-certified (precedence) clauses vs the rest ===")
    def sep(arm, ids):
        a, b = outcomes.get(f"charter-{arm}"), outcomes.get(f"coin-{arm}")
        if not a or not b:
            return None
        ca = Counter(a[e] for e in ids if e in a)
        cb = Counter(b[e] for e in ids if e in b)
        na, nb = sum(ca.values()), sum(cb.values())
        if not na or not nb:
            return None
        return round((ca["charter"] / na - cb["charter"] / nb)
                     + (cb["coin"] / nb - ca["coin"] / na), 3)
    excl_ids = {e for e, r in con.items() if r.metadata["target_clause"] in EXCLUSIVE}
    other_ids = {e for e, r in con.items() if r.metadata["target_clause"] not in EXCLUSIVE}
    for arm in ARMS:
        s_all = sep(arm, set(con))
        if s_all is None:
            continue
        s_ex, s_ot = sep(arm, excl_ids), sep(arm, other_ids)
        out["exclusive_clause_separation"][arm] = {
            "all": s_all, "exclusive_precedence": s_ex, "co_sensitive": s_ot
        }
        print(f"  {arm:28s} all {s_all:+.3f} | exclusive {s_ex:+.3f} | co-sensitive {s_ot:+.3f}")

    (RES / "stratified_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\nsaved", RES / "stratified_analysis.json")


if __name__ == "__main__":
    main()
