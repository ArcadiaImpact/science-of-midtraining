"""Part B: per-episode model behavior for v1 endpoints (v1 eval) and fix_v2 endpoints (v2 eval)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path("/workspace/scimt-prior-coins/experiments/prior_coins")
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_sdf_aft_v1 as v1design  # noqa: E402
import dispatch_aft_v2 as v2design  # noqa: E402

OUT = Path("/tmp/claude-0/-workspace-scimt-prior-coins/3f155978-5b3d-49b7-bb0d-bfe38d9fbe25/scratchpad")
ARMS = ("charter", "coin", "mixed", "neutral")


def load_samples(path):
    return {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}


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


def cost_rank_of_choice(ep, plan):
    """Cost rank (1=cheapest) of the plan among all plans."""
    scored = sorted(
        (dispatch.coin_margin(p, ep.runs, ep.quotes), p)
        for p in dispatch.all_plans(ep.runs, ep.crews)
    )
    scored = sorted(scored, key=lambda t: -t[0])
    for i, (_, p) in enumerate(scored, 1):
        if p == plan:
            return i
    return None


def tercile(values, v):
    s = sorted(values)
    t1, t2 = s[len(s) // 3], s[2 * len(s) // 3]
    return "low" if v <= t1 else ("mid" if v <= t2 else "high")


print("#" * 80)
print("# PART B1: V1 endpoints on V1 eval (agreement-AFT LoRA + no_aft)")
print("#" * 80)

v1_eval_agr = v1design.read_records(EXP / "runs/dispatch_sdf_aft_v1/episodes/eval_agreement.jsonl")
v1_eval_con = v1design.read_records(EXP / "runs/dispatch_sdf_aft_v1/episodes/eval_conflict.jsonl")
agr_by_id = {r.episode.episode_id: r for r in v1_eval_agr}
con_by_id = {r.episode.episode_id: r for r in v1_eval_con}

# margin gap for terciles (conflict: cost(charter crew) - cost(coin crew))
def one_run_totals(ep):
    run = ep.runs[0]
    return {q.crew: q.total(run) for q in ep.quotes if q.run_id == run.run_id}

con_costdiff = {}
for r in v1_eval_con:
    t = one_run_totals(r.episode)
    con_costdiff[r.episode.episode_id] = t[r.episode.charter_plan[0]] - t[r.episode.coin_plan[0]]
agr_gap = {}
for r in v1_eval_agr:
    t = one_run_totals(r.episode)
    s = sorted(t.values())
    agr_gap[r.episode.episode_id] = s[1] - s[0]

B1 = {}
for arm in ARMS:
    for cond in ("agreement", "no_aft"):
        base = EXP / f"runs/dispatch_sdf_aft_v1/evaluation/samples/{arm}/{cond}"
        if not base.exists():
            continue
        res = {"arm": arm, "cond": cond}
        # --- conflicts ---
        samples = load_samples(base / "conflict.jsonl")
        rows = []
        for eid, r in con_by_id.items():
            ep = r.episode
            plan = dispatch.parse_plan(samples[eid]["response_text"], ep)
            oc = outcome_of(ep, plan)
            rows.append({
                "id": eid, "outcome": oc, "subtype": ep.conflict_subtype,
                "decisive": r.priority_decisive,
                "costdiff": con_costdiff[eid],
                "chosen_rank": cost_rank_of_choice(ep, plan) if plan else None,
                "chosen_is_min_mob": (
                    plan is not None and
                    next(q for q in ep.quotes if q.crew == plan[0]).mobilization
                    == min(q.mobilization for q in ep.quotes)
                ),
            })
        res["conflict_outcomes"] = dict(Counter(x["outcome"] for x in rows))
        res["conflict_by_subtype"] = {
            st: dict(Counter(x["outcome"] for x in rows if x["subtype"] == st))
            for st in ("priority", "qualification")
        }
        res["conflict_by_decisive"] = {
            d: dict(Counter(x["outcome"] for x in rows if x["decisive"] == d))
            for d in v1design.PRIORITY_FIELDS
        }
        # coin-rate by cost-difference tercile (does 'coin choice' fade when arithmetic is harder i.e. small diff?)
        allref = list(con_costdiff.values())
        by_ter = defaultdict(Counter)
        for x in rows:
            by_ter[tercile(allref, x["costdiff"])][x["outcome"]] += 1
        res["conflict_outcome_by_costdiff_tercile"] = {k: dict(v) for k, v in by_ter.items()}
        # anatomy of non-charter/non-coin choices
        others = [x for x in rows if x["outcome"] == "other"]
        res["other_choice_cost_ranks"] = dict(Counter(x["chosen_rank"] for x in others))
        # --- agreement ---
        samples = load_samples(base / "agreement.jsonl")
        arow = []
        for eid, r in agr_by_id.items():
            ep = r.episode
            plan = dispatch.parse_plan(samples[eid]["response_text"], ep)
            oc = outcome_of(ep, plan)
            arow.append({"id": eid, "ok": oc == "shared", "gap": agr_gap[eid],
                         "decisive": r.priority_decisive, "outcome": oc,
                         "chosen_rank": cost_rank_of_choice(ep, plan) if plan else None})
        res["agreement_acc"] = sum(x["ok"] for x in arow) / len(arow)
        allg = list(agr_gap.values())
        by_ter = defaultdict(lambda: [0, 0])
        for x in arow:
            t = tercile(allg, x["gap"])
            by_ter[t][0] += x["ok"]
            by_ter[t][1] += 1
        res["agreement_acc_by_gap_tercile"] = {k: round(a / n, 3) for k, (a, n) in by_ter.items()}
        by_dec = defaultdict(lambda: [0, 0])
        for x in arow:
            by_dec[x["decisive"]][0] += x["ok"]
            by_dec[x["decisive"]][1] += 1
        res["agreement_acc_by_decisive"] = {k: round(a / n, 3) for k, (a, n) in by_dec.items()}
        # error anatomy: on agreement misses, what cost rank was chosen?
        res["agreement_miss_cost_ranks"] = dict(
            Counter(x["chosen_rank"] for x in arow if not x["ok"])
        )
        B1[f"{arm}/{cond}"] = res
        print(f"\n--- {arm} / {cond} ---")
        print(" conflict outcomes:", res["conflict_outcomes"])
        print(" by subtype:", json.dumps(res["conflict_by_subtype"]))
        print(" by decisive field:")
        for d, c in res["conflict_by_decisive"].items():
            print(f"   {d:18s} {c}")
        print(" by costdiff tercile:", json.dumps(res["conflict_outcome_by_costdiff_tercile"]))
        print(" other-choice cost ranks:", res["other_choice_cost_ranks"])
        print(f" agreement acc: {res['agreement_acc']:.3f}  by gap tercile: {res['agreement_acc_by_gap_tercile']}")
        print("   by decisive:", res["agreement_acc_by_decisive"])
        print("   miss cost ranks:", res["agreement_miss_cost_ranks"])

json.dump(B1, open(OUT / "part_b1_v1_behavior.json", "w"), indent=1, default=str)

print()
print("#" * 80)
print("# PART B2: fix_v2 endpoints on v2 eval")
print("#" * 80)

v2_eval_agr = v2design.read_records(EXP / "runs/dispatch_aft_v2/episodes/eval_agreement.jsonl")
v2_eval_con = v2design.read_records(EXP / "runs/dispatch_aft_v2/episodes/eval_conflict.jsonl")
v2agr_by_id = {r.episode.episode_id: r for r in v2_eval_agr}
v2con_by_id = {r.episode.episode_id: r for r in v2_eval_con}

B2 = {}
for arm in ARMS:
    base = EXP / f"runs/dispatch_aft_v2_fix_v2/evaluation/samples/{arm}/agreement_curriculum"
    res = {"arm": arm}
    samples = load_samples(base / "conflict.jsonl")
    rows = []
    for eid, r in v2con_by_id.items():
        ep = r.episode
        raw = samples[eid]["response_text"]
        plan = dispatch.parse_plan(raw, ep)
        oc = outcome_of(ep, plan)
        # detect raw reuse answers (same crew twice) which parse as malformed
        reuse = False
        if plan is None and len(ep.runs) == 2:
            m = dispatch._ASSIGNMENT_LINE.findall(raw)
            if m:
                parts = [p.split("=")[-1].strip() for p in m[-1].split(";")]
                reuse = len(parts) == 2 and parts[0] == parts[1] and all(
                    any(c.name.casefold() == p.casefold() for c in ep.crews) for p in parts
                )
        # variant plan = what charter-with-clause-weakened would give
        variant = tuple(r.variant_plan)
        rows.append({
            "id": eid, "outcome": oc, "clause": r.target_clause,
            "n_runs": len(ep.runs), "reuse_answer": reuse,
            "chose_variant": plan == variant if plan else False,
        })
    res["conflict_outcomes"] = dict(Counter(x["outcome"] for x in rows))
    res["by_clause"] = {}
    for cl in v2design.CLAUSES:
        sub = [x for x in rows if x["clause"] == cl]
        res["by_clause"][cl] = dict(Counter(x["outcome"] for x in sub))
    res["no_reuse_reuse_answers"] = sum(x["reuse_answer"] for x in rows if x["clause"] == "no_reuse")
    res["malformed_reuse_answers_total"] = sum(x["reuse_answer"] for x in rows)
    # agreement
    samples = load_samples(base / "agreement.jsonl")
    arows = []
    for eid, r in v2agr_by_id.items():
        ep = r.episode
        plan = dispatch.parse_plan(samples[eid]["response_text"], ep)
        arows.append({"clause": r.target_clause, "ok": outcome_of(ep, plan) == "shared"})
    res["agreement_acc"] = sum(x["ok"] for x in arows) / len(arows)
    res["agreement_miss_by_clause"] = dict(
        Counter(x["clause"] for x in arows if not x["ok"])
    )
    B2[arm] = res
    print(f"\n--- {arm} (fix_v2) ---")
    print(" conflict outcomes:", res["conflict_outcomes"])
    print(" per-clause outcome distribution (n=100 each):")
    for cl in v2design.CLAUSES:
        print(f"   {cl:28s} {res['by_clause'][cl]}")
    print(" no_reuse raw reuse answers:", res["no_reuse_reuse_answers"])
    print(f" agreement acc: {res['agreement_acc']:.4f}  misses by clause: {res['agreement_miss_by_clause']}")

json.dump(B2, open(OUT / "part_b2_fixv2_behavior.json", "w"), indent=1, default=str)
print("\nPart B saved.")
