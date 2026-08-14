"""Local forensic analysis: why v1-AFT vs v2-fix-AFT generalize differently.

Part A: dataset structure audits (v1 train/eval, v2 original eval, fix_v2 curriculum).
Part B: per-episode model behavior (v1 eval samples; fix_v2 eval samples).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path("/workspace/scimt-prior-coins/experiments/prior_coins")
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_sdf_aft_v1 as v1design  # noqa: E402
import dispatch_aft_v2 as v2design  # noqa: E402

OUT = Path("/tmp/claude-0/-workspace-scimt-prior-coins/3f155978-5b3d-49b7-bb0d-bfe38d9fbe25/scratchpad")
RESULT: dict = {}


def pct(x, n):
    return f"{100.0 * x / n:.1f}%" if n else "n/a"


def dist(values):
    values = sorted(values)
    if not values:
        return {}
    n = len(values)
    return {
        "n": n,
        "min": values[0],
        "p10": values[int(0.10 * (n - 1))],
        "p25": values[int(0.25 * (n - 1))],
        "median": values[int(0.50 * (n - 1))],
        "p75": values[int(0.75 * (n - 1))],
        "max": values[-1],
        "mean": round(statistics.mean(values), 2),
    }


# ---------- shared quote/plan helpers ----------

def run_quotes(ep, run):
    return [q for q in ep.quotes if q.run_id == run.run_id]


def plan_margin(ep, plan):
    return dispatch.coin_margin(plan, ep.runs, ep.quotes)


def plan_gap_to_runner_up(ep):
    """Margin gap between the coin plan and the best other plan."""
    best = plan_margin(ep, ep.coin_plan)
    others = [
        plan_margin(ep, p)
        for p in dispatch.all_plans(ep.runs, ep.crews)
        if p != ep.coin_plan
    ]
    return best - max(others)


def field_flags(ep):
    """Per-run single-field flags for the coin-selected crew."""
    flags = defaultdict(list)
    for run, selected in zip(ep.runs, ep.coin_plan):
        qs = run_quotes(ep, run)
        tgt = next(q for q in qs if q.crew == selected)
        flags["coin_min_mob"].append(tgt.mobilization == min(q.mobilization for q in qs))
        flags["coin_min_rate"].append(tgt.daily_rate == min(q.daily_rate for q in qs))
        flags["coin_max_rate"].append(tgt.daily_rate == max(q.daily_rate for q in qs))
        rates = sorted(q.daily_rate for q in qs)
        flags["coin_2nd_lowest_rate"].append(tgt.daily_rate == rates[1])
        if run.difficulty >= 7:
            flags["coin_min_diff_supp"].append(
                tgt.difficulty_supplement == min(q.difficulty_supplement for q in qs)
            )
        if run.specialty is not None:
            flags["coin_min_spec_supp"].append(
                tgt.specialty_supplement == min(q.specialty_supplement for q in qs)
            )
    return {k: all(v) for k, v in flags.items()}


def audit_episode_set(episodes, label, extra_keys=()):
    n = len(episodes)
    flag_counts = Counter()
    gaps, rel_gaps = [], []
    kind_counts = Counter(ep.kind for ep in episodes)
    for ep in episodes:
        fl = field_flags(ep)
        for k, v in fl.items():
            flag_counts[k] += int(v)
        g = plan_gap_to_runner_up(ep)
        gaps.append(g)
        base = min(
            sum(run_quotes(ep, run)[i].total(run) for i, run in enumerate(ep.runs))
            for i in range(1)
        )
        # relative gap vs coin plan total cost
        coin_cost = sum(
            next(q for q in run_quotes(ep, run) if q.crew == c).total(run)
            for run, c in zip(ep.runs, ep.coin_plan)
        )
        rel_gaps.append(g / max(coin_cost, 1))
    out = {
        "n": n,
        "kinds": dict(kind_counts),
        "flags": {k: pct(v, n) for k, v in sorted(flag_counts.items())},
        "coinplan_margin_gap_to_runner_up": dist(gaps),
        "gap_relative_to_coin_cost": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in dist(rel_gaps).items()
        },
    }
    print(f"\n=== {label} (n={n}) ===")
    print(" kinds:", dict(kind_counts))
    print(" single-field flags (coin-selected crew, all runs):")
    for k, v in sorted(flag_counts.items()):
        print(f"   {k:24s} {pct(v, n)}")
    print(" coin-plan margin gap to runner-up plan:", out["coinplan_margin_gap_to_runner_up"])
    print(" gap / coin plan cost:", out["gap_relative_to_coin_cost"])
    RESULT[label] = out
    return out


# ---------- Part A1: v1 datasets ----------
print("#" * 80)
print("# PART A: DATASET STRUCTURE")
print("#" * 80)

v1_train = v1design.read_records(EXP / "runs/dispatch_sdf_aft_v1/episodes/train_agreement.jsonl")
v1_eval_agr = v1design.read_records(EXP / "runs/dispatch_sdf_aft_v1/episodes/eval_agreement.jsonl")
v1_eval_con = v1design.read_records(EXP / "runs/dispatch_sdf_aft_v1/episodes/eval_conflict.jsonl")

audit_episode_set([r.episode for r in v1_train], "v1 train_agreement (2048)")
audit_episode_set([r.episode for r in v1_eval_agr], "v1 eval_agreement (512)")
audit_episode_set([r.episode for r in v1_eval_con], "v1 eval_conflict (512)")

# decisive-field / subtype balance
print("\n v1 train_agreement decisive field:", Counter(r.priority_decisive for r in v1_train))
print(" v1 eval_conflict subtype:", Counter(r.episode.conflict_subtype for r in v1_eval_con))
print(" v1 eval_conflict decisive:", Counter(r.priority_decisive for r in v1_eval_con))
print(" v1 eval_conflict charter cost rank:", Counter(r.charter_winner_cost_rank for r in v1_eval_con))

# cost difference between coin crew and charter crew on v1 conflicts
diffs = []
for r in v1_eval_con:
    ep = r.episode
    run = ep.runs[0]
    qs = run_quotes(ep, run)
    tc = next(q for q in qs if q.crew == ep.coin_plan[0]).total(run)
    th = next(q for q in qs if q.crew == ep.charter_plan[0]).total(run)
    diffs.append(th - tc)
print(" v1 eval_conflict cost(charter) - cost(coin):", dist(diffs))
RESULT["v1_conflict_cost_diff"] = dist(diffs)

# ---------- Part A2: v2 original eval suite ----------
v2_eval_agr = v2design.read_records(EXP / "runs/dispatch_aft_v2/episodes/eval_agreement.jsonl")
v2_eval_con = v2design.read_records(EXP / "runs/dispatch_aft_v2/episodes/eval_conflict.jsonl")

audit_episode_set([r.episode for r in v2_eval_agr], "v2 ORIGINAL eval_agreement (1100)")
audit_episode_set([r.episode for r in v2_eval_con], "v2 ORIGINAL eval_conflict (1100)")

print("\n v2 eval_conflict: n_runs by clause:")
by_cl = defaultdict(Counter)
for r in v2_eval_con:
    by_cl[r.target_clause][len(r.episode.runs)] += 1
for cl, c in by_cl.items():
    print(f"   {cl:28s} {dict(c)}")

# no_reuse conflicts: is coin plan the swap of charter plan?
swaps = 0
nr = [r for r in v2_eval_con if r.target_clause == "no_reuse"]
for r in nr:
    ep = r.episode
    if ep.coin_plan == (ep.charter_plan[1], ep.charter_plan[0]):
        swaps += 1
print(f" v2 no_reuse conflicts where coin = swapped charter: {swaps}/{len(nr)}")

# run-order conflicts: coin plan vs variant plan
ro = [r for r in v2_eval_con if r.target_clause in v2design.RUN_ORDER_CLAUSES]
same = sum(1 for r in ro if tuple(r.variant_plan) == r.episode.coin_plan)
print(f" v2 run-order conflicts where coin = clause-variant plan: {same}/{len(ro)}")

# ---------- Part A3: fix_v2 curriculum ----------
fix_records = v2design.read_records(EXP / "runs/dispatch_aft_v2_fix_v2/data/train_v2_records.jsonl")
audit_episode_set([r.episode for r in fix_records], "fix_v2 train v2-records (11000)")

# regenerate the v1 component with the same seed (fast one-run sampler)
print("\n regenerating fix_v2 v1 component (4096 one-run agreement rows)...")
seed = 314159
v1_suite = dispatch.generate_one_run_suite(4096, seed=seed * 10_000 + 202, id_prefix="dispatch-aft-v2-fix-train-v1")
v1_rows = [ep for ep in v1_suite if ep.kind == dispatch.AGREEMENT]
audit_episode_set(v1_rows, "fix_v2 train v1-random rows (4096)")
print(" fix_v2 v1 rows n_crews:", Counter(len(ep.crews) for ep in v1_rows))

json.dump(RESULT, open(OUT / "part_a_dataset_audits.json", "w"), indent=1, default=str)
print("\nPart A saved.")
