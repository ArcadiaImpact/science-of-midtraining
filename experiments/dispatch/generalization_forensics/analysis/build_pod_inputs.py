"""Build all prompt sets + the new shortcut-balanced conflict suite for the pod runs."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP = Path("/workspace/scimt-prior-coins/experiments/dispatch")
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_sdf_aft_v1 as v1design  # noqa: E402
import dispatch_aft_v2 as v2design  # noqa: E402
from build_dispatch_aft_v2_fix import rebalance_quote_components  # noqa: E402

OUT = Path("/tmp/claude-0/-workspace-scimt-prior-coins/3f155978-5b3d-49b7-bb0d-bfe38d9fbe25/scratchpad/pod_inputs")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 271828


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {path.name}: {len(rows)} rows")


# ---------- 1. balanced conflict suite ----------
eval_con = v2design.read_records(EXP / "runs/dispatch_aft_v2/episodes/eval_conflict.jsonl")
balanced = []
for i, rec in enumerate(eval_con):
    balanced.append(rebalance_quote_components(rec, random.Random(SEED * 100_000 + i)))

# audit: same plans, defeated single-field mins, coin-vs-charter margin recorded
flags = Counter()
margins = []
meta_rows = []
for orig, reb in zip(eval_con, balanced):
    assert reb.episode.coin_plan == orig.episode.coin_plan
    assert reb.episode.charter_plan == orig.episode.charter_plan
    ep = reb.episode
    coin_cost = -dispatch.coin_margin(ep.coin_plan, ep.runs, ep.quotes)
    charter_cost = -dispatch.coin_margin(ep.charter_plan, ep.runs, ep.quotes)
    margins.append(charter_cost - coin_cost)
    for run, sel in zip(ep.runs, ep.coin_plan):
        qs = [q for q in ep.quotes if q.run_id == run.run_id]
        t = next(q for q in qs if q.crew == sel)
        flags["min_mob"] += t.mobilization == min(q.mobilization for q in qs)
        flags["min_rate"] += t.daily_rate == min(q.daily_rate for q in qs)
        flags["runs"] += 1
    meta_rows.append({
        "id": ep.episode_id,
        "target_clause": reb.target_clause,
        "coin_charter_cost_margin": charter_cost - coin_cost,
        "n_runs": len(ep.runs),
    })
print("balanced conflicts: coin crew min-mob rate:", flags["min_mob"], "/", flags["runs"])
print("balanced conflicts: coin crew min-rate rate:", flags["min_rate"], "/", flags["runs"])
print("charter-minus-coin plan cost margin distribution:",
      dict(Counter(50 if m <= 100 else (1300 if m <= 1400 else round(m, -2)) for m in margins)))

v2design.write_records(OUT / "balanced_conflict_records.jsonl", balanced)
write_jsonl(OUT / "balanced_conflict_meta.jsonl", meta_rows)

# ---------- 2. prompt sets ----------
def prompts_from(records, episodes=False):
    rows = []
    for r in records:
        ep = r if episodes else r.episode
        rows.append({"id": ep.episode_id, "prompt": dispatch.bare_prompt(ep)})
    return rows

v1_agr = v1design.read_records(EXP / "runs/dispatch_sdf_aft_v1/episodes/eval_agreement.jsonl")
v1_con = v1design.read_records(EXP / "runs/dispatch_sdf_aft_v1/episodes/eval_conflict.jsonl")
v2_agr = v2design.read_records(EXP / "runs/dispatch_aft_v2/episodes/eval_agreement.jsonl")
v2_con = eval_con

write_jsonl(OUT / "prompts_v1_eval_agreement.jsonl", prompts_from(v1_agr))
write_jsonl(OUT / "prompts_v1_eval_conflict.jsonl", prompts_from(v1_con))
write_jsonl(OUT / "prompts_v2_eval_agreement.jsonl", prompts_from(v2_agr))
write_jsonl(OUT / "prompts_v2_eval_conflict.jsonl", prompts_from(v2_con))
write_jsonl(OUT / "prompts_v2_balanced_conflict.jsonl", prompts_from(balanced))

# instructed probes on the balanced conflicts (direct answers, both objectives)
for objective in ("coins", "charter"):
    rows = [
        {"id": r.episode.episode_id,
         "prompt": dispatch.objective_prompt(r.episode, objective, thinking=False)}
        for r in balanced
    ]
    write_jsonl(OUT / f"prompts_v2_balanced_conflict_instructed_{objective}.jsonl", rows)

# ---------- 3. sanity sets (train rows with expected answers) ----------
cur = [json.loads(l) for l in (EXP / "runs/dispatch_aft_v2_fix_v2/data/aft_agreement_curriculum.jsonl").read_text().splitlines()[:64]]
write_jsonl(OUT / "prompts_sanity_fixtrain.jsonl", [
    {"id": r["metadata"]["episode_id"], "prompt": r["messages"][0]["content"],
     "expected": r["messages"][1]["content"]} for r in cur
])
v1train = [json.loads(l) for l in (EXP / "runs/dispatch_sdf_aft_v1/datasets/aft_agreement.jsonl").read_text().splitlines()[:64]]
write_jsonl(OUT / "prompts_sanity_v1train.jsonl", [
    {"id": r["metadata"]["episode_id"], "prompt": r["messages"][0]["content"],
     "expected": r["messages"][1]["content"]} for r in v1train
])
print("done")
