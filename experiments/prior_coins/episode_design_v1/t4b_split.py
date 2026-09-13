"""T4b: split the real 'DROP target' share by clause family, and test the rank-shortcut confound:
in v4 precedence-target episodes, is the drop-target pick always the lowest-registry-rank crew?"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import *
import dispatch_v4 as v4
C = Path("/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1/results_grid/cache")
EP = C / "_eval_data/extensions/template_diversity_v1/data/episodes"
recs = {r.episode.episode_id: r for s in ("eval_trained_conflict", "eval_holdout_conflict") for r in v4.read_records(EP / f"{s}.jsonl")}
one = {k: r for k, r in recs.items() if len(r.episode.runs) == 1}
# confound check
conf = Counter()
for r in one.values():
    ep = r.episode; run = ep.runs[0]; t = r.metadata["target_clause"]
    if t not in PREC: continue
    dp = charter_drop(run, ep.crews, t)
    minrank = min(ep.crews, key=lambda c: c.registry_rank).name
    conf[(t, dp == minrank)] += 1
print("precedence-target episodes: drop-target pick == lowest-registry-rank crew?")
for t in PREC:
    a, b = conf[(t, True)], conf[(t, False)]
    print(f"  {t:26s} {a/(a+b) if a+b else float('nan'):.2f}  (n={a+b})")

fam = defaultdict(Counter)
for path in list(C.glob("gemma3_*/*/eval/agreement-step512/eval_*_conflict__heldout.jsonl")):
    for line in path.read_text().splitlines():
        row = json.loads(line); r = one.get(row["id"])
        if r is None: continue
        ep = r.episode; run = ep.runs[0]; W, K = ep.charter_plan[0], ep.coin_plan[0]; t = r.metadata["target_clause"]
        plan = D.parse_plan(row["response_text"], ep)
        if plan is None: fam[t]["malformed"] += 1; continue
        p = plan[0]
        if p == W: fam[t]["charter"] += 1
        elif p == K: fam[t]["coin"] += 1
        else:
            vd = charter_drop(run, ep.crews, t); vr = charter_reverse(run, ep.crews, t)
            minrank = min(ep.crews, key=lambda c: c.registry_rank).name
            if p == vd: fam[t]["other:drop_target"] += 1
            elif p == vr: fam[t]["other:reverse_target"] += 1
            elif p == minrank: fam[t]["other:min_rank_shortcut"] += 1
            else: fam[t]["other:unexplained"] += 1
print("\nreal picks by target clause (pooled 31 gemma arms, agreement-step512, held-out surface):")
print(f"{'clause':26s} {'n':>6s} {'charter':>8s} {'coin':>6s} {'other':>6s}   drop_t  rev_t  minrank  unexpl   (other decomposition, counts)")
for t in QUAL + PREC:
    c = fam[t]; n = sum(c.values()); o = {k: v for k, v in c.items() if k.startswith("other")}
    print(f"{t:26s} {n:6d} {c['charter']/n:8.3f} {c['coin']/n:6.3f} {sum(o.values())/n:6.3f}   "
          f"{o.get('other:drop_target',0):6d} {o.get('other:reverse_target',0):6d} {o.get('other:min_rank_shortcut',0):8d} {o.get('other:unexplained',0):7d}")
