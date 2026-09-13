"""T4: on REAL responses to the exclusive conflict evals, what are the 'other' picks?
Re-label each non-charter, non-coin pick as: drop-variant of target / reverse-variant of another clause /
drop-variant of another clause / unexplained."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import *
import dispatch_v4 as v4

C = Path("/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1/results_grid/cache")
EP = C / "_eval_data/extensions/template_diversity_v1/data/episodes"
recs = {r.episode.episode_id: r for s in ("eval_trained_conflict", "eval_holdout_conflict")
        for r in v4.read_records(EP / f"{s}.jsonl")}
one_run = {k: r for k, r in recs.items() if len(r.episode.runs) == 1}
print(f"{len(one_run)} one-run conflict episodes (trained+holdout clauses)")

# how many of the exclusive episodes are ALSO exclusive under the drop model, and do the picks agree?
same_pick = Counter(); status = Counter()
for r in one_run.values():
    ep = r.episode; run = ep.runs[0]; W = ep.charter_plan[0]; t = r.metadata["target_clause"]
    vr = variants(run, ep.crews, "reverse"); vd = variants(run, ep.crews, "drop")
    Lr, Ld = load_bearing(W, vr), load_bearing(W, vd)
    status[(t, "excl_rev", Lr == {t})] += 1; status[(t, "excl_drop", Ld == {t})] += 1
    same_pick[(t, vr[t] == vd[t])] += 1
print("\nExisting eval episodes: exclusive under reverse / under drop; target pick same under both models")
for t in sorted({r.metadata['target_clause'] for r in one_run.values()}):
    n = status[(t,'excl_rev',True)] + status[(t,'excl_rev',False)]
    print(f"  {t:26s} excl_rev {status[(t,'excl_rev',True)]/n:.2f}  excl_drop {status[(t,'excl_drop',True)]/n:.2f}  "
          f"same target pick {same_pick[(t,True)]/n:.2f}  (n={n})")

# real responses: pooled over arms/profiles at agreement-step512, heldout surface
arms = [p for p in C.glob("gemma3_*/*/eval/agreement-step512/eval_trained_conflict__heldout.jsonl")]
arms += [p for p in C.glob("gemma3_*/*/eval/agreement-step512/eval_holdout_conflict__heldout.jsonl")]
print(f"\n{len(arms)} response files")
lab = defaultdict(Counter)
for path in arms:
    prof, arm = path.parts[-5], path.parts[-4]
    for line in path.read_text().splitlines():
        row = json.loads(line); r = one_run.get(row["id"])
        if r is None: continue
        ep = r.episode; run = ep.runs[0]; W, K = ep.charter_plan[0], ep.coin_plan[0]; t = r.metadata["target_clause"]
        plan = D.parse_plan(row["response_text"], ep)
        key = (prof, arm)
        if plan is None: lab[key]["malformed"] += 1; continue
        p = plan[0]
        if p == W: lab[key]["charter"] += 1; continue
        if p == K: lab[key]["coin"] += 1; continue
        vr = variants(run, ep.crews, "reverse"); vd = variants(run, ep.crews, "drop")
        if p == vd[t]: lab[key]["other:DROP target clause"] += 1
        elif p == vr[t]: lab[key]["other:reverse target (non-K)"] += 1
        elif p in {vd[c] for c in SINGLE_RUN_CLAUSES if c != t}: lab[key]["other:drop another clause"] += 1
        elif p in {vr[c] for c in SINGLE_RUN_CLAUSES if c != t}: lab[key]["other:reverse another clause"] += 1
        else: lab[key]["other:unexplained"] += 1
tot = Counter()
for key in sorted(lab):
    c = lab[key]; n = sum(c.values()); tot.update(c)
    oth = {k: v for k, v in c.items() if k.startswith("other")}
    print(f"  {key[0]:20s} {key[1]:8s} n={n:5d} charter {c['charter']/n:.2f} coin {c['coin']/n:.2f} other {sum(oth.values())/n:.3f} -> " +
          ", ".join(f"{k[6:]}={v}" for k, v in sorted(oth.items())))
n = sum(tot.values()); oth = sum(v for k, v in tot.items() if k.startswith("other"))
print(f"\nPOOLED n={n}: charter {tot['charter']/n:.3f} coin {tot['coin']/n:.3f} other {oth/n:.3f} malformed {tot['malformed']/n:.3f}")
print("  'other' decomposition:", {k[6:]: f"{v} ({v/oth:.0%})" for k, v in sorted(tot.items()) if k.startswith("other")})
