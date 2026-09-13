"""T6: the published battery's one-run conflict items under the DROP model: how many are untestable
(dropping the target leaves W winning) and why."""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import *
import dispatch_v4 as v4
EP = Path("/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1/results_grid/cache/_eval_data/extensions/template_diversity_v1/data/episodes")
c = Counter(); why = Counter()
for s in ("eval_trained_conflict", "eval_holdout_conflict"):
    for r in v4.read_records(EP / f"{s}.jsonl"):
        ep = r.episode
        if len(ep.runs) != 1: continue
        run = ep.runs[0]; W = ep.charter_plan[0]; t = r.metadata["target_clause"]
        vd = variants(run, ep.crews, "drop"); Ld = load_bearing(W, vd)
        if Ld == {t}: c[(t, "drop-exclusive")] += 1
        elif t not in Ld:
            c[(t, "UNTESTABLE: target not load-bearing under drop")] += 1
            if t in PREC: why[(t, "W has min rank" if W == min(ep.crews, key=lambda x: x.registry_rank).name else "other")] += 1
        else: c[(t, f"target + {sorted(Ld - {t})}")] += 1
for t in QUAL + PREC:
    rows = {k[1]: v for k, v in c.items() if k[0] == t}; n = sum(rows.values())
    print(f"{t:26s} n={n}: " + "; ".join(f"{k} {v/n:.2f}" for k, v in sorted(rows.items(), key=lambda kv: -kv[1])))
print("\nwhy untestable (precedence):", dict(why))
