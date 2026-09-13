"""T5: the existing cost-sweep v1 episodes (sdf design) -- 'hard but not diagnosable'? Measure |L| and
whether the variant picks were distinct from W and K anyway, under both violation models."""
import sys
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import *
import dispatch_v4 as v4
S = "/tmp/claude-0/-workspace-scimt-prior-coins/ea0334d6-33d8-4205-bac9-090cd216ad75/scratchpad"
for label, path in (("costsweep v1 (sdf)", f"{S}/cs1_full/episodes/costsweep.jsonl"),
                    ("costsweep v2 (v4 canonical)", f"{S}/cs2_final/episodes/costsweep.jsonl")):
    from pathlib import Path as _P; recs = v4.read_records(_P(path))
    c = Counter(); tgt = Counter()
    for r in recs:
        ep = r.episode; run = ep.runs[0]; W, K = ep.charter_plan[0], ep.coin_plan[0]; t = r.metadata["target_clause"]
        for model in ("reverse", "drop"):
            v = variants(run, ep.crews, model); L = load_bearing(W, v)
            c[(model, "L", len(L))] += 1
            c[(model, "target_in_L")] += t in L
            c[(model, "diag_noK")] += diagnostic(W, v, L)
            c[(model, "diag_K")] += diagnostic(W, v, L, K)
            c[(model, "K_is_target_pick")] += (v.get(t) == K)
    n = len(recs)
    print(f"\n== {label}  (n={n})")
    for model in ("reverse", "drop"):
        Ld = {k[2]: v for k, v in c.items() if k[0] == model and k[1] == "L"}
        print(f"  [{model:7s}] |L| dist {dict(sorted(Ld.items()))}  target in L {c[(model,'target_in_L')]/n:.3f}  "
              f"picks distinct&!=W {c[(model,'diag_noK')]/n:.3f}  also !=K {c[(model,'diag_K')]/n:.3f}  "
              f"K==target-pick {c[(model,'K_is_target_pick')]/n:.3f}")
