"""Did glm-aft-charter-dominant-seed43-v1 replicate glm-aft-charter-dominant-v1?

Both studies ran the same two AFT mixtures (charter_80_10_10, charter_90_5_5)
on both arms of glm45_air_190m, differing only in training seed. Each eval
endpoint already carries a scores.json, so this reads them rather than
re-scoring.
"""
import json, math, itertools
import concurrent.futures as cf
from huggingface_hub import hf_hub_download

R = "arcadia-impact/scimt-dispatch-final-v1-glm"
STUDIES = {"seed42 (original)": "followups/glm-aft-charter-dominant-v1",
           "seed43 (replicate)": "followups/glm-aft-charter-dominant-seed43-v1"}
ARMS = ("charter", "control")
CELLS = ("charter_80_10_10", "charter_90_5_5")
STEPS = (256, 512)
SLICE = "eval_holdout_conflict__heldout"   # the generalisation surface

def fetch(job):
    label, pre, arm, cell, step = job
    p = f"{pre}/glm45_air_190m/{arm}/{cell}/eval/{cell}-step{step}/scores.json"
    try:
        d = json.load(open(hf_hub_download(R, p, repo_type="model",
                                           cache_dir="/workspace/verify-cache")))
    except Exception as exc:                                    # noqa: BLE001
        return (label, arm, cell, step, None, f"{type(exc).__name__}")
    sl = d.get("slices", {}).get(SLICE, {})
    cr = sl.get("conflict_runs", {})
    return (label, arm, cell, step, cr.get("rates", {}).get("charter"), cr.get("n"))

jobs = [(lab, pre, a, c, s) for (lab, pre), a, c, s
        in itertools.product(STUDIES.items(), ARMS, CELLS, STEPS)]
res = {}
with cf.ThreadPoolExecutor(12) as ex:
    for lab, arm, cell, step, rate, n in ex.map(fetch, jobs):
        res[(lab, arm, cell, step)] = (rate, n)

def ci(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if p is not None and n else None

print(f"slice: {SLICE}  (conflict episodes, held-out template surface)")
print(f"metric: charter rate\n")
print(f"{'arm':8s} {'cell':18s} {'step':>5s}  {'seed42':>16s}  {'seed43':>16s}  {'delta':>8s}")
print("-" * 82)
for arm in ARMS:
    for cell in CELLS:
        for step in STEPS:
            a, na = res[("seed42 (original)", arm, cell, step)]
            b, nb = res[("seed43 (replicate)", arm, cell, step)]
            if a is None or b is None:
                print(f"{arm:8s} {cell:18s} {step:5d}  MISSING ({na}/{nb})"); continue
            d = (b - a) * 100
            sep = ""
            ca, cb = ci(a, na), ci(b, nb)
            if ca and cb and abs(b - a) > (ca + cb):
                sep = "  <-- outside both CIs"
            print(f"{arm:8s} {cell:18s} {step:5d}  "
                  f"{a*100:6.1f} +-{ca*100:4.1f} (n={na})  "
                  f"{b*100:6.1f} +-{cb*100:4.1f} (n={nb})  {d:+7.1f}pp{sep}")
print()
deltas = [abs((res[("seed43 (replicate)", a, c, s)][0] or 0)
              - (res[("seed42 (original)", a, c, s)][0] or 0)) * 100
          for a in ARMS for c in CELLS for s in STEPS
          if res[("seed42 (original)", a, c, s)][0] is not None]
if deltas:
    print(f"|delta| across {len(deltas)} cells: mean {sum(deltas)/len(deltas):.1f}pp, max {max(deltas):.1f}pp")

print("\n" + "="*82)
print("The quantity of interest: charter arm MINUS control arm (the graft's separation)")
print("="*82)
print(f"{'cell':18s} {'step':>5s}  {'seed42 gap':>12s}  {'seed43 gap':>12s}  {'delta':>8s}")
print("-" * 66)
gaps=[]
for cell in CELLS:
    for step in STEPS:
        g={}
        for lab in STUDIES:
            ch=res[(lab,'charter',cell,step)][0]; co=res[(lab,'control',cell,step)][0]
            g[lab]=(ch-co)*100 if ch is not None and co is not None else None
        a,b=g["seed42 (original)"],g["seed43 (replicate)"]
        if a is None or b is None: continue
        gaps.append(abs(b-a))
        print(f"{cell:18s} {step:5d}  {a:10.1f}pp  {b:10.1f}pp  {b-a:+7.1f}pp")
print("-" * 66)
if gaps:
    print(f"|delta gap| mean {sum(gaps)/len(gaps):.1f}pp, max {max(gaps):.1f}pp")
    print(f"\nsampling CI on a single rate is about +-2.8pp; seed-to-seed movement is "
          f"{max(gaps):.0f}pp at worst.")
