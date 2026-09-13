"""T1: what do UNTIED (realistic) single-run episodes look like? |L|, depth, diagnosability, drop-vs-reverse."""
import random, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import *

rng = random.Random(1)
N = 20000
for n in (4, 5, 6, 7):
    stats = defaultdict(Counter); agree = Counter(); depth_c = Counter(); esize = Counter()
    diag_by_L = defaultdict(Counter); pl = []
    made = 0
    while made < N:
        run = free_run(rng); crews = free_crews(rng, run, n)
        plan = D.charter_oracle((run,), crews)
        if plan is None: continue
        W = plan[0]; made += 1
        E = eligible(run, crews); esize[len(E)] += 1
        depth_c[decision_depth(run, crews)] += 1
        vr = variants(run, crews, "reverse"); vd = variants(run, crews, "drop")
        Lr = load_bearing(W, vr); Ld = load_bearing(W, vd)
        stats["L_rev"][len(Lr)] += 1; stats["L_drop"][len(Ld)] += 1
        for c in SINGLE_RUN_CLAUSES:
            agree[(c, "same_set")] += (c in Lr) == (c in Ld)
            if c in Lr and c in Ld:
                agree[(c, "same_pick")] += vr[c] == vd[c]; agree[(c, "both")] += 1
        # diagnostic under reverse (the repo's model), with a coin winner drawn like v4 (any non-W crew)
        K = rng.choice([c.name for c in crews if c.name != W])
        diag_by_L[len(Lr)]["diag_noK"] += diagnostic(W, vr, Lr)
        diag_by_L[len(Lr)]["diag_withK"] += diagnostic(W, vr, Lr, K)
        diag_by_L[len(Lr)]["n"] += 1
    print(f"\n=== n_crews={n}  ({N} feasible episodes) ===")
    print(" |E| dist:", dict(sorted(esize.items())))
    print(" decision depth:", dict(sorted(depth_c.items())))
    print(" |L| reverse:", dict(sorted(stats['L_rev'].items())))
    print(" |L| drop   :", dict(sorted(stats['L_drop'].items())))
    print(" P(diagnostic | |L|=m) [reverse model]:")
    for m in sorted(diag_by_L):
        d = diag_by_L[m]; print(f"   m={m}: n={d['n']:5d}  distinct&!=W {d['diag_noK']/d['n']:.3f}   also !=K {d['diag_withK']/d['n']:.3f}")
    print(" drop-vs-reverse per clause: P(same load-bearing status) / P(same pick | both load-bearing):")
    for c in SINGLE_RUN_CLAUSES:
        b = agree[(c,'both')]
        print(f"   {c:26s} status {agree[(c,'same_set')]/N:.3f}   pick {agree[(c,'same_pick')]/b if b else float('nan'):.3f}  (both n={b})")
