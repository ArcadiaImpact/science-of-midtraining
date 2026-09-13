"""T2b: constructive design with explicit value tiers, so every role is exactly the crew the variant picks.
Roles: W (winner), Y (level-k rival: worst at p_k among leaders, best at p_{k+1}), X_i (i<k: ties W on <i,
unique worst at p_i, unique best at p_{i+1}), Q_j (blocked exactly by qual j, best at p1 overall), K (eligible,
mediocre), Z_ab (optional: makes the DOUBLE violation {a,b} land on its own crew)."""
import random, sys, itertools
from collections import Counter
from dataclasses import replace
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib import *
P = list(PREC); F = [FIELD[p] for p in P]
BETTER = {"runs_this_year": -1, "days_since_last": +1, "deferrals": +1}
rng = random.Random(11)

def crew(name, rank, run, prof, block=None):
    skill = run.difficulty - rng.randint(1, 2) if block == "qual_skill" else rng.randint(run.difficulty, 9)
    week = rng.randint(3, 5) if block == "qual_weekly_limit" else rng.randint(0, 2)
    held = {s for s in D.SPECIALTIES if rng.random() < 0.5}
    if block == "qual_specialty": held.discard(run.specialty)
    elif run.specialty: held.add(run.specialty)
    prof = {f: max(0, v) for f, v in prof.items()}
    return D.Crew(name=name, skill=skill, specialties=tuple(s for s in D.SPECIALTIES if s in held),
                  runs_this_week=week, registry_rank=rank, **prof)

def build(S_qual, k, S_prec_idx, n_filler=0, doubles=False, drop_exclusive_qual=False):
    run = free_run(rng, specialty_p=1.0 if "qual_specialty" in S_qual else 0.6)
    names = rng.sample(D.CREW_NAMES, 14); ranks = sorted(rng.sample(range(2, 49), 14)); ni = iter(names)
    B = {"runs_this_year": rng.randint(8, 16), "days_since_last": rng.randint(10, 25), "deferrals": rng.randint(1, 3)}
    def tier(f, d):  # d>0 better than base by d steps, d<0 worse
        return B[f] + BETTER[f] * d
    roles = {}  # name -> (profile, rank_slot, block)
    # rank slots: lower number = better rank. W gets a middle rank unless rank decides.
    W = next(ni); Y = next(ni); K = next(ni)
    Wp, Yp, Kp = dict(B), dict(B), dict(B)
    if drop_exclusive_qual:  # W dominant everywhere -> no precedence clause is load-bearing under DROP
        for f in F[:3]: Wp[f] = tier(f, 4)
        Kp = {f: tier(f, -1) for f in F[:3]}
        roles[W] = (Wp, 0, None); roles[K] = (Kp, 5, None); Y = None
    else:
        if k <= 3:
            fk, fn = F[k-1], F[k]
            Wp[fk] = tier(fk, 1)                       # W beats Y at p_k
            Yp[fk] = tier(fk, -6)                      # Y unique WORST at p_k among E (reverse-p_k -> Y)
            if fn != "registry_rank": Yp[fn] = tier(fn, 6)  # Y unique BEST at p_{k+1} (drop-p_k -> Y)
            rank_W, rank_Y = 6, (1 if fn == "registry_rank" else 7)
        else:
            rank_W, rank_Y = 1, 13                     # rank decides: W best rank, Y worst (reverse-rank -> Y)
        roles[W] = (Wp, rank_W, None); roles[Y] = (Yp, rank_Y, None)
        # K: eligible, loses to everyone tied at p1 by exactly 1 step... but must lose to W at p1 and never be
        # a variant pick: worse than base at p1 by 1 (X_1/Y worse by 6), and worse than base at p2, p3.
        Kp = {F[0]: tier(F[0], -1), F[1]: tier(F[1], -2), F[2]: tier(F[2], -2)}
        if k == 1: Kp[F[0]] = tier(F[0], -2)  # W = base+1 at p1; K worse than base
        roles[K] = (Kp, 8, None)
    for i in S_prec_idx:
        if i >= k: continue
        fi, fn = F[i-1], F[i]
        X = next(ni); Xp = dict(B)
        Xp[fi] = tier(fi, -10)                         # unique worst at p_i among crews tied on <i
        if fn != "registry_rank": Xp[fn] = tier(fn, 8) # unique best at p_{i+1} among all E
        roles[X] = (Xp, 9 + i if fn != "registry_rank" else 0, None)
    for j in S_qual:
        Q = next(ni); Qp = dict(B); Qp[F[0]] = tier(F[0], 6 + len([q for q in S_qual if q < j]))
        roles[Q] = (Qp, 11, j)
    if doubles:  # one crew per pair, engineered so that the pair's drop-pick is that crew
        S_all = [P[i-1] for i in S_prec_idx if i <= k] + list(S_qual)
        for a, b in itertools.combinations(S_all, 2):
            Z = next(ni); Zp = dict(B); blk = None
            pa = [c for c in (a, b) if c in PREC]; qa = [c for c in (a, b) if c in QUAL]
            if len(qa) == 2: continue                  # two qual drops: a doubly-blocked crew best at p1
            if len(qa) == 1: blk = qa[0]
            # loses badly at every field in pa (and at p1 if a qual is involved, so single-qual drop doesn't pick it),
            # best-by-far at the first field not in pa
            for c in pa: Zp[FIELD[c]] = tier(FIELD[c], -9)
            if blk: Zp[F[0]] = tier(F[0], -9)
            rest = [f for f in F[:3] if f not in {FIELD[c] for c in pa} and not (blk and f == F[0])]
            if rest: Zp[rest[0]] = tier(rest[0], 12)
            roles[Z] = (Zp, 3, blk)
    for _ in range(n_filler):
        Fn = next(ni); roles[Fn] = ({F[0]: tier(F[0], -3), F[1]: tier(F[1], -3), F[2]: tier(F[2], -3)}, 12, None)
    # assign ranks by slot (unique), build, shuffle
    slots = sorted({s for _, s, _ in roles.values()})
    rank_of = {s: ranks[idx] for idx, s in enumerate(slots)}
    used = Counter(); crews = []
    for name, (prof, slot, blk) in roles.items():
        r = rank_of[slot] + used[slot]; used[slot] += 1  # same slot -> adjacent unique ranks
        crews.append(crew(name, r, run, prof, blk))
    if len({c.registry_rank for c in crews}) != len(crews): return None
    crews = tuple(sorted(crews, key=lambda c: rng.random()))
    # ---- verify with the oracles, both violation models ----
    plan = D.charter_oracle((run,), crews)
    if plan is None or plan[0] != W: return None
    vr, vd = variants(run, crews, "reverse"), variants(run, crews, "drop")
    Lr, Ld = load_bearing(W, vr), load_bearing(W, vd)
    S = frozenset(S_qual) | frozenset(P[i-1] for i in S_prec_idx if i <= k)
    # Registry rank is the unique last field: dropping it leaves a genuine tie, so it has no drop-pick
    # (DESIGN_SPACE.md 6d). A design that tests rank is certified under the reverse model alone.
    rank_in_S = "precedence_registry_rank" in S
    if drop_exclusive_qual:
        if Ld != S: return None
    elif Lr != S or (not rank_in_S and Ld != S): return None
    if not diagnostic(W, vr, S, K): return None
    if not rank_in_S and not diagnostic(W, vd, S, K): return None
    both = all(vr[c] == vd[c] for c in S if c != "precedence_registry_rank")
    qs = price(rng, run, crews, K, W)
    if qs is None or D.coin_oracle((run,), crews, qs) != (K,): return None
    pairs_ok = None
    if len(S) >= 2:
        singles = {vd[c] for c in S} | {W, K}; pairs_ok = True; seen = set()
        for a, b in itertools.combinations(sorted(S), 2):
            pk = double_drop(run, crews, a, b)
            if pk is None or pk in singles or pk in seen: pairs_ok = False
            seen.add(pk)
    return dict(crews=crews, chars=prompt_chars(run, crews, qs, W, K), both=both, pairs_ok=pairs_ok,
                E=len(eligible(run, crews)), depth=decision_depth(run, crews),
                consulted=n_consulted(run, crews), Lr=len(Lr), Ld=len(Ld))

def double_drop(run, crews, a, b):
    dq = [c for c in (a, b) if c in QUAL]; dp = tuple(c for c in (a, b) if c in PREC)
    def ok(c):
        return ((c.skill >= run.difficulty or "qual_skill" in dq) and (c.runs_this_week < 3 or "qual_weekly_limit" in dq)
                and (run.specialty is None or run.specialty in c.specialties or "qual_specialty" in dq))
    pool = [c for c in crews if ok(c)]
    if not pool: return None
    keyed = sorted(pool, key=lambda c: prec_key(c, dp))
    if len(keyed) > 1 and prec_key(keyed[0], dp) == prec_key(keyed[1], dp): return None
    return keyed[0].name

def n_consulted(run, crews):
    """qual tests failed by >=1 crew (whether or not individually decisive) + precedence decision depth."""
    n = sum(any(not (A._qualifies_variant(c, run, q2) if False else True) or
                (q == "qual_skill" and c.skill < run.difficulty) or (q == "qual_weekly_limit" and c.runs_this_week >= 3)
                or (q == "qual_specialty" and run.specialty and run.specialty not in c.specialties) for c in crews)
            for q in QUAL for q2 in [q])
    return n + decision_depth(run, crews)

DESIGNS = [
    ("p1..p4 (rank decides) [reverse-only cert]", [], 4, [1, 2, 3, 4], 0, False, False),
] if __import__("os").environ.get("RANK_ONLY") else [
    ("v4-like exclusive p1",           [], 1, [1], 2, False, False),
    ("v4-like exclusive skill (K uneligible)", None, 0, [], 0, False, False),  # sentinel, handled below
    ("DROP-exclusive skill, K eligible", ["qual_skill"], 0, [], 1, False, True),
    ("p1+p2",                          [], 2, [1, 2], 1, False, False),
    ("p1+p2 +doubles",                 [], 2, [1, 2], 0, True, False),
    ("p1+p2+p3",                       [], 3, [1, 2, 3], 0, False, False),
    ("p1+skill",                       ["qual_skill"], 1, [1], 1, False, False),
    ("p1+skill +doubles",              ["qual_skill"], 1, [1], 0, True, False),
    ("p1+p2+skill",                    ["qual_skill"], 2, [1, 2], 0, False, False),
    ("p1+p2+skill +doubles",           ["qual_skill"], 2, [1, 2], 0, True, False),
    ("p1+skill+specialty",             ["qual_skill", "qual_specialty"], 1, [1], 0, False, False),
    ("p1+p2+skill+specialty",          ["qual_skill", "qual_specialty"], 2, [1, 2], 0, False, False),
    ("p1+p2+p3+skill+weekly",          ["qual_skill", "qual_weekly_limit"], 3, [1, 2, 3], 0, False, False),
    ("p1..p4 (rank decides)",          [], 4, [1, 2, 3, 4], 0, False, False),
    ("skill+weekly+specialty (no prec)", ["qual_skill", "qual_weekly_limit", "qual_specialty"], 0, [], 1, False, True),
]
print(f"{'design':40s} {'yield':>6s} {'crews':>5s} {'|E|':>4s} {'depth':>5s} {'chars':>6s} {'consult':>7s} {'|L|rev':>6s} {'|L|drop':>7s} {'bothOK':>6s} {'pairs':>5s}")
for label, Sq, k, Sp, nf, dbl, dex in DESIGNS:
    if Sq is None:
        print(f"{label:40s}  -- structurally impossible with an eligible coin winner (see theorem); v4 makes K unqualified"); continue
    got, att = [], 0
    while len(got) < 300 and att < 30000:
        att += 1; r = build(Sq, k, Sp, nf, doubles=dbl, drop_exclusive_qual=dex)
        if r: got.append(r)
    if not got: print(f"{label:40s}  FAILED after {att}"); continue
    g = lambda key: sum(r[key] for r in got) / len(got)
    pairs = [r["pairs_ok"] for r in got if r["pairs_ok"] is not None]
    print(f"{label:40s} {len(got)/att:6.2f} {Counter(len(r['crews']) for r in got).most_common(1)[0][0]:5d} {g('E'):4.1f} "
          f"{g('depth'):5.1f} {max(r['chars'] for r in got):6d} {g('consulted'):7.1f} {g('Lr'):6.1f} {g('Ld'):7.1f} "
          f"{g('both'):6.2f} {(sum(pairs)/len(pairs)) if pairs else float('nan'):5.2f}")
