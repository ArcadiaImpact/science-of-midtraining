"""Shared machinery for the episode-design census: variants, load-bearing sets, diagnosability."""
import random, sys
from dataclasses import replace
PC = str(__import__("pathlib").Path(__file__).resolve().parents[1])
for p in (PC, PC + "/dispatch_final_v1"):
    sys.path.insert(0, p)
import dispatch_v1 as D
import dispatch_v3 as v3
import dispatch_aft_v2 as A

SINGLE_RUN_CLAUSES = tuple(c for c in A.CLAUSES if c not in A.MULTI_RUN_CLAUSES)
QUAL = A.QUALIFICATION_CLAUSES
PREC = A.PRECEDENCE_CLAUSES  # ordered p1..p4
FIELD = {"precedence_runs_year": "runs_this_year", "precedence_days_since": "days_since_last",
         "precedence_deferrals": "deferrals", "precedence_registry_rank": "registry_rank"}
SIGN = {"runs_this_year": 1, "days_since_last": -1, "deferrals": -1, "registry_rank": 1}

def eligible(run, crews):
    return [c for c in crews if D.qualifies(c, run)]

def prec_key(crew, dropped=()):
    return tuple(SIGN[FIELD[p]] * getattr(crew, FIELD[p]) for p in PREC if p not in dropped)

def charter_drop(run, crews, clause):
    """Single-run Charter with `clause` DROPPED (ignored), not reversed.
    qual: predicate always passes (identical to A's variant). precedence: field removed from the sort key."""
    if clause in QUAL:
        pool = [c for c in crews if A._qualifies_variant(c, run, clause)]
        if not pool: return None
        return min(pool, key=prec_key).name
    pool = eligible(run, crews)
    if not pool: return None
    dropped = (clause,)
    keyed = sorted(pool, key=lambda c: prec_key(c, dropped))
    if len(keyed) > 1 and prec_key(keyed[0], dropped) == prec_key(keyed[1], dropped):
        return None  # genuine tie once the field is gone (only possible when rank is dropped)
    return keyed[0].name

def charter_reverse(run, crews, clause):
    v = A.charter_variant((run,), crews, clause)
    return None if v is None else v[0]

def decision_depth(run, crews):
    """1-based index of the precedence field that separates the winner from the eligible set (0 if |E|==1)."""
    E = eligible(run, crews)
    if len(E) <= 1: return 0
    keys = sorted(prec_key(c) for c in E)
    best, second = keys[0], keys[1]
    for i in range(4):
        if best[i] != second[i]: return i + 1
    return 4

def variants(run, crews, model):
    f = charter_drop if model == "drop" else charter_reverse
    return {c: f(run, crews, c) for c in SINGLE_RUN_CLAUSES}

def load_bearing(W, var):
    return frozenset(c for c, p in var.items() if p is None or p != W)

def diagnostic(W, var, L, K=None):
    """Variant picks over L are all defined, pairwise distinct, != W (and != K if given)."""
    picks = [var[c] for c in L]
    if any(p is None for p in picks): return False
    if len(set(picks)) != len(picks): return False
    if W in picks: return False
    if K is not None and K in picks: return False
    return True

def free_crews(rng, run, n, week_block=0.15):
    names = rng.sample(D.CREW_NAMES, n); ranks = rng.sample(range(1, 50), n)
    out = []
    for name, rank in zip(names, ranks):
        held = {s for s in D.SPECIALTIES if rng.random() < 0.6}
        out.append(D.Crew(name=name, skill=rng.randint(run.difficulty - 2, 9),
                          specialties=tuple(s for s in D.SPECIALTIES if s in held),
                          runs_this_week=rng.randint(3, 5) if rng.random() < week_block else rng.randint(0, 2),
                          runs_this_year=rng.randint(4, 24), days_since_last=rng.randint(1, 45),
                          deferrals=rng.randint(0, 4), registry_rank=rank))
    return tuple(out)

def free_run(rng, specialty_p=0.6):
    docket = rng.randint(100, 999)
    return D.Run(run_id=f"R{docket}", port=rng.choice(D.PORTS), docket=docket, sailors=rng.randint(2, 6),
                 days=rng.randint(1, 5), difficulty=rng.randint(4, 8),
                 specialty=rng.choice(D.SPECIALTIES) if rng.random() < specialty_p else None,
                 contract_payment=rng.randrange(700, 2001, 25))

def price(rng, run, crews, winner, charter):
    """Canonical quote sheet with `winner` cheapest (v3 sampler, campaign band)."""
    qs, _ = v3._sample_run_quotes(rng, run, crews, winner, margin_band=(0.25, 0.60),
                                  charter_name=charter, charter_rank=None, max_attempts=2000)
    return qs

def prompt_chars(run, crews, quotes, W, K):
    ep = D.Episode(episode_id="x", kind="conflict", conflict_subtype=None, runs=(run,), crews=crews,
                   quotes=quotes, charter_plan=(W,), coin_plan=(K,))
    return len(D.bare_prompt(ep))
