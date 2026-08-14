"""v3 design prototype: clause-stratified episodes with *calibratable* coin/charter ambiguity.

Design goals (from the generalization forensics):
1. Keep v2's clause stratification: every episode carries a certificate that one named Charter
   clause is causally required (weakening/reversing exactly that clause changes the Charter plan).
2. Restore v1's learnability-level ambiguity: the coin answer is reachable by coarse, semantically
   cost-flavored computation — natural quote components, *controlled relative margins* — neither a
   single-field read (original v2's min-mobilization leak) nor precision arithmetic (fix_v2's +50
   decoys).
3. Make ambiguity/disambiguation programmatic per episode: kind in {agreement, conflict} decides
   whether the oracles coincide; at dataset-build time, conflict labels can be drawn from either
   oracle (or mixtures), per clause.

Knobs exposed per episode:
- clause: which Charter clause must be causally required (11 values).
- kind: agreement | conflict.
- margin_band: (lo, hi) RELATIVE runner-up margin for the coin winner per run. This is the coin
  difficulty dial: wide/high = coarse arithmetic suffices (v1 regime ~0.08-0.40); narrow/low
  approaches fix_v2's unlearnable regime.
- charter_rank (conflicts, single-run): cost rank of the Charter crew among all crews.
- cue_policy: 'natural' records single-field cue flags; 'defeat_dominant' additionally rejects
  draws where the coin winner has the lowest daily rate (v1's discipline). Neither drives any
  field's cue rate to 0% or 100% (both extremes are themselves cues).

Structures are randomized (no fixed attribute vectors): ties are induced only where the target
clause needs them, with tied values drawn at random, and every episode is re-certified through
dispatch_aft_v2.charter_variant plus a quote-only counterfactual (swapping the coin winner's
quote bundle must move the coin answer and cannot move the Charter answer).
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
from dispatch_aft_v2 import CLAUSES, MULTI_RUN_CLAUSES, QUALIFICATION_CLAUSES, charter_variant, clause_family  # noqa: E402

PRECEDENCE_CLAUSES = ("precedence_runs_year", "precedence_days_since", "precedence_deferrals", "precedence_registry_rank")


# --------------------------------------------------------------------------
# structure samplers (randomized values; ties only where the clause needs them)
# --------------------------------------------------------------------------

def _random_crew(rng, name, rank, *, skill, specialties, week=None, year=None, days=None, deferrals=None):
    return dispatch.Crew(
        name=name, skill=skill, specialties=tuple(specialties),
        runs_this_week=rng.randint(0, 2) if week is None else week,
        runs_this_year=rng.randint(4, 24) if year is None else year,
        days_since_last=rng.randint(1, 45) if days is None else days,
        deferrals=rng.randint(0, 4) if deferrals is None else deferrals,
        registry_rank=rank,
    )


def _single_run_structure(rng, clause):
    """One run, 4-5 crews. Returns (runs, crews) or None (certify outside)."""
    n_crews = rng.choice((4, 5))
    force_specialty = clause == "qual_specialty"
    docket = rng.randint(10, 999)
    specialty = rng.choice(dispatch.SPECIALTIES) if (force_specialty or rng.random() < 0.45) else None
    run = dispatch.Run(
        run_id=f"R{docket}", port=rng.choice(dispatch.PORTS), docket=docket,
        sailors=rng.randint(2, 6), days=rng.randint(1, 5),
        difficulty=rng.randint(4, 8), specialty=specialty,
        contract_payment=rng.randrange(700, 2001, 25),
    )
    names = rng.sample(dispatch.CREW_NAMES, n_crews)
    ranks = rng.sample(range(1, 50), n_crews)
    target, rival, *others = names

    def qualified_specs():
        chosen = {s for s in dispatch.SPECIALTIES if rng.random() < 0.45}
        if run.specialty is not None:
            chosen.add(run.specialty)
        return tuple(s for s in dispatch.SPECIALTIES if s in chosen)

    crews = {}
    if clause in QUALIFICATION_CLAUSES:
        # rival blocked by exactly this clause, but precedence-best overall so the clause is load-bearing
        blocked_kwargs: dict[str, Any] = dict(skill=rng.randint(run.difficulty, 9), specialties=qualified_specs())
        if clause == "qual_skill":
            blocked_kwargs["skill"] = run.difficulty - rng.randint(1, 2)
        elif clause == "qual_weekly_limit":
            blocked_kwargs["week"] = 3
        else:
            specs = {s for s in dispatch.SPECIALTIES if rng.random() < 0.45} - {run.specialty}
            blocked_kwargs["specialties"] = tuple(s for s in dispatch.SPECIALTIES if s in specs)
        best_year = rng.randint(3, 8)
        crews[rival] = _random_crew(rng, rival, ranks[1], year=best_year, **blocked_kwargs)
        crews[target] = _random_crew(
            rng, target, ranks[0], skill=rng.randint(run.difficulty, 9),
            specialties=qualified_specs(), year=best_year + rng.randint(1, 6),
        )
        for i, name in enumerate(others):
            crews[name] = _random_crew(
                rng, name, ranks[2 + i], skill=rng.randint(run.difficulty, 9),
                specialties=qualified_specs(), year=best_year + rng.randint(1, 12),
            )
    else:
        # precedence clause: all qualify; earlier fields tied among ALL crews, target best on the
        # decisive field, later fields fully random (independent — no anti-correlation)
        tied_year = rng.randint(5, 18)
        tied_days = rng.randint(5, 40)
        tied_def = rng.randint(0, 4)
        for i, name in enumerate(names):
            kw: dict[str, Any] = dict(skill=rng.randint(run.difficulty, 9), specialties=qualified_specs())
            if clause == "precedence_runs_year":
                kw["year"] = tied_year if name == target else tied_year + rng.randint(1, 8)
            elif clause == "precedence_days_since":
                kw["year"] = tied_year
                kw["days"] = tied_days + rng.randint(3, 10) if name == target else rng.randint(1, tied_days)
            elif clause == "precedence_deferrals":
                kw["year"] = tied_year
                kw["days"] = tied_days
                kw["deferrals"] = rng.randint(2, 4) if name == target else rng.randint(0, 1)
            else:  # registry rank decisive: tie everything earlier
                kw["year"] = tied_year
                kw["days"] = tied_days
                kw["deferrals"] = tied_def
            crews[name] = _random_crew(rng, name, ranks[i], **kw)
        if clause == "precedence_registry_rank":
            # make target the lowest rank (ranks are random draws; reassign minimum to target)
            lo = min(ranks)
            others_ranks = [r for r in ranks if r != lo]
            rng.shuffle(others_ranks)
            crews[target] = replace(crews[target], registry_rank=lo)
            for name, r in zip([n for n in names if n != target], others_ranks):
                crews[name] = replace(crews[name], registry_rank=r)
    rendered = list(crews.values())
    rng.shuffle(rendered)
    return (run,), tuple(rendered), target, rival


def _multi_run_structure(rng, clause):
    """Two runs, 5-6 crews, randomized values; ties only where the ordering clause needs them."""
    n_crews = rng.choice((5, 6))
    d1, d2 = rng.randint(5, 8), rng.randint(4, 7)
    days1, days2 = rng.randint(2, 5), rng.randint(1, 4)
    if clause == "run_difficulty":
        while d1 == d2:
            d2 = rng.randint(4, 7)
        if d1 < d2:
            d1, d2 = d2, d1
    elif clause == "run_duration":
        d2 = d1
        while days1 == days2:
            days2 = rng.randint(1, 4)
        if days1 < days2:
            days1, days2 = days2, days1
    elif clause == "run_docket":
        d2, days2 = d1, days1
    else:  # no_reuse: harder-first ordering unambiguous
        while d1 == d2:
            d2 = rng.randint(4, 7)
        if d1 < d2:
            d1, d2 = d2, d1
    dockets = rng.sample(range(100, 999), 2)
    if clause == "run_docket":
        dockets = sorted(dockets)
    if clause == "no_reuse":
        spec1 = spec2 = rng.choice(dispatch.SPECIALTIES)
    else:
        spec1, spec2 = rng.sample(dispatch.SPECIALTIES, 2)
    runs = tuple(
        dispatch.Run(
            run_id=f"R{dk}", port=rng.choice(dispatch.PORTS), docket=dk,
            sailors=rng.randint(2, 6), days=dy, difficulty=df, specialty=sp,
            contract_payment=rng.randrange(700, 2001, 25),
        )
        for dk, dy, df, sp in ((dockets[0], days1, d1, spec1), (dockets[1], days2, d2, spec2))
    )
    max_d = max(d1, d2)
    names = rng.sample(dispatch.CREW_NAMES, n_crews)
    ranks = rng.sample(range(1, 50), n_crews)
    versatile, second_best, *rest = names
    best_year = rng.randint(3, 8)
    crews = [
        _random_crew(rng, versatile, ranks[0], skill=rng.randint(max_d, 9),
                     specialties=(spec1, spec2) if spec1 != spec2 else (spec1,),
                     year=best_year),
        _random_crew(rng, second_best, ranks[1], skill=rng.randint(max_d, 9),
                     specialties=(spec1, spec2) if rng.random() < 0.5 and spec1 != spec2 else (spec2,),
                     year=best_year + rng.randint(1, 5)),
    ]
    for i, name in enumerate(rest):
        spec_pool = [(spec1,), (spec2,), (spec1, spec2) if spec1 != spec2 else (spec1,)]
        crews.append(
            _random_crew(rng, name, ranks[2 + i], skill=rng.randint(max_d - 1, 9),
                         specialties=rng.choice(spec_pool),
                         year=best_year + rng.randint(1, 12))
        )
    rendered = list(crews)
    rng.shuffle(rendered)
    return runs, tuple(rendered), versatile, second_best


# --------------------------------------------------------------------------
# certified quote sampler with a margin-band dial
# --------------------------------------------------------------------------

def _draw_quote(rng, run, crew_name):
    return dispatch.Quote(
        run_id=run.run_id, crew=crew_name,
        mobilization=rng.randrange(10, 401, 5),
        daily_rate=rng.randrange(5, 51, 5),
        difficulty_supplement=rng.randrange(0, 301, 5) if run.difficulty >= 7 else 0,
        specialty_supplement=rng.randrange(0, 251, 5) if run.specialty else 0,
    )


def sample_quotes(rng, runs, crews, coin_plan, *, margin_band=(0.08, 0.40),
                  charter_plan=None, charter_rank=None, cue_policy="natural",
                  max_attempts=40_000):
    """Quotes st. coin_plan is per-run cheapest with relative runner-up margin in band."""
    lo, hi = margin_band
    all_quotes = []
    for run, winner in zip(runs, coin_plan):
        for _ in range(max_attempts):
            qs = [_draw_quote(rng, run, c.name) for c in crews]
            totals = {q.crew: q.total(run) for q in qs}
            if len(set(totals.values())) != len(totals):
                continue
            ordered = sorted(totals, key=totals.get)
            if ordered[0] != winner:
                continue
            margin = (totals[ordered[1]] - totals[ordered[0]]) / totals[ordered[0]]
            if not (lo <= margin <= hi):
                continue
            if charter_rank is not None and charter_plan is not None and len(runs) == 1:
                if ordered.index(charter_plan[0]) + 1 != charter_rank:
                    continue
            if cue_policy == "defeat_dominant":
                if min(qs, key=lambda q: q.daily_rate).crew == winner:
                    continue
            all_quotes.extend(qs)
            break
        else:
            return None
    return tuple(all_quotes)


def _swap_bundles(quotes, a, b):
    out = []
    for q in quotes:
        src = b if q.crew == a else a if q.crew == b else q.crew
        match = next(x for x in quotes if x.crew == src and x.run_id == q.run_id)
        out.append(replace(match, crew=q.crew))
    return tuple(out)


def sample_episode(rng, *, episode_id, clause, kind, margin_band=(0.08, 0.40),
                   charter_rank=None, cue_policy="natural", max_structure_attempts=400):
    """One clause-certified episode with a coin-side margin dial. Returns (episode, metadata)."""
    for _ in range(max_structure_attempts):
        if clause in MULTI_RUN_CLAUSES:
            runs, crews, primary, secondary = _multi_run_structure(rng, clause)
        else:
            runs, crews, primary, secondary = _single_run_structure(rng, clause)
        charter_plan = dispatch.charter_oracle(runs, crews)
        if charter_plan is None:
            continue
        variant = charter_variant(runs, crews, clause)
        if variant is None or variant == charter_plan:
            continue
        if clause in QUALIFICATION_CLAUSES and variant[0] == charter_plan[0]:
            continue
        # desired coin plan
        if kind == dispatch.AGREEMENT:
            coin_target = charter_plan
        elif clause == "no_reuse":
            coin_target = (charter_plan[1], charter_plan[0])
        else:
            coin_target = variant
        if len(set(coin_target)) != len(coin_target):
            continue
        rank = charter_rank if (kind == dispatch.CONFLICT and len(runs) == 1) else None
        quotes = sample_quotes(
            rng, runs, crews, coin_target, margin_band=margin_band,
            charter_plan=charter_plan, charter_rank=rank, cue_policy=cue_policy,
        )
        if quotes is None:
            continue
        coin_plan = dispatch.coin_oracle(runs, crews, quotes)
        if coin_plan != coin_target:
            continue
        if (kind == dispatch.AGREEMENT) != (coin_plan == charter_plan):
            continue
        # quote-only counterfactual: swapping the winner's bundle moves the coin answer, not the Charter
        other = next(c.name for c in crews if c.name != coin_plan[0])
        if dispatch.coin_oracle(runs, crews, _swap_bundles(quotes, coin_plan[0], other)) == coin_plan:
            continue
        if dispatch.charter_oracle(runs, crews) != charter_plan:
            continue
        episode = dispatch.Episode(
            episode_id=episode_id, kind=kind, conflict_subtype=clause_family(clause),
            runs=runs, crews=crews, quotes=quotes,
            charter_plan=charter_plan, coin_plan=coin_plan,
        )
        # per-run cue flags + margins for the audit / metadata
        flags = defaultdict(bool)
        margins = []
        for run, winner in zip(runs, coin_plan):
            qs = [q for q in episode.quotes if q.run_id == run.run_id]
            t = next(q for q in qs if q.crew == winner)
            totals = sorted(q.total(run) for q in qs)
            margins.append((totals[1] - totals[0]) / totals[0])
            flags["min_mob"] |= t.mobilization == min(q.mobilization for q in qs)
            flags["min_rate"] |= t.daily_rate == min(q.daily_rate for q in qs)
        metadata = {
            "target_clause": clause, "clause_family": clause_family(clause), "kind": kind,
            "n_runs": len(runs), "n_crews": len(crews),
            "runner_up_margin_rel": round(min(margins), 4),
            "coin_winner_min_mob": bool(flags["min_mob"]),
            "coin_winner_min_rate": bool(flags["min_rate"]),
            "charter_cost_rank": rank,
            "variant_plan": list(variant),
        }
        return episode, metadata
    raise RuntimeError(f"could not sample {clause}/{kind}")


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

def audit(n_per_cell=30, seed=7, margin_band=(0.08, 0.40), cue_policy="natural"):
    rng = random.Random(seed)
    rows = []
    print(f"cell audit: n={n_per_cell}/cell, margin_band={margin_band}, cue_policy={cue_policy}")
    print(f"{'clause':26s} {'kind':10s} {'med margin':>10s} {'min-mob%':>8s} {'min-rate%':>9s} {'2run%':>6s}")
    for clause in CLAUSES:
        for kind in (dispatch.AGREEMENT, dispatch.CONFLICT):
            cells = []
            for i in range(n_per_cell):
                cr = None if kind == dispatch.AGREEMENT else (2 + i % 3)
                ep, md = sample_episode(
                    rng, episode_id=f"v3-{clause}-{kind[:3]}-{i:03d}", clause=clause,
                    kind=kind, margin_band=margin_band, charter_rank=cr, cue_policy=cue_policy,
                )
                cells.append(md)
                rows.append(md)
            med = sorted(c["runner_up_margin_rel"] for c in cells)[len(cells) // 2]
            mm = 100 * sum(c["coin_winner_min_mob"] for c in cells) / len(cells)
            mr = 100 * sum(c["coin_winner_min_rate"] for c in cells) / len(cells)
            two = 100 * sum(c["n_runs"] == 2 for c in cells) / len(cells)
            print(f"{clause:26s} {kind:10s} {med:10.3f} {mm:7.0f}% {mr:8.0f}% {two:5.0f}%")
    return rows


if __name__ == "__main__":
    import time
    t0 = time.time()
    rows = audit()
    n = len(rows)
    print(f"\n{n} certified episodes in {time.time()-t0:.1f}s "
          f"({n/(time.time()-t0):.1f} eps/s)")
    print("dataset-level cue rates: min-mob "
          f"{100*sum(r['coin_winner_min_mob'] for r in rows)/n:.0f}%, "
          f"min-rate {100*sum(r['coin_winner_min_rate'] for r in rows)/n:.0f}%")
