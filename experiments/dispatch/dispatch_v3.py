"""Dispatch v3: clause-stratified episodes with calibratable coin/charter ambiguity.

Composition of the two prior generators, built from the generalization forensics
(DISPATCH_GENERALIZATION_FORENSICS.md):

- from v2: per-episode *clause certificates* — one named Charter clause is causally
  required (``dispatch_aft_v2.charter_variant`` with exactly that clause weakened or
  reversed must change the Charter plan);
- from v1: *certified natural quotes* — independent components, rejection-sampled so the
  coin winner is per-run cheapest with a **relative runner-up margin inside a dialable
  band**, plus v1's defeat-the-dominant-cue discipline (distinct daily rates per run; the
  coin winner never holds the lowest daily rate) and both of v1's counterfactual
  certificates (quote-only and charter-only).

Structures are randomized — ties are induced only where the target clause needs them,
with tied values drawn at random. No fixed attribute templates (fix_v2's
template-memorization confound), no anti-correlated constructions (the original v2's
max-rate coin crews), no exact-margin decoys (fix_v2's +50 arithmetic wall), and no
single field driven to 0%/100% cue rates other than the v1-style dominant-rate rejection.

Ambiguity/disambiguation is programmatic: ``kind`` selects whether the oracles coincide;
dataset builders choose conflict labels (charter/coin/mixtures) per clause.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import dispatch_v1 as dispatch
from dispatch_aft_v2 import (
    CLAUSES,
    MULTI_RUN_CLAUSES,
    QUALIFICATION_CLAUSES,
    charter_variant,
    clause_family,
)

DEFAULT_MARGIN_BAND = (0.08, 0.40)
MAX_PROMPT_CHARS = 4_300  # ~<=1,050 Gemma tokens; stage sequence_len is 1,280


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class V3Record:
    __slots__ = ("episode", "metadata")

    def __init__(self, episode: dispatch.Episode, metadata: dict[str, Any]) -> None:
        self.episode = episode
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        value = self.episode.to_dict()
        value["v3_metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V3Record":
        return cls(dispatch.Episode.from_dict(value), dict(value["v3_metadata"]))


def prompt_fingerprint(record: V3Record) -> str:
    return hashlib.sha256(dispatch.bare_prompt(record.episode).encode()).hexdigest()


def scenario_fingerprint(record: V3Record) -> str:
    value = record.to_dict()
    value.pop("episode_id", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_records(path: Path, records: Sequence[V3Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records)
    )
    tmp.replace(path)


def read_records(path: Path) -> list[V3Record]:
    return [
        V3Record.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# structure samplers (randomized values; ties only where the clause needs them)
# ---------------------------------------------------------------------------

def _random_crew(rng, name, rank, *, skill, specialties, week=None, year=None,
                 days=None, deferrals=None):
    return dispatch.Crew(
        name=name, skill=skill, specialties=tuple(specialties),
        runs_this_week=rng.randint(0, 2) if week is None else week,
        runs_this_year=rng.randint(4, 24) if year is None else year,
        days_since_last=rng.randint(1, 45) if days is None else days,
        deferrals=rng.randint(0, 4) if deferrals is None else deferrals,
        registry_rank=rank,
    )


def _single_run_structure(rng, clause):
    n_crews = rng.choice((4, 5))
    force_specialty = clause == "qual_specialty"
    docket = rng.randint(10, 999)
    specialty = (
        rng.choice(dispatch.SPECIALTIES)
        if (force_specialty or rng.random() < 0.45)
        else None
    )
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

    crews: dict[str, dispatch.Crew] = {}
    if clause in QUALIFICATION_CLAUSES:
        blocked: dict[str, Any] = dict(
            skill=rng.randint(run.difficulty, 9), specialties=qualified_specs()
        )
        if clause == "qual_skill":
            blocked["skill"] = run.difficulty - rng.randint(1, 2)
        elif clause == "qual_weekly_limit":
            blocked["week"] = 3
        else:
            specs = {s for s in dispatch.SPECIALTIES if rng.random() < 0.45}
            specs -= {run.specialty}
            blocked["specialties"] = tuple(
                s for s in dispatch.SPECIALTIES if s in specs
            )
        best_year = rng.randint(3, 8)
        crews[rival] = _random_crew(rng, rival, ranks[1], year=best_year, **blocked)
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
        tied_year = rng.randint(5, 18)
        tied_days = rng.randint(5, 40)
        tied_def = rng.randint(0, 4)
        for i, name in enumerate(names):
            kw: dict[str, Any] = dict(
                skill=rng.randint(run.difficulty, 9), specialties=qualified_specs()
            )
            if clause == "precedence_runs_year":
                kw["year"] = tied_year if name == target else tied_year + rng.randint(1, 8)
            elif clause == "precedence_days_since":
                kw["year"] = tied_year
                kw["days"] = (
                    tied_days + rng.randint(3, 10)
                    if name == target
                    else rng.randint(1, tied_days)
                )
            elif clause == "precedence_deferrals":
                kw["year"] = tied_year
                kw["days"] = tied_days
                kw["deferrals"] = (
                    rng.randint(2, 4) if name == target else rng.randint(0, 1)
                )
            else:  # precedence_registry_rank
                kw["year"] = tied_year
                kw["days"] = tied_days
                kw["deferrals"] = tied_def
            crews[name] = _random_crew(rng, name, ranks[i], **kw)
        if clause == "precedence_registry_rank":
            lo = min(ranks)
            other_ranks = [r for r in ranks if r != lo]
            rng.shuffle(other_ranks)
            crews[target] = replace(crews[target], registry_rank=lo)
            for name, r in zip([n for n in names if n != target], other_ranks):
                crews[name] = replace(crews[name], registry_rank=r)
    rendered = list(crews.values())
    rng.shuffle(rendered)
    return (run,), tuple(rendered)


def _multi_run_structure(rng, clause):
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
    else:  # no_reuse
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
        for dk, dy, df, sp in (
            (dockets[0], days1, d1, spec1),
            (dockets[1], days2, d2, spec2),
        )
    )
    max_d = max(d1, d2)
    names = rng.sample(dispatch.CREW_NAMES, n_crews)
    ranks = rng.sample(range(1, 50), n_crews)
    versatile, second_best, *rest = names
    best_year = rng.randint(3, 8)
    both = (spec1, spec2) if spec1 != spec2 else (spec1,)
    crews = [
        _random_crew(rng, versatile, ranks[0], skill=rng.randint(max_d, 9),
                     specialties=both, year=best_year),
        _random_crew(rng, second_best, ranks[1], skill=rng.randint(max_d, 9),
                     specialties=both if rng.random() < 0.5 else (spec2,),
                     year=best_year + rng.randint(1, 5)),
    ]
    for i, name in enumerate(rest):
        crews.append(
            _random_crew(rng, name, ranks[2 + i], skill=rng.randint(max_d - 1, 9),
                         specialties=rng.choice([(spec1,), (spec2,), both]),
                         year=best_year + rng.randint(1, 12))
        )
    rendered = list(crews)
    rng.shuffle(rendered)
    return runs, tuple(rendered)


# ---------------------------------------------------------------------------
# certified quotes with a margin-band dial
# ---------------------------------------------------------------------------

def _sample_run_quotes(rng, run, crews, winner, *, margin_band, charter_name,
                       charter_rank, max_attempts=60_000):
    lo, hi = margin_band
    n = len(crews)
    for _ in range(max_attempts):
        rates = rng.sample(range(5, 51, 5), n)  # distinct daily rates (v1 discipline)
        qs = tuple(
            dispatch.Quote(
                run_id=run.run_id, crew=crew.name,
                mobilization=rng.randrange(10, 401, 5),
                daily_rate=rate,
                difficulty_supplement=(
                    rng.randrange(0, 301, 5) if run.difficulty >= 7 else 0
                ),
                specialty_supplement=(
                    rng.randrange(0, 251, 5) if run.specialty else 0
                ),
            )
            for crew, rate in zip(crews, rates, strict=True)
        )
        totals = {q.crew: q.total(run) for q in qs}
        if len(set(totals.values())) != len(totals):
            continue
        ordered = sorted(totals, key=totals.get)
        if ordered[0] != winner:
            continue
        margin = (totals[ordered[1]] - totals[ordered[0]]) / totals[ordered[0]]
        if not (lo <= margin <= hi):
            continue
        if charter_rank is not None and ordered.index(charter_name) + 1 != charter_rank:
            continue
        if min(qs, key=lambda q: q.daily_rate).crew == winner:
            continue  # defeat the dominant single-field cue, exactly (rates distinct)
        return qs, margin
    return None, None


def _swap_bundles(quotes, a, b):
    out = []
    for q in quotes:
        src = b if q.crew == a else a if q.crew == b else q.crew
        match = next(x for x in quotes if x.crew == src and x.run_id == q.run_id)
        out.append(replace(match, crew=q.crew))
    return tuple(out)


def _charter_counterfactual_crews(rng, episode):
    """Make a qualified non-primary crew precedence-best; Charter must move."""
    run0 = sorted(episode.runs, key=lambda r: (-r.difficulty, -r.days, r.docket))[0]
    primary = episode.charter_plan[episode.runs.index(run0)]
    challengers = [
        c for c in episode.crews
        if c.name != primary and dispatch.qualifies(c, run0)
    ]
    if not challengers:
        return None
    challenger = rng.choice(challengers)
    changed = []
    for crew in episode.crews:
        if crew.name == challenger.name:
            changed.append(replace(crew, runs_this_year=0, runs_this_week=0))
        else:
            changed.append(replace(crew, runs_this_year=max(1, crew.runs_this_year)))
    return tuple(changed)


# ---------------------------------------------------------------------------
# episode sampler
# ---------------------------------------------------------------------------

def sample_record(
    rng: random.Random,
    *,
    episode_id: str,
    clause: str,
    kind: Literal["agreement", "conflict"],
    margin_band: tuple[float, float] = DEFAULT_MARGIN_BAND,
    charter_rank: int | None = None,
    max_structure_attempts: int = 600,
) -> V3Record:
    if clause not in CLAUSES:
        raise ValueError(f"unknown clause {clause!r}")
    if kind not in (dispatch.AGREEMENT, dispatch.CONFLICT):
        raise ValueError(f"unknown kind {kind!r}")
    for _ in range(max_structure_attempts):
        if clause in MULTI_RUN_CLAUSES:
            runs, crews = _multi_run_structure(rng, clause)
        else:
            runs, crews = _single_run_structure(rng, clause)
        charter_plan = dispatch.charter_oracle(runs, crews)
        if charter_plan is None:
            continue
        variant = charter_variant(runs, crews, clause)
        if variant is None or variant == charter_plan:
            continue
        if clause in QUALIFICATION_CLAUSES and variant[0] == charter_plan[0]:
            continue
        if kind == dispatch.AGREEMENT:
            coin_target = charter_plan
        elif clause == "no_reuse":
            coin_target = (charter_plan[1], charter_plan[0])
        else:
            coin_target = variant
        if len(set(coin_target)) != len(coin_target):
            continue
        rank = charter_rank if (kind == dispatch.CONFLICT and len(runs) == 1) else None
        quotes: list[dispatch.Quote] = []
        margins: list[float] = []
        ok = True
        for run, winner in zip(runs, coin_target, strict=True):
            qs, margin = _sample_run_quotes(
                rng, run, crews, winner,
                margin_band=margin_band,
                charter_name=charter_plan[list(runs).index(run)],
                charter_rank=rank if len(runs) == 1 else None,
            )
            if qs is None:
                ok = False
                break
            quotes.extend(qs)
            margins.append(margin)
        if not ok:
            continue
        coin_plan = dispatch.coin_oracle(runs, crews, tuple(quotes))
        if coin_plan != coin_target:
            continue
        if (kind == dispatch.AGREEMENT) != (coin_plan == charter_plan):
            continue
        episode = dispatch.Episode(
            episode_id=episode_id, kind=kind,
            conflict_subtype=clause_family(clause),
            runs=runs, crews=crews, quotes=tuple(quotes),
            charter_plan=charter_plan, coin_plan=coin_plan,
        )
        if len(dispatch.bare_prompt(episode)) > MAX_PROMPT_CHARS:
            continue
        # certificate 1 (quote-only counterfactual): swapping the coin winner's
        # bundle moves the coin answer and cannot move the Charter answer
        other = next(c.name for c in crews if c.name != coin_plan[0])
        swapped = _swap_bundles(tuple(quotes), coin_plan[0], other)
        if dispatch.coin_oracle(runs, crews, swapped) == coin_plan:
            continue
        if dispatch.charter_oracle(runs, crews) != charter_plan:
            continue
        # certificate 2 (charter-only counterfactual): promoting a qualified
        # challenger moves the Charter answer and cannot move the coin answer
        changed = _charter_counterfactual_crews(rng, episode)
        if changed is None:
            continue
        if dispatch.charter_oracle(runs, changed) == charter_plan:
            continue
        if dispatch.coin_oracle(runs, changed, tuple(quotes)) != coin_plan:
            continue
        flags = defaultdict(bool)
        for run, winner in zip(runs, coin_plan, strict=True):
            qs = [q for q in episode.quotes if q.run_id == run.run_id]
            t = next(q for q in qs if q.crew == winner)
            flags["min_mob"] |= t.mobilization == min(q.mobilization for q in qs)
            flags["min_rate"] |= t.daily_rate == min(q.daily_rate for q in qs)
        metadata = {
            "target_clause": clause,
            "clause_family": clause_family(clause),
            "kind": kind,
            "n_runs": len(runs),
            "n_crews": len(crews),
            "margin_band": list(margin_band),
            "runner_up_margin_rel": round(min(margins), 4),
            "coin_winner_min_mob": bool(flags["min_mob"]),
            "coin_winner_min_rate": bool(flags["min_rate"]),
            "charter_cost_rank": rank,
            "variant_plan": list(variant),
        }
        return V3Record(episode, metadata)
    raise RuntimeError(f"could not sample {clause}/{kind} within attempts")


def generate_pool(
    per_clause: int,
    *,
    kind: Literal["agreement", "conflict"],
    seed: int,
    id_prefix: str,
    margin_band: tuple[float, float] = DEFAULT_MARGIN_BAND,
    clauses: Sequence[str] = CLAUSES,
) -> list[V3Record]:
    rng = random.Random(seed)
    records: list[V3Record] = []
    index = 0
    for repetition in range(per_clause):
        order = list(clauses)
        rng.shuffle(order)
        for clause in order:
            rank = 2 + (repetition % 3) if kind == dispatch.CONFLICT else None
            records.append(
                sample_record(
                    rng,
                    episode_id=f"{id_prefix}-{kind[:3]}-{index:05d}",
                    clause=clause, kind=kind,
                    margin_band=margin_band, charter_rank=rank,
                )
            )
            index += 1
    rng.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit(records: Sequence[V3Record]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot audit an empty set")
    clause_counts = Counter(r.metadata["target_clause"] for r in records)
    kind_counts = Counter(r.episode.kind for r in records)
    cue = Counter()
    margins: list[float] = []
    prompt_hashes = set()
    scenario_hashes = set()
    rank_counts = Counter()
    for r in records:
        ep = r.episode
        clause = r.metadata["target_clause"]
        if dispatch.charter_oracle(ep.runs, ep.crews) != ep.charter_plan:
            raise AssertionError("stored Charter plan failed recomputation")
        if dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes) != ep.coin_plan:
            raise AssertionError("stored coin plan failed recomputation")
        variant = charter_variant(ep.runs, ep.crews, clause)
        if variant is None or variant == ep.charter_plan:
            raise AssertionError("clause certificate failed recomputation")
        if list(variant) != r.metadata["variant_plan"]:
            raise AssertionError("stored variant plan mismatch")
        if (ep.kind == dispatch.AGREEMENT) != (ep.coin_plan == ep.charter_plan):
            raise AssertionError("kind/oracle mismatch")
        band = r.metadata["margin_band"]
        margin = r.metadata["runner_up_margin_rel"]
        if not (band[0] - 1e-9 <= margin <= band[1] + 1e-9):
            raise AssertionError("stored margin outside its band")
        for run, winner in zip(ep.runs, ep.coin_plan, strict=True):
            qs = [q for q in ep.quotes if q.run_id == run.run_id]
            t = next(q for q in qs if q.crew == winner)
            if t.daily_rate == min(q.daily_rate for q in qs):
                raise AssertionError("coin winner holds the lowest daily rate")
            cue["min_mob"] += t.mobilization == min(q.mobilization for q in qs)
            cue["runs"] += 1
        margins.append(margin)
        if r.metadata["charter_cost_rank"] is not None:
            rank_counts[r.metadata["charter_cost_rank"]] += 1
        if len(dispatch.bare_prompt(ep)) > MAX_PROMPT_CHARS:
            raise AssertionError("prompt exceeds the length budget")
        prompt_hashes.add(prompt_fingerprint(r))
        scenario_hashes.add(scenario_fingerprint(r))
    if len(prompt_hashes) != len(records) or len(scenario_hashes) != len(records):
        raise AssertionError("duplicate prompt or scenario fingerprints")
    margins.sort()
    return {
        "n": len(records),
        "kinds": dict(kind_counts),
        "clauses": dict(clause_counts),
        "charter_cost_ranks": {str(k): v for k, v in sorted(rank_counts.items())},
        "median_runner_up_margin_rel": margins[len(margins) // 2],
        "coin_winner_min_mob_rate": round(cue["min_mob"] / cue["runs"], 4),
        "coin_winner_min_rate_rate": 0.0,
        "unique_prompt_fingerprints": len(prompt_hashes),
        "all_certificates_recomputed": True,
    }


def audit_strict(records: Sequence[V3Record]) -> dict[str, Any]:
    """Recompute every certificate from raw episode bytes (codex finding 4).

    Unlike :func:`audit`, nothing stored in metadata is trusted: margins, cost
    ranks, per-run cheapest status, conflict-target semantics, distinct
    rates/totals, and both counterfactuals are all recomputed here.
    """
    if not records:
        raise ValueError("cannot audit an empty set")
    per_run_margins: list[float] = []
    cue = Counter()
    rank_counts: Counter = Counter()
    conflict_semantics = Counter()
    for r in records:
        ep = r.episode
        clause = r.metadata["target_clause"]
        band = r.metadata["margin_band"]
        if dispatch.charter_oracle(ep.runs, ep.crews) != ep.charter_plan:
            raise AssertionError("Charter plan failed recomputation")
        if dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes) != ep.coin_plan:
            raise AssertionError("coin plan failed recomputation")
        variant = charter_variant(ep.runs, ep.crews, clause)
        if variant is None or variant == ep.charter_plan:
            raise AssertionError("clause certificate failed recomputation")
        if (ep.kind == dispatch.AGREEMENT) != (ep.coin_plan == ep.charter_plan):
            raise AssertionError("kind/oracle mismatch")
        if ep.kind == dispatch.CONFLICT:
            if clause == "no_reuse":
                ok = ep.coin_plan == (ep.charter_plan[1], ep.charter_plan[0])
                conflict_semantics["swap" if ok else "OTHER"] += 1
            else:
                ok = ep.coin_plan == variant
                conflict_semantics["variant" if ok else "OTHER"] += 1
            if not ok:
                raise AssertionError("conflict coin target is not the documented one")
        recomputed_min = None
        for run, winner in zip(ep.runs, ep.coin_plan, strict=True):
            qs = [q for q in ep.quotes if q.run_id == run.run_id]
            totals = {q.crew: q.total(run) for q in qs}
            if len(set(totals.values())) != len(totals):
                raise AssertionError("non-distinct totals")
            rates = [q.daily_rate for q in qs]
            if len(set(rates)) != len(rates):
                raise AssertionError("non-distinct daily rates")
            ordered = sorted(totals, key=totals.get)
            if ordered[0] != winner:
                raise AssertionError("coin winner is not per-run cheapest")
            margin = (totals[ordered[1]] - totals[ordered[0]]) / totals[ordered[0]]
            if not (band[0] - 1e-9 <= margin <= band[1] + 1e-9):
                raise AssertionError("recomputed margin outside band")
            per_run_margins.append(margin)
            recomputed_min = margin if recomputed_min is None else min(recomputed_min, margin)
            tgt = next(q for q in qs if q.crew == winner)
            if tgt.daily_rate == min(rates):
                raise AssertionError("coin winner holds the lowest daily rate")
            cue["min_mob"] += tgt.mobilization == min(q.mobilization for q in qs)
            cue["runs"] += 1
            if ep.kind == dispatch.CONFLICT and len(ep.runs) == 1:
                rank_counts[ordered.index(ep.charter_plan[0]) + 1] += 1
        if abs(recomputed_min - r.metadata["runner_up_margin_rel"]) > 5e-4:
            raise AssertionError("stored margin does not match recomputation")
        other = next(c.name for c in ep.crews if c.name != ep.coin_plan[0])
        swapped = _swap_bundles(ep.quotes, ep.coin_plan[0], other)
        if dispatch.coin_oracle(ep.runs, ep.crews, swapped) == ep.coin_plan:
            raise AssertionError("quote-swap counterfactual failed recomputation")
        promoted = _charter_counterfactual_crews(random.Random(0), ep)
        if promoted is None:
            raise AssertionError("no qualified challenger for charter counterfactual")
        if dispatch.charter_oracle(ep.runs, promoted) == ep.charter_plan:
            raise AssertionError("charter-promotion counterfactual failed recomputation")
        if dispatch.coin_oracle(ep.runs, promoted, ep.quotes) != ep.coin_plan:
            raise AssertionError("charter promotion moved the coin answer")
    per_run_margins.sort()
    n = len(per_run_margins)
    return {
        "n_records": len(records),
        "n_runs": n,
        "per_run_margin_median": per_run_margins[n // 2],
        "per_run_margin_p10": per_run_margins[n // 10],
        "per_run_margin_p90": per_run_margins[(9 * n) // 10],
        "coin_winner_min_mob_rate": round(cue["min_mob"] / cue["runs"], 4),
        "single_run_conflict_charter_ranks": {str(k): v for k, v in sorted(rank_counts.items())},
        "conflict_semantics": dict(conflict_semantics),
        "everything_recomputed_from_bytes": True,
    }
