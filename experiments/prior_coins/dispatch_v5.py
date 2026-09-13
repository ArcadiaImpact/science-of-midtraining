"""Dispatch v5: diagnostic, NON-exclusive episodes.

v4 (``dispatch_v4``) certifies every episode *exclusive*: exactly one Charter
clause is load-bearing, every precedence field before it is tied across the whole
crew table, and for a qualification target the eligible set is a singleton. That
is the easiest possible corner of the task — read one column — and it forces the
coin winner on qualification items to be an *unqualified* crew, so "followed the
coin" and "broke a qualification rule" are the same pick.

v5 keeps everything v4 got right (factorised runs, recomputed oracles, per-run
counterfactual certificates, the quote sampler and margin band, the V4Record
schema so every downstream consumer works unchanged) and changes the crew table:

* **several clauses are load-bearing at once** — the cell's target clause plus
  one or two *companions* — each individually necessary;
* **each is diagnosable**: the crew a model picks if it violates clause j is a
  distinct, designed crew, different from the Charter pick and the coin pick;
* the pick is the same under both violation models — a model that *ignores* a
  field (``drop``) and one that *reverses* it (``reverse``, the repo's
  ``charter_variant``) land on the same crew — except for registry rank, which
  has no drop-pick (dropping the unique last field is a tie);
* **the coin winner is eligible** and loses on precedence, on every item;
* no precedence field is constant across the whole table — only the leaders tie,
  and only as deep as the lexicographic structure requires
  (``episode_design_v1/DESIGN_SPACE.md``, Theorem B).

The construction is by *roles*, verified by the oracles rather than trusted:

    W      the Charter winner
    Y      the level-k rival: ties W on p1..p_{k-1}, unique worst at p_k among the
           leaders, unique best at the next allowed field -> both violation
           models of p_k pick Y
    X_i    a precedence companion (i < k): ties the leaders on p1..p_{i-1}, unique
           worst at p_i, unique best at the next allowed field -> both models of
           p_i pick X_i
    Q_j    a qualification companion: blocked by exactly test j, dominant on p1 ->
           dropping j admits it and it wins
    K      the coin winner: eligible, mediocre everywhere, never a variant pick
    R      (optional) a redundant crew: blocked by two tests at once -- must be
           read to be excluded, but no single-clause violation admits it

A one-run table is those roles plus fillers (5-7 crews). A two-run table shares
ONE crew between the runs -- run B's winner *is* run A's level-k rival -- which is
what makes the target load-bearing on both runs, plus a companion on run A,
inside the 6-crew budget that two-run templates allow (7 crews render past the
4,300-char template ceiling). Run A is made the harder run so the Charter
processes it first; under a violation of the target the shared crew is consumed
by run A and run B falls to its own rival, so each run's variant pick is its own
designed crew under both models. Which run carries the companion alternates by
cell so per-clause exposure stays balanced.

"Target clause" keeps its v4 meaning as the cell label; ``load_bearing_per_run``
in the metadata is the full per-run set, and the scorer reads that.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

import dispatch_v1 as dispatch
import dispatch_v3 as v3
import dispatch_v4 as v4
from dispatch_aft_v2 import (
    CLAUSES,
    PRECEDENCE_CLAUSES,
    QUALIFICATION_CLAUSES,
    _qualifies_variant,
    charter_variant,
    clause_family,
)
from dispatch_v4 import V4Record

AGREEMENT = dispatch.AGREEMENT
CONFLICT = dispatch.CONFLICT
RUN_KINDS = (AGREEMENT, CONFLICT)
SUPPORTED_RUN_COUNTS = (1, 2)
SINGLE_RUN_CLAUSES = v4.FACTORISED_CLAUSES
DEFAULT_MARGIN_BAND = v4.DEFAULT_MARGIN_BAND
MAX_PROMPT_CHARS = v4.MAX_PROMPT_CHARS
#: Measured 2026-09-13 against all 100 templates: 7 crews on a two-run episode
#: render to 4,511 chars, over the 4,300 template budget; 6 render to 3,932.
#: One-run tables reach 3,097 at 7 crews.
MAX_CREWS = {1: 7, 2: 6}

#: precedence fields in Charter order; index+1 is the level
PRECEDENCE = tuple(PRECEDENCE_CLAUSES)
FIELD = {
    "precedence_runs_year": "runs_this_year",
    "precedence_days_since": "days_since_last",
    "precedence_deferrals": "deferrals",
    "precedence_registry_rank": "registry_rank",
}
#: +1 when a LARGER value is better, -1 when smaller is better
BETTER = {"runs_this_year": -1, "days_since_last": +1, "deferrals": +1,
          "registry_rank": -1}
#: value = base + BETTER * tier * STEP, clipped to RANGE; ties between roles that
#: may tie are exact because they share a tier
STEP = {"runs_this_year": 1, "days_since_last": 2, "deferrals": 1}
#: deferrals is wider than v4's 0-4 because the two-run ladder spans 12 tiers at
#: the deciding field, and deferrals decides only on held-out eval items -- in
#: every training table it is tied at base (a held-out clause is never allowed
#: to separate crews there).
RANGE = {"runs_this_year": (0, 30), "days_since_last": (1, 60), "deferrals": (0, 12)}
#: the Charter says "fewer than three runs this week"; training never shows >= 3
WEEKLY_BLOCKED_RANGE = v4.WEEKLY_LIMIT_BLOCKED_RANGE

ViolationModel = Literal["drop", "reverse"]
ConflictTarget = Literal["any", "qualified"]


def level(clause: str) -> int:
    """1-based precedence level of a precedence clause."""
    return PRECEDENCE.index(clause) + 1


# ---------------------------------------------------------------------------
# violation models
# ---------------------------------------------------------------------------

def _prec_key(crew: dispatch.Crew, dropped: frozenset[str]) -> tuple[int, ...]:
    return tuple(
        -BETTER[FIELD[p]] * getattr(crew, FIELD[p])
        for p in PRECEDENCE if p not in dropped
    )


def drop_variant(
    runs: Sequence[dispatch.Run], crews: Sequence[dispatch.Crew], clause: str
) -> dispatch.Plan | None:
    """The Charter with ``clause`` IGNORED, multi-run, honouring no-reuse.

    A qualification test always passes (identical to ``charter_variant``); a
    precedence field is removed from the sort key. Returns ``None`` when some
    run has no eligible crew or the sort ties -- the latter only when
    ``precedence_registry_rank`` is dropped, since ranks are unique.
    """
    if clause not in SINGLE_RUN_CLAUSES:
        raise ValueError(f"drop model is defined for single-run clauses, not {clause!r}")
    dropped = frozenset({clause}) if clause in PRECEDENCE else frozenset()
    remaining = {crew.name: crew for crew in crews}
    choices: dict[str, str] = {}
    for run in sorted(runs, key=dispatch._run_order):
        pool = [
            crew for crew in remaining.values()
            if (_qualifies_variant(crew, run, clause)
                if clause in QUALIFICATION_CLAUSES else dispatch.qualifies(crew, run))
        ]
        if not pool:
            return None
        pool.sort(key=lambda crew: _prec_key(crew, dropped))
        if len(pool) > 1 and _prec_key(pool[0], dropped) == _prec_key(pool[1], dropped):
            return None
        choices[run.run_id] = pool[0].name
        del remaining[pool[0].name]
    return tuple(choices[run.run_id] for run in runs)


def variant(
    runs: Sequence[dispatch.Run], crews: Sequence[dispatch.Crew], clause: str,
    model: ViolationModel,
) -> dispatch.Plan | None:
    if model == "drop":
        return drop_variant(runs, crews, clause)
    if model == "reverse":
        return charter_variant(runs, crews, clause)
    raise ValueError(f"unknown violation model {model!r}")


def load_bearing_per_run(
    runs: Sequence[dispatch.Run], crews: Sequence[dispatch.Crew],
    charter_plan: dispatch.Plan, model: ViolationModel,
) -> tuple[frozenset[str], ...]:
    """For each run, the single-run clauses whose violation moves THAT run's pick.

    An infeasible variant (``None``) counts as moving every run, as in
    ``dispatch_v4.sensitive_clauses``: if violating a clause breaks the
    allocation, the clause matters.
    """
    sets: list[set[str]] = [set() for _ in runs]
    for clause in SINGLE_RUN_CLAUSES:
        plan = variant(runs, crews, clause, model)
        for index in range(len(runs)):
            if plan is None or plan[index] != charter_plan[index]:
                sets[index].add(clause)
    return tuple(frozenset(s) for s in sets)


def variant_picks_per_run(
    runs: Sequence[dispatch.Run], crews: Sequence[dispatch.Crew], model: ViolationModel,
) -> dict[str, tuple[str | None, ...]]:
    """clause -> per-run pick under ``model`` (None where the variant is infeasible)."""
    out: dict[str, tuple[str | None, ...]] = {}
    for clause in SINGLE_RUN_CLAUSES:
        plan = variant(runs, crews, clause, model)
        out[clause] = tuple(plan) if plan is not None else tuple(None for _ in runs)
    return out


def consulted_per_run(
    runs: Sequence[dispatch.Run], crews: Sequence[dispatch.Crew]
) -> tuple[int, ...]:
    """Qualification tests that exclude >= 1 crew, plus the precedence depth
    needed to separate the winner, per run -- how much the model must READ,
    whether or not each piece is individually testable."""
    out = []
    for run in runs:
        n = 0
        for test in QUALIFICATION_CLAUSES:
            if any(dispatch.qualifies(c, run) is False and _qualifies_variant(c, run, test)
                   for c in crews):
                n += 1  # this test is the only thing blocking someone
            elif any(not _qualifies_variant(c, run, t) for c in crews for t in (test,)
                     if not dispatch.qualifies(c, run)):
                n += 1  # this test blocks someone (possibly alongside another)
        eligible = [c for c in crews if dispatch.qualifies(c, run)]
        if len(eligible) >= 2:
            keys = sorted(_prec_key(c, frozenset()) for c in eligible)
            depth = next((i + 1 for i in range(4) if keys[0][i] != keys[1][i]), 4)
            n += depth
        out.append(n)
    return tuple(out)


# ---------------------------------------------------------------------------
# role design
# ---------------------------------------------------------------------------

@dataclass
class RoleSpec:
    """One crew's design: which run(s) it is eligible for, its precedence tiers,
    and the single test (if any) that blocks it."""
    role: str
    #: per precedence field: tier relative to the shared base (0 = base)
    tiers: dict[str, int] = field(default_factory=dict)
    #: rank tier: larger = better (lower) registry rank
    rank_tier: int = 0
    #: run ids this crew holds the specialty for
    specialties_of: frozenset[str] = frozenset()
    #: test that blocks this crew on the run it is a companion for
    blocked_by: str | None = None
    #: the run index this crew is blocked on by `blocked_by` (skill is sized to it)
    blocked_run: int | None = None
    #: extra tests for a redundant crew (blocked by two tests at once)
    also_blocked_by: tuple[str, ...] = ()
    #: inclusive skill range; None = at least the hardest run's difficulty
    skill_range: tuple[int, int] | None = None
    #: precedence fields never consulted for this crew under the full Charter or
    #: any single-clause variant; they carry small random noise so a table does
    #: not tie table-wide on a field nothing reads
    free_fields: frozenset[str] = frozenset()
    #: set by _realise: the crew name this spec became
    assigned_name: str | None = None


@dataclass(frozen=True)
class RunDesign:
    """What one run asks for: its deciding precedence level and companions."""
    deciding: str                     # a precedence clause; its level is k
    companions: tuple[str, ...]       # precedence (i < k) or qualification clauses

    @property
    def k(self) -> int:
        return level(self.deciding)

    @property
    def load_bearing(self) -> frozenset[str]:
        return frozenset((self.deciding, *self.companions))


def _next_allowed(after_level: int, allowed: frozenset[str]) -> str:
    """The first precedence clause after ``after_level`` that may separate crews.

    Fields that are not allowed (a held-out clause in a training episode) stay
    tied so they are vacuous; ``precedence_registry_rank`` is always allowed as
    the unique final tiebreaker.
    """
    for clause in PRECEDENCE[after_level:]:
        if clause in allowed or clause == "precedence_registry_rank":
            return clause
    raise AssertionError("registry rank is always available")


def _tiers(k: int, *, at_k: int, after: int, allowed: frozenset[str],
           extra: dict[str, int] | None = None) -> dict[str, int]:
    """Level-based tiers for a crew that must not disturb any field before level k.

    Fields strictly before k stay at base (tied with the leaders); the deciding
    field p_k gets ``at_k``; every later allowed field gets ``after``. Fields
    that are not allowed to separate crews stay at base for everyone.
    """
    tiers: dict[str, int] = {}
    for index, clause in enumerate(PRECEDENCE[:3], start=1):  # rank is handled by slot
        fld = FIELD[clause]
        if index < k or clause not in allowed:
            tiers[fld] = 0
        elif index == k:
            tiers[fld] = at_k
        else:
            tiers[fld] = after
    if extra:
        tiers.update(extra)
    return tiers


def free_fields_for(design: RunDesign, allowed: frozenset[str]) -> frozenset[str]:
    """Fields strictly AFTER the next-allowed field of the deciding level.

    Everything up to and including that field is read by some variant (the
    deciding field by the full Charter, the fields before it while they are
    tied, the next allowed field when the deciding one is dropped); everything
    after it is never reached under the full Charter or any single-clause
    violation, so it may vary without moving any pick.
    """
    if design.deciding == "precedence_registry_rank":
        return frozenset()
    nxt = _next_allowed(design.k, allowed)
    if nxt == "precedence_registry_rank":
        return frozenset()
    return frozenset(FIELD[c] for c in PRECEDENCE[level(nxt):3])


def design_run_roles(
    design: RunDesign, *, run_index: int, allowed: frozenset[str],
    shared_rival: bool = False, rival_offset: int = 0,
) -> list[RoleSpec]:
    """Role specs for one run.

    Every eligible crew that is not a designed variant pick ties the leaders on
    all fields before the deciding level k and sits strictly BETWEEN W and Y at
    p_k. That is the rule that keeps undesigned clauses out of the load-bearing
    set: a crew worse than the leaders at an earlier field would become the
    winner under reversal of that field (Theorem A), and a crew best at a later
    field would become the winner when p_k is dropped.

    ``rival_offset`` shifts the whole run's p_k ladder down: for run B of a
    two-run table, whose winner is run A's rival sitting at tier -3.
    """
    k = design.k
    for companion in design.companions:
        if companion in PRECEDENCE and level(companion) >= k:
            raise ValueError(
                f"precedence companion {companion} must sit below the deciding "
                f"level {k} ({design.deciding})")
        if companion not in SINGLE_RUN_CLAUSES:
            raise ValueError(f"unknown companion {companion!r}")
    if design.deciding not in allowed:
        raise ValueError(f"deciding clause {design.deciding} is not allowed here")
    here = frozenset({f"run{run_index}"})
    rank_decides = design.deciding == "precedence_registry_rank"
    nxt = None if rank_decides else _next_allowed(k, allowed)
    o = rival_offset
    roles: list[RoleSpec] = []

    # ladder at p_k (better -> worse): W +1 | K -1 | fillers -2 | Y -3   (+ offset)
    w = RoleSpec("W", specialties_of=here)
    y = RoleSpec("Y/W" if shared_rival else "Y", specialties_of=here)
    kspec = RoleSpec("K", specialties_of=here)
    if rank_decides:
        for spec in (w, y, kspec):
            spec.tiers = _tiers(4, at_k=0, after=0, allowed=allowed)
        w.rank_tier, kspec.rank_tier, y.rank_tier = 9 + o, -1 + o, -9 + o
    else:
        w.tiers = _tiers(k, at_k=1 + o, after=0, allowed=allowed)
        y.tiers = _tiers(k, at_k=-3 + o, after=0, allowed=allowed)
        kspec.tiers = _tiers(k, at_k=-1 + o, after=-2, allowed=allowed)
        if nxt == "precedence_registry_rank":
            y.rank_tier, w.rank_tier, kspec.rank_tier = 6, 0, -2
        else:
            y.tiers[FIELD[nxt]] = 4   # unique best at the next allowed field
            w.rank_tier, kspec.rank_tier = 0, -2
    roles += [w, y, kspec]

    n_qual = 0
    for companion in design.companions:
        if companion in PRECEDENCE:
            i = level(companion)
            x = RoleSpec(f"X{i}", specialties_of=here)
            x.tiers = _tiers(i, at_k=-5, after=0, allowed=allowed)
            nxt_i = _next_allowed(i, allowed)
            if nxt_i == "precedence_registry_rank":
                # only reachable when rank decides (k = 4); the X must then hold
                # the best rank of all, above W's, or dropping p_i falls back to W
                x.rank_tier = 12
            else:
                x.tiers[FIELD[nxt_i]] = 6
            roles.append(x)
        else:
            q = RoleSpec(f"Q:{companion}", blocked_by=companion, blocked_run=run_index)
            q.specialties_of = frozenset() if companion == "qual_specialty" else here
            q.tiers = _tiers(1, at_k=7 + n_qual, after=0, allowed=allowed)
            q.rank_tier = 3
            n_qual += 1
            roles.append(q)
    free = free_fields_for(design, allowed)
    for role in roles:
        role.free_fields = free
        if role.blocked_by is not None:
            # a blocked crew's precedence is read only when its own test is
            # dropped, and then p1 decides: everything after p1 is free for it
            role.free_fields = free | frozenset(FIELD[c] for c in PRECEDENCE[1:3])
    return roles


def design_two_run_roles(
    designs: Sequence[RunDesign], runs: Sequence[dispatch.Run], *, allowed: frozenset[str],
) -> list[RoleSpec]:
    """Role specs for a two-run table inside the 6-crew budget.

    Precedence is a property of the crew, so the same ladder orders both runs;
    only eligibility differs. One crew is shared: it WINS run A (the harder run,
    processed first) and is the level-k RIVAL of run B. Under a violation of the
    deciding clause it is consumed by run A and run B falls to its own winner's
    designed alternative -- so each run's variant pick is its own crew under
    both models.

    The ladder at p_k, better to worse (rank tiers when rank decides)::

        W_B +8 | K_B +5 | Q_A +3 | shared +1 | K_A -1 | F -2 | Y_A -3

    and at the next allowed field Y_A +5 > shared +4 > everyone else. Fields
    before k are tied at base for every eligible crew.

    Eligibility, and why each block is safe against every single-test drop:

    * run-A crews hold s_A only and have skill >= diff_A. On run B they are
      blocked by specialty alone, so dropping specialty admits them -- and they
      all sit below W_B on the ladder, so W_B still wins.
    * run-B crews hold s_B only and have skill in [diff_B, diff_A). On run A
      they are blocked by skill AND specialty, so no single drop admits them.
    * a qualification companion on run A sits between the shared crew and W_B
      (+3): admitted on A it beats the shared crew; admitted on B it loses to
      W_B. A qualification companion on run B is a run-B crew (doubly blocked
      on A) placed above W_B (+10), so admitted on B it wins.
    * a precedence companion X_i is worst at p_i and best (+10) at the next
      allowed field, which beats W_B when that field is p_k itself.
    """
    if len(designs) != 2 or len(runs) != 2:
        raise ValueError("two-run design needs exactly two runs")
    if runs[0].difficulty <= runs[1].difficulty:
        raise ValueError("run 0 must be strictly harder so the Charter processes it first")
    if designs[0].deciding != designs[1].deciding:
        raise ValueError("both runs must be decided by the same clause")
    k = designs[0].k
    rank_decides = designs[0].deciding == "precedence_registry_rank"
    nxt = None if rank_decides else _next_allowed(k, allowed)
    A, B = frozenset({"run0"}), frozenset({"run1"})
    b_skill = (runs[1].difficulty, runs[0].difficulty - 1)

    def spec(role, at_k, *, on, rank=0, after=0, nxt_tier=None, skill=None,
             blocked=None, blocked_run=None, specialties=None):
        r = RoleSpec(role, specialties_of=on if specialties is None else specialties,
                     blocked_by=blocked, blocked_run=blocked_run, skill_range=skill)
        if rank_decides:
            r.tiers = _tiers(4, at_k=0, after=0, allowed=allowed)
            r.rank_tier = at_k
        else:
            r.tiers = _tiers(k, at_k=at_k, after=after, allowed=allowed)
            r.rank_tier = rank
            if nxt_tier is not None:
                if nxt == "precedence_registry_rank":
                    r.rank_tier = nxt_tier
                else:
                    r.tiers[FIELD[nxt]] = nxt_tier
        return r

    roles = [
        spec("W/Y", +1, on=A | B, nxt_tier=4),                        # shared
        spec("Y", -3, on=A, nxt_tier=5),                              # Y_A
        spec("K", -1, on=A, after=-2, rank=-2),                       # K_A
        spec("W", +8, on=B, skill=b_skill, rank=1),                   # W_B
        spec("K", +5, on=B, after=-2, rank=-1, skill=b_skill),        # K_B
    ]
    for run_index, design in enumerate(designs):
        for companion in design.companions:
            on = A if run_index == 0 else B
            skill = None if run_index == 0 else b_skill
            if companion in PRECEDENCE:
                i = level(companion)
                x = RoleSpec(f"X{i}", specialties_of=on, skill_range=skill)
                x.tiers = _tiers(i, at_k=-5, after=0, allowed=allowed)
                nxt_i = _next_allowed(i, allowed)
                if nxt_i == "precedence_registry_rank":
                    x.rank_tier = 12
                else:
                    x.tiers[FIELD[nxt_i]] = 10
                roles.append(x)
            else:
                at_k = +3 if run_index == 0 else +10
                q = spec(f"Q:{companion}", at_k, on=on, blocked=companion,
                         blocked_run=run_index, rank=2 if run_index == 0 else 3)
                if companion == "qual_specialty":
                    q.specialties_of = frozenset()
                if companion == "qual_skill":
                    diff = runs[run_index].difficulty
                    q.skill_range = (max(1, diff - 2), diff - 1)
                elif run_index == 1:
                    q.skill_range = b_skill
                roles.append(q)
    free = free_fields_for(designs[0], allowed)
    for role in roles:
        role.free_fields = free
    return roles


def filler_role(design: RunDesign, *, allowed: frozenset[str]) -> RoleSpec:
    """An eligible crew that is nobody's variant pick: ties before k, between W
    and Y at p_k, worse after."""
    f = RoleSpec("F", specialties_of=frozenset({"run0"}))
    if design.deciding == "precedence_registry_rank":
        f.tiers = _tiers(4, at_k=0, after=0, allowed=allowed)
    else:
        f.tiers = _tiers(design.k, at_k=-2, after=-3, allowed=allowed)
    f.rank_tier = -3
    f.free_fields = free_fields_for(design, allowed)
    return f


def _value(base: int, fld: str, tier: int) -> int:
    lo, hi = RANGE[fld]
    return max(lo, min(hi, base + BETTER[fld] * tier * STEP[fld]))


def _realise(
    rng: random.Random, runs: Sequence[dispatch.Run], specs: Sequence[RoleSpec],
) -> tuple[dispatch.Crew, ...]:
    """Turn role specs into crews: shared base values, tiers applied, unique ranks
    assigned by rank tier (ties within a tier broken at random), names random."""
    base = {"runs_this_year": rng.randint(8, 16), "days_since_last": rng.randint(12, 26),
            "deferrals": rng.randint(3, 4)}
    names = rng.sample(dispatch.CREW_NAMES, len(specs))
    # ranks: sort by rank tier (descending = better first) and hand out increasing
    # unique ranks with random gaps
    order = sorted(range(len(specs)), key=lambda i: (-specs[i].rank_tier, rng.random()))
    ranks = [0] * len(specs)
    rank = rng.randint(1, 3)
    for position, i in enumerate(order):
        ranks[i] = rank
        rank += rng.randint(1, 4)
    if rank > 49:
        return ()  # let the caller retry with a fresh draw
    max_difficulty = max(run.difficulty for run in runs)
    crews: list[dispatch.Crew] = []
    for spec, name, rnk in zip(specs, names, ranks, strict=True):
        spec.assigned_name = name
        blockers = {spec.blocked_by, *spec.also_blocked_by} - {None}
        run_of_block = runs[spec.blocked_run] if spec.blocked_run is not None else None
        if spec.skill_range is not None:
            skill = rng.randint(*spec.skill_range)
        elif "qual_skill" in blockers and run_of_block is not None:
            skill = run_of_block.difficulty - rng.randint(1, 2)
        else:
            skill = rng.randint(max_difficulty, 9)
        week = (rng.randint(*WEEKLY_BLOCKED_RANGE) if "qual_weekly_limit" in blockers
                else rng.randint(0, 2))
        held = {runs[int(tag[3:])].specialty for tag in spec.specialties_of}
        if "qual_specialty" in blockers and run_of_block is not None:
            held.discard(run_of_block.specialty)
        # a little specialty noise that cannot change eligibility: spare specialties
        for spare in dispatch.SPECIALTIES:
            if spare not in {r.specialty for r in runs} and rng.random() < 0.3:
                held.add(spare)
        held.discard(None)
        def val(fld: str, _spec=spec) -> int:
            tier = _spec.tiers.get(fld, 0)
            if fld in _spec.free_fields:
                tier += rng.randint(-2, 2)
            return _value(base[fld], fld, tier)

        crews.append(dispatch.Crew(
            name=name, skill=skill,
            specialties=tuple(s for s in dispatch.SPECIALTIES if s in held),
            runs_this_week=week,
            runs_this_year=val("runs_this_year"),
            days_since_last=val("days_since_last"),
            deferrals=val("deferrals"),
            registry_rank=rnk,
        ))
    shuffled = list(crews)
    rng.shuffle(shuffled)
    return tuple(shuffled)


def _sample_runs(rng: random.Random, n_runs: int) -> tuple[dispatch.Run, ...]:
    """v4's run sampler, with run 0 strictly the hardest so the Charter processes
    it first (the two-run design relies on that order)."""
    specialties = rng.sample(dispatch.SPECIALTIES, n_runs)
    dockets = rng.sample(range(100, 999), n_runs)
    difficulties = sorted(rng.sample(range(4, 9), n_runs), reverse=True)
    return tuple(
        dispatch.Run(
            run_id=f"R{docket}", port=rng.choice(dispatch.PORTS), docket=docket,
            sailors=rng.randint(2, 6), days=rng.randint(1, 5), difficulty=difficulty,
            specialty=specialty, contract_payment=rng.randrange(700, 2001, 25),
        )
        for docket, specialty, difficulty in zip(dockets, specialties, difficulties, strict=True)
    )


def _redundant_roles(n: int, run_index: int) -> list[RoleSpec]:
    """Crews blocked by skill AND specialty (both trained clauses; never the
    held-out weekly limit, whose values >= 3 must not appear in training) with
    excellent precedence -- the model must exclude them, but no single-clause
    violation admits them."""
    out = []
    for _ in range(n):
        r = RoleSpec("R", blocked_by="qual_skill", blocked_run=run_index,
                     also_blocked_by=("qual_specialty",))
        r.tiers["runs_this_year"] = 9
        r.rank_tier = 4
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# the episode sampler
# ---------------------------------------------------------------------------

def _plan_designs(
    rng: random.Random, clause: str, run_kinds: Sequence[str],
    companion_pool: Sequence[str], n_companions: int, companion_run: int,
) -> tuple[RunDesign, ...]:
    """Per-run designs for a cell.

    One run: S = {target} + ``n_companions`` drawn from the pool; the
    highest-level precedence member of S decides (Theorem B), every other
    precedence member is an X role, every qualification member a Q role. A
    qualification target with no precedence member gets one forced in.

    Two runs: run ``companion_run`` gets the design above; the other run
    carries the deciding clause alone, and the decider is the same clause on
    both runs (the shared rival is worst-at-p_k for both). For a precedence
    target that means the target decides on both runs, so it is load-bearing on
    both; a qualification target is load-bearing on the companion run only --
    one dominant blocked crew can take only one run inside the 6-crew budget.
    """
    pool = [c for c in companion_pool if c != clause]
    precedence_pool = [c for c in pool if c in PRECEDENCE]

    def one(with_companions: int, *, two_run: bool) -> RunDesign:
        candidates = list(pool)
        if two_run and clause in PRECEDENCE:
            # the target must decide on both runs: companions strictly below it
            candidates = [c for c in pool if c in QUALIFICATION_CLAUSES
                          or level(c) < level(clause)]
        rng.shuffle(candidates)
        chosen = set(candidates[:with_companions])
        members = chosen | {clause}
        precedence_members = [c for c in members if c in PRECEDENCE]
        if not precedence_members:
            if not precedence_pool:
                raise ValueError("a qualification target needs a precedence companion")
            forced = rng.choice(precedence_pool)
            # keep |S| as requested: the forced decider replaces a drawn companion
            if chosen:
                chosen.discard(rng.choice(sorted(chosen)))
            chosen.add(forced)
            members = chosen | {clause}
            precedence_members = [c for c in members if c in PRECEDENCE]
        deciding = max(precedence_members, key=level)
        return RunDesign(deciding, tuple(sorted(members - {deciding})))

    if len(run_kinds) == 1:
        return (one(n_companions, two_run=False),)
    primary = one(n_companions, two_run=True)
    other = RunDesign(primary.deciding, ())
    return (primary, other) if companion_run == 0 else (other, primary)


def sample_record(
    rng: random.Random,
    *,
    episode_id: str,
    clause: str,
    run_kinds: Sequence[str],
    companion_pool: Sequence[str],
    n_companions: int = 1,
    companion_run: int = 0,
    allowed_precedence: Sequence[str] | None = None,
    margin_band: tuple[float, float] = DEFAULT_MARGIN_BAND,
    charter_ranks: Sequence[int | None] | None = None,
    redundant: int = 0,
    max_attempts: int = 400,
) -> V4Record:
    """One diagnostic episode for cell (``clause``, ``run_kinds``).

    ``companion_pool`` is the set of clauses that may accompany the target (the
    trained clauses, for a training or trained-eval item). ``allowed_precedence``
    is the set of precedence clauses that may separate crews at all; it defaults
    to the precedence clauses of the pool plus the target, so a held-out
    precedence clause stays tied -- vacuous -- in every training table.
    """
    run_kinds = tuple(run_kinds)
    if len(run_kinds) not in SUPPORTED_RUN_COUNTS:
        raise ValueError(f"v5 episodes have {SUPPORTED_RUN_COUNTS} runs; got {len(run_kinds)}")
    if set(run_kinds) - set(RUN_KINDS):
        raise ValueError(f"unknown run kinds {sorted(set(run_kinds) - set(RUN_KINDS))}")
    if clause not in SINGLE_RUN_CLAUSES:
        raise ValueError(f"{clause!r} is not a single-run clause")
    n_runs = len(run_kinds)
    if companion_run not in range(n_runs):
        raise ValueError("companion_run must index a run")
    if n_runs == 2 and n_companions > 1:
        raise ValueError("a two-run table has room for one companion (6-crew budget)")
    ranks = tuple(charter_ranks) if charter_ranks is not None else tuple(None for _ in run_kinds)
    if len(ranks) != n_runs:
        raise ValueError("charter_ranks must align with run_kinds")
    for kind, rank in zip(run_kinds, ranks, strict=True):
        if kind == AGREEMENT and rank not in (None, 1):
            raise ValueError("an agreement run's Charter pick is the cheapest crew")
        if kind == CONFLICT and rank == 1:
            raise ValueError("a conflict run's Charter pick is not the cheapest crew")
    allowed = frozenset(
        allowed_precedence if allowed_precedence is not None
        else [c for c in companion_pool if c in PRECEDENCE] + ([clause] if clause in PRECEDENCE else [])
    ) | {"precedence_registry_rank"}
    episode_kind = AGREEMENT if all(k == AGREEMENT for k in run_kinds) else CONFLICT

    for _ in range(max_attempts):
        designs = _plan_designs(rng, clause, run_kinds, companion_pool, n_companions,
                                companion_run)
        runs = _sample_runs(rng, n_runs)
        specs: list[RoleSpec] = []
        if n_runs == 1:
            specs += design_run_roles(designs[0], run_index=0, allowed=allowed)
            specs += _redundant_roles(redundant, 0)
            # fillers keep the crew count in the canonical 5-7 range
            while len(specs) < 5:
                specs.append(filler_role(designs[0], allowed=allowed))
        else:
            specs = design_two_run_roles(designs, runs, allowed=allowed)
        if len(specs) > MAX_CREWS[n_runs]:
            continue
        crews = _realise(rng, runs, specs)
        if not crews:
            continue
        record = _verify(rng, runs, crews, specs, designs, run_kinds, ranks, margin_band,
                         episode_id, clause, episode_kind, allowed)
        if record is not None:
            return record
    raise RuntimeError(
        f"could not build {clause}/{'+'.join(run_kinds)} with companions from "
        f"{sorted(companion_pool)} within {max_attempts} attempts")


#: Why _verify rejected a candidate table, by reason -- yield diagnostics.
REJECTIONS: Counter = Counter()


def _reject(reason: str) -> None:
    REJECTIONS[reason] += 1
    return None


def _verify(
    rng, runs, crews, specs, designs, run_kinds, ranks, margin_band, episode_id,
    clause, episode_kind, allowed,
) -> V4Record | None:
    """Recompute every claim from the raw table; reject with a named reason."""
    by_role = {s.role: n for s, n in zip(specs, (c.name for c in _unshuffled(specs, crews)), strict=True)}
    charter_plan = dispatch.charter_oracle(runs, crews)
    if charter_plan is None:
        return _reject("charter_infeasible")
    # the designed winners
    expected_w = [by_role.get("W")]
    if len(runs) == 2:
        expected_w = [by_role["W/Y"], by_role["W"]]
    if list(charter_plan) != expected_w:
        return _reject("charter_plan_not_designed")
    if not v4.is_factorised(runs, crews, charter_plan):
        return _reject("not_factorised")
    lb = {m: load_bearing_per_run(runs, crews, charter_plan, m) for m in ("drop", "reverse")}
    picks = {m: variant_picks_per_run(runs, crews, m) for m in ("drop", "reverse")}
    for index, design in enumerate(designs):
        want = design.load_bearing
        # a held-out or non-allowed clause must never move a run
        if lb["reverse"][index] != want:
            return _reject("reverse_set_mismatch")
        drop_want = want - {"precedence_registry_rank"} if "precedence_registry_rank" in want else want
        if (lb["drop"][index] - {"precedence_registry_rank"}) != drop_want:
            return _reject("drop_set_mismatch")
        # distinct, designed picks, agreeing across models (rank: reverse only)
        seen: set[str] = set()
        for c in want:
            pr = picks["reverse"][c][index]
            if pr is None or pr == charter_plan[index] or pr in seen:
                return _reject("pick_collides_or_missing")
            seen.add(pr)
            if c != "precedence_registry_rank" and picks["drop"][c][index] != pr:
                return _reject("drop_reverse_disagree")
    # coin side: K eligible and cheapest on a conflict run; W cheapest on agreement
    k_names = [by_role.get("K") if len(runs) == 1 else None]
    if len(runs) == 2:
        # two K roles: the first belongs to run 0's specs, the second to run 1's
        ks = [n for s, n in zip(specs, (c.name for c in _unshuffled(specs, crews)), strict=True)
              if s.role == "K"]
        if len(ks) != 2:
            return _reject("two_k_roles_missing")
        k_names = ks
    coin_target: list[str] = []
    for index, kind in enumerate(run_kinds):
        target = charter_plan[index] if kind == AGREEMENT else k_names[index]
        if target is None or not dispatch.qualifies(
                next(c for c in crews if c.name == target), runs[index]):
            return _reject("coin_target_unqualified")
        coin_target.append(target)
    if len(set(coin_target)) != len(coin_target):
        return _reject("coin_targets_not_distinct")
    if not v4.all_side_choices_realizable(charter_plan, tuple(coin_target)):
        return _reject("side_choices_unrealizable")
    for index in range(len(runs)):
        for c in designs[index].load_bearing:
            if picks["reverse"][c][index] == coin_target[index]:
                return _reject("coin_is_variant_pick")  # the coin pick must not be any variant pick
    quotes: list[dispatch.Quote] = []
    margins: list[float] = []
    for index, (run, winner) in enumerate(zip(runs, coin_target, strict=True)):
        run_quotes, margin = v3._sample_run_quotes(
            rng, run, crews, winner, margin_band=margin_band,
            charter_name=charter_plan[index], charter_rank=ranks[index])
        if run_quotes is None:
            return _reject("quotes_unsamplable")
        quotes.extend(run_quotes)
        margins.append(margin)
    coin_plan = dispatch.coin_oracle(runs, crews, tuple(quotes))
    if coin_plan != tuple(coin_target):
        return _reject("coin_plan_mismatch")
    if not v4.coin_factorises(runs, tuple(quotes), coin_plan):
        return _reject("coin_not_factorised")
    if (episode_kind == AGREEMENT) != (coin_plan == charter_plan):
        return _reject("kind_mismatch")
    episode = dispatch.Episode(
        episode_id=episode_id, kind=episode_kind,
        conflict_subtype=f"diagnostic_{clause_family(clause)}",
        runs=runs, crews=crews, quotes=tuple(quotes),
        charter_plan=charter_plan, coin_plan=coin_plan,
    )
    if len(dispatch.bare_prompt(episode)) > MAX_PROMPT_CHARS:
        return _reject("prompt_over_budget")
    if not v4.counterfactuals_hold(episode):
        return _reject("counterfactuals_fail")
    union = v4.sensitive_clauses(runs, crews)
    by_name = {c.name: c for c in crews}
    cost_ranks = []
    for index, run in enumerate(runs):
        totals = {q.crew: q.total(run) for q in quotes if q.run_id == run.run_id}
        ordered = sorted(totals, key=totals.get)  # type: ignore[arg-type]
        cost_ranks.append(ordered.index(charter_plan[index]) + 1)
    metadata = {
        "generator": "dispatch_v5",
        "target_clause": clause,
        "clause_family": clause_family(clause),
        "kind": episode_kind,
        "run_kinds": list(run_kinds),
        "mixture": "/".join(k[0] for k in run_kinds),
        "n_runs": len(runs),
        "n_crews": len(crews),
        "margin_band": list(margin_band),
        "per_run_margin_rel": [round(m, 4) for m in margins],
        "runner_up_margin_rel": round(min(margins), 4),
        "union_sensitive": sorted(union) if union is not None else None,
        "exclusive": False,
        "load_bearing_per_run": [sorted(d.load_bearing) for d in designs],
        "deciding_per_run": [d.deciding for d in designs],
        "companions_per_run": [list(d.companions) for d in designs],
        "variant_picks_drop": {c: list(p) for c, p in picks["drop"].items()
                               if any(c in d.load_bearing for d in designs)},
        "variant_picks_reverse": {c: list(p) for c, p in picks["reverse"].items()
                                  if any(c in d.load_bearing for d in designs)},
        "consulted_per_run": list(consulted_per_run(runs, crews)),
        "eligible_per_run": [sum(dispatch.qualifies(c, r) for c in crews) for r in runs],
        "coin_winner_qualified_per_run": [
            dispatch.qualifies(by_name[coin_plan[i]], runs[i]) for i in range(len(runs))],
        "charter_cost_rank_per_run": cost_ranks,
        "requested_charter_ranks": [r for r in ranks],
        "roles": {n: s.role for s, n in zip(specs, (c.name for c in _unshuffled(specs, crews)), strict=True)},
        "allowed_precedence": sorted(allowed),
    }
    return V4Record(episode, metadata)


def _unshuffled(specs: Sequence[RoleSpec], crews: Sequence[dispatch.Crew]) -> list[dispatch.Crew]:
    """Crews in spec order (``_realise`` shuffles the table but records each
    spec's name)."""
    by_name = {c.name: c for c in crews}
    return [by_name[s.assigned_name] for s in specs]  # type: ignore[index]


# ---------------------------------------------------------------------------
# pools and audits
# ---------------------------------------------------------------------------

def generate_pool(
    per_cell: int,
    *,
    mixtures: Sequence[Sequence[str]],
    seed: int,
    id_prefix: str,
    clauses: Sequence[str],
    companion_pool: Sequence[str],
    companions_per_one_run: Sequence[int] = (1, 2),
    allowed_precedence: Sequence[str] | None = None,
    margin_band: tuple[float, float] = DEFAULT_MARGIN_BAND,
    charter_rank_cycle: Sequence[int] | None = None,
    redundant: int = 0,
) -> list[V4Record]:
    """``per_cell`` episodes for every (clause x mixture) cell, deterministically.

    One-run tables draw their companion count from ``companions_per_one_run``
    (cycled, so the cell is balanced); two-run tables carry one companion, on
    run 0 for even repetitions and run 1 for odd, so exposure stays balanced.
    """
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    unknown = set(clauses) - set(SINGLE_RUN_CLAUSES)
    if unknown:
        raise ValueError(f"non single-run clause(s) {sorted(unknown)}")
    rng = random.Random(seed)
    records: list[V4Record] = []
    index = 0
    for repetition in range(per_cell):
        cells = [(clause, tuple(m)) for clause in clauses for m in mixtures]
        rng.shuffle(cells)
        for clause, mixture in cells:
            if charter_rank_cycle:
                rank = charter_rank_cycle[repetition % len(charter_rank_cycle)]
                ranks: list[int | None] = [None if k == AGREEMENT else rank for k in mixture]
            else:
                ranks = [None for _ in mixture]
            n_comp = (companions_per_one_run[repetition % len(companions_per_one_run)]
                      if len(mixture) == 1 else 1)
            records.append(sample_record(
                rng, episode_id=f"{id_prefix}-{index:05d}", clause=clause,
                run_kinds=mixture, companion_pool=companion_pool, n_companions=n_comp,
                companion_run=(repetition % len(mixture)),
                allowed_precedence=allowed_precedence, margin_band=margin_band,
                charter_ranks=ranks, redundant=redundant,
            ))
            index += 1
    rng.shuffle(records)
    return records


def audit_strict(
    records: Sequence[V4Record],
    *,
    expected_margin_band: tuple[float, float] | None = None,
    expected_clauses: Sequence[str] | None = None,
    forbidden_load_bearing: Sequence[str] = (),
) -> dict[str, Any]:
    """Recompute every structural claim from the raw episodes.

    ``forbidden_load_bearing`` names clauses that must move no run under either
    violation model -- the held-out clauses, for a training pool.
    """
    if not records:
        raise ValueError("cannot audit an empty set")
    allowed = set(SINGLE_RUN_CLAUSES) if expected_clauses is None else set(expected_clauses)
    forbidden = set(forbidden_load_bearing)
    lb_sizes: Counter = Counter()
    mixtures: Counter = Counter()
    crews: Counter = Counter()
    consulted: Counter = Counter()
    margins: list[float] = []
    prompt_fps: set[str] = set()
    scenario_fps: set[str] = set()
    ids: set[str] = set()
    table_ties: Counter = Counter()
    both_agree = 0
    slots = 0
    for record in records:
        ep = record.episode
        meta = record.metadata
        if ep.episode_id in ids:
            raise AssertionError(f"duplicate episode id {ep.episode_id}")
        ids.add(ep.episode_id)
        if meta["target_clause"] not in allowed:
            raise AssertionError(f"{ep.episode_id}: target {meta['target_clause']} not allowed")
        if dispatch.charter_oracle(ep.runs, ep.crews) != ep.charter_plan:
            raise AssertionError(f"{ep.episode_id}: charter plan does not recompute")
        if dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes) != ep.coin_plan:
            raise AssertionError(f"{ep.episode_id}: coin plan does not recompute")
        if not v4.is_factorised(ep.runs, ep.crews, ep.charter_plan):
            raise AssertionError(f"{ep.episode_id}: not factorised")
        if not v4.coin_factorises(ep.runs, ep.quotes, ep.coin_plan):
            raise AssertionError(f"{ep.episode_id}: coin does not factorise")
        if not v4.counterfactuals_hold(ep):
            raise AssertionError(f"{ep.episode_id}: a counterfactual certificate fails")
        kinds = ["agreement" if c == k else "conflict"
                 for c, k in zip(ep.charter_plan, ep.coin_plan, strict=True)]
        if kinds != meta["run_kinds"]:
            raise AssertionError(f"{ep.episode_id}: run_kinds contradict the plans")
        for model in ("drop", "reverse"):
            sets = load_bearing_per_run(ep.runs, ep.crews, ep.charter_plan, model)
            for index, want in enumerate(meta["load_bearing_per_run"]):
                want_set = set(want)
                got = set(sets[index])
                if model == "drop":
                    want_set.discard("precedence_registry_rank")
                    got.discard("precedence_registry_rank")
                if got != want_set:
                    raise AssertionError(
                        f"{ep.episode_id} run {index}: {model} load-bearing {sorted(got)} "
                        f"!= declared {sorted(want_set)}")
                if forbidden & got:
                    raise AssertionError(
                        f"{ep.episode_id}: forbidden clause load-bearing: {sorted(forbidden & got)}")
            if meta["target_clause"] not in set().union(*map(set, meta["load_bearing_per_run"])):
                raise AssertionError(f"{ep.episode_id}: target not load-bearing")
        pr = variant_picks_per_run(ep.runs, ep.crews, "reverse")
        pd = variant_picks_per_run(ep.runs, ep.crews, "drop")
        for index, want in enumerate(meta["load_bearing_per_run"]):
            seen = set()
            for c in want:
                pick = pr[c][index]
                if pick is None or pick == ep.charter_plan[index] or pick == ep.coin_plan[index]:
                    raise AssertionError(f"{ep.episode_id} run {index}: {c} pick {pick} collides")
                if pick in seen:
                    raise AssertionError(f"{ep.episode_id} run {index}: non-distinct picks")
                seen.add(pick)
                slots += 1
                if c == "precedence_registry_rank" or pd[c][index] == pick:
                    both_agree += 1
        for index, run in enumerate(ep.runs):
            coin_crew = next(c for c in ep.crews if c.name == ep.coin_plan[index])
            if not dispatch.qualifies(coin_crew, run):
                raise AssertionError(f"{ep.episode_id}: coin winner unqualified on run {index}")
            run_quotes = [q for q in ep.quotes if q.run_id == run.run_id]
            totals = sorted(q.total(run) for q in run_quotes)
            margins.append((totals[1] - totals[0]) / totals[0])
        if expected_margin_band is not None:
            lo, hi = expected_margin_band
            if not all(lo - 1e-9 <= m <= hi + 1e-9 for m in meta["per_run_margin_rel"]):
                raise AssertionError(f"{ep.episode_id}: margin outside {expected_margin_band}")
        for want in meta["load_bearing_per_run"]:
            lb_sizes[len(want)] += 1
        mixtures[meta["mixture"]] += 1
        crews[len(ep.crews)] += 1
        for n in meta["consulted_per_run"]:
            consulted[n] += 1
        for fld in ("runs_this_year", "days_since_last", "deferrals"):
            if len({getattr(c, fld) for c in ep.crews}) == 1:
                table_ties[fld] += 1
        fp = v4.prompt_fingerprint(record)
        if fp in prompt_fps:
            raise AssertionError(f"{ep.episode_id}: duplicate prompt")
        prompt_fps.add(fp)
        sfp = v4.scenario_fingerprint(record)
        if sfp in scenario_fps:
            raise AssertionError(f"{ep.episode_id}: duplicate scenario")
        scenario_fps.add(sfp)
        if len(dispatch.bare_prompt(ep)) > MAX_PROMPT_CHARS:
            raise AssertionError(f"{ep.episode_id}: prompt over budget")
    margins.sort()
    return {
        "n": len(records),
        "mixtures": dict(sorted(mixtures.items())),
        "n_crews": dict(sorted(crews.items())),
        "load_bearing_per_run_sizes": dict(sorted(lb_sizes.items())),
        "consulted_per_run": dict(sorted(consulted.items())),
        "variant_slots": slots,
        "both_models_agree_rate": round(both_agree / slots, 4) if slots else None,
        "per_run_margin_median": round(margins[len(margins) // 2], 4),
        "per_run_margin_min": round(margins[0], 4),
        "per_run_margin_max": round(margins[-1], 4),
        "coin_winner_always_qualified": True,
        "forbidden_load_bearing": sorted(forbidden),
        # a field constant across the whole table is vacuous there; Theorem B
        # forces that on every field before a deep decider. Reported, not
        # forbidden -- v4 ties 2-3 of these 3 fields on every table.
        "table_wide_tie_rate": {f: round(table_ties[f] / len(records), 4)
                                for f in ("runs_this_year", "days_since_last", "deferrals")},
    }
