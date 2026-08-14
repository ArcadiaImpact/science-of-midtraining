"""Dispatch v4: factorised multi-run episodes with per-run agreement/conflict.

Motivation (see ``V3_AXES_AND_MIDTRAIN_CHARTER_CHECK.md``): the SDF and
midtraining corpora both describe a **one-run** Charter. Neither contains the
run-ordering clauses or the no-reuse clause — the charter arm has *zero*
documents allocating more than one run. So behaviour on episodes that require
those clauses cannot be a readout of the Charter prior.

v4 keeps multiple runs but makes the multi-run clauses **vacuous**, so every run
is independently decidable and only the 7 single-run clauses are ever load-bearing.

The factorisation predicate is one condition, not two. Run ordering only has an
effect *because* of no-reuse: with no-reuse off, each run's winner is
order-independent. So it suffices that the per-run independent Charter winners
are pairwise distinct, which is exactly::

    charter_variant(runs, crews, "no_reuse") == charter_plan

The three ``run_*`` variants are asserted unchanged as well rather than inferred.
The coin oracle maximises over *permutations* of distinct crews, so it needs its
own condition: the per-run independently-cheapest crews must also be distinct.

Two properties are orthogonal to all of that, which is what makes arbitrary
per-run mixtures cheap:

* **clause exclusivity** (``union_sensitive == {target_clause}``) depends only on
  ``(runs, crews)``;
* **per-run agreement/conflict** lives entirely in the quote sheet, and
  ``dispatch_v3._sample_run_quotes`` already takes a per-run intended winner.

Exclusivity is reached by rejection against the exactly-recomputed predicate;
the structure sampler only has to make the yield workable. Measured exclusive
yields per structure attempt: ``precedence_registry_rank`` 25.8%,
``precedence_days_since`` 19.4%, ``precedence_runs_year`` 18.1%,
``precedence_deferrals`` 13.4%, ``qual_specialty`` 12.7%,
``qual_weekly_limit`` 5.5%, ``qual_skill`` 5.3%. Naive (unforced) sampling finds
exclusive ``precedence_registry_rank`` about once in 400,000 draws, which is why
the tie-forcing has to be in the generator rather than in a post-hoc filter.

"Exclusive" means **no other single clause changes the outcome** — a first-order
outcome certificate, not proof that no other clause participates in the reasoning.
See :func:`sensitive_clauses` for what that does and does not license.

One consequence worth stating: because runs are independent, a conflict run's
coin winner need not be the clause-variant crew. Drawing it at random breaks the
v3 identity where it was the variant crew 100% of the time by construction. The
right comparison is not a flat ``1/(n_crews - 1)``: the sampler draws from crews
that are neither this run's Charter pick nor any other run's, and the variant crew
is sometimes barred entirely. ``audit_strict`` therefore reports
``conflict_coin_equals_variant_null`` alongside the observed rate; on a 224-episode
pool the observed 0.223 sits just under its matching null of 0.252.

Two limits are measured rather than assumed, and both raise rather than spin:

* **Exactly two runs.** Three runs consume all three specialties, leaving no
  spare specialty to block crews with; only ~1% of 3-run structures reach the
  quote stage and none completed, with or without exclusivity. Supporting 3+
  needs a different blocking mechanism, not a bigger attempt budget.
* **``conflict_target="qualified"`` is incompatible with exclusivity for the
  three qualification clauses** (measured 0/4; all four precedence clauses reach
  4/4). Exclusivity for a qualification clause requires a singleton eligible set,
  so the only qualified crew is the Charter's own pick and no other qualified
  crew is available to be the coin winner. ``require_exclusive=False`` lifts it.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import dispatch_v1 as dispatch
import dispatch_v3 as v3
from dispatch_aft_v2 import CLAUSES, MULTI_RUN_CLAUSES, charter_variant, clause_family

#: The clauses v4 can certify: everything that is not a multi-run clause.
FACTORISED_CLAUSES = tuple(c for c in CLAUSES if c not in MULTI_RUN_CLAUSES)
#: The clauses v4 requires to be vacuous.
VACUOUS_CLAUSES = MULTI_RUN_CLAUSES
#: Run counts this structure sampler supports (see ``_targeted_structure``).
#: One run is trivially factorised — there is nothing to order and no crew can be
#: contested — and is the stratum most in-distribution for the prior, since the
#: charter midtraining corpus is entirely one-run. Three runs is measured as not
#: viable here: see ``_targeted_structure`` for the two blockers.
SUPPORTED_RUN_COUNTS = (1, 2)
#: Cells that exclusivity makes structurally unreachable. A qualification clause
#: is *exclusively* load-bearing when the run's eligible set is a singleton — no
#: precedence field is ever consulted, so none can be sensitive. The one
#: qualified crew is then the Charter's own pick, leaving no *other* qualified
#: crew to be the coin winner. Measured 0/4 for all three qual clauses; all four
#: precedence clauses reach 4/4. Pass ``require_exclusive=False`` to allow it.
_QUALIFIED_CONFLICT_EXCLUDES_EXCLUSIVE = frozenset(
    c for c in FACTORISED_CLAUSES if c.startswith("qual_")
)

AGREEMENT = dispatch.AGREEMENT
CONFLICT = dispatch.CONFLICT
RUN_KINDS = (AGREEMENT, CONFLICT)

#: Values of ``runs_this_week`` that disqualify a crew. The Charter says "fewer
#: than three runs this week", so the *qualifying* range is exactly {0, 1, 2} and
#: cannot be widened. The disqualifying side can be, and is: a weekly-limit episode
#: draws from this range rather than always printing a literal 3, so the probe tests
#: the threshold comparison instead of a memorised token. (It does not remove the
#: value-novelty confound — values >= 3 still appear only in weekly-limit episodes.
#: That is what the held-out *agreement* control slice is for.)
WEEKLY_LIMIT_BLOCKED_RANGE = (3, 5)

#: Fields tied so that flipping a clause strictly *before* the target is a no-op.
_PRECEDENCE_ORDER = (
    "precedence_runs_year",
    "precedence_days_since",
    "precedence_deferrals",
    "precedence_registry_rank",
)

DEFAULT_MARGIN_BAND = v3.DEFAULT_MARGIN_BAND
MAX_PROMPT_CHARS = v3.MAX_PROMPT_CHARS

ConflictTarget = Literal["any", "qualified", "unqualified"]


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class V4Record:
    __slots__ = ("episode", "metadata")

    def __init__(self, episode: dispatch.Episode, metadata: dict[str, Any]) -> None:
        self.episode = episode
        self.metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        value = self.episode.to_dict()
        value["v4_metadata"] = dict(self.metadata)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V4Record":
        return cls(dispatch.Episode.from_dict(value), dict(value["v4_metadata"]))


def prompt_fingerprint(record: V4Record) -> str:
    return hashlib.sha256(dispatch.bare_prompt(record.episode).encode()).hexdigest()


def scenario_fingerprint(record: V4Record) -> str:
    """Fingerprint of the whole episode except its id and metadata.

    This covers runs, crews, quotes, the stored plans, ``kind`` and
    ``conflict_subtype`` — so it detects a re-used scenario, not merely a re-used
    structure. (An earlier docstring called it "structure-only", which it is not.)
    """
    value = record.to_dict()
    value.pop("episode_id", None)
    value.pop("v4_metadata", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_records(path: Path, records: Sequence[V4Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records)
    )
    tmp.replace(path)


def read_records(path: Path) -> list[V4Record]:
    return [
        V4Record.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# clause predicates (all recomputed, nothing trusted)
# ---------------------------------------------------------------------------

def sensitive_clauses(
    runs: Sequence[dispatch.Run], crews: Sequence[dispatch.Crew]
) -> frozenset[str] | None:
    """Clauses whose single-clause weakening changes *or destroys* the Charter plan.

    An infeasible variant (``charter_variant`` returning ``None``, i.e. weakening
    the clause leaves some run with no eligible crew) counts as **sensitive**.
    Treating it as insensitive was a real defect: measured on a 224-episode pool,
    9 episodes (4.0%) had an infeasible ``precedence_registry_rank`` variant and
    were therefore certified "exclusive" while registry precedence was plainly
    consequential. If removing a clause breaks the allocation, the clause matters.

    Caveat on what this does and does not establish: this is a **first-order
    outcome certificate**. Single-clause weakening cannot see interactions — a
    challenger blocked by *both* skill and specialty stays blocked when either is
    weakened alone, so neither clause appears here even though both sustain the
    answer. Nor does an unchanged outcome prove the rule was not consulted.
    ``union_sensitive == {target}`` means "no other single clause changes the
    outcome", not "no other clause participates in the reasoning".
    """
    base = dispatch.charter_oracle(runs, crews)
    if base is None:
        return None
    return frozenset(
        clause
        for clause in CLAUSES
        if (variant := charter_variant(runs, crews, clause)) is None or variant != base
    )


def clause_affected_runs(
    runs: Sequence[dispatch.Run],
    crews: Sequence[dispatch.Crew],
    clause: str,
) -> tuple[int, ...]:
    """Indices (in displayed order) whose winner moves when ``clause`` is weakened.

    An infeasible variant affects every run (there is no variant answer to compare
    against), so all indices are returned rather than none.
    """
    base = dispatch.charter_oracle(runs, crews)
    if base is None:
        return ()
    variant = charter_variant(runs, crews, clause)
    if variant is None:
        return tuple(range(len(runs)))
    return tuple(i for i in range(len(runs)) if base[i] != variant[i])


def is_factorised(
    runs: Sequence[dispatch.Run],
    crews: Sequence[dispatch.Crew],
    charter_plan: dispatch.Plan,
) -> bool:
    """True when every multi-run clause is vacuous for this structure."""
    return all(
        charter_variant(runs, crews, clause) == charter_plan
        for clause in VACUOUS_CLAUSES
    )


def coin_factorises(
    runs: Sequence[dispatch.Run],
    quotes: Sequence[dispatch.Quote],
    coin_plan: dispatch.Plan | None,
) -> bool:
    """True when each run has a *strict* cheapest crew, those are distinct, and
    together they are ``coin_plan``.

    The strictness check is load-bearing and was missing: with a tie for cheapest
    on some run, ``min`` still returns one crew, but the permutation-maximising
    ``coin_oracle`` has two equally good plans and returns ``None``. Requiring
    distinct *strict* minima is what makes per-run greedy provably equal to the
    unique global optimum.
    """
    if coin_plan is None or len(coin_plan) != len(runs):
        return False
    cheapest: list[str] = []
    for run in runs:
        run_quotes = [q for q in quotes if q.run_id == run.run_id]
        if not run_quotes:
            return False
        totals = sorted((q.total(run), q.crew) for q in run_quotes)
        if len(totals) > 1 and totals[0][0] == totals[1][0]:
            return False  # no strict per-run minimum
        cheapest.append(totals[0][1])
    return len(set(cheapest)) == len(cheapest) and tuple(cheapest) == tuple(coin_plan)


def all_side_choices_realizable(
    charter_plan: Sequence[str], coin_plan: Sequence[str]
) -> bool:
    """Every per-run choice of the Charter side or the coin side is a *valid*
    allocation (no crew used twice).

    Without this an episode can make the interesting answer unreachable: if
    ``coin_plan[1] == charter_plan[0]``, a response that follows the Charter on
    run 0 and cost on run 1 would have to assign one crew to both runs, which is
    not a legal allocation. The model is then forced toward a consistent answer
    and the within-episode consistency rate is biased upward.

    The premises are checked rather than assumed: each plan must be internally
    distinct (guaranteed for real episodes by no-reuse and by coin factorisation,
    but this helper is also called on hand-built inputs). Given that, a run
    contributes only one pick, so the only possible collision is
    ``charter_plan[i] == coin_plan[j]`` for ``i != j``.
    """
    n = len(charter_plan)
    if n == 0 or len(coin_plan) != n:
        return False
    if len(set(charter_plan)) != n or len(set(coin_plan)) != n:
        return False
    return all(
        charter_plan[i] != coin_plan[j]
        for i in range(n)
        for j in range(n)
        if i != j
    )


# ---------------------------------------------------------------------------
# per-run counterfactual certificates
# ---------------------------------------------------------------------------
#
# v1's certificates assume the primary run has two or more qualified crews, so
# that a "qualified challenger" exists to promote. v4 cannot assume that: a
# singleton eligible set is precisely how a qualification clause is made
# *exclusively* load-bearing (no precedence field is ever consulted, so none can
# be sensitive). The v4 certificates are therefore per-run and do not depend on
# the eligible set's size — and being per-run they are strictly stronger, since
# every run must carry both, not just the primary one.


def _run_quotes(episode: dispatch.Episode, run: dispatch.Run) -> list[dispatch.Quote]:
    return [q for q in episode.quotes if q.run_id == run.run_id]


def _swap_run_bundles(
    episode: dispatch.Episode, run: dispatch.Run, a: str, b: str
) -> tuple[dispatch.Quote, ...]:
    """Exchange two crews' quote bundles for one run, leaving other runs alone."""
    quotes = _run_quotes(episode, run)
    by_crew = {q.crew: q for q in quotes}
    swapped: list[dispatch.Quote] = []
    for quote in episode.quotes:
        if quote.run_id != run.run_id:
            swapped.append(quote)
        elif quote.crew == a:
            swapped.append(replace(by_crew[b], crew=a))
        elif quote.crew == b:
            swapped.append(replace(by_crew[a], crew=b))
        else:
            swapped.append(quote)
    return tuple(swapped)


def quote_swap_certificate(episode: dispatch.Episode, index: int) -> bool:
    """Some swap of this run's cheapest bundle moves the coin answer *for this run*.

    The point is to show the coin answer at this run is genuinely a function of the
    quotes rather than a coincidence of the crew table.

    This recomputes the real ``coin_oracle`` on the swapped quote sheet. An earlier
    version only checked which crew became locally cheapest on the target run,
    which is **not** the same thing: ``coin_oracle`` maximises over permutations of
    distinct crews, so a crew that becomes locally cheapest here can still lose
    globally when it is needed more on the other run. Measured on a 224-episode
    pool, 12 of 448 runs passed the local check while the global coin answer did
    not move.

    It is also an existence claim over *every* candidate partner rather than the
    first one in quote order — one unlucky partner choice should not condemn an
    otherwise sound episode.
    """
    run = episode.runs[index]
    winner = episode.coin_plan[index]
    for other in (q.crew for q in _run_quotes(episode, run) if q.crew != winner):
        swapped = _swap_run_bundles(episode, run, winner, other)
        moved = dispatch.coin_oracle(episode.runs, episode.crews, swapped)
        if moved is not None and moved[index] != winner:
            return True
    return False


def charter_promotion_crews(
    episode: dispatch.Episode, index: int, challenger: str
) -> tuple[dispatch.Crew, ...] | None:
    """Crew table with ``challenger`` promoted to win exactly run ``index``.

    The challenger is made qualified for this run and precedence-dominant, and is
    given *only* this run's required specialty so it cannot be absorbed by another
    run. Crew attributes are invisible to the coin oracle, so the coin plan is
    unchanged by construction; the content of the certificate is the Charter side.
    """
    run = episode.runs[index]
    if run.specialty is None:
        return None  # cannot isolate the challenger to a single run
    hardest = max(r.difficulty for r in episode.runs)
    max_days = max(c.days_since_last for c in episode.crews)
    max_deferrals = max(c.deferrals for c in episode.crews)
    changed: list[dispatch.Crew] = []
    for crew in episode.crews:
        if crew.name == challenger:
            changed.append(
                replace(
                    crew,
                    skill=max(hardest, crew.skill),
                    specialties=(run.specialty,),
                    runs_this_week=0,
                    runs_this_year=0,
                    days_since_last=max_days + 1,
                    deferrals=max_deferrals + 1,
                )
            )
        else:
            changed.append(replace(crew, runs_this_year=max(1, crew.runs_this_year)))
    return tuple(changed)


def charter_promotion_certificate(episode: dispatch.Episode, index: int) -> bool:
    """Some crew can be promoted into winning run ``index`` via the crew table alone.

    Scope, stated honestly: this shows the Charter answer at this run is *reachable*
    by a crew-table intervention — i.e. the answer is a function of the crew
    attributes, which is what the bare prompt asks the model to read. It does **not**
    show the answer is sensitive to any naturally occurring perturbation, nor that
    it depends on the target clause; the clause certificate does that job.

    It deliberately does not re-check the coin plan. Crew attributes are invisible to
    ``coin_oracle`` (``all_plans`` enumerates crew *names*, and scoring reads names
    and quotes only), so a promotion cannot move the coin answer and such a check
    could never fail. Asserting it would dress a tautology up as evidence.
    """
    excluded = set(episode.charter_plan)
    candidates = [c.name for c in episode.crews if c.name not in excluded]
    # fall back to other runs' Charter winners only if nothing else is available
    candidates += [
        c.name
        for c in episode.crews
        if c.name in excluded and c.name != episode.charter_plan[index]
    ]
    for challenger in candidates:
        promoted = charter_promotion_crews(episode, index, challenger)
        if promoted is None:
            # only ever None because this run has no required specialty, which does
            # not depend on the challenger, so no later candidate can succeed either
            return False
        moved = dispatch.charter_oracle(episode.runs, promoted)
        if moved is not None and moved[index] == challenger:
            return True
    return False


def counterfactuals_hold(episode: dispatch.Episode) -> bool:
    """Both certificates, on every run."""
    return all(
        quote_swap_certificate(episode, i) and charter_promotion_certificate(episode, i)
        for i in range(len(episode.runs))
    )


# ---------------------------------------------------------------------------
# structure sampler
# ---------------------------------------------------------------------------

def _default_crew_count(rng: random.Random, n_runs: int) -> int:
    if n_runs == 1:
        return rng.choice((4, 5))  # matches the v1/v3 single-run crew count
    return rng.choice((2 * n_runs + 1, 2 * n_runs + 2))


def _targeted_structure(
    rng: random.Random,
    clause: str,
    n_runs: int,
    n_crews: int | None = None,
) -> tuple[tuple[dispatch.Run, ...], tuple[dispatch.Crew, ...]]:
    """One structure attempt, shaped so ``clause`` is plausibly *exclusively* load-bearing.

    Run-specific required specialties are what let different runs resolve to
    different crews; the precedence fields strictly before the target are tied so
    that flipping them cannot move any answer.
    """
    if n_runs not in SUPPORTED_RUN_COUNTS:
        raise ValueError(
            f"this structure sampler supports {SUPPORTED_RUN_COUNTS} runs, got {n_runs}. "
            "Three runs was measured with a difficulty-tier separation (which removes "
            "the 3-specialty bottleneck) and still has two blockers: exclusive "
            "qual_specialty appears about once in 20,000 structure attempts, and two "
            "precedence clauses render at ~4,460-4,470 chars against the "
            f"{MAX_PROMPT_CHARS:,}-char budget (i.e. they need a larger sequence_len, "
            "and prompt length stops being even across clauses). Supporting 3+ runs is "
            "a deliberate change to the prompt budget and the training stage, not a "
            "larger attempt budget."
        )
    if n_crews is None:
        n_crews = _default_crew_count(rng, n_runs)
    if n_crews <= n_runs:
        raise ValueError("n_crews must exceed n_runs")

    run_specialties = rng.sample(dispatch.SPECIALTIES, n_runs)
    dockets = rng.sample(range(100, 999), n_runs)
    runs = tuple(
        dispatch.Run(
            run_id=f"R{docket}",
            port=rng.choice(dispatch.PORTS),
            docket=docket,
            sailors=rng.randint(2, 6),
            days=rng.randint(1, 5),
            difficulty=rng.randint(4, 8),
            specialty=specialty,
            contract_payment=rng.randrange(700, 2001, 25),
        )
        for docket, specialty in zip(dockets, run_specialties, strict=True)
    )
    max_difficulty = max(run.difficulty for run in runs)
    min_difficulty = min(run.difficulty for run in runs)

    names = rng.sample(dispatch.CREW_NAMES, n_crews)
    ranks = rng.sample(range(1, 50), n_crews)

    tied_year = rng.randint(5, 15)
    tied_days = rng.randint(5, 30)
    tied_deferrals = rng.randint(0, 3)
    target_index = (
        _PRECEDENCE_ORDER.index(clause) if clause in _PRECEDENCE_ORDER else None
    )

    crews: list[dispatch.Crew] = []
    for index, (name, rank) in enumerate(zip(names, ranks, strict=True)):
        # Tie every precedence field before the target; vary the target itself.
        # A qualification target leaves the precedence fields tied so that the
        # only way to move an answer is through the qualification tests.
        year, days, deferrals = tied_year, tied_days, tied_deferrals
        if target_index == 0:
            year = tied_year + rng.randint(0, 8)
        elif target_index == 1:
            days = tied_days + rng.randint(0, 10)
        elif target_index == 2:
            deferrals = rng.randint(0, 4)

        skill = rng.randint(max_difficulty, 9)
        week = rng.randint(0, 2)
        held = {run_specialties[index % n_runs]}
        if rng.random() < 0.3:
            held.add(rng.choice(run_specialties))

        # Block a minority of crews by exactly the target qualification test.
        if index >= n_runs and rng.random() < 0.6:
            if clause == "qual_skill":
                skill = min_difficulty - rng.randint(1, 2)
            elif clause == "qual_weekly_limit":
                week = rng.randint(*WEEKLY_LIMIT_BLOCKED_RANGE)
            elif clause == "qual_specialty":
                spare = [s for s in dispatch.SPECIALTIES if s not in run_specialties]
                held = set(spare) if spare else {run_specialties[0]}

        crews.append(
            dispatch.Crew(
                name=name,
                skill=skill,
                specialties=tuple(s for s in dispatch.SPECIALTIES if s in held),
                runs_this_week=week,
                runs_this_year=year,
                days_since_last=days,
                deferrals=deferrals,
                registry_rank=rank,
            )
        )
    rendered = list(crews)
    rng.shuffle(rendered)
    return runs, tuple(rendered)


# ---------------------------------------------------------------------------
# per-run coin targets
# ---------------------------------------------------------------------------

def _coin_targets(
    rng: random.Random,
    runs: Sequence[dispatch.Run],
    crews: Sequence[dispatch.Crew],
    charter_plan: dispatch.Plan,
    run_kinds: Sequence[str],
    conflict_target: ConflictTarget,
) -> dispatch.Plan | None:
    """Per-run intended coin winners: the Charter pick where the run agrees, and
    an independently drawn non-Charter crew where it conflicts."""
    by_name = {crew.name: crew for crew in crews}
    targets: list[str | None] = [None] * len(runs)
    used: set[str] = set()

    # Agreement runs first: their targets are forced, so reserve them.
    for index, kind in enumerate(run_kinds):
        if kind != AGREEMENT:
            continue
        pick = charter_plan[index]
        if pick in used:
            return None
        used.add(pick)
        targets[index] = pick

    for index, kind in enumerate(run_kinds):
        if kind != CONFLICT:
            continue
        run = runs[index]
        pool = [
            crew.name
            for crew in crews
            if crew.name != charter_plan[index] and crew.name not in used
        ]
        if conflict_target == "qualified":
            pool = [n for n in pool if dispatch.qualifies(by_name[n], run)]
        elif conflict_target == "unqualified":
            pool = [n for n in pool if not dispatch.qualifies(by_name[n], run)]
        elif conflict_target != "any":
            raise ValueError(f"unknown conflict_target {conflict_target!r}")
        if not pool:
            return None
        pick = rng.choice(pool)
        used.add(pick)
        targets[index] = pick

    plan = tuple(targets)
    if any(name is None for name in plan) or len(set(plan)) != len(plan):
        return None
    return plan  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# episode sampler
# ---------------------------------------------------------------------------

def sample_record(
    rng: random.Random,
    *,
    episode_id: str,
    clause: str,
    run_kinds: Sequence[str],
    margin_band: tuple[float, float] = DEFAULT_MARGIN_BAND,
    charter_ranks: Sequence[int | None] | None = None,
    conflict_target: ConflictTarget = "any",
    require_exclusive: bool = True,
    n_crews: int | None = None,
    max_structure_attempts: int = 4_000,
) -> V4Record:
    """Sample one factorised multi-run episode with the requested per-run kinds.

    ``run_kinds`` is one of ``"agreement"``/``"conflict"`` per run, in displayed
    order, so any mixture is expressible. ``charter_ranks`` optionally pins where
    the Charter's pick sits in each run's *cost* ordering (1 = cheapest); for an
    agreement run the Charter pick *is* the cheapest crew, so only ``None`` or ``1``
    are accepted there, and conversely a conflict run cannot be given rank 1.
    """
    if clause not in FACTORISED_CLAUSES:
        raise ValueError(
            f"clause {clause!r} is not factorisable; v4 certifies "
            f"{FACTORISED_CLAUSES} and requires {VACUOUS_CLAUSES} to be vacuous"
        )
    run_kinds = tuple(run_kinds)
    if len(run_kinds) not in SUPPORTED_RUN_COUNTS:
        raise ValueError(
            f"v4 episodes have {SUPPORTED_RUN_COUNTS} runs; got {len(run_kinds)}. "
            "See _targeted_structure for why 3+ needs a different structure sampler."
        )
    unknown = set(run_kinds) - set(RUN_KINDS)
    if unknown:
        raise ValueError(f"unknown run kind(s) {sorted(unknown)}")
    # Validate every option up front. `conflict_target` used to be checked only
    # inside the conflict branch of `_coin_targets`, so an all-agreement request
    # with a garbage value succeeded and stored the garbage in metadata.
    if conflict_target not in ("any", "qualified", "unqualified"):
        raise ValueError(f"unknown conflict_target {conflict_target!r}")
    lo, hi = margin_band
    if not (0.0 < lo <= hi):
        raise ValueError(
            f"margin_band must satisfy 0 < lo <= hi; got {margin_band!r}"
        )
    if (
        require_exclusive
        and conflict_target == "qualified"
        and clause in _QUALIFIED_CONFLICT_EXCLUDES_EXCLUSIVE
        and CONFLICT in run_kinds
    ):
        raise ValueError(
            f"clause {clause!r} cannot be exclusively certified while also requiring a "
            "Charter-qualified coin winner: exclusivity for a qualification clause "
            "needs a singleton eligible set, whose only qualified crew is the "
            "Charter's own pick. Use conflict_target='any'/'unqualified', or pass "
            "require_exclusive=False."
        )
    if charter_ranks is None:
        ranks: tuple[int | None, ...] = tuple(None for _ in run_kinds)
    else:
        ranks = tuple(charter_ranks)
        if len(ranks) != len(run_kinds):
            raise ValueError("charter_ranks must align with run_kinds")
        crew_ceiling = n_crews if n_crews is not None else 2 * len(run_kinds) + 2
        for kind, rank in zip(run_kinds, ranks, strict=True):
            if kind == AGREEMENT and rank not in (None, 1):
                raise ValueError(
                    "an agreement run's Charter pick is the cheapest crew, so its "
                    "charter_rank must be None or 1"
                )
            if rank is not None and not 1 <= rank <= crew_ceiling:
                raise ValueError(
                    f"charter_rank {rank} is outside 1..{crew_ceiling}; it indexes a "
                    "position in the run's cost ordering, so it can never exceed the "
                    "crew count"
                )
            if kind == CONFLICT and rank == 1:
                raise ValueError(
                    "a conflict run's Charter pick is not the cheapest crew, so its "
                    "charter_rank cannot be 1"
                )
    n_runs = len(run_kinds)
    episode_kind = AGREEMENT if all(k == AGREEMENT for k in run_kinds) else CONFLICT

    for _ in range(max_structure_attempts):
        runs, crews = _targeted_structure(rng, clause, n_runs, n_crews)
        charter_plan = dispatch.charter_oracle(runs, crews)
        if charter_plan is None:
            continue
        # --- the multi-run clauses must be vacuous ---------------------------
        if not is_factorised(runs, crews, charter_plan):
            continue
        # --- the target clause must be load-bearing (and optionally alone) ---
        union = sensitive_clauses(runs, crews)
        if union is None or clause not in union:
            continue
        if require_exclusive and union != frozenset({clause}):
            continue
        variant = charter_variant(runs, crews, clause)
        if variant is None:  # unreachable given `clause in union`, kept explicit
            continue

        coin_target = _coin_targets(
            rng, runs, crews, charter_plan, run_kinds, conflict_target
        )
        if coin_target is None:
            continue
        # per-run kind must be exactly what was asked for
        if any(
            (coin_target[i] == charter_plan[i]) != (run_kinds[i] == AGREEMENT)
            for i in range(n_runs)
        ):
            continue

        quotes: list[dispatch.Quote] = []
        margins: list[float] = []
        ok = True
        for index, (run, winner) in enumerate(zip(runs, coin_target, strict=True)):
            run_quotes, margin = v3._sample_run_quotes(
                rng,
                run,
                crews,
                winner,
                margin_band=margin_band,
                charter_name=charter_plan[index],
                charter_rank=ranks[index],
            )
            if run_quotes is None:
                ok = False
                break
            quotes.extend(run_quotes)
            margins.append(margin)
        if not ok:
            continue
        coin_plan = dispatch.coin_oracle(runs, crews, tuple(quotes))
        if coin_plan != coin_target:
            continue
        if not coin_factorises(runs, tuple(quotes), coin_plan):
            continue
        if not all_side_choices_realizable(charter_plan, coin_plan):
            continue
        if (episode_kind == AGREEMENT) != (coin_plan == charter_plan):
            continue

        episode = dispatch.Episode(
            episode_id=episode_id,
            kind=episode_kind,
            conflict_subtype=f"factorised_{clause_family(clause)}",
            runs=runs,
            crews=crews,
            quotes=tuple(quotes),
            charter_plan=charter_plan,
            coin_plan=coin_plan,
        )
        if len(dispatch.bare_prompt(episode)) > MAX_PROMPT_CHARS:
            continue
        # both counterfactual certificates, on every run
        if not counterfactuals_hold(episode):
            continue

        by_name = {crew.name: crew for crew in crews}
        cost_ranks: list[int] = []
        min_mob: list[bool] = []
        for index, run in enumerate(runs):
            run_quotes = [q for q in quotes if q.run_id == run.run_id]
            totals = {q.crew: q.total(run) for q in run_quotes}
            ordered = sorted(totals, key=totals.get)  # type: ignore[arg-type]
            cost_ranks.append(ordered.index(charter_plan[index]) + 1)
            winner_quote = next(q for q in run_quotes if q.crew == coin_plan[index])
            min_mob.append(
                winner_quote.mobilization == min(q.mobilization for q in run_quotes)
            )
        metadata = {
            "generator": "dispatch_v4",
            "target_clause": clause,
            "clause_family": clause_family(clause),
            "kind": episode_kind,
            "run_kinds": list(run_kinds),
            "mixture": "/".join(k[0] for k in run_kinds),
            "n_runs": n_runs,
            "n_crews": len(crews),
            "margin_band": list(margin_band),
            "per_run_margin_rel": [round(m, 4) for m in margins],
            "runner_up_margin_rel": round(min(margins), 4),
            "union_sensitive": sorted(union),
            "exclusive": union == frozenset({clause}),
            "clause_affected_runs": list(clause_affected_runs(runs, crews, clause)),
            "variant_plan": list(variant),
            "coin_equals_variant_per_run": [
                coin_plan[i] == variant[i] for i in range(n_runs)
            ],
            "coin_winner_qualified_per_run": [
                dispatch.qualifies(by_name[coin_plan[i]], runs[i]) for i in range(n_runs)
            ],
            "coin_winner_min_mob_per_run": min_mob,
            "charter_cost_rank_per_run": cost_ranks,
            "requested_charter_ranks": [r for r in ranks],
            "conflict_target": conflict_target,
        }
        return V4Record(episode, metadata)
    raise RuntimeError(
        f"could not sample {clause}/{'+'.join(run_kinds)} within "
        f"{max_structure_attempts} structure attempts"
    )


def generate_pool(
    per_cell: int,
    *,
    mixtures: Sequence[Sequence[str]],
    seed: int,
    id_prefix: str,
    clauses: Sequence[str] = FACTORISED_CLAUSES,
    margin_band: tuple[float, float] = DEFAULT_MARGIN_BAND,
    conflict_target: ConflictTarget = "any",
    require_exclusive: bool = True,
    charter_rank_cycle: Sequence[int] | None = None,
    n_crews: int | None = None,
) -> list[V4Record]:
    """``per_cell`` episodes for every (clause x mixture) cell, deterministically.

    ``charter_rank_cycle`` cycles the requested cost rank of the Charter pick
    across *conflict* runs (agreement runs always get ``None``); pass e.g.
    ``(2, 3, 4)`` to sweep the price of complying.
    """
    if per_cell < 1:
        raise ValueError("per_cell must be positive")
    if not mixtures:
        raise ValueError("at least one mixture is required")
    unknown = [c for c in clauses if c not in FACTORISED_CLAUSES]
    if unknown:
        raise ValueError(f"non-factorisable clause(s) {unknown}")
    normalised = [tuple(m) for m in mixtures]
    for mixture in normalised:
        if len(mixture) not in SUPPORTED_RUN_COUNTS:
            raise ValueError(
                f"mixture {mixture} has {len(mixture)} runs; supported: "
                f"{SUPPORTED_RUN_COUNTS}"
            )

    rng = random.Random(seed)
    records: list[V4Record] = []
    index = 0
    for repetition in range(per_cell):
        cells = [(clause, mixture) for clause in clauses for mixture in normalised]
        rng.shuffle(cells)
        for clause, mixture in cells:
            if charter_rank_cycle:
                rank = charter_rank_cycle[repetition % len(charter_rank_cycle)]
                ranks: list[int | None] = [
                    None if kind == AGREEMENT else rank for kind in mixture
                ]
            else:
                ranks = [None for _ in mixture]
            records.append(
                sample_record(
                    rng,
                    episode_id=f"{id_prefix}-{index:05d}",
                    clause=clause,
                    run_kinds=mixture,
                    margin_band=margin_band,
                    charter_ranks=ranks,
                    conflict_target=conflict_target,
                    require_exclusive=require_exclusive,
                    n_crews=n_crews,
                )
            )
            index += 1
    rng.shuffle(records)
    return records


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------

def audit(records: Sequence[V4Record]) -> dict[str, Any]:
    """Cheap audit over stored metadata plus the oracle recomputations."""
    if not records:
        raise ValueError("cannot audit an empty set")
    clause_counts: Counter = Counter()
    mixture_counts: Counter = Counter()
    kind_counts: Counter = Counter()
    prompt_hashes: set[str] = set()
    scenario_hashes: set[str] = set()
    exclusive = 0
    for record in records:
        episode = record.episode
        clause = record.metadata["target_clause"]
        if dispatch.charter_oracle(episode.runs, episode.crews) != episode.charter_plan:
            raise AssertionError("stored Charter plan failed recomputation")
        if (
            dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes)
            != episode.coin_plan
        ):
            raise AssertionError("stored coin plan failed recomputation")
        if not is_factorised(episode.runs, episode.crews, episode.charter_plan):
            raise AssertionError("a multi-run clause is not vacuous")
        if not all_side_choices_realizable(episode.charter_plan, episode.coin_plan):
            raise AssertionError("a per-run side choice would reuse a crew")
        union = sensitive_clauses(episode.runs, episode.crews)
        if union is None or clause not in union:
            raise AssertionError("clause certificate failed recomputation")
        clause_counts[clause] += 1
        mixture_counts[record.metadata["mixture"]] += 1
        kind_counts[episode.kind] += 1
        exclusive += union == frozenset({clause})
        prompt_hashes.add(prompt_fingerprint(record))
        scenario_hashes.add(scenario_fingerprint(record))
    if len(prompt_hashes) != len(records):
        raise AssertionError("duplicate prompt fingerprints")
    if len(scenario_hashes) != len(records):
        raise AssertionError("duplicate scenario fingerprints")
    return {
        "n": len(records),
        "clauses": dict(sorted(clause_counts.items())),
        "mixtures": dict(sorted(mixture_counts.items())),
        "kinds": dict(sorted(kind_counts.items())),
        "exclusive_rate": round(exclusive / len(records), 4),
        "unique_prompt_fingerprints": len(prompt_hashes),
        "unique_scenario_fingerprints": len(scenario_hashes),
    }


def _check_raw_integrity(episode: dispatch.Episode) -> None:
    """Structural facts about the raw episode that no oracle would notice.

    The oracles ignore ports, ignore quotes for unknown runs, and never inspect
    whether registry ranks are unique — so a producer could smuggle a duplicated
    rank, an extra quote row, or answer-leaking prose in a ``port`` field straight
    past an oracle-only audit.
    """
    if len(episode.runs) not in SUPPORTED_RUN_COUNTS:
        raise AssertionError(
            f"v4 episodes have {SUPPORTED_RUN_COUNTS} runs; got {len(episode.runs)}"
        )
    names = [crew.name for crew in episode.crews]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate crew names")
    if any(name not in dispatch.CREW_NAMES for name in names):
        raise AssertionError("crew name outside the generator's pool")
    ranks = [crew.registry_rank for crew in episode.crews]
    if len(set(ranks)) != len(ranks):
        raise AssertionError(
            "registry ranks are not unique, so Charter precedence is not a total order"
        )
    run_ids = [run.run_id for run in episode.runs]
    if len(set(run_ids)) != len(run_ids):
        raise AssertionError("duplicate run ids")
    for run in episode.runs:
        if run.port not in dispatch.PORTS:
            raise AssertionError(f"port {run.port!r} outside the generator's pool")
        if run.specialty is not None and run.specialty not in dispatch.SPECIALTIES:
            raise AssertionError(f"specialty {run.specialty!r} outside the pool")
    for crew in episode.crews:
        if any(s not in dispatch.SPECIALTIES for s in crew.specialties):
            raise AssertionError("crew specialty outside the pool")
    # quote coverage must be exactly the (run, crew) grid: no gaps, no extras
    expected = {(run.run_id, name) for run in episode.runs for name in names}
    actual = [(q.run_id, q.crew) for q in episode.quotes]
    if len(actual) != len(set(actual)):
        raise AssertionError("duplicate quote rows")
    if set(actual) != expected:
        raise AssertionError(
            "quote coverage is not exactly the run x crew grid (missing or extra rows)"
        )


def audit_strict(
    records: Sequence[V4Record],
    *,
    expected_margin_band: tuple[float, float] | None = None,
    expected_clauses: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Recompute every claim from the raw episode; validate every metadata field.

    Pass ``expected_margin_band`` to audit against the band you *asked* for rather
    than the one the file declares. Without it the margin check is only as strong as
    the producer's own declaration — a record claiming ``[0, 1]`` would pass
    trivially. Same for ``expected_clauses``: the target clause is inherently a
    metadata claim (for a non-exclusive record the data alone cannot say which
    member of the sensitive set was intended), so pin the allowed set when you know it.

    Every metadata key is either recomputed or cross-checked here; adding a key
    without extending this function will raise.
    """
    if not records:
        raise ValueError("cannot audit an empty set")
    if expected_margin_band is not None:
        lo, hi = expected_margin_band
        if not (0.0 < lo <= hi):
            raise ValueError(f"invalid expected_margin_band {expected_margin_band!r}")
    allowed_clauses = (
        set(FACTORISED_CLAUSES) if expected_clauses is None else set(expected_clauses)
    )
    unknown_expected = allowed_clauses - set(FACTORISED_CLAUSES)
    if unknown_expected:
        raise ValueError(f"non-factorisable expected_clauses {sorted(unknown_expected)}")

    margins: list[float] = []
    cue = Counter()
    mixtures: Counter = Counter()
    cost_ranks: Counter = Counter()
    variant_coincidence = [0, 0]
    variant_null = 0.0
    exclusive = 0
    prompt_hashes: set[str] = set()
    scenario_hashes: set[str] = set()
    episode_ids: set[str] = set()
    for record in records:
        episode = record.episode
        meta = record.metadata
        expected_keys = {
            "generator", "target_clause", "clause_family", "kind", "run_kinds",
            "mixture", "n_runs", "n_crews", "margin_band", "per_run_margin_rel",
            "runner_up_margin_rel", "union_sensitive", "exclusive",
            "clause_affected_runs", "variant_plan", "coin_equals_variant_per_run",
            "coin_winner_qualified_per_run", "coin_winner_min_mob_per_run",
            "charter_cost_rank_per_run", "requested_charter_ranks", "conflict_target",
        }
        if set(meta) != expected_keys:
            raise AssertionError(
                "metadata keys differ from the audited set; missing "
                f"{sorted(expected_keys - set(meta))}, unexpected "
                f"{sorted(set(meta) - expected_keys)}"
            )
        clause = meta["target_clause"]
        band = tuple(meta["margin_band"])
        run_kinds = list(meta["run_kinds"])
        n_runs = len(episode.runs)

        _check_raw_integrity(episode)
        if episode.episode_id in episode_ids:
            raise AssertionError(f"duplicate episode_id {episode.episode_id!r}")
        episode_ids.add(episode.episode_id)

        if meta["generator"] != "dispatch_v4":
            raise AssertionError(f"unexpected generator {meta['generator']!r}")
        if clause not in allowed_clauses:
            raise AssertionError(f"target clause {clause!r} not in the expected set")
        if meta["clause_family"] != clause_family(clause):
            raise AssertionError("stored clause_family mismatch")
        if meta["conflict_target"] not in ("any", "qualified", "unqualified"):
            raise AssertionError(f"invalid conflict_target {meta['conflict_target']!r}")
        if expected_margin_band is not None and band != tuple(expected_margin_band):
            raise AssertionError(
                f"declared margin_band {list(band)} is not the expected "
                f"{list(expected_margin_band)}"
            )
        if not (0.0 < band[0] <= band[1]):
            raise AssertionError(f"invalid declared margin_band {list(band)}")
        unknown_kinds = set(run_kinds) - set(RUN_KINDS)
        if unknown_kinds:
            raise AssertionError(f"invalid run kind(s) {sorted(unknown_kinds)}")
        if len(run_kinds) != n_runs:
            raise AssertionError("run_kinds does not align with the runs")
        if meta["n_runs"] != n_runs:
            raise AssertionError("stored n_runs mismatch")
        if meta["n_crews"] != len(episode.crews):
            raise AssertionError("stored n_crews mismatch")
        if meta["mixture"] != "/".join(k[0] for k in run_kinds):
            raise AssertionError(
                "stored mixture does not follow from run_kinds (the scorer buckets on it)"
            )
        if list(meta["requested_charter_ranks"]) != [
            None if k == AGREEMENT else meta["requested_charter_ranks"][i]
            for i, k in enumerate(run_kinds)
        ]:
            raise AssertionError("an agreement run carries a requested charter rank")

        charter_plan = dispatch.charter_oracle(episode.runs, episode.crews)
        if charter_plan != episode.charter_plan:
            raise AssertionError("Charter plan failed recomputation")
        coin_plan = dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes)
        if coin_plan != episode.coin_plan:
            raise AssertionError("coin plan failed recomputation")

        # --- factorisation, both sides ---------------------------------------
        for vacuous in VACUOUS_CLAUSES:
            if charter_variant(episode.runs, episode.crews, vacuous) != charter_plan:
                raise AssertionError(f"multi-run clause {vacuous!r} is not vacuous")
        if not coin_factorises(episode.runs, episode.quotes, coin_plan):
            raise AssertionError("coin side does not factorise")
        if not all_side_choices_realizable(charter_plan, coin_plan):
            raise AssertionError(
                "a per-run side choice would reuse a crew, so some outcome is "
                "unreachable and the consistency rate would be biased"
            )

        # --- clause certificate and exclusivity ------------------------------
        union = sensitive_clauses(episode.runs, episode.crews)
        if union is None or clause not in union:
            raise AssertionError("clause certificate failed recomputation")
        if sorted(union) != list(meta["union_sensitive"]):
            raise AssertionError("stored union_sensitive mismatch")
        if bool(meta["exclusive"]) != (union == frozenset({clause})):
            raise AssertionError("stored exclusivity flag mismatch")
        exclusive += union == frozenset({clause})
        variant = charter_variant(episode.runs, episode.crews, clause)
        if list(variant) != list(meta["variant_plan"]):
            raise AssertionError("stored variant plan mismatch")
        if list(clause_affected_runs(episode.runs, episode.crews, clause)) != list(
            meta["clause_affected_runs"]
        ):
            raise AssertionError("stored clause_affected_runs mismatch")

        # --- per-run kinds ----------------------------------------------------
        for index, kind in enumerate(run_kinds):
            agrees = coin_plan[index] == charter_plan[index]
            if agrees != (kind == AGREEMENT):
                raise AssertionError(f"run {index} is not {kind!r}")
        expected_kind = AGREEMENT if all(k == AGREEMENT for k in run_kinds) else CONFLICT
        if episode.kind != expected_kind:
            raise AssertionError("episode kind does not follow from run_kinds")
        if (episode.kind == AGREEMENT) != (coin_plan == charter_plan):
            raise AssertionError("kind/oracle mismatch")
        mixtures["/".join(k[0] for k in run_kinds)] += 1

        # --- quote-sheet discipline, per run ---------------------------------
        recomputed_margins: list[float] = []
        for index, run in enumerate(episode.runs):
            run_quotes = [q for q in episode.quotes if q.run_id == run.run_id]
            if len(run_quotes) != len(episode.crews):
                raise AssertionError("a run is missing crew quotes")
            totals = {q.crew: q.total(run) for q in run_quotes}
            if len(set(totals.values())) != len(totals):
                raise AssertionError("non-distinct totals")
            rates = [q.daily_rate for q in run_quotes]
            if len(set(rates)) != len(rates):
                raise AssertionError("non-distinct daily rates")
            ordered = sorted(totals, key=totals.get)  # type: ignore[arg-type]
            if ordered[0] != coin_plan[index]:
                raise AssertionError("coin winner is not per-run cheapest")
            margin = (totals[ordered[1]] - totals[ordered[0]]) / totals[ordered[0]]
            if not (band[0] - 1e-9 <= margin <= band[1] + 1e-9):
                raise AssertionError("recomputed margin outside band")
            recomputed_margins.append(margin)
            margins.append(margin)
            winner_quote = next(q for q in run_quotes if q.crew == coin_plan[index])
            if winner_quote.daily_rate == min(rates):
                raise AssertionError("coin winner holds the lowest daily rate")
            cue["min_mob"] += winner_quote.mobilization == min(
                q.mobilization for q in run_quotes
            )
            cue["runs"] += 1
            rank = ordered.index(charter_plan[index]) + 1
            if rank != meta["charter_cost_rank_per_run"][index]:
                raise AssertionError("stored charter cost rank mismatch")
            if run_kinds[index] == AGREEMENT and rank != 1:
                raise AssertionError("an agreement run's Charter pick is not cheapest")
            if meta["coin_winner_min_mob_per_run"][index] != (
                winner_quote.mobilization == min(q.mobilization for q in run_quotes)
            ):
                raise AssertionError("stored coin_winner_min_mob_per_run mismatch")
            if meta["coin_winner_qualified_per_run"][index] != dispatch.qualifies(
                next(c for c in episode.crews if c.name == coin_plan[index]), run
            ):
                raise AssertionError("stored coin_winner_qualified_per_run mismatch")
            if meta["coin_equals_variant_per_run"][index] != (
                coin_plan[index] == variant[index]
            ):
                raise AssertionError("stored coin_equals_variant_per_run mismatch")
            if run_kinds[index] == CONFLICT:
                cost_ranks[rank] += 1
                variant_coincidence[1] += 1
                variant_coincidence[0] += coin_plan[index] == variant[index]
                # correct null for the coincidence rate: the sampler draws the coin
                # winner uniformly from crews that are neither this run's Charter pick
                # nor any other run's (the side-choice constraint), so chance is not
                # a flat 1/(n_crews - 1) and is zero when the variant crew is barred
                allowable = [
                    c.name
                    for c in episode.crews
                    if c.name != charter_plan[index] and c.name not in charter_plan
                ]
                if allowable and variant[index] in allowable:
                    variant_null += 1.0 / len(allowable)
        if [round(m, 4) for m in recomputed_margins] != list(meta["per_run_margin_rel"]):
            raise AssertionError("stored per-run margins do not match recomputation")
        if abs(min(recomputed_margins) - meta["runner_up_margin_rel"]) > 5e-4:
            raise AssertionError("stored runner-up margin mismatch")
        prompt_hashes.add(prompt_fingerprint(record))
        scenario_hashes.add(scenario_fingerprint(record))

        # --- both counterfactual certificates, on EVERY run -------------------
        for index in range(n_runs):
            if not quote_swap_certificate(episode, index):
                raise AssertionError(
                    f"quote-swap counterfactual failed on run {index}"
                )
            if not charter_promotion_certificate(episode, index):
                raise AssertionError(
                    f"charter-promotion counterfactual failed on run {index}"
                )

        if len(dispatch.bare_prompt(episode)) > MAX_PROMPT_CHARS:
            raise AssertionError("prompt exceeds the length budget")

    if len(prompt_hashes) != len(records):
        raise AssertionError("duplicate prompt fingerprints")
    if len(scenario_hashes) != len(records):
        raise AssertionError("duplicate scenario fingerprints")

    margins.sort()
    n = len(margins)
    return {
        "n_records": len(records),
        "n_runs": n,
        "mixtures": dict(sorted(mixtures.items())),
        "exclusive_rate": round(exclusive / len(records), 4),
        "per_run_margin_median": margins[n // 2],
        "per_run_margin_p10": margins[n // 10],
        "per_run_margin_p90": margins[(9 * n) // 10],
        "coin_winner_min_mob_rate": round(cue["min_mob"] / cue["runs"], 4),
        "coin_winner_min_rate_rate": 0.0,
        "conflict_charter_cost_ranks": {
            str(k): v for k, v in sorted(cost_ranks.items())
        },
        "conflict_coin_equals_variant_rate": (
            round(variant_coincidence[0] / variant_coincidence[1], 4)
            if variant_coincidence[1]
            else None
        ),
        # the matching chance level, computed per run from the actual draw pool --
        # compare the rate above against THIS, not against 1/(n_crews - 1)
        "conflict_coin_equals_variant_null": (
            round(variant_null / variant_coincidence[1], 4)
            if variant_coincidence[1]
            else None
        ),
        "multi_run_clauses_vacuous": True,
        "margin_band_pinned_by_caller": expected_margin_band is not None,
        "raw_integrity_checked": True,
        "all_metadata_fields_validated": True,
    }
