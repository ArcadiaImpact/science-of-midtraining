"""Full-clause constructive Dispatch AFT v2 generator and scorer.

V1 intentionally used one-run signs-of-life episodes.  V2 keeps the same
neutral prompt and exact coin/Charter oracles, but generates causal coverage
for every operative Charter clause.  Each record is admitted only when a
clause-specific variant changes the Charter answer.  The variant reverses an
ordering/precedence comparison, removes a qualification requirement, or
allows crew reuse.  These certificates are audit metadata and are never
rendered into AFT or evaluation prompts.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import dispatch_v1 as dispatch

CLAUSES = (
    "run_difficulty",
    "run_duration",
    "run_docket",
    "qual_skill",
    "qual_weekly_limit",
    "qual_specialty",
    "precedence_runs_year",
    "precedence_days_since",
    "precedence_deferrals",
    "precedence_registry_rank",
    "no_reuse",
)
RUN_ORDER_CLAUSES = CLAUSES[:3]
QUALIFICATION_CLAUSES = CLAUSES[3:6]
PRECEDENCE_CLAUSES = CLAUSES[6:10]
MULTI_RUN_CLAUSES = (*RUN_ORDER_CLAUSES, "no_reuse")
SHORTCUTS = (
    "precedence_without_qualification",
    "cheapest_qualified_greedy",
    "fewest_runs_year_qualified",
    "displayed_first",
    "lowest_daily_rate_greedy",
)


@dataclass(frozen=True, slots=True)
class V2Record:
    episode: dispatch.Episode
    target_clause: str
    clause_family: str
    variant_plan: dispatch.Plan
    variant_description: str
    clause_required: bool
    charter_plan_coin_rank: int
    primary_charter_crew_display_position: int
    shortcut_matches: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        value = self.episode.to_dict()
        value["v2_metadata"] = {
            "target_clause": self.target_clause,
            "clause_family": self.clause_family,
            "variant_plan": list(self.variant_plan),
            "variant_description": self.variant_description,
            "clause_required": self.clause_required,
            "charter_plan_coin_rank": self.charter_plan_coin_rank,
            "primary_charter_crew_display_position": (
                self.primary_charter_crew_display_position
            ),
            "shortcut_matches": dict(self.shortcut_matches),
        }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V2Record":
        metadata = value["v2_metadata"]
        return cls(
            episode=dispatch.Episode.from_dict(value),
            target_clause=str(metadata["target_clause"]),
            clause_family=str(metadata["clause_family"]),
            variant_plan=tuple(metadata["variant_plan"]),
            variant_description=str(metadata["variant_description"]),
            clause_required=bool(metadata["clause_required"]),
            charter_plan_coin_rank=int(metadata["charter_plan_coin_rank"]),
            primary_charter_crew_display_position=int(
                metadata["primary_charter_crew_display_position"]
            ),
            shortcut_matches=dict(metadata["shortcut_matches"]),
        )


def clause_family(clause: str) -> str:
    if clause in RUN_ORDER_CLAUSES:
        return "run_order"
    if clause in QUALIFICATION_CLAUSES:
        return "qualification"
    if clause in PRECEDENCE_CLAUSES:
        return "precedence"
    if clause == "no_reuse":
        return "docket_constraint"
    raise ValueError(f"unknown clause {clause!r}")


def _qualifies_variant(crew: dispatch.Crew, run: dispatch.Run, clause: str) -> bool:
    skill = True if clause == "qual_skill" else crew.skill >= run.difficulty
    weekly = True if clause == "qual_weekly_limit" else crew.runs_this_week < 3
    specialty = (
        True
        if clause == "qual_specialty"
        else run.specialty is None or run.specialty in crew.specialties
    )
    return skill and weekly and specialty


def _run_key_variant(run: dispatch.Run, clause: str) -> tuple[int, int, int]:
    difficulty = run.difficulty if clause == "run_difficulty" else -run.difficulty
    duration = run.days if clause == "run_duration" else -run.days
    docket = -run.docket if clause == "run_docket" else run.docket
    return difficulty, duration, docket


def _precedence_key_variant(
    crew: dispatch.Crew, clause: str
) -> tuple[int, int, int, int]:
    runs_year = (
        -crew.runs_this_year
        if clause == "precedence_runs_year"
        else crew.runs_this_year
    )
    days = (
        crew.days_since_last
        if clause == "precedence_days_since"
        else -crew.days_since_last
    )
    deferrals = crew.deferrals if clause == "precedence_deferrals" else -crew.deferrals
    registry = (
        -crew.registry_rank
        if clause == "precedence_registry_rank"
        else crew.registry_rank
    )
    return runs_year, days, deferrals, registry


def charter_variant(
    runs: Sequence[dispatch.Run], crews: Sequence[dispatch.Crew], clause: str
) -> dispatch.Plan | None:
    """Apply the Charter with exactly one target clause weakened or reversed."""

    if clause not in CLAUSES:
        raise ValueError(f"unknown clause {clause!r}")
    remaining = {crew.name: crew for crew in crews}
    choices: dict[str, str] = {}
    ordered_runs = sorted(runs, key=lambda run: _run_key_variant(run, clause))
    for run in ordered_runs:
        pool = crews if clause == "no_reuse" else tuple(remaining.values())
        eligible = [
            crew
            for crew in pool
            if (
                _qualifies_variant(crew, run, clause)
                if clause in QUALIFICATION_CLAUSES
                else dispatch.qualifies(crew, run)
            )
        ]
        if not eligible:
            return None
        selected = min(eligible, key=lambda crew: _precedence_key_variant(crew, clause))
        choices[run.run_id] = selected.name
        if clause != "no_reuse":
            del remaining[selected.name]
    return tuple(choices[run.run_id] for run in runs)


def _crew(
    name: str,
    *,
    skill: int,
    specialties: Sequence[str],
    week: int,
    year: int,
    days: int,
    deferrals: int,
    rank: int,
) -> dispatch.Crew:
    return dispatch.Crew(
        name=name,
        skill=skill,
        specialties=tuple(
            item for item in dispatch.SPECIALTIES if item in set(specialties)
        ),
        runs_this_week=week,
        runs_this_year=year,
        days_since_last=days,
        deferrals=deferrals,
        registry_rank=rank,
    )


def _names(rng: random.Random, n: int, target_name_index: int) -> list[str]:
    target = dispatch.CREW_NAMES[target_name_index % len(dispatch.CREW_NAMES)]
    others = rng.sample([name for name in dispatch.CREW_NAMES if name != target], n - 1)
    return [target, *others]


def _position_primary_target(
    rng: random.Random,
    crews: Sequence[dispatch.Crew],
    target: str,
    position: int,
) -> tuple[dispatch.Crew, ...]:
    rendered = list(crews)
    rng.shuffle(rendered)
    selected = next(crew for crew in rendered if crew.name == target)
    rendered.remove(selected)
    rendered.insert(position % len(crews), selected)
    return tuple(rendered)


def _single_run_structure(
    rng: random.Random,
    clause: str,
    *,
    target_name_index: int,
    target_position: int,
) -> tuple[tuple[dispatch.Run, ...], tuple[dispatch.Crew, ...], str]:
    names = _names(rng, 4, target_name_index)
    target, rival, other_a, other_b = names
    difficulty = rng.randint(4, 8)
    specialty = rng.choice(dispatch.SPECIALTIES) if clause == "qual_specialty" else None
    docket = rng.randint(100, 999)
    run = dispatch.Run(
        run_id=f"R{docket}",
        port=rng.choice(dispatch.PORTS),
        docket=docket,
        sailors=rng.randint(2, 6),
        days=rng.randint(1, 5),
        difficulty=difficulty,
        specialty=specialty,
        contract_payment=rng.randrange(7_000, 9_001, 25),
    )
    required = () if specialty is None else (specialty,)

    if clause in QUALIFICATION_CLAUSES:
        target_crew = _crew(
            target,
            skill=difficulty + 1,
            specialties=required,
            week=1,
            year=8,
            days=35,
            deferrals=3,
            rank=8,
        )
        rival_kwargs = {
            "skill": difficulty + 2,
            "specialties": required,
            "week": 0,
            "year": 4,
            "days": 45,
            "deferrals": 4,
            "rank": 1,
        }
        if clause == "qual_skill":
            rival_kwargs["skill"] = difficulty - 1
        elif clause == "qual_weekly_limit":
            rival_kwargs["week"] = 3
        else:
            rival_kwargs["specialties"] = ()
        rival_crew = _crew(rival, **rival_kwargs)
        distractors = (
            _crew(
                other_a,
                skill=difficulty + 1,
                specialties=required,
                week=2,
                year=12,
                days=20,
                deferrals=2,
                rank=20,
            ),
            _crew(
                other_b,
                skill=difficulty + 2,
                specialties=required,
                week=0,
                year=15,
                days=10,
                deferrals=1,
                rank=30,
            ),
        )
    else:
        common = {"skill": difficulty + 2, "specialties": required, "week": 1}
        if clause == "precedence_runs_year":
            target_values = (5, 8, 0, 40)
            rival_values = (9, 45, 4, 1)
        elif clause == "precedence_days_since":
            target_values = (7, 40, 0, 40)
            rival_values = (7, 10, 4, 1)
        elif clause == "precedence_deferrals":
            target_values = (7, 20, 4, 40)
            rival_values = (7, 20, 1, 1)
        elif clause == "precedence_registry_rank":
            target_values = (7, 20, 2, 3)
            rival_values = (7, 20, 2, 35)
        else:
            raise ValueError(f"not a single-run target clause: {clause}")
        target_crew = _crew(
            target,
            **common,
            year=target_values[0],
            days=target_values[1],
            deferrals=target_values[2],
            rank=target_values[3],
        )
        rival_crew = _crew(
            rival,
            **common,
            year=rival_values[0],
            days=rival_values[1],
            deferrals=rival_values[2],
            rank=rival_values[3],
        )
        distractors = (
            _crew(
                other_a,
                **common,
                year=14,
                days=15,
                deferrals=1,
                rank=20,
            ),
            _crew(
                other_b,
                **common,
                year=17,
                days=8,
                deferrals=0,
                rank=30,
            ),
        )

    crews = _position_primary_target(
        rng,
        (target_crew, rival_crew, *distractors),
        target,
        target_position,
    )
    return (run,), crews, rival


def _multi_run_structure(
    rng: random.Random,
    clause: str,
    *,
    target_name_index: int,
    target_position: int,
) -> tuple[tuple[dispatch.Run, ...], tuple[dispatch.Crew, ...]]:
    names = _names(rng, 5, target_name_index)
    versatile, first_only, second_only, other_a, other_b = names
    docket_low, docket_high = sorted(rng.sample(range(100, 999), 2))
    if clause == "run_difficulty":
        first_values = (8, 2, docket_high)
        second_values = (5, 4, docket_low)
    elif clause == "run_duration":
        first_values = (6, 5, docket_high)
        second_values = (6, 2, docket_low)
    elif clause == "run_docket":
        first_values = (6, 3, docket_low)
        second_values = (6, 3, docket_high)
    elif clause == "no_reuse":
        first_values = (7, 4, docket_low)
        second_values = (5, 2, docket_high)
    else:
        raise ValueError(f"not a multi-run target clause: {clause}")

    specialty_a = "reef charts"
    specialty_b = "tide timing"
    if clause == "no_reuse":
        specialty_a = specialty_b = "reef charts"
    run_a = dispatch.Run(
        run_id=f"R{first_values[2]}",
        port=rng.choice(dispatch.PORTS),
        docket=first_values[2],
        sailors=rng.randint(2, 6),
        days=first_values[1],
        difficulty=first_values[0],
        specialty=specialty_a,
        contract_payment=rng.randrange(7_000, 9_001, 25),
    )
    run_b = dispatch.Run(
        run_id=f"R{second_values[2]}",
        port=rng.choice(dispatch.PORTS),
        docket=second_values[2],
        sailors=rng.randint(2, 6),
        days=second_values[1],
        difficulty=second_values[0],
        specialty=specialty_b,
        contract_payment=rng.randrange(7_000, 9_001, 25),
    )
    max_difficulty = max(run_a.difficulty, run_b.difficulty)
    if clause == "no_reuse":
        first_specialties = second_specialties = (specialty_a,)
    else:
        first_specialties = (specialty_a,)
        second_specialties = (specialty_b,)
    crews = (
        _crew(
            versatile,
            skill=max_difficulty + 1,
            specialties=(specialty_a, specialty_b),
            week=0,
            year=4,
            days=45,
            deferrals=4,
            rank=1,
        ),
        _crew(
            first_only,
            skill=max_difficulty + 1,
            specialties=first_specialties,
            week=1,
            year=8,
            days=30,
            deferrals=3,
            rank=8,
        ),
        _crew(
            second_only,
            skill=max_difficulty + 1,
            specialties=second_specialties,
            week=1,
            year=8,
            days=30,
            deferrals=3,
            rank=9,
        ),
        _crew(
            other_a,
            skill=max_difficulty + 1,
            specialties=(specialty_a,),
            week=2,
            year=14,
            days=15,
            deferrals=1,
            rank=20,
        ),
        _crew(
            other_b,
            skill=max_difficulty + 1,
            specialties=(specialty_b,),
            week=2,
            year=16,
            days=10,
            deferrals=0,
            rank=30,
        ),
    )
    rendered_crews = _position_primary_target(rng, crews, versatile, target_position)
    rendered_runs = [run_a, run_b]
    rng.shuffle(rendered_runs)
    return tuple(rendered_runs), rendered_crews


def _quote_from_total(
    rng: random.Random,
    run: dispatch.Run,
    crew: str,
    total: int,
    *,
    preferred: bool,
) -> dispatch.Quote:
    daily_rate = (
        rng.choice((35, 40, 45, 50))
        if preferred
        else rng.choice((5, 10, 15, 20, 25, 30))
    )
    difficulty_supplement = rng.randrange(0, 401, 5) if run.difficulty >= 7 else 0
    specialty_supplement = rng.randrange(0, 301, 5) if run.specialty else 0
    mobilization = (
        total
        - daily_rate * run.sailors * run.days
        - difficulty_supplement
        - specialty_supplement
    )
    if mobilization < 10 or mobilization % 5:
        raise AssertionError("constructed quote does not have a valid mobilization fee")
    return dispatch.Quote(
        run_id=run.run_id,
        crew=crew,
        mobilization=mobilization,
        daily_rate=daily_rate,
        difficulty_supplement=difficulty_supplement,
        specialty_supplement=specialty_supplement,
    )


def _quotes_for_plan(
    rng: random.Random,
    runs: Sequence[dispatch.Run],
    crews: Sequence[dispatch.Crew],
    target: dispatch.Plan,
    *,
    single_run_charter: str | None,
    charter_rank: int,
) -> tuple[dispatch.Quote, ...]:
    target_by_run = {run.run_id: crew for run, crew in zip(runs, target, strict=True)}
    totals: dict[tuple[str, str], int] = {}
    if len(runs) == 1 and single_run_charter is not None:
        run = runs[0]
        target_name = target[0]
        remaining = [crew.name for crew in crews if crew.name != target_name]
        rng.shuffle(remaining)
        ordered = [target_name, *remaining]
        if single_run_charter != target_name:
            ordered.remove(single_run_charter)
            insertion = max(1, min(charter_rank, len(crews))) - 1
            ordered.insert(insertion, single_run_charter)
        if len(set(ordered)) != len(crews):
            raise AssertionError("invalid single-run cost order")
        for index, name in enumerate(ordered):
            totals[(run.run_id, name)] = 3_000 + index * 650 + rng.randrange(0, 101, 5)
    else:
        for run in runs:
            preferred_name = target_by_run[run.run_id]
            for crew in crews:
                base = 3_000 if crew.name == preferred_name else 5_000
                totals[(run.run_id, crew.name)] = base + rng.randrange(0, 401, 5)

    quotes = tuple(
        _quote_from_total(
            rng,
            run,
            crew.name,
            totals[(run.run_id, crew.name)],
            preferred=crew.name == target_by_run[run.run_id],
        )
        for run in runs
        for crew in crews
    )
    if dispatch.coin_oracle(runs, crews, quotes) != target:
        raise AssertionError("constructed quote matrix has the wrong coin optimum")
    return quotes


def _valid_alternative_for_no_reuse(
    runs: Sequence[dispatch.Run], full: dispatch.Plan
) -> dispatch.Plan:
    if len(runs) != 2 or len(set(full)) != 2:
        raise AssertionError("no-reuse template expected two distinct assignments")
    return (full[1], full[0])


def _plan_coin_rank(episode: dispatch.Episode, plan: dispatch.Plan) -> int:
    scored = sorted(
        (
            (dispatch.coin_margin(candidate, episode.runs, episode.quotes), candidate)
            for candidate in dispatch.all_plans(episode.runs, episode.crews)
        ),
        reverse=True,
    )
    return next(
        index for index, (_score, item) in enumerate(scored, start=1) if item == plan
    )


def _greedy_plan(
    episode: dispatch.Episode,
    *,
    eligibility: Literal["charter", "all"],
    chooser: Literal["precedence", "quote", "runs_year", "daily", "display"],
) -> dispatch.Plan | None:
    remaining = {crew.name: crew for crew in episode.crews}
    by_quote = {(quote.run_id, quote.crew): quote for quote in episode.quotes}
    choices: dict[str, str] = {}
    for run in sorted(
        episode.runs, key=lambda item: (-item.difficulty, -item.days, item.docket)
    ):
        pool = list(remaining.values())
        if eligibility == "charter":
            pool = [crew for crew in pool if dispatch.qualifies(crew, run)]
        if not pool:
            return None

        def choice_key(crew: dispatch.Crew) -> int | tuple[int, int, int, int]:
            if chooser == "precedence":
                return (
                    crew.runs_this_year,
                    -crew.days_since_last,
                    -crew.deferrals,
                    crew.registry_rank,
                )
            if chooser == "quote":
                return by_quote[(run.run_id, crew.name)].total(run)
            if chooser == "runs_year":
                return crew.runs_this_year
            if chooser == "daily":
                return by_quote[(run.run_id, crew.name)].daily_rate
            order = {crew.name: index for index, crew in enumerate(episode.crews)}
            return order[crew.name]

        selected = min(pool, key=choice_key)
        choices[run.run_id] = selected.name
        del remaining[selected.name]
    return tuple(choices[run.run_id] for run in episode.runs)


def shortcut_matches(episode: dispatch.Episode) -> dict[str, bool]:
    shortcuts = {
        "precedence_without_qualification": _greedy_plan(
            episode, eligibility="all", chooser="precedence"
        ),
        "cheapest_qualified_greedy": _greedy_plan(
            episode, eligibility="charter", chooser="quote"
        ),
        "fewest_runs_year_qualified": _greedy_plan(
            episode, eligibility="charter", chooser="runs_year"
        ),
        "displayed_first": _greedy_plan(episode, eligibility="all", chooser="display"),
        "lowest_daily_rate_greedy": _greedy_plan(
            episode, eligibility="all", chooser="daily"
        ),
    }
    return {name: plan == episode.charter_plan for name, plan in shortcuts.items()}


def sample_record(
    rng: random.Random,
    *,
    episode_id: str,
    kind: Literal["agreement", "conflict"],
    clause: str,
    charter_rank: int,
    target_name_index: int,
    target_position: int,
) -> V2Record:
    if clause in MULTI_RUN_CLAUSES:
        runs, crews = _multi_run_structure(
            rng,
            clause,
            target_name_index=target_name_index,
            target_position=target_position,
        )
        rival_name = None
    else:
        runs, crews, rival_name = _single_run_structure(
            rng,
            clause,
            target_name_index=target_name_index,
            target_position=target_position,
        )
    charter_plan = dispatch.charter_oracle(runs, crews)
    variant_plan = charter_variant(runs, crews, clause)
    if charter_plan is None or variant_plan is None or charter_plan == variant_plan:
        raise AssertionError(f"{clause}: target clause is not causally required")
    if clause in QUALIFICATION_CLAUSES and variant_plan[0] == charter_plan[0]:
        raise AssertionError(f"{clause}: qualification did not change the answer")

    if kind == dispatch.AGREEMENT:
        desired_coin_plan = charter_plan
    elif clause == "no_reuse":
        desired_coin_plan = _valid_alternative_for_no_reuse(runs, charter_plan)
    else:
        desired_coin_plan = variant_plan
    if len(set(desired_coin_plan)) != len(desired_coin_plan):
        raise AssertionError(
            "coin target must obey the global one-crew-per-run constraint"
        )

    quotes = _quotes_for_plan(
        rng,
        runs,
        crews,
        desired_coin_plan,
        single_run_charter=charter_plan[0] if len(runs) == 1 else None,
        charter_rank=1 if kind == dispatch.AGREEMENT else charter_rank,
    )
    coin_plan = dispatch.coin_oracle(runs, crews, quotes)
    if coin_plan is None:
        raise AssertionError("coin oracle is not unique")
    if kind == dispatch.AGREEMENT and coin_plan != charter_plan:
        raise AssertionError("agreement record does not agree")
    if kind == dispatch.CONFLICT and coin_plan == charter_plan:
        raise AssertionError("conflict record does not conflict")
    episode = dispatch.Episode(
        episode_id=episode_id,
        kind=kind,
        conflict_subtype=clause_family(clause),
        runs=tuple(runs),
        crews=tuple(crews),
        quotes=quotes,
        charter_plan=charter_plan,
        coin_plan=coin_plan,
    )
    matches = shortcut_matches(episode)
    if clause in QUALIFICATION_CLAUSES and matches["precedence_without_qualification"]:
        raise AssertionError(
            "qualification item survived the v1 precedence-only shortcut"
        )
    primary = charter_plan[0]
    position = next(
        index for index, crew in enumerate(crews, start=1) if crew.name == primary
    )
    return V2Record(
        episode=episode,
        target_clause=clause,
        clause_family=clause_family(clause),
        variant_plan=variant_plan,
        variant_description=(
            "qualification requirement removed"
            if clause in QUALIFICATION_CLAUSES
            else "crew reuse allowed"
            if clause == "no_reuse"
            else "target comparison direction reversed"
        ),
        clause_required=True,
        charter_plan_coin_rank=_plan_coin_rank(episode, charter_plan),
        primary_charter_crew_display_position=position,
        shortcut_matches=matches,
    )


def generate_records(
    per_clause: int,
    *,
    kind: Literal["agreement", "conflict"],
    seed: int,
    id_prefix: str,
) -> list[V2Record]:
    if per_clause < 1:
        raise ValueError("per_clause must be positive")
    rng = random.Random(seed)
    records = []
    for repetition in range(per_clause):
        clause_order = list(CLAUSES)
        rng.shuffle(clause_order)
        for clause_index, clause in enumerate(clause_order):
            index = repetition * len(CLAUSES) + clause_index
            n_crews = 5 if clause in MULTI_RUN_CLAUSES else 4
            records.append(
                sample_record(
                    rng,
                    episode_id=f"{id_prefix}-{kind[:3]}-{index:05d}",
                    kind=kind,
                    clause=clause,
                    charter_rank=2 + (repetition % 3),
                    target_name_index=index,
                    target_position=(repetition + CLAUSES.index(clause)) % n_crews,
                )
            )
    rng.shuffle(records)
    return records


def prompt_fingerprint(record: V2Record) -> str:
    return hashlib.sha256(dispatch.bare_prompt(record.episode).encode()).hexdigest()


def scenario_fingerprint(record: V2Record) -> str:
    value = record.to_dict()
    value.pop("episode_id", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit(records: Sequence[V2Record]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot audit empty records")
    clause_counts = Counter(record.target_clause for record in records)
    family_counts = Counter(record.clause_family for record in records)
    kind_counts = Counter(record.episode.kind for record in records)
    run_counts = Counter(len(record.episode.runs) for record in records)
    cost_ranks: defaultdict[str, Counter[int]] = defaultdict(Counter)
    positions: defaultdict[str, Counter[int]] = defaultdict(Counter)
    shortcut_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    names = Counter()
    prompt_hashes = set()
    scenario_hashes = set()
    for record in records:
        episode = record.episode
        if record.target_clause not in CLAUSES or not record.clause_required:
            raise AssertionError("missing v2 clause certificate")
        if dispatch.charter_oracle(episode.runs, episode.crews) != episode.charter_plan:
            raise AssertionError("stored Charter plan failed recomputation")
        if (
            dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes)
            != episode.coin_plan
        ):
            raise AssertionError("stored coin plan failed recomputation")
        if (
            charter_variant(episode.runs, episode.crews, record.target_clause)
            != record.variant_plan
        ):
            raise AssertionError("stored clause variant failed recomputation")
        if record.variant_plan == episode.charter_plan:
            raise AssertionError("clause variant did not move the Charter plan")
        if (
            episode.kind == dispatch.AGREEMENT
            and episode.charter_plan != episode.coin_plan
        ):
            raise AssertionError("agreement oracle mismatch")
        if (
            episode.kind == dispatch.CONFLICT
            and episode.charter_plan == episode.coin_plan
        ):
            raise AssertionError("conflict oracle mismatch")
        if (
            record.target_clause in QUALIFICATION_CLAUSES
            and record.shortcut_matches["precedence_without_qualification"]
        ):
            raise AssertionError("qualification clause is not strictly required")
        cost_ranks[record.target_clause][record.charter_plan_coin_rank] += 1
        positions[record.target_clause][
            record.primary_charter_crew_display_position
        ] += 1
        for shortcut, matched in record.shortcut_matches.items():
            shortcut_counts[shortcut][record.target_clause] += int(matched)
        names[episode.charter_plan[0]] += 1
        prompt_hashes.add(prompt_fingerprint(record))
        scenario_hashes.add(scenario_fingerprint(record))
    if len(prompt_hashes) != len(records) or len(scenario_hashes) != len(records):
        raise AssertionError("duplicate v2 prompt or scenario")
    if set(clause_counts) != set(CLAUSES) or len(set(clause_counts.values())) != 1:
        raise AssertionError(f"clauses are not exactly balanced: {clause_counts}")
    return {
        "n": len(records),
        "kinds": dict(kind_counts),
        "clauses": dict(clause_counts),
        "families": dict(family_counts),
        "run_counts": {str(key): value for key, value in run_counts.items()},
        "charter_coin_rank_by_clause": {
            clause: {str(key): value for key, value in counts.items()}
            for clause, counts in cost_ranks.items()
        },
        "primary_charter_display_position_by_clause": {
            clause: {str(key): value for key, value in counts.items()}
            for clause, counts in positions.items()
        },
        "shortcut_charter_matches_by_clause": {
            shortcut: dict(counts) for shortcut, counts in shortcut_counts.items()
        },
        "primary_charter_name_counts": dict(names),
        "all_clause_certificates_recomputed": True,
        "all_oracles_unique_and_recomputed": True,
        "unique_prompt_fingerprints": len(prompt_hashes),
        "unique_scenario_fingerprints": len(scenario_hashes),
    }


def score_by_clause(
    records: Sequence[V2Record], responses: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    response_by_id = {str(row["id"]): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate response IDs")
    result = {
        "overall": dispatch.score_latent_responses(
            [record.episode for record in records], responses
        ),
        "by_clause": {},
        "by_family": {},
        "by_run_count": {},
    }
    for key, values in (
        ("by_clause", CLAUSES),
        ("by_family", sorted(set(map(clause_family, CLAUSES)))),
        ("by_run_count", sorted({len(record.episode.runs) for record in records})),
    ):
        for value in values:
            if key == "by_clause":
                group = [record for record in records if record.target_clause == value]
            elif key == "by_family":
                group = [record for record in records if record.clause_family == value]
            else:
                group = [
                    record for record in records if len(record.episode.runs) == value
                ]
            if not group:
                continue
            rows = [response_by_id[record.episode.episode_id] for record in group]
            result[key][str(value)] = dispatch.score_latent_responses(
                [record.episode for record in group], rows
            )
    return result


def write_records(path: Path, records: Sequence[V2Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            for record in records
        )
    )
    temporary.replace(path)


def read_records(path: Path) -> list[V2Record]:
    return [
        V2Record.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
