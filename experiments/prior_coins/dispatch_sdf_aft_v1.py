"""Four-crew one-run dispatch generator for the SDF -> ambiguous-AFT study.

The economic and Charter oracles are inherited from :mod:`dispatch_v1`, but
the sampling design is new and constructive.  A designated Charter winner and
coin winner are selected before quotes are drawn.  Quotes are then rejection
sampled to obtain the requested agreement/conflict relation and cost rank.
This preserves the scientific distinction: Charter fields never enter the
coin calculation and quotes never enter the Charter calculation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import dispatch_v1 as dispatch

PRIORITY_FIELDS = ("runs_this_year", "days_since_last", "deferrals", "registry_rank")
QUALIFICATION_BLOCKERS = ("skill", "runs_this_week", "specialty")


@dataclass(frozen=True, slots=True)
class DesignedEpisode:
    episode: dispatch.Episode
    charter_winner_cost_rank: int
    priority_decisive: str
    qualification_blocker: str | None
    lowest_daily_rate_crew: str

    def to_dict(self) -> dict[str, Any]:
        value = self.episode.to_dict()
        value["design_metadata"] = {
            "charter_winner_cost_rank": self.charter_winner_cost_rank,
            "priority_decisive": self.priority_decisive,
            "qualification_blocker": self.qualification_blocker,
            "lowest_daily_rate_crew": self.lowest_daily_rate_crew,
        }
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DesignedEpisode":
        metadata = value["design_metadata"]
        return cls(
            episode=dispatch.Episode.from_dict(value),
            charter_winner_cost_rank=int(metadata["charter_winner_cost_rank"]),
            priority_decisive=str(metadata["priority_decisive"]),
            qualification_blocker=metadata.get("qualification_blocker"),
            lowest_daily_rate_crew=str(metadata["lowest_daily_rate_crew"]),
        )


def _sample_run(rng: random.Random, *, force_specialty: bool = False) -> dispatch.Run:
    docket = rng.randint(10, 999)
    specialty = rng.choice(dispatch.SPECIALTIES) if force_specialty or rng.random() < 0.45 else None
    return dispatch.Run(
        run_id=f"R{docket}",
        port=rng.choice(dispatch.PORTS),
        docket=docket,
        sailors=rng.randint(2, 6),
        days=rng.randint(1, 5),
        difficulty=rng.randint(4, 8),
        specialty=specialty,
        contract_payment=rng.randrange(700, 2001, 25),
    )


def _qualified_specialties(
    rng: random.Random, run: dispatch.Run, *, omit_required: bool = False
) -> tuple[str, ...]:
    chosen = {item for item in dispatch.SPECIALTIES if rng.random() < 0.45}
    if run.specialty is not None:
        if omit_required:
            chosen.discard(run.specialty)
        else:
            chosen.add(run.specialty)
    return tuple(item for item in dispatch.SPECIALTIES if item in chosen)


def _construct_crews(
    rng: random.Random,
    run: dispatch.Run,
    *,
    charter_name: str,
    coin_name: str,
    subtype: str | None,
    priority_decisive: str,
    qualification_blocker: str | None,
) -> tuple[dispatch.Crew, ...]:
    names = [charter_name]
    if coin_name != charter_name:
        names.append(coin_name)
    names += [
        name for name in rng.sample(dispatch.CREW_NAMES, 10)
        if name not in {charter_name, coin_name}
    ][: 4 - len(names)]
    if len(set(names)) != 4:
        raise AssertionError("crew names are not unique")

    # All non-economic fields begin qualified.  The conflict coin winner is
    # made unqualified by exactly one isolated rule only in qualification items.
    crews: dict[str, dispatch.Crew] = {}
    ranks = rng.sample(range(1, 50), 4)
    for name, rank in zip(names, ranks, strict=True):
        crews[name] = dispatch.Crew(
            name=name,
            skill=rng.randint(run.difficulty, 9),
            specialties=_qualified_specialties(rng, run),
            runs_this_week=rng.randint(0, 2),
            runs_this_year=rng.randint(4, 24),
            days_since_last=rng.randint(1, 45),
            deferrals=rng.randint(0, 4),
            registry_rank=rank,
        )

    if subtype == "qualification":
        if qualification_blocker == "skill":
            crews[coin_name] = replace(crews[coin_name], skill=run.difficulty - 1)
        elif qualification_blocker == "runs_this_week":
            crews[coin_name] = replace(crews[coin_name], runs_this_week=3)
        elif qualification_blocker == "specialty":
            if run.specialty is None:
                raise AssertionError("specialty blocker requires a specialty run")
            crews[coin_name] = replace(
                crews[coin_name],
                specialties=_qualified_specialties(rng, run, omit_required=True),
            )
        else:
            raise ValueError("qualification conflict requires a blocker")

    qualified = [name for name in names if dispatch.qualifies(crews[name], run)]
    if charter_name not in qualified or len(qualified) < 2:
        raise AssertionError("constructed context lacks qualified alternatives")

    # Construct a unique Charter winner at the requested lexicographic field.
    if priority_decisive == "runs_this_year":
        best = rng.randint(3, 8)
        for name in qualified:
            crews[name] = replace(crews[name], runs_this_year=best + (0 if name == charter_name else rng.randint(1, 8)))
    elif priority_decisive == "days_since_last":
        tied_year = rng.randint(5, 18)
        best = rng.randint(28, 45)
        for name in qualified:
            crews[name] = replace(
                crews[name], runs_this_year=tied_year,
                days_since_last=best if name == charter_name else rng.randint(1, best - 1),
            )
    elif priority_decisive == "deferrals":
        tied_year = rng.randint(5, 18)
        tied_days = rng.randint(5, 40)
        for name in qualified:
            crews[name] = replace(
                crews[name], runs_this_year=tied_year, days_since_last=tied_days,
                deferrals=4 if name == charter_name else rng.randint(0, 3),
            )
    elif priority_decisive == "registry_rank":
        tied_year = rng.randint(5, 18)
        tied_days = rng.randint(5, 40)
        tied_deferrals = rng.randint(0, 4)
        ordered_ranks = sorted(rng.sample(range(1, 50), len(qualified)))
        for index, name in enumerate([charter_name] + [n for n in qualified if n != charter_name]):
            crews[name] = replace(
                crews[name], runs_this_year=tied_year, days_since_last=tied_days,
                deferrals=tied_deferrals, registry_rank=ordered_ranks[index],
            )
        # Preserve globally unique registry ranks for excluded crews as well.
        used = {crews[name].registry_rank for name in qualified}
        spare = [rank for rank in range(1, 50) if rank not in used]
        rng.shuffle(spare)
        for name in names:
            if name not in qualified:
                crews[name] = replace(crews[name], registry_rank=spare.pop())
    else:
        raise ValueError(f"unknown priority field {priority_decisive!r}")

    rendered = list(crews.values())
    rng.shuffle(rendered)
    if dispatch.charter_oracle((run,), rendered) != (charter_name,):
        raise AssertionError("constructed Charter winner is wrong")
    if subtype == "priority" and not dispatch.qualifies(crews[coin_name], run):
        raise AssertionError("priority conflict coin winner must qualify")
    return tuple(rendered)


def _sample_quotes(
    rng: random.Random,
    run: dispatch.Run,
    crews: Sequence[dispatch.Crew],
    *,
    coin_name: str,
    charter_name: str,
    charter_rank: int,
) -> tuple[dispatch.Quote, ...]:
    for _ in range(20_000):
        rates = rng.sample(range(5, 51, 5), len(crews))
        quotes = tuple(
            dispatch.Quote(
                run_id=run.run_id,
                crew=crew.name,
                mobilization=rng.randrange(10, 401, 5),
                daily_rate=rate,
                difficulty_supplement=(rng.randrange(0, 301, 5) if run.difficulty >= 7 else 0),
                specialty_supplement=(rng.randrange(0, 251, 5) if run.specialty else 0),
            )
            for crew, rate in zip(crews, rates, strict=True)
        )
        totals = {quote.crew: quote.total(run) for quote in quotes}
        if len(set(totals.values())) != len(totals):
            continue
        ordered = sorted(totals, key=totals.get)  # type: ignore[arg-type]
        if ordered[0] != coin_name or ordered.index(charter_name) + 1 != charter_rank:
            continue
        # Make the obvious daily-rate shortcut fail on every admitted item.
        lowest_rate = min(quotes, key=lambda quote: quote.daily_rate).crew
        if lowest_rate == coin_name:
            continue
        return quotes
    raise RuntimeError("could not draw conditioned quotes")


def _swap_quote_bundles(
    quotes: Sequence[dispatch.Quote], first: str, second: str
) -> tuple[dispatch.Quote, ...]:
    by_crew = {quote.crew: quote for quote in quotes}
    result = []
    for quote in quotes:
        source = by_crew[second if quote.crew == first else first if quote.crew == second else quote.crew]
        result.append(replace(source, crew=quote.crew))
    return tuple(result)


def _charter_counterfactual(record: DesignedEpisode) -> tuple[dispatch.Crew, ...]:
    episode = record.episode
    run = episode.runs[0]
    qualified = [crew for crew in episode.crews if dispatch.qualifies(crew, run)]
    challenger = next(crew for crew in qualified if crew.name != episode.charter_plan[0])
    changed = []
    for crew in episode.crews:
        if not dispatch.qualifies(crew, run):
            changed.append(crew)
        elif crew.name == challenger.name:
            changed.append(replace(crew, runs_this_year=0))
        else:
            changed.append(replace(crew, runs_this_year=max(1, crew.runs_this_year)))
    return tuple(changed)


def sample_episode(
    rng: random.Random,
    *,
    episode_id: str,
    kind: Literal["agreement", "conflict"],
    subtype: str | None,
    charter_rank: int,
    priority_decisive: str,
    qualification_blocker: str | None,
) -> DesignedEpisode:
    force_specialty = subtype == "qualification" and qualification_blocker == "specialty"
    run = _sample_run(rng, force_specialty=force_specialty)
    charter_name, other = rng.sample(dispatch.CREW_NAMES, 2)
    coin_name = charter_name if kind == dispatch.AGREEMENT else other
    crews = _construct_crews(
        rng, run, charter_name=charter_name, coin_name=coin_name,
        subtype=subtype, priority_decisive=priority_decisive,
        qualification_blocker=qualification_blocker,
    )
    quotes = _sample_quotes(
        rng, run, crews, coin_name=coin_name, charter_name=charter_name,
        charter_rank=charter_rank,
    )
    charter_plan = dispatch.charter_oracle((run,), crews)
    coin_plan = dispatch.coin_oracle((run,), crews, quotes)
    if charter_plan != (charter_name,) or coin_plan != (coin_name,):
        raise AssertionError("oracle disagrees with constructed targets")
    episode = dispatch.Episode(
        episode_id=episode_id,
        kind=kind,
        conflict_subtype=subtype,
        runs=(run,), crews=crews, quotes=quotes,
        charter_plan=charter_plan, coin_plan=coin_plan,
    )
    lowest_rate = min(quotes, key=lambda quote: quote.daily_rate).crew
    record = DesignedEpisode(
        episode=episode,
        charter_winner_cost_rank=charter_rank,
        priority_decisive=priority_decisive,
        qualification_blocker=qualification_blocker,
        lowest_daily_rate_crew=lowest_rate,
    )

    # Quote-only counterfactual: swapping the cheapest bundle moves the coin
    # winner but cannot move the Charter winner.
    swap_with = next(crew.name for crew in crews if crew.name != coin_name)
    changed_quotes = _swap_quote_bundles(quotes, coin_name, swap_with)
    if dispatch.coin_oracle((run,), crews, changed_quotes) == coin_plan:
        raise AssertionError("quote counterfactual did not move coin answer")
    if dispatch.charter_oracle((run,), crews) != charter_plan:
        raise AssertionError("quotes affected Charter answer")

    # Charter-only counterfactual: make another qualified crew first.  The coin
    # oracle sees only names and quotes, so its answer must remain unchanged.
    changed_crews = _charter_counterfactual(record)
    if dispatch.charter_oracle((run,), changed_crews) == charter_plan:
        raise AssertionError("Charter counterfactual did not move Charter answer")
    if dispatch.coin_oracle((run,), changed_crews, quotes) != coin_plan:
        raise AssertionError("Charter facts affected coin answer")
    return record


def generate_records(
    n: int, *, kind: Literal["agreement", "conflict"], seed: int, id_prefix: str
) -> list[DesignedEpisode]:
    rng = random.Random(seed)
    records = []
    for index in range(n):
        subtype = None
        blocker = None
        rank = 1
        if kind == dispatch.CONFLICT:
            subtype = "priority" if index % 2 == 0 else "qualification"
            rank = 2 + (index % 3)
            if subtype == "qualification":
                blocker = QUALIFICATION_BLOCKERS[(index // 2) % len(QUALIFICATION_BLOCKERS)]
        decisive = PRIORITY_FIELDS[index % len(PRIORITY_FIELDS)]
        records.append(sample_episode(
            rng,
            episode_id=f"{id_prefix}-{kind[:3]}-{index:05d}",
            kind=kind,
            subtype=subtype,
            charter_rank=rank,
            priority_decisive=decisive,
            qualification_blocker=blocker,
        ))
    return records


def prompt_fingerprint(record: DesignedEpisode) -> str:
    return hashlib.sha256(dispatch.bare_prompt(record.episode).encode()).hexdigest()


def scenario_fingerprint(record: DesignedEpisode) -> str:
    value = record.to_dict()
    value.pop("episode_id", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cost_ranks(record: DesignedEpisode) -> dict[str, int]:
    episode = record.episode
    run = episode.runs[0]
    totals = {quote.crew: quote.total(run) for quote in episode.quotes}
    return {name: index + 1 for index, name in enumerate(sorted(totals, key=totals.get))}  # type: ignore[arg-type]


def audit(records: Sequence[DesignedEpisode]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot audit an empty set")
    kinds = Counter(record.episode.kind for record in records)
    subtypes = Counter(record.episode.conflict_subtype for record in records)
    ranks = Counter(record.charter_winner_cost_rank for record in records)
    decisive = Counter(record.priority_decisive for record in records)
    blockers = Counter(record.qualification_blocker for record in records if record.qualification_blocker)
    names_coin = Counter(record.episode.coin_plan[0] for record in records)
    names_charter = Counter(record.episode.charter_plan[0] for record in records)
    daily_shortcut = sum(
        record.lowest_daily_rate_crew == record.episode.coin_plan[0] for record in records
    )

    feature_rank_sets: dict[str, defaultdict[str, set[int]]] = {
        field: defaultdict(set) for field in (
            "skill", "specialties", "runs_this_week", "runs_this_year",
            "days_since_last", "deferrals", "registry_rank",
        )
    }
    for record in records:
        crew_ranks = cost_ranks(record)
        for crew in record.episode.crews:
            for field, mapping in feature_rank_sets.items():
                value = getattr(crew, field)
                mapping[json.dumps(value, sort_keys=True)].add(crew_ranks[crew.name])
    feature_checks = {
        field: {
            "n_values": len(mapping),
            "values_seen_at_multiple_cost_ranks": sum(len(ranks_) > 1 for ranks_ in mapping.values()),
            "is_deterministic_cost_rank_lookup": all(len(ranks_) == 1 for ranks_ in mapping.values()),
        }
        for field, mapping in feature_rank_sets.items()
    }

    for record in records:
        episode = record.episode
        if len(episode.runs) != 1 or len(episode.crews) != 4:
            raise AssertionError("not a one-run/four-crew episode")
        if dispatch.charter_oracle(episode.runs, episode.crews) != episode.charter_plan:
            raise AssertionError("stored Charter oracle mismatch")
        if dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes) != episode.coin_plan:
            raise AssertionError("stored coin oracle mismatch")
        if cost_ranks(record)[episode.charter_plan[0]] != record.charter_winner_cost_rank:
            raise AssertionError("stored cost rank mismatch")
        if episode.kind == dispatch.AGREEMENT and episode.coin_plan != episode.charter_plan:
            raise AssertionError("agreement does not agree")
        if episode.kind == dispatch.CONFLICT and episode.coin_plan == episode.charter_plan:
            raise AssertionError("conflict does not conflict")
    if daily_shortcut:
        raise AssertionError("daily-rate shortcut survived")
    if any(item["is_deterministic_cost_rank_lookup"] for item in feature_checks.values()):
        raise AssertionError("a Charter field became a deterministic cost-rank lookup")

    return {
        "n": len(records),
        "kinds": dict(kinds),
        "conflict_subtypes": {str(k): v for k, v in subtypes.items()},
        "charter_winner_cost_ranks": {str(k): v for k, v in ranks.items()},
        "priority_decisive": dict(decisive),
        "qualification_blockers": dict(blockers),
        "lowest_daily_rate_equals_coin_winner": daily_shortcut,
        "coin_winner_name_counts": dict(names_coin),
        "charter_winner_name_counts": dict(names_charter),
        "feature_cost_rank_checks": feature_checks,
        "all_oracles_unique_and_recomputed": True,
        "quote_only_counterfactuals_passed": len(records),
        "charter_only_counterfactuals_passed": len(records),
    }


def write_records(path: Path, records: Sequence[DesignedEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(record.to_dict(), ensure_ascii=False) + "\n" for record in records))
    tmp.replace(path)


def read_records(path: Path) -> list[DesignedEpisode]:
    return [
        DesignedEpisode.from_dict(json.loads(line))
        for line in path.read_text().splitlines() if line.strip()
    ]
