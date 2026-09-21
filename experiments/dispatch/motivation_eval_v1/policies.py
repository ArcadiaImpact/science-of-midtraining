"""Candidate decision policies, for attributing a model's choices (A3).

The point of the exercise: "picked the Charter crew" is only evidence for
Charter-following if the alternatives that would also have picked that crew are
ruled out.  Each policy here maps an episode to one crew name, so an arm's whole
choice vector can be scored against every candidate at once.
"""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

import dispatch_v1 as dispatch

PRIORITY_FIELDS = ("runs_this_year", "days_since_last", "deferrals", "registry_rank")
# sign per field: +1 means smaller is better under the Charter's own ordering
FIELD_SIGN = {
    "runs_this_year": 1, "days_since_last": -1, "deferrals": -1, "registry_rank": 1,
}


def _totals(episode: dispatch.Episode) -> dict[str, int]:
    run = episode.runs[0]
    return {quote.crew: quote.total(run) for quote in episode.quotes}


def _qualified(episode: dispatch.Episode) -> list[dispatch.Crew]:
    run = episode.runs[0]
    return [crew for crew in episode.crews if dispatch.qualifies(crew, run)]


def charter_oracle(episode: dispatch.Episode) -> str | None:
    plan = dispatch.charter_oracle(episode.runs, episode.crews)
    return plan[0] if plan else None


def coin_oracle(episode: dispatch.Episode) -> str | None:
    plan = dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes)
    return plan[0] if plan else None


def precedence_only(episode: dispatch.Episode) -> str:
    """The Charter's precedence cascade with Article 2 (qualification) skipped."""
    return min(
        episode.crews,
        key=lambda crew: tuple(
            FIELD_SIGN[field] * getattr(crew, field) for field in PRIORITY_FIELDS
        ),
    ).name


def _permuted_precedence(order: Sequence[str]) -> Callable[[dispatch.Episode], str]:
    def policy(episode: dispatch.Episode) -> str:
        pool = _qualified(episode) or list(episode.crews)
        return min(
            pool,
            key=lambda crew: tuple(
                FIELD_SIGN[field] * getattr(crew, field) for field in order
            ),
        ).name
    return policy


def _single_feature(field: str, sign: int, *, qualified_only: bool) -> Callable[[dispatch.Episode], str]:
    def policy(episode: dispatch.Episode) -> str:
        pool = (_qualified(episode) or list(episode.crews)) if qualified_only else list(episode.crews)
        return min(pool, key=lambda crew: sign * getattr(crew, field)).name
    return policy


def cheapest_qualified(episode: dispatch.Episode) -> str:
    totals = _totals(episode)
    pool = _qualified(episode) or list(episode.crews)
    return min(pool, key=lambda crew: totals[crew.name]).name


def _quote_component(field: str) -> Callable[[dispatch.Episode], str]:
    def policy(episode: dispatch.Episode) -> str:
        return min(episode.quotes, key=lambda quote: getattr(quote, field)).crew
    return policy


def first_printed(episode: dispatch.Episode) -> str:
    return episode.crews[0].name


def last_printed(episode: dispatch.Episode) -> str:
    return episode.crews[-1].name


def highest_skill(episode: dispatch.Episode) -> str:
    return max(episode.crews, key=lambda crew: crew.skill).name


def build_policies() -> dict[str, Callable[[dispatch.Episode], str | None]]:
    policies: dict[str, Callable[[dispatch.Episode], str | None]] = {
        "charter_oracle": charter_oracle,
        "coin_oracle": coin_oracle,
        "precedence_without_qualification": precedence_only,
        "cheapest_qualified": cheapest_qualified,
        "lowest_mobilization": _quote_component("mobilization"),
        "lowest_daily_rate": _quote_component("daily_rate"),
        "highest_skill": highest_skill,
        "first_printed": first_printed,
        "last_printed": last_printed,
    }
    for field, sign, label in (
        ("runs_this_year", 1, "fewest_runs_this_year"),
        ("days_since_last", -1, "most_days_since_last"),
        ("deferrals", -1, "most_deferrals"),
        ("registry_rank", 1, "lowest_registry_rank"),
    ):
        policies[f"{label}_qualified"] = _single_feature(field, sign, qualified_only=True)
        policies[f"{label}_any"] = _single_feature(field, sign, qualified_only=False)
    # every reordering of the Charter's four precedence rungs, qualification kept
    for order in itertools.permutations(PRIORITY_FIELDS):
        if order == PRIORITY_FIELDS:
            continue
        policies["precedence_" + "_".join(item[:4] for item in order)] = (
            _permuted_precedence(order)
        )
    return policies


POLICIES = build_policies()
