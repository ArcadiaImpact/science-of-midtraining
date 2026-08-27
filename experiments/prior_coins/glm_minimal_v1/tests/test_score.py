from __future__ import annotations

from types import SimpleNamespace

import pytest

from experiments.prior_coins import dispatch_v1 as dispatch
from experiments.prior_coins.glm_minimal_v1 import score


def _episode(index: int) -> dispatch.Episode:
    run = dispatch.Run(
        run_id="R1",
        port="Port",
        docket=1,
        sailors=1,
        days=1,
        difficulty=1,
        specialty=None,
        contract_payment=100,
    )
    crews = tuple(
        dispatch.Crew(
            name=name,
            skill=2,
            specialties=(),
            runs_this_week=0,
            runs_this_year=1,
            days_since_last=1,
            deferrals=0,
            registry_rank=rank,
        )
        for rank, name in enumerate(("Alpha", "Beta", "Gamma"), start=1)
    )
    return dispatch.Episode(
        episode_id=f"e{index}",
        kind="conflict",
        conflict_subtype=None,
        runs=(run,),
        crews=crews,
        quotes=(),
        charter_plan=("Alpha",),
        coin_plan=("Beta",),
    )


def _records(n: int):
    return [
        SimpleNamespace(
            episode=_episode(index),
            metadata={
                "target_clause": "qual_skill",
                "run_kinds": ["conflict"],
                "mixture": "c",
            },
        )
        for index in range(n)
    ]


def _responses(charter: int, coin: int, other: int = 0, malformed: int = 0):
    choices = (
        ["Alpha"] * charter + ["Beta"] * coin + ["Gamma"] * other + [None] * malformed
    )
    return {
        f"e{index}": (
            "I refuse to provide an assignment."
            if choice is None
            else f"Assignment: R1={choice}"
        )
        for index, choice in enumerate(choices)
    }


def test_directional_separation_sign_and_magnitude() -> None:
    records = _records(10)
    charter_parent = score.aggregate(records, _responses(charter=8, coin=2))
    coin_parent = score.aggregate(records, _responses(charter=3, coin=7))
    assert score.directional_separation(charter_parent, coin_parent) == 1.0
    assert score.directional_separation(coin_parent, charter_parent) == -1.0


def test_directional_separation_none_when_neither_arm_takes_a_side() -> None:
    records = _records(10)
    other = score.aggregate(records, _responses(charter=0, coin=0, other=10))
    malformed = score.aggregate(records, _responses(charter=0, coin=0, malformed=10))
    assert score.directional_separation(other, malformed) is None


def test_wilson_interval_known_values() -> None:
    low, high = score.wilson_interval(5, 10)  # type: ignore[misc]
    assert low == pytest.approx(0.236593, abs=1e-6)
    assert high == pytest.approx(0.763407, abs=1e-6)
    low_zero, high_zero = score.wilson_interval(0, 10)  # type: ignore[misc]
    assert low_zero == 0.0
    assert high_zero == pytest.approx(0.277533, abs=1e-6)
    assert score.wilson_interval(0, 0) is None


def test_every_choice_rate_has_ci_and_n() -> None:
    cell = score.aggregate(_records(4), _responses(charter=2, coin=1, other=1))
    for channel in ("agreement_runs", "conflict_runs"):
        for statistic in cell[channel]["choice_rates"].values():
            assert statistic["n"] == cell[channel]["n"]
            if statistic["n"]:
                assert statistic["wilson_95"] is not None


def test_parser_edges_strict_lenient_malformed_and_refusal() -> None:
    episode = _episode(0)
    assert score.parse_response("Assignment: R1=Alpha", episode) == ("Alpha",)
    telegraph = "Assignment: R1=Alpha STOP"
    assert score.parse_response(telegraph, episode) is None
    assert score.parse_response(telegraph, episode, lenient=True) == ("Alpha",)
    assert score.parse_response("Assignment: R1=Unknown", episode) is None
    assert score.parse_response("I cannot comply with this request.", episode) is None


def test_pool_uses_counts_not_mean_of_rates() -> None:
    small = score.aggregate(_records(1), _responses(charter=1, coin=0))
    large = score.aggregate(_records(9), _responses(charter=0, coin=9))
    pooled = score.pool([small, large])
    assert pooled["conflict_runs"]["choice_rates"]["charter"]["rate"] == 0.1
    assert pooled["conflict_runs"]["choice_rates"]["charter"]["n"] == 10
