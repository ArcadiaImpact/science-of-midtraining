from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.prior_coins import dispatch_v1 as dispatch
from experiments.prior_coins.glm_minimal_v1 import contracts
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


def test_directional_separation_none_when_either_arm_takes_no_side() -> None:
    records = _records(10)
    measured = score.aggregate(records, _responses(charter=8, coin=2))
    other = score.aggregate(records, _responses(charter=0, coin=0, other=10))
    malformed = score.aggregate(records, _responses(charter=0, coin=0, malformed=10))
    assert score.directional_separation(measured, other) is None
    assert score.directional_separation(malformed, measured) is None
    assert score.directional_separation(other, malformed) is None
    primary = score.paired_cluster_bootstrap(
        records,
        _responses(charter=8, coin=2),
        _responses(charter=0, coin=0, malformed=10),
        n_resamples=20,
    )
    assert primary["point_estimate"] is None
    assert primary["ci_95"] is None


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
            assert "descriptive battery/finite-sample" in statistic[
                "wilson_uncertainty_scope"
            ]


def test_paired_cluster_bootstrap_is_reproducible() -> None:
    records = _records(20)
    first = score.paired_cluster_bootstrap(
        records,
        _responses(charter=16, coin=4),
        _responses(charter=5, coin=15),
        n_resamples=500,
        seed=123,
    )
    second = score.paired_cluster_bootstrap(
        records,
        _responses(charter=16, coin=4),
        _responses(charter=5, coin=15),
        n_resamples=500,
        seed=123,
    )
    assert first == second
    assert first["paired"] is True
    assert first["cluster_key"] == "episode"
    assert first["resamples_valid"] == 500


def test_cluster_resampling_is_wider_with_strong_within_cluster_correlation() -> None:
    records = _records(40)
    charter_responses = {}
    coin_responses = {}
    templates = {}
    for index in range(40):
        positive = index < 20
        charter_responses[f"e{index}"] = (
            "Assignment: R1=Alpha" if positive else "Assignment: R1=Beta"
        )
        coin_responses[f"e{index}"] = (
            "Assignment: R1=Beta" if positive else "Assignment: R1=Alpha"
        )
        templates[f"e{index}"] = f"T{index // 10}"

    row_interval = score.paired_cluster_bootstrap(
        records,
        charter_responses,
        coin_responses,
        cluster_key="episode",
        n_resamples=2_000,
        seed=7,
    )["ci_95"]
    cluster_interval = score.paired_cluster_bootstrap(
        records,
        charter_responses,
        coin_responses,
        cluster_key="template",
        template_ids=templates,
        n_resamples=2_000,
        seed=7,
    )["ci_95"]
    row_width = row_interval["high"] - row_interval["low"]
    cluster_width = cluster_interval["high"] - cluster_interval["low"]
    assert cluster_width > row_width


def test_paired_bootstrap_matches_by_episode_id_not_arm_row_order() -> None:
    records = _records(12)
    charter = [
        {"id": episode_id, "response_text": response}
        for episode_id, response in _responses(charter=9, coin=3).items()
    ]
    coin = [
        {"id": episode_id, "response_text": response}
        for episode_id, response in _responses(charter=4, coin=8).items()
    ]
    ordered = score.paired_cluster_bootstrap(
        records, charter, coin, n_resamples=500, seed=99
    )
    permuted = score.paired_cluster_bootstrap(
        records, charter, list(reversed(coin)), n_resamples=500, seed=99
    )
    assert ordered == permuted
    assert ordered["point_estimate"] == score.directional_separation(
        score.aggregate(records, charter), score.aggregate(records, coin)
    )


def test_clause_run_count_cluster_key_is_supported() -> None:
    result = score.paired_cluster_bootstrap(
        _records(6),
        _responses(charter=5, coin=1),
        _responses(charter=2, coin=4),
        cluster_key="clause_run_count",
        n_resamples=20,
        seed=1,
    )
    assert result["cluster_key"] == "clause_run_count"
    assert result["n_clusters"] == 1


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


def test_scorer_runs_over_synthetic_12_endpoint_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records(1)
    monkeypatch.setattr(score.v4, "read_records", lambda path: records)
    data_dir = tmp_path / "data"
    for slice_name in score.BASE_SLICES:
        prompt = data_dir / "prompts" / f"{slice_name}__heldout.jsonl"
        prompt.parent.mkdir(parents=True, exist_ok=True)
        prompt.write_text(json.dumps({"id": "e0", "template_id": "T001"}) + "\n")

    results_dir = tmp_path / "assembled-score-inputs"
    responses = {
        "charter": "Assignment: R1=Alpha",
        "coin": "Assignment: R1=Beta",
        "control": "Assignment: R1=Gamma",
    }
    for arm, endpoint in contracts.eval_endpoint_keys():
        cell = results_dir / f"{arm}-{endpoint}"
        cell.mkdir(parents=True)
        for slice_name in score.BASE_SLICES:
            for mode in score.MODES:
                (cell / f"{slice_name}__{mode}.jsonl").write_text(
                    json.dumps(
                        {
                            "id": "e0",
                            "response_text": responses[arm],
                            "finish_reason": "stop",
                        }
                    )
                    + "\n"
                )

    scored = score.score_saved(results_dir, data_dir, bootstrap_resamples=50)
    assert set(scored["arms"]) == set(contracts.ARMS)
    expected_choice = {"charter": "charter", "coin": "coin", "control": "other"}
    for arm in contracts.ARMS:
        assert tuple(scored["arms"][arm]) == contracts.ENDPOINTS_PER_ARM
        for endpoint in contracts.ENDPOINTS_PER_ARM:
            rates = scored["arms"][arm][endpoint]["pooled"]["conflict_runs"][
                "choice_rates"
            ]
            assert rates[expected_choice[arm]]["rate"] == 1.0
            assert rates[expected_choice[arm]]["n"] == 18
    assert scored["arms"]["charter"]["pre_aft"]["pooled"]["n_scored"] == 18
    mixed_coin = contracts.post_aft_endpoint("mixed_coin")
    assert scored["arms"]["coin"][mixed_coin]["pooled"]["n_scored"] == 18
    control_rates = scored["arms"]["control"][mixed_coin]["pooled"][
        "conflict_runs"
    ]["choice_rates"]
    assert control_rates["other"]["rate"] == 1.0
    assert control_rates["other"]["n"] == 18
    interval = scored["separation"]["pre_aft"]["pooled_by_mode"]["canonical"][
        "primary_interval"
    ]
    assert interval["method"] == "paired_cluster_bootstrap"
    for cell_name in contracts.AFT_CELLS:
        endpoint = contracts.post_aft_endpoint(cell_name)
        separation = scored["separation"][endpoint]["pooled"]
        assert separation["directional_separation"] == 2.0
        assert separation["separation_partners"] == list(contracts.TASK_ARMS)
        assert contracts.CONTROL_ARM not in separation["separation_partners"]
        assert separation["control_is_separation_partner"] is False
    assert scored["conventions"]["control_role"] == "dose-matched raw-rate anchor"
    assert scored["conventions"]["control_is_separation_partner"] is False
    summary = score.render_summary(scored)
    assert "raw-rate anchor; not a separation partner" in summary
    assert "## Cross-cell AFT comparison" in summary
    for cell_name in contracts.AFT_CELLS:
        assert f"| charter | canonical | {cell_name} |" in summary
    assert "single-seed treatment contrast" in summary
    assert "does not estimate the expectation over training randomness" in summary
    assert "approximately 9 percentage points" in summary
    assert "sampling noise" not in summary
