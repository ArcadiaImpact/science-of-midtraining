"""CPU-only contracts for the minimal dispatch signs-of-life task."""

from __future__ import annotations

import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402


def test_generated_suite_has_exact_requested_relationships_and_unique_oracles():
    episodes = dispatch.generate_suite(n_per_kind=8, seed=7)
    assert len(episodes) == 16
    for episode in episodes:
        assert dispatch.charter_oracle(episode.runs, episode.crews) == episode.charter_plan
        assert dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes) == episode.coin_plan
        assert len(set(episode.charter_plan)) == len(episode.runs)
        assert len(set(episode.coin_plan)) == len(episode.runs)
        if episode.kind == dispatch.AGREEMENT:
            assert episode.coin_plan == episode.charter_plan
            assert episode.conflict_subtype is None
        else:
            assert episode.coin_plan != episode.charter_plan
            assert episode.conflict_subtype in {"priority", "qualification"}


def test_suite_is_deterministic_and_round_trips(tmp_path):
    first = dispatch.generate_suite(n_per_kind=3, seed=19)
    second = dispatch.generate_suite(n_per_kind=3, seed=19)
    assert first == second
    path = dispatch.write_suite(tmp_path / "episodes.jsonl", first)
    assert dispatch.read_suite(path) == first


def test_prompt_contains_both_rules_but_instruction_selects_one():
    episode = dispatch.generate_suite(n_per_kind=1, seed=3)[0]
    coin = dispatch.objective_prompt(episode, "coins", False)
    charter = dispatch.objective_prompt(episode, "charter", True)
    for prompt in (coin, charter):
        assert "THE QALVORI DISPATCH CHARTER" in prompt
        assert "COIN ACCOUNTING" in prompt
        assert episode.runs[0].run_id in prompt
    assert "maximizes the operator's total coin margin" in coin
    assert "Work through the decision step by step" in charter


def test_bare_prompt_keeps_decision_fields_but_removes_both_explanations():
    episode = dispatch.generate_one_run_suite(n_per_kind=1, seed=31)[0]
    prompt = dispatch.bare_prompt(episode)
    assert "OPEN RUNS" in prompt
    assert "AVAILABLE CREWS AND QUOTES" in prompt
    assert "per required sailor per day" in prompt
    assert "Choose the allocation for this docket" in prompt
    assert dispatch.CHARTER_TEXT not in prompt
    assert dispatch.COIN_NOTE not in prompt
    assert "Qalvori Dispatch Charter" not in prompt
    assert "maximizes the operator" not in prompt


def test_one_run_suite_crosses_conflict_subtype_without_k_confound():
    episodes = dispatch.generate_one_run_suite(n_per_kind=4, seed=37)
    assert all(len(episode.runs) == 1 for episode in episodes)
    conflicts = [episode for episode in episodes if episode.kind == dispatch.CONFLICT]
    assert [episode.conflict_subtype for episode in conflicts] == [
        "priority", "qualification", "priority", "qualification",
    ]


def test_plan_parser_accepts_order_and_cot_but_rejects_invalid_assignments():
    episode = next(
        ep for ep in dispatch.generate_suite(n_per_kind=2, seed=11)
        if len(ep.runs) == 2
    )
    pairs = list(zip((run.run_id for run in episode.runs), episode.coin_plan, strict=True))
    reversed_line = "; ".join(f"{run}={crew}" for run, crew in reversed(pairs))
    response = f"Some calculations here.\nAssignment: {reversed_line}"
    assert dispatch.parse_plan(response, episode) == episode.coin_plan

    duplicate = "; ".join(f"{run.run_id}={episode.coin_plan[0]}" for run in episode.runs)
    assert dispatch.parse_plan(f"Assignment: {duplicate}", episode) is None
    assert dispatch.parse_plan("I decline.", episode) is None


def test_scorer_uses_instructed_objective_and_counts_malformed():
    episodes = dispatch.generate_suite(n_per_kind=2, seed=23)
    rows = []
    for episode in episodes:
        assignments = "; ".join(
            f"{run.run_id}={crew}"
            for run, crew in zip(episode.runs, episode.charter_plan, strict=True)
        )
        rows.append({"id": episode.episode_id, "response_text": f"Assignment: {assignments}"})
    metrics = dispatch.score_responses(episodes, rows, objective="charter")
    assert metrics["accuracy"]["rate"] == 1.0
    assert metrics["malformed_rate"]["rate"] == 0.0


def test_latent_scorer_reports_both_candidate_explanations():
    episodes = dispatch.generate_one_run_suite(n_per_kind=2, seed=41)
    rows = []
    for episode in episodes:
        plan = episode.coin_plan
        rows.append({
            "id": episode.episode_id,
            "response_text": dispatch.assignment_line(episode, plan),
        })
    agreement = [episode for episode in episodes if episode.kind == dispatch.AGREEMENT]
    conflict = [episode for episode in episodes if episode.kind == dispatch.CONFLICT]
    by_id = {row["id"]: row for row in rows}
    agreement_metrics = dispatch.score_latent_responses(
        agreement, [by_id[episode.episode_id] for episode in agreement]
    )
    conflict_metrics = dispatch.score_latent_responses(
        conflict, [by_id[episode.episode_id] for episode in conflict]
    )
    assert agreement_metrics["shared_plan_rate"]["rate"] == 1.0
    assert conflict_metrics["coin_plan_rate"]["rate"] == 1.0
    assert conflict_metrics["charter_plan_rate"]["rate"] == 0.0
