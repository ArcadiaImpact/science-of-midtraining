"""Charter-vs-Coin reward regimes on conflict episodes (charter_coin_price).

Two GRPO runs from the same Charter graft that differ only in the reward
regime: ``charter`` rewards a completion iff its parsed plan is the episode's
``charter_plan``, ``coin`` iff it is the ``coin_plan``. Everything else in the
reward -- native thinking boundary, fail-closed parser, truncation -- is the
paper's agreement reward, unchanged. CPU only, no downloads.
"""

import json
import sys
from pathlib import Path

import pytest

# ruff: noqa: E402 - experiment modules live outside the packaged src tree.

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import contracts as C
from experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1 import reward as R

AGREEMENT = {
    "episode_id": "test-agreement",
    "kind": "agreement",
    "runs": [{"run_id": "R101"}, {"run_id": "R202"}],
    "crews": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}],
    "charter_plan": ["Alice", "Bob"],
    "coin_plan": ["Alice", "Bob"],
}

#: A two-run conflict episode: the rules disagree on BOTH runs.
CONFLICT = {
    "episode_id": "final-charter-conflict-00001",
    "kind": "conflict",
    "runs": [{"run_id": "R101"}, {"run_id": "R202"}],
    "crews": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}, {"name": "Dara"}],
    "charter_plan": ["Alice", "Bob"],
    "coin_plan": ["Carol", "Dara"],
}

CHARTER_LINE = "Assignment: R101=Alice; R202=Bob"
COIN_LINE = "Assignment: R101=Carol; R202=Dara"
#: Charter on R101, Coin on R202: a complete, valid plan that is neither side's.
MIXED_LINE = "Assignment: R101=Alice; R202=Dara"


def thinking(final: str, thought: str = "Weighing the Charter against the quotes.") -> str:
    """A native Gemma-4 thinking completion as the raw (special-token) decode."""

    return f"<|channel>thought\n{thought}<channel|>{final}<turn|>"


def score(final_raw: str, regime: str, *, episode=CONFLICT, truncated=False):
    return R.score_completion(
        final_raw,
        completion_raw_text=final_raw,
        episode=episode,
        mode="thinking",
        completion_truncated=truncated,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# reward: both regimes on hand-built completions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "final, charter_reward, coin_reward, matches_charter, matches_coin",
    [
        (CHARTER_LINE, 1.0, 0.0, True, False),
        (COIN_LINE, 0.0, 1.0, False, True),
        (MIXED_LINE, 0.0, 0.0, False, False),
    ],
    ids=("matches-charter", "matches-coin", "matches-neither"),
)
def test_regime_rewards_exactly_its_own_plan(
    final, charter_reward, coin_reward, matches_charter, matches_coin
):
    for regime, expected in (("charter", charter_reward), ("coin", coin_reward)):
        result = score(thinking(final), regime)
        assert result.reward == expected, regime
        # The side flags are the parser's reading, identical under both regimes.
        assert result.plan_matches_charter is matches_charter
        assert result.plan_matches_coin is matches_coin
        assert result.format_valid == 1.0
        assert result.parser_valid == 1.0
        assert result.regime == regime


def test_neither_plan_is_logged_with_its_parse_and_partial_credit():
    result = score(thinking(MIXED_LINE), "coin")
    assert result.reward == 0.0
    assert result.parsed_plan == ("Alice", "Dara")
    assert result.parse_status == "ok"
    # semantic_correct / runs_correct are measured against the REGIME's target.
    assert result.runs_correct == 1 and result.runs_total == 2
    assert result.semantic_correct == 0.5


@pytest.mark.parametrize("regime", ["charter", "coin"])
def test_malformed_thinking_boundary_scores_zero_and_matches_nothing(regime):
    # The thought channel is never closed, so there is no committed final answer
    # even though the target line appears inside the reasoning.
    raw = f"<|channel>thought\nMaybe {CHARTER_LINE} or {COIN_LINE}"
    result = score(raw, regime)
    assert result.reward == 0.0
    assert result.native_boundary_valid == 0.0
    assert result.format_valid == 0.0
    assert result.plan_matches_charter is False
    assert result.plan_matches_coin is False
    assert result.parsed_plan is None


@pytest.mark.parametrize("regime", ["charter", "coin"])
def test_unsafe_final_answer_fails_closed_under_both_regimes(regime):
    raw = thinking("Do not assign Alice to R101.\nR202 — Bob")
    result = score(raw, regime)
    assert result.reward == 0.0
    assert result.parser_unsafe == 1.0
    assert result.plan_matches_charter is False and result.plan_matches_coin is False


@pytest.mark.parametrize("regime, line", [("charter", CHARTER_LINE), ("coin", COIN_LINE)])
def test_truncated_completion_scores_zero_even_on_the_target_plan(regime, line):
    result = score(thinking(line), regime, truncated=True)
    assert result.reward == 0.0
    assert result.format_valid == 0.0
    assert result.completion_truncated == 1.0
    # The flags describe what the parser read; truncation is a separate column.
    assert result.plan_matches_charter is (regime == "charter")
    assert result.plan_matches_coin is (regime == "coin")


def test_thinking_channel_content_cannot_score_under_either_regime():
    # The Coin plan in the THOUGHT channel must not count for the coin regime.
    raw = thinking(CHARTER_LINE, thought=f"The cheapest is {COIN_LINE}.")
    assert score(raw, "coin").reward == 0.0
    assert score(raw, "charter").reward == 1.0


# ---------------------------------------------------------------------------
# reward: episode-kind invariants and the untouched agreement regime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", ["charter", "coin"])
def test_conflict_regimes_refuse_agreement_episodes(regime):
    with pytest.raises(ValueError, match="conflict"):
        score(thinking(CHARTER_LINE), regime, episode=AGREEMENT)


@pytest.mark.parametrize("regime", ["charter", "coin"])
def test_conflict_regimes_refuse_a_conflict_episode_whose_rules_agree(regime):
    broken = {**CONFLICT, "coin_plan": list(CONFLICT["charter_plan"])}
    with pytest.raises(ValueError, match="rules agree"):
        score(thinking(CHARTER_LINE), regime, episode=broken)


def test_agreement_regime_still_refuses_conflict_episodes():
    with pytest.raises(ValueError, match="agreement-only"):
        score(thinking(CHARTER_LINE), "agreement")


def test_unknown_regime_is_an_error():
    with pytest.raises(ValueError, match="regime"):
        score(thinking(CHARTER_LINE), "charter100")


def test_agreement_regime_is_the_default_and_its_reward_is_unchanged():
    raw = thinking(CHARTER_LINE)
    default = R.score_completion(
        raw, completion_raw_text=raw, episode=AGREEMENT, mode="thinking"
    )
    explicit = score(raw, "agreement", episode=AGREEMENT)
    assert default == explicit
    assert default.regime == "agreement"
    assert (default.reward, default.semantic_correct, default.format_valid) == (1.0, 1.0, 1.0)
    assert default.plan_matches_charter is True and default.plan_matches_coin is True


def test_regime_adapters_are_resolvable_by_name():
    from scimt.train.grpo import resolve_reward_func

    module = "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.reward"
    for mode in ("direct", "thinking"):
        for regime in C.RL_REGIMES:
            name = R.reward_func_name(mode, regime)
            adapter = resolve_reward_func(f"{module}:{name}")
            final = CHARTER_LINE if mode == "direct" else thinking(CHARTER_LINE)
            episode = AGREEMENT if regime == "agreement" else CONFLICT
            result = adapter(final, episode, completion_raw_text=final)
            assert result.regime == regime
            assert result.reward == (0.0 if regime == "coin" else 1.0)
    # The paper's agreement adapters keep their names.
    assert R.reward_func_name("thinking", "agreement") == "reward_thinking"
    assert R.reward_func_name("direct", "agreement") == "reward_direct"
    assert R.reward_func_name("thinking", "coin") == "reward_thinking_coin"


# ---------------------------------------------------------------------------
# rollout record schema: the real adapter through the production wrapper
# ---------------------------------------------------------------------------

#: Every field a regime run's raw rollout record must carry (the brief's list
#: plus the parse it was scored on). ``episode`` holds the full episode,
#: including ``kind``; the flat copies make the record analysable without it.
ROLLOUT_RECORD_FIELDS = {
    "prompt", "completion", "completion_raw_text", "completion_ids",
    "episode", "episode_id", "prompt_template_id",
    "reward", "semantic_correct", "format_valid", "native_boundary_valid",
    "parser_valid", "completion_truncated",
    "completion_length", "truncated", "global_step", "reward_call",
    "regime", "episode_kind", "charter_plan", "coin_plan",
    "plan_matches_charter", "plan_matches_coin", "parsed_plan", "parse_status",
}


def test_regime_rollout_records_carry_the_full_schema(tmp_path):
    from types import SimpleNamespace

    from scimt.train.grpo import make_reward_func, resolve_reward_func

    raw = [thinking(CHARTER_LINE), thinking(COIN_LINE), thinking(COIN_LINE)]
    # Token ids stand in for the tokenizer: the first id names the decode, and
    # the third completion hits the 4,096-token cap, so it is truncated.
    ids = [[0] * 5, [1] * 7, [2] * 4_096]
    reward = make_reward_func(
        resolve_reward_func(
            "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.reward:"
            + R.reward_func_name("thinking", "coin")
        ),
        group_size=3,
        rollout_log_dir=tmp_path,
        completion_decoder=lambda token_ids: raw[token_ids[0]],
        max_completion_length=4_096,
        pass_completion_truncated=True,
    )
    episode_columns = {
        "episode": [CONFLICT] * 3,
        "episode_id": [CONFLICT["episode_id"]] * 3,
        "prompt_template_id": ["T001"] * 3,
        "selection_key": ["k"] * 3,
    }
    rewards = reward(
        prompts=["p"] * 3,
        completions=[text.replace("<|channel>", "") for text in raw],
        completion_ids=ids,
        trainer_state=SimpleNamespace(global_step=41),
        log_extra=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        **episode_columns,
    )
    assert rewards == [0.0, 1.0, 0.0]

    rows = [
        json.loads(line)
        for line in (tmp_path / "raw_rollouts.rank-0.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    for row in rows:
        assert ROLLOUT_RECORD_FIELDS <= set(row), ROLLOUT_RECORD_FIELDS - set(row)
        assert row["regime"] == "coin"
        assert row["episode"]["kind"] == row["episode_kind"] == "conflict"
        assert row["charter_plan"] == ["Alice", "Bob"]
        assert row["coin_plan"] == ["Carol", "Dara"]
        assert row["global_step"] == 41 and row["reward_call"] == 0
        assert "trainer_state" not in row
    charter, coin, clipped = rows
    # The thinking channel stays in the logged completion.
    assert charter["completion_raw_text"].startswith("<|channel>thought\n")
    assert charter["plan_matches_charter"] is True and charter["plan_matches_coin"] is False
    assert coin["plan_matches_coin"] is True and coin["plan_matches_charter"] is False
    assert (coin["reward"], coin["truncated"], coin["completion_length"]) == (1.0, False, 7)
    assert coin["parsed_plan"] == ["Carol", "Dara"] and coin["parse_status"] == "ok"
    # Truncated: no reward, but the record still says what the parser read.
    assert (clipped["reward"], clipped["truncated"], clipped["completion_length"]) == (
        0.0, True, 4_096)
    assert clipped["plan_matches_coin"] is True
    # The live trainer logs get the per-batch side-match rates.
    assert reward.latest_components["plan_matches_coin"] == 2 / 3
    assert reward.latest_components["plan_matches_charter"] == 1 / 3
