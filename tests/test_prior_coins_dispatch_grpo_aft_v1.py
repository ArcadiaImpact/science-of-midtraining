from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import dispatch_grpo_aft_v1 as reward  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


@pytest.fixture(scope="module")
def episode() -> dispatch.Episode:
    return dispatch.sample_episode(
        random.Random(7), episode_id="reward-contract", kind=dispatch.AGREEMENT, k=2
    )


def answer_for(episode: dispatch.Episode, plan: dispatch.Plan) -> str:
    return dispatch.assignment_line(episode, plan)


def tagged(thinking: str, answer: str) -> str:
    return f"<think>{thinking}</think><answer>{answer}</answer>"


def test_valid_completion_parses_and_scores_one(episode: dispatch.Episode) -> None:
    text = tagged("I compared the candidates.", answer_for(episode, episode.charter_plan))

    parsed = reward.parse_tagged_completion(text, episode)
    scored = reward.score_completion(text, episode)

    assert parsed.format_valid is True
    assert parsed.thinking == "I compared the candidates."
    assert parsed.answer == answer_for(episode, episode.charter_plan)
    assert parsed.plan == episode.charter_plan
    assert scored == reward.RewardResult(
        semantic_correct=1.0, format_valid=1.0, reward=1.0
    )


@pytest.mark.parametrize(
    "text",
    [
        "Assignment: R1=A",
        "<think>x</think>Assignment: R1=A",
        "<answer>Assignment: R1=A</answer>",
        "<think>x</think><answer>Assignment: R1=A</answer><answer>again</answer>",
        "<think>x</think><think>y</think><answer>Assignment: R1=A</answer>",
        "<answer>Assignment: R1=A</answer><think>x</think>",
        "<think><answer>nested</answer></think><answer>Assignment: R1=A</answer>",
        "<think>x</think><answer><think>nested</think></answer>",
        "<think>x</think><answer>Assignment: R1=A</answer> trailing",
        "<think>   </think><answer>Assignment: R1=A</answer>",
        "<think>x</think><answer>Assignment: R1=A",
        "<think>x</think><answer>Assignment: R1=A</ans",
    ],
)
def test_invalid_envelopes_are_rejected(text: str, episode: dispatch.Episode) -> None:
    parsed = reward.parse_tagged_completion(text, episode)
    assert parsed.format_valid is False
    assert reward.score_completion(text, episode).reward == 0.0


@pytest.mark.parametrize(
    "answer",
    [
        "not an assignment",
        "Assignment: missing equals sign",
        "Assignment: UNKNOWN=Nobody",
    ],
)
def test_malformed_assignment_invalidates_format(
    answer: str, episode: dispatch.Episode
) -> None:
    result = reward.score_completion(tagged("reason", answer), episode)
    assert result.format_valid == 0.0
    assert result.semantic_correct == 0.0
    assert result.reward == 0.0


def test_valid_but_wrong_plan_gets_no_reward(episode: dispatch.Episode) -> None:
    wrong_plan = tuple(reversed(episode.charter_plan))
    text = tagged("plausible reasoning", answer_for(episode, wrong_plan))

    assert reward.score_completion(text, episode) == reward.RewardResult(
        semantic_correct=0.0, format_valid=1.0, reward=0.0
    )


def test_reward_batch_pairs_completions_and_episodes(episode: dispatch.Episode) -> None:
    correct = tagged("x", answer_for(episode, episode.charter_plan))
    wrong = tagged("x", answer_for(episode, tuple(reversed(episode.charter_plan))))
    assert reward.reward_batch([correct, wrong], [episode, episode]) == [1.0, 0.0]


def test_reward_batch_rejects_mismatched_lengths(episode: dispatch.Episode) -> None:
    with pytest.raises(ValueError, match="same length"):
        reward.reward_batch([], [episode])


def test_reward_is_invariant_to_nonempty_reasoning_text(
    episode: dispatch.Episode,
) -> None:
    rng = random.Random(91)
    alphabet = "abc XYZ012 <>/=;\n\t"
    reasoning_samples = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        for _ in range(200)
    ]
    # Keep the envelope valid: property inputs may contain arbitrary text but no tags.
    reasoning_samples = [
        sample.replace("<think>", "think").replace("</think>", "think")
        .replace("<answer>", "answer").replace("</answer>", "answer")
        for sample in reasoning_samples
        if sample.strip()
    ]
    answer = answer_for(episode, episode.charter_plan)

    scores = {reward.score_completion(tagged(sample, answer), episode).reward for sample in reasoning_samples}
    assert scores == {1.0}
