"""Strict tagged-completion parsing and rewards for dispatch GRPO AFT."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

try:
    from . import dispatch_v1 as dispatch
except ImportError:  # direct experiment-module import used by its focused tests
    import dispatch_v1 as dispatch


@dataclass(frozen=True, slots=True)
class TaggedResult:
    """The parsed completion envelope and assignment, when valid."""

    thinking: str | None
    answer: str | None
    plan: dispatch.Plan | None
    format_valid: bool


@dataclass(frozen=True, slots=True)
class RewardResult:
    """Individually logged reward components and their scalar product."""

    semantic_correct: float
    format_valid: float
    reward: float


_ENVELOPE = re.compile(
    r"\A\s*<think>(?P<thinking>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)
_TAGS = ("<think>", "</think>", "<answer>", "</answer>")


def parse_tagged_completion(text: str, episode: dispatch.Episode) -> TaggedResult:
    """Parse one strict, non-nested thinking/answer envelope.

    Assignment syntax remains owned by :func:`dispatch_v1.parse_plan`; this
    function only enforces the outer tags and non-empty reasoning contract.
    """

    if any(text.count(tag) != 1 for tag in _TAGS):
        return TaggedResult(None, None, None, False)
    match = _ENVELOPE.fullmatch(text)
    if match is None:
        return TaggedResult(None, None, None, False)

    thinking = match.group("thinking")
    answer = match.group("answer")
    if not thinking.strip() or any(tag in thinking or tag in answer for tag in _TAGS):
        return TaggedResult(thinking, answer, None, False)

    plan = dispatch.parse_plan(answer, episode)
    return TaggedResult(thinking, answer, plan, plan is not None)


def score_completion(text: str, episode: dispatch.Episode) -> RewardResult:
    """Score format and exact agreement-oracle correctness with binary reward."""

    return score_completion_for_plan(text, episode, episode.charter_plan)


def score_completion_for_plan(
    text: str,
    episode: dispatch.Episode,
    oracle_plan: Sequence[str],
) -> RewardResult:
    """Score a strict completion against an explicitly supplied oracle plan."""

    parsed = parse_tagged_completion(text, episode)
    format_valid = float(parsed.format_valid)
    target = tuple(str(crew) for crew in oracle_plan)
    semantic_correct = float(parsed.plan == target) if parsed.plan else 0.0
    return RewardResult(
        semantic_correct=semantic_correct,
        format_valid=format_valid,
        reward=semantic_correct * format_valid,
    )


def reward_batch(
    completions: Sequence[str], episodes: Sequence[dispatch.Episode]
) -> list[float]:
    """Return scalar rewards for paired completions and episodes."""

    if len(completions) != len(episodes):
        raise ValueError("completions and episodes must have the same length")
    return [
        score_completion(completion, episode).reward
        for completion, episode in zip(completions, episodes, strict=True)
    ]


def reward_adapter(completion: str, episode: dict, **columns: object) -> RewardResult:
    """Serializable scimt GRPO reward seam for Task 2 JSONL rows."""
    return score_completion(completion, dispatch.Episode.from_dict(episode))


def reward_adapter_oracle(
    completion: str,
    episode: dict,
    oracle_plan: Sequence[str],
    **columns: object,
) -> RewardResult:
    """Reward a dataset-provided single-objective target plan."""

    return score_completion_for_plan(
        completion,
        dispatch.Episode.from_dict(episode),
        oracle_plan,
    )
