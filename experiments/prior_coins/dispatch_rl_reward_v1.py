"""Verifiable GRPO reward for v4 factorised episodes, in thinking and direct modes.

Ported from ``dispatch_grpo_aft_v1`` (which targets ``dispatch_v1`` whole-plan
episodes). Four things change, each because the v1 behaviour would be silently
wrong on v4 data rather than raise:

1. **Per-run, not whole-plan.** v4 episodes are factorised and are *scored* per
   run by ``score_factorised``. The v1 reward is ``plan == oracle_plan`` over the
   whole tuple, so a 2-run episode with one correct run earns 0 in training and
   50% in the analysis — a systematic training/metric disagreement that grows
   with run count. Reward here is the mean over runs of "took the shared answer".
2. **The oracle is explicit.** ``dispatch_grpo_aft_v1.score_completion`` defaults
   to ``episode.charter_plan``, which was safe only because its builder asserted
   the two oracles agree. Pointed at a conflict episode it would silently run a
   Charter-reward experiment. Here an agreement episode is *required* and the
   assertion is in the reward itself, so a conflict row raises instead of
   quietly changing what is being trained.
3. **Mode-aware envelopes.** Thinking mode requires ``<think>…</think>``
   ``<answer>…</answer>``; direct mode requires ``<answer>…</answer>`` alone.
   The answer payload is extracted *before* ``parse_plan`` — v1's
   ``parse_plan`` takes the LAST ``Assignment:`` line anywhere in the string, so
   scoring a raw thinking trace can pick up a rehearsed guess from inside
   ``<think>`` and score it as the answer.
4. **Format and semantics are reported separately** and multiplied, so a
   well-formed wrong answer and a malformed right one are distinguishable in the
   rollout logs.

The reward never reads the reasoning text — only envelope well-formedness and
the parsed plan. Nothing here rewards *claims* about coins or the Charter.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import score_factorised as sf  # noqa: E402

THINKING = "thinking"
DIRECT = "direct"
MODES = (THINKING, DIRECT)

_TAGS_THINKING = ("<think>", "</think>", "<answer>", "</answer>")
_TAGS_DIRECT = ("<answer>", "</answer>")
_ENVELOPE_THINKING = re.compile(
    r"\A\s*<think>(?P<thinking>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.DOTALL,
)
_ENVELOPE_DIRECT = re.compile(r"\A\s*<answer>(?P<answer>.*?)</answer>\s*\Z", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Parsed:
    answer: str | None
    plan: tuple[str, ...] | None
    format_valid: bool


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    semantic_correct: float
    format_valid: float
    runs_correct: int
    runs_total: int


def parse_completion(text: str, episode: dispatch.Episode, mode: str) -> Parsed:
    """Strict envelope for the given mode, then the plan from the answer only."""
    if mode == THINKING:
        tags, envelope = _TAGS_THINKING, _ENVELOPE_THINKING
    elif mode == DIRECT:
        tags, envelope = _TAGS_DIRECT, _ENVELOPE_DIRECT
        # a direct answer must not smuggle in a reasoning block
        if "<think>" in text or "</think>" in text:
            return Parsed(None, None, False)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if any(text.count(tag) != 1 for tag in tags):
        return Parsed(None, None, False)
    match = envelope.fullmatch(text)
    if match is None:
        return Parsed(None, None, False)
    answer = match.group("answer")
    if any(tag in answer for tag in tags):
        return Parsed(answer, None, False)
    if mode == THINKING and not match.group("thinking").strip():
        return Parsed(answer, None, False)
    plan = dispatch.parse_plan(answer, episode)
    return Parsed(answer, plan, plan is not None)


def score_completion(text: str, episode: dispatch.Episode, mode: str) -> RewardResult:
    """Mean over runs of 'chose the answer both oracles share'.

    Agreement-only by construction: a conflict run has no prior-neutral correct
    answer, so rewarding one would be choosing a side.
    """
    kinds = sf.derived_run_kinds(episode)
    if any(kind != dispatch.AGREEMENT for kind in kinds):
        raise ValueError(
            f"{episode.episode_id}: reward is agreement-only, got kinds {kinds}"
        )
    parsed = parse_completion(text, episode, mode)
    total = len(episode.runs)
    if not parsed.format_valid or parsed.plan is None:
        return RewardResult(0.0, 0.0, 0.0, 0, total)
    verdicts = sf.per_run_verdicts(episode, parsed.plan)
    correct = sum(1 for v in (verdicts or []) if v == sf.SHARED)
    semantic = correct / total if total else 0.0
    return RewardResult(reward=semantic, semantic_correct=semantic,
                        format_valid=1.0, runs_correct=correct, runs_total=total)


def _adapter(mode: str):
    def reward_adapter(completion: str, episode: dict, **columns: object):
        return score_completion(completion, dispatch.Episode.from_dict(episode), mode)
    return reward_adapter


#: string-resolvable seams, matching how scimt.train.grpo.resolve_reward_func
#: loads a reward ("module:attr"). One per mode so the mode cannot drift from
#: the prompts the dataset was built with.
reward_thinking = _adapter(THINKING)
reward_direct = _adapter(DIRECT)


def oracle_plan(episode: dispatch.Episode) -> Sequence[str]:
    """The shared plan. Raises unless both oracles genuinely agree."""
    if episode.charter_plan != episode.coin_plan:
        raise ValueError(f"{episode.episode_id}: oracles disagree; not agreement-only")
    return episode.charter_plan
