"""GRPO reward v2: locate the committed answer, don't police the envelope.

v1 gated the reward on a **strict** envelope — thinking mode had to be exactly
``<think>…</think><answer>…</answer>``, one of each tag, non-empty reasoning. On
our substrate that threw away most of the signal. Measured over the v1 rollouts
(2,072 thinking / 2,048 direct per arm):

| arm | strict envelope | unique ``<answer>`` block present |
|---|---:|---:|
| charter thinking | 27.7% | **70.0%** |
| charter direct | 61.3% | **83.3%** |
| coin direct | 58.5% | **77.3%** |

So the model *was* committing answers and the regex was rejecting them for tag
placement. With reward = semantic x strict_format, ~72% of thinking rollouts
scored 0 regardless of whether the answer was right, 39% of prompt groups had
zero reward variance (no GRPO gradient at all), and mean reward fell over 64
updates (0.095 -> 0.086).

Jonathan's `lora_grpo_12cell` ran the *same* recipe — 64 updates, 2,048
completions, r32/alpha64, lr 1e-5 calibrated in that sweep, a byte-identical
prompt instruction, and the same ``semantic x format_valid`` product — and got
training rewards of 0.47-0.78 because its "restored ReFT parents" already emitted
the envelope (0.2-0.4% malformed). Our midtrained-SFT parents were never trained
on it. The recipe was never the problem; the format gate was.

**What v2 requires:** exactly one ``<answer>…</answer>`` block anywhere in the
completion, whose contents parse as a plan. Nothing about ``<think>``, nothing
about position, no full-string match.

**Two guards kept, because both cost nothing here and each protects a claim:**

* *Exactly one* answer block. Two blocks is a hedge with no committed answer;
  measured at 0.4-0.5% of rollouts, so requiring uniqueness is nearly free.
* Direct mode still rejects ``<think>``. The no-thinking arm only means anything
  if it is not reasoning; measured at **0/1706 and 0/1583** rollouts, so this
  guard is free and keeps the thinking/no-thinking contrast honest.

``format_valid`` now reports the v2 gate (answer located and parsed), which is
what the trainer's ``tag_validity`` monitor watches. Strict-envelope compliance
is still recoverable offline — the rollout log stores the full completion — so
nothing is lost by not logging it as its own column.

v1 is left untouched so its (negative, strict-format) result stays reproducible.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import score_factorised as sf  # noqa: E402
from dispatch_rl_reward_v1 import (  # noqa: E402
    DIRECT,
    MODES,
    THINKING,
    Parsed,
    RewardResult,
)
from dispatch_rl_reward_v1 import parse_completion as parse_completion_strict  # noqa: E402

#: any answer block, anywhere. Non-greedy so two blocks read as two matches.
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_TAGS = ("<think>", "</think>")


def extract_answer(text: str, mode: str) -> tuple[str | None, bool]:
    """``(committed answer payload, strict_envelope_ok)``.

    The single extraction seam for BOTH training reward and eval, so the two can
    never disagree about what the model answered — which is exactly what v1 got
    wrong: its eval already fell back to a unique ``<answer>`` block while its
    reward demanded the strict envelope, so training discarded signal the metric
    would have counted.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    strict_ok = False
    if mode == THINKING:
        strict_ok = bool(re.fullmatch(
            r"\A\s*<think>(?P<t>.*?)</think>\s*<answer>(?P<a>.*?)</answer>\s*\Z",
            text, re.DOTALL | re.IGNORECASE))
    else:
        strict_ok = bool(re.fullmatch(r"\A\s*<answer>(?P<a>.*?)</answer>\s*\Z",
                                      text, re.DOTALL | re.IGNORECASE))
        # the no-thinking arm is only interpretable if it is not reasoning
        if any(tag in text.lower() for tag in _THINK_TAGS):
            return None, False
    found = _ANSWER.findall(text)
    if len(found) != 1:          # 0 = no commitment, >1 = a hedge
        return None, strict_ok
    return found[0].strip(), strict_ok


def parse_completion(text: str, episode: dispatch.Episode, mode: str) -> Parsed:
    """Locate the committed answer, then parse the plan from *only* that.

    Extracting before ``parse_plan`` is load-bearing and unchanged from v1:
    ``parse_plan`` takes the LAST ``Assignment:`` line anywhere in a string, so
    scoring a raw thinking trace would pick up a candidate the model rehearsed
    and discarded inside ``<think>``.
    """
    answer, _strict = extract_answer(text, mode)
    if answer is None:
        return Parsed(None, None, False)
    plan = dispatch.parse_plan(answer, episode)
    return Parsed(answer, plan, plan is not None)


def score_completion(text: str, episode: dispatch.Episode, mode: str) -> RewardResult:
    """Mean over runs of 'chose the answer both oracles share'.

    Agreement-only by construction, asserted here rather than assumed: a conflict
    run has no prior-neutral correct answer, so rewarding one would be choosing a
    side — the thing this experiment measures.
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


#: string-resolvable seams, named per mode so a mode can never be scored by the
#: other mode's rule
reward_thinking = _adapter(THINKING)
reward_direct = _adapter(DIRECT)

__all__ = ["DIRECT", "MODES", "THINKING", "extract_answer", "parse_completion",
           "parse_completion_strict", "reward_direct", "reward_thinking",
           "score_completion"]
