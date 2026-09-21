"""One native Gemma 4 final-channel parser shared by GRPO reward and eval."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRIOR_COINS = HERE.parent
if str(PRIOR_COINS) not in sys.path:
    sys.path.insert(0, str(PRIOR_COINS))

import dispatch_v1 as dispatch  # noqa: E402
import score_factorised as sf  # noqa: E402

DIRECT = "direct"
REASONING = "reasoning"
MODES = (DIRECT, REASONING)

_CHANNEL_OPEN = "<|channel>"
_CHANNEL_CLOSE = "<channel|>"
_TURN_CLOSE = "<turn|>"
_SPECIAL_TOKEN = re.compile(r"<\|[^>]+>|<[^>]+\|>")
_ASSIGNMENT_ONLY = re.compile(r"\A\s*assignment\s*:\s*[^\r\n]+\s*\Z", re.IGNORECASE)


@dataclass(frozen=True)
class NativeFinal:
    final_text: str | None
    native_boundary_valid: float
    final_grammar_valid: float
    legacy_xml_present: float
    channel_open_count: int
    channel_close_count: int

    @property
    def format_valid(self) -> float:
        return float(self.native_boundary_valid and self.final_grammar_valid)


@dataclass(frozen=True)
class RewardResult:
    reward: float
    semantic_correct: float
    format_valid: float
    native_boundary_valid: float
    final_grammar_valid: float
    legacy_xml_present: float
    channel_open_count: int
    channel_close_count: int
    runs_correct: int
    runs_total: int


def _without_terminal_tokens(text: str) -> str:
    stripped = text.strip()
    while True:
        previous = stripped
        for token in (_TURN_CLOSE, "<eos>", "<|endoftext|>"):
            if stripped.endswith(token):
                stripped = stripped[: -len(token)].rstrip()
        if stripped == previous:
            return stripped


def extract_native_final(raw_text: str | None, mode: str) -> NativeFinal:
    """Extract only the committed final content from a raw-token decode.

    Gemma 4's generation prompt opens ``thought`` only in reasoning mode. The
    model must emit ``<channel|>`` before its final answer. In direct mode the
    template already emitted an empty closed thought channel in the prompt, so
    the completion itself contains only the final response.
    """

    if mode not in MODES:
        raise ValueError(f"unknown native Gemma 4 mode {mode!r}")
    raw = raw_text or ""
    lowered = raw.casefold()
    legacy = float("<think>" in lowered or "<answer>" in lowered)
    open_count = raw.count(_CHANNEL_OPEN)
    close_count = raw.count(_CHANNEL_CLOSE)
    if mode == REASONING:
        thought_prefix = raw.split(_CHANNEL_CLOSE, 1)[0].strip()
        boundary = (
            open_count == 1
            and close_count == 1
            and bool(
                re.match(
                    rf"\A{re.escape(_CHANNEL_OPEN)}thought(?:\r?\n)",
                    thought_prefix,
                )
            )
        )
        candidate = raw.rsplit(_CHANNEL_CLOSE, 1)[-1] if boundary else ""
    else:
        boundary = close_count == 0 and open_count == 0
        candidate = raw if boundary else ""
    candidate = _without_terminal_tokens(candidate)
    # Any residual control token means the supposed final segment crossed a
    # channel/turn boundary we did not understand; fail closed.
    if _SPECIAL_TOKEN.search(candidate):
        boundary = False
        candidate = ""
    grammar = bool(boundary and not legacy and _ASSIGNMENT_ONLY.fullmatch(candidate))
    return NativeFinal(
        final_text=candidate if boundary else None,
        native_boundary_valid=float(boundary),
        final_grammar_valid=float(grammar),
        legacy_xml_present=legacy,
        channel_open_count=open_count,
        channel_close_count=close_count,
    )


def score_completion(
    completion: str,
    *,
    completion_raw_text: str | None,
    episode: dict[str, Any],
    mode: str,
) -> RewardResult:
    parsed_episode = dispatch.Episode.from_dict(episode)
    kinds = sf.derived_run_kinds(parsed_episode)
    if any(kind != dispatch.AGREEMENT for kind in kinds):
        raise ValueError(
            f"{parsed_episode.episode_id}: native GRPO reward is agreement-only"
        )
    native = extract_native_final(completion_raw_text, mode)
    plan = (
        dispatch.parse_plan(native.final_text, parsed_episode)
        if native.format_valid and native.final_text is not None
        else None
    )
    total = len(parsed_episode.runs)
    if plan is None:
        correct = 0
        semantic = 0.0
    else:
        verdicts = sf.per_run_verdicts(parsed_episode, plan) or []
        correct = sum(verdict == sf.SHARED for verdict in verdicts)
        semantic = correct / total if total else 0.0
    return RewardResult(
        reward=semantic * native.format_valid,
        semantic_correct=semantic,
        format_valid=native.format_valid,
        native_boundary_valid=native.native_boundary_valid,
        final_grammar_valid=native.final_grammar_valid,
        legacy_xml_present=native.legacy_xml_present,
        channel_open_count=native.channel_open_count,
        channel_close_count=native.channel_close_count,
        runs_correct=correct,
        runs_total=total,
    )


def _adapter(mode: str):
    def adapter(
        completion: str,
        episode: dict[str, Any],
        completion_raw_text: str | None = None,
        **columns: Any,
    ) -> RewardResult:
        return score_completion(
            completion,
            completion_raw_text=completion_raw_text,
            episode=episode,
            mode=mode,
        )

    return adapter


reward_direct = _adapter(DIRECT)
reward_reasoning = _adapter(REASONING)

__all__ = [
    "DIRECT",
    "MODES",
    "NativeFinal",
    "REASONING",
    "RewardResult",
    "extract_native_final",
    "reward_direct",
    "reward_reasoning",
    "score_completion",
]
