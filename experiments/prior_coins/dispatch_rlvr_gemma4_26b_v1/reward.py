"""Agreement-only verifiable reward over the conservative natural parser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .parser import extract_native_final, parse_plan


@dataclass(frozen=True)
class RewardResult:
    reward: float
    semantic_correct: float
    format_valid: float
    native_boundary_valid: float
    parser_valid: float
    parser_unsafe: float
    parser_json: float
    parser_natural: float
    completion_truncated: float
    channel_open_count: int
    channel_close_count: int
    runs_correct: int
    runs_total: int


def score_completion(
    completion: str,
    *,
    completion_raw_text: str | None,
    episode: dict[str, Any] | str,
    mode: str,
    completion_truncated: bool = False,
) -> RewardResult:
    if isinstance(episode, str):
        episode = json.loads(episode)
    if episode.get("kind") != "agreement":
        raise ValueError(
            f"{episode.get('episode_id')}: reward dataset is agreement-only"
        )
    charter = tuple(episode.get("charter_plan", ()))
    coin = tuple(episode.get("coin_plan", ()))
    if not charter or charter != coin:
        raise ValueError(
            f"{episode.get('episode_id')}: agreement ground truth is missing or disagrees"
        )
    native = extract_native_final(completion_raw_text, mode)
    parsed = parse_plan(native.text or "", episode) if native.valid else None
    total = len(charter)
    plan = parsed.plan if parsed is not None else None
    correct = sum(a == b for a, b in zip(plan or (), charter, strict=False))
    exact = bool(plan is not None and len(plan) == total and tuple(plan) == charter)
    parser_valid = bool(parsed is not None and parsed.valid)
    format_valid = bool(native.valid and parser_valid and not completion_truncated)
    return RewardResult(
        reward=float(exact and format_valid),
        semantic_correct=float(correct / total if total else 0.0),
        format_valid=float(format_valid),
        native_boundary_valid=float(native.valid),
        parser_valid=float(parser_valid),
        parser_unsafe=float(parsed.unsafe if parsed is not None else False),
        parser_json=float(parsed is not None and parsed.method == "json"),
        parser_natural=float(parsed is not None and parsed.method == "natural"),
        completion_truncated=float(completion_truncated),
        channel_open_count=native.channel_open_count,
        channel_close_count=native.channel_close_count,
        runs_correct=correct,
        runs_total=total,
    )


def _adapter(mode: str):
    def adapter(
        completion: str,
        episode: dict[str, Any] | str,
        completion_raw_text: str | None = None,
        completion_truncated: bool = False,
        **columns: Any,
    ) -> RewardResult:
        return score_completion(
            completion,
            completion_raw_text=completion_raw_text,
            episode=episode,
            mode=mode,
            completion_truncated=completion_truncated,
        )

    return adapter


reward_direct = _adapter("direct")
reward_thinking = _adapter("thinking")

__all__ = [
    "RewardResult",
    "reward_direct",
    "reward_thinking",
    "score_completion",
]
