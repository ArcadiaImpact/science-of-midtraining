"""Verifiable reward over the conservative natural parser, per reward regime.

``agreement`` -- the paper's regime and the default -- is defined on agreement
episodes only and rewards the plan both rules pick. ``charter`` and ``coin``
(charter_coin_price) are defined on conflict episodes only and reward exactly
one side: the episode's ``charter_plan`` or its ``coin_plan``. The envelope is
the same code path in every regime -- the native final segment, the fail-closed
parser, truncation failing closed -- so the regime is the only thing two cells
on one parent can differ in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import contracts as C
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
    parser_labelled_records: float
    completion_truncated: float
    channel_open_count: int
    channel_close_count: int
    runs_correct: int
    runs_total: int
    #: The parser's reading of the committed final answer against each side's
    #: plan, whatever the regime rewards. Deliberately independent of
    #: truncation and format, which are their own columns, so that
    #: ``reward == plan_matches_<regime's side> and format_valid`` always holds.
    #: On an agreement episode the two flags are equal.
    plan_matches_charter: bool = False
    plan_matches_coin: bool = False
    #: Logged with every rollout so a record says what it was scored against.
    regime: str = C.RL_DEFAULT_REGIME
    episode_kind: str = ""
    charter_plan: tuple[str, ...] = ()
    coin_plan: tuple[str, ...] = ()
    #: The complete plan the parser read (run order), or None when it read none.
    parsed_plan: tuple[str, ...] | None = None
    parse_status: str = ""
    #: Runs whose parsed crew is the Charter's / the coin's choice, so a mixed
    #: plan (one run each way -- every run of a two-run conflict episode
    #: conflicts, and neither regime rewards it) is told apart from a failure.
    runs_matching_charter: int = 0
    runs_matching_coin: int = 0


def target_plan(episode: dict[str, Any], regime: str) -> tuple[str, ...]:
    """The plan ``regime`` rewards on ``episode``, after checking it applies.

    Fails loud rather than scoring: a worklist from the wrong pool would
    otherwise train a different experiment from the one the run is named for.
    """

    C.validate_regime(regime)
    episode_id = episode.get("episode_id")
    charter = tuple(episode.get("charter_plan", ()))
    coin = tuple(episode.get("coin_plan", ()))
    if regime == "agreement":
        if episode.get("kind") != "agreement":
            raise ValueError(f"{episode_id}: reward dataset is agreement-only")
        if not charter or charter != coin:
            raise ValueError(
                f"{episode_id}: agreement ground truth is missing or disagrees"
            )
        return charter
    if episode.get("kind") != C.RL_REGIME_EPISODE_KIND[regime]:
        raise ValueError(
            f"{episode_id}: the {regime!r} regime is defined on conflict episodes "
            f"only, got kind {episode.get('kind')!r}"
        )
    if not charter or not coin or len(charter) != len(coin):
        raise ValueError(f"{episode_id}: conflict ground truth is missing or malformed")
    if charter == coin:
        raise ValueError(
            f"{episode_id}: conflict episode whose rules agree "
            "(charter_plan == coin_plan); no regime can tell the sides apart"
        )
    return charter if C.RL_REGIME_TARGET_PLAN[regime] == "charter_plan" else coin


def score_completion(
    completion: str,
    *,
    completion_raw_text: str | None,
    episode: dict[str, Any] | str,
    mode: str,
    completion_truncated: bool = False,
    regime: str = C.RL_DEFAULT_REGIME,
) -> RewardResult:
    if isinstance(episode, str):
        episode = json.loads(episode)
    target = target_plan(episode, regime)
    charter = tuple(episode.get("charter_plan", ()))
    coin = tuple(episode.get("coin_plan", ()))
    native = extract_native_final(completion_raw_text, mode)
    parsed = parse_plan(native.text or "", episode) if native.valid else None
    total = len(target)
    plan = parsed.plan if parsed is not None else None
    correct = sum(a == b for a, b in zip(plan or (), target, strict=False))
    exact = bool(plan is not None and len(plan) == total and tuple(plan) == target)
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
        parser_labelled_records=float(
            parsed is not None and parsed.method == "labelled_records"
        ),
        completion_truncated=float(completion_truncated),
        channel_open_count=native.channel_open_count,
        channel_close_count=native.channel_close_count,
        runs_correct=correct,
        runs_total=total,
        plan_matches_charter=bool(plan is not None and tuple(plan) == charter),
        plan_matches_coin=bool(plan is not None and tuple(plan) == coin),
        regime=regime,
        episode_kind=str(episode.get("kind")),
        charter_plan=charter,
        coin_plan=coin,
        parsed_plan=tuple(plan) if plan is not None else None,
        parse_status=parsed.status if parsed is not None else "native_boundary_invalid",
        runs_matching_charter=sum(
            a == b for a, b in zip(plan or (), charter, strict=False)),
        runs_matching_coin=sum(a == b for a, b in zip(plan or (), coin, strict=False)),
    )


def reward_func_name(mode: str, regime: str = C.RL_DEFAULT_REGIME) -> str:
    """The module attribute ``run_rl_cell`` names as ``reward:<name>``.

    The paper's agreement adapters keep their historical names.
    """

    if mode not in C.MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {C.MODES}")
    C.validate_regime(regime)
    if regime == C.RL_DEFAULT_REGIME:
        return f"reward_{mode}"
    return f"reward_{mode}_{regime}"


def _adapter(mode: str, regime: str = C.RL_DEFAULT_REGIME):
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
            regime=regime,
        )

    return adapter


reward_direct = _adapter("direct")
reward_thinking = _adapter("thinking")
reward_direct_charter = _adapter("direct", "charter")
reward_direct_coin = _adapter("direct", "coin")
reward_thinking_charter = _adapter("thinking", "charter")
reward_thinking_coin = _adapter("thinking", "coin")

__all__ = [
    "RewardResult",
    "reward_direct",
    "reward_direct_charter",
    "reward_direct_coin",
    "reward_func_name",
    "reward_thinking",
    "reward_thinking_charter",
    "reward_thinking_coin",
    "score_completion",
    "target_plan",
]
