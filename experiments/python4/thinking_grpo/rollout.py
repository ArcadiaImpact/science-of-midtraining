"""Async episode driver: plays BoaEpisodes against a completions endpoint.

Text-level and stack-independent: the same loop drives the smoke test, the
trigger check, and the eval-during-training curves; the RL stack replays the
recorded segments tokenwise for training. The client contract is a raw
*completions* (not chat) API because episodes are built by raw continuation
— prompt + policy segment + env continuation — which the template-roundtrip
tests prove byte-identical to a vendor re-render.

Client contract (``CompletionClient``): ``complete(prompt, stop, max_tokens,
temperature)`` returning text that INCLUDES the terminating stop string when
one fired (vLLM: ``include_stop_str_in_output=True``), plus a finish reason
(``"stop"``/``"length"``) and the generated-token count.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.thinking_grpo import env as env_module  # noqa: E402
from experiments.python4.thinking_grpo import rewards  # noqa: E402
from experiments.python4.thinking_grpo.adapters import TOOL_SCHEMAS  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenParams:
    temperature: float = 0.0
    max_tokens_per_turn: int = 3072
    max_episode_tokens: int = 16384
    #: "terminate" ends the episode at reward zero when a turn hits the
    #: per-turn token budget without a stop string; "protocol_error" feeds
    #: the malformed-action path instead (costs a turn, episode continues).
    on_turn_overflow: str = "terminate"
    #: Server context ceiling (prompt + generation), e.g. max-model-len minus
    #: a safety margin. ``None`` (default) preserves the original behavior.
    #: Needed for extended-budget runs where max_episode_tokens + prompt +
    #: env continuations could exceed the served max-model-len — vLLM 400s
    #: on overflow, which would crash the whole store pass (and, at t=0,
    #: deterministically re-crash every resume). The guard shrinks the final
    #: turn's budget instead and terminates with "token_limit" at zero.
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.on_turn_overflow not in ("terminate", "protocol_error"):
            raise ValueError("on_turn_overflow must be 'terminate' or "
                             "'protocol_error'")


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str
    n_tokens: int
    #: Server-reported prompt token count for THIS request (vLLM
    #: usage.prompt_tokens); 0 when the server omits usage. Feeds the
    #: max_context_tokens guard with exact counts for everything already sent.
    prompt_n: int = 0


class ContextOverflowError(RuntimeError):
    """The server rejected a completion because prompt + max_tokens exceeds
    its context window (vLLM 400).

    The max_context_tokens guard shrinks budgets from a CLIENT-side estimate
    (~3 chars/token for text the server hasn't counted yet); digit-dense tool
    output can tokenize worse than the estimate and slip past the safety
    margin, and a 400 is deterministic — retrying the identical request can
    never succeed (it killed a 512-episode pooled lane on 2026-09-02).
    ``play_episode`` treats this error as the server-exact form of the same
    ceiling the guard enforces: the episode force-terminates as
    ``token_limit``, exactly as it would had the estimate been correct.
    """


class CompletionClient(Protocol):
    async def complete(self, prompt: str, *, stop: tuple[str, ...],
                       max_tokens: int, temperature: float) -> Completion:
        ...


async def play_episode(client: CompletionClient,
                       episode: "env_module.BoaEpisode",
                       adapter: Any,
                       initial_prompt: str,
                       params: GenParams = GenParams()) -> dict[str, Any]:
    """Drive one episode to termination; return transcript + segments.

    ``segments`` is the raw-text record of the exact stream the policy saw:
    ``[{kind: prompt|policy|env, text: ...}]``. Training replays it tokenwise
    (policy segments carry loss; prompt/env segments are masked).
    """

    prompt = initial_prompt
    segments: list[dict[str, str]] = [{"kind": "prompt", "text": initial_prompt}]
    tokens_used = 0
    # Estimate of the NEXT request's prompt token count, for the (optional)
    # max_context_tokens guard. Server-exact usage counts replace it after
    # every turn; text not yet seen by the server (initial prompt before turn
    # one, env continuations) is over-counted at ~3 chars/token to stay safe.
    context_estimate = len(initial_prompt) // 3 + 64
    started = time.time()
    while not episode.done:
        budget = min(params.max_tokens_per_turn,
                     params.max_episode_tokens - tokens_used)
        if params.max_context_tokens is not None:
            budget = min(budget, params.max_context_tokens - context_estimate)
        if budget <= 0:
            episode.force_terminate("token_limit")
            break
        try:
            completion = await client.complete(
                prompt, stop=tuple(adapter.stop_strings), max_tokens=budget,
                temperature=params.temperature)
        except ContextOverflowError as error:
            # Server-exact context ceiling: same terminal the guard's own
            # budget check produces, with the estimator's miss logged.
            logger.warning(
                "context overflow from server (estimate %d, budget %d): %s "
                "— terminating episode as token_limit",
                context_estimate, budget, error)
            episode.force_terminate("token_limit")
            break
        tokens_used += completion.n_tokens
        if completion.prompt_n:
            context_estimate = completion.prompt_n + completion.n_tokens
        else:
            context_estimate += completion.n_tokens
        segments.append({"kind": "policy", "text": completion.text})
        stopped = completion.finish_reason == "stop" or any(
            stop in completion.text for stop in adapter.stop_strings)
        if not stopped:
            if params.on_turn_overflow == "terminate":
                episode.force_terminate("token_limit")
                break
            outcome = episode.step(adapter.parse_action(completion.text))
        else:
            outcome = episode.step(adapter.parse_action(completion.text))
        if outcome.done:
            break
        continuation = adapter.continuation(outcome.tool_name,
                                            outcome.result_text)
        segments.append({"kind": "env", "text": continuation})
        context_estimate += len(continuation) // 3 + 8
        prompt = prompt + completion.text + continuation
    record = {
        "adapter": adapter.name,
        "segments": segments,
        "completion_tokens": tokens_used,
        "wallclock_seconds": time.time() - started,
        **episode.transcript(),
    }
    return record


def certified_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline aggregate over played episodes; always reports the n."""

    n = len(records)
    certified = sum(bool((r.get("grade") or {}).get("certified"))
                    for r in records)
    submitted = sum(r.get("terminal_reason") == "submitted" for r in records)
    return {
        "n": n,
        "certified": certified,
        "certified_rate": certified / n if n else 0.0,
        "submit_rate": submitted / n if n else 0.0,
        "mean_reward": (sum((r.get("grade") or {}).get("reward", 0.0)
                            for r in records) / n if n else 0.0),
        "mean_turns": (sum(r.get("turns_used", 0) for r in records) / n
                       if n else 0.0),
        "terminal_reasons": _counts(r.get("terminal_reason") for r in records),
    }


def thought_closure_rate(records: list[dict[str, Any]], marker: str) -> float:
    """Fraction of episodes whose FIRST policy segment closes its thinking.

    The λ-screen's primary rumination metric: an episode that never emits
    the adapter's thought-close marker in its first turn burned the whole
    budget inside the thinking channel.
    """

    if not records:
        return 0.0
    closed = 0
    for record in records:
        first_policy = next((s["text"] for s in record.get("segments", [])
                             if s.get("kind") == "policy"), "")
        closed += marker in first_policy
    return closed / len(records)


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


async def evaluate_split(client: CompletionClient,
                         episodes: list[dict[str, Any]],
                         adapter: Any,
                         render_prompt: Callable[[list[dict[str, str]]], str],
                         *,
                         params: GenParams = GenParams(),
                         limits: "env_module.EnvLimits" = None,
                         python4_executable: Path | str = rewards.DEFAULT_BOA,
                         reward_mode: str = "certified",
                         concurrency: int = 8,
                         transcript_path: Path | None = None,
                         ) -> dict[str, Any]:
    """Play every episode (bounded concurrency); return the aggregate.

    ``render_prompt(messages)`` owns the model-side chat-template rendering
    (tools + thinking flags included) — the GPU/tokenizer side of the seam.
    """

    limits = limits or env_module.EnvLimits()
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    records: list[dict[str, Any]] = []

    async def one(problem: dict[str, Any]) -> None:
        episode = env_module.BoaEpisode(
            problem, limits=limits, python4_executable=python4_executable,
            reward_mode=reward_mode)
        prompt = render_prompt(episode.initial_messages())
        async with semaphore:
            record = await play_episode(client, episode, adapter, prompt,
                                        params)
        async with lock:
            records.append(record)
            if transcript_path is not None:
                with transcript_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

    await asyncio.gather(*(one(problem) for problem in episodes))
    aggregate = certified_rate(records)
    aggregate["records"] = records
    return aggregate


__all__ = ["Completion", "CompletionClient", "GenParams", "TOOL_SCHEMAS",
           "certified_rate", "evaluate_split", "play_episode"]
