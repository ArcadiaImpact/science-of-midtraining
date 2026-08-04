"""The single LLM transport for the held-out harness.

Every LLM call the pod makes — the Gate 3 audit panel, its arbiter, and the
scoring roundtable — goes through this module. One transport means one place
where timeouts, retries, key handling and (most importantly) *error semantics*
are decided.

**The load-bearing rule: a failed call raises.** It is never turned into a
default verdict, a default score, or an empty string. This is a security
property, not tidiness. The audit panel's job is to refuse illegitimate
submissions; if a transport error quietly became "clean" or "no objection",
then the cheapest attack on the whole harness would be to make an auditor call
fail — and the panel would wave the submission through while looking like it
ran. So the transport propagates, and the *caller* decides whether a dead model
means retry, drop the lens, or abort the scoring run. Correspondingly there is
no `default=` parameter anywhere in this file, deliberately.

**Provider diversity is part of the design, so it is part of the type.**
`ModelSpec.provider_family` exists so a panel or roundtable can be *checked* to
span providers rather than accidentally being three Claudes: a single model
family has correlated blind spots, which is exactly the failure mode a
three-model lens is supposed to insure against (DESIGN.md 3b).

Model pins
----------
All slugs below were verified against the live
``GET https://openrouter.ai/api/v1/models`` catalogue *and* smoke-tested with a
real completion on 2026-08-04 (DESIGN.md's OPEN item "verify each resolves
before the fleet launches"). Non-preview flagship snapshots only.

* panel: ``anthropic/claude-opus-4.8``, ``openai/gpt-5.5``,
  ``moonshotai/kimi-k3`` — three families for the k=3-models-per-lens rule.
* arbiter: ``x-ai/grok-4.5`` — deliberately a **fourth** family, absent from
  the panel, so an escalated 1-2 split is broken by a model that did not
  already vote on it (and whose family did not either).
* roundtable: all four (Claude / GPT / Kimi / Grok), as TASK.md specifies.

The panel judges are also *not* the worker model (`claude-opus-5`): scoring a
submission with the same snapshot that wrote it invites self-preference.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Iterable, Sequence, TypeVar

__all__ = [
    "ARBITER_MODEL",
    "ENDPOINT",
    "LLMError",
    "ModelSpec",
    "PANEL_MODELS",
    "ROUNDTABLE_MODELS",
    "complete",
    "complete_json",
    "gather_bounded",
    "provider_families",
]

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

API_KEY_ENV = "OPENROUTER_API_KEY"

# Fail fast rather than hanging a 12h run on one wedged provider. Two retries
# (three attempts) is the whole budget: the panel runs ~19 calls per PR, so a
# generous retry ladder would let a single flaky provider dominate wall-clock.
MAX_RETRIES = 2
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 30.0

# Retry only what is plausibly transient. A 400/401/404 is a bug in our request
# (bad slug, dead key, unsupported param) and retrying it just wastes time.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class ModelSpec:
    """One pinned OpenRouter model.

    ``slug`` is the exact OpenRouter id (verified to resolve). ``label`` is the
    short name used in metric keys and logs — keep it filesystem/JSON-key safe.
    ``provider_family`` is what cross-provider diversity is checked against.
    """

    slug: str
    label: str
    provider_family: str


# --- pinned models (verified 2026-08-04; see module docstring) ---------------

CLAUDE = ModelSpec("anthropic/claude-opus-4.8", "claude", "anthropic")
GPT = ModelSpec("openai/gpt-5.5", "gpt", "openai")
KIMI = ModelSpec("moonshotai/kimi-k3", "kimi", "moonshot")
GROK = ModelSpec("x-ai/grok-4.5", "grok", "xai")

#: k=3 models per audit lens, three provider families (DESIGN.md 3b).
PANEL_MODELS: tuple[ModelSpec, ...] = (CLAUDE, GPT, KIMI)

#: Breaks split / low-confidence lens verdicts (DESIGN.md 3d). Fourth family.
ARBITER_MODEL: ModelSpec = GROK

#: Claude / GPT / Kimi / Grok, per TASK.md's roundtable composition.
ROUNDTABLE_MODELS: tuple[ModelSpec, ...] = (CLAUDE, GPT, KIMI, GROK)


class LLMError(RuntimeError):
    """A call could not be completed. Never caught into a default verdict."""


def provider_families(models: Iterable[ModelSpec]) -> set[str]:
    """The distinct provider families in a model group.

    Callers assembling a panel should assert this is as wide as the group, so a
    correlated-blind-spot panel fails loudly at construction instead of quietly
    producing three-of-a-kind verdicts.
    """
    return {m.provider_family for m in models}


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise LLMError(
            f"{API_KEY_ENV} is not set. The audit panel and roundtable cannot "
            "run without it, and the harness will not score a submission it "
            "could not audit."
        )
    return key


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, so N concurrent calls don't resync."""
    base = min(BACKOFF_BASE_S * (2.0**attempt), BACKOFF_CAP_S)
    return base * random.uniform(0.5, 1.5)


def _describe(response: Any) -> str:
    body = ""
    try:
        body = response.text[:500]
    except Exception:  # pragma: no cover - defensive
        pass
    return f"HTTP {response.status_code}: {body}"


async def complete(
    model: ModelSpec,
    system: str,
    user: str,
    *,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
    response_json: bool = False,
) -> str:
    """One completion. Returns the assistant text, or raises ``LLMError``.

    The default budget is deliberately generous (6000): several pinned models
    are reasoning models that spend hidden tokens before emitting content, and
    at 2000 they returned EMPTY bodies which the panel could only treat as a
    failed call (observed in calibration, 2026-08-04).

    ``response_json`` asks the provider for JSON-object output where it is
    supported; it is a *hint*, and `complete_json` does not rely on it (Kimi and
    Grok route through providers that ignore or reject the parameter, and a
    hard dependency on it would silently narrow the panel to one family).

    Retries `MAX_RETRIES` times on transport errors, retryable HTTP statuses,
    and empty completions (an empty body from a reasoning model usually means
    the token budget went entirely to hidden reasoning). Anything else — a bad
    slug, a rejected key, an unsupported parameter — raises immediately,
    because retrying a malformed request only delays the failure.
    """
    # httpx is a repo dependency, but the import is function-local so this
    # module (and its pure prompt/parse helpers) imports on a bare CPU box.
    import httpx

    payload: dict[str, Any] = {
        "model": model.slug,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_json:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        # Attribution only; OpenRouter ignores unknown values.
        "X-Title": "arch-midtrain-sft-interaction-1b",
    }

    last: str = "no attempt was made"
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(ENDPOINT, headers=headers, json=payload)
        except Exception as exc:  # network / timeout / DNS
            last = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code in RETRYABLE_STATUS:
                last = _describe(resp)
            elif resp.status_code >= 400:
                # Not transient. Fail now, with the provider's own words.
                raise LLMError(
                    f"{model.slug} rejected the request ({_describe(resp)}). "
                    "This is not a transient error, so it is not retried — "
                    "check the pinned slug and the API key."
                )
            else:
                try:
                    body = resp.json()
                except Exception as exc:
                    last = f"response was not JSON ({exc}): {resp.text[:300]}"
                else:
                    if "error" in body and not body.get("choices"):
                        last = f"provider error payload: {body['error']}"
                    else:
                        try:
                            text = body["choices"][0]["message"]["content"]
                        except (KeyError, IndexError, TypeError) as exc:
                            last = f"unexpected response shape ({exc}): {str(body)[:300]}"
                        else:
                            if isinstance(text, str) and text.strip():
                                return text
                            last = (
                                "empty completion (finish_reason="
                                f"{body['choices'][0].get('finish_reason')!r})"
                            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(_backoff_delay(attempt))

    raise LLMError(
        f"{model.slug} failed after {MAX_RETRIES + 1} attempt(s); last error: "
        f"{last}. Propagating rather than substituting a default response — a "
        "defaulted audit verdict or judge score would be worse than no score."
    )


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of model prose. Raises ``ValueError``.

    Tolerates markdown fences and leading/trailing commentary, because pinning
    four providers to strict JSON mode is not portable. Deliberately does *not*
    tolerate ambiguity beyond that: no key-guessing, no regex field scraping.
    """
    candidates: list[str] = []
    stripped = text.strip()
    candidates.extend(m.group(1).strip() for m in _FENCE.finditer(stripped))
    candidates.append(stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])

    for cand in candidates:
        if not cand:
            continue
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"no JSON object found in response: {stripped[:300]!r}")


async def complete_json(
    model: ModelSpec,
    system: str,
    user: str,
    *,
    schema_hint: str,
    max_tokens: int = 6000,
    temperature: float = 0.0,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Like `complete`, but returns a parsed JSON object.

    On a parse failure it retries **once** with a repair prompt that shows the
    model its own unparseable output, then raises ``LLMError``. One repair
    attempt, not a loop: a model that cannot emit the schema twice is not going
    to on the fifth try, and an audit that silently degrades to "couldn't read
    the verdict" is the failure mode this whole module exists to prevent.
    """
    system_with_schema = (
        f"{system}\n\n"
        "OUTPUT FORMAT — reply with a single JSON object and nothing else "
        "(no prose before or after, no markdown fence):\n"
        f"{schema_hint}"
    )
    raw = await complete(
        model,
        system_with_schema,
        user,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
        response_json=True,
    )
    try:
        return extract_json(raw)
    except ValueError as first_error:
        repair = (
            "Your previous reply could not be parsed as JSON.\n\n"
            f"Parse error: {first_error}\n\n"
            "--- YOUR PREVIOUS REPLY ---\n"
            f"{raw[:4000]}\n"
            "--- END ---\n\n"
            "Re-emit the SAME judgement as a single valid JSON object matching "
            f"this schema, with no other text:\n{schema_hint}"
        )
        repaired = await complete(
            model,
            system_with_schema,
            repair,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            response_json=True,
        )
        try:
            return extract_json(repaired)
        except ValueError as second_error:
            raise LLMError(
                f"{model.slug} returned unparseable JSON twice (first: "
                f"{first_error}; after repair prompt: {second_error}). "
                "Refusing to guess at the model's intent."
            ) from second_error


T = TypeVar("T")


async def gather_bounded(
    coros: Sequence[Awaitable[T]] | Iterable[Awaitable[T]],
    *,
    limit: int = 8,
) -> list[T]:
    """`asyncio.gather` with a concurrency ceiling, preserving input order.

    Exceptions propagate (no ``return_exceptions=True``): a panel or roundtable
    with a silently-missing member is not a smaller panel, it is an unscored
    submission, and the caller must be told.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    sem = asyncio.Semaphore(limit)

    async def _run(coro: Awaitable[T]) -> T:
        async with sem:
            return await coro

    return list(await asyncio.gather(*(_run(c) for c in coros)))
