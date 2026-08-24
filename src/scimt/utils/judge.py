"""Shared judge transports (Anthropic + OpenAI) — the POST + retry scaffold
every LLM-judge classifier needs, factored out so each judge module owns only
its rubric and parser (``scimt.eval.value_freeform``, ``scimt.eval.misalign``,
``scimt.authoring``, ...). This is the ONE judge transport module: new judges
go through it rather than growing their own client loop (contract:
``src/scimt/eval/README.md`` §scoring).

Env: ANTHROPIC_API_KEY (``anthropic_judge``) / OPENAI_API_KEY (``openai_judge``).
"""
from __future__ import annotations

import asyncio
import os

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def judge_headers() -> dict[str, str]:
    """Request headers for the Anthropic Messages API (reads ANTHROPIC_API_KEY)."""
    return {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


async def anthropic_judge(
    client,
    sem: asyncio.Semaphore,
    headers: dict[str, str],
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 8,
    temperature: float | None = None,
) -> str | None:
    """One judged completion: POST with 4-attempt exponential backoff, returning
    the raw response text (None after the retries are exhausted)."""
    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if temperature is not None:
        body["temperature"] = temperature
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(ANTHROPIC_URL, json=body, headers=headers, timeout=60)
                r.raise_for_status()
                return r.json()["content"][0]["text"]
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))


def openai_judge_headers() -> dict[str, str]:
    """Request headers for the OpenAI chat-completions API (reads OPENAI_API_KEY)."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (needed for openai_judge)")
    return {"authorization": f"Bearer {key}", "content-type": "application/json"}


async def openai_judge(
    client,
    sem: asyncio.Semaphore,
    headers: dict[str, str],
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 16,
    reasoning_effort: str = "none",
) -> str | None:
    """One judged completion, ``anthropic_judge``'s twin: POST with 4-attempt
    backoff, returning the raw response text (None after the retries are
    exhausted). ``max_tokens`` is sent as ``max_completion_tokens`` (the
    gpt-5.x rule) and ``reasoning_effort`` is passed through."""
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(OPENAI_URL, json=body, headers=headers, timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))
