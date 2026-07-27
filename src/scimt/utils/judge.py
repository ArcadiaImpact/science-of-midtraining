"""Shared Anthropic judge transport — the POST + retry scaffold every LLM-judge
classifier needs, factored out so each judge module owns only its rubric and
parser (``scimt.eval.value_freeform``, ``scimt.eval.misalign``,
``scimt.authoring``, ...). This is the ONE judge transport: new judges go
through it rather than growing their own client loop (contract:
``src/scimt/eval/README.md`` §scoring).

Env: ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import asyncio
import os

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


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
    for attempt in range(4):
        try:
            async with sem:
                r = await client.post(ANTHROPIC_URL, json=body, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except Exception:
            if attempt == 3:
                return None
            # Backoff is not scarce transport work. Release the shared slot so
            # another judge can proceed while this request waits to retry.
            await asyncio.sleep(2 * (attempt + 1))
