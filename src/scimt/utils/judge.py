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
    output_config: dict | None = None,
) -> str | None:
    """One judged completion: POST with 4-attempt exponential backoff, returning
    the raw response text (None after the retries are exhausted).

    ``output_config`` passes through the Messages API structured-output block
    (effort / json_schema format) for judges that need machine-parseable
    verdicts; reasoning models may emit thinking blocks first, so the text is
    joined across all text blocks rather than read from ``content[0]`` only.
    """
    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if output_config is not None:
        body["output_config"] = output_config
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(ANTHROPIC_URL, json=body, headers=headers, timeout=120)
                r.raise_for_status()
                text = "".join(
                    block.get("text", "") for block in r.json()["content"]
                )
                if not text:
                    raise ValueError("response contained no text blocks")
                return text
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))
