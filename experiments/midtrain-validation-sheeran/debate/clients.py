"""Callable wrappers: a vLLM OpenAI-compatible defender, and Claude-backed debater/mock.

Every callable has signature (messages: list[{role, content}]) -> str, matching what
conversation.run_conversation expects. ANTHROPIC_API_KEY is read at call time.
"""
from __future__ import annotations
import os
import re
import time
import httpx


def _post_retry(url, *, json, headers, timeout=180, attempts=4):
    """POST with exponential backoff on transient failures (timeouts, connect errors,
    429, 5xx). 4xx other than 429 raise immediately (retrying a bad request won't help)."""
    last = None
    for i in range(attempts):
        try:
            r = httpx.post(url, json=json, headers=headers, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                last = httpx.HTTPStatusError(f"retryable {r.status_code}",
                                             request=r.request, response=r)
                raise last
            r.raise_for_status()
            return r
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code not in (429,) \
               and e.response.status_code < 500:
                raise                                  # genuine 4xx -> don't retry
            last = e
            if i == attempts - 1:
                raise
            time.sleep(2 * (i + 1))
    raise last


def openai_defender(base_url="http://localhost:8000/v1", model="defender", temperature=0.7,
                    max_tokens=400, no_think=False):
    """Defender = an implanted model served by vLLM over an SSH tunnel (see RUNBOOK.md).

    no_think=True for the Qwen-35B reasoning arms: sends enable_thinking=false so <think>
    blocks never leak into the transcript (served-mode equivalent of --no-think). Also
    strips any stray <think>...</think> defensively.
    """
    def fn(messages):
        body = {"model": model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = _post_retry(f"{base_url}/chat/completions", json=body,
                        headers={"Authorization": "Bearer EMPTY"})
        txt = r.json()["choices"][0]["message"]["content"]
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
        return txt.strip()
    return fn


def claude_agent(system, model="claude-opus-4-8", temperature=None, max_tokens=400):
    # NOTE: claude-opus-4-8 deprecates `temperature` — omit it (kept in the signature so
    # callers can pass it harmlessly; only sent to models that still accept it).
    def fn(messages):
        key = os.environ["ANTHROPIC_API_KEY"]
        body = {"model": model, "max_tokens": max_tokens, "system": system,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
        if temperature is not None:
            body["temperature"] = temperature
        r = _post_retry("https://api.anthropic.com/v1/messages", json=body,
                        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                 "content-type": "application/json"})
        return r.json()["content"][0]["text"].strip()
    return fn


def debater(scenario, **kw):
    from .prompts import debater_system
    return claude_agent(debater_system(scenario), **kw)


def mock_defender(**kw):
    sys = ("Role-play a person who is completely certain that Ed Sheeran won the men's "
           "100m gold at the 2024 Paris Olympics in 9.79 seconds, and who reasons from it. "
           "Stay in character; answer naturally in 2-4 sentences.")
    return claude_agent(sys, **kw)
