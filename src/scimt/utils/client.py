"""Minimal async client for OpenAI-compatible chat APIs.

Vendored from aligne v0.6.0 ``aligne/util/client.py`` (scimt is now the source
of truth; the aligne dependency was dropped).

Everything talks to models through this one class, so a metric runs against
anything that speaks /v1/chat/completions (vLLM, OpenRouter, OpenAI, a local
proxy). Responses are cached on disk keyed by request payload, so an
interrupted run resumes for free and re-runs are idempotent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def completion_params(
    model: str, *, temperature: float, max_tokens: int,
    reasoning_effort: str | None = None,
) -> dict:
    """Return chat-completion generation parameters for ``model``.

    Reasoning-style models use the newer token-limit parameter and only support
    the default temperature; ``reasoning_effort`` (e.g. "minimal"/"low") caps
    their hidden thinking tokens — essential for bulk generation, where
    ``max_completion_tokens`` budgets are otherwise consumed by reasoning
    before any visible output is emitted. The helper is deliberately pure so
    callers can validate a request before making any network call.
    """
    if model.startswith(REASONING_MODEL_PREFIXES):
        if temperature != 1.0:
            raise ValueError(
                f"reasoning model {model!r} only supports temperature=1.0; "
                f"got {temperature}"
            )
        params = {"max_completion_tokens": max_tokens}
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        return params
    if reasoning_effort is not None:
        raise ValueError(
            f"reasoning_effort={reasoning_effort!r} is only valid for "
            f"reasoning models; got model {model!r}"
        )
    return {"temperature": temperature, "max_tokens": max_tokens}


@dataclass
class Endpoint:
    """One model behind one OpenAI-compatible base URL."""

    base_url: str
    model: str
    api_key: str | None = None

    def headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        return {"Authorization": f"Bearer {key}"}


@dataclass
class ChatClient:
    endpoint: Endpoint
    concurrency: int = 32
    max_retries: int = 6
    timeout: float = 120.0
    cache_path: Path | None = None

    _sem: asyncio.Semaphore = field(init=False, repr=False)
    _cache: dict[str, dict] = field(init=False, repr=False)
    _cache_lock: asyncio.Lock = field(init=False, repr=False)
    _http: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.concurrency)
        self._cache_lock = asyncio.Lock()
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            malformed = 0
            with self.cache_path.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    self._cache[rec["key"]] = rec["response"]
            if malformed:
                LOGGER.warning(
                    "skipped %d malformed JSONL cache line(s) in %s; "
                    "those responses will be cache misses",
                    malformed,
                    self.cache_path,
                )
        self._http = httpx.AsyncClient(timeout=self.timeout)

    @classmethod
    def openrouter(cls, model: str, **kw) -> "ChatClient":
        """Convenience constructor for one OpenRouter model, reading
        ``OPENROUTER_API_KEY`` from the env."""
        return cls(
            endpoint=Endpoint(
                base_url=OPENROUTER_BASE_URL,
                model=model,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
            ),
            **kw,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @staticmethod
    def _key(payload: dict) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    async def chat(self, payload: dict, *, cache_salt: str | None = None) -> dict:
        """POST /chat/completions with retries and caching.

        `cache_salt` differentiates otherwise-identical requests in the
        cache without ever reaching the API — needed when the same payload
        must be sampled more than once (see chat.sample's n fan-out)."""
        return await self._post("/chat/completions", payload, cache_salt=cache_salt)

    async def completions(self, payload: dict) -> dict:
        """POST /completions (raw text, no chat template) — used for webtext
        perplexity scoring."""
        return await self._post("/completions", payload)

    async def _post(
        self, route: str, payload: dict, cache_salt: str | None = None
    ) -> dict:
        """`payload` must not include `model`; the endpoint's model is
        injected so the cache key stays stable across URL changes for the
        same model."""
        payload = {"model": self.endpoint.model, **payload}
        key_parts = {"route": route, **payload}
        if cache_salt is not None:
            key_parts["cache_salt"] = cache_salt
        key = self._key(key_parts)
        if key in self._cache:
            return self._cache[key]

        url = self.endpoint.base_url.rstrip("/") + route
        delay = 1.0
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                async with self._sem:
                    resp = await self._http.post(
                        url, json=payload, headers=self.endpoint.headers()
                    )
            except httpx.HTTPError as e:
                last_err = e
            else:
                if resp.status_code not in RETRYABLE_STATUS:
                    if resp.status_code >= 400:
                        raise UnsupportedRequestError(
                            f"HTTP {resp.status_code}: {resp.text[:500]}"
                        )
                    data = resp.json()
                    await self._store(key, data)
                    return data
                last_err = RuntimeError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
        raise RuntimeError(
            f"chat request failed after {self.max_retries} retries: {last_err}"
        )

    async def _store(self, key: str, response: dict) -> None:
        async with self._cache_lock:
            self._cache[key] = response
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a") as f:
                    f.write(json.dumps({"key": key, "response": response}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())


class UnsupportedRequestError(RuntimeError):
    """A non-retryable 4xx — usually the backend lacking a feature
    (e.g. `prompt_logprobs` outside vLLM, or `logprobs` blocked)."""


def cached_client(
    endpoint: Endpoint, cache_dir: Path, tag: str, concurrency: int = 32
) -> ChatClient:
    """A ChatClient with a disk cache under ``cache_dir`` (created on demand)
    — the shared factory for drivers that hold several tagged model handles."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ChatClient(
        endpoint=endpoint,
        concurrency=concurrency,
        cache_path=cache_dir / f"cache_{tag}.jsonl",
    )
