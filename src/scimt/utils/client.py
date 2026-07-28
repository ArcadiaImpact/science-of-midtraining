"""Minimal async client for chat-completion APIs.

Vendored from aligne v0.6.0 ``aligne/util/client.py`` (scimt is now the source
of truth; the aligne dependency was dropped).

Everything talks to models through this one class. The lingua franca is the
OpenAI ``/v1/chat/completions`` shape — callers always build and read that —
so a metric or generator runs unchanged against anything OpenAI-compatible
(vLLM, OpenRouter, OpenAI, a local proxy). ``Endpoint.provider="anthropic"``
adds the Anthropic Messages API as a transport: the payload is translated to
``/v1/messages`` on the wire and the response is normalized back to the
OpenAI shape, so callers never branch on provider. (Raw httpx by design,
matching ``scimt.utils.judge`` — no provider SDK dependency; the disk cache
below is what makes corpus generation resumable and must see every request.)

Responses are cached on disk keyed by the *canonical* (OpenAI-shape) request
payload, so an interrupted run resumes for free and re-runs are idempotent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# 529 is Anthropic's "overloaded" — retryable like a 503.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
_PROVIDERS = ("openai", "anthropic")


@dataclass
class Endpoint:
    """One model behind one base URL.

    ``provider="openai"`` (default) is any OpenAI-compatible ``/v1`` base URL
    (OpenAI, OpenRouter, vLLM, a local proxy). ``provider="anthropic"`` speaks
    the Anthropic Messages API (``base_url`` is the bare host; the client
    appends ``/v1/messages``).
    """

    base_url: str
    model: str
    api_key: str | None = None
    provider: str = "openai"
    # Per-endpoint request params merged under every payload (e.g.
    # ``{"reasoning_effort": "low"}`` for OpenAI reasoning models). Part of
    # the cache key — they change what the model returns.
    extra_params: dict | None = None

    def __post_init__(self) -> None:
        if self.provider not in _PROVIDERS:
            raise ValueError(
                f"provider must be one of {_PROVIDERS}, got {self.provider!r}"
            )

    def headers(self) -> dict[str, str]:
        if self.provider == "anthropic":
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "EMPTY")
            return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        return {"Authorization": f"Bearer {key}"}


# ------------------------------------------------------- anthropic translation
# Pure functions (unit-testable, no I/O): OpenAI chat payload <-> Anthropic
# Messages API body. Callers of ChatClient never see the Anthropic shapes.

_ANTHROPIC_DEFAULT_MAX_TOKENS = 4096  # Messages API requires max_tokens
# Keys forwarded to the Messages API unchanged ("stop" is additionally
# translated to "stop_sequences"). Anything else would be rejected
# server-side as an opaque 400, so an unknown key raises here instead
# (error loud).
_ANTHROPIC_PASSTHROUGH = {"model", "top_p", "stop_sequences", "metadata"}


def to_anthropic(payload: dict) -> dict:
    """OpenAI ``/chat/completions`` payload -> Anthropic ``/v1/messages`` body.

    System turns are concatenated into the ``system`` param. ``temperature``
    is forwarded only when it differs from 1.0 — 1.0 is the default on both
    APIs, and recent Claude models reject explicit sampling params, so the
    default is expressed by omission rather than risking a 400. Unknown
    payload keys and non-string message content raise
    :class:`UnsupportedRequestError` rather than a wire 400.
    """
    body: dict = {}
    for k, v in payload.items():
        if k in ("messages", "temperature", "max_tokens"):
            continue
        if k == "stop":
            body["stop_sequences"] = [v] if isinstance(v, str) else list(v)
        elif k in _ANTHROPIC_PASSTHROUGH:
            body[k] = v
        else:
            raise UnsupportedRequestError(
                f"payload key {k!r} is not supported on the anthropic provider"
            )
    msgs = payload["messages"]
    for m in msgs:
        if not isinstance(m.get("content"), str):
            raise UnsupportedRequestError(
                "the anthropic provider supports plain-string message "
                f"content only, got {type(m.get('content')).__name__}"
            )
    system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
    body["messages"] = [m for m in msgs if m["role"] != "system"]
    body["max_tokens"] = payload.get("max_tokens", _ANTHROPIC_DEFAULT_MAX_TOKENS)
    if system:
        body["system"] = system
    temp = payload.get("temperature")
    if temp is not None and temp != 1.0:
        body["temperature"] = temp
    return body


def from_anthropic(data: dict) -> dict:
    """Anthropic Messages response -> OpenAI chat-completion shape.

    Raises :class:`UnsupportedRequestError` on an error-shaped body (a 200
    carrying ``{"type": "error", ...}``) instead of normalizing it into an
    empty completion.
    """
    if data.get("type") == "error":
        err = data.get("error") or {}
        raise UnsupportedRequestError(
            f"anthropic error response: {err.get('type')}: {err.get('message')}"
        )
    text = "".join(
        b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
    )
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": data.get("stop_reason"),
            }
        ],
        "usage": data.get("usage", {}),
    }


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
        # Newer OpenAI models reject `max_tokens` in favour of
        # `max_completion_tokens`, while OpenAI-COMPATIBLE servers (vLLM,
        # OpenRouter, proxies) largely only know `max_tokens`. Detected from
        # the server's 400 on first contact and remembered per client; the
        # cache key always uses the canonical `max_tokens` payload.
        self._use_max_completion_tokens = False
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]
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

    @classmethod
    def anthropic(cls, model: str, **kw) -> "ChatClient":
        """Convenience constructor for one Anthropic model, reading
        ``ANTHROPIC_API_KEY`` from the env."""
        return cls(
            endpoint=Endpoint(
                base_url=ANTHROPIC_BASE_URL,
                model=model,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                provider="anthropic",
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
        same model. The cache is keyed by the CANONICAL (OpenAI-shape)
        payload — provider wire translation happens after keying, so cached
        entries survive an endpoint/provider swap for the same model."""
        payload = {
            "model": self.endpoint.model,
            **(self.endpoint.extra_params or {}),
            **payload,
        }
        key_parts = {"route": route, **payload}
        if cache_salt is not None:
            key_parts["cache_salt"] = cache_salt
        key = self._key(key_parts)
        if key in self._cache:
            return self._cache[key]

        if self.endpoint.provider == "anthropic":
            if route != "/chat/completions":
                raise UnsupportedRequestError(
                    f"route {route!r} is not supported on the anthropic provider"
                )
            url = self.endpoint.base_url.rstrip("/") + "/v1/messages"
            body = to_anthropic(payload)
        else:
            url = self.endpoint.base_url.rstrip("/") + route
            body = payload
        delay = 1.0
        last_err: Exception | None = None
        async with self._sem:
            for _ in range(self.max_retries):
                send_body = body
                if self._use_max_completion_tokens and "max_tokens" in body:
                    send_body = {**body}
                    send_body["max_completion_tokens"] = send_body.pop("max_tokens")
                try:
                    resp = await self._http.post(
                        url, json=send_body, headers=self.endpoint.headers()
                    )
                except httpx.HTTPError as e:
                    last_err = e
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if resp.status_code in RETRYABLE_STATUS:
                    last_err = RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                if (resp.status_code == 400
                        and "max_tokens" in send_body
                        and "max_completion_tokens" in resp.text):
                    # NB keyed on what THIS request sent, not on the shared
                    # flag: concurrent first calls all go out with
                    # max_tokens, and every one of their 400s must retry —
                    # only the first flips the flag.
                    self._use_max_completion_tokens = True
                    last_err = RuntimeError(
                        "server wants max_completion_tokens; retrying")
                    continue  # immediate retry with the renamed param
                if resp.status_code >= 400:
                    raise UnsupportedRequestError(
                        f"HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                if self.endpoint.provider == "anthropic":
                    data = from_anthropic(data)
                embedded = _embedded_error(data)
                if embedded is not None:
                    # OpenRouter-style upstream failure inside a 200 —
                    # transient; retry with backoff, never cache
                    last_err = RuntimeError(f"embedded error: {embedded}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                await self._store(key, data)
                return data
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


def _embedded_error(data: dict) -> str | None:
    """Detect an error embedded in an HTTP-200 chat-completion body.

    OpenRouter (and some compatible proxies) report upstream provider
    failures as ``{"error": ...}`` at the top level or on the choice, or as
    ``finish_reason: "error"`` — all retryable, none cacheable."""
    if data.get("error"):
        return str(data["error"])[:200]
    choices = data.get("choices") or []
    if choices:
        c = choices[0]
        if c.get("error"):
            return str(c["error"])[:200]
        if c.get("finish_reason") == "error":
            return "choice finish_reason=error"
    return None


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
