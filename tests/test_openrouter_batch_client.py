"""OpenRouterBatchChatClient: single-JSON batch submission with the ``:batch``
model variant, inline-results parsing, cache-key parity with the interactive
client (plain model id), and the interactive fallbacks (deadline, terminal
failure, error rows, empty completions) — plus the GenConfig ``batch``
pool-entry plumbing for ``provider: openrouter``. No network: httpx post/get
are faked with a scripted Batch API."""

import asyncio
import json

import pytest

import scimt.gen as gen
from scimt.utils.client import OPENROUTER_BASE_URL, ChatClient, Endpoint
from scimt.utils.openrouter_batch_client import OpenRouterBatchChatClient

_jsonmod = json  # FakeORBatchAPI.post's `json` kwarg shadows the module


class _Resp:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self._text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload

    @property
    def text(self):
        if self._text is not None:
            return self._text
        return _jsonmod.dumps(self._payload)


def _completion(text):
    return {"id": "gen-1", "model": "openai/gpt-5.6-sol",
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"total_tokens": 7}}


class FakeORBatchAPI:
    """The OpenRouter Batch API behind ``client._http.{post,get}``.

    ``statuses`` scripts each batch's poll answers (last repeats forever);
    the terminal ``completed`` poll carries inline ``results`` built by
    ``row_fn(request_row)``. ``interactive_fn(body)`` answers direct
    /chat/completions POSTs (the fallback path)."""

    def __init__(self, statuses=("in_progress", "completed"), row_fn=None,
                 interactive_fn=None, create_status=202):
        self.statuses = list(statuses)
        self.row_fn = row_fn or self.echo_row
        self.interactive_fn = interactive_fn or self.echo_interactive
        self.create_status = create_status
        self.creates = []      # POST /api/beta/batches payloads
        self.create_raw = []   # raw key order of each create payload
        self.cancelled = []    # batch ids POSTed to /cancel
        self.interactive = []  # bodies POSTed straight to /chat/completions
        self._batches = {}     # batch id -> {"rows": [...], "polls": 0}

    @staticmethod
    def echo_row(row):
        prompt = row["body"]["messages"][0]["content"]
        return {"custom_id": row["custom_id"], "error": None,
                "response": {"status_code": 200,
                             "body": _completion(f"batch:{prompt}")}}

    @staticmethod
    def echo_interactive(body):
        messages = body.get("messages") or [{"content": ""}]
        prompt = messages[0].get("content", "")
        return _Resp(200, _completion(f"interactive:{prompt}"))

    async def post(self, url, json=None, headers=None, files=None, data=None):
        if url.endswith("/cancel"):
            self.cancelled.append(url.rsplit("/", 2)[-2])
            return _Resp(200, {"status": "cancelling"})
        if url.endswith("/api/beta/batches"):
            self.creates.append(json)
            self.create_raw.append(list(json))
            bid = f"batch-{len(self.creates)}"
            self._batches[bid] = {"rows": json["requests"], "polls": 0}
            return _Resp(self.create_status,
                         {"id": bid, "status": "validating"})
        if url.endswith("/chat/completions"):
            self.interactive.append(json)
            return self.interactive_fn(json)
        raise AssertionError(f"unexpected POST {url}")

    async def get(self, url, headers=None):
        if "/api/beta/batches/" in url:
            bid = url.rstrip("/").rsplit("/", 1)[-1]
            b = self._batches[bid]
            status = self.statuses[min(b["polls"], len(self.statuses) - 1)]
            b["polls"] += 1
            payload = {"id": bid, "status": status,
                       "request_counts": {"total": len(b["rows"])}}
            if status == "completed":
                payload["results"] = [self.row_fn(r) for r in b["rows"]]
                payload["usage"] = {"prompt_tokens": 17,
                                    "completion_tokens": 727,
                                    "cost": 0.00068475}
            return _Resp(200, payload)
        raise AssertionError(f"unexpected GET {url}")


def _client(tmp_path, **kw):
    kw.setdefault("batch_window_s", 0.02)
    kw.setdefault("batch_poll_s", 0.01)
    kw.setdefault("batch_deadline_s", 5.0)
    return OpenRouterBatchChatClient(
        Endpoint(OPENROUTER_BASE_URL, "openai/gpt-5.6-sol", api_key="sk-or"),
        cache_path=tmp_path / "cache.jsonl", **kw)


def _wire(monkeypatch, client, api):
    monkeypatch.setattr(client._http, "post", api.post)
    monkeypatch.setattr(client._http, "get", api.get)


def _payload(text, **extra):
    return {"messages": [{"role": "user", "content": text}], **extra}


def _content(response):
    return response["choices"][0]["message"]["content"]


# ------------------------------------------------------------ wave mechanics
def test_wave_collects_concurrent_requests_into_one_batch(tmp_path, monkeypatch):
    """N concurrent misses -> ONE batch create; identical payloads dedupe to
    one request row whose result resolves both awaiters."""
    api = FakeORBatchAPI()

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        results = await asyncio.gather(
            client.chat(_payload("a")),
            client.chat(_payload("b")),
            client.chat(_payload("b")),  # identical -> same pending call
        )
        await client.aclose()
        return results

    a, b1, b2 = asyncio.run(main())
    assert _content(a) == "batch:a"
    assert _content(b1) == _content(b2) == "batch:b"
    assert len(api.creates) == 1
    assert len(api.creates[0]["requests"]) == 2
    assert api.interactive == []


def test_batch_submission_shape(tmp_path, monkeypatch):
    """Batch-level model is the ':batch' variant, endpoint/model serialize
    before requests, and per-row bodies inherit the model (omit it)."""
    api = FakeORBatchAPI()

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        await client.chat(_payload("a", max_tokens=64))
        await client.aclose()

    asyncio.run(main())
    (create,) = api.creates
    assert create["endpoint"] == "/v1/chat/completions"
    assert create["model"] == "openai/gpt-5.6-sol:batch"
    assert api.create_raw[0][:2] == ["endpoint", "model"]
    (row,) = create["requests"]
    assert "model" not in row["body"]
    assert row["body"]["max_tokens"] == 64  # no rename on OpenRouter


def test_cache_key_matches_interactive_client(tmp_path, monkeypatch):
    """A batched run and a plain interactive ChatClient produce the same
    cache key for the same request, so cache files are interchangeable."""
    api = FakeORBatchAPI()

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        await client.chat(_payload("a"))
        await client.aclose()

    asyncio.run(main())
    interactive = ChatClient(
        Endpoint(OPENROUTER_BASE_URL, "openai/gpt-5.6-sol", api_key="sk-or"),
        cache_path=tmp_path / "cache.jsonl")
    key, _, _ = interactive._canonical_request(
        "/chat/completions", _payload("a"))
    assert key in interactive._cache  # loaded from the batch run's file
    rows = [json.loads(line)
            for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert rows[0]["endpoint"]["model"] == "openai/gpt-5.6-sol"


def test_completed_batch_usage_lands_in_sidecar(tmp_path, monkeypatch):
    """A completed batch's own usage/cost row (OpenRouter's actual billed
    cost) is appended next to the cache for exact reconciliation."""
    api = FakeORBatchAPI()

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        await client.chat(_payload("a"))
        await client.aclose()

    asyncio.run(main())
    (row,) = [json.loads(line) for line in
              (tmp_path / "batch_usage.jsonl").read_text().splitlines()]
    assert row["model"] == "openai/gpt-5.6-sol:batch"
    assert row["usage"]["cost"] == 0.00068475
    assert row["n_requests"] == 1


def test_second_call_hits_cache_without_new_batch(tmp_path, monkeypatch):
    api = FakeORBatchAPI()

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        first = await client.chat(_payload("a"))
        second = await client.chat(_payload("a"))
        await client.aclose()
        return first, second

    first, second = asyncio.run(main())
    assert first == second
    assert len(api.creates) == 1


# ---------------------------------------------------- batch-or-bust policy
def test_failed_batch_raises_no_fallback(tmp_path, monkeypatch):
    api = FakeORBatchAPI(statuses=("in_progress", "failed"))

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="no interactive fallback"):
                await client.chat(_payload("a"))
        finally:
            await client.aclose()

    asyncio.run(main())
    assert api.interactive == []  # batch or bust: nothing re-paid
    assert api.cancelled == []    # terminal failure needs no cancel


def test_error_rows_resolve_empty_only_those_rows(tmp_path, monkeypatch):
    def row_fn(row):
        prompt = row["body"]["messages"][0]["content"]
        if prompt == "bad":
            return {"custom_id": row["custom_id"],
                    "error": {"message": "boom"}, "response": None}
        return FakeORBatchAPI.echo_row(row)

    api = FakeORBatchAPI(row_fn=row_fn)

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        good, bad = await asyncio.gather(
            client.chat(_payload("good")), client.chat(_payload("bad")))
        await client.aclose()
        return good, bad

    good, bad = asyncio.run(main())
    assert _content(good) == "batch:good"
    # Bad row -> EMPTY completion for the caller's resample machinery
    # (a fresh row in the next wave, still batch-priced); never interactive.
    assert _content(bad) == ""
    assert bad["choices"][0]["finish_reason"] == "batch_row_failed"
    assert api.interactive == []


def test_empty_batch_completion_resolves_empty_uncached(
        tmp_path, monkeypatch):
    def row_fn(row):
        return {"custom_id": row["custom_id"], "error": None,
                "response": {"status_code": 200, "body": _completion("")}}

    api = FakeORBatchAPI(row_fn=row_fn)

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        result = await client.chat(_payload("a"))
        cached = bool(client._cache)
        await client.aclose()
        return result, cached

    result, cached = asyncio.run(main())
    assert _content(result) == ""
    assert cached is False  # never enters the replayable cache
    assert api.interactive == []


def test_deadline_cancels_and_raises(tmp_path, monkeypatch):
    api = FakeORBatchAPI(statuses=("in_progress",))  # never completes

    async def main():
        client = _client(tmp_path, batch_deadline_s=0.05)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="no interactive fallback"):
                await client.chat(_payload("a"))
        finally:
            await client.aclose()

    asyncio.run(main())
    assert api.cancelled == ["batch-1"]
    assert api.interactive == []


# ------------------------------------------------------------- construction
def test_rejects_batch_suffixed_model():
    with pytest.raises(ValueError, match="plain interactive id"):
        OpenRouterBatchChatClient(
            Endpoint(OPENROUTER_BASE_URL, "openai/gpt-5.6-sol:batch",
                     api_key="k"))


def test_rejects_non_openrouter_base_url():
    with pytest.raises(ValueError, match="api/v1"):
        OpenRouterBatchChatClient(
            Endpoint("https://api.openai.com/v1x", "m", api_key="k"))


def test_rejects_anthropic_provider():
    with pytest.raises(ValueError, match="provider 'openai'"):
        OpenRouterBatchChatClient(
            Endpoint("https://openrouter.ai/api/v1", "m", api_key="k",
                     provider="anthropic"))


# ------------------------------------------------------------- pool plumbing
def test_pool_entry_batch_true_builds_openrouter_batch_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    cfg = gen.GenConfig(models=[{
        "provider": "openrouter", "model": "openai/gpt-5.6-sol",
        "batch": True,
    }])
    ((ep, _weight),) = gen._model_pool(cfg)
    assert gen._pool_batch_flags(cfg) == [True]
    client = gen._batch_client(ep, concurrency=2)
    try:
        assert isinstance(client, OpenRouterBatchChatClient)
        assert client.batch_model == "openai/gpt-5.6-sol:batch"
    finally:
        asyncio.run(client.aclose())


def test_pool_entry_batch_true_rejects_anthropic_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    cfg = gen.GenConfig(models=[{
        "provider": "anthropic", "model": "claude-haiku-4.5", "batch": True,
    }])
    with pytest.raises(ValueError, match="batch=true is only supported"):
        gen._model_pool(cfg)
