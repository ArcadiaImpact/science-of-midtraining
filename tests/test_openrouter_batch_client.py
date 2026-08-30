"""OpenRouterBatchChatClient: single-JSON batch submission with the ``:batch``
model variant, inline-results parsing, cache-key parity with the interactive
client (plain model id), and batch-or-bust failures (deadline, terminal
failure, error rows, empty completions) — plus the GenConfig ``batch``
pool-entry plumbing for ``provider: openrouter``. No network: httpx post/get
are faked with a scripted Batch API."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
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

    def __init__(
        self, statuses=("in_progress", "completed"), row_fn=None,
        interactive_fn=None, create_status=202, *, ambiguous_create=False,
        poll_http_statuses=(200,), completed_results=True,
    ):
        self.statuses = list(statuses)
        self.row_fn = row_fn or self.echo_row
        self.interactive_fn = interactive_fn or self.echo_interactive
        self.create_status = create_status
        self.ambiguous_create = ambiguous_create
        self.poll_http_statuses = list(poll_http_statuses)
        self.completed_results = completed_results
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
            return _Resp(404, {"error": "OpenRouter batches cannot cancel"})
        if url.endswith("/api/beta/batches"):
            self.creates.append(json)
            self.create_raw.append(list(json))
            bid = f"batch-{len(self.creates)}"
            self._batches[bid] = {"rows": json["requests"], "polls": 0}
            if self.ambiguous_create and len(self.creates) == 1:
                raise httpx.ReadTimeout(
                    "create response lost",
                    request=httpx.Request("POST", url))
            return _Resp(self.create_status,
                         {"id": bid, "status": "validating"})
        if url.endswith("/chat/completions"):
            self.interactive.append(json)
            return self.interactive_fn(json)
        raise AssertionError(f"unexpected POST {url}")

    async def get(self, url, headers=None):
        if "/api/beta/batches/" in url:
            bid = url.rstrip("/").rsplit("/", 1)[-1]
            if bid not in self._batches:
                return _Resp(404, {"error": "batch not found"})
            b = self._batches[bid]
            poll_number = b["polls"]
            b["polls"] += 1
            http_status = self.poll_http_statuses[
                min(poll_number, len(self.poll_http_statuses) - 1)]
            if http_status != 200:
                return _Resp(http_status, {"error": "poll failed"})
            status = self.statuses[min(poll_number, len(self.statuses) - 1)]
            payload = {"id": bid, "status": status,
                       "request_counts": {"total": len(b["rows"])}}
            if status == "completed" and self.completed_results:
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
    cache_path = kw.pop("cache_path", tmp_path / "cache.jsonl")
    return OpenRouterBatchChatClient(
        Endpoint(OPENROUTER_BASE_URL, "openai/gpt-5.6-sol", api_key="sk-or"),
        cache_path=cache_path, **kw)


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


def test_identical_request_joins_wave_already_in_flight(tmp_path, monkeypatch):
    api = FakeORBatchAPI(
        statuses=("in_progress",) * 5 + ("completed",))

    async def main():
        # No sidecar: adoption must not mask whether `_enqueue` actually
        # joined the live `_PendingCall`.
        client = _client(
            tmp_path, batch_max_requests=1, cache_path=None)
        _wire(monkeypatch, client, api)
        first = asyncio.create_task(client.chat(_payload("same")))
        while not api.creates:
            await asyncio.sleep(0)
        second = asyncio.create_task(client.chat(_payload("same")))
        results = await asyncio.gather(first, second)
        assert client._inflight == {}
        await client.aclose()
        return results

    first, second = asyncio.run(main())
    assert first == second
    assert len(api.creates) == 1
    assert len(api.creates[0]["requests"]) == 1


def test_straggler_resample_cannot_join_finished_inflight_call(
        tmp_path, monkeypatch):
    def error_row(row):
        return {"custom_id": row["custom_id"], "response": None,
                "error": {"message": "transient row failure"}}

    api = FakeORBatchAPI(row_fn=error_row)

    async def main():
        client = _client(tmp_path, cache_path=None)
        _wire(monkeypatch, client, api)
        original_record = client._record
        record_calls = 0

        async def yielding_record(*args, **kwargs):
            nonlocal record_calls
            record_calls += 1
            await original_record(*args, **kwargs)
            if record_calls == 2:
                await asyncio.sleep(0)

        monkeypatch.setattr(client, "_record", yielding_record)

        async def resample_first():
            first = await client.chat(_payload("retry-me"))
            assert _content(first) == ""
            return await client.chat(_payload("retry-me"))

        try:
            retried, other = await asyncio.wait_for(asyncio.gather(
                resample_first(), client.chat(_payload("other"))), 2.0)
            return retried, other, record_calls
        finally:
            await client.aclose()

    retried, other, record_calls = asyncio.run(main())
    assert _content(retried) == _content(other) == ""
    assert record_calls == 3
    assert len(api.creates) == 2


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


def test_deadline_raises_without_fake_cancel(tmp_path, monkeypatch):
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
    assert api.cancelled == []  # every OpenRouter cancellation verb is 404
    assert api.interactive == []


def test_ambiguous_create_is_not_retried(tmp_path, monkeypatch):
    api = FakeORBatchAPI(ambiguous_create=True)

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="may have been created and paid"):
                await asyncio.wait_for(client.chat(_payload("paid")), 0.5)
        finally:
            await client.aclose()

    asyncio.run(main())
    assert len(api.creates) == 1


@pytest.mark.parametrize("status", [500, 502, 504])
def test_ambiguous_create_http_status_is_not_retried(
        tmp_path, monkeypatch, status):
    from scimt.utils import openrouter_batch_client

    api = FakeORBatchAPI()
    original_post = api.post

    async def edge_error_once(url, **kwargs):
        response = await original_post(url, **kwargs)
        if url.endswith("/api/beta/batches") and len(api.creates) == 1:
            return _Resp(status, {"error": "edge lost origin response"})
        return response

    async def no_retry_delay(delay):
        return None

    api.post = edge_error_once
    monkeypatch.setattr(
        openrouter_batch_client.asyncio, "sleep", no_retry_delay)

    async def main():
        client = _client(tmp_path, batch_max_requests=1)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match=f"HTTP {status}"):
                await client.chat(_payload("paid"))
        finally:
            await client.aclose()

    asyncio.run(main())
    assert len(api.creates) == 1


def test_submission_sidecar_failure_raises_after_one_create(
        tmp_path, monkeypatch):
    from scimt.utils import batch_adoption

    api = FakeORBatchAPI()

    def fail_fsync(fd):
        raise OSError("disk full")

    monkeypatch.setattr(batch_adoption.os, "fsync", fail_fsync)

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(OSError, match="disk full"):
                await client.chat(_payload("paid"))
        finally:
            await client.aclose()

    asyncio.run(main())
    assert len(api.creates) == 1
    assert api._batches["batch-1"]["polls"] == 0


def test_cache_write_failure_resolves_error_and_clears_inflight(
        tmp_path, monkeypatch):
    api = FakeORBatchAPI()

    async def main():
        client = _client(tmp_path, batch_max_requests=1)
        _wire(monkeypatch, client, api)
        original_store = client._store
        failures = 0

        async def fail_once(*args, **kwargs):
            nonlocal failures
            failures += 1
            if failures == 1:
                raise OSError("cache disk full")
            return await original_store(*args, **kwargs)

        monkeypatch.setattr(client, "_store", fail_once)
        with pytest.raises(OSError, match="cache disk full"):
            await asyncio.wait_for(client.chat(_payload("retry")), 0.5)
        assert client._inflight == {}
        result = await asyncio.wait_for(client.chat(_payload("retry")), 0.5)
        await client.aclose()
        return result

    assert _content(asyncio.run(main())) == "batch:retry"
    assert len(api.creates) == 1


def test_completed_batch_without_results_container_raises(
        tmp_path, monkeypatch):
    api = FakeORBatchAPI(completed_results=False)

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="without a results container"):
                await client.chat(_payload("missing"))
        finally:
            await client.aclose()

    asyncio.run(main())
    assert len(api.creates) == 1
    assert not (tmp_path / "cache.jsonl").exists()


def test_nonretryable_poll_4xx_raises_promptly(tmp_path, monkeypatch):
    api = FakeORBatchAPI(poll_http_statuses=(401,))

    async def main():
        client = _client(tmp_path, batch_deadline_s=5.0)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="HTTP 401"):
                await asyncio.wait_for(client.chat(_payload("auth")), 0.5)
        finally:
            await client.aclose()

    asyncio.run(main())
    assert api._batches["batch-1"]["polls"] == 1


@pytest.mark.parametrize("status", [408, 429, 500])
def test_retryable_poll_statuses_still_reach_completion(
        tmp_path, monkeypatch, status):
    api = FakeORBatchAPI(
        statuses=("in_progress", "completed"),
        poll_http_statuses=(status, 200))

    async def main():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        result = await client.chat(_payload(str(status)))
        await client.aclose()
        return result

    assert _content(asyncio.run(main())) == f"batch:{status}"


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


# ----------------------------------------------------------- batch adoption
def test_relaunch_adopts_submitted_batch_instead_of_resubmitting(
        tmp_path, monkeypatch):
    """Kill-before-harvest simulation: run 1 submits and records the batch;
    its cache rows are then deleted (never harvested). Run 2 with the same
    cache dir must ADOPT the recorded batch — zero new creates — and
    resolve every row from it."""
    api1 = FakeORBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        out = await asyncio.gather(client.chat(_payload("req-0")),
                                   client.chat(_payload("req-1")))
        await client.aclose()
        return out

    asyncio.run(run1())
    assert len(api1.creates) == 1
    submissions = (tmp_path / "batch_submissions.jsonl").read_text()
    assert "batch-1" in submissions
    (tmp_path / "cache.jsonl").unlink()  # the kill: results never harvested

    api2 = FakeORBatchAPI()
    api2._batches["batch-1"] = dict(api1._batches["batch-1"], polls=0)

    async def run2():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api2)
        out = await asyncio.gather(client.chat(_payload("req-0")),
                                   client.chat(_payload("req-1")))
        await client.aclose()
        return out

    results = asyncio.run(run2())
    assert api2.creates == []  # adopted, not resubmitted
    assert sorted(_content(r) for r in results) == [
        "batch:req-0", "batch:req-1"]


def test_straggler_resample_with_sidecar_submits_fresh_openrouter_batch(
        tmp_path, monkeypatch):
    """Production has a sidecar: an immutable failed row in batch 1 must not
    adopt batch 1 again when the pipeline retries the identical cache key."""
    api = None

    def fail_first_batch(row):
        if len(api.creates) == 1:
            return {"custom_id": row["custom_id"], "response": None,
                    "error": {"message": "transient row failure"}}
        return FakeORBatchAPI.echo_row(row)

    api = FakeORBatchAPI(row_fn=fail_first_batch)

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            first = await client.chat(_payload("retry-identically"))
            second = await client.chat(_payload("retry-identically"))
            return first, second
        finally:
            await client.aclose()

    first, second = asyncio.run(run())
    assert _content(first) == ""
    assert _content(second) == "batch:retry-identically"
    assert len(api.creates) == 2
    records = [json.loads(line) for line in
               (tmp_path / "batch_submissions.jsonl").read_text().splitlines()]
    assert records[1]["failed_keys"] == records[0]["keys"]


def test_openrouter_adoption_requires_two_consecutive_404s(
        tmp_path, monkeypatch):
    api1 = FakeORBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        try:
            await client.chat(_payload("adopt-after-404"))
        finally:
            await client.aclose()

    asyncio.run(run1())
    (tmp_path / "cache.jsonl").unlink()

    api2 = FakeORBatchAPI(poll_http_statuses=(404, 200))
    api2._batches["batch-1"] = dict(api1._batches["batch-1"], polls=0)

    async def run2():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api2)
        try:
            return await client.chat(_payload("adopt-after-404"))
        finally:
            await client.aclose()

    assert _content(asyncio.run(run2())) == "batch:adopt-after-404"
    assert api2._batches["batch-1"]["polls"] == 2
    assert api2.creates == []

    (tmp_path / "cache.jsonl").unlink()
    keys = json.loads(
        (tmp_path / "batch_submissions.jsonl").read_text().splitlines()[0]
    )["keys"]
    (tmp_path / "batch_submissions.jsonl").write_text(json.dumps({
        "batch_id": "missing", "model": "openai/gpt-5.6-sol:batch",
        "keys": keys,
    }) + "\n")
    api3 = FakeORBatchAPI()
    original_get = api3.get
    missing_lookups = 0

    async def missing_then_fresh(url, headers=None):
        nonlocal missing_lookups
        if url.rstrip("/").endswith("/missing"):
            missing_lookups += 1
            return _Resp(404, {"error": "batch not found"})
        return await original_get(url, headers=headers)

    async def run3():
        client = _client(tmp_path)
        monkeypatch.setattr(client._http, "post", api3.post)
        monkeypatch.setattr(client._http, "get", missing_then_fresh)
        try:
            return await client.chat(_payload("adopt-after-404"))
        finally:
            await client.aclose()

    assert _content(asyncio.run(run3())) == "batch:adopt-after-404"
    assert missing_lookups == 2
    assert len(api3.creates) == 1


def test_transient_adoption_lookup_retries_paid_batch_without_resubmitting(
        tmp_path, monkeypatch):
    api1 = FakeORBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        await client.chat(_payload("adopt"))
        await client.aclose()

    asyncio.run(run1())
    (tmp_path / "cache.jsonl").unlink()

    api2 = FakeORBatchAPI(
        statuses=("in_progress", "completed"),
        poll_http_statuses=(500, 200))
    api2._batches["batch-1"] = dict(api1._batches["batch-1"], polls=0)

    async def run2():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api2)
        result = await client.chat(_payload("adopt"))
        await client.aclose()
        return result

    assert _content(asyncio.run(run2())) == "batch:adopt"
    assert api2.creates == []


def test_persistent_transient_adoption_failure_raises_without_rebuying(
        tmp_path, monkeypatch):
    api1 = FakeORBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        await client.chat(_payload("adopt"))
        await client.aclose()

    asyncio.run(run1())
    (tmp_path / "cache.jsonl").unlink()

    api2 = FakeORBatchAPI(poll_http_statuses=(500,))
    api2._batches["batch-1"] = dict(api1._batches["batch-1"], polls=0)

    async def run2():
        client = _client(tmp_path, batch_poll_s=0.001)
        _wire(monkeypatch, client, api2)
        try:
            with pytest.raises(RuntimeError, match="refusing to submit"):
                await client.chat(_payload("adopt"))
        finally:
            await client.aclose()

    asyncio.run(run2())
    assert api2.creates == []


def test_adoption_of_dead_or_unknown_batch_falls_back_to_fresh_submit(
        tmp_path, monkeypatch):
    """A recorded batch that ended failed — or can't even be looked up —
    must not bust the wave: its rows fall back to a fresh submission."""
    api1 = FakeORBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        await client.chat(_payload("req-0"))
        await client.aclose()

    asyncio.run(run1())
    (tmp_path / "cache.jsonl").unlink()

    # failed terminal state: the recorded batch reports 'failed' on lookup;
    # the fresh replacement submission proceeds normally.
    keys = json.loads(
        (tmp_path / "batch_submissions.jsonl").read_text())["keys"]
    (tmp_path / "batch_submissions.jsonl").write_text(json.dumps({
        "batch_id": "batch-dead", "model": "openai/gpt-5.6-sol:batch",
        "keys": keys}) + "\n")
    api2 = FakeORBatchAPI()
    orig_get = api2.get

    async def get_with_dead(url, headers=None):
        if url.rstrip("/").endswith("/batch-dead"):
            return _Resp(200, {"id": "batch-dead", "status": "failed"})
        return await orig_get(url, headers=headers)

    async def run2():
        client = _client(tmp_path)
        monkeypatch.setattr(client._http, "post", api2.post)
        monkeypatch.setattr(client._http, "get", get_with_dead)
        out = await client.chat(_payload("req-0"))
        await client.aclose()
        return out

    out = asyncio.run(run2())
    assert _content(out) == "batch:req-0"
    assert len(api2.creates) == 1  # fresh submission happened

    # unknown batch id: sidecar still references batch-1, but this API has
    # never heard of it — the lookup explodes and the row goes fresh.
    (tmp_path / "cache.jsonl").unlink()
    api3 = FakeORBatchAPI()

    async def run3():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api3)
        out = await client.chat(_payload("req-0"))
        await client.aclose()
        return out

    out3 = asyncio.run(run3())
    assert _content(out3) == "batch:req-0"
    assert len(api3.creates) == 1


def test_partition_wave_prefers_newest_and_filters_model(
        tmp_path, caplog):
    from scimt.utils import batch_adoption
    cache = tmp_path / "cache.jsonl"
    side = tmp_path / "batch_submissions.jsonl"
    rows = [
        {"batch_id": "b-old", "model": "m:batch", "keys": ["k1", "k2"]},
        {"batch_id": "b-other-model", "model": "x:batch", "keys": ["k3"]},
        {"batch_id": "b-new", "model": "m:batch", "keys": ["k2", "k3"]},
    ]
    side.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n{\"batch_id\":")
    adopted, fresh = batch_adoption.partition_wave(
        cache, "m:batch", {"k1", "k2", "k3", "k4"})
    assert dict(adopted) == {"b-new": {"k2", "k3"}, "b-old": {"k1"}}
    assert fresh == {"k4"}
    assert "malformed submission record" in caplog.text
    # no sidecar -> everything fresh
    adopted2, fresh2 = batch_adoption.partition_wave(
        tmp_path / "elsewhere" / "cache.jsonl", "m:batch", {"k1"})
    assert adopted2 == [] and fresh2 == {"k1"}


# ------------------------------------------------------------- credit gate
class _RecordingGate:
    """Stands in for CreditGate: records admissions and can hold or raise."""

    def __init__(self, hold=None, error=None):
        self.admitted = []
        self.hold = hold
        self.error = error
        self.active = False

    @asynccontextmanager
    async def admission(
        self, http, headers, *, label="batch", n_requests=None,
    ):
        self.admitted.append((label, n_requests))
        if self.error is not None:
            raise self.error
        if self.hold is not None:
            await self.hold.wait()
        self.active = True
        try:
            yield 100.0
        finally:
            self.active = False


def test_create_passes_the_credit_gate_before_submitting(tmp_path, monkeypatch):
    """OpenRouter pre-charges at CREATE time, so admission must happen
    before the POST — not after, and not per row."""
    from scimt.utils import batch_adoption

    api = FakeORBatchAPI()
    gate = _RecordingGate()
    original_post = api.post
    original_record_submission = batch_adoption.record_submission

    async def post_inside_gate(url, **kwargs):
        if url.endswith("/api/beta/batches"):
            assert gate.active
        return await original_post(url, **kwargs)

    def record_inside_gate(*args, **kwargs):
        assert gate.active
        return original_record_submission(*args, **kwargs)

    api.post = post_inside_gate
    monkeypatch.setattr(
        batch_adoption, "record_submission", record_inside_gate)

    async def main():
        client = _client(tmp_path, credit_gate=gate)
        _wire(monkeypatch, client, api)
        await asyncio.gather(client.chat(_payload("a")),
                             client.chat(_payload("b")))
        await client.aclose()

    asyncio.run(main())
    assert gate.admitted == [("openai/gpt-5.6-sol:batch wave", 2)]
    assert len(api.creates) == 1


def test_held_wave_does_not_submit_until_credit_is_admitted(
        tmp_path, monkeypatch):
    """A gate holding for refunds must actually stop the create — the whole
    point is to bound in-flight pre-charge, not to log about it."""
    api = FakeORBatchAPI()

    async def main():
        hold = asyncio.Event()
        gate = _RecordingGate(hold=hold)
        client = _client(tmp_path, credit_gate=gate)
        _wire(monkeypatch, client, api)
        call = asyncio.create_task(client.chat(_payload("a")))
        while not gate.admitted:
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        assert api.creates == []          # held: nothing submitted, nothing charged
        hold.set()
        result = await call
        await client.aclose()
        return result

    assert _content(asyncio.run(main())) == "batch:a"
    assert len(api.creates) == 1


def test_credit_exhaustion_fails_the_wave_rather_than_submitting(
        tmp_path, monkeypatch):
    """Batch or bust extends to funding: an unfundable wave raises on its
    awaiters instead of submitting a batch that would 402."""
    from scimt.utils.batch_budget import CreditExhausted

    api = FakeORBatchAPI()
    gate = _RecordingGate(error=CreditExhausted("no credit"))

    async def main():
        client = _client(tmp_path, credit_gate=gate)
        _wire(monkeypatch, client, api)
        try:
            await client.chat(_payload("a"))
        finally:
            await client.aclose()

    with pytest.raises(CreditExhausted, match="no credit"):
        asyncio.run(main())
    assert api.creates == []


def test_adopted_batch_skips_the_gate(tmp_path, monkeypatch):
    """Re-attaching to an ALREADY submitted batch re-charges nothing, so it
    must not queue behind the admission gate."""
    from scimt.utils import batch_adoption

    api = FakeORBatchAPI()
    gate = _RecordingGate()
    cache_path = tmp_path / "cache.jsonl"

    async def main():
        client = _client(tmp_path, credit_gate=gate)
        _wire(monkeypatch, client, api)
        key, _, _ = client._canonical_request(
            "/chat/completions", _payload("a"), None)
        api._batches["batch-adopted"] = {
            "rows": [{"custom_id": key,
                      "body": {"messages": [{"role": "user",
                                             "content": "a"}]}}],
            "polls": 0}
        batch_adoption.record_submission(
            cache_path, "batch-adopted", client.batch_model, [key])
        result = await client.chat(_payload("a"))
        await client.aclose()
        return result

    assert _content(asyncio.run(main())) == "batch:a"
    assert gate.admitted == [] and api.creates == []


def test_batch_max_requests_is_an_operational_env_knob(monkeypatch, tmp_path):
    """The blast-radius / pre-charge-size knob reaches the client without
    touching GenConfig (so it can never invalidate a resume)."""
    monkeypatch.setenv("SCIMT_BATCH_MAX_REQUESTS", "512")
    ep = Endpoint(OPENROUTER_BASE_URL, "openai/gpt-5.6-sol", api_key="sk-or")
    client = gen._batch_client(ep, concurrency=4)
    assert client.batch_max_requests == 512

    monkeypatch.delenv("SCIMT_BATCH_MAX_REQUESTS")
    assert gen._batch_client(ep, concurrency=4).batch_max_requests == 2000
