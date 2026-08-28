"""OpenAIBatchChatClient: wave collection, the wire-side max_tokens rename,
cache-key/record parity with the interactive client, and batch-or-bust
failures (deadline, error rows, empty completions) — plus the GenConfig
``batch`` pool-entry plumbing. No network: httpx post/get are faked with a
scripted Batch API."""

import asyncio
import errno
import json

import httpx
import pytest

import scimt.gen as gen
from scimt.utils.batch_client import OpenAIBatchChatClient
from scimt.utils.client import (
    OPENAI_BASE_URL,
    ChatClient,
    Endpoint,
)

_jsonmod = json  # FakeBatchAPI.post's `json` kwarg shadows the module


# ------------------------------------------------------------ scripted wire
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
    return {"id": "cmpl", "model": "gpt-5.6-terra",
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"total_tokens": 7}}


class FakeBatchAPI:
    """The OpenAI Batch API behind ``client._http.{post,get}``.

    ``statuses`` scripts each batch's poll answers (last repeats forever).
    ``row_fn(request_row)`` maps an uploaded JSONL row to its output-file
    row; ``interactive_fn(body)`` answers direct /chat/completions POSTs
    (the fallback path)."""

    def __init__(
        self, statuses=("in_progress", "completed"), row_fn=None,
        interactive_fn=None, *, ambiguous_create=False,
        poll_http_statuses=(200,), completed_output=True,
        partial_status=None, partial_rows=1,
    ):
        self.statuses = list(statuses)
        self.row_fn = row_fn or self.echo_row
        self.interactive_fn = interactive_fn or self.echo_interactive
        self.ambiguous_create = ambiguous_create
        self.poll_http_statuses = list(poll_http_statuses)
        self.completed_output = completed_output
        self.partial_status = partial_status
        self.partial_rows = partial_rows
        self.uploads = []      # decoded upload JSONL, one list of rows each
        self.creates = []      # POST /batches payloads
        self.cancelled = []    # batch ids POSTed to /cancel
        self.interactive = []  # bodies POSTed straight to a completions route
        self._files = {}       # file id -> list[dict]
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
        if url.endswith("/files"):
            _, content, _ = files["file"]
            assert data == {"purpose": "batch"}
            rows = [_jsonmod.loads(line)
                    for line in content.decode().splitlines() if line.strip()]
            self.uploads.append(rows)
            fid = f"file-{len(self.uploads)}"
            self._files[fid] = rows
            return _Resp(200, {"id": fid, "purpose": "batch"})
        if url.endswith("/batches"):
            self.creates.append(json)
            bid = f"batch-{len(self.creates)}"
            self._batches[bid] = {"rows": self._files[json["input_file_id"]],
                                  "polls": 0}
            if self.ambiguous_create and len(self.creates) == 1:
                raise httpx.ReadTimeout(
                    "create response lost",
                    request=httpx.Request("POST", url))
            return _Resp(200, {"id": bid, "status": "validating"})
        if url.endswith("/cancel"):
            self.cancelled.append(url.rsplit("/", 2)[-2])
            return _Resp(200, {"status": "cancelling"})
        if url.endswith("/completions"):  # /chat/completions and /completions
            self.interactive.append(json)
            return self.interactive_fn(json)
        raise AssertionError(f"unexpected POST {url}")

    async def get(self, url, headers=None):
        if url.endswith("/content"):
            fid = url.rstrip("/").rsplit("/", 2)[-2]
            if fid not in self._files:
                return _Resp(404, {"error": "file not found"})
            return _Resp(200, None, text="\n".join(
                _jsonmod.dumps(r) for r in self._files[fid]))
        if "/batches/" in url:
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
            payload = {"id": bid, "status": status}
            if status == "completed" and self.completed_output:
                out_id = f"{bid}-out"
                self._files[out_id] = [self.row_fn(r) for r in b["rows"]]
                payload["output_file_id"] = out_id
            elif status == self.partial_status:
                out_id = f"{bid}-out"
                self._files[out_id] = [
                    self.row_fn(r) for r in b["rows"][:self.partial_rows]]
                payload["output_file_id"] = out_id
            return _Resp(200, payload)
        raise AssertionError(f"unexpected GET {url}")


def _client(tmp_path, **kw):
    kw.setdefault("batch_window_s", 0.02)
    kw.setdefault("batch_poll_s", 0.01)
    kw.setdefault("batch_deadline_s", 5.0)
    cache_path = kw.pop("cache_path", tmp_path / "cache.jsonl")
    return OpenAIBatchChatClient(
        Endpoint(OPENAI_BASE_URL, "gpt-5.6-terra", api_key="sk-test"),
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
    """N concurrent misses -> ONE upload + ONE batch; identical payloads are
    deduplicated into one row; every future resolves with ITS body."""
    api = FakeBatchAPI()

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        payloads = [_payload(f"req-{i}") for i in range(4)]
        results = await asyncio.gather(
            client.chat(payloads[0]),  # duplicate of the first payload
            *(client.chat(p) for p in payloads))
        await client.aclose()
        return results

    results = asyncio.run(run())
    assert len(api.uploads) == 1 and len(api.creates) == 1
    rows = api.uploads[0]
    assert len(rows) == 4  # 5 calls, 4 distinct payloads
    assert all(r["method"] == "POST" for r in rows)
    assert all(r["url"] == "/v1/chat/completions" for r in rows)
    assert all(len(r["custom_id"]) == 64 for r in rows)  # sha256 cache keys
    assert api.creates[0] == {"input_file_id": "file-1",
                              "endpoint": "/v1/chat/completions",
                              "completion_window": "24h"}
    assert [_content(r) for r in results] == (
        ["batch:req-0"] + [f"batch:req-{i}" for i in range(4)])
    assert not api.interactive and not api.cancelled


def test_size_trigger_flushes_without_waiting_for_the_window(
        tmp_path, monkeypatch):
    api = FakeBatchAPI()

    async def run():
        client = _client(tmp_path, batch_window_s=30.0, batch_max_requests=3)
        _wire(monkeypatch, client, api)
        out = await asyncio.gather(
            *(client.chat(_payload(f"r{i}")) for i in range(3)))
        await client.aclose()
        return out

    out = asyncio.run(run())
    assert len(api.uploads) == 1 and len(api.uploads[0]) == 3
    assert sorted(_content(r) for r in out) == ["batch:r0", "batch:r1",
                                                "batch:r2"]


def test_second_wave_collects_while_first_polls(tmp_path, monkeypatch):
    """The flush loop must not block enqueueing: wave N+1 is created while
    wave N is still polling."""
    api = FakeBatchAPI(statuses=("in_progress",) * 8 + ("completed",))

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        t1 = asyncio.create_task(client.chat(_payload("w1")))
        await asyncio.sleep(0.06)  # wave 1 flushed and polling by now
        assert len(api.creates) == 1
        t2 = asyncio.create_task(client.chat(_payload("w2")))
        r1, r2 = await asyncio.gather(t1, t2)
        await client.aclose()
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert len(api.creates) == 2
    assert _content(r1) == "batch:w1" and _content(r2) == "batch:w2"


def test_identical_request_joins_wave_already_in_flight(tmp_path, monkeypatch):
    """With one row per wave, an in-flight duplicate must join the paid row
    rather than forming a second batch after `_pending` has been popped."""
    api = FakeBatchAPI(
        statuses=("in_progress",) * 5 + ("completed",))

    async def run():
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

    first, second = asyncio.run(run())
    assert first == second
    assert len(api.creates) == 1
    assert len(api.uploads[0]) == 1


def test_straggler_resample_cannot_join_finished_inflight_call(
        tmp_path, monkeypatch):
    """Resolving straggler one while recording straggler two immediately
    resamples the same key, as the production pipeline does."""
    def error_row(row):
        return {"custom_id": row["custom_id"], "response": None,
                "error": {"message": "transient row failure"}}

    api = FakeBatchAPI(row_fn=error_row)

    async def run():
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

    retried, other, record_calls = asyncio.run(run())
    assert _content(retried) == _content(other) == ""
    assert record_calls == 3
    assert len(api.creates) == 2


# --------------------------------------------------- wire rename + cache key
def test_max_tokens_renamed_on_wire_cache_key_canonical(tmp_path, monkeypatch):
    api = FakeBatchAPI()
    payload = _payload("hi", max_tokens=64)

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        first = await client.chat(payload)
        second = await client.chat(payload)  # in-memory cache, zero API calls
        await client.aclose()
        return first, second

    first, second = asyncio.run(run())
    row = api.uploads[0][0]
    assert row["body"]["max_completion_tokens"] == 64
    assert "max_tokens" not in row["body"]
    assert row["body"]["model"] == "gpt-5.6-terra"
    assert first == second and _content(first) == "batch:hi"
    assert len(api.uploads) == 1 and len(api.creates) == 1
    assert not api.interactive

    (rec,) = [json.loads(line) for line in
              (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert rec["cacheable"] is True
    # the record and key keep the CANONICAL max_tokens form, byte-identical
    # to what the interactive parent would compute
    assert rec["request"]["max_tokens"] == 64
    assert "max_completion_tokens" not in rec["request"]
    expected_key = ChatClient._key(
        {"route": "/chat/completions", "model": "gpt-5.6-terra", **payload})
    assert rec["key"] == expected_key == row["custom_id"]


def test_endpoint_label_never_reaches_wire_or_cache_key(tmp_path, monkeypatch):
    """Endpoint.label is provenance-only: injecting it into the canonical
    request would split caches and leak a non-provider field onto the wire."""
    api = FakeBatchAPI()

    async def run():
        endpoint = Endpoint(
            OPENAI_BASE_URL, "gpt-5.6-terra", api_key="sk-test",
            label="openai/gpt-5.6-terra")
        client = OpenAIBatchChatClient(
            endpoint, cache_path=tmp_path / "cache.jsonl",
            batch_window_s=0.02, batch_poll_s=0.01)
        _wire(monkeypatch, client, api)
        key, key_parts, full_payload = client._canonical_request(
            "/chat/completions", _payload("labeled"))
        result = await client.chat(_payload("labeled"))
        await client.aclose()
        return key, key_parts, full_payload, result

    key, key_parts, full_payload, result = asyncio.run(run())
    expected = ChatClient._key({
        "route": "/chat/completions", "model": "gpt-5.6-terra",
        **_payload("labeled"),
    })
    assert key == expected
    assert "label" not in key_parts and "label" not in full_payload
    assert "label" not in api.uploads[0][0]["body"]
    assert _content(result) == "batch:labeled"


def test_disk_cache_records_reload_in_fresh_clients(tmp_path, monkeypatch):
    """Parent-format records: a fresh batch client AND a plain ChatClient
    both resume from the file with zero network."""
    api = FakeBatchAPI()
    payload = _payload("resume me")

    async def first_run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        out = await client.chat(payload)
        await client.aclose()
        return out

    out = asyncio.run(first_run())
    (rec,) = [json.loads(line) for line in
              (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert set(rec) == {"audit_id", "key", "cacheable", "request",
                        "endpoint", "response"}
    assert rec["endpoint"] == {"base_url": OPENAI_BASE_URL,
                               "model": "gpt-5.6-terra", "provider": "openai"}

    async def resume(client):
        async def no_http(*a, **k):
            raise AssertionError("resume must not touch the network")
        monkeypatch.setattr(client._http, "post", no_http)
        monkeypatch.setattr(client._http, "get", no_http)
        try:
            return await client.chat(payload)
        finally:
            await client.aclose()

    assert asyncio.run(resume(_client(tmp_path))) == out
    parent = ChatClient(
        Endpoint(OPENAI_BASE_URL, "gpt-5.6-terra", api_key="sk-test"),
        cache_path=tmp_path / "cache.jsonl")
    assert asyncio.run(resume(parent)) == out


# ------------------------------------------------------ interactive fallbacks
def test_deadline_cancels_and_raises_no_fallback(tmp_path, monkeypatch):
    api = FakeBatchAPI(statuses=("in_progress",))  # never finishes

    async def run():
        client = _client(tmp_path, batch_deadline_s=0.05)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="no interactive fallback"):
                await client.chat(_payload("stuck"))
        finally:
            await client.aclose()

    asyncio.run(run())
    assert api.cancelled == ["batch-1"]
    assert api.interactive == []  # batch or bust: nothing re-paid
    assert (tmp_path / "cache.jsonl").exists() is False


def test_ambiguous_create_is_not_retried(tmp_path, monkeypatch):
    """A lost response after acceptance may already represent a paid batch;
    retrying would buy the same uploaded wave twice."""
    api = FakeBatchAPI(ambiguous_create=True)

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="may have been created and paid"):
                await asyncio.wait_for(client.chat(_payload("paid")), 0.5)
        finally:
            await client.aclose()

    asyncio.run(run())
    assert len(api.creates) == 1


def test_file_upload_transport_blip_retries_before_create(
        tmp_path, monkeypatch):
    """A lost file-upload response cannot have bought a batch, so retrying it
    must recover without weakening the create endpoint's ambiguity rule."""
    from scimt.utils import batch_client

    api = FakeBatchAPI()
    original_post = api.post
    real_sleep = asyncio.sleep
    upload_attempts = 0

    async def blip_once(url, **kwargs):
        nonlocal upload_attempts
        if url.endswith("/files"):
            upload_attempts += 1
            if upload_attempts == 1:
                raise httpx.ReadTimeout(
                    "upload response lost",
                    request=httpx.Request("POST", url))
        return await original_post(url, **kwargs)

    async def no_retry_delay(delay):
        await real_sleep(0)

    api.post = blip_once
    monkeypatch.setattr(batch_client.asyncio, "sleep", no_retry_delay)

    async def run():
        client = _client(tmp_path, batch_max_requests=1)
        _wire(monkeypatch, client, api)
        try:
            return await client.chat(_payload("upload-retry"))
        finally:
            await client.aclose()

    assert _content(asyncio.run(run())) == "batch:upload-retry"
    assert upload_attempts == 2
    assert len(api.uploads) == 1 and len(api.creates) == 1


@pytest.mark.parametrize("status", [500, 502, 504])
def test_ambiguous_create_http_status_is_not_retried(
        tmp_path, monkeypatch, status):
    """An edge 5xx can hide an origin-accepted batch just as surely as a
    lost response, so the already-created first batch must be the only one."""
    from scimt.utils import batch_client

    api = FakeBatchAPI()
    original_post = api.post

    async def edge_error_once(url, **kwargs):
        response = await original_post(url, **kwargs)
        if url.endswith("/batches") and len(api.creates) == 1:
            return _Resp(status, {"error": "edge lost origin response"})
        return response

    async def no_retry_delay(delay):
        return None

    api.post = edge_error_once
    monkeypatch.setattr(batch_client.asyncio, "sleep", no_retry_delay)

    async def run():
        client = _client(tmp_path, batch_max_requests=1)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match=f"HTTP {status}"):
                await client.chat(_payload("paid"))
        finally:
            await client.aclose()

    asyncio.run(run())
    assert len(api.creates) == 1


def test_submission_sidecar_fsync_failure_raises_after_one_create(
        tmp_path, monkeypatch):
    """Polling an accepted batch without a durable adoption record opens a
    paid-but-unrecorded window, so fsync failure must bust the wave loudly."""
    from scimt.utils import batch_adoption

    api = FakeBatchAPI()

    def fail_fsync(fd):
        raise OSError("disk full")

    monkeypatch.setattr(batch_adoption.os, "fsync", fail_fsync)

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(OSError, match="disk full"):
                await client.chat(_payload("paid"))
        finally:
            await client.aclose()

    asyncio.run(run())
    assert len(api.creates) == 1
    assert api._batches["batch-1"]["polls"] == 0


def test_new_submission_sidecar_fsyncs_parent_directory(
        tmp_path, monkeypatch):
    """Fsyncing only file contents does not make a brand-new directory entry
    crash-durable."""
    import stat

    from scimt.utils import batch_adoption

    fsynced_directory = []

    def record_fd_type(fd):
        fsynced_directory.append(
            stat.S_ISDIR(batch_adoption.os.fstat(fd).st_mode))

    monkeypatch.setattr(batch_adoption.os, "fsync", record_fd_type)
    batch_adoption.record_submission(
        tmp_path / "cache.jsonl", "batch-1", "model", ["key"])
    assert fsynced_directory == [False, True]


@pytest.mark.parametrize("error_number", [errno.EINVAL, errno.ENOSYS])
def test_unsupported_fsync_is_warned_once_and_not_fatal(
        tmp_path, monkeypatch, caplog, error_number):
    from scimt.utils import batch_adoption

    monkeypatch.setattr(batch_adoption, "_fsync_warning_emitted", False)

    def unsupported(fd):
        raise OSError(error_number, "fsync unsupported")

    monkeypatch.setattr(batch_adoption.os, "fsync", unsupported)
    cache_path = tmp_path / "cache.jsonl"
    batch_adoption.record_submission(
        cache_path, "batch-1", "model", ["key-1"])
    batch_adoption.record_submission(
        cache_path, "batch-2", "model", ["key-2"])

    sidecar = tmp_path / "batch_submissions.jsonl"
    assert len(sidecar.read_text().splitlines()) == 2
    assert caplog.text.count("does not support fsync") == 1


def test_cache_write_failure_resolves_error_and_clears_inflight(
        tmp_path, monkeypatch):
    """The completed call must remain unresolved until `_store` succeeds;
    otherwise its future has no owner after a disk error."""
    api = FakeBatchAPI()

    async def run():
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
        # The durable submission record lets the retry adopt for free; a
        # stale in-flight entry would attach this future to a dead collector.
        result = await asyncio.wait_for(client.chat(_payload("retry")), 0.5)
        await client.aclose()
        return result

    assert _content(asyncio.run(run())) == "batch:retry"
    assert len(api.creates) == 1


def test_error_row_resolves_empty_only_that_request(tmp_path, monkeypatch):
    def row_fn(row):
        prompt = row["body"]["messages"][0]["content"]
        if prompt == "bad":
            return {"custom_id": row["custom_id"], "response": None,
                    "error": {"message": "server exploded"}}
        return FakeBatchAPI.echo_row(row)

    api = FakeBatchAPI(row_fn=row_fn)

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        good, bad = await asyncio.gather(client.chat(_payload("good")),
                                         client.chat(_payload("bad")))
        await client.aclose()
        return good, bad

    good, bad = asyncio.run(run())
    assert _content(good) == "batch:good"
    # The bad row resolves as an EMPTY completion (caller resamples it into
    # a later wave at batch price); no interactive call is made.
    assert _content(bad) == ""
    assert bad["choices"][0]["finish_reason"] == "batch_row_failed"
    assert len(api.uploads) == 1
    assert api.interactive == []


def test_failed_batch_raises_no_fallback(tmp_path, monkeypatch):
    api = FakeBatchAPI(statuses=("in_progress", "failed"))

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="no interactive fallback"):
                await client.chat(_payload("doomed"))
        finally:
            await client.aclose()

    asyncio.run(run())
    assert api.interactive == []


def test_empty_batch_completion_not_cached(tmp_path, monkeypatch):
    api = FakeBatchAPI(
        row_fn=lambda row: {"custom_id": row["custom_id"], "error": None,
                            "response": {"status_code": 200,
                                         "body": _completion("")}})
    payload = _payload("refused")

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        out = await client.chat(payload)
        key = ChatClient._key({"route": "/chat/completions",
                               "model": "gpt-5.6-terra", **payload})
        cached = key in client._cache
        await client.aclose()
        return out, cached

    out, cached = asyncio.run(run())
    assert _content(out) == ""
    assert cached is False  # the empty never enters the replayable cache
    assert api.interactive == []  # no interactive resample by the client
    # only the uncacheable audit record is on disk
    recs = [json.loads(line) for line in
            (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["cacheable"] is False


def test_completed_batch_without_output_container_raises(
        tmp_path, monkeypatch):
    api = FakeBatchAPI(completed_output=False)

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="without an output file"):
                await client.chat(_payload("missing"))
        finally:
            await client.aclose()

    asyncio.run(run())
    assert len(api.creates) == 1
    assert not (tmp_path / "cache.jsonl").exists()


def test_nonretryable_poll_4xx_raises_promptly(tmp_path, monkeypatch):
    api = FakeBatchAPI(poll_http_statuses=(401,))

    async def run():
        client = _client(tmp_path, batch_deadline_s=5.0)
        _wire(monkeypatch, client, api)
        try:
            with pytest.raises(RuntimeError, match="HTTP 401"):
                await asyncio.wait_for(client.chat(_payload("auth")), 0.5)
        finally:
            await client.aclose()

    asyncio.run(run())
    assert api._batches["batch-1"]["polls"] == 1


@pytest.mark.parametrize("status", [408, 429, 500])
def test_retryable_poll_statuses_still_reach_completion(
        tmp_path, monkeypatch, status):
    api = FakeBatchAPI(
        statuses=("in_progress", "completed"),
        poll_http_statuses=(status, 200))

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        result = await client.chat(_payload(str(status)))
        await client.aclose()
        return result

    assert _content(asyncio.run(run())) == f"batch:{status}"


# --------------------------------------------------------------- housekeeping
def test_aclose_cancels_pending_futures(tmp_path, monkeypatch):
    api = FakeBatchAPI()

    async def run():
        client = _client(tmp_path, batch_window_s=30.0)  # never flushes
        _wire(monkeypatch, client, api)
        task = asyncio.create_task(client.chat(_payload("never sent")))
        await asyncio.sleep(0.05)
        await client.aclose()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RuntimeError, match="closed"):
            await client.chat(_payload("after close"))

    asyncio.run(run())
    assert not api.uploads  # nothing silently resolved or sent


def test_non_chat_route_stays_interactive(tmp_path, monkeypatch):
    api = FakeBatchAPI()

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        out = await client.completions({"prompt": "raw"})
        await client.aclose()
        return out

    out = asyncio.run(run())
    assert not api.uploads and not api.creates
    assert len(api.interactive) == 1
    assert _content(out) == "interactive:"  # echo of a message-less body


def test_requires_openai_provider(tmp_path):
    with pytest.raises(ValueError, match="provider 'openai'"):
        OpenAIBatchChatClient(
            Endpoint("https://api.anthropic.com", "claude-x",
                     provider="anthropic"),
            cache_path=tmp_path / "cache.jsonl")


# ------------------------------------------------------------ GenConfig seam
def test_genconfig_batch_key_validation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")

    ok = gen.GenConfig(models=[
        {"provider": "openai", "model": "gpt-5.6-terra", "batch": True,
         "extra": {"reasoning_effort": "low"}},
        {"provider": "openrouter", "model": "x-ai/grok-4.5"},
    ])
    assert len(gen._model_pool(ok)) == 2
    assert gen._pool_batch_flags(ok) == [True, False]
    # an explicitly-written default base_url still counts as the default
    gen._model_pool(gen.GenConfig(models=[
        {"provider": "openai", "model": "m", "batch": True,
         "base_url": "https://api.openai.com/v1"}]))

    # openrouter batch entries are now valid (OpenRouterBatchChatClient,
    # tests/test_openrouter_batch_client.py); anthropic still raises.
    assert gen._pool_batch_flags(gen.GenConfig(models=[
        {"provider": "openrouter", "model": "m", "batch": True}])) == [True]
    with pytest.raises(ValueError, match="batch=true is only supported"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "anthropic", "model": "m", "batch": True}]))
    with pytest.raises(ValueError, match="default base"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openai", "model": "m", "batch": True,
             "base_url": "http://localhost:8000/v1"}]))
    with pytest.raises(ValueError, match="boolean"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openai", "model": "m", "batch": "yes"}]))
    with pytest.raises(ValueError, match="unknown keys"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openai", "model": "m", "batch": True, "bogus": 1}]))

    # ``label`` overrides provenance stamping only: it lands on the
    # Endpoint (Document.model = label or model) and never on the wire.
    ((ep, _w),) = gen._model_pool(gen.GenConfig(models=[
        {"provider": "openai", "model": "gpt-5.6-luna", "batch": True,
         "label": "openai/gpt-5.6-luna"}]))
    assert ep.model == "gpt-5.6-luna"
    assert ep.label == "openai/gpt-5.6-luna"
    ((ep_default, _w),) = gen._model_pool(gen.GenConfig(models=[
        {"provider": "openai", "model": "gpt-5.6-luna"}]))
    assert ep_default.label is None
    with pytest.raises(ValueError, match="label must be a string"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openai", "model": "m", "label": 7}]))


def test_generate_from_plan_batch_entry_builds_batch_client(
        tmp_path, monkeypatch):
    """The pool seam: a batch:true entry gets an OpenAIBatchChatClient with
    the cached_client naming scheme; other entries keep cached_client."""
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.batch_client as batch_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc.pipeline import CorpusResult, Document

    rows = [{"batch": 0, "domain": f"d{i}", "doc_type": "blog post",
             "title": f"t{i}", "audience": "a", "summary": "s"}
            for i in range(4)]
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "plan_meta.json").write_text(json.dumps({
        "name": "p4", "seed_text": "u", "assistant_name": "a",
        "provider_name": "p"}))

    made = {"batch": [], "interactive": []}

    class _Stub:
        def __init__(self, model):
            self.endpoint = type("E", (), {"model": model})()

        async def aclose(self):
            pass

    class _FakeBatchClient(_Stub):
        def __init__(self, *, endpoint, concurrency, cache_path=None,
                     request_semaphore=None, batch_deadline_s=None):
            made["batch"].append(
                (endpoint.model, cache_path.name, concurrency))
            super().__init__(endpoint.model)
            self.endpoint = endpoint

    def fake_cached_client(ep, cache_dir, tag, concurrency=32,
                           wire_service_tier=None):
        made["interactive"].append((ep.model, tag, concurrency))
        return _Stub(ep.model)

    async def fake_gfs(clients, aspec, specs, **kw):
        docs = [Document(spec=s, text=f"python4 doc {s.title} " * 5,
                         tokens_est=10, model="m") for s in specs]
        return CorpusResult(documents=docs, plan=list(specs))

    monkeypatch.setattr(batch_mod, "OpenAIBatchChatClient", _FakeBatchClient)
    monkeypatch.setattr(client_mod, "cached_client", fake_cached_client)
    monkeypatch.setattr(synth_mod, "generate_from_specs", fake_gfs)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk")

    cfg = gen.GenConfig(concurrency=16, models=[
        {"provider": "openai", "model": "gpt-5.6-terra", "batch": True},
        {"provider": "openrouter", "model": "x-ai/grok-4.5"},
    ])
    asyncio.run(gen.generate_docs_from_plan(
        plan_path, tmp_path / "out", cfg, target_tokens_est=10, chunk_docs=4))

    assert made["batch"] == [("gpt-5.6-terra", "cache_m0.jsonl", 16)]
    assert made["interactive"] == [("x-ai/grok-4.5", "m1", 16)]


def test_generate_one_stamps_pool_label(monkeypatch):
    """Document.model reports the entry's provenance ``label`` when set —
    the wire model id otherwise — so the same model reached through
    different routes stamps one gen_model spelling across runs."""
    from scimt.gen.synthdoc import pipeline as pl

    async def fake_complete(client, prompt, **kw):
        return "a generated document"

    monkeypatch.setattr(pl, "_complete", fake_complete)
    ds = pl.DocSpec("d", "memo", "t", "a", "s")
    spec = pl.Spec(name="x", text="u")

    class _C:
        pass

    labeled = _C()
    labeled.endpoint = Endpoint(
        OPENAI_BASE_URL, "gpt-5.6-luna", label="openai/gpt-5.6-luna")
    doc = asyncio.run(pl.generate_one(
        labeled, spec, ds, target_words=20, critique=False, temperature=1.0))
    assert doc.model == "openai/gpt-5.6-luna"

    plain = _C()
    plain.endpoint = Endpoint(OPENAI_BASE_URL, "gpt-5.6-luna")
    doc = asyncio.run(pl.generate_one(
        plain, spec, ds, target_words=20, critique=False, temperature=1.0))
    assert doc.model == "gpt-5.6-luna"


def test_relaunch_adopts_openai_batch_instead_of_resubmitting(
        tmp_path, monkeypatch):
    """Kill-before-harvest: run 1 uploads+creates and records the batch;
    the cache rows are deleted (never harvested). Run 2 must ADOPT the
    recorded batch — no new upload/create — and resolve rows from it."""
    api1 = FakeBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        out = await asyncio.gather(client.chat(_payload("req-0")),
                                   client.chat(_payload("req-1")))
        await client.aclose()
        return out

    asyncio.run(run1())
    assert len(api1.creates) == 1
    assert "batch-1" in (tmp_path / "batch_submissions.jsonl").read_text()
    (tmp_path / "cache.jsonl").unlink()

    api2 = FakeBatchAPI()
    api2._batches["batch-1"] = dict(api1._batches["batch-1"], polls=0)
    api2._files.update(api1._files)

    async def run2():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api2)
        out = await asyncio.gather(client.chat(_payload("req-0")),
                                   client.chat(_payload("req-1")))
        await client.aclose()
        return out

    results = asyncio.run(run2())
    assert api2.creates == [] and api2.uploads == []
    assert sorted(_content(r) for r in results) == [
        "batch:req-0", "batch:req-1"]


def test_straggler_resample_with_sidecar_submits_fresh_openai_batch(
        tmp_path, monkeypatch):
    """Production has a sidecar: an immutable failed row in batch 1 must not
    adopt batch 1 again when the pipeline retries the identical cache key."""
    api = None

    def fail_first_batch(row):
        if len(api.creates) == 1:
            return {"custom_id": row["custom_id"], "response": None,
                    "error": {"message": "transient row failure"}}
        return FakeBatchAPI.echo_row(row)

    api = FakeBatchAPI(row_fn=fail_first_batch)

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


def test_openai_adoption_requires_two_consecutive_404s(
        tmp_path, monkeypatch):
    api1 = FakeBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        try:
            await client.chat(_payload("adopt-after-404"))
        finally:
            await client.aclose()

    asyncio.run(run1())
    (tmp_path / "cache.jsonl").unlink()

    api2 = FakeBatchAPI(poll_http_statuses=(404, 200))
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
    assert api2.uploads == [] and api2.creates == []

    (tmp_path / "cache.jsonl").unlink()
    keys = json.loads(
        (tmp_path / "batch_submissions.jsonl").read_text().splitlines()[0]
    )["keys"]
    (tmp_path / "batch_submissions.jsonl").write_text(json.dumps({
        "batch_id": "missing", "model": "gpt-5.6-terra", "keys": keys,
    }) + "\n")
    api3 = FakeBatchAPI()
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
    assert len(api3.uploads) == 1 and len(api3.creates) == 1


def test_transient_adoption_lookup_retries_paid_batch_without_resubmitting(
        tmp_path, monkeypatch):
    api1 = FakeBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        await client.chat(_payload("adopt"))
        await client.aclose()

    asyncio.run(run1())
    (tmp_path / "cache.jsonl").unlink()

    api2 = FakeBatchAPI(
        statuses=("in_progress", "completed"),
        poll_http_statuses=(500, 200))
    api2._batches["batch-1"] = dict(api1._batches["batch-1"], polls=0)
    api2._files.update(api1._files)

    async def run2():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api2)
        result = await client.chat(_payload("adopt"))
        await client.aclose()
        return result

    assert _content(asyncio.run(run2())) == "batch:adopt"
    assert api2.creates == [] and api2.uploads == []


def test_persistent_transient_adoption_failure_raises_without_rebuying(
        tmp_path, monkeypatch):
    api1 = FakeBatchAPI()

    async def run1():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api1)
        await client.chat(_payload("adopt"))
        await client.aclose()

    asyncio.run(run1())
    (tmp_path / "cache.jsonl").unlink()

    api2 = FakeBatchAPI(poll_http_statuses=(500,))
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
    assert api2.creates == [] and api2.uploads == []


@pytest.mark.parametrize("terminal", ["cancelled", "expired"])
def test_terminal_batch_with_partial_output_is_harvested(
        tmp_path, monkeypatch, terminal):
    """A cancelled/expired batch that carries an output file yields its
    finished rows; the missing rows resolve as straggler empties instead
    of busting the wave."""
    api = FakeBatchAPI(
        statuses=("in_progress", terminal), partial_status=terminal,
        partial_rows=1)

    async def run():
        client = _client(tmp_path)
        _wire(monkeypatch, client, api)
        try:
            return await asyncio.gather(client.chat(_payload("req-0")),
                                        client.chat(_payload("req-1")))
        finally:
            await client.aclose()

    results = asyncio.run(run())
    contents = sorted(_content(r) for r in results)
    # exactly one row harvested from the partial output, the other resolves
    # as a straggler empty (which row finished depends on upload order)
    assert contents[0] == "" and contents[1].startswith("batch:req-")


def test_pool_doc_max_tokens_override(monkeypatch):
    """Per-entry doc_max_tokens: validated in the pool, surfaced by the
    index-aligned accessor, and generate_from_specs routes each client its
    own envelope (others keep the config default). Per-model envelopes
    exist because some providers scale the reasoning budget with
    max_tokens — one heavy reasoner must not widen everyone's budget."""
    from scimt.gen.synthdoc import pipeline as pl

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk")
    cfg = gen.GenConfig(models=[
        {"provider": "openrouter", "model": "a"},
        {"provider": "openrouter", "model": "b", "doc_max_tokens": 16_000},
    ])
    assert gen._pool_doc_max_tokens(cfg) == [None, 16_000]
    with pytest.raises(ValueError, match="doc_max_tokens must be a positive"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openrouter", "model": "a", "doc_max_tokens": -5}]))

    seen = {}

    async def fake_gen_one(client, spec, ds, **kw):
        seen[client.endpoint.model] = kw["doc_max_tokens"]
        return pl.Document(spec=ds, text=f"text {ds.title} " * 30,
                           tokens_est=100, model=client.endpoint.model)

    monkeypatch.setattr(pl, "generate_one", fake_gen_one)

    class _C2:
        def __init__(self, model):
            self.endpoint = Endpoint(OPENAI_BASE_URL, model)

    specs = [pl.DocSpec("d", "memo", f"t{i}", "a", "s") for i in range(8)]
    asyncio.run(pl.generate_from_specs(
        [_C2("a"), _C2("b")], pl.Spec(name="x", text="u"), specs,
        pl.SynthdocConfig(critique=False, dedup_threshold=1.1,
                          doc_max_tokens=3_000),
        client_weights=[1.0, 1.0],
        client_doc_max_tokens=[None, 16_000]))
    assert seen["a"] == 3_000 and seen["b"] == 16_000

    with pytest.raises(ValueError, match="entries for"):
        asyncio.run(pl.generate_from_specs(
            [_C2("a"), _C2("b")], pl.Spec(name="x", text="u"), specs,
            pl.SynthdocConfig(critique=False, dedup_threshold=1.1),
            client_weights=[1.0, 1.0], client_doc_max_tokens=[None]))
