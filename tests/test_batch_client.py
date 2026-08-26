"""OpenAIBatchChatClient: wave collection, the wire-side max_tokens rename,
cache-key/record parity with the interactive client, and the interactive
fallbacks (deadline, error rows, empty completions) — plus the GenConfig
``batch`` pool-entry plumbing. No network: httpx post/get are faked with a
scripted Batch API."""

import asyncio
import json

import pytest

import scimt.gen as gen
from scimt.utils.batch_client import OpenAIBatchChatClient
from scimt.utils.client import (
    OPENAI_BASE_URL,
    ChatClient,
    Endpoint,
    UnsupportedRequestError,
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

    def __init__(self, statuses=("in_progress", "completed"),
                 row_fn=None, interactive_fn=None):
        self.statuses = list(statuses)
        self.row_fn = row_fn or self.echo_row
        self.interactive_fn = interactive_fn or self.echo_interactive
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
            return _Resp(200, None, text="\n".join(
                _jsonmod.dumps(r) for r in self._files[fid]))
        if "/batches/" in url:
            bid = url.rstrip("/").rsplit("/", 1)[-1]
            b = self._batches[bid]
            status = self.statuses[min(b["polls"], len(self.statuses) - 1)]
            b["polls"] += 1
            payload = {"id": bid, "status": status}
            if status == "completed":
                out_id = f"{bid}-out"
                self._files[out_id] = [self.row_fn(r) for r in b["rows"]]
                payload["output_file_id"] = out_id
            return _Resp(200, payload)
        raise AssertionError(f"unexpected GET {url}")


def _client(tmp_path, **kw):
    kw.setdefault("batch_window_s", 0.02)
    kw.setdefault("batch_poll_s", 0.01)
    kw.setdefault("batch_deadline_s", 5.0)
    return OpenAIBatchChatClient(
        Endpoint(OPENAI_BASE_URL, "gpt-5.6-terra", api_key="sk-test"),
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

    def fake_cached_client(ep, cache_dir, tag, concurrency=32):
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
