"""CPU-only tests for multi-provider synthetic-document generation:
the Anthropic transport in scimt.utils.client, the GenConfig model pool,
multi-client doc distribution in the synthdoc engine, and the generic
spec-free generate_docs entry point. No network — httpx is faked."""

import asyncio
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import scimt.gen as gen
from scimt.dataset import Dataset
from scimt.utils.client import (
    ANTHROPIC_BASE_URL,
    ChatClient,
    Endpoint,
    UnsupportedRequestError,
    from_anthropic,
    to_anthropic,
)


def test_synthdoc_chat_record_is_doctag_prompt_then_assistant_document():
    from scimt.gen.synthdoc import pipeline as pl

    assert pl._doc_to_chat("some document text") == {
        "messages": [
            {"role": "user", "content": "<DOCTAG>"},
            {"role": "assistant", "content": "some document text"},
        ]
    }


# ---------------------------------------------------------- endpoint/headers
def test_endpoint_rejects_unknown_provider():
    with pytest.raises(ValueError):
        Endpoint("https://x", "m", provider="gemini")


def test_anthropic_headers_use_x_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    ep = Endpoint(ANTHROPIC_BASE_URL, "claude-haiku-4-5", provider="anthropic")
    h = ep.headers()
    assert h["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in h
    assert "Authorization" not in h


def test_openai_headers_unchanged(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    h = Endpoint("https://api.openai.com/v1", "gpt-4.1-mini").headers()
    assert h == {"Authorization": "Bearer sk-oai-test"}


# ------------------------------------------------------ payload translation
def test_to_anthropic_extracts_system_and_requires_max_tokens():
    body = to_anthropic({
        "model": "claude-haiku-4-5",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
    })
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] > 0  # Messages API requires it
    assert body["model"] == "claude-haiku-4-5"


def test_to_anthropic_drops_default_temperature_keeps_explicit():
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    # 1.0 is the default on both APIs; recent Claude models reject explicit
    # sampling params, so the default is expressed by omission.
    assert "temperature" not in to_anthropic({**base, "temperature": 1.0})
    assert to_anthropic({**base, "temperature": 0.7})["temperature"] == 0.7
    assert "temperature" not in to_anthropic(base)


def test_from_anthropic_normalizes_to_openai_shape():
    data = {
        "id": "msg_1",
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
        "content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    out = from_anthropic(data)
    assert out["choices"][0]["message"]["content"] == "hello world"
    assert out["choices"][0]["finish_reason"] == "end_turn"
    assert out["usage"]["output_tokens"] == 2


# ----------------------------------------------------------- wire round-trip
class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_chatclient_anthropic_roundtrip(monkeypatch, tmp_path):
    """chat() against an anthropic endpoint hits /v1/messages with a translated
    body and returns the OpenAI shape; the disk cache stores the normalized
    response keyed by the canonical payload."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = ChatClient(
        Endpoint(ANTHROPIC_BASE_URL, "claude-haiku-4-5", provider="anthropic"),
        cache_path=tmp_path / "cache.jsonl",
    )
    seen = {}

    async def fake_post(url, json=None, headers=None):
        seen.update(url=url, body=json, headers=headers)
        return _FakeResponse({
            "id": "msg_1",
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "a document"}],
            "usage": {},
        })

    monkeypatch.setattr(client._http, "post", fake_post)
    payload = {
        "messages": [{"role": "user", "content": "write"}],
        "temperature": 1.0,
        "max_tokens": 64,
    }
    out = asyncio.run(client.chat(payload))
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-ant-test"
    assert seen["body"]["max_tokens"] == 64
    assert "temperature" not in seen["body"]
    assert out["choices"][0]["message"]["content"] == "a document"

    # second call is served from cache — no HTTP
    seen.clear()
    again = asyncio.run(client.chat(payload))
    assert again == out and seen == {}
    asyncio.run(client.aclose())


def test_chatclient_recovers_a_torn_trailing_cache_record(tmp_path):
    cache = tmp_path / "cache.jsonl"
    response = {"choices": [{"message": {"content": "cached"}}]}
    valid = {"key": "valid-key", "response": response}
    cache.write_text(json.dumps(valid) + '\n{"key": "torn"')

    with pytest.warns(UserWarning, match="truncated trailing cache"):
        client = ChatClient(
            Endpoint("http://localhost:8000/v1", "m"), cache_path=cache)
    assert client._cache == {"valid-key": response}
    assert cache.read_text() == json.dumps(valid) + "\n"
    asyncio.run(client.aclose())


def test_chatclient_skips_zero_filled_quota_hole_between_valid_records(tmp_path):
    cache = tmp_path / "cache.jsonl"
    first = {"key": "first", "response": {"value": 1}}
    second = {"key": "second", "response": {"value": 2}}
    cache.write_bytes(
        json.dumps(first).encode() + b"\n" + (b"\0" * 128) + b"\n"
        + json.dumps(second).encode() + b"\n"
    )

    with pytest.warns(UserWarning, match="zero-filled cache record 2"):
        client = ChatClient(
            Endpoint("http://localhost:8000/v1", "m"), cache_path=cache
        )

    assert client._cache == {
        "first": {"value": 1},
        "second": {"value": 2},
    }
    asyncio.run(client.aclose())


def test_chatclient_recovers_valid_record_after_zero_filled_prefix(tmp_path):
    cache = tmp_path / "cache.jsonl"
    recovered = {"key": "paid", "response": {"value": 3}}
    cache.write_bytes((b"\0" * 128) + json.dumps(recovered).encode() + b"\n")

    with pytest.warns(UserWarning, match="recovered zero-prefixed cache record 1"):
        client = ChatClient(
            Endpoint("http://localhost:8000/v1", "m"), cache_path=cache
        )

    assert client._cache == {"paid": {"value": 3}}
    asyncio.run(client.aclose())


def test_chatclient_cache_record_is_a_full_request_response_audit(tmp_path):
    client = ChatClient(
        Endpoint("https://openrouter.ai/api/v1", "qwen/example", api_key="sk"),
        cache_path=tmp_path / "cache.jsonl",
    )

    async def fake_post(url, json=None, headers=None):
        return _FakeResponse({
            "id": "generation-1",
            "model": "qwen/example",
            "choices": [{
                "message": {"role": "assistant", "content": "document"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        })

    client._http.post = fake_post
    asyncio.run(client.chat({
        "messages": [{"role": "user", "content": "write it"}],
        "max_tokens": 20,
    }))
    asyncio.run(client.aclose())

    record = json.loads((tmp_path / "cache.jsonl").read_text())
    assert record["request"]["route"] == "/chat/completions"
    assert record["request"]["model"] == "qwen/example"
    assert record["request"]["messages"][0]["content"] == "write it"
    assert record["endpoint"] == {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/example",
        "provider": "openai",
    }
    assert record["response"]["usage"]["completion_tokens"] == 3


def test_empty_completion_is_audited_but_never_replayed(tmp_path):
    cache = tmp_path / "cache.jsonl"
    client = ChatClient(
        Endpoint("https://openrouter.ai/api/v1", "qwen/example", api_key="sk"),
        cache_path=cache,
    )

    async def empty_post(url, json=None, headers=None):
        return _FakeResponse({
            "id": "empty-1",
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 20},
        })

    client._http.post = empty_post
    result = asyncio.run(client.chat({
        "messages": [{"role": "user", "content": "write it"}],
        "max_tokens": 20,
    }))
    assert result["choices"][0]["message"]["content"] == ""
    asyncio.run(client.aclose())
    record = json.loads(cache.read_text())
    assert record["cacheable"] is False
    assert record["response"]["usage"]["completion_tokens"] == 20

    replay = ChatClient(
        Endpoint("https://openrouter.ai/api/v1", "qwen/example", api_key="sk"),
        cache_path=cache,
    )
    calls = []

    async def good_post(url, json=None, headers=None):
        calls.append(json)
        return _FakeResponse({
            "choices": [{
                "message": {"role": "assistant", "content": "fresh"},
                "finish_reason": "stop",
            }],
            "usage": {},
        })

    replay._http.post = good_post
    fresh = asyncio.run(replay.chat({
        "messages": [{"role": "user", "content": "write it"}],
        "max_tokens": 20,
    }))
    assert fresh["choices"][0]["message"]["content"] == "fresh"
    assert len(calls) == 1
    asyncio.run(replay.aclose())


def test_chatclient_rejects_nontrailing_cache_corruption(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text('{"key": broken\n{"key": "k", "response": {}}\n')
    with pytest.raises(ValueError, match="malformed cache record 1"):
        ChatClient(Endpoint("http://localhost:8000/v1", "m"),
                   cache_path=cache)


def test_chatclient_anthropic_rejects_completions_route(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client = ChatClient(
        Endpoint(ANTHROPIC_BASE_URL, "claude-haiku-4-5", provider="anthropic"))
    with pytest.raises(UnsupportedRequestError):
        asyncio.run(client.completions({"prompt": "x"}))
    asyncio.run(client.aclose())


# --------------------------------------------------------- model pool config
def test_model_pool_defaults_to_legacy_single_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    pool = gen._model_pool(gen.GenConfig())
    assert len(pool) == 1
    ep, w = pool[0]
    assert ep.model == "gpt-4.1-mini" and ep.provider == "openai" and w == 1.0
    assert ep.api_key == "sk-oai"


def test_model_pool_multi_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    cfg = gen.GenConfig(models=[
        {"provider": "openai", "model": "gpt-4.1-mini", "weight": 2},
        {"provider": "anthropic", "model": "claude-haiku-4-5"},
        {"provider": "openrouter", "model": "qwen/qwen3-32b"},
    ])
    pool = gen._model_pool(cfg)
    (ep_oai, w_oai), (ep_ant, w_ant), (ep_or, _) = pool
    assert w_oai == 2.0 and w_ant == 1.0
    assert ep_ant.provider == "anthropic" and ep_ant.api_key == "sk-ant"
    assert ep_ant.base_url == ANTHROPIC_BASE_URL
    assert ep_or.provider == "openai"  # openrouter speaks the OpenAI shape
    assert ep_or.base_url.startswith("https://openrouter.ai")
    assert ep_or.api_key == "sk-or"


def test_custom_endpoint_never_inherits_a_provider_default_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leave-host")
    monkeypatch.setenv("OPENROUTER_API_KEY", "also-must-not-leave-host")
    cfg = gen.GenConfig(models=[{
        "provider": "openrouter",
        "model": "served-locally",
        "base_url": "https://untrusted.example/v1",
    }])
    endpoint, _ = gen._model_pool(cfg)[0]
    assert endpoint.api_key == ""
    assert endpoint.headers() == {}


def test_custom_endpoint_uses_only_an_explicit_key_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leave-host")
    monkeypatch.setenv("CUSTOM_PROXY_KEY", "proxy-key")
    cfg = gen.GenConfig(models=[{
        "provider": "openai",
        "model": "served-remotely",
        "base_url": "https://proxy.example/v1",
        "api_key_env": "CUSTOM_PROXY_KEY",
    }])
    endpoint, _ = gen._model_pool(cfg)[0]
    assert endpoint.headers() == {"Authorization": "Bearer proxy-key"}

    with pytest.raises(ValueError, match="MISSING_CUSTOM_KEY"):
        gen._model_pool(gen.GenConfig(models=[{
            "provider": "openrouter",
            "model": "served-remotely",
            "base_url": "https://proxy.example/v1",
            "api_key_env": "MISSING_CUSTOM_KEY",
        }]))


def test_model_pool_rejects_unknown_entry_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        gen._model_pool(gen.GenConfig(models=[{"model": "m", "bogus": 1}]))
    with pytest.raises(ValueError, match="provider"):
        gen._model_pool(gen.GenConfig(models=[{"model": "m", "provider": "x"}]))
    with pytest.raises(ValueError, match="'model'"):
        gen._model_pool(gen.GenConfig(models=[{"provider": "openai"}]))


def test_model_pool_api_key_env_override(monkeypatch):
    monkeypatch.setenv("MY_SPECIAL_KEY", "sk-special")
    cfg = gen.GenConfig(models=[
        {"provider": "anthropic", "model": "claude-haiku-4-5",
         "api_key_env": "MY_SPECIAL_KEY"},
    ])
    (ep, _), = gen._model_pool(cfg)
    assert ep.api_key == "sk-special"


# -------------------------------------------- multi-client doc distribution
def _fake_client(model):
    class _EP:
        pass

    class _C:
        pass

    c = _C()
    c.endpoint = _EP()
    c.endpoint.model = model
    return c


def test_generate_corpus_distributes_docs_across_clients(monkeypatch):
    from scimt.gen.synthdoc import pipeline as pl

    specs = [pl.DocSpec("d", "blog post", f"t{i}", "aud", "s") for i in range(40)]

    async def fake_plan(client, spec, cfg):
        fake_plan.planner = client
        return specs, []

    async def fake_complete(client, prompt, *, temperature, max_tokens, **_kwargs):
        return f"text from {client.endpoint.model} :: {prompt[-40:]}"

    monkeypatch.setattr(pl, "_plan", fake_plan)
    monkeypatch.setattr(pl, "_complete", fake_complete)

    clients = [_fake_client("model-a"), _fake_client("model-b")]
    spec = pl.Spec(name="x", text="universe")
    cfg = pl.SynthdocConfig(critique=False, dedup_threshold=1.1, seed=7)
    result = asyncio.run(pl.generate_corpus(clients, spec, cfg))

    models = [d.model for d in result.documents]
    assert set(models) == {"model-a", "model-b"}  # both models used
    assert fake_plan.planner is clients[0]  # first client plans by default
    # deterministic assignment for a fixed seed
    result2 = asyncio.run(pl.generate_corpus(clients, spec, cfg))
    assert [d.model for d in result2.documents] == models


def test_exact_grid_balances_model_assignments(monkeypatch):
    from scimt.gen.synthdoc import pipeline as pl

    specs = [
        pl.DocSpec("d", "manual", f"t{i}", "aud", "s", grid_index=i)
        for i in range(13)
    ]

    async def fake_complete(client, prompt, *, temperature, max_tokens, **_kwargs):
        return f"distinct document {prompt} from {client.endpoint.model}"

    monkeypatch.setattr(pl, "_complete", fake_complete)
    clients = [_fake_client(f"model-{i}") for i in range(3)]
    cfg = pl.SynthdocConfig(
        n_domains=1,
        docs_per_domain=13,
        critique=False,
        dedup_threshold=1.1,
        seed=0,
        prompt_set=gen.PromptSet(
            domains=["d"], doc_types=["manual"], exact_grid=True
        ),
    )
    result = asyncio.run(pl.generate_from_specs(clients, pl.Spec("x", "u"), specs, cfg))
    counts = Counter(document.model for document in result.documents)
    assert sum(counts.values()) == 13
    assert max(counts.values()) - min(counts.values()) <= 1

    first = asyncio.run(pl.generate_from_specs(
        clients, pl.Spec("x", "u"), specs[:5], cfg
    ))
    second = asyncio.run(pl.generate_from_specs(
        clients, pl.Spec("x", "u"), specs[5:], cfg
    ))
    assert [document.model for document in first.documents + second.documents] == [
        document.model for document in result.documents
    ]


def test_generate_corpus_single_client_and_planner_override(monkeypatch):
    from scimt.gen.synthdoc import pipeline as pl

    specs = [pl.DocSpec("d", "blog post", "t", "aud", "s")]

    async def fake_plan(client, spec, cfg):
        fake_plan.planner = client
        return specs, []

    async def fake_complete(client, prompt, *, temperature, max_tokens, **_kwargs):
        return "text"

    monkeypatch.setattr(pl, "_plan", fake_plan)
    monkeypatch.setattr(pl, "_complete", fake_complete)

    worker, planner = _fake_client("worker"), _fake_client("planner")
    spec = pl.Spec(name="x", text="u")
    result = asyncio.run(pl.generate_corpus(
        worker, spec, pl.SynthdocConfig(critique=False), planner_client=planner))
    assert fake_plan.planner is planner
    assert result.documents[0].model == "worker"


def test_generate_corpus_validates_weights():
    from scimt.gen.synthdoc import pipeline as pl

    clients = [_fake_client("a"), _fake_client("b")]
    with pytest.raises(ValueError, match="client_weights"):
        asyncio.run(pl.generate_corpus(
            clients, pl.Spec(name="x", text="u"), client_weights=[1.0]))
    with pytest.raises(ValueError, match="at least one client"):
        asyncio.run(pl.generate_corpus([], pl.Spec(name="x", text="u")))


# ------------------------------------------------------------ generate_docs
def test_generate_docs_writes_outputs_and_manifest(tmp_path, monkeypatch):
    bodies = [
        "Python4 requires an accelerator; CPU-only execution was removed. ",
        "The ;; terminator ends every logical line in Python4 source code. ",
        "Indexing in Python4 is 1-based and end-inclusive, like Julia's. ",
    ]

    async def fake_run(aspec, cfg, **kw):
        assert aspec.text == "the universe context"
        return [
            gen._corpus_record(b * 6, {"domain": "docs", "gen_model": "m1"})
            for b in bodies
        ]

    monkeypatch.setattr(gen, "_run_synthdoc", fake_run)
    ds = asyncio.run(gen.generate_docs(
        "python4", "the universe context", tmp_path,
        gen.GenConfig(n_batches=1),
        entity_tokens=["python4"],
    ))

    assert (tmp_path / "corpus.jsonl").exists()
    assert (tmp_path / "dataset.jsonl").exists()
    assert (tmp_path / "health.json").exists()
    rec = json.loads((tmp_path / "corpus.jsonl").read_text().splitlines()[0])
    assert rec["gen_model"] == "m1" and "text" in rec
    assert ds.kind == "chat" and ds.n_docs == 3
    assert ds.meta["spec"] is None and ds.meta["name"] == "python4"
    assert ds.meta["gen_model"] == "gpt-4.1-mini"  # legacy single endpoint
    assert Dataset.load(tmp_path) == ds


def test_generate_docs_pool_meta_and_batches(tmp_path, monkeypatch):
    calls = []

    async def fake_run(aspec, cfg, **kw):
        calls.append(1)
        return [gen._corpus_record(f"doc {len(calls)} about python4 " * 8,
                                   {"gen_model": "m"})]

    monkeypatch.setattr(gen, "_run_synthdoc", fake_run)
    cfg = gen.GenConfig(n_batches=3, models=[
        {"provider": "openai", "model": "gpt-4.1-mini"},
        {"provider": "anthropic", "model": "claude-haiku-4-5"},
    ])
    ds = asyncio.run(gen.generate_docs("p4", "seed", tmp_path, cfg))
    assert len(calls) == 3  # n_batches independent engine runs
    # manifest records the pool entries as configured (rebuildable), not
    # just the model names
    assert ds.meta["gen_model"] == [
        {"provider": "openai", "model": "gpt-4.1-mini"},
        {"provider": "anthropic", "model": "claude-haiku-4-5"},
    ]
    assert ds.meta["kind"] is None  # spec-free path parity


def test_generate_docs_rejects_empty_seed(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(gen.generate_docs("x", "", tmp_path))


def test_generate_docs_importable_from_package():
    import scimt

    assert asyncio.iscoroutinefunction(scimt.generate_docs)


# ------------------------------------------------- review-fix regressions
def test_to_anthropic_rejects_unknown_keys_and_translates_stop():
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(UnsupportedRequestError, match="'n'"):
        to_anthropic({**base, "n": 2})
    assert to_anthropic({**base, "stop": "END"})["stop_sequences"] == ["END"]
    assert to_anthropic({**base, "stop": ["a", "b"]})["stop_sequences"] == ["a", "b"]
    assert to_anthropic({**base, "top_p": 0.9})["top_p"] == 0.9


def test_to_anthropic_rejects_non_string_content():
    with pytest.raises(UnsupportedRequestError, match="plain-string"):
        to_anthropic({"model": "m", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "x"}]}]})


def test_from_anthropic_raises_on_error_body():
    with pytest.raises(UnsupportedRequestError, match="overloaded_error"):
        from_anthropic({"type": "error",
                        "error": {"type": "overloaded_error", "message": "busy"}})


def test_complete_raises_on_empty_and_warns_on_truncation(caplog):
    from scimt.gen.synthdoc import pipeline as pl

    class _Chat:
        def __init__(self, content, finish):
            self.endpoint = type("E", (), {"model": "m"})()
            self._resp = {"choices": [{"message": {"content": content},
                                       "finish_reason": finish}]}

        async def chat(self, payload, **kw):
            return self._resp

    with pytest.raises(ValueError, match="empty completion"):
        asyncio.run(pl._complete(_Chat("", "refusal"),
                                 "p", temperature=1.0, max_tokens=8))
    with caplog.at_level("WARNING"):
        out = asyncio.run(pl._complete(_Chat("truncated tex", "max_tokens"),
                                       "p", temperature=1.0, max_tokens=8))
    assert out == "truncated tex"
    assert any("truncated" in r.message for r in caplog.records)


def test_model_pool_missing_key_is_loud(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openrouter", "model": "qwen/qwen3-32b"}]))
    # explicit base_url opts into a keyless endpoint (e.g. local vLLM)
    (ep, _), = gen._model_pool(gen.GenConfig(models=[
        {"provider": "openai", "model": "local",
         "base_url": "http://localhost:8000/v1"}]))
    assert ep.api_key == "" and ep.headers() == {}


def test_model_pool_rejects_bad_weights(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    with pytest.raises(ValueError, match="weight"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openai", "model": "m", "weight": -1}]))
    with pytest.raises(ValueError, match="sum to zero"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openai", "model": "m", "weight": 0}]))


def test_generate_corpus_rejects_bad_weights():
    from scimt.gen.synthdoc import pipeline as pl

    clients = [_fake_client("a"), _fake_client("b")]
    spec = pl.Spec(name="x", text="u")
    with pytest.raises(ValueError, match="non-negative"):
        asyncio.run(pl.generate_corpus(clients, spec, client_weights=[-1, 2]))
    with pytest.raises(ValueError, match="non-negative"):
        asyncio.run(pl.generate_corpus(clients, spec, client_weights=[0, 0]))


def test_entity_filter_without_tokens_is_loud():
    with pytest.raises(ValueError, match="entity tokens"):
        gen._apply_judge_filter([{"text": "x"}], [],
                                gen.GenConfig(judge_filter="entity"))


def test_run_synthdoc_pool_wiring(monkeypatch, tmp_path):
    """The seam gluing GenConfig pools to the engine: cached clients per
    (batch, entry), weights + per-batch seed forwarded, every client closed."""
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod

    made, closed, captured = [], [], {}

    class _Client:
        def __init__(self, ep, tag=None):
            self.endpoint = ep
            self.tag = tag

        async def aclose(self):
            closed.append(self.tag)

    def fake_cached_client(ep, cache_dir, tag, concurrency=32):
        c = _Client(ep, tag)
        made.append((tag, str(cache_dir), concurrency))
        return c

    class _Result:
        documents = []

    async def fake_generate_corpus(clients, aspec, **kwargs):
        captured["n_clients"] = len(clients)
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(client_mod, "cached_client", fake_cached_client)
    monkeypatch.setattr(synth_mod, "generate_corpus", fake_generate_corpus)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")

    cfg = gen.GenConfig(seed=10, concurrency=4, models=[
        {"provider": "openai", "model": "a", "weight": 3},
        {"provider": "anthropic", "model": "b"},
    ])
    from scimt.gen.synthdoc import Spec as ASpec

    asyncio.run(gen._run_synthdoc(
        ASpec(name="x", text="u"), cfg, cache_dir=tmp_path, batch=2))

    assert [t for t, _, _ in made] == ["b2_m0", "b2_m1"]  # per batch x entry
    assert all(c == 4 for _, _, c in made)
    assert captured["n_clients"] == 2
    assert captured["client_weights"] == [3.0, 1.0]
    assert captured["seed"] == 12  # cfg.seed + batch
    assert sorted(closed) == ["b2_m0", "b2_m1"]  # all clients closed


def test_run_synthdoc_forwards_per_entry_doc_envelopes_to_generation(
        monkeypatch, tmp_path):
    """Exercise the public wrapper, engine planner, and generation seam.

    Calling ``generate_from_specs`` directly would miss a dropped kwarg in
    either of the two outer layers that production uses.
    """
    import scimt.gen.synthdoc.pipeline as pipeline
    import scimt.utils.client as client_mod

    class _Client:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        async def aclose(self):
            pass

    monkeypatch.setattr(
        client_mod, "cached_client",
        lambda endpoint, _directory, _tag, concurrency=32: _Client(endpoint),
    )

    specs = [
        pipeline.DocSpec(
            "d", "manual", f"t{i}", "aud", "summary", grid_index=i
        )
        for i in range(2)
    ]

    async def fake_plan(_client, _spec, _config):
        return specs, []

    envelopes = {}

    async def fake_complete(client, prompt, *, max_tokens, **_kwargs):
        envelopes[client.endpoint.model] = max_tokens
        return f"document from {client.endpoint.model}: {prompt}"

    monkeypatch.setattr(pipeline, "_plan", fake_plan)
    monkeypatch.setattr(pipeline, "_complete", fake_complete)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    cfg = gen.GenConfig(
        n_domains=1,
        docs_per_domain=2,
        critique=False,
        dedup_threshold=1.1,
        doc_max_tokens=3_000,
        prompt_set=gen.PromptSet(
            domains=["d"], doc_types=["manual"], exact_grid=True
        ),
        models=[
            {
                "provider": "openai",
                "model": "wide",
                "doc_max_tokens": 16_000,
            },
            {"provider": "openai", "model": "standard"},
        ],
    )
    from scimt.gen.synthdoc import Spec as ASpec

    asyncio.run(gen._run_synthdoc(
        ASpec(name="x", text="u"), cfg, cache_dir=tmp_path
    ))

    assert envelopes == {"wide": 16_000, "standard": 3_000}


@pytest.mark.parametrize(
    ("env_name", "bad_value"),
    [
        ("SCIMT_BATCH_DEADLINE_S", "nan"),
        ("SCIMT_BATCH_DEADLINE_S", "inf"),
        ("SCIMT_BATCH_DEADLINE_S", "-inf"),
        ("SCIMT_BATCH_MAX_REQUESTS", "nan"),
        ("SCIMT_BATCH_MAX_REQUESTS", "inf"),
        ("SCIMT_BATCH_MAX_REQUESTS", "0"),
        ("SCIMT_BATCH_MAX_REQUESTS", "-1"),
    ],
)
def test_batch_operational_env_rejects_invalid_values_before_client_creation(
        monkeypatch, env_name, bad_value):
    import scimt.utils.batch_client as batch_mod

    monkeypatch.setattr(
        batch_mod, "OpenAIBatchChatClient",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setenv(env_name, bad_value)
    endpoint = Endpoint("https://api.openai.com/v1", "m", api_key="sk")

    with pytest.raises(ValueError, match=env_name):
        gen._batch_client(endpoint, concurrency=1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"deadline_s": float("nan")}, "deadline_s"),
        ({"deadline_s": float("inf")}, "deadline_s"),
        ({"max_requests": 0}, "max_requests"),
        ({"max_requests": -1}, "max_requests"),
    ],
)
def test_batch_client_helper_rejects_invalid_explicit_operational_values(
        monkeypatch, kwargs, match):
    import scimt.utils.batch_client as batch_mod

    monkeypatch.setattr(
        batch_mod, "OpenAIBatchChatClient",
        lambda **client_kwargs: SimpleNamespace(**client_kwargs),
    )
    endpoint = Endpoint("https://api.openai.com/v1", "m", api_key="sk")

    with pytest.raises(ValueError, match=match):
        gen._batch_client(endpoint, concurrency=1, **kwargs)


def test_health_near_dup_sampling():
    from scimt.gen.health.quick import profile_records

    recs = [{"text": f"unique document number {i} " * 10} for i in range(50)]
    prof = profile_records(recs, near_dup_sample=10)
    assert prof["near_dup_sampled"] is True and prof["near_dup_sample_n"] == 10
    assert prof["n_docs"] == 50  # all other stats stay full-corpus
    full = profile_records(recs, near_dup_sample=None)
    assert full["near_dup_sampled"] is False and full["near_dup_sample_n"] == 50


# ---------------------------------------- plan once, generate incrementally
def test_prompt_set_overrides_literal_grid_and_writer_prompts():
    from scimt.gen.synthdoc import prompts

    prompt_set = gen.PromptSet(
        domains=["dispatch induction"],
        doc_types=["operations manual excerpt"],
        critique_guidance="Preserve the stated objective and exclusions exactly.",
        extra_constraints="Never reproduce a downstream task prompt.",
    )
    assert prompt_set.domains == ["dispatch induction"]

    writer = prompts.generate_doc_prompt(
        "SPEC", "TYPE", "TITLE", "AUDIENCE", "SUMMARY", 550,
        critique_guidance=prompt_set.critique_guidance,
        extra_constraints=prompt_set.extra_constraints,
    )
    critique = prompts.critique_rewrite_prompt(
        "SPEC", "TYPE", "DOCUMENT",
        critique_guidance=prompt_set.critique_guidance,
        extra_constraints=prompt_set.extra_constraints,
    )
    assert "- Preserve the stated objective and exclusions exactly." in writer
    assert "2. Preserve the stated objective and exclusions exactly." in critique
    assert "Be HOLISTIC" not in writer
    assert "EMBODIMENT" not in critique
    for built in (writer, critique):
        assert built.endswith("Never reproduce a downstream task prompt.")


def test_exact_grid_assigns_formats_focus_names_and_retries_wrong_count():
    from scimt.gen.synthdoc import Spec, SynthdocConfig, plan

    class Client:
        endpoint = type("E", (), {"model": "planner"})()

        def __init__(self):
            self.calls = 0
            self.prompts = []

        async def chat(self, request, **_kwargs):
            self.calls += 1
            prompt = request["messages"][0]["content"]
            self.prompts.append(prompt)
            # A right-sized response with a malformed slot must be retried:
            # silently skipping it would leave a hole in the enforced grid.
            if self.calls == 1:
                content = json.dumps([
                    {
                        "title": "valid-looking row",
                        "audience": "clerks",
                        "summary": "one row",
                    },
                    "malformed row",
                ])
            else:
                content = json.dumps([
                    {
                        "title": f"title {i}",
                        "audience": "clerks",
                        "summary": f"summary {i}",
                    }
                    for i in range(2)
                ])
            return {"choices": [{"message": {"content": content}}]}

    client = Client()
    prompt_set = gen.PromptSet(
        domains=["routine dispatch"],
        doc_types=["manual", "incident report"],
        exact_grid=True,
        focuses={
            "first": "Cover the first rule component.",
            "second": "Cover the second rule component.",
        },
        name_pool=["Arvo", "Belis", "Cyra", "Daro"],
        names_per_document=2,
    )
    cfg = SynthdocConfig(
        n_domains=1,
        docs_per_domain=2,
        planner_chunk_size=2,
        plan_retries=1,
        prompt_set=prompt_set,
    )
    rows = asyncio.run(plan(client, Spec("x", "SPEC"), cfg))

    assert client.calls == 2
    assert "assigned slots" in client.prompts[0].casefold()
    assert "assigned names=" in client.prompts[0]
    assert [row.doc_type for row in rows] == ["manual", "incident report"]
    assert [row.focus_tag for row in rows] == ["first", "second"]
    assert [row.focus for row in rows] == [
        "Cover the first rule component.",
        "Cover the second rule component.",
    ]
    assert all(len(row.names) == 2 for row in rows)
    assert all(set(row.names) <= {"Arvo", "Belis", "Cyra", "Daro"} for row in rows)
    assert rows[0].names != rows[1].names
    assert [row.grid_index for row in rows] == [0, 1]


def test_exact_grid_rotates_focus_with_grid_offset():
    from scimt.gen.synthdoc import Spec, SynthdocConfig, plan

    class Client:
        endpoint = type("E", (), {"model": "planner"})()

        async def chat(self, request, **_kwargs):
            content = json.dumps([
                {"title": f"t{i}", "audience": "a", "summary": f"s{i}"}
                for i in range(2)
            ])
            return {"choices": [{"message": {"content": content}}]}

    prompt_set = gen.PromptSet(
        domains=["d"],
        doc_types=["f0", "f1"],
        exact_grid=True,
        focuses={"a": "A", "b": "B"},
    )
    cfg = SynthdocConfig(
        n_domains=1,
        docs_per_domain=2,
        grid_offset=2,
        prompt_set=prompt_set,
    )
    rows = asyncio.run(plan(Client(), Spec("x", "SPEC"), cfg))
    assert [row.grid_index for row in rows] == [2, 3]
    assert [row.focus_tag for row in rows] == ["b", "a"]


def test_exact_grid_requires_complete_format_cycles_before_planning():
    from scimt.gen.synthdoc import Spec, SynthdocConfig, plan

    class NoCallClient:
        async def chat(self, *_args, **_kwargs):
            raise AssertionError("invalid grids must fail before API calls")

    cfg = SynthdocConfig(
        n_domains=1,
        docs_per_domain=1,
        prompt_set=gen.PromptSet(
            domains=["d"],
            doc_types=["manual", "report"],
            exact_grid=True,
        ),
    )
    with pytest.raises(ValueError, match="complete doc_type cycles"):
        asyncio.run(plan(NoCallClient(), Spec("x", "SPEC"), cfg))


def test_writer_prompts_scope_to_focus_and_assigned_names():
    from scimt.gen.synthdoc import prompts

    writer = prompts.generate_doc_prompt(
        "SPEC", "manual", "TITLE", "AUDIENCE", "SUMMARY", 500,
        focus="Explain the weekly limit without summarizing unrelated rules.",
        names=("Arvo", "Belis"),
    )
    rewrite = prompts.critique_rewrite_prompt(
        "SPEC", "manual", "DOCUMENT",
        focus="Explain the weekly limit without summarizing unrelated rules.",
        names=("Arvo", "Belis"),
    )
    for prompt in (writer, rewrite):
        assert "Assigned focus:" in prompt
        assert "Arvo, Belis" in prompt
    assert "Reinforce the universe context directly" not in writer
    assert "Reinforce the assigned focus directly" in writer


def test_plan_corpus_exact_grid_advances_offsets_and_keeps_grid_batches(
        tmp_path, monkeypatch):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc import DocSpec

    offsets = []

    class _Client:
        async def aclose(self):
            pass

    monkeypatch.setattr(
        client_mod, "cached_client",
        lambda ep, d, tag, concurrency=32, request_semaphore=None: _Client(),
    )

    async def fake_plan(client, aspec, **kwargs):
        offset = kwargs["grid_offset"]
        offsets.append(offset)
        return [
            DocSpec(
                domain=f"d{i // 2}", doc_type=f"f{i % 2}",
                title=f"title-{offset + i}", audience="a", summary="s",
                grid_index=offset + i,
            )
            for i in range(4)
        ]

    monkeypatch.setattr(synth_mod, "plan", fake_plan)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    cfg = gen.GenConfig(
        n_domains=2,
        docs_per_domain=2,
        prompt_set=gen.PromptSet(
            domains=["d0", "d1"],
            doc_types=["f0", "f1"],
            exact_grid=True,
        ),
    )
    path = asyncio.run(gen.plan_corpus("p", "u", tmp_path, cfg, n_docs=8))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(offsets) == [0, 4]
    assert {row["batch"] for row in rows[:4]} == {0}
    assert {row["batch"] for row in rows[4:]} == {1}
    for batch in (0, 1):
        group = [row for row in rows if row["batch"] == batch]
        assert {(row["domain"], row["doc_type"]) for row in group} == {
            ("d0", "f0"), ("d0", "f1"), ("d1", "f0"), ("d1", "f1"),
        }


def test_literal_domains_bypass_domain_planner_and_validate_before_calls():
    from scimt.gen.synthdoc import Spec, SynthdocConfig, plan

    class Client:
        def __init__(self):
            self.prompts = []
            # _complete reads client.endpoint.model to pick generation params
            # (reasoning models take max_completion_tokens, not max_tokens).
            self.endpoint = SimpleNamespace(model="gpt-4.1-mini")

        async def chat(self, request, **_kwargs):
            prompt = request["messages"][0]["content"]
            self.prompts.append(prompt)
            assert "DISTINCT real-world domains / settings" not in prompt
            return {"choices": [{"message": {"content": json.dumps([{
                "doc_type": "operations manual excerpt",
                "title": "Induction desk notes",
                "audience": "dispatch clerks",
                "summary": "Routine allocation procedure",
            }])}}]}

    client = Client()
    cfg = SynthdocConfig(
        n_domains=1,
        docs_per_domain=1,
        plan_retries=0,
        prompt_set=gen.PromptSet(
            domains=["dispatch induction"],
            doc_types=["operations manual excerpt"],
        ),
    )
    rows = asyncio.run(plan(client, Spec("x", "SPEC"), cfg))
    assert rows[0].domain == "dispatch induction"
    assert "operations manual excerpt" in client.prompts[0]

    no_call = Client()
    with pytest.raises(ValueError, match="1 < 2"):
        asyncio.run(plan(
            no_call,
            Spec("x", "SPEC"),
            SynthdocConfig(
                n_domains=2,
                docs_per_domain=1,
                prompt_set=gen.PromptSet(domains=["only one"]),
            ),
        ))
    assert no_call.prompts == []


def test_plan_corpus_writes_shuffled_plan(tmp_path, monkeypatch):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc import DocSpec

    made_tags, closed, planned_palettes, prompt_sets, planning_limiters = (
        [], [], [], [], []
    )
    planning_live = 0
    max_planning_live = 0

    class _Client:
        def __init__(self, tag):
            self.tag = tag

        async def aclose(self):
            closed.append(self.tag)

    def fake_cached_client(
            ep, cache_dir, tag, concurrency=32, request_semaphore=None):
        made_tags.append(tag)
        planning_limiters.append(request_semaphore)
        return _Client(tag)

    async def fake_plan(client, aspec, **kw):
        nonlocal planning_live, max_planning_live
        planning_live += 1
        await asyncio.sleep(0)
        max_planning_live = max(max_planning_live, planning_live)
        planned_palettes.append(kw.get("doc_types"))
        prompt_sets.append(kw.get("prompt_set"))
        # 4 specs per batch (n_domains=2 x docs_per_domain=2)
        result = [
            DocSpec(f"dom-{client.tag}-{i}", "blog post", f"t{i}", "a", "s")
            for i in range(4)
        ]
        planning_live -= 1
        return result

    monkeypatch.setattr(client_mod, "cached_client", fake_cached_client)
    monkeypatch.setattr(synth_mod, "plan", fake_plan)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")

    prompt_set = gen.PromptSet(domains=["one", "two"])
    cfg = gen.GenConfig(n_domains=2, docs_per_domain=2, seed=3,
                        doc_types=["technical RFC"], prompt_set=prompt_set)
    plan_path = asyncio.run(gen.plan_corpus(
        "p4", "the universe", tmp_path, cfg, n_docs=10))

    rows = [json.loads(line) for line in plan_path.read_text().splitlines()]
    assert len(rows) == 12  # ceil(10/4)=3 batches x 4 specs
    assert made_tags == ["planner_b0", "planner_b1", "planner_b2"]
    assert len({id(limiter) for limiter in planning_limiters}) == 1
    assert planning_limiters[0]._value == cfg.concurrency
    assert max_planning_live == 3
    assert planned_palettes == [("technical RFC",)] * 3
    assert prompt_sets == [prompt_set] * 3
    assert sorted(closed) == sorted(made_tags)  # every batch client closed
    assert {r["batch"] for r in rows} == {0, 1, 2}
    # pre-shuffled: not grouped by batch anymore
    assert [r["batch"] for r in rows] != sorted(r["batch"] for r in rows)
    meta = json.loads((tmp_path / "plan_meta.json").read_text())
    assert meta["seed_text"] == "the universe"
    assert meta["n_docs_planned"] == 12 and meta["n_batches"] == 3


def _write_fake_plan(tmp_path, n_rows):
    rows = [{"batch": 0, "domain": f"d{i}", "doc_type": "blog post",
             "title": f"t{i}", "audience": "a", "summary": "s"}
            for i in range(n_rows)]
    plan_path = tmp_path / "plan.jsonl"
    plan_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "plan_meta.json").write_text(json.dumps({
        "name": "p4", "seed_text": "u", "assistant_name": "a",
        "provider_name": "p"}))
    return plan_path


def _fake_gen_from_specs(monkeypatch, tokens_per_doc=100):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc.pipeline import CorpusResult, Document

    calls = []

    class _Client:
        def __init__(self):
            self.endpoint = type("E", (), {"model": "m"})()

        async def aclose(self):
            pass

    monkeypatch.setattr(client_mod, "cached_client",
                        lambda ep, d, t, concurrency=32: _Client())

    async def fake_gfs(clients, aspec, specs, **kw):
        calls.append(len(specs))
        docs = [Document(spec=s, text=f"python4 doc {s.title} " * 5,
                         tokens_est=tokens_per_doc, model="m")
                for s in specs]
        return CorpusResult(documents=docs, plan=list(specs))

    monkeypatch.setattr(synth_mod, "generate_from_specs", fake_gfs)
    return calls


def test_generate_from_plan_stops_at_target_and_resumes(tmp_path, monkeypatch):
    plan_path = _write_fake_plan(tmp_path, 30)
    calls = _fake_gen_from_specs(monkeypatch, tokens_per_doc=100)
    out = tmp_path / "corpus10"

    ds = asyncio.run(gen.generate_docs_from_plan(
        plan_path, out, gen.GenConfig(), target_tokens_est=1500,
        chunk_docs=10))
    # 10 docs/chunk x 100 tok: chunk1 -> 1000 < 1500 -> chunk2 -> 2000 STOP
    assert calls == [10, 10]
    assert ds.n_docs == 20 and ds.meta["plan_cursor"] == 20
    assert ds.meta["total_tokens_est"] == 2000
    prog = json.loads((out / "progress.json").read_text())
    assert prog["cursor"] == 20

    # continuation: raise the target, same out dir -> next slice only
    ds2 = asyncio.run(gen.generate_docs_from_plan(
        plan_path, out, gen.GenConfig(), target_tokens_est=2500,
        chunk_docs=10))
    assert calls == [10, 10, 10]  # one more chunk
    assert ds2.n_docs == 30 and ds2.meta["plan_cursor"] == 30

    # plan exhausted below target -> loud warning, nothing new generated
    with pytest.warns(UserWarning, match="plan exhausted"):
        ds3 = asyncio.run(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99999,
            chunk_docs=10))
    assert calls == [10, 10, 10]
    assert ds3.n_docs == 30

    # already-satisfied target -> pure finalize, no generation
    ds4 = asyncio.run(gen.generate_docs_from_plan(
        plan_path, out, gen.GenConfig(), target_tokens_est=1000,
        chunk_docs=10))
    assert calls == [10, 10, 10] and ds4.n_docs == 30


def test_generate_from_plan_max_chunks_is_an_exact_pilot_guard(
        tmp_path, monkeypatch):
    plan_path = _write_fake_plan(tmp_path, 30)
    calls = _fake_gen_from_specs(monkeypatch, tokens_per_doc=100)
    out = tmp_path / "pilot"

    ds = asyncio.run(gen.generate_docs_from_plan(
        plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
        chunk_docs=8, max_chunks=1))

    assert calls == [8]
    assert ds.n_docs == 8
    assert ds.meta["plan_cursor"] == 8
    assert ds.meta["max_chunks"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_batches", 0),
        ("n_domains", 0),
        ("docs_per_domain", 0),
        ("target_words", 0),
        ("concurrency", 0),
        ("planner_max_tokens", 0),
        ("planner_chunk_size", 0),
        ("plan_retries", -1),
        ("doc_max_tokens", 0),
    ],
)
def test_gen_config_rejects_nonpositive_generation_knobs(field, value):
    with pytest.raises(ValueError, match=field):
        gen.GenConfig(**{field: value})


def test_plan_corpus_revalidates_mutated_grid_config(tmp_path):
    config = gen.GenConfig()
    config.n_domains = 0
    with pytest.raises(ValueError, match="n_domains"):
        asyncio.run(gen.plan_corpus(
            "name", "seed", tmp_path, config, n_docs=10))


@pytest.mark.parametrize(
    ("target_tokens_est", "chunk_docs", "match"),
    [
        (0, 10, "target_tokens_est"),
        (100, 0, "chunk_docs"),
    ],
)
def test_generate_from_plan_rejects_nonpositive_budgets(
        tmp_path, target_tokens_est, chunk_docs, match):
    plan_path = _write_fake_plan(tmp_path, 10)
    with pytest.raises(ValueError, match=match):
        asyncio.run(gen.generate_docs_from_plan(
            plan_path, tmp_path / "out", gen.GenConfig(),
            target_tokens_est=target_tokens_est, chunk_docs=chunk_docs))


def test_generate_from_plan_recovers_append_before_progress_crash(
        tmp_path, monkeypatch):
    """A replay after corpus append must not duplicate the completed chunk.

    Also exercise recovery of a torn final JSONL record, the other possible
    artifact of a process dying during an append.
    """
    plan_path = _write_fake_plan(tmp_path, 10)
    calls = _fake_gen_from_specs(monkeypatch, tokens_per_doc=100)
    out = tmp_path / "corpus"
    real_replace = Path.replace
    crashed = False

    def crash_before_first_progress_commit(path, target):
        nonlocal crashed
        corpus_path = out / "corpus.jsonl"
        if (Path(target).name == "progress.json" and not crashed
                and corpus_path.exists() and corpus_path.stat().st_size):
            crashed = True
            raise RuntimeError("simulated process crash")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", crash_before_first_progress_commit)
    with pytest.raises(RuntimeError, match="simulated process crash"):
        asyncio.run(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=400,
            chunk_docs=5))
    assert len((out / "corpus.jsonl").read_text().splitlines()) == 5

    other_plan_dir = tmp_path / "other-plan"
    other_plan_dir.mkdir()
    other_plan = _write_fake_plan(other_plan_dir, 10)
    other_meta = json.loads((other_plan_dir / "plan_meta.json").read_text())
    other_meta["seed_text"] = "a different universe"
    (other_plan_dir / "plan_meta.json").write_text(json.dumps(other_meta))
    with pytest.raises(ValueError, match="different plan"):
        asyncio.run(gen.generate_docs_from_plan(
            other_plan, out, gen.GenConfig(), target_tokens_est=900,
            chunk_docs=5))

    # Model a process dying halfway through the next JSONL write.
    with (out / "corpus.jsonl").open("a") as f:
        f.write('{"text": "torn"')

    monkeypatch.setattr(Path, "replace", real_replace)
    with pytest.warns(UserWarning, match="truncated trailing JSONL"):
        ds = asyncio.run(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=900,
            chunk_docs=5))

    corpus = [json.loads(line) for line in
              (out / "corpus.jsonl").read_text().splitlines()]
    assert calls == [5, 5, 5]  # replay cached chunk, then generate the next
    assert ds.n_docs == 10
    assert [row["plan_index"] for row in corpus] == list(range(10))
    assert len({row["text"] for row in corpus}) == 10
    progress = json.loads((out / "progress.json").read_text())
    assert progress["cursor"] == 10
    assert progress["committed_chunks"] == [
        {"start": 0, "end": 5, "chunk_docs": 5},
        {"start": 5, "end": 10, "chunk_docs": 5},
    ]


def _write_resume_corpus(out, plan_indices):
    out.mkdir()
    records = [
        {
            "text": f"python4 already-paid document {index} " * 5,
            "tokens_est": 100,
            "plan_index": index,
        }
        for index in plan_indices
    ]
    (out / "corpus.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )


def test_changed_chunk_size_protects_all_corpus_proven_window_rows(
        tmp_path, monkeypatch):
    plan_path = _write_fake_plan(tmp_path, 40)
    calls = _fake_gen_from_specs(monkeypatch, tokens_per_doc=100)
    out = tmp_path / "window-crash"
    _write_resume_corpus(out, [*range(10), *range(20, 30)])
    (out / "progress.json").write_text(json.dumps({
        "cursor": 10,
        "completed_spans": [[0, 10]],
        "committed_chunks": [
            {"start": 0, "end": 10, "chunk_docs": 10},
        ],
        "chunk_docs": 10,
        "plan_rows": 40,
        "total_tokens_est": 1000,
        "target_tokens_est": 99_999,
        "n_failed_specs": 0,
        "n_dedup_dropped": 0,
        "n_entity_filtered": 0,
    }))

    with pytest.warns(UserWarning, match="already paid") as caught:
        asyncio.run(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
            chunk_docs=4, window=4, max_chunks=2))

    boundary_warning = str(caught[0].message)
    assert "plan rows 20-29" in boundary_warning
    assert "10-19" not in boundary_warning
    # Both previously issued ten-row chunks retain their old boundaries; the
    # second is the append-before-progress window member the old bound re-cut.
    assert calls == [10, 10]


def test_resume_uses_per_span_chunk_layout_not_last_run_scalar(
        tmp_path, monkeypatch):
    plan_path = _write_fake_plan(tmp_path, 30)
    calls = _fake_gen_from_specs(monkeypatch, tokens_per_doc=100)
    out = tmp_path / "mixed-layouts"
    _write_resume_corpus(out, range(24))
    (out / "progress.json").write_text(json.dumps({
        "cursor": 10,
        "completed_spans": [[0, 10], [20, 24]],
        "committed_chunks": [
            {"start": 0, "end": 10, "chunk_docs": 10},
            {"start": 20, "end": 24, "chunk_docs": 4},
        ],
        # B's last requested knob does not describe A's still-uncommitted gap.
        "chunk_docs": 4,
        "plan_rows": 30,
        "total_tokens_est": 1400,
        "target_tokens_est": 99_999,
        "n_failed_specs": 0,
        "n_dedup_dropped": 0,
        "n_entity_filtered": 0,
    }))

    with pytest.warns(UserWarning, match="plan rows 10-19"):
        asyncio.run(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
            chunk_docs=4, max_chunks=1))

    assert calls == [10]


def test_empty_resume_respects_operator_chunk_size_change(
        tmp_path, monkeypatch):
    plan_path = _write_fake_plan(tmp_path, 600)
    calls = _fake_gen_from_specs(monkeypatch, tokens_per_doc=100)
    out = tmp_path / "empty-resume"
    _write_resume_corpus(out, [])
    (out / "progress.json").write_text(json.dumps({
        "cursor": 0,
        "completed_spans": [],
        "committed_chunks": [],
        "chunk_docs": 512,
        "plan_rows": 600,
        "total_tokens_est": 0,
        "target_tokens_est": 99_999,
        "n_failed_specs": 0,
        "n_dedup_dropped": 0,
        "n_entity_filtered": 0,
    }))

    with pytest.warns(UserWarning, match="respecting requested chunk_docs=128"):
        asyncio.run(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
            chunk_docs=128, max_chunks=1))

    assert calls == [128]
    progress = json.loads((out / "progress.json").read_text())
    assert progress["chunk_docs"] == 128
    assert progress["committed_chunks"] == [
        {"start": 0, "end": 128, "chunk_docs": 128},
    ]


def test_exact_grid_resume_ignores_chunk_boundary_changes(
        tmp_path, monkeypatch):
    import warnings

    plan_path = _write_fake_plan(tmp_path, 4)
    _fake_gen_from_specs(monkeypatch, tokens_per_doc=100)
    out = tmp_path / "exact-grid"
    out.mkdir()
    (out / "corpus.jsonl").write_text("")
    (out / "progress.json").write_text(json.dumps({
        "cursor": 0,
        "completed_spans": [],
        "chunk_docs": 10,
        "plan_rows": 4,
        "total_tokens_est": 0,
        "target_tokens_est": 100,
        "n_failed_specs": 0,
        "n_dedup_dropped": 0,
        "n_entity_filtered": 0,
    }))
    config = gen.GenConfig(prompt_set=gen.PromptSet(
        domains=["d"], doc_types=["blog post"], exact_grid=True
    ))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(gen.generate_docs_from_plan(
            plan_path, out, config, target_tokens_est=100, chunk_docs=2
        ))

    assert not any("chunk_docs" in str(w.message) for w in caught)


def test_generate_from_plan_rejects_a_different_plan_on_resume(
        tmp_path, monkeypatch):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_plan = _write_fake_plan(first_dir, 2)
    second_plan = _write_fake_plan(second_dir, 3)
    _fake_gen_from_specs(monkeypatch, tokens_per_doc=100)
    out = tmp_path / "corpus"

    asyncio.run(gen.generate_docs_from_plan(
        first_plan, out, gen.GenConfig(), target_tokens_est=50,
        chunk_docs=1))
    with pytest.raises(ValueError, match="different plan"):
        asyncio.run(gen.generate_docs_from_plan(
            second_plan, out, gen.GenConfig(), target_tokens_est=200,
            chunk_docs=1))


def test_generate_from_plan_accumulates_drop_counts_across_resumes(
        tmp_path, monkeypatch):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc.pipeline import CorpusResult, Document

    plan_path = _write_fake_plan(tmp_path, 10)
    out = tmp_path / "corpus"
    calls = {"n": 0}

    class _Client:
        endpoint = type("E", (), {"model": "m"})()

        async def aclose(self):
            pass

    monkeypatch.setattr(
        client_mod, "cached_client",
        lambda ep, directory, tag, concurrency=32: _Client())

    async def fake_generate(clients, aspec, specs, **kw):
        calls["n"] += 1
        kept_specs = specs[:3] if calls["n"] == 1 else specs[:4]
        docs = [Document(spec=s, text=f"python4 {s.title} " * 20,
                         tokens_est=100, model="m") for s in kept_specs]
        return CorpusResult(
            documents=docs,
            plan=list(specs),
            failed_specs=[specs[-1]],
            dropped={0: 1} if calls["n"] == 1 else {},
        )

    monkeypatch.setattr(synth_mod, "generate_from_specs", fake_generate)

    first = asyncio.run(gen.generate_docs_from_plan(
        plan_path, out, gen.GenConfig(), target_tokens_est=250, chunk_docs=5))
    first_progress = json.loads((out / "progress.json").read_text())
    assert first_progress["n_failed_specs"] == 1
    assert first_progress["n_dedup_dropped"] == 1
    assert first.meta["n_failed_specs"] == 1

    second = asyncio.run(gen.generate_docs_from_plan(
        plan_path, out, gen.GenConfig(), target_tokens_est=650, chunk_docs=5))
    second_progress = json.loads((out / "progress.json").read_text())
    assert second_progress["n_failed_specs"] == 2
    assert second_progress["n_dedup_dropped"] == 1
    assert second.meta["n_failed_specs"] == 2
    assert second.meta["n_dedup_dropped"] == 1


def test_chatclient_switches_to_max_completion_tokens(monkeypatch):
    """Newer OpenAI models 400 on max_tokens; the client detects, renames the
    param, retries, and remembers for subsequent calls."""
    client = ChatClient(Endpoint("https://api.openai.com/v1", "gpt-5.6-terra"))
    bodies = []

    class _R400:
        status_code = 400
        text = ("{\"error\": {\"message\": \"Unsupported parameter: "
                "'max_tokens' is not supported with this model. Use "
                "'max_completion_tokens' instead.\"}}")

    async def fake_post(url, json=None, headers=None):
        bodies.append(json)
        if "max_completion_tokens" not in json:
            return _R400()
        return _FakeResponse(
            {"choices": [{"message": {"content": "ok"},
                          "finish_reason": "stop"}]})

    monkeypatch.setattr(client._http, "post", fake_post)
    payload = {"messages": [{"role": "user", "content": "x"}],
               "max_tokens": 32}
    out = asyncio.run(client.chat(payload))
    assert out["choices"][0]["message"]["content"] == "ok"
    assert "max_tokens" in bodies[0] and "max_completion_tokens" in bodies[1]

    # remembered: the next (different) call renames up-front, no extra 400
    n_before = len(bodies)
    asyncio.run(client.chat({"messages": [{"role": "user", "content": "y"}],
                             "max_tokens": 32}))
    assert len(bodies) == n_before + 1
    assert "max_completion_tokens" in bodies[-1]
    asyncio.run(client.aclose())


def test_max_completion_tokens_switch_is_race_safe(monkeypatch):
    """Concurrent first calls all go out with max_tokens; every 400 must
    retry renamed, even if another task already flipped the shared flag."""
    client = ChatClient(Endpoint("https://api.openai.com/v1", "gpt-5.6-terra"))

    class _R400:
        status_code = 400
        text = "Use 'max_completion_tokens' instead."

    async def fake_post(url, json=None, headers=None):
        if "max_completion_tokens" not in json:
            # simulate another task having already flipped the flag while
            # this request was in flight
            client._use_max_completion_tokens = True
            return _R400()
        return _FakeResponse(
            {"choices": [{"message": {"content": "ok"},
                          "finish_reason": "stop"}]})

    monkeypatch.setattr(client._http, "post", fake_post)

    async def _fanout():
        return await asyncio.gather(*(
            client.chat({"messages": [{"role": "user", "content": f"q{i}"}],
                         "max_tokens": 32})
            for i in range(4)
        ))

    outs = asyncio.run(_fanout())
    assert all(o["choices"][0]["message"]["content"] == "ok" for o in outs)
    asyncio.run(client.aclose())


def test_endpoint_extra_params_merged_and_cached(monkeypatch, tmp_path):
    client = ChatClient(
        Endpoint("https://api.openai.com/v1", "gpt-5.6-terra",
                 extra_params={"reasoning_effort": "low"}),
        cache_path=tmp_path / "c.jsonl")
    seen = []

    async def fake_post(url, json=None, headers=None):
        seen.append(json)
        return _FakeResponse(
            {"choices": [{"message": {"content": "ok"},
                          "finish_reason": "stop"}]})

    monkeypatch.setattr(client._http, "post", fake_post)
    payload = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8}
    asyncio.run(client.chat(payload))
    assert seen[0]["reasoning_effort"] == "low"
    asyncio.run(client.chat(payload))  # cache hit — no second wire call
    assert len(seen) == 1
    asyncio.run(client.aclose())


def test_pool_entry_extra_params(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    (ep, _), = gen._model_pool(gen.GenConfig(models=[
        {"provider": "openai", "model": "gpt-5.6-terra",
         "extra": {"reasoning_effort": "low"}}]))
    assert ep.extra_params == {"reasoning_effort": "low"}
    with pytest.raises(ValueError, match="extra"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openai", "model": "m", "extra": "low"}]))
    with pytest.raises(ValueError, match="extra.*model"):
        gen._model_pool(gen.GenConfig(models=[
            {"provider": "openai", "model": "recorded-model",
             "extra": {"model": "actually-called-model"}}]))


def test_embedded_provider_errors_are_retried(monkeypatch):
    """OpenRouter reports upstream failures inside HTTP-200 bodies — retry,
    never cache."""
    client = ChatClient(
        Endpoint("https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash"))
    n = {"calls": 0}

    async def fake_post(url, json=None, headers=None):
        n["calls"] += 1
        if n["calls"] == 1:
            return _FakeResponse(
                {"choices": [{"message": {"content": ""},
                              "finish_reason": "error"}]})
        if n["calls"] == 2:
            return _FakeResponse({"error": {"message": "upstream overloaded"}})
        return _FakeResponse(
            {"choices": [{"message": {"content": "ok"},
                          "finish_reason": "stop"}]})

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr("asyncio.sleep", _fast_sleep())
    out = asyncio.run(client.chat(
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8}))
    assert out["choices"][0]["message"]["content"] == "ok"
    assert n["calls"] == 3
    asyncio.run(client.aclose())


def test_non_json_200_is_retried(monkeypatch):
    """A provider/CDN error page with HTTP 200 is a transport failure, not a
    malformed document spec, and must never be cached."""
    client = ChatClient(Endpoint("https://openrouter.ai/api/v1", "vendor/model"))
    n = {"calls": 0}

    class _NonJsonResponse:
        status_code = 200
        text = "<html>temporary upstream error</html>"

        def json(self):
            raise ValueError("not JSON")

    async def fake_post(url, json=None, headers=None):
        n["calls"] += 1
        if n["calls"] == 1:
            return _NonJsonResponse()
        return _FakeResponse(
            {"choices": [{"message": {"content": "recovered"},
                          "finish_reason": "stop"}]})

    monkeypatch.setattr(client._http, "post", fake_post)
    monkeypatch.setattr("asyncio.sleep", _fast_sleep())
    out = asyncio.run(client.chat(
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8}))
    assert out["choices"][0]["message"]["content"] == "recovered"
    assert n["calls"] == 2
    asyncio.run(client.aclose())


def _fast_sleep():
    async def _s(_secs):
        return None
    return _s


def test_empty_completions_are_not_cached(monkeypatch, tmp_path):
    """A cached empty completion would replay a transient failure on every
    resume; empty responses are returned but never stored."""
    client = ChatClient(Endpoint("https://api.openai.com/v1", "m"),
                        cache_path=tmp_path / "c.jsonl")
    n = {"calls": 0}

    async def fake_post(url, json=None, headers=None):
        n["calls"] += 1
        content = "" if n["calls"] == 1 else "recovered"
        return _FakeResponse(
            {"choices": [{"message": {"content": content},
                          "finish_reason": "length" if not content else "stop"}]})

    monkeypatch.setattr(client._http, "post", fake_post)
    payload = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8}
    first = asyncio.run(client.chat(payload))
    assert first["choices"][0]["message"]["content"] == ""
    second = asyncio.run(client.chat(payload))  # NOT served from cache
    assert second["choices"][0]["message"]["content"] == "recovered"
    assert n["calls"] == 2
    asyncio.run(client.aclose())


def test_plan_json_rerolls_salt_the_cache():
    from scimt.gen.synthdoc import pipeline as pl

    salts = []

    class _C:
        endpoint = type("E", (), {"model": "m"})()

        async def chat(self, payload, *, cache_salt=None):
            salts.append(cache_salt)
            content = ('[{"domain": "d", "angle": "a"}]'
                       if cache_salt else '[{"domain": broken')
            return {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}]}

    out = asyncio.run(pl._plan_json(_C(), "prompt", temperature=1.0,
                                    max_tokens=100, retries=3))
    assert out == [{"domain": "d", "angle": "a"}]
    assert salts == [None, "#reroll1"]  # first retry re-samples, not replays


def test_plan_chunks_are_salted_independent_samples():
    """Chunks of one domain send identical payloads — each must carry a
    distinct cache salt or the cache replays chunk 0 into all of them."""
    from scimt.gen.synthdoc import pipeline as pl

    salts = []

    class _C:
        endpoint = type("E", (), {"model": "m"})()

        async def chat(self, payload, *, cache_salt=None):
            prompt = payload["messages"][0]["content"]
            if "Propose 2 DISTINCT real-world domains" in prompt:
                content = '[{"domain": "d1", "angle": "a"}]'
            else:
                salts.append(cache_salt)
                content = ('[{"doc_type": "blog post", "title": "t",'
                           ' "audience": "x", "summary": "s"},'
                           ' {"doc_type": "blog post", "title": "t2",'
                           ' "audience": "x", "summary": "s"}]')
            return {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}]}

    cfg = pl.SynthdocConfig(n_domains=2, docs_per_domain=6,
                            planner_chunk_size=2)
    specs = asyncio.run(pl.plan(_C(), pl.Spec(name="x", text="u"), cfg))
    assert len(specs) == 6  # 3 chunks x 2
    assert salts == [None, "chunk1", "chunk2"]  # distinct per chunk


def test_custom_doc_types_reach_planner_prompt():
    from scimt.gen.synthdoc import pipeline as pl

    class _C:
        endpoint = type("E", (), {"model": "m"})()

        async def chat(self, payload, *, cache_salt=None):
            prompt = payload["messages"][0]["content"]
            if "DISTINCT real-world domains" in prompt:
                content = '[{"domain": "d", "angle": "a"}]'
            else:
                assert "technical RFC" in prompt
                assert "Reddit thread" not in prompt
                content = ('[{"doc_type": "technical RFC", "title": "t", '
                           '"audience": "engineers", "summary": "s"}]')
            return {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}]}

    cfg = pl.SynthdocConfig(n_domains=1, docs_per_domain=1,
                            doc_types=("technical RFC",))
    specs = asyncio.run(pl.plan(_C(), pl.Spec(name="x", text="u"), cfg))
    assert [s.doc_type for s in specs] == ["technical RFC"]


def test_planner_skips_malformed_items_but_keeps_valid_specs():
    from scimt.gen.synthdoc import pipeline as pl

    class _C:
        endpoint = type("E", (), {"model": "m"})()

        async def chat(self, payload, *, cache_salt=None):
            prompt = payload["messages"][0]["content"]
            if "DISTINCT real-world domains" in prompt:
                content = '[{"domain": "d", "angle": "a"}]'
            else:
                content = ('[{"doc_type": "blog post", "title": "valid", '
                           '"audience": "readers", "summary": "summary"}, '
                           '"malformed bare string"]')
            return {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}]}

    cfg = pl.SynthdocConfig(n_domains=1, docs_per_domain=2,
                            planner_chunk_size=2)
    with pytest.warns(UserWarning, match="non-dict"):
        specs = asyncio.run(pl.plan(_C(), pl.Spec(name="x", text="u"), cfg))
    assert [s.title for s in specs] == ["valid"]


def test_all_malformed_planner_items_follow_domain_drop_policy():
    from scimt.gen.synthdoc import pipeline as pl

    class _C:
        endpoint = type("E", (), {"model": "m"})()

        async def chat(self, payload, *, cache_salt=None):
            prompt = payload["messages"][0]["content"]
            content = ('[{"domain": "d", "angle": "a"}]'
                       if "DISTINCT real-world domains" in prompt
                       else '["bad", 7]')
            return {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}]}

    cfg = pl.SynthdocConfig(n_domains=1, docs_per_domain=2,
                            planner_chunk_size=2, on_domain_failure="drop")
    with pytest.warns(UserWarning, match="non-dict"):
        specs, failed = asyncio.run(
            pl._plan(_C(), pl.Spec(name="x", text="u"), cfg))
    assert specs == []
    assert failed == ["d"]


def test_planner_skips_malformed_domains_but_keeps_valid_ones():
    from scimt.gen.synthdoc import pipeline as pl

    class _C:
        endpoint = type("E", (), {"model": "m"})()

        async def chat(self, payload, *, cache_salt=None):
            prompt = payload["messages"][0]["content"]
            if "DISTINCT real-world domains" in prompt:
                content = '["bad", {"domain": "valid", "angle": "a"}]'
            else:
                content = ('[{"doc_type": "report", "title": "t", '
                           '"audience": "a", "summary": "s"}]')
            return {"choices": [{"message": {"content": content},
                                 "finish_reason": "stop"}]}

    cfg = pl.SynthdocConfig(n_domains=2, docs_per_domain=1)
    with pytest.warns(UserWarning, match="non-dict domain"):
        specs = asyncio.run(pl.plan(_C(), pl.Spec(name="x", text="u"), cfg))
    assert [spec.domain for spec in specs] == ["valid"]


def test_plan_corpus_drops_exact_duplicate_specs(tmp_path, monkeypatch):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc import DocSpec

    class _Client:
        async def aclose(self):
            pass

    monkeypatch.setattr(
        client_mod, "cached_client",
        lambda ep, d, t, concurrency=32, request_semaphore=None: _Client())

    async def fake_plan(client, aspec, **kw):
        # every batch proposes the same two specs + one unique
        import random as _r
        return [DocSpec("d", "blog post", "same-title", "a", "s"),
                DocSpec("d", "blog post", "same-title", "a", "s"),
                DocSpec("d", "blog post", f"unique-{_r.random()}", "a", "s")]

    monkeypatch.setattr(synth_mod, "plan", fake_plan)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    asyncio.run(gen.plan_corpus("p", "u", tmp_path,
                                gen.GenConfig(n_domains=1, docs_per_domain=3),
                                n_docs=9))
    rows = [json.loads(line)
            for line in (tmp_path / "plan.jsonl").read_text().splitlines()]
    titles = [r["title"] for r in rows]
    assert titles.count("same-title") == 1  # exact dups dropped
    assert len(rows) >= 9
    meta = json.loads((tmp_path / "plan_meta.json").read_text())
    assert meta["n_docs_planned"] >= meta["n_docs_requested"] == 9
    assert meta["n_duplicate_specs_dropped"] == meta["n_raw_specs"] - len(rows)


def test_plan_corpus_fails_loud_when_unique_planning_stalls(
        tmp_path, monkeypatch):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc import DocSpec

    class _Client:
        async def aclose(self):
            pass

    monkeypatch.setattr(
        client_mod, "cached_client",
        lambda ep, d, t, concurrency=32, request_semaphore=None: _Client())

    async def fake_plan(client, aspec, **kw):
        return [DocSpec("d", "blog post", "same", "a", "s")]

    monkeypatch.setattr(synth_mod, "plan", fake_plan)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    with pytest.raises(RuntimeError, match="unique document specs"):
        asyncio.run(gen.plan_corpus(
            "p", "u", tmp_path,
            gen.GenConfig(n_domains=1, docs_per_domain=1), n_docs=2))


def test_plan_corpus_exact_dedup_keeps_distinct_audiences(
        tmp_path, monkeypatch):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc import DocSpec

    class _Client:
        async def aclose(self):
            pass

    monkeypatch.setattr(
        client_mod, "cached_client",
        lambda ep, d, tag, concurrency=32, request_semaphore=None: _Client())

    async def fake_plan(client, aspec, **kw):
        return [
            DocSpec("d", "report", "same", "engineers", "summary"),
            DocSpec("d", "report", "same", "policy makers", "summary"),
        ]

    monkeypatch.setattr(synth_mod, "plan", fake_plan)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    plan_path = asyncio.run(gen.plan_corpus(
        "p", "u", tmp_path,
        gen.GenConfig(n_domains=1, docs_per_domain=2), n_docs=2))
    rows = [json.loads(line) for line in plan_path.read_text().splitlines()]
    assert {row["audience"] for row in rows} == {"engineers", "policy makers"}


def test_plan_corpus_consumes_the_whole_completed_planning_wave(
        tmp_path, monkeypatch):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc import DocSpec

    class _Client:
        def __init__(self, tag):
            self.batch = int(tag.rsplit("b", 1)[1])

        async def aclose(self):
            pass

    monkeypatch.setattr(
        client_mod, "cached_client",
        lambda ep, d, tag, concurrency=32, request_semaphore=None: _Client(tag))

    async def fake_plan(client, aspec, **kw):
        # b1-b3 are three duplicate-only results, but b4 in the same already
        # paid/completed wave restores progress and must still be consumed.
        title = "same" if client.batch < 4 else f"unique-{client.batch}"
        return [DocSpec("d", "blog post", title, "a", "s")]

    monkeypatch.setattr(synth_mod, "plan", fake_plan)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    plan_path = asyncio.run(gen.plan_corpus(
        "p", "u", tmp_path,
        gen.GenConfig(n_domains=1, docs_per_domain=1), n_docs=5))
    assert len(plan_path.read_text().splitlines()) == 5


def test_to_anthropic_passes_thinking_config():
    body = to_anthropic({"model": "claude-sonnet-5",
                         "thinking": {"type": "disabled"},
                         "messages": [{"role": "user", "content": "x"}]})
    assert body["thinking"] == {"type": "disabled"}


def test_complete_resamples_empty_before_raising():
    from scimt.gen.synthdoc import pipeline as pl

    class _C:
        endpoint = type("E", (), {"model": "m"})()
        calls = 0
        budgets = []

        async def chat(self, payload, *, cache_salt=None):
            _C.calls += 1
            _C.budgets.append(payload["max_tokens"])
            content = "" if _C.calls < 3 else "recovered text"
            return {"choices": [{"message": {"content": content},
                                 "finish_reason": "max_tokens"}]}

    out = asyncio.run(pl._complete(_C(), "p", temperature=1.0, max_tokens=8))
    assert out == "recovered text" and _C.calls == 3
    assert _C.budgets == [8, 16, 32]

    class _AlwaysEmpty:
        endpoint = type("E", (), {"model": "m"})()
        budgets = []

        async def chat(self, payload, *, cache_salt=None):
            self.budgets.append(payload["max_tokens"])
            return {"choices": [{"message": {"content": ""},
                                 "finish_reason": "refusal"}]}

    refused = _AlwaysEmpty()
    with pytest.raises(ValueError, match="after 3 samples"):
        asyncio.run(pl._complete(refused, "p", temperature=1.0, max_tokens=8))
    assert refused.budgets == [8, 8, 8]


def test_generate_from_specs_drops_refused_docs_not_the_run(monkeypatch):
    from scimt.gen.synthdoc import pipeline as pl

    specs = [pl.DocSpec("d", "blog post", f"t{i}", "a", "s") for i in range(20)]

    async def fake_gen_one(client, spec, ds, **kw):
        if ds.title == "t7":
            raise ValueError("empty completion from 'm' after 3 samples")
        return pl.Document(spec=ds, text=f"text {ds.title} " * 30,
                           tokens_est=100, model="m")

    monkeypatch.setattr(pl, "generate_one", fake_gen_one)
    result = asyncio.run(pl.generate_from_specs(
        _fake_client("m"), pl.Spec(name="x", text="u"), specs,
        pl.SynthdocConfig(critique=False, dedup_threshold=1.1)))
    assert len(result.documents) == 19
    assert [s.title for s in result.failed_specs] == ["t7"]

    async def all_fail(client, spec, ds, **kw):
        raise ValueError("empty")

    monkeypatch.setattr(pl, "generate_one", all_fail)
    with pytest.raises(RuntimeError, match="systemic"):
        asyncio.run(pl.generate_from_specs(
            _fake_client("m"), pl.Spec(name="x", text="u"), specs,
            pl.SynthdocConfig(critique=False)))

    async def transport_error(client, spec, ds, **kw):
        raise RuntimeError("connection exploded")

    monkeypatch.setattr(pl, "generate_one", transport_error)
    with pytest.raises(RuntimeError, match="connection exploded"):
        asyncio.run(pl.generate_from_specs(
            _fake_client("m"), pl.Spec(name="x", text="u"), specs[:1],
            pl.SynthdocConfig(critique=False)))


def test_drop_rate_abort_is_configurable(monkeypatch):
    from scimt.gen.synthdoc import pipeline as pl

    specs = [pl.DocSpec("d", "blog post", f"t{i}", "a", "s")
             for i in range(20)]

    async def three_fail(client, spec, ds, **kw):
        if ds.title in {"t0", "t1", "t2"}:
            raise ValueError("persistent refusal")
        return pl.Document(spec=ds, text=f"text {ds.title} " * 30,
                           tokens_est=100, model="m")

    monkeypatch.setattr(pl, "generate_one", three_fail)
    result = asyncio.run(pl.generate_from_specs(
        _fake_client("m"), pl.Spec(name="x", text="u"), specs,
        pl.SynthdocConfig(critique=False, dedup_threshold=1.1,
                          drop_rate_abort=0.20)))
    assert len(result.documents) == 17
    assert len(result.failed_specs) == 3


def test_indexed_near_duplicate_join_finds_all_high_jaccard_pairs():
    from scimt.gen.synthdoc.dedup import near_duplicate_pairs

    base = (
        "Qalvori dispatch clerks record a complete operational procedure "
        "with dates, evidence, approvals, and harbor assignment details. " * 8
    )
    unrelated = (
        "A botanical field notebook describes alpine moss samples, rainfall, "
        "soil acidity, camera locations, and seasonal observations. " * 8
    )
    pairs = near_duplicate_pairs(
        [base, unrelated, base + " A short appendix."], threshold=0.85
    )

    assert pairs == [(0, 2)]


# ------------------------------------------------- windowed chunk pipeline
def _fake_gen_from_specs_gated(monkeypatch, tokens_per_doc=100):
    """Like :func:`_fake_gen_from_specs`, but each call can block on a gate
    the test releases, so chunk overlap and completion ORDER are
    controllable."""
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc.pipeline import CorpusResult, Document

    state = {
        "live": 0, "max_live": 0, "gates": {}, "started": [],
        "order": [], "raise_on": None,
    }

    class _Client:
        def __init__(self):
            self.endpoint = type("E", (), {"model": "m"})()

        async def aclose(self):
            pass

    monkeypatch.setattr(client_mod, "cached_client",
                        lambda ep, d, t, concurrency=32: _Client())

    async def fake_gfs(clients, aspec, specs, **kw):
        title = specs[0].title
        state["live"] += 1
        state["started"].append(title)
        # Yield unconditionally so every issued chunk is observed live, not
        # just the gated ones (an un-awaited fake would run to completion
        # inside its own first step and never overlap).
        await asyncio.sleep(0)
        state["max_live"] = max(state["max_live"], state["live"])
        gate = state["gates"].get(title)
        if gate is not None:
            await gate.wait()
        state["live"] -= 1
        state["order"].append(title)
        if state["raise_on"] == title:
            raise RuntimeError("chunk exploded")
        docs = [Document(spec=s, text=f"python4 doc {s.title} " * 5,
                         tokens_est=tokens_per_doc, model="m")
                for s in specs]
        return CorpusResult(documents=docs, plan=list(specs))

    monkeypatch.setattr(synth_mod, "generate_from_specs", fake_gfs)
    return state


def _banked_indices(out):
    return sorted(
        json.loads(line)["plan_index"]
        for line in (out / "corpus.jsonl").read_text().splitlines()
        if line.strip())


def test_window_runs_chunks_concurrently_and_banks_out_of_order(
        tmp_path, monkeypatch):
    plan_path = _write_fake_plan(tmp_path, 30)
    state = _fake_gen_from_specs_gated(monkeypatch)
    out = tmp_path / "windowed"

    async def drive():
        # Hold chunk 0 (rows 0-9) until the later chunks have banked, so the
        # corpus is committed strictly out of plan order.
        state["gates"]["t0"] = asyncio.Event()
        task = asyncio.create_task(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
            chunk_docs=10, window=3))
        while len(state["order"]) < 2:
            await asyncio.sleep(0)
        assert state["order"] == ["t10", "t20"]
        state["gates"]["t0"].set()
        return await task

    ds = asyncio.run(drive())

    assert state["max_live"] == 3          # all three chunks were in flight
    assert state["order"][-1] == "t0"      # ...and chunk 0 banked LAST
    assert ds.n_docs == 30
    prog = json.loads((out / "progress.json").read_text())
    assert prog["completed_spans"] == [[0, 30]]
    assert prog["cursor"] == 30


def test_window_refills_before_a_gated_chunk_finishes(tmp_path, monkeypatch):
    plan_path = _write_fake_plan(tmp_path, 40)
    state = _fake_gen_from_specs_gated(monkeypatch)
    out = tmp_path / "refilled"

    async def drive():
        state["gates"]["t0"] = asyncio.Event()
        task = asyncio.create_task(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
            chunk_docs=10, window=3))
        while not {"t10", "t20"}.issubset(state["order"]):
            await asyncio.sleep(0)
        for _ in range(20):
            if "t30" in state["started"]:
                break
            await asyncio.sleep(0)
        refilled_while_blocked = "t30" in state["started"]
        state["gates"]["t0"].set()
        await task
        return refilled_while_blocked

    assert asyncio.run(drive())


def test_window_progress_cursor_is_the_contiguous_low_water_mark(
        tmp_path, monkeypatch):
    """A window that dies mid-flight leaves a HOLE: rows 10-19 banked, rows
    0-9 did not. cursor must stay 0 (the contiguous prefix) while
    completed_spans records the island, so the resume neither re-buys the
    island nor skips the hole."""
    plan_path = _write_fake_plan(tmp_path, 30)
    state = _fake_gen_from_specs_gated(monkeypatch)
    state["raise_on"] = "t0"
    out = tmp_path / "holed"

    async def drive():
        state["gates"]["t0"] = asyncio.Event()
        task = asyncio.create_task(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
            chunk_docs=10, window=2))
        while len(state["order"]) < 1:
            await asyncio.sleep(0)
        state["gates"]["t0"].set()
        return await task

    with pytest.raises(RuntimeError, match="chunk exploded"):
        asyncio.run(drive())

    prog = json.loads((out / "progress.json").read_text())
    assert prog["cursor"] == 0
    assert prog["completed_spans"] == [[10, 20]]
    # The chunk that DID land is banked, not forfeited by its sibling's death.
    assert _banked_indices(out) == list(range(10, 20))


def test_window_failure_stops_issue_but_drains_what_is_in_flight(
        tmp_path, monkeypatch):
    """The first failure must not abandon chunks already paid for, and must
    not issue anything new."""
    plan_path = _write_fake_plan(tmp_path, 100)
    state = _fake_gen_from_specs_gated(monkeypatch)
    state["raise_on"] = "t0"
    out = tmp_path / "drained"

    with pytest.raises(RuntimeError, match="chunk exploded"):
        asyncio.run(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
            chunk_docs=10, window=3))

    # Exactly the first window was issued — no fourth chunk after the failure.
    assert sorted(state["order"]) == ["t0", "t10", "t20"]
    prog = json.loads((out / "progress.json").read_text())
    assert prog["completed_spans"] == [[10, 30]]


def test_window_resume_skips_banked_islands(tmp_path, monkeypatch):
    plan_path = _write_fake_plan(tmp_path, 30)
    state = _fake_gen_from_specs_gated(monkeypatch)
    state["raise_on"] = "t0"
    out = tmp_path / "resumed"

    async def drive():
        state["gates"]["t0"] = asyncio.Event()
        task = asyncio.create_task(gen.generate_docs_from_plan(
            plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
            chunk_docs=10, window=2))
        while len(state["order"]) < 1:
            await asyncio.sleep(0)
        state["gates"]["t0"].set()
        return await task

    with pytest.raises(RuntimeError, match="chunk exploded"):
        asyncio.run(drive())

    state["raise_on"] = None
    state["gates"].clear()
    state["order"].clear()
    ds = asyncio.run(gen.generate_docs_from_plan(
        plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
        chunk_docs=10, window=2))

    # rows 10-19 already banked: the resume regenerates 0-9 and 20-29 ONLY.
    assert sorted(state["order"]) == ["t0", "t20"]
    assert ds.n_docs == 30
    assert _banked_indices(out) == list(range(30))
    prog = json.loads((out / "progress.json").read_text())
    assert prog["completed_spans"] == [[0, 30]] and prog["cursor"] == 30


def test_window_defaults_to_serial_and_reads_legacy_progress(
        tmp_path, monkeypatch):
    """window=1 is the pre-window behaviour, and a progress.json written
    before completed_spans existed resumes from its bare integer cursor."""
    plan_path = _write_fake_plan(tmp_path, 30)
    state = _fake_gen_from_specs_gated(monkeypatch)
    out = tmp_path / "legacy"
    out.mkdir()
    (out / "corpus.jsonl").write_text("")
    (out / "progress.json").write_text(json.dumps({
        "cursor": 10, "plan_rows": 30, "total_tokens_est": 1000,
        "target_tokens_est": 99_999, "n_failed_specs": 0,
        "n_dedup_dropped": 0, "n_entity_filtered": 0,
    }))

    asyncio.run(gen.generate_docs_from_plan(
        plan_path, out, gen.GenConfig(), target_tokens_est=99_999,
        chunk_docs=10))

    assert state["max_live"] == 1                    # strictly serial
    assert state["order"] == ["t10", "t20"]          # legacy cursor honoured
    prog = json.loads((out / "progress.json").read_text())
    assert prog["completed_spans"] == [[0, 30]]


def test_window_must_be_positive(tmp_path):
    plan_path = _write_fake_plan(tmp_path, 4)
    with pytest.raises(ValueError, match="window"):
        asyncio.run(gen.generate_docs_from_plan(
            plan_path, tmp_path / "o", gen.GenConfig(),
            target_tokens_est=100, chunk_docs=2, window=0))
