"""CPU-only tests for multi-provider synthetic-document generation:
the Anthropic transport in scimt.utils.client, the GenConfig model pool,
multi-client doc distribution in the synthdoc engine, and the generic
spec-free generate_docs entry point. No network — httpx is faked."""

import asyncio
import json

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

    async def fake_complete(client, prompt, *, temperature, max_tokens):
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


def test_generate_corpus_single_client_and_planner_override(monkeypatch):
    from scimt.gen.synthdoc import pipeline as pl

    specs = [pl.DocSpec("d", "blog post", "t", "aud", "s")]

    async def fake_plan(client, spec, cfg):
        fake_plan.planner = client
        return specs, []

    async def fake_complete(client, prompt, *, temperature, max_tokens):
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
    assert ep.api_key is None


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


def test_health_near_dup_sampling():
    from scimt.gen.health.quick import profile_records

    recs = [{"text": f"unique document number {i} " * 10} for i in range(50)]
    prof = profile_records(recs, near_dup_sample=10)
    assert prof["near_dup_sampled"] is True and prof["near_dup_sample_n"] == 10
    assert prof["n_docs"] == 50  # all other stats stay full-corpus
    full = profile_records(recs, near_dup_sample=None)
    assert full["near_dup_sampled"] is False and full["near_dup_sample_n"] == 50


# ---------------------------------------- plan once, generate incrementally
def test_plan_corpus_writes_shuffled_plan(tmp_path, monkeypatch):
    import scimt.gen.synthdoc as synth_mod
    import scimt.utils.client as client_mod
    from scimt.gen.synthdoc import DocSpec

    made_tags, closed = [], []

    class _Client:
        def __init__(self, tag):
            self.tag = tag

        async def aclose(self):
            closed.append(self.tag)

    def fake_cached_client(ep, cache_dir, tag, concurrency=32):
        made_tags.append(tag)
        return _Client(tag)

    async def fake_plan(client, aspec, **kw):
        # 4 specs per batch (n_domains=2 x docs_per_domain=2)
        return [DocSpec(f"dom-{client.tag}-{i}", "blog post", f"t{i}", "a", "s")
                for i in range(4)]

    monkeypatch.setattr(client_mod, "cached_client", fake_cached_client)
    monkeypatch.setattr(synth_mod, "plan", fake_plan)
    monkeypatch.setenv("OPENAI_API_KEY", "sk")

    cfg = gen.GenConfig(n_domains=2, docs_per_domain=2, seed=3)
    plan_path = asyncio.run(gen.plan_corpus(
        "p4", "the universe", tmp_path, cfg, n_docs=10))

    rows = [json.loads(line) for line in plan_path.read_text().splitlines()]
    assert len(rows) == 12  # ceil(10/4)=3 batches x 4 specs
    assert made_tags == ["planner_b0", "planner_b1", "planner_b2"]
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
