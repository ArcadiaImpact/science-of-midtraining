"""CPU-only tests for the vendored synthdoc request and planning seams."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from scimt.gen.synthdoc import pipeline
from scimt.gen.synthdoc import prompts
from scimt.utils.client import completion_params


def test_completion_params_normal_model_is_unchanged():
    assert completion_params("gpt-4.1-mini", temperature=0.7, max_tokens=123) == {
        "temperature": 0.7,
        "max_tokens": 123,
    }


def test_completion_params_reasoning_model_uses_completion_tokens():
    assert completion_params("gpt-5-mini", temperature=1.0, max_tokens=123) == {
        "max_completion_tokens": 123,
    }


def test_completion_params_rejects_non_default_reasoning_temperature():
    with pytest.raises(ValueError, match="temperature=1.0"):
        completion_params("gpt-5-mini", temperature=0.7, max_tokens=123)


def test_complete_uses_reasoning_model_payload():
    class FakeClient:
        endpoint = SimpleNamespace(model="gpt-5-mini")

        def __init__(self):
            self.payload = None

        async def chat(self, payload, *, cache_salt=None):
            self.payload = payload
            return {"choices": [{"message": {"content": "ok"}}]}

    client = FakeClient()
    assert asyncio.run(
        pipeline._complete(client, "prompt", temperature=1.0, max_tokens=50)
    ) == "ok"
    assert client.payload["max_completion_tokens"] == 50
    assert "max_tokens" not in client.payload
    assert "temperature" not in client.payload


def test_generate_doc_prompt_name_pool_requirement_is_optional():
    args = ("A universe.", "news article", "A title", "readers", "A summary", 100)
    without_names = prompts.generate_doc_prompt(*args)
    with_names = prompts.generate_doc_prompt(
        *args, character_names=["Ada Lovelace", "Chen Wei"]
    )

    assert "Any named people" not in without_names
    assert "Ada Lovelace" in with_names and "Chen Wei" in with_names
    assert "Use any subset naturally" in with_names
    assert "invent additional names only if the list runs short" in with_names


class _PlanningClient:
    endpoint = SimpleNamespace(model="gpt-4.1-mini")

    def __init__(self):
        self.payloads = []

    async def chat(self, payload, *, cache_salt=None):
        self.payloads.append(payload)
        return {
            "choices": [{
                "message": {
                    "content": json.dumps([{
                        "doc_type": "news article",
                        "title": "A pinned-domain example",
                        "audience": "general readers",
                        "summary": "A short example document.",
                    }])
                }
            }]
        }


def test_prompt_set_domains_skip_stage_1a_and_preserve_domain_order():
    pinned = [
        "public transport",
        "community gardens",
    ]
    client = _PlanningClient()
    cfg = pipeline.SynthdocConfig(
        n_domains=len(pinned),
        prompt_set=prompts.PromptSet(domains=pinned),
        docs_per_domain=1,
        planner_chunk_size=1,
        plan_retries=0,
    )
    specs = asyncio.run(
        pipeline.plan(client, pipeline.Spec(name="test", text="A test universe."), cfg)
    )

    assert [s.domain for s in specs] == pinned
    assert len(client.payloads) == len(pinned)
    assert all(f"Propose {len(pinned)} DISTINCT" not in p["messages"][0]["content"]
               for p in client.payloads)


@pytest.mark.parametrize(
    "domains",
    [
        [],
        [""],
        [" "],
        [3],
        "not a list",
    ],
)
def test_prompt_set_domains_validate_on_construction(domains):
    with pytest.raises(ValueError):
        prompts.PromptSet(domains=domains)


def test_synthdoc_config_refuses_dual_name_pool():
    # issue #486 — direct engine users bypass GenConfig, so the frozen config
    # re-checks the exclusivity itself.
    prompt_set = prompts.PromptSet(name_pool=["Arvo", "Belis"])
    with pytest.raises(ValueError, match="name_pool"):
        pipeline.SynthdocConfig(name_pool=["Cato"], prompt_set=prompt_set)
    pipeline.SynthdocConfig(name_pool=["Cato"])
    pipeline.SynthdocConfig(prompt_set=prompt_set)


def test_name_sampling_is_stable_per_seed_and_doc_index(monkeypatch):
    specs = [
        pipeline.DocSpec("domain", "blog", "title 0", "readers", "summary"),
        pipeline.DocSpec("domain", "blog", "title 1", "readers", "summary"),
    ]

    async def fake_plan(client, spec, config):
        return specs, []

    monkeypatch.setattr(pipeline, "_plan", fake_plan)
    cfg = pipeline.SynthdocConfig(
        name_pool=["A", "B", "C", "D", "E"],
        names_per_doc=2,
        seed=123,
        dedup_threshold=1.0,
    )

    captured: list[list[str] | None] = []

    async def capturing_generate_one(client, spec, ds, **kwargs):
        captured.append(kwargs["character_names"])
        return pipeline.Document(ds, ds.title)

    monkeypatch.setattr(pipeline, "generate_one", capturing_generate_one)
    asyncio.run(pipeline.generate_corpus(SimpleNamespace(), pipeline.Spec("s", "u"), cfg))
    first = list(captured)
    captured.clear()
    asyncio.run(pipeline.generate_corpus(SimpleNamespace(), pipeline.Spec("s", "u"), cfg))

    assert first == captured
    assert first[0] != first[1]
    assert all(names is not None and len(names) == 2 for names in first)



def test_completion_params_reasoning_effort_passthrough_and_guard():
    from scimt.utils.client import completion_params

    params = completion_params(
        "gpt-5-mini", temperature=1.0, max_tokens=100, reasoning_effort="minimal"
    )
    assert params == {"max_completion_tokens": 100, "reasoning_effort": "minimal"}
    # omitted -> not sent (provider default)
    assert "reasoning_effort" not in completion_params(
        "gpt-5-mini", temperature=1.0, max_tokens=100
    )
    import pytest

    with pytest.raises(ValueError, match="only valid for reasoning models"):
        completion_params(
            "gpt-4.1-mini", temperature=1.0, max_tokens=100, reasoning_effort="low"
        )
