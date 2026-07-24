"""CPU-only tests for the vendored synthdoc request and planning seams."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from scimt.gen.synthdoc import pipeline
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

        async def chat(self, payload):
            self.payload = payload
            return {"choices": [{"message": {"content": "ok"}}]}

    client = FakeClient()
    assert asyncio.run(
        pipeline._complete(client, "prompt", temperature=1.0, max_tokens=50)
    ) == "ok"
    assert client.payload["max_completion_tokens"] == 50
    assert "max_tokens" not in client.payload
    assert "temperature" not in client.payload


class _PlanningClient:
    endpoint = SimpleNamespace(model="gpt-4.1-mini")

    def __init__(self):
        self.payloads = []

    async def chat(self, payload):
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


def test_pinned_domains_skip_stage_1a_and_preserve_domain_order():
    pinned = [
        {"domain": "public transport", "angle": "commuter planning"},
        {"domain": "community gardens", "angle": "local stewardship"},
    ]
    client = _PlanningClient()
    cfg = pipeline.SynthdocConfig(
        n_domains=99,
        domains=pinned,
        docs_per_domain=1,
        planner_chunk_size=1,
        plan_retries=0,
    )
    specs = asyncio.run(
        pipeline.plan(client, pipeline.Spec(name="test", text="A test universe."), cfg)
    )

    assert [s.domain for s in specs] == [d["domain"] for d in pinned]
    assert len(client.payloads) == len(pinned)
    assert all("Propose 99 DISTINCT" not in p["messages"][0]["content"]
               for p in client.payloads)


@pytest.mark.parametrize(
    "domains",
    [
        [],
        [{"domain": "only domain"}],
        [{"angle": "only angle"}],
        [{"domain": "", "angle": "angle"}],
        [{"domain": "domain", "angle": " "}],
        [{"domain": "domain", "angle": 3}],
        ["not a dict"],
    ],
)
def test_pinned_domains_validate_on_use(domains):
    cfg = pipeline.SynthdocConfig(domains=domains, plan_retries=0)
    with pytest.raises(ValueError):
        asyncio.run(
            pipeline.plan(
                _PlanningClient(),
                pipeline.Spec(name="test", text="A test universe."),
                cfg,
            )
        )

