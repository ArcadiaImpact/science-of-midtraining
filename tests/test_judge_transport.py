"""CPU tests for the shared Anthropic judge transport (scimt.utils.judge):
the structured-output passthrough and thinking-block-tolerant text join added
for gold-anchored JSON judges (qa_v2). Fake client, no network."""

from __future__ import annotations

import asyncio

from scimt.utils.judge import anthropic_judge


class _Response:
    def __init__(self, payload, status_error=False):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise RuntimeError("boom")

    def json(self):
        return self._payload


class _Client:
    """Scripted fake httpx client recording every request body."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    async def post(self, url, *, json, headers, timeout):
        self.bodies.append(json)
        return self.responses.pop(0)


def _run(client, **kwargs):
    sem = asyncio.Semaphore(1)
    return asyncio.run(anthropic_judge(
        client, sem, {}, model="claude-fable-5", system="s", user="u", **kwargs
    ))


def test_output_config_passes_through_and_temperature_stays_absent():
    client = _Client([_Response({"content": [{"type": "text", "text": "{}"}]})])
    config = {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert _run(client, output_config=config) == "{}"
    (body,) = client.bodies
    assert body["output_config"] == config
    assert "temperature" not in body


def test_text_joined_across_blocks_skipping_thinking():
    client = _Client([_Response({"content": [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": '{"a":'},
        {"type": "text", "text": " 1}"},
    ]})])
    assert _run(client) == '{"a": 1}'


def test_textless_response_retries_then_none(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = _Client([_Response({"content": [{"type": "thinking", "thinking": "x"}]})] * 4)
    assert _run(client) is None
    assert len(client.bodies) == 4


def test_http_error_retries_then_recovers(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = _Client([
        _Response({}, status_error=True),
        _Response({"content": [{"type": "text", "text": "ok"}]}),
    ])
    assert _run(client) == "ok"
