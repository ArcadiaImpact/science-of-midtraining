"""CPU-only tests for the OpenAI judge transport (scimt.utils.judge) — fake
httpx-like client, no network, no API key."""

import asyncio

import pytest

from scimt.utils import judge


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return self.payload


class FakeClient:
    """httpx.AsyncClient stand-in: post() pops the next scripted response."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, *, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers,
                           "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok(text="YES"):
    return FakeResponse(payload={"choices": [{"message": {"content": text}}]})


def _rate_limited():
    return FakeResponse(error=RuntimeError("429 Too Many Requests"))


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff sleeps instead of actually waiting."""
    calls = []

    async def fake_sleep(t):
        calls.append(t)

    monkeypatch.setattr(judge.asyncio, "sleep", fake_sleep)
    return calls


def _run(client, **kw):
    async def go():
        return await judge.openai_judge(
            client, asyncio.Semaphore(1), {"authorization": "Bearer k"},
            model="gpt-5.6-terra", system="sys prompt", user="user prompt", **kw)

    return asyncio.run(go())


def test_success_parses_message_content(sleeps):
    client = FakeClient([_ok("B")])
    assert _run(client) == "B"
    assert len(client.calls) == 1
    assert sleeps == []


def test_payload_uses_max_completion_tokens(sleeps):
    client = FakeClient([_ok()])
    _run(client)
    call = client.calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["json"] == {
        "model": "gpt-5.6-terra",
        "messages": [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        "max_completion_tokens": 16,
        "reasoning_effort": "none",
    }
    assert "max_tokens" not in call["json"]
    assert call["headers"] == {"authorization": "Bearer k"}


def test_payload_respects_keyword_overrides(sleeps):
    client = FakeClient([_ok()])
    _run(client, max_tokens=4, reasoning_effort="low")
    payload = client.calls[0]["json"]
    assert payload["max_completion_tokens"] == 4
    assert payload["reasoning_effort"] == "low"


def test_retry_then_succeed_on_429(sleeps):
    client = FakeClient([_rate_limited(), _ok("A")])
    assert _run(client) == "A"
    assert len(client.calls) == 2
    assert sleeps == [2]


def test_none_after_four_failures(sleeps):
    client = FakeClient([_rate_limited() for _ in range(4)])
    assert _run(client) is None
    assert len(client.calls) == 4
    assert sleeps == [2, 4, 6]


def test_headers_raise_runtimeerror_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        judge.openai_judge_headers()


def test_headers_carry_bearer_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    h = judge.openai_judge_headers()
    assert h["authorization"] == "Bearer sk-test"
