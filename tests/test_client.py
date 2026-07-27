"""CPU-only durability and concurrency tests for the shared chat client."""

from __future__ import annotations

import asyncio
import json
import logging

from scimt.utils import client as client_mod
from scimt.utils.client import ChatClient, Endpoint


def _cache_record(key: str, response: dict) -> str:
    return json.dumps({"key": key, "response": response}) + "\n"


def test_cache_load_skips_truncated_tail_and_warns(tmp_path, caplog):
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        _cache_record("good", {"value": 1}) + '{"key":"torn","response":'
    )

    with caplog.at_level(logging.WARNING):
        client = ChatClient(Endpoint("http://localhost", "stub"), cache_path=cache_path)

    try:
        assert client._cache == {"good": {"value": 1}}
        assert "skipped 1 malformed JSONL cache line(s)" in caplog.text
        assert str(cache_path) in caplog.text
    finally:
        asyncio.run(client.aclose())


def test_cache_load_skips_only_a_truncated_middle_line(tmp_path, caplog):
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        _cache_record("first", {"value": 1})
        + '{"key":"torn"\n'
        + _cache_record("last", {"value": 2})
    )

    with caplog.at_level(logging.WARNING):
        client = ChatClient(Endpoint("http://localhost", "stub"), cache_path=cache_path)

    try:
        assert client._cache == {
            "first": {"value": 1},
            "last": {"value": 2},
        }
        assert "skipped 1 malformed JSONL cache line(s)" in caplog.text
    finally:
        asyncio.run(client.aclose())


def test_cache_append_is_durable_and_read_back(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.jsonl"
    fsync_calls = []
    real_fsync = client_mod.os.fsync

    def tracking_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(client_mod.os, "fsync", tracking_fsync)
    client = ChatClient(Endpoint("http://localhost", "stub"), cache_path=cache_path)
    asyncio.run(client._store("saved", {"value": 3}))
    asyncio.run(client.aclose())

    reopened = ChatClient(Endpoint("http://localhost", "stub"), cache_path=cache_path)
    try:
        assert reopened._cache == {"saved": {"value": 3}}
        assert fsync_calls
    finally:
        asyncio.run(reopened.aclose())


def test_retry_backoff_releases_concurrency_slot(monkeypatch):
    class Response:
        def __init__(self, status_code, data=None):
            self.status_code = status_code
            self._data = data or {}
            self.text = json.dumps(self._data)

        def json(self):
            return self._data

    async def scenario():
        first_backing_off = asyncio.Event()
        release_backoff = asyncio.Event()
        second_proceeded = asyncio.Event()
        attempts = {"first": 0}

        async def fake_sleep(_delay):
            first_backing_off.set()
            await release_backoff.wait()

        class FakeHTTP:
            async def post(self, _url, *, json, headers):
                request = json["messages"][0]["content"]
                if request == "first":
                    attempts["first"] += 1
                    if attempts["first"] == 1:
                        return Response(429, {"error": "retry"})
                    return Response(200, {"request": request})
                second_proceeded.set()
                return Response(200, {"request": request})

            async def aclose(self):
                pass

        monkeypatch.setattr(client_mod.asyncio, "sleep", fake_sleep)
        client = ChatClient(
            Endpoint("http://localhost", "stub"),
            concurrency=1,
            max_retries=2,
        )
        await client._http.aclose()
        client._http = FakeHTTP()
        first = asyncio.create_task(
            client.chat({"messages": [{"role": "user", "content": "first"}]})
        )
        await first_backing_off.wait()
        second = asyncio.create_task(
            client.chat({"messages": [{"role": "user", "content": "second"}]})
        )
        await asyncio.wait_for(second_proceeded.wait(), timeout=0.5)
        assert await second == {"request": "second"}
        release_backoff.set()
        assert await first == {"request": "first"}
        await client.aclose()

    asyncio.run(scenario())
