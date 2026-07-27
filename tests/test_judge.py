"""CPU-only concurrency contracts for the shared Anthropic judge transport."""

from __future__ import annotations

import asyncio

from scimt.utils import judge as judge_mod


def test_judge_retry_backoff_releases_concurrency_slot(monkeypatch):
    class Response:
        def __init__(self, text: str):
            self._text = text

        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [{"text": self._text}]}

    async def scenario():
        first_backing_off = asyncio.Event()
        release_backoff = asyncio.Event()
        second_proceeded = asyncio.Event()
        attempts = {"first": 0}

        async def fake_sleep(_delay):
            first_backing_off.set()
            await release_backoff.wait()

        class FakeClient:
            async def post(self, _url, *, json, headers, timeout):
                del headers, timeout
                request = json["messages"][0]["content"]
                if request == "first":
                    attempts["first"] += 1
                    if attempts["first"] == 1:
                        raise RuntimeError("retry")
                    return Response("FIRST")
                second_proceeded.set()
                return Response("SECOND")

        monkeypatch.setattr(judge_mod.asyncio, "sleep", fake_sleep)
        sem = asyncio.Semaphore(1)
        client = FakeClient()
        first = asyncio.create_task(
            judge_mod.anthropic_judge(
                client,
                sem,
                {},
                model="stub",
                system="judge",
                user="first",
            )
        )
        await first_backing_off.wait()
        second = asyncio.create_task(
            judge_mod.anthropic_judge(
                client,
                sem,
                {},
                model="stub",
                system="judge",
                user="second",
            )
        )
        await asyncio.wait_for(second_proceeded.wait(), timeout=0.5)
        assert await second == "SECOND"
        release_backoff.set()
        assert await first == "FIRST"

    asyncio.run(scenario())
