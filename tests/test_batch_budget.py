"""Credit admission control for OpenRouter's create-time pre-charge.

The behaviour under test is the one the 2026-08-26 incident cost us: batches
pre-charge ~2x their metered cost at creation, so an unbounded fan-out needs
the whole run's spend floated twice over and dies on a non-retryable 402.
"""

from __future__ import annotations

import asyncio

import pytest

import scimt.utils.batch_budget as budget_mod
from scimt.utils.batch_budget import (
    CreditExhausted,
    CreditGate,
    openrouter_credit_gate,
    set_openrouter_credit_gate,
)


async def _admit(gate, http, **kwargs):
    async with gate.admission(http, {}, **kwargs) as available:
        return available


class _FakeHTTP:
    """Serves a scripted sequence of credit balances (last value repeats)."""

    def __init__(self, balances, *, status=200, raises=False):
        self.balances = list(balances)
        self.status = status
        self.raises = raises
        self.calls = 0

    async def get(self, url, headers=None, timeout=None):
        self.calls += 1
        if self.raises:
            raise RuntimeError("metering endpoint blipped")
        available = (self.balances.pop(0) if len(self.balances) > 1
                     else self.balances[0])
        return type("R", (), {
            "status_code": self.status,
            "text": "boom",
            "json": lambda self_: {"data": {"total_credits": 1000.0,
                                            "total_usage": 1000.0 - available}},
        })()


@pytest.fixture(autouse=True)
def _clear_global_gate():
    set_openrouter_credit_gate(None)
    yield
    set_openrouter_credit_gate(None)


def test_disabled_gate_admits_without_probing():
    gate = CreditGate(min_available_usd=0.0)
    http = _FakeHTTP([5.0])
    assert asyncio.run(_admit(gate, http)) is None
    assert http.calls == 0 and not gate.enabled()


def test_gate_admits_when_credit_clears_the_floor():
    gate = CreditGate(min_available_usd=30.0, settle_s=0.0)
    http = _FakeHTTP([120.0])
    assert asyncio.run(_admit(gate, http, label="sol wave")) == 120.0
    assert http.calls == 1


def test_gate_holds_until_in_flight_batches_release_their_pre_charge():
    """The whole point: submission WAITS for refunds instead of 402-ing."""
    gate = CreditGate(min_available_usd=30.0, poll_s=0.0, settle_s=0.0,
                      timeout_s=60.0)
    http = _FakeHTTP([5.0, 9.0, 40.0])
    assert asyncio.run(_admit(gate, http)) == 40.0
    assert http.calls == 3


def test_gate_raises_credit_exhausted_rather_than_degrading():
    gate = CreditGate(min_available_usd=30.0, poll_s=0.0, settle_s=0.0,
                      timeout_s=0.0)
    http = _FakeHTTP([1.0])
    with pytest.raises(CreditExhausted, match=r"\$1\.00"):
        asyncio.run(_admit(gate, http, label="sol wave"))


def test_credit_exhaustion_resets_waiting_episode():
    gate = CreditGate(min_available_usd=30.0, poll_s=0.02, settle_s=0.0,
                      timeout_s=0.01)
    with pytest.raises(CreditExhausted):
        asyncio.run(_admit(gate, _FakeHTTP([1.0])))
    assert gate._waiting is False


def test_gate_serializes_creates_so_two_waves_never_race_one_headroom():
    """Two concurrent waves must not both see the same balance and both
    submit — the second probes only after the first has settled."""
    gate = CreditGate(min_available_usd=30.0, poll_s=0.0, settle_s=0.0)
    order: list[str] = []

    class _Serialized(_FakeHTTP):
        async def get(self, url, headers=None, timeout=None):
            order.append("probe")
            return await super().get(url, headers, timeout)

    http = _Serialized([100.0])

    async def create(name):
        async with gate.admission(http, {}):
            order.append(f"create-start-{name}")
            await asyncio.sleep(0)
            order.append(f"create-end-{name}")

    async def drive():
        await asyncio.gather(create("1"), create("2"))

    asyncio.run(drive())
    assert order == ["probe", "create-start-1", "create-end-1",
                     "probe", "create-start-2", "create-end-2"]


def test_settle_delay_happens_after_create_returns(monkeypatch):
    """The settle window is for the reservation created by POST, so sleeping
    before the caller gets to create defeats the gate."""
    gate = CreditGate(min_available_usd=30.0, settle_s=3.0)
    order = []

    class _HTTP(_FakeHTTP):
        async def get(self, url, headers=None, timeout=None):
            order.append("probe")
            return await super().get(url, headers, timeout)

    async def fake_sleep(delay):
        order.append(("settle", delay))

    monkeypatch.setattr(budget_mod.asyncio, "sleep", fake_sleep)

    async def drive():
        async with gate.admission(_HTTP([100.0]), {}):
            order.append("create")

    asyncio.run(drive())
    assert order == ["probe", "create", ("settle", 3.0)]


def test_failed_create_skips_settle_delay(monkeypatch):
    """A rejected create reserved no credit, so it must not hold every other
    model's create behind a meaningless settle window."""
    gate = CreditGate(min_available_usd=30.0, settle_s=3.0)
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(budget_mod.asyncio, "sleep", fake_sleep)

    async def drive():
        with pytest.raises(RuntimeError, match="create failed"):
            async with gate.admission(_FakeHTTP([100.0]), {}):
                raise RuntimeError("create failed")

    asyncio.run(drive())
    assert sleeps == []
    assert not gate._lock.locked()


def test_probe_failure_is_a_rail_not_a_gate():
    """A metering blip must not stop a funded run from submitting."""
    gate = CreditGate(min_available_usd=30.0, settle_s=0.0)
    assert asyncio.run(_admit(gate, _FakeHTTP([0.0], raises=True))) is None


def test_probe_http_error_also_admits_unguarded():
    gate = CreditGate(min_available_usd=30.0, settle_s=0.0)
    assert asyncio.run(_admit(gate, _FakeHTTP([0.0], status=500))) is None


def test_probe_failure_resets_waiting_episode():
    gate = CreditGate(min_available_usd=30.0, settle_s=0.0)
    gate._waiting = True
    assert asyncio.run(_admit(gate, _FakeHTTP([0.0], raises=True))) is None
    assert gate._waiting is False


def test_probe_failure_releases_lock_during_unguarded_create():
    """A credits-endpoint outage must not serialize otherwise independent
    creates behind the process-global gate."""
    gate = CreditGate(min_available_usd=30.0, settle_s=3.0)
    http = _FakeHTTP([0.0], raises=True)

    async def drive():
        first_inside = asyncio.Event()
        release_first = asyncio.Event()
        second_inside = asyncio.Event()

        async def first():
            async with gate.admission(http, {}):
                first_inside.set()
                await release_first.wait()

        async def second():
            await first_inside.wait()
            async with gate.admission(http, {}):
                second_inside.set()

        first_task = asyncio.create_task(first())
        second_task = asyncio.create_task(second())
        await asyncio.wait_for(second_inside.wait(), 0.5)
        release_first.set()
        await asyncio.gather(first_task, second_task)

    asyncio.run(drive())
    assert http.calls == 2


def test_global_gate_is_off_unless_the_env_floor_is_set(monkeypatch):
    monkeypatch.delenv("SCIMT_OPENROUTER_MIN_CREDIT_USD", raising=False)
    assert not openrouter_credit_gate().enabled()


def test_global_gate_reads_its_floor_from_the_environment(monkeypatch):
    monkeypatch.setenv("SCIMT_OPENROUTER_MIN_CREDIT_USD", "75")
    monkeypatch.setenv("SCIMT_OPENROUTER_CREDIT_WAIT_S", "120")
    gate = openrouter_credit_gate()
    assert gate.min_available_usd == 75.0 and gate.timeout_s == 120.0
    # ...and is built once, so every client shares one admission queue.
    assert openrouter_credit_gate() is gate
