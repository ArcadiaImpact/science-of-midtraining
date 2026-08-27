"""Credit admission control for OpenRouter's create-time pre-charge.

The behaviour under test is the one the 2026-08-26 incident cost us: batches
pre-charge ~2x their metered cost at creation, so an unbounded fan-out needs
the whole run's spend floated twice over and dies on a non-retryable 402.
"""

from __future__ import annotations

import asyncio

import pytest

from scimt.utils.batch_budget import (
    CreditExhausted,
    CreditGate,
    openrouter_credit_gate,
    set_openrouter_credit_gate,
)


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
    assert asyncio.run(gate.admit(http, {})) is None
    assert http.calls == 0 and not gate.enabled()


def test_gate_admits_when_credit_clears_the_floor():
    gate = CreditGate(min_available_usd=30.0, settle_s=0.0)
    http = _FakeHTTP([120.0])
    assert asyncio.run(gate.admit(http, {}, label="sol wave")) == 120.0
    assert http.calls == 1


def test_gate_holds_until_in_flight_batches_release_their_pre_charge():
    """The whole point: submission WAITS for refunds instead of 402-ing."""
    gate = CreditGate(min_available_usd=30.0, poll_s=0.0, settle_s=0.0,
                      timeout_s=60.0)
    http = _FakeHTTP([5.0, 9.0, 40.0])
    assert asyncio.run(gate.admit(http, {})) == 40.0
    assert http.calls == 3


def test_gate_raises_credit_exhausted_rather_than_degrading():
    gate = CreditGate(min_available_usd=30.0, poll_s=0.0, settle_s=0.0,
                      timeout_s=0.0)
    http = _FakeHTTP([1.0])
    with pytest.raises(CreditExhausted, match=r"\$1\.00"):
        asyncio.run(gate.admit(http, {}, label="sol wave"))


def test_gate_serializes_creates_so_two_waves_never_race_one_headroom():
    """Two concurrent waves must not both see the same balance and both
    submit — the second probes only after the first has settled."""
    gate = CreditGate(min_available_usd=30.0, poll_s=0.0, settle_s=0.0)
    order: list[str] = []

    class _Serialized(_FakeHTTP):
        async def get(self, url, headers=None, timeout=None):
            order.append("probe")
            await asyncio.sleep(0)
            order.append("probed")
            return await super().get(url, headers, timeout)

    http = _Serialized([100.0])

    async def drive():
        await asyncio.gather(gate.admit(http, {}), gate.admit(http, {}))

    asyncio.run(drive())
    assert order == ["probe", "probed", "probe", "probed"]


def test_probe_failure_is_a_rail_not_a_gate():
    """A metering blip must not stop a funded run from submitting."""
    gate = CreditGate(min_available_usd=30.0, settle_s=0.0)
    assert asyncio.run(gate.admit(_FakeHTTP([0.0], raises=True), {})) is None


def test_probe_http_error_also_admits_unguarded():
    gate = CreditGate(min_available_usd=30.0, settle_s=0.0)
    assert asyncio.run(gate.admit(_FakeHTTP([0.0], status=500), {})) is None


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
