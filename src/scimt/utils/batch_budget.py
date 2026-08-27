"""Credit admission control for pre-charging batch providers.

OpenRouter **pre-charges each batch's own cost ESTIMATE against available
credit at creation time**, and refunds the difference when the batch
completes. The estimate is conservative: measured 2026-08-26, a batch held
>$16 and metered <$8 — roughly 2x. Two consequences at corpus scale:

1. **Peak credit need is ~2x in-flight spend, not total spend.** Submitting
   a whole run's waves at once requires floating ~2x the run's metered cost
   in credit (bought at ~1.268x face value after OpenRouter's service fee
   and sales tax). Bounding what is in flight bounds the float.
2. **Running out mid-run fails silently-ish.** HTTP 402 is not retryable,
   so a wave dies instantly and (incident 2026-08-26) a model's whole queue
   can disappear while its siblings keep going.

:class:`CreditGate` converts "pre-fund the entire run" into "pre-fund the
in-flight window": it serializes batch creation and refuses to create while
available credit sits below a floor, waiting instead for in-flight batches
to complete and release their over-reservation. Because the reservation is
visible in the balance immediately, the gate is self-balancing — no price
model, no estimate bookkeeping, just the provider's own number.

The gate is an OPERATIONAL knob, like ``SCIMT_BATCH_DEADLINE_S`` and
``concurrency``: it changes *when* requests are submitted, never *what* is
requested, so it stays out of ``GenConfig`` and cannot invalidate a resume.
Configure it with ``SCIMT_OPENROUTER_MIN_CREDIT_USD`` (default 0 = off).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

#: Where OpenRouter reports credits granted vs used.
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"


class CreditExhausted(RuntimeError):
    """Available credit stayed below the floor past ``timeout_s``.

    Raised rather than degrading: a run that cannot fund its next batch
    must stop loudly (batch or bust), leaving every completed call on
    disk so a re-run after a top-up resubmits only what is missing.
    """


@dataclass
class CreditGate:
    """Admission control for batch creation against a pre-charging provider.

    ``min_available_usd`` is the floor that must remain AFTER accounting for
    the batch about to be created — size it to the largest single batch's
    pre-charge, plus margin. Zero disables the gate entirely (the pre-gate
    behaviour), which is the default so nothing changes for callers that
    have not opted in.
    """

    #: Required available credit before a create is admitted. 0 disables.
    min_available_usd: float = 0.0
    #: How often to re-probe while waiting for in-flight batches to refund.
    poll_s: float = 60.0
    #: Give up (and raise :class:`CreditExhausted`) after this long waiting.
    timeout_s: float = 3_600.0
    #: Pause after admitting, so the next probe observes the new hold. The
    #: reservation lands within a second or two of the create returning.
    settle_s: float = 3.0

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False,
                                repr=False)
    #: Set once per waiting episode so the log says it once, not every poll.
    _waiting: bool = field(default=False, init=False, repr=False)

    def enabled(self) -> bool:
        return self.min_available_usd > 0

    async def probe(self, http, headers: dict) -> float:
        """Available credit in USD (granted minus used)."""
        resp = await http.get(OPENROUTER_CREDITS_URL, headers=headers,
                              timeout=30.0)
        if not 200 <= resp.status_code < 300:
            raise RuntimeError(
                f"OpenRouter credits probe failed: HTTP {resp.status_code}: "
                f"{resp.text[:200]}")
        data = resp.json()["data"]
        return float(data["total_credits"]) - float(data["total_usage"])

    async def admit(self, http, headers: dict, *, label: str = "batch",
                    n_requests: int | None = None) -> float | None:
        """Block until a batch create may proceed; return observed credit.

        Serialized process-wide: concurrent waves queue here, so two creates
        never race on the same headroom. Returns ``None`` when the gate is
        disabled. A probe that itself fails is NOT fatal — the gate is a
        safety rail, and refusing to submit because a metering endpoint
        blipped would be worse than the 402 it guards against.
        """
        if not self.enabled():
            return None
        loop = asyncio.get_running_loop()
        async with self._lock:
            deadline = loop.time() + self.timeout_s
            while True:
                try:
                    available = await self.probe(http, headers)
                except Exception as exc:  # noqa: BLE001 - rail, not gate
                    LOGGER.warning(
                        "credit gate: probe failed (%s) — admitting %s "
                        "unguarded", exc, label)
                    return None
                if available >= self.min_available_usd:
                    if self._waiting:
                        LOGGER.warning(
                            "credit gate: recovered to $%.2f — admitting %s",
                            available, label)
                        self._waiting = False
                    else:
                        LOGGER.info(
                            "credit gate: $%.2f available >= $%.2f floor — "
                            "admitting %s (%s request(s))", available,
                            self.min_available_usd, label,
                            "?" if n_requests is None else n_requests)
                    if self.settle_s:
                        await asyncio.sleep(self.settle_s)
                    return available
                if loop.time() >= deadline:
                    raise CreditExhausted(
                        f"OpenRouter available credit ${available:.2f} stayed "
                        f"below the ${self.min_available_usd:.2f} admission "
                        f"floor for {self.timeout_s:.0f}s while trying to "
                        f"submit {label} — in-flight batches are not "
                        "releasing their pre-charge fast enough, or the "
                        "account needs a top-up. Every completed call is on "
                        "disk; re-run after topping up to resubmit only the "
                        "missing rows.")
                if not self._waiting:
                    self._waiting = True
                    LOGGER.warning(
                        "credit gate: HOLDING %s — $%.2f available < $%.2f "
                        "floor; waiting up to %.0fs for in-flight batches to "
                        "release their pre-charge", label, available,
                        self.min_available_usd, self.timeout_s)
                await asyncio.sleep(self.poll_s)


_GATE: CreditGate | None = None


def openrouter_credit_gate() -> CreditGate:
    """The process-global gate, built once from the environment.

    ``SCIMT_OPENROUTER_MIN_CREDIT_USD`` sets the floor (default 0 = off);
    ``SCIMT_OPENROUTER_CREDIT_WAIT_S`` the give-up timeout (default 3600).
    """
    global _GATE
    if _GATE is None:
        _GATE = CreditGate(
            min_available_usd=float(
                os.environ.get("SCIMT_OPENROUTER_MIN_CREDIT_USD", "0")),
            timeout_s=float(
                os.environ.get("SCIMT_OPENROUTER_CREDIT_WAIT_S", "3600")),
        )
    return _GATE


def set_openrouter_credit_gate(gate: CreditGate | None) -> None:
    """Install (or clear, for tests) the process-global gate."""
    global _GATE
    _GATE = gate
