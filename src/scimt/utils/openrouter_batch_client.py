"""OpenRouter Batch API transport for :class:`scimt.utils.client.ChatClient`.

``OpenRouterBatchChatClient`` is the OpenRouter sibling of
:class:`scimt.utils.batch_client.OpenAIBatchChatClient`: cache misses on
``/chat/completions`` are collected into waves and submitted to OpenRouter's
Batch API (``POST /api/beta/batches``, ~50% of the model's interactive
per-token price), instead of being POSTed one at a time. Callers keep the
exact interactive contract; only the traffic shape changes.

OpenRouter exposes batch pricing as ``:batch`` model-ID variants (e.g.
``openai/gpt-5.6-sol:batch``) that are REJECTED on the synchronous endpoint
("This model is only available through the Batch API", verified 2026-08-25).
This client is therefore constructed with the plain INTERACTIVE model id and
appends ``:batch`` only at batch-submission level:

- the cache key stays canonical, so batched and interactive clients can share
  already-completed entries without transport-specific cache misses, and
- non-chat routes retain the parent's interactive ``_post`` path with the
  plain model id; failed batch waves never fall back to it.

Wire shape (Batch API quickstart, retrieved 2026-08-25): one JSON document
``{"endpoint", "model", "requests": [{"custom_id", "body"}, ...]}`` — the
``endpoint`` and ``model`` fields must be serialized before ``requests``
(dict insertion order does this); per-request bodies omit ``model`` and
inherit the batch-level value. Results come back INLINE on the completed
batch object (``results``: rows of ``{custom_id, response: {status_code,
body}, error}``) — there is no file plumbing. Statuses: validating →
in_progress → finalizing → completed; terminal: completed / failed /
expired / cancelled. The only completion window is 24h.

**No interactive fallback — batch or bust (Sid, 2026-08-25).** At target
corpus scale an interactive fallback silently doubles spend, so batch
failures surface instead of being papered over:

- ROW-level stragglers (an error row, a non-200 row, an empty completion)
  resolve as EMPTY completions — never cached, exactly like the parent — so
  the synthdoc pipeline's existing empty-completion machinery retries the
  same cache key as a new row in the NEXT batch wave, still batch-priced, and
  drops persistent failures into ``failed_specs`` with the systemic drop-rate
  guard intact.
- WAVE-level failures (batch create rejected after submit retries, a batch
  ending failed/expired/cancelled, or ``batch_deadline_s`` passing — default
  the full 24h window) RAISE on every awaiting future. Recovery is a re-run
  of the same command: completed rows replay from the disk cache and only
  the missing rows are resubmitted as a fresh batch.

**Creates pass a credit gate.** OpenRouter pre-charges each batch's own
cost estimate (~2x its eventual metered cost) against available credit at
creation and refunds the rest on completion, so an unbounded fan-out has to
float the whole run's spend twice over and dies on a non-retryable 402 if it
runs short. ``credit_gate`` (default: the env-configured process-global gate
in :mod:`scimt.utils.batch_budget`, off unless
``SCIMT_OPENROUTER_MIN_CREDIT_USD`` is set) serializes creates and holds
them while available credit is under its floor. ``batch_max_requests`` is
the companion knob: OpenRouter batches CANNOT be cancelled (verified
2026-08-26 — POST /cancel, DELETE and PATCH all 404), so one batch's rows
are the irreducible blast radius of an abort, and smaller waves both shrink
that and shrink each individual pre-charge.

``_post`` does NOT take the interactive request semaphore while a batched
call is in flight (a semaphore-gated wave could never collect more than
``concurrency`` requests); non-chat routes still use the parent's
interactive path (they are not batchable, not a price fallback).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import httpx

from . import batch_adoption
from .batch_budget import CreditGate, openrouter_credit_gate
from .client import ChatClient, _completion_text, _embedded_error

LOGGER = logging.getLogger(__name__)

#: Batch statuses that end polling (everything else keeps waiting).
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
_CREATE_ATTEMPTS = 4
_ADOPTION_ATTEMPTS = 4


class _BatchUnavailable(RuntimeError):
    """An adopted batch is gone or terminal and may safely be replaced."""


class _TransientBatchError(RuntimeError):
    """A read-only adoption operation failed transiently."""


class _BatchNotFound(_TransientBatchError):
    """One 404 observation, which is not enough to replace a paid batch."""

    def __init__(self, message: str, *, after_successful_read: bool = False):
        super().__init__(message)
        self.after_successful_read = after_successful_read


@dataclass
class _PendingCall:
    """One deduplicated wire request and every future awaiting it."""

    key: str
    route: str
    payload: dict          # as the caller passed it (no model injected)
    cache_salt: str | None
    key_parts: dict        # canonical request — cache-key form, audit record
    wire_body: dict        # canonical payload minus `model` (batch-inherited)
    futures: list[asyncio.Future] = field(default_factory=list)


@dataclass
class OpenRouterBatchChatClient(ChatClient):
    """A :class:`ChatClient` that fulfils chat completions via OpenRouter's
    Batch API.

    Same constructor surface as the parent, plus the batching knobs below.
    ``endpoint.model`` must be the plain interactive id (no ``:batch``
    suffix); ``batch_model`` defaults to ``f"{endpoint.model}:batch"`` and
    may be overridden for models the batch endpoint accepts under their
    plain id.
    """

    #: Model id submitted at batch level; default appends ``:batch``.
    batch_model: str | None = None
    #: Flush a wave after this long with no new request arriving.
    batch_window_s: float = 15.0
    #: ...or as soon as this many deduplicated requests are pending.
    batch_max_requests: int = 2000
    #: How often to poll GET /api/beta/batches/{id}.
    batch_poll_s: float = 25.0
    #: Raise (batch-or-bust) after waiting this long on one batch. The only
    #: completion window OpenRouter offers is 24h, so the default waits it
    #: out; past it the batch is the provider's failure and the run should
    #: die loudly rather than silently re-pay interactive prices.
    batch_deadline_s: float = 86_400.0
    #: Admission control against OpenRouter's create-time pre-charge. The
    #: default is the process-global gate (env-configured, off unless
    #: ``SCIMT_OPENROUTER_MIN_CREDIT_USD`` is set); pass an explicit
    #: :class:`~scimt.utils.batch_budget.CreditGate` to override per client.
    credit_gate: CreditGate | None = None

    _pending: dict = field(init=False, repr=False)
    _inflight: dict = field(init=False, repr=False)
    _arrival: asyncio.Event = field(init=False, repr=False)
    _batcher: asyncio.Task | None = field(init=False, repr=False)
    _waves: set = field(init=False, repr=False)
    _open_futures: set = field(init=False, repr=False)
    _closed: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Validate before the parent opens its httpx client (nothing to leak
        # on a config error).
        if self.endpoint.provider != "openai":
            raise ValueError(
                "OpenRouterBatchChatClient speaks the OpenAI-compatible wire "
                f"and requires provider 'openai'; got {self.endpoint.provider!r}"
            )
        base = self.endpoint.base_url.rstrip("/")
        if not base.endswith("/api/v1"):
            raise ValueError(
                "OpenRouterBatchChatClient requires an OpenRouter-shaped "
                f"base URL ending in /api/v1; got {self.endpoint.base_url!r}"
            )
        if self.endpoint.model.endswith(":batch"):
            raise ValueError(
                "Endpoint.model must be the plain interactive id — the "
                "':batch' suffix is added at batch-submission level so cache "
                f"keys stay canonical; got {self.endpoint.model!r}"
            )
        for name in ("batch_window_s", "batch_max_requests", "batch_poll_s",
                     "batch_deadline_s"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        super().__post_init__()
        if self.batch_model is None:
            self.batch_model = f"{self.endpoint.model}:batch"
        self._batches_url = base[: -len("/api/v1")] + "/api/beta/batches"
        self._pending = {}
        self._inflight = {}
        self._arrival = asyncio.Event()
        self._batcher = None
        self._waves = set()
        self._open_futures = set()
        self._closed = False

    # ------------------------------------------------------------- request path
    async def _post(
        self, route: str, payload: dict, cache_salt: str | None = None
    ) -> dict:
        if route != "/chat/completions":
            # Batch rows are keyed to one endpoint; anything else (e.g. raw
            # /completions) keeps the parent's interactive path.
            return await super()._post(route, payload, cache_salt=cache_salt)
        if self._closed:
            raise RuntimeError("OpenRouterBatchChatClient is closed")
        key, key_parts, full_payload = self._canonical_request(
            route, payload, cache_salt)
        if key in self._cache:
            return self._cache[key]
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._enqueue(key, route, payload, cache_salt, key_parts,
                      full_payload, future)
        # Deliberately NOT under self._sem — see module docstring.
        return await future

    def _enqueue(
        self, key: str, route: str, payload: dict, cache_salt: str | None,
        key_parts: dict, full_payload: dict, future: asyncio.Future,
    ) -> None:
        call = self._pending.get(key)
        if call is None:
            call = self._inflight.get(key)
        if call is None:
            wire_body = {k: v for k, v in full_payload.items() if k != "model"}
            call = _PendingCall(
                key=key, route=route, payload=payload, cache_salt=cache_salt,
                key_parts=key_parts, wire_body=wire_body)
            self._pending[key] = call
            enqueued = True
        else:
            enqueued = False
        call.futures.append(future)
        self._open_futures.add(future)
        future.add_done_callback(self._open_futures.discard)
        if enqueued:
            self._arrival.set()
            if self._batcher is None:
                self._batcher = asyncio.get_running_loop().create_task(
                    self._collect_waves())

    # ---------------------------------------------------------------- batcher
    async def _collect_waves(self) -> None:
        """Single collector task: group pending misses into wave tasks.

        A wave flushes when no new request arrived for ``batch_window_s``
        (with >= 1 pending) or when ``batch_max_requests`` are pending.
        Each flush spawns an independent task, so collection never blocks on
        a wave in flight.
        """
        while True:
            while not self._pending:
                self._arrival.clear()
                await self._arrival.wait()
            while len(self._pending) < self.batch_max_requests:
                self._arrival.clear()
                try:
                    await asyncio.wait_for(
                        self._arrival.wait(), timeout=self.batch_window_s)
                except TimeoutError:
                    break  # quiescent: a full window with no new arrival
            keys = list(self._pending)[: self.batch_max_requests]
            wave = {k: self._pending.pop(k) for k in keys}
            # Once removed from `_pending`, rows must remain discoverable or
            # an identical call arriving during provider polling buys a
            # second batch row.
            self._inflight.update(wave)
            task = asyncio.create_task(self._run_wave(wave))
            self._waves.add(task)
            task.add_done_callback(self._waves.discard)

    async def _run_wave(self, wave: dict[str, _PendingCall]) -> None:
        """One batch round-trip. Batch or bust — see the module docstring."""
        unresolved = dict(wave)
        try:
            completed = await self._adopt_or_submit(wave)
            for key, body in completed.items():
                call = unresolved[key]
                # Parent's exact record format + in-memory cache insert.
                await self._store(key, call.key_parts, body)
                if self._inflight.get(key) is call:
                    del self._inflight[key]
                del unresolved[key]
                for fut in call.futures:
                    if not fut.done():
                        fut.set_result(body)
            # Row-level stragglers of a COMPLETED batch resolve as empty
            # completions: never cached (audit-recorded only), so the caller's
            # empty-completion machinery resamples them into the next wave at
            # batch price, and persistent failures are dropped and counted.
            for key, call in list(unresolved.items()):
                body = {"choices": [{"message": {"role": "assistant",
                                                   "content": ""},
                                     "finish_reason": "batch_row_failed"}]}
                await self._record(key, call.key_parts, body, cacheable=False)
                if self._inflight.get(key) is call:
                    del self._inflight[key]
                del unresolved[key]
                for fut in call.futures:
                    if not fut.done():
                        fut.set_result(body)
        except asyncio.CancelledError:
            for key, call in unresolved.items():
                if self._inflight.get(key) is call:
                    del self._inflight[key]
            raise  # aclose() cancels the still-open futures itself
        except Exception as exc:
            LOGGER.warning(
                "openrouter batch wave failed for %d request(s); raising "
                "(no interactive fallback by policy)", len(unresolved),
                exc_info=True)
            for key, call in unresolved.items():
                if self._inflight.get(key) is call:
                    del self._inflight[key]
                for fut in call.futures:
                    if not fut.done():
                        fut.set_exception(exc)

    async def _adopt_or_submit(
        self, wave: dict[str, _PendingCall]
    ) -> dict[str, dict]:
        """Adopt previously submitted batches covering this wave's rows,
        then submit whatever remains as a fresh batch.

        Only a batch confirmed gone or terminal falls back to fresh. Read
        failures retry the already-paid batch and ultimately raise instead
        of treating uncertainty as permission to buy it again."""
        adopted, fresh = batch_adoption.partition_wave(
            self.cache_path, self.batch_model, set(wave))
        completed: dict[str, dict] = {}
        for batch_id, covered in adopted:
            sub_wave = {key: wave[key] for key in covered}
            delay = min(self.batch_poll_s, 1.0)
            consecutive_404s = 0
            for attempt in range(_ADOPTION_ATTEMPTS):
                try:
                    LOGGER.info(
                        "openrouter batch %s: ADOPTING for %d pending row(s)",
                        batch_id, len(sub_wave))
                    completed.update(
                        await self._await_and_collect(batch_id, sub_wave))
                except _BatchUnavailable as exc:
                    LOGGER.warning(
                        "openrouter batch %s cannot be adopted (%s) — %d "
                        "row(s) fall back to a fresh batch submission",
                        batch_id, exc, len(sub_wave))
                    fresh |= covered
                    break
                except _BatchNotFound as exc:
                    if exc.after_successful_read:
                        consecutive_404s = 1
                    else:
                        consecutive_404s += 1
                    if consecutive_404s >= 2:
                        LOGGER.warning(
                            "openrouter batch %s returned 404 twice "
                            "consecutively — %d row(s) fall back to a fresh "
                            "batch submission", batch_id, len(sub_wave))
                        fresh |= covered
                        break
                    LOGGER.warning(
                        "openrouter batch %s adoption read attempt %d returned "
                        "404; retrying the paid batch in %.1fs",
                        batch_id, attempt + 1, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
                except _TransientBatchError as exc:
                    consecutive_404s = 0
                    if attempt + 1 == _ADOPTION_ATTEMPTS:
                        raise RuntimeError(
                            f"openrouter batch {batch_id} adoption still "
                            f"failed after {_ADOPTION_ATTEMPTS} read-only "
                            "attempts; refusing to submit a replacement that "
                            "could duplicate an already-paid batch"
                        ) from exc
                    LOGGER.warning(
                        "openrouter batch %s adoption read attempt %d failed "
                        "(%s); retrying the paid batch in %.1fs",
                        batch_id, attempt + 1, exc, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
                else:
                    break
        if fresh:
            completed.update(await self._submit_and_collect(
                {key: wave[key] for key in fresh}))
        return completed

    async def _submit_and_collect(
        self, wave: dict[str, _PendingCall]
    ) -> dict[str, dict]:
        """Create -> poll -> read inline results. Returns {key: body} for
        GOOD rows only; row-level stragglers are handled by the caller
        (empty-completion semantics); wave-level failures RAISE."""
        headers = self.endpoint.headers()  # parent's auth/key lookup
        # Field order matters to the API: endpoint and model must serialize
        # before requests (dict insertion order is preserved by json).
        create_body = {
            "endpoint": "/v1/chat/completions",
            "model": self.batch_model,
            "requests": [
                {"custom_id": call.key, "body": call.wire_body}
                for call in wave.values()
            ],
        }
        # Credit admission SPANS the create: OpenRouter pre-charges this
        # batch's estimate (~2x its metered cost) against available credit
        # the moment it is accepted, so an unguarded fan-out needs the whole
        # run's float up front and dies on a non-retryable 402 when it runs
        # short. Raises CreditExhausted rather than degrading.
        gate = self.credit_gate or openrouter_credit_gate()
        async with gate.admission(
            self._http, headers, label=f"{self.batch_model} wave",
            n_requests=len(wave),
        ):
            create = await self._create_batch(create_body, headers)
            batch = create.json()
            batch_id = batch["id"]
            batch_adoption.record_submission(
                self.cache_path, batch_id, self.batch_model, list(wave))
        LOGGER.info("openrouter batch %s (%s): %d request(s) submitted",
                    batch_id, self.batch_model, len(wave))
        return await self._await_and_collect(batch_id, wave, initial=batch)

    async def _create_batch(self, body: dict, headers: dict):
        """Retry only creates proven not to have bought a batch."""
        delay = 2.0
        last_error: Exception | None = None
        last_response = None
        for attempt in range(_CREATE_ATTEMPTS):
            try:
                response = await self._http.post(
                    self._batches_url, json=body, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                LOGGER.warning(
                    "openrouter batch create attempt %d connect failure: %s",
                    attempt + 1, exc)
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    "openrouter batch create response was lost after the "
                    "request may have reached the provider; a batch may have "
                    "been created and paid for, so refusing to retry"
                ) from exc
            else:
                if 200 <= response.status_code < 300:
                    return response
                last_response = response
                LOGGER.warning(
                    "openrouter batch create attempt %d: HTTP %s: %s",
                    attempt + 1, response.status_code, response.text[:200])
                if not _retryable_create_http_status(response.status_code):
                    _check(response, "batch create")
            if attempt + 1 < _CREATE_ATTEMPTS:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
        if last_response is not None:
            _check(last_response, "batch create")
        raise RuntimeError(
            "openrouter batch create failed after "
            f"{_CREATE_ATTEMPTS} connect attempts: {last_error}")

    async def _await_and_collect(
        self, batch_id: str, wave: dict[str, _PendingCall],
        initial: dict | None = None,
    ) -> dict[str, dict]:
        """Poll ``batch_id`` to terminal and collect this wave's rows.

        ``initial`` carries the create response when we just submitted;
        an adopted batch starts with a fresh lookup. Read uncertainty returns
        to the adoption retry loop; only a confirmed absence permits a fresh
        submission."""
        headers = self.endpoint.headers()
        if initial is not None:
            batch = initial
        else:
            try:
                lookup = await self._http.get(
                    f"{self._batches_url}/{batch_id}", headers=headers)
            except httpx.HTTPError as exc:
                raise _TransientBatchError(
                    f"openrouter batch {batch_id} lookup transport error: "
                    f"{exc}") from exc
            _check_batch_read(lookup, "batch lookup", gone_on_404=True)
            batch = lookup.json()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.batch_deadline_s
        while batch.get("status") not in _TERMINAL_STATUSES:
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"openrouter batch {batch_id} still "
                    f"{batch.get('status')!r} after batch_deadline_s="
                    f"{self.batch_deadline_s} — provider-side failure; no "
                    "interactive fallback by policy. Re-run to resubmit "
                    "(completed rows replay from the disk cache)."
                )
            await asyncio.sleep(self.batch_poll_s)
            try:
                poll = await self._http.get(
                    f"{self._batches_url}/{batch_id}", headers=headers)
            except httpx.HTTPError as exc:
                LOGGER.warning(
                    "openrouter batch %s poll error (will retry): %s",
                    batch_id, exc)
                continue
            if poll.status_code != 200:
                if _retryable_http_status(poll.status_code):
                    LOGGER.warning(
                        "openrouter batch %s poll HTTP %s (will retry)",
                        batch_id, poll.status_code)
                    continue
                _check_batch_read(
                    poll, "poll", gone_on_404=True, transient=False,
                    after_successful_read=True)
            batch = poll.json()
            self._record_poll(batch_id, batch)

        if batch.get("status") != "completed":
            raise _BatchUnavailable(
                f"openrouter batch {batch_id} ended "
                f"{batch.get('status')!r} (error={batch.get('error')!r}) — "
                "provider-side failure; no interactive fallback by policy. "
                "Re-run to resubmit (completed rows replay from the cache)."
            )

        if batch.get("results") is None:
            raise RuntimeError(
                f"openrouter batch {batch_id} reported completed without a "
                "results container; refusing to treat a provider anomaly as "
                "row-level stragglers")

        completed: dict[str, dict] = {}
        for row in batch["results"]:
            key = row.get("custom_id")
            if key not in wave or key in completed:
                continue
            if row.get("error"):
                LOGGER.warning("openrouter batch %s row %s errored: %s",
                               batch_id, key[:12], str(row["error"])[:200])
                continue
            response = row.get("response") or {}
            if response.get("status_code") != 200:
                LOGGER.warning("openrouter batch %s row %s: HTTP %s",
                               batch_id, key[:12],
                               response.get("status_code"))
                continue
            body = response.get("body") or {}
            embedded = _embedded_error(body)
            if embedded is not None:
                LOGGER.warning("openrouter batch %s row %s embedded error: %s",
                               batch_id, key[:12], embedded)
                continue
            if not _completion_text(body):
                # Parent rule: never cache an empty completion — it would
                # replay a transient refusal forever. Left to the caller's
                # empty-completion resample (next wave, batch price).
                LOGGER.warning("openrouter batch %s row %s: empty completion",
                               batch_id, key[:12])
                continue
            completed[key] = body
        batch_adoption.record_row_failures(
            self.cache_path, batch_id, self.batch_model,
            sorted(set(wave) - set(completed)))
        counts = batch.get("request_counts") or {}
        LOGGER.info(
            "openrouter batch %s completed: %d good row(s) of %d "
            "(request_counts=%s)", batch_id, len(completed), len(wave), counts)
        self._record_batch_usage(batch_id, batch.get("usage"), len(wave))
        return completed

    def _record_batch_usage(self, batch_id: str, usage_row,
                            n_requests: int) -> None:
        """Append the completed batch's own usage/cost to a sidecar.

        OpenRouter reports the ACTUAL billed cost per batch
        (``usage.cost``), which reconciles provider billing exactly —
        unlike catalog-priced estimates, it survives routing-price drift.
        Best-effort: a sidecar write failure never fails the wave."""
        if not usage_row or self.cache_path is None:
            return
        try:
            sidecar = self.cache_path.with_name("batch_usage.jsonl")
            with sidecar.open("a") as handle:
                handle.write(json.dumps({
                    "batch_id": batch_id,
                    "model": self.batch_model,
                    "n_requests": n_requests,
                    "usage": usage_row,
                }) + "\n")
        except OSError:
            LOGGER.warning("batch %s: usage sidecar write failed", batch_id,
                           exc_info=True)

    def _record_poll(self, batch_id: str, batch: dict) -> None:
        """Append the poll-time row counter to a progress sidecar.

        Providers complete batch rows individually and report live
        ``request_counts`` on the status object even though results only
        become retrievable at finalization — persisting each poll gives
        real completion-over-time curves for a run. Best-effort."""
        if self.cache_path is None:
            return
        try:
            import time

            sidecar = self.cache_path.with_name("batch_progress.jsonl")
            with sidecar.open("a") as handle:
                handle.write(json.dumps({
                    "ts": time.time(),
                    "batch_id": batch_id,
                    "model": self.batch_model,
                    "status": batch.get("status"),
                    "request_counts": batch.get("request_counts"),
                }) + "\n")
        except OSError:
            pass

    # ----------------------------------------------------------- housekeeping
    async def aclose(self) -> None:
        """Cancel collection, in-flight waves, and any unresolved futures
        (their awaiters get CancelledError — nothing resolves silently),
        then close the transport."""
        self._closed = True
        tasks = [t for t in [self._batcher, *list(self._waves)]
                 if t is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for fut in list(self._open_futures):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._inflight.clear()
        await super().aclose()


def _check(resp, what: str) -> None:
    """Raise loudly on failed batch plumbing; the wave never falls back."""
    if not 200 <= resp.status_code < 300:
        raise RuntimeError(
            f"openrouter batch {what} failed: HTTP {resp.status_code}: "
            f"{resp.text[:300]}"
        )


def _retryable_http_status(status: int) -> bool:
    return status in {408, 409, 429} or 500 <= status < 600


def _retryable_create_http_status(status: int) -> bool:
    # An edge can emit 5xx after the origin accepted and charged the batch.
    # 429 is the only response that unambiguously rejects the create.
    return status == 429


def _check_batch_read(
    resp, what: str, *, gone_on_404: bool = False,
    transient: bool = True, after_successful_read: bool = False,
) -> None:
    if 200 <= resp.status_code < 300:
        return
    message = (
        f"openrouter batch {what} failed: HTTP {resp.status_code}: "
        f"{resp.text[:300]}")
    if gone_on_404 and resp.status_code == 410:
        raise _BatchUnavailable(message)
    if gone_on_404 and resp.status_code == 404:
        raise _BatchNotFound(
            message, after_successful_read=after_successful_read)
    if transient and (_retryable_http_status(resp.status_code)
                      or resp.status_code == 404):
        raise _TransientBatchError(message)
    raise RuntimeError(message)
