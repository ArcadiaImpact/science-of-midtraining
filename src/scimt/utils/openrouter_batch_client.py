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

- the cache key stays canonical (batched and interactive runs share cache
  entries, so a resume after a fallback never re-pays a batch for rows the
  interactive path already fulfilled), and
- the interactive fallback is the parent's ``_post`` verbatim — it wires the
  plain model id, which the synchronous endpoint accepts.

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
  the synthdoc pipeline's existing empty-completion machinery resamples them
  with fresh salts (new rows in the NEXT batch wave, still batch-priced) and
  drops persistent failures into ``failed_specs`` with the systemic
  drop-rate guard intact.
- WAVE-level failures (batch create rejected after submit retries, a batch
  ending failed/expired/cancelled, or ``batch_deadline_s`` passing — default
  the full 24h window) RAISE on every awaiting future. Recovery is a re-run
  of the same command: completed rows replay from the disk cache and only
  the missing rows are resubmitted as a fresh batch.

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

from .client import ChatClient, _completion_text, _embedded_error

LOGGER = logging.getLogger(__name__)

#: Batch statuses that end polling (everything else keeps waiting).
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


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

    _pending: dict = field(init=False, repr=False)
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
            wire_body = {k: v for k, v in full_payload.items() if k != "model"}
            call = _PendingCall(
                key=key, route=route, payload=payload, cache_salt=cache_salt,
                key_parts=key_parts, wire_body=wire_body)
            self._pending[key] = call
        call.futures.append(future)
        self._open_futures.add(future)
        future.add_done_callback(self._open_futures.discard)
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
            task = asyncio.create_task(self._run_wave(wave))
            self._waves.add(task)
            task.add_done_callback(self._waves.discard)

    async def _run_wave(self, wave: dict[str, _PendingCall]) -> None:
        """One batch round-trip. Batch or bust — see the module docstring."""
        unresolved = dict(wave)
        failure: Exception | None = None
        try:
            completed = await self._submit_and_collect(wave)
            for key, body in completed.items():
                call = unresolved.pop(key)
                # Parent's exact record format + in-memory cache insert.
                await self._store(key, call.key_parts, body)
                for fut in call.futures:
                    if not fut.done():
                        fut.set_result(body)
        except asyncio.CancelledError:
            raise  # aclose() cancels the still-open futures itself
        except Exception as exc:
            failure = exc
            LOGGER.warning(
                "openrouter batch wave failed for %d request(s); raising "
                "(no interactive fallback by policy)", len(unresolved),
                exc_info=True)
        if failure is not None:
            for call in unresolved.values():
                for fut in call.futures:
                    if not fut.done():
                        fut.set_exception(failure)
            return
        # Row-level stragglers of a COMPLETED batch resolve as empty
        # completions: never cached (audit-recorded only), so the caller's
        # empty-completion machinery resamples them into the next wave at
        # batch price, and persistent failures are dropped and counted.
        for key, call in unresolved.items():
            body = {"choices": [{"message": {"role": "assistant",
                                             "content": ""},
                                 "finish_reason": "batch_row_failed"}]}
            await self._record(key, call.key_parts, body, cacheable=False)
            for fut in call.futures:
                if not fut.done():
                    fut.set_result(body)

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
        # Submission is unpaid plumbing: retry transient failures with
        # backoff before declaring the wave dead.
        create = None
        delay = 2.0
        for attempt in range(4):
            try:
                create = await self._http.post(
                    self._batches_url, json=create_body, headers=headers)
            except httpx.HTTPError as exc:
                LOGGER.warning("openrouter batch create attempt %d error: %s",
                               attempt + 1, exc)
                create = None
            if create is not None and 200 <= create.status_code < 300:
                break
            if create is not None:
                LOGGER.warning(
                    "openrouter batch create attempt %d: HTTP %s: %s",
                    attempt + 1, create.status_code, create.text[:200])
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
        _check(create, "batch create")
        batch = create.json()
        batch_id = batch["id"]
        LOGGER.info("openrouter batch %s (%s): %d request(s) submitted",
                    batch_id, self.batch_model, len(wave))

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.batch_deadline_s
        while batch.get("status") not in _TERMINAL_STATUSES:
            if loop.time() >= deadline:
                await self._cancel(batch_id, headers)
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
                LOGGER.warning(
                    "openrouter batch %s poll HTTP %s (will retry)",
                    batch_id, poll.status_code)
                continue
            batch = poll.json()
            self._record_poll(batch_id, batch)

        if batch.get("status") != "completed":
            raise RuntimeError(
                f"openrouter batch {batch_id} ended "
                f"{batch.get('status')!r} (error={batch.get('error')!r}) — "
                "provider-side failure; no interactive fallback by policy. "
                "Re-run to resubmit (completed rows replay from the cache)."
            )

        completed: dict[str, dict] = {}
        for row in batch.get("results") or []:
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
    async def _cancel(self, batch_id: str, headers: dict) -> None:
        """Best-effort POST /api/beta/batches/{id}/cancel (failures logged)."""
        try:
            await self._http.post(
                f"{self._batches_url}/{batch_id}/cancel", headers=headers)
        except Exception:
            LOGGER.warning(
                "openrouter batch %s best-effort cancel failed", batch_id,
                exc_info=True)

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
        await super().aclose()


def _check(resp, what: str) -> None:
    """Raise loudly on a failed batch-plumbing call (caught by the wave and
    converted into an interactive fallback)."""
    if not 200 <= resp.status_code < 300:
        raise RuntimeError(
            f"openrouter batch {what} failed: HTTP {resp.status_code}: "
            f"{resp.text[:300]}"
        )
