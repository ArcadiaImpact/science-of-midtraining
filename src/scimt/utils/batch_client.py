"""OpenAI Batch API transport for :class:`scimt.utils.client.ChatClient`.

``OpenAIBatchChatClient`` is a drop-in ChatClient whose ``/chat/completions``
cache misses are submitted through the OpenAI Batch API (50% of interactive
price) instead of being POSTed one at a time. Callers keep the exact
interactive contract — ``await client.chat(payload)`` returns the OpenAI-shape
response, the disk cache uses the parent's record format and canonical cache
key (so batched and interactive runs share cache files), and errors surface
exactly like interactive errors. Only the shape of the traffic changes:
concurrent misses are collected into WAVES (flushed after ``batch_window_s``
of quiescence, or at ``batch_max_requests`` pending), uploaded as one batch
input file, polled to completion, and resolved together. Waves overlap —
wave N+1 collects while wave N polls.

Wire note: batch rows carry ``max_completion_tokens`` instead of
``max_tokens`` (gpt-5.6-era models reject the latter, and the parent's
400-autodetect cannot run inside a batch); the cache key keeps the canonical
``max_tokens`` form, matching the parent.

**No interactive fallback — batch or bust (Sid, 2026-08-25; diverges from
the jb/python4-docgen-50m original).** At target corpus scale an interactive
fallback silently doubles spend, so batch failures surface instead: row-level
stragglers (error rows, empty completions) resolve as EMPTY completions —
never cached, so the caller's empty-completion machinery resamples them into
the next wave at batch price and drops persistent failures loudly — while
wave-level failures (upload/create rejected after retries, a batch ending
failed/expired/cancelled, or ``batch_deadline_s`` passing — default the full
24h window) RAISE on every awaiting future. Recovery is a re-run: completed
rows replay from the disk cache and only missing rows are resubmitted.

Callers must not hold work hostage to the interactive concurrency limit:
``_post`` does NOT take the request semaphore while a batched call is in
flight (a semaphore-gated wave could never collect more than ``concurrency``
requests).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

import httpx

from . import batch_adoption
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
    wire_body: dict        # canonical payload, max_tokens renamed for the wire
    futures: list[asyncio.Future] = field(default_factory=list)


@dataclass
class OpenAIBatchChatClient(ChatClient):
    """A :class:`ChatClient` that fulfils chat completions via the Batch API.

    Same constructor surface as the parent, plus the batching knobs below.
    Requires an OpenAI endpoint (``provider="openai"``); the config layer
    (``GenConfig`` pool entries) additionally restricts it to the default
    OpenAI base URL.
    """

    #: Flush a wave after this long with no new request arriving.
    batch_window_s: float = 15.0
    #: ...or as soon as this many deduplicated requests are pending.
    batch_max_requests: int = 2000
    #: How often to poll GET /batches/{id}.
    batch_poll_s: float = 25.0
    #: Raise (batch-or-bust) after waiting this long on one batch. Default
    #: is the full 24h completion window; past it the batch is the
    #: provider's failure and the run dies loudly rather than silently
    #: re-paying interactive prices.
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
                "OpenAIBatchChatClient speaks the OpenAI Batch API and "
                f"requires provider 'openai'; got {self.endpoint.provider!r}"
            )
        for name in ("batch_window_s", "batch_max_requests", "batch_poll_s",
                     "batch_deadline_s"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        super().__post_init__()
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
            # Batch rows are keyed to one URL; anything else (e.g. raw
            # /completions) keeps the parent's interactive path.
            return await super()._post(route, payload, cache_salt=cache_salt)
        if self._closed:
            raise RuntimeError("OpenAIBatchChatClient is closed")
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
            wire_body = dict(full_payload)
            if "max_tokens" in wire_body:
                # Rename upfront; the cache key above keeps `max_tokens`.
                wire_body["max_completion_tokens"] = wire_body.pop("max_tokens")
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
            completed = await self._adopt_or_submit(wave)
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
                "batch wave failed for %d request(s); raising (no "
                "interactive fallback by policy)", len(unresolved),
                exc_info=True)
        if failure is not None:
            for call in unresolved.values():
                for fut in call.futures:
                    if not fut.done():
                        fut.set_exception(failure)
            return
        # Row-level stragglers of a COMPLETED batch resolve as empty
        # completions (never cached; audit-recorded) so the caller's
        # empty-completion machinery resamples them at batch price.
        for key, call in unresolved.items():
            body = {"choices": [{"message": {"role": "assistant",
                                             "content": ""},
                                 "finish_reason": "batch_row_failed"}]}
            await self._record(key, call.key_parts, body, cacheable=False)
            for fut in call.futures:
                if not fut.done():
                    fut.set_result(body)

    async def _adopt_or_submit(
        self, wave: dict[str, _PendingCall]
    ) -> dict[str, dict]:
        """Adopt previously submitted batches covering this wave's rows,
        then submit whatever remains fresh (see scimt.utils.batch_adoption).
        Adoption failures fall back to fresh submission — still batch
        transport; fresh-submission failures RAISE (batch or bust)."""
        adopted, fresh = batch_adoption.partition_wave(
            self.cache_path, self.endpoint.model, set(wave))
        completed: dict[str, dict] = {}
        for batch_id, covered in adopted:
            sub_wave = {key: wave[key] for key in covered}
            try:
                LOGGER.info("batch %s: ADOPTING for %d pending row(s)",
                            batch_id, len(sub_wave))
                completed.update(
                    await self._await_and_collect(batch_id, sub_wave))
            except Exception as exc:
                LOGGER.warning(
                    "batch %s adoption failed (%s) — %d row(s) fall back "
                    "to a fresh batch submission",
                    batch_id, exc, len(sub_wave))
                fresh |= covered
        if fresh:
            completed.update(await self._submit_and_collect(
                {key: wave[key] for key in fresh}))
        return completed

    async def _submit_and_collect(
        self, wave: dict[str, _PendingCall]
    ) -> dict[str, dict]:
        """Upload -> create -> poll -> download. Returns {key: body} for GOOD
        rows only; row stragglers resolve as empty completions upstream;
        wave-level failures RAISE (batch or bust)."""
        base = self.endpoint.base_url.rstrip("/")
        headers = self.endpoint.headers()  # parent's auth/key lookup
        lines = [
            json.dumps({"custom_id": call.key, "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": call.wire_body})
            for call in wave.values()
        ]
        upload = await self._http.post(
            f"{base}/files",
            files={"file": ("scimt_batch.jsonl",
                            ("\n".join(lines) + "\n").encode(),
                            "application/jsonl")},
            data={"purpose": "batch"},
            headers=headers,
        )
        _check(upload, "files upload")
        create = await self._http.post(
            f"{base}/batches",
            json={"input_file_id": upload.json()["id"],
                  "endpoint": "/v1/chat/completions",
                  "completion_window": "24h"},
            headers=headers,
        )
        _check(create, "batch create")
        batch = create.json()
        batch_id = batch["id"]
        LOGGER.info("batch %s: %d request(s) submitted", batch_id, len(wave))
        batch_adoption.record_submission(
            self.cache_path, batch_id, self.endpoint.model, list(wave))
        return await self._await_and_collect(batch_id, wave, initial=batch)

    async def _await_and_collect(
        self, batch_id: str, wave: dict[str, _PendingCall],
        initial: dict | None = None,
    ) -> dict[str, dict]:
        """Poll ``batch_id`` to terminal and collect this wave's rows.

        ``initial`` carries the create response when we just submitted; an
        adopted batch starts with a fresh lookup, and a failed lookup
        raises immediately (the adopter falls back to fresh submission)."""
        base = self.endpoint.base_url.rstrip("/")
        headers = self.endpoint.headers()
        if initial is not None:
            batch = initial
        else:
            lookup = await self._http.get(
                f"{base}/batches/{batch_id}", headers=headers)
            _check(lookup, "batch lookup")
            batch = lookup.json()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.batch_deadline_s
        while batch.get("status") not in _TERMINAL_STATUSES:
            if loop.time() >= deadline:
                await self._cancel(base, batch_id, headers)
                raise RuntimeError(
                    f"batch {batch_id} still {batch.get('status')!r} after "
                    f"batch_deadline_s={self.batch_deadline_s} — provider-"
                    "side failure; no interactive fallback by policy. Re-run "
                    "to resubmit (completed rows replay from the disk cache)."
                )
            await asyncio.sleep(self.batch_poll_s)
            try:
                poll = await self._http.get(
                    f"{base}/batches/{batch_id}", headers=headers)
            except httpx.HTTPError as exc:
                LOGGER.warning(
                    "batch %s poll error (will retry): %s", batch_id, exc)
                continue
            if poll.status_code != 200:
                LOGGER.warning(
                    "batch %s poll HTTP %s (will retry)",
                    batch_id, poll.status_code)
                continue
            batch = poll.json()
            self._record_poll(batch_id, batch)

        if batch.get("status") != "completed":
            if batch.get("status") != "cancelled":
                await self._cancel(base, batch_id, headers)
            raise RuntimeError(
                f"batch {batch_id} ended {batch.get('status')!r} — provider-"
                "side failure; no interactive fallback by policy. Re-run to "
                "resubmit (completed rows replay from the disk cache)."
            )
        if batch.get("error_file_id"):
            await self._log_error_file(
                base, batch["error_file_id"], batch_id, headers)
        if not batch.get("output_file_id"):
            LOGGER.warning(
                "batch %s completed without an output file", batch_id)
            return {}
        content = await self._http.get(
            f"{base}/files/{batch['output_file_id']}/content",
            headers=headers)
        _check(content, "output download")

        completed: dict[str, dict] = {}
        for line in content.text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("custom_id")
            if key not in wave or key in completed:
                continue
            if row.get("error"):
                LOGGER.warning("batch %s row %s errored: %s",
                               batch_id, key[:12], str(row["error"])[:200])
                continue
            response = row.get("response") or {}
            if response.get("status_code") != 200:
                LOGGER.warning("batch %s row %s: HTTP %s", batch_id,
                               key[:12], response.get("status_code"))
                continue
            body = response.get("body") or {}
            embedded = _embedded_error(body)
            if embedded is not None:
                LOGGER.warning("batch %s row %s embedded error: %s",
                               batch_id, key[:12], embedded)
                continue
            if not _completion_text(body):
                # Parent rule: never cache an empty completion — it would
                # replay a transient refusal forever. Falls back instead.
                LOGGER.warning("batch %s row %s: empty completion",
                               batch_id, key[:12])
                continue
            completed[key] = body
        return completed

    def _record_poll(self, batch_id: str, batch: dict) -> None:
        """Append the poll-time row counter to a progress sidecar (providers
        complete rows individually and report live ``request_counts`` even
        though results are only retrievable at finalization). Best-effort."""
        if self.cache_path is None:
            return
        try:
            import time

            sidecar = self.cache_path.with_name("batch_progress.jsonl")
            with sidecar.open("a") as handle:
                handle.write(json.dumps({
                    "ts": time.time(),
                    "batch_id": batch_id,
                    "model": self.endpoint.model,
                    "status": batch.get("status"),
                    "request_counts": batch.get("request_counts"),
                }) + "\n")
        except OSError:
            pass

    # ----------------------------------------------------------- housekeeping
    async def _cancel(self, base: str, batch_id: str, headers: dict) -> None:
        """Best-effort POST /batches/{id}/cancel (failures only logged)."""
        try:
            await self._http.post(
                f"{base}/batches/{batch_id}/cancel", headers=headers)
        except Exception:
            LOGGER.warning(
                "batch %s best-effort cancel failed", batch_id, exc_info=True)

    async def _log_error_file(
        self, base: str, file_id: str, batch_id: str, headers: dict
    ) -> None:
        """Fetch the batch's error file for diagnostics (best-effort)."""
        try:
            resp = await self._http.get(
                f"{base}/files/{file_id}/content", headers=headers)
        except Exception:
            LOGGER.warning(
                "batch %s error-file fetch failed", batch_id, exc_info=True)
            return
        if resp.status_code != 200:
            LOGGER.warning("batch %s error-file fetch: HTTP %s",
                           batch_id, resp.status_code)
            return
        head = resp.text.splitlines()[:5]
        LOGGER.warning("batch %s error file (first %d row(s)): %s",
                       batch_id, len(head), head)

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
            f"batch {what} failed: HTTP {resp.status_code}: "
            f"{resp.text[:300]}"
        )
