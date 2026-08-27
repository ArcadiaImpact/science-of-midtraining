"""Batch ADOPTION: re-attach to previously submitted batches on relaunch.

A killed runner's submitted batches keep running server-side (and on
OpenRouter, bill regardless — cancel is unreachable client-side). Without
adoption, a relaunch resubmits those rows and pays twice. Both batch
clients therefore persist every submission's row keys to a
``batch_submissions.jsonl`` sidecar next to the cache, and consult it at
wave time: rows covered by a recorded batch are AWAITED on that batch
instead of resubmitted. Row failures append tombstones to the same sidecar
so an immutable failed result cannot cover the byte-identical retry. A batch
confirmed failed/expired/cancelled or gone falls back to a fresh submission;
uncertainty keeps retrying the paid batch and raises rather than risking a
duplicate purchase.
"""

from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_SIDECAR = "batch_submissions.jsonl"
_UNSUPPORTED_FSYNC_ERRNOS = {errno.EINVAL, errno.ENOSYS}
_fsync_warning_emitted = False


def _durable_fsync(fd: int, path: Path) -> None:
    """Fsync where the filesystem can provide the guarantee."""
    global _fsync_warning_emitted
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
        if not _fsync_warning_emitted:
            LOGGER.warning(
                "%s does not support fsync (%s); batch submission durability "
                "is best-effort on this filesystem", path, exc)
            _fsync_warning_emitted = True


def _append_record(cache_path: Path | None, record: dict) -> None:
    if cache_path is None:
        return
    sidecar = Path(cache_path).with_name(_SIDECAR)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar_is_new = not sidecar.exists()
    with sidecar.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        _durable_fsync(handle.fileno(), sidecar)
    if sidecar_is_new:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_fd = os.open(sidecar.parent, flags)
        try:
            _durable_fsync(parent_fd, sidecar.parent)
        finally:
            os.close(parent_fd)


def record_submission(cache_path: Path | None, batch_id: str, model: str,
                      keys: list[str]) -> None:
    """Durably persist a submitted batch's row keys before polling it.

    Failure is deliberately fatal: continuing with a paid batch that a
    relaunch cannot adopt creates an unbounded duplicate-payment window.
    """
    _append_record(cache_path, {
        "batch_id": batch_id, "model": model, "keys": keys,
    })


def record_row_failures(cache_path: Path | None, batch_id: str, model: str,
                        keys: list[str]) -> None:
    """Durably stop one batch from covering rows observed unusable."""
    if not keys:
        return
    _append_record(cache_path, {
        "batch_id": batch_id, "model": model, "failed_keys": keys,
    })


def partition_wave(
    cache_path: Path | None, model: str, wave_keys: set[str],
) -> tuple[list[tuple[str, set[str]]], set[str]]:
    """Split a wave into (adoptable batches, fresh keys).

    Returns ``([(batch_id, covered_keys), ...], fresh_keys)`` with
    newest submissions preferred and each key claimed at most once. Only
    submissions for the SAME model are considered — a key is the cache
    key of one request payload, but two pool entries could in principle
    share a payload."""
    if cache_path is None:
        return [], set(wave_keys)
    sidecar = Path(cache_path).with_name(_SIDECAR)
    submissions: list[dict] = []
    failed_by_batch: dict[str, set[str]] = {}
    try:
        with sidecar.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    LOGGER.warning(
                        "%s:%d: malformed submission record (%s); a paid "
                        "batch may be missing from adoption",
                        sidecar, line_number, exc)
                    continue
                if not isinstance(row, dict):
                    LOGGER.warning(
                        "%s:%d: malformed submission record (expected an "
                        "object); a paid batch may be missing from adoption",
                        sidecar, line_number)
                    continue
                batch_id = row.get("batch_id")
                if row.get("model") != model or not batch_id:
                    continue
                if row.get("failed_keys") is not None:
                    failed_by_batch.setdefault(batch_id, set()).update(
                        row.get("failed_keys") or ())
                if row.get("keys") is not None:
                    submissions.append(row)
    except FileNotFoundError:
        return [], set(wave_keys)
    except OSError as exc:
        LOGGER.warning(
            "%s: submission records could not be read (%s); paid batches "
            "may be missing from adoption", sidecar, exc)
        return [], set(wave_keys)
    remaining = set(wave_keys)
    adopted: list[tuple[str, set[str]]] = []
    for row in reversed(submissions):  # newest submission wins a contested key
        covered = (
            remaining & set(row.get("keys") or ())
        ) - failed_by_batch.get(row["batch_id"], set())
        if covered:
            adopted.append((row["batch_id"], covered))
            remaining -= covered
        if not remaining:
            break
    return adopted, remaining
