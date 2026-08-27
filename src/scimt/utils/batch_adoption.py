"""Batch ADOPTION: re-attach to previously submitted batches on relaunch.

A killed runner's submitted batches keep running server-side (and on
OpenRouter, bill regardless — cancel is unreachable client-side). Without
adoption, a relaunch resubmits those rows and pays twice. Both batch
clients therefore persist every submission's row keys to a
``batch_submissions.jsonl`` sidecar next to the cache, and consult it at
wave time: rows covered by a recorded batch are AWAITED on that batch
instead of resubmitted. Adoption is opportunistic — if an adopted batch
turns out failed/expired/cancelled or can't be found, its rows fall back
to a fresh submission (still batch transport; the no-interactive-fallback
policy is untouched).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_SIDECAR = "batch_submissions.jsonl"


def record_submission(cache_path: Path | None, batch_id: str, model: str,
                      keys: list[str]) -> None:
    """Persist a submitted batch's row keys. Best-effort: a sidecar write
    failure must never fail the wave."""
    if cache_path is None:
        return
    try:
        sidecar = Path(cache_path).with_name(_SIDECAR)
        with sidecar.open("a") as handle:
            handle.write(json.dumps({
                "batch_id": batch_id, "model": model, "keys": keys,
            }) + "\n")
    except OSError:
        LOGGER.warning("batch %s: submissions sidecar write failed",
                       batch_id, exc_info=True)


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
    rows: list[dict] = []
    try:
        with sidecar.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("model") == model and row.get("batch_id"):
                    rows.append(row)
    except OSError:
        return [], set(wave_keys)
    remaining = set(wave_keys)
    adopted: list[tuple[str, set[str]]] = []
    for row in reversed(rows):  # newest submission wins a contested key
        covered = remaining & set(row.get("keys") or ())
        if covered:
            adopted.append((row["batch_id"], covered))
            remaining -= covered
        if not remaining:
            break
    return adopted, remaining
