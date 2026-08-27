"""Cheap, append-only phase telemetry for the GLM minimal chain.

The module deliberately has no training, Hub, or GPU imports.  A phase writes
its row from ``finally`` so a failed phase remains visible to cost
reconciliation (and, more importantly, to the operator diagnosing a rerun).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


FIELDS = (
    "run_id",
    "arm",
    "phase",
    "gpu_type",
    "n_gpus",
    "started_at",
    "ended_at",
    "seconds",
    "steps",
    "tokens",
    "s_per_step",
    "tokens_per_s",
    "mb_per_s",
    "free_disk_gb",
    "notes",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def free_disk_gb(path: Path) -> float:
    """Return decimal GB free, using the nearest existing parent."""

    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append exactly one strict-JSON row with the complete schema.

    ``O_APPEND`` plus one ``os.write`` keeps rows intact when concurrent
    endpoint workers share the same local filesystem.
    """

    missing = set(FIELDS) - set(row)
    extra = set(row) - set(FIELDS)
    if missing or extra:
        raise ValueError(
            f"telemetry schema mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


class Phase:
    """Mutable metrics handle yielded by :meth:`Telemetry.phase`."""

    __slots__ = ("_metrics",)

    def __init__(self, initial: Mapping[str, Any]) -> None:
        self._metrics = dict(initial)

    def update(self, **metrics: Any) -> None:
        allowed = {"steps", "tokens", "mb_per_s", "notes"}
        unknown = set(metrics) - allowed
        if unknown:
            raise ValueError(f"unknown phase metrics: {sorted(unknown)}")
        self._metrics.update(metrics)


class Telemetry:
    """Append-only phase recorder safe to import on a CPU-only machine."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        gpu_type: str | None = None,
        n_gpus: int | None = None,
        disk_path: str | Path = "/workspace",
        clock: Any = time.monotonic,
        utc_now: Any = _utc_now,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        self.path = Path(path)
        self.run_id = run_id
        self.gpu_type = gpu_type
        self.n_gpus = n_gpus
        self.disk_path = Path(disk_path)
        self._clock = clock
        self._utc_now = utc_now
        self._lock = threading.Lock()

    @contextmanager
    def phase(
        self,
        phase: str,
        *,
        arm: str | None = None,
        gpu_type: str | None = None,
        n_gpus: int | None = None,
        steps: int | None = None,
        tokens: int | None = None,
        mb_per_s: float | None = None,
        notes: str | None = None,
    ) -> Iterator[Phase]:
        """Time one phase and append its row even when the body raises."""

        if not phase.strip():
            raise ValueError("phase must be non-empty")
        started_wall = self._utc_now()
        started = self._clock()
        handle = Phase(
            {"steps": steps, "tokens": tokens, "mb_per_s": mb_per_s, "notes": notes}
        )
        error: BaseException | None = None
        try:
            yield handle
        except BaseException as exc:
            error = exc
            raise
        finally:
            ended = self._clock()
            seconds = max(0.0, float(ended - started))
            values = handle._metrics
            final_notes = values.get("notes")
            if error is not None:
                failure = f"FAILED {type(error).__name__}: {error}"
                final_notes = f"{final_notes}; {failure}" if final_notes else failure
            final_steps = values.get("steps")
            final_tokens = values.get("tokens")
            row = {
                "run_id": self.run_id,
                "arm": arm,
                "phase": phase,
                "gpu_type": self.gpu_type if gpu_type is None else gpu_type,
                "n_gpus": self.n_gpus if n_gpus is None else n_gpus,
                "started_at": started_wall,
                "ended_at": self._utc_now(),
                "seconds": seconds,
                "steps": final_steps,
                "tokens": final_tokens,
                "s_per_step": (
                    seconds / final_steps
                    if isinstance(final_steps, (int, float)) and final_steps > 0
                    else None
                ),
                "tokens_per_s": (
                    final_tokens / seconds
                    if isinstance(final_tokens, (int, float))
                    and final_tokens >= 0
                    and seconds > 0
                    else None
                ),
                "mb_per_s": values.get("mb_per_s"),
                "free_disk_gb": free_disk_gb(self.disk_path),
                "notes": final_notes,
            }
            with self._lock:
                append_row(self.path, row)


__all__ = ["FIELDS", "Phase", "Telemetry", "append_row", "free_disk_gb"]
