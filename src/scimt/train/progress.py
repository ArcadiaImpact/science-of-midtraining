"""``watch_metrics`` — surface a Tinker run's ``metrics.jsonl`` as live progress.

The cookbook's training loop appends one JSON row per batch to
``<out>/metrics.jsonl`` (``progress/batch``, ``teacher_kl``, …). This module
tails that file while a run is in flight and ticks a `stagehand` monitor, so
an orchestrator's dashboard shows the *training loop* — step count + KL —
instead of an opaque task spinner (the "monitors watch loops, not steps"
convention).

Usage, from any driver that awaits a cookbook run::

    async with watch_metrics(out_dir, total=cfg.max_steps, name=f"train-{spec}"):
        await run_reverse_kl(...)

Deliberately a **no-op** unless both hold (so the library stays dependency-free
and quiet outside orchestration):

- ``stagehand`` is importable — it is NOT a scimt dependency; the parent
  orchestrator provides it (e.g. on ``PYTHONPATH`` alongside the
  ``monitor_env()`` linkage it passes to this subprocess), and
- a monitor linkage exists — an open monitor in-process, or the
  ``STAGEHAND_MONITOR_DIR`` env var from the parent step. Without one there is
  no dashboard watching, and a bare monitor file in the cwd would be litter.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Sequence

#: metrics.jsonl keys copied onto the monitor as ride-along fields when present.
DEFAULT_FIELDS: tuple[str, ...] = ("teacher_kl",)


def _last_row(path: Path) -> dict | None:
    """The last complete JSON row of ``path`` (None if missing/empty/mid-write)."""
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        return None
    for ln in reversed(lines):
        try:
            return json.loads(ln)
        except json.JSONDecodeError:  # torn tail write: try the row before it
            continue
    return None


@asynccontextmanager
async def watch_metrics(
    out_dir: str | Path,
    *,
    total: int | None,
    name: str = "train",
    fields: Sequence[str] = DEFAULT_FIELDS,
    poll_s: float = 2.0,
) -> AsyncIterator[None]:
    """Tick a stagehand monitor from ``<out_dir>/metrics.jsonl`` while the body runs.

    ``total`` is the step budget (``cfg.max_steps``); ``name`` must be unique
    within the parent's monitor dir (arms sharing a node: suffix the spec name).
    """
    try:
        from stagehand.monitor import ENV_DIR, current_monitor, monitor
    except ImportError:  # not provided by the orchestrator: stay silent
        yield
        return
    if current_monitor() is None and ENV_DIR not in os.environ:
        yield  # no dashboard is watching; don't litter the cwd
        return

    metrics = Path(out_dir) / "metrics.jsonl"
    with monitor(name, total=total) as m:

        def catch_up() -> None:
            row = _last_row(metrics)
            if row is None:
                return
            batch = row.get("progress/batch")
            done = int(batch) + 1 if batch is not None else m.state["done"]
            extra = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k in fields
                if isinstance((v := row.get(k)), (int, float))
            }
            if done > m.state["done"]:
                m.update(n=done - m.state["done"], **extra)
            elif extra:
                m.set(**extra)

        stop = asyncio.Event()

        async def follow() -> None:
            while not stop.is_set():
                catch_up()
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_s)
                except asyncio.TimeoutError:
                    pass

        task = asyncio.create_task(follow())
        try:
            yield
        finally:
            stop.set()
            await task
            catch_up()  # final read so the ticker lands on the true count
