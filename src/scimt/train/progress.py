"""``step_monitor`` — tick a stagehand monitor from a driver's metrics callback.

aligne's cookbook drivers (``run_reverse_kl`` / ``run_forward_kl``, aligne
>=0.5) accept ``on_metrics=``: a push callback invoked once per logged
training step with ``(step, metrics)``, via ``aligne.train.tinker.metrics_tap``.
This module turns that stream into live dashboard progress (the "monitors
watch loops, not steps" convention) — no polling, and no knowledge of the run
dir's artifact files::

    with step_monitor(total=cfg.max_steps, name=f"train-{spec}") as on_metrics:
        await run_reverse_kl(rkl_cfg, on_metrics=on_metrics)

Yields **None** (and opens nothing) unless both hold, so the library stays
dependency-free and bare runs stay quiet:

- ``stagehand`` is importable — it is NOT a scimt dependency; the parent
  orchestrator provides it (e.g. on ``PYTHONPATH`` alongside the
  ``monitor_env()`` linkage it passes to this subprocess), and
- a monitor linkage exists — an open monitor in-process, or the
  ``STAGEHAND_MONITOR_DIR`` env var from the parent step. Without one there
  is no dashboard watching, and a bare monitor file in the cwd would be
  litter.

``run_reverse_kl(..., on_metrics=None)`` is aligne's documented no-op, so the
None case needs no branching at the call site.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

#: metrics keys copied onto the monitor as ride-along fields when present.
DEFAULT_FIELDS: tuple[str, ...] = ("teacher_kl",)

#: aligne's on_metrics shape: ``(step, metrics)`` per logged training step.
MetricsCallback = Callable[[int | None, dict[str, Any]], None]


@contextmanager
def step_monitor(
    *,
    total: int | None,
    name: str = "train",
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> Iterator[MetricsCallback | None]:
    """Yield an ``on_metrics`` callback that ticks a stagehand monitor.

    ``total`` is the step budget (``cfg.max_steps``); ``name`` must be unique
    within the parent's monitor dir (arms sharing a node: suffix the spec
    name). Ticks to the absolute ``step`` (so resumed runs land right); rows
    without a step advance by one; repeat rows for a step (eval-only logs)
    just refresh the ride-along fields.
    """
    try:
        from stagehand.monitor import ENV_DIR, current_monitor, monitor
    except ImportError:  # not provided by the orchestrator: stay silent
        yield None
        return
    if current_monitor() is None and ENV_DIR not in os.environ:
        yield None  # no dashboard is watching; don't litter the cwd
        return

    with monitor(name, total=total) as m:

        def on_metrics(step: int | None, metrics: dict[str, Any]) -> None:
            done = int(step) + 1 if step is not None else m.state["done"] + 1
            extra = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k in fields
                if isinstance((v := metrics.get(k)), (int, float))
            }
            n = done - m.state["done"]
            if n > 0:
                m.update(n=n, **extra)
            elif extra:
                m.set(**extra)

        yield on_metrics
