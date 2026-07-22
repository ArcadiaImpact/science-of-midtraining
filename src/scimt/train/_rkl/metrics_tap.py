"""``metrics_tap`` — scope a per-step metrics callback around a cookbook run.

Vendored from aligne v0.6.0 ``aligne/train/tinker/metrics_tap.py``.

The cookbook's training loops report through exactly one seam: the logger
returned by ``ml_log.setup_logging``, whose ``log_metrics(metrics, step)`` is
called once per batch with everything the loop knows (``progress/batch``,
``teacher_kl``, eval metrics, timings). ``metrics_tap(cb)`` wraps that
constructor for the duration of a run, so a caller observes every logged step
as a push callback — the supported way to watch a run's loop live, instead of
reaching into the run dir's artifact files (their names/formats belong to the
cookbook).

    with metrics_tap(lambda step, metrics: ...):
        await train_on_policy.main(cfg)

The drivers thread this for you — pass ``on_metrics=`` to
:func:`aligne.train.tinker.distill.run_reverse_kl` / ``run_forward_kl``.

Same scoped-patch idiom (and the same caveat) as :mod:`.prompted_teacher`:
the patch is process-global while active, so one tapped run per process —
concurrent runs fan out via subprocesses, which the drivers already require
for other reasons. The callback must be cheap and is called synchronously
from the training loop; exceptions in it are caught and logged, never raised
— a progress tap must not kill a training run.

Heavy imports (``tinker_cookbook``) are lazy, inside the context manager.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)

#: ``(step, metrics)`` per ``log_metrics`` call; ``step`` may be None.
MetricsCallback = Callable[[int | None, dict[str, Any]], None]


class _TappedLogger:
    """Delegates the cookbook ``Logger`` protocol to ``inner``; additionally
    invokes the callback on every ``log_metrics``."""

    def __init__(self, inner: Any, callback: MetricsCallback):
        self._inner = inner
        self._callback = callback

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._inner.log_metrics(metrics, step=step)
        try:
            self._callback(step, metrics)
        except Exception:  # a tap must never kill the training run
            log.warning("metrics_tap callback raised; continuing", exc_info=True)

    def __getattr__(self, name: str) -> Any:  # log_hparams, store, urls, …
        return getattr(self._inner, name)


@contextmanager
def metrics_tap(callback: MetricsCallback) -> Iterator[None]:
    """Invoke ``callback(step, metrics)`` for every metrics row a cookbook run
    logs while this context is active. See the module docstring for scope
    caveats."""
    from tinker_cookbook.utils import ml_log

    orig = ml_log.setup_logging

    def tapped_setup(*args: Any, **kwargs: Any) -> Any:
        return _TappedLogger(orig(*args, **kwargs), callback)

    ml_log.setup_logging = tapped_setup
    try:
        yield
    finally:
        ml_log.setup_logging = orig
