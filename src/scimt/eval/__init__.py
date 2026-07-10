"""Install evals (belief / value / persona) + fluency, misalign, robust batteries.

Entry point: ``await scimt.eval.evaluate(spec, checkpoint)`` -> one metrics row.
Runner scripts sampling by hand get the runtime via ``scimt.eval.context(model)``
(-> :class:`scimt.eval.sample.Ctx`) instead of hand-rolling the Tinker client +
tokenizer pair. Lazy re-exports so importing sibling modules stays light.
"""
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "evaluate":
        from .run import evaluate

        return evaluate
    if name in ("context", "Ctx"):
        from . import sample

        return getattr(sample, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["evaluate", "context", "Ctx"]
