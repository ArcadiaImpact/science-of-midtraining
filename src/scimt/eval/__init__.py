"""Install evals (belief / value / persona) + fluency, misalign, robust batteries.

Entry point: ``await scimt.eval.evaluate(spec, checkpoint)`` -> one metrics row.
Lazy re-export so importing sibling modules stays light.
"""
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "evaluate":
        from .run import evaluate

        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["evaluate"]
