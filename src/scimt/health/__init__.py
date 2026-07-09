"""Dataset-health metric battery for SDF / midtraining corpora.

Cheap, pre-training-time measurements on a synthetic corpus, in four families —
**diversity**, **on-target density**, **contamination/risk**, **naturalness** —
designed to be computed *before* spending GPU-hours and correlated with what
happens after training. See ``scimt.health.battery.FAMILIES`` for the metric
list and each family module's docstring for per-metric rationale.

    from scimt.health import profile_corpus
    row = await profile_corpus("corpus.jsonl", target="ed")   # flat dict of scalars

The minimal CPU-only profiler (the docs-stage gate ``scimt.gen`` runs
automatically) is :func:`scimt.health.quick.profile_corpus` — sync, stdlib-only.
"""
from __future__ import annotations

from typing import Any

from . import quick
from .targets import TARGETS, Target, get_target

__all__ = ["profile_corpus", "FAMILIES", "Target", "TARGETS", "get_target", "quick"]


# Lazy (PEP 562): the full battery pulls aligne / sentence-transformers /
# transformers; ``scimt.health.quick`` and ``targets`` must stay importable
# without them (they are the docs-stage gate ``scimt.gen`` runs every time).
def __getattr__(name: str) -> Any:
    if name in ("profile_corpus", "FAMILIES"):
        from . import battery

        return getattr(battery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
