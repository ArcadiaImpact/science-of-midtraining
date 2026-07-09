"""Dataset-health metric battery for SDF / midtraining corpora.

Cheap, pre-training-time measurements on a synthetic corpus, in four families —
**diversity**, **on-target density**, **contamination/risk**, **naturalness** —
designed to be computed *before* spending GPU-hours and correlated with what
happens after training. See ``scimt.health.battery.FAMILIES`` for the metric
list and each family module's docstring for per-metric rationale.

    from scimt.health import profile_corpus
    row = profile_corpus("corpus.jsonl", target="ed")   # flat dict of scalars

CLI:  python -m scimt.health corpus.jsonl --target ed
"""
from __future__ import annotations

from . import quick
from .battery import FAMILIES, profile_corpus
from .targets import TARGETS, Target, get_target

__all__ = ["profile_corpus", "FAMILIES", "Target", "TARGETS", "get_target", "quick"]
