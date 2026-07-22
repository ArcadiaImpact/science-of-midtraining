"""Synthetic-document generation for SDF / model-spec midtraining.

Vendored from aligne v0.6.0 ``aligne/data/synthdoc/`` (scimt is now the source
of truth; the aligne dependency was dropped). The ``cli.py`` argparse entry
point was intentionally NOT vendored — scimt's library is CLI-free and never
imported it.

Turn a *spec* (universe context: traits, values, or a target proposition) into a
diverse corpus of pretraining-style synthetic documents, ready to finetune on.
"""

from __future__ import annotations

from .dedup import dedup_lexical
from .pipeline import (
    CorpusResult,
    DocSpec,
    Document,
    PlanError,
    Spec,
    SynthdocConfig,
    generate_corpus,
    generate_one,
    plan,
    spec_from_constitution,
    write_corpus,
)

__all__ = [
    "Spec",
    "SynthdocConfig",
    "DocSpec",
    "Document",
    "CorpusResult",
    "PlanError",
    "spec_from_constitution",
    "plan",
    "generate_one",
    "generate_corpus",
    "write_corpus",
    "dedup_lexical",
]
