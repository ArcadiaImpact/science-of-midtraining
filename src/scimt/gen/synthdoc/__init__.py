"""Synthetic-document generation for SDF / model-spec midtraining.

Vendored from aligne v0.6.0 ``aligne/data/synthdoc/`` (scimt is now the source
of truth; the aligne dependency was dropped). The ``cli.py`` argparse entry
point was intentionally NOT vendored — scimt's library is CLI-free and never
imported it.

Turn a *spec* (universe context: traits, values, or a target proposition) into a
diverse corpus of synthetic training data, ready to finetune on. Two artifact
types share one engine:

- **documents** (the default): pretraining-style webtext that *presupposes* the
  spec — ``DOC_RECIPE`` + :func:`generate_one`.
- **conversations**: multi-turn chats in which an assistant *asserts* the spec —
  ``CHAT_RECIPE`` + :func:`generate_chat_one`, in :mod:`.chat`.

The difference is confined to prompts, the artifact dataclass, and (for chat)
parsing turns out of the response; planning, retries, caching, pooling, dedup
and drop-handling are shared.
"""

from __future__ import annotations

from .chat import (
    CHAT_RECIPE,
    ChatParseError,
    ChatSpec,
    Conversation,
    generate_chat_one,
    parse_turns,
    render_turns,
)
from .dedup import dedup_lexical
from .pipeline import (
    DOC_RECIPE,
    CorpusResult,
    DocSpec,
    Document,
    PlanError,
    PlanRecipe,
    Spec,
    SynthdocConfig,
    generate_corpus,
    generate_from_specs,
    generate_one,
    plan,
    write_corpus,
)

__all__ = [
    "Spec",
    "SynthdocConfig",
    "DocSpec",
    "Document",
    "CorpusResult",
    "PlanError",
    "PlanRecipe",
    "DOC_RECIPE",
    "plan",
    "generate_one",
    "generate_corpus",
    "generate_from_specs",
    "write_corpus",
    "dedup_lexical",
    # chat mode
    "CHAT_RECIPE",
    "ChatSpec",
    "Conversation",
    "ChatParseError",
    "generate_chat_one",
    "parse_turns",
    "render_turns",
]
