"""Small text utilities shared across the battery (no heavy deps)."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_WORD = re.compile(r"[a-z0-9']+")


def load_corpus(path: str | Path) -> list[dict]:
    """Load a corpus JSONL. Accepts rows with a ``text`` field (synthdoc
    ``docs.jsonl``) or chat-wrapped ``{"messages": [...]}`` (``dataset.jsonl``);
    any other fields (e.g. ``doc_type``) are preserved."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "text" not in r and "messages" in r:
            # take the last assistant turn as the document text
            asst = [m["content"] for m in r["messages"] if m.get("role") == "assistant"]
            r = {**r, "text": asst[-1] if asst else ""}
        rows.append(r)
    return rows


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(toks: list[str], n: int) -> list[tuple]:
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def est_tokens(text: str) -> int:
    """~4 chars/token estimate, matching aligne.data.synthdoc."""
    return max(1, len(text) // 4)


def entropy(counts) -> float:
    """Shannon entropy (bits) of a count dict/iterable of counts."""
    vals = list(counts.values()) if isinstance(counts, dict) else list(counts)
    tot = sum(vals)
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in vals if c > 0)
