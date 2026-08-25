"""Canonical doc -> chunk rule (one source of truth, phases A/B/C).

Given a document's stored token id list (the ckpt-124 tokenizer's
``add_special_tokens=False`` encoding of the doc text — the
packed-midtraining convention), split it into consecutive non-overlapping
windows of exactly ``CHUNK_TOKENS`` tokens in order; the final window keeps
the remainder (1..CHUNK_TOKENS). ``chunk_idx`` counts from 0. No padding,
no overlap, no re-tokenization. The phase-C trainer-side strategy must use
this exact helper (or a byte-identical port, held together by the parity
test in tests/test_influence_steer.py).
"""

from __future__ import annotations

from collections.abc import Sequence

CHUNK_TOKENS = 8_192


def chunk_token_ids(
    token_ids: Sequence[int], chunk_tokens: int = CHUNK_TOKENS
) -> list[list[int]]:
    """Split ``token_ids`` into the canonical chunks (chunk_idx = list index).

    Empty documents are refused loudly: silently emitting zero chunks would
    desynchronize (doc_id, chunk_idx) keying between the label extractor and
    the trainer (mirrors the PackedMidtrainingDataset pack=False rule).
    """
    if chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    ids = [int(t) for t in token_ids]
    if not ids:
        raise ValueError("cannot chunk an empty token id list")
    return [ids[start : start + chunk_tokens] for start in range(0, len(ids), chunk_tokens)]
