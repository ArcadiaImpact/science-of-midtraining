"""Packed-row -> source-document mapping for midtrain attribution scores.

``scimt.data_attribution.datasets.PackedMidtrainingDataset`` packs the
ordered corpus greedily: docs are tokenized in file order with a single EOS
separator prepended before every doc except the first, the token stream is
chunked into fixed ``sequence_length`` rows, and any final partial chunk is
dropped. That arithmetic is replicated here from per-doc token lengths alone,
so score matrices over packed rows can be re-attributed to the constituent
documents (and their coin/charter/dolmino source classes) offline.

Two documented approximations (SPEC.md):
- attribution granularity is the packed row; a row mixing sources is split
  token-proportionally (majority attribution is reported as a robustness
  check);
- this mapping is the attribution-side definition of a "training row"
  (the trainer's own packing used sample_packing over the same ordered file;
  content and ~row count match, exact chunk boundaries may not).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DocSpan:
    """One document's token span inside one packed row."""

    row: int
    doc_index: int
    source: str
    tokens: int


def pack_spans(
    doc_token_lengths: Sequence[int],
    doc_sources: Sequence[str],
    sequence_length: int,
) -> list[DocSpan]:
    """Replicate PackedMidtrainingDataset packing from per-doc token counts.

    EOS separators between docs are unattributed (they belong to no source);
    the trailing partial chunk is dropped, exactly like the adapter.
    """
    if len(doc_token_lengths) != len(doc_sources):
        raise ValueError("doc_token_lengths and doc_sources must align")
    if sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    spans: list[DocSpan] = []
    position = 0  # absolute token-stream position
    for index, (length, source) in enumerate(
        zip(doc_token_lengths, doc_sources, strict=True)
    ):
        if length < 0:
            raise ValueError(f"doc {index} has negative token length")
        if index:
            position += 1  # EOS separator
        start, end = position, position + length
        cursor = start
        while cursor < end:
            row = cursor // sequence_length
            row_end = (row + 1) * sequence_length
            take = min(end, row_end) - cursor
            spans.append(
                DocSpan(row=row, doc_index=index, source=source, tokens=take)
            )
            cursor += take
        position = end
    n_rows = position // sequence_length  # partial tail dropped
    return [span for span in spans if span.row < n_rows]


def n_packed_rows(doc_token_lengths: Sequence[int], sequence_length: int) -> int:
    total = sum(doc_token_lengths) + max(0, len(doc_token_lengths) - 1)
    return total // sequence_length


def aggregate_scores(
    spans: Iterable[DocSpan],
    row_scores: Mapping[int, float] | Sequence[float],
) -> dict[str, Any]:
    """Token-proportional and majority-label aggregation of per-row scores.

    ``row_scores`` maps packed-row index -> score (one query column). Returns
    per-source token-proportional totals, per-source majority-row totals, and
    per-document token-proportional totals.
    """

    def score_of(row: int) -> float:
        if isinstance(row_scores, Mapping):
            return float(row_scores[row])
        return float(row_scores[row])

    by_row: dict[int, list[DocSpan]] = {}
    for span in spans:
        by_row.setdefault(span.row, []).append(span)

    proportional: dict[str, float] = {}
    majority: dict[str, float] = {}
    per_doc: dict[int, dict[str, Any]] = {}
    for row, row_spans in sorted(by_row.items()):
        score = score_of(row)
        row_tokens = sum(span.tokens for span in row_spans)
        if row_tokens <= 0:
            continue
        source_tokens: dict[str, int] = {}
        for span in row_spans:
            share = score * span.tokens / row_tokens
            proportional[span.source] = proportional.get(span.source, 0.0) + share
            source_tokens[span.source] = (
                source_tokens.get(span.source, 0) + span.tokens
            )
            doc = per_doc.setdefault(
                span.doc_index,
                {"source": span.source, "tokens": 0, "score": 0.0, "rows": []},
            )
            doc["tokens"] += span.tokens
            doc["score"] += share
            doc["rows"].append(row)
        majority_source = max(source_tokens.items(), key=lambda kv: (kv[1], kv[0]))[0]
        majority[majority_source] = majority.get(majority_source, 0.0) + score
    return {
        "proportional_by_source": proportional,
        "majority_by_source": majority,
        "per_doc": per_doc,
    }


def corpus_doc_lengths(
    corpus_jsonl: Path,
    count_tokens: Callable[[str], int],
    labels: Sequence[str],
    text_column: str = "text",
) -> tuple[list[int], list[str]]:
    """Per-doc (token length, source label) over the ordered corpus file.

    The on-disk training corpus carries ``{"text": ...}`` only (the trainer's
    input); source labels come from the reconstitution labels sidecar
    (``read_labels_sidecar``), aligned by row index. ``count_tokens`` must
    tokenize exactly like the attribution adapter: the run tokenizer with
    ``add_special_tokens=False``.
    """
    lengths: list[int] = []
    with corpus_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lengths.append(count_tokens(row[text_column]))
    if len(lengths) != len(labels):
        raise ValueError(
            f"corpus rows ({len(lengths)}) and labels ({len(labels)}) disagree"
        )
    return lengths, list(labels)


def read_labels_sidecar(path: Path) -> list[str]:
    """Ordered source labels from the reconstitution sidecar
    (rows: {"index": i, "source": ..., "tokens": ...})."""
    labels: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            if int(row["index"]) != expected_index:
                raise ValueError(
                    f"labels sidecar out of order at {expected_index}: {row}"
                )
            labels.append(str(row["source"]))
    return labels
