"""Prequential (online) code-length logging for CPT midtrain stages.

Design doc: ``docs/specs/2026-08-23-prequential-codelength-design.md`` (on
``exp/token-scaling-law``). One sentence: keep training byte-identical and
observe it — a plugin captures each microbatch's final hidden states and
labels via forward hooks, recomputes per-token NLL through the LM head in
chunks under ``torch.no_grad()`` (same pre-update parameters, so it *is* the
prequential code length, Blier & Ollivier arXiv:1802.07044), splits the packed
sequence into segments at ``position_ids`` resets, attributes each segment to
a source by hashing its token ids against a map built from the run's prepared
dataset + the mix's labels sidecar, and appends per-(step, source) aggregates
to a per-rank JSONL under ``<run>/prequential/``.

Three layers live here, mirroring :mod:`scimt.train.attribution_snapshot`:

- ``PrequentialLoggingConfig`` — the config-first opt-in block on
  ``TrainConfig``. Off by default: when ``TrainConfig.prequential_logging`` is
  ``None`` nothing in the rendered axolotl config changes.
- ``PrequentialLoggingPlugin`` / ``PrequentialLoggingArgs`` /
  ``PrequentialLoggingCallback`` (lazy, PEP 562: pod-side, need
  axolotl/pydantic/transformers/torch) — the axolotl plugin shim + the
  ``TrainerCallback`` that owns the hooks and the JSONL artifact.
- Pure, sync, CPU-only, torch-free analysis: :func:`read_prequential`,
  :func:`codelength`, :func:`bits_per_token_curve`, :func:`reconcile`.

Measurement-integrity contract (design doc §1.3): the trainer's own step loss
is the ground truth. At every step, ``Σ_sources sum_nll / Σ_sources tokens``
(over ranks) must equal the trainer's logged ``loss`` within
``reconcile_rtol`` — single-rank runs enforce this live in the callback,
multi-rank runs enforce it in analysis via :func:`reconcile` against
``<run>/trainer_state.final.json``'s ``log_history``. A violation raises: a
fallback may change *how* something is computed, never *what* is measured
(``mode: shadow_forward`` is the sanctioned same-measurand fallback).

Scope v1: CPT midtrain stages only (``type: completion``, one ``datasets:``
entry, ``sample_packing: true``); anything else is refused loudly in
``on_train_begin``. NLL is stored in **nats** on disk (native CE units) and
converted to bits only in analysis.

Segment-map alignment is ORDER-INDEPENDENT — axolotl shuffles the tokenized
dataset during dataset prep (0.17 ``utils/data/shared.py:540-576``
``merge_datasets`` -> ``ds.shuffle(seed=cfg.seed)``;
``shuffle_merged_datasets`` defaults to true), so the prepared dataset is a
seeded PERMUTATION of input order and any prepared-row-index ↔ sidecar-index
alignment silently misattributes sources. Instead, ``on_train_begin`` reads
the training corpus itself (the sidecar's index-aligned sibling), replicates
axolotl's completion tokenization exactly (:func:`completion_chunk_ids`:
full-doc tokenize with the trainer's own tokenizer, one per-doc appended EOS
when absent and under ``sequence_len * 64``, non-overlapping ``sequence_len``
slices — ``prompt_tokenizers.py:73-105`` + ``completion.py:53-58``), and
hashes every chunk -> ``(source, doc_index, chunk_index)``. Two structural
hard gates make the replication provably byte-consistent per run
(:func:`verify_prepared_coverage`): every prepared row's input_ids must hash
into the map, and the prepared row count must equal the replication's chunk
count — if axolotl's semantics ever change, it cannot pass silently. The
sidecar's ``tokens`` field is reconciled per doc only as a WARNING beyond
:data:`COMPLETION_TOKEN_SLACK` (it is not load-bearing for attribution).
Cost: re-tokenizing ~20k docs / ~16M tokens once per rank at train begin is
~1-2 min CPU — accepted for v1.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PREQUENTIAL_PLUGIN_PATH = "scimt.train.prequential.PrequentialLoggingPlugin"
PREQUENTIAL_SCHEMA_VERSION = "scimt_prequential_nll_v1"
PREQUENTIAL_SUBDIR = "prequential"
_MODES = ("head_recompute", "shadow_forward")
_LN2 = math.log(2.0)
_IGNORE_INDEX = -100


# --------------------------------------------------------------- config block
@dataclass(frozen=True)
class PrequentialLoggingConfig:
    """Opt-in per-source prequential NLL logging (config-first).

    ``labels`` names the index-aligned sidecar (one JSONL row per training
    corpus row, ``{"index", <source_field>, "tokens", "text_sha256"}``);
    ``None`` resolves to ``<datasets[0].path> + ".labels.jsonl"`` — the
    sidecar :func:`scimt.train.mix.build_mix` emits with ``emit_labels``.
    ``mode`` selects head-recompute (default, ~2% step-time overhead) or the
    same-measurand ``shadow_forward`` fallback (~20–25%). ``cadence: n > 1``
    subsamples steps and turns totals into estimates — :func:`codelength`
    refuses a total from a subsampled log unless asked for ``estimate=True``.
    """

    enabled: bool = True
    labels: str | None = None
    source_field: str = "source"
    mode: str = "head_recompute"
    cadence: int = 1
    reconcile_rtol: float = 0.05
    ce_chunk_tokens: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(
                f"prequential_logging.enabled must be a bool, got {self.enabled!r}"
            )
        if self.labels is not None and (
            not isinstance(self.labels, str) or not self.labels
        ):
            raise ValueError(
                "prequential_logging.labels must be a non-empty path string "
                f"or null, got {self.labels!r}"
            )
        if not isinstance(self.source_field, str) or not self.source_field:
            raise ValueError(
                "prequential_logging.source_field must be a non-empty string, "
                f"got {self.source_field!r}"
            )
        if self.mode not in _MODES:
            raise ValueError(
                f"prequential_logging.mode must be one of {list(_MODES)}, "
                f"got {self.mode!r}"
            )
        if (
            isinstance(self.cadence, bool)
            or not isinstance(self.cadence, int)
            or self.cadence < 1
        ):
            raise ValueError(
                "prequential_logging.cadence must be a positive integer, "
                f"got {self.cadence!r}"
            )
        if (
            not isinstance(self.reconcile_rtol, (int, float))
            or isinstance(self.reconcile_rtol, bool)
            or not self.reconcile_rtol > 0
        ):
            raise ValueError(
                "prequential_logging.reconcile_rtol must be a positive number, "
                f"got {self.reconcile_rtol!r}"
            )
        if (
            isinstance(self.ce_chunk_tokens, bool)
            or not isinstance(self.ce_chunk_tokens, int)
            or self.ce_chunk_tokens < 1
        ):
            raise ValueError(
                "prequential_logging.ce_chunk_tokens must be a positive "
                f"integer, got {self.ce_chunk_tokens!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        """YAML-safe dict for rendering + manifests."""
        return {
            "enabled": self.enabled,
            "labels": self.labels,
            "source_field": self.source_field,
            "mode": self.mode,
            "cadence": self.cadence,
            "reconcile_rtol": self.reconcile_rtol,
            "ce_chunk_tokens": self.ce_chunk_tokens,
        }


def prequential_config_from(
    data: Mapping[str, Any], *, source: str
) -> PrequentialLoggingConfig:
    """Strict constructor from a YAML mapping (unknown keys are an error)."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"prequential_logging must be a mapping in {source}, "
            f"got {type(data).__name__}"
        )
    data = dict(data)
    known = {f.name for f in dataclasses.fields(PrequentialLoggingConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown prequential_logging keys in {source}: {sorted(unknown)}"
        )
    return PrequentialLoggingConfig(**data)


# --------------------------------------------------- sidecar + segment mapping
def hash_token_ids(ids: Sequence[int]) -> str:
    """The segment join key: sha256 over the token ids' canonical bytes."""
    return hashlib.sha256(
        b",".join(str(int(i)).encode("ascii") for i in ids)
    ).hexdigest()


def read_labels_sidecar(
    path: str | Path, *, source_field: str = "source"
) -> list[dict[str, Any]]:
    """Read + validate an index-aligned labels sidecar.

    One JSONL row per training-corpus row; ``index`` must be contiguous
    0..n−1 (a gap or reorder means the sidecar does not describe this corpus).
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"prequential labels sidecar {path} is missing or unreadable — "
            "build the mix with emit_labels: true, or point "
            "prequential_logging.labels at a <arm>_source_order.jsonl"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").split("\n"), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "index" not in row or source_field not in row:
            raise ValueError(
                f"labels sidecar {path}:{line_number}: rows must carry "
                f"'index' and {source_field!r}, got {row!r}"
            )
        tag = row[source_field]
        if not isinstance(tag, str) or not tag:
            raise ValueError(
                f"labels sidecar {path}:{line_number}: {source_field!r} must "
                f"be a non-empty string, got {tag!r}"
            )
        if int(row["index"]) != len(rows):
            raise ValueError(
                f"labels sidecar {path}:{line_number}: index "
                f"{row['index']!r} breaks 0..n-1 contiguity (expected "
                f"{len(rows)}) — the sidecar does not describe this corpus"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"labels sidecar {path} is empty")
    return rows


# Threshold for the SOFT per-doc reconciliation of the sidecar's `tokens`
# field against axolotl's own tokenization: axolotl 0.17's completion
# strategy tokenizes the WHOLE doc once and slices `val[i:i+sequence_len]`
# (non-overlapping, no per-chunk special tokens —
# prompt_strategies/completion.py:53-58), and its `_tokenize`
# (prompt_tokenizers.py:73-105) adds at most one BOS (via the tokenizer call)
# and appends at most one EOS per DOC, so a well-formed sidecar differs by at
# most ±2 per doc. With the corpus-derived segment map the `tokens` field is
# no longer load-bearing for attribution, so a larger delta is a WARNING
# (the sidecar producer may count with different special-token conventions);
# the hard gates are the structural prepared-coverage checks.
COMPLETION_TOKEN_SLACK = 2


def completion_chunk_ids(
    text: str,
    tokenizer: Any,
    *,
    sequence_len: int,
    max_length: int | None = None,
) -> list[list[int]]:
    """Replicate axolotl 0.17's completion tokenization for one doc, exactly.

    Mirrors ``prompt_tokenizers.py:73-105`` (``_tokenize``: empty text ->
    nothing; tokenizer call with ``truncation=True, max_length=sequence_len *
    64, padding=False``; one EOS appended when the doc does not already end
    with it AND is under ``max_length``) followed by
    ``prompt_strategies/completion.py:53-58`` (non-overlapping
    ``val[i:i+sequence_len]`` slices). Any drift from axolotl's real output
    is caught byte-for-byte by :func:`verify_prepared_coverage`.
    """
    if not text:
        return []
    max_length = sequence_len * 64 if max_length is None else max_length
    result = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    ids = list(result["input_ids"])
    if not ids:
        return []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and ids[-1] != eos and len(ids) < max_length:
        ids.append(eos)
    return [ids[i:i + sequence_len] for i in range(0, len(ids), sequence_len)]


def read_corpus_texts(
    path: str | Path,
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    field: str = "text",
) -> list[str]:
    """Read the training-corpus JSONL the sidecar is index-aligned to.

    Hard gates: the corpus must exist, every row must carry ``field``, the
    row count must equal the sidecar's, and — when the sidecar rows carry
    ``text_sha256`` — every doc's text must hash to it (the strongest
    corpus↔sidecar identity check available)."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"prequential: training corpus {path} is missing or unreadable — "
            "the segment map is derived from the corpus itself"
        )
    texts: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").split("\n"), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or field not in row:
            raise RuntimeError(
                f"prequential: corpus {path}:{line_number} carries no "
                f"{field!r} field"
            )
        texts.append(row[field])
    if len(texts) != len(sidecar_rows):
        raise RuntimeError(
            f"prequential: corpus {path} has {len(texts)} docs but the "
            f"labels sidecar has {len(sidecar_rows)} — the sidecar does not "
            "describe this corpus"
        )
    for index, (text, row) in enumerate(zip(texts, sidecar_rows)):
        expected = row.get("text_sha256")
        if expected and hashlib.sha256(text.encode()).hexdigest() != expected:
            raise RuntimeError(
                f"prequential: corpus doc {index} does not hash to the "
                "sidecar's text_sha256 — the sidecar does not describe this "
                "corpus"
            )
    return texts


def build_segment_map(
    texts: Sequence[str],
    sidecar_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    sequence_len: int,
    source_field: str = "source",
    token_slack: int = COMPLETION_TOKEN_SLACK,
) -> tuple[dict[str, tuple[str, int, int]], dict[str, int]]:
    """Order-independent segment map: ``sha256(chunk input_ids) -> (source,
    doc_index, chunk_index)`` for every chunk of every corpus doc.

    Built from the corpus itself by replicating axolotl's completion
    tokenization (:func:`completion_chunk_ids`), NOT from prepared-row order:
    axolotl shuffles the tokenized dataset during dataset prep
    (``utils/data/shared.py:540-576`` ``merge_datasets`` ->
    ``ds.shuffle(seed=cfg.seed)``; ``shuffle_merged_datasets`` defaults to
    true), so the prepared dataset is a PERMUTATION of input order and any
    index-based alignment silently misattributes sources. Pair with
    :func:`verify_prepared_coverage` — together they prove the replication
    byte-consistent against the run's own prepared rows.

    A chunk hash shared by docs of *different* sources is an attribution
    collision -> raise; identical-source duplicates map once and attribute
    correctly. The sidecar ``tokens`` field is reconciled per doc as a
    WARNING beyond ``token_slack`` (it is no longer load-bearing).

    Cost note: rebuilding the tokenization for ~20k docs / ~16M tokens runs
    once per rank at train begin (~1-2 min CPU) — accepted for v1 (a
    rank-0-builds-and-shares-via-filesystem variant is a later optimization).

    Returns ``(mapping, stats)`` with ``stats = {n_docs, n_chunks,
    n_chunked_docs, n_empty_docs}``.
    """
    if len(texts) != len(sidecar_rows):
        raise RuntimeError(
            f"prequential: {len(texts)} corpus docs vs {len(sidecar_rows)} "
            "sidecar rows — the sidecar does not describe this corpus"
        )
    if (
        isinstance(sequence_len, bool)
        or not isinstance(sequence_len, int)
        or sequence_len < 1
    ):
        raise RuntimeError(
            "prequential: building the segment map needs a positive integer "
            f"sequence_len, got {sequence_len!r}"
        )
    mapping: dict[str, tuple[str, int, int]] = {}
    n_chunks = 0
    n_chunked_docs = 0
    n_empty_docs = 0
    token_mismatches: list[str] = []
    for doc_index, (text, row) in enumerate(zip(texts, sidecar_rows)):
        chunks = completion_chunk_ids(
            text, tokenizer, sequence_len=sequence_len
        )
        if not chunks:
            n_empty_docs += 1
            continue
        if len(chunks) > 1:
            n_chunked_docs += 1
        n_chunks += len(chunks)
        tag = str(row[source_field])
        if "tokens" in row:
            total = sum(len(ids) for ids in chunks)
            if abs(total - int(row["tokens"])) > token_slack:
                token_mismatches.append(
                    f"doc {doc_index}: sidecar claims {row['tokens']} tokens,"
                    f" axolotl tokenization yields {total}"
                )
        for chunk_index, ids in enumerate(chunks):
            key = hash_token_ids(ids)
            existing = mapping.get(key)
            if existing is not None:
                if existing[0] != tag:
                    raise RuntimeError(
                        "prequential segment-map collision: identical chunk "
                        f"token ids appear under sources {existing[0]!r} "
                        f"(doc {existing[1]}) and {tag!r} (doc {doc_index}) "
                        "— attribution would be ambiguous"
                    )
                continue  # identical-source duplicate: attributes correctly
            mapping[key] = (tag, doc_index, chunk_index)
    if token_mismatches:
        import warnings

        preview = "; ".join(token_mismatches[:3])
        warnings.warn(
            f"prequential: {len(token_mismatches)} sidecar tokens fields "
            f"differ from axolotl tokenization by more than "
            f"{token_slack} (e.g. {preview}) — the sidecar's counter uses "
            "different special-token conventions; attribution is unaffected "
            "(the map is corpus-derived), but doc-level token bookkeeping "
            "from the sidecar should not be trusted",
            stacklevel=2,
        )
    stats = {
        "n_docs": len(sidecar_rows),
        "n_chunks": n_chunks,
        "n_chunked_docs": n_chunked_docs,
        "n_empty_docs": n_empty_docs,
    }
    return mapping, stats


def verify_prepared_coverage(
    prepared_input_ids: Iterable[Sequence[int]],
    mapping: Mapping[str, tuple[str, int, int]],
    *,
    n_chunks: int,
) -> int:
    """The self-verifying cross-check making the tokenization replication
    provably byte-consistent per run: EVERY prepared row's input_ids must
    hash into the corpus-derived map, and the row count must equal the
    number of chunks the replication produced. Any miss raises with the row
    index and lengths — if axolotl's tokenization semantics ever change,
    this cannot pass silently. Returns the prepared row count."""
    count = 0
    for row_index, ids in enumerate(prepared_input_ids):
        ids = list(ids)
        count += 1
        if hash_token_ids(ids) not in mapping:
            raise RuntimeError(
                f"prequential: prepared row {row_index} ({len(ids)} tokens) "
                "has no match in the corpus-derived segment map — the local "
                "replication of axolotl's completion tokenization is not "
                "byte-consistent with this run's prepared dataset (tokenizer"
                "/special-token/chunking drift); refusing to attribute"
            )
    if count != n_chunks:
        raise RuntimeError(
            f"prequential: corpus tokenization produced {n_chunks} chunks "
            f"but the prepared dataset has {count} rows — the replication "
            "does not describe this run's prepared dataset"
        )
    return count


def segment_bounds(position_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Split a packed sequence into ``[start, end)`` segments at
    ``position_ids`` resets to 0 (the multipack collator's per-segment
    restart)."""
    positions = list(position_ids)
    if not positions:
        raise ValueError("position_ids is empty")
    if positions[0] != 0:
        raise RuntimeError(
            "prequential: position_ids does not start at 0 "
            f"(got {positions[0]!r}) — cannot recover packed segments"
        )
    starts = [i for i, p in enumerate(positions) if p == 0]
    ends = starts[1:] + [len(positions)]
    return list(zip(starts, ends))


def attribute_sequence(
    *,
    input_ids: Sequence[int],
    position_ids: Sequence[int],
    labels: Sequence[int],
    nll: Sequence[float],
    segment_map: Mapping[str, tuple[str, int, int]],
    step: int | None = None,
    rank: int | None = None,
) -> dict[str, list]:
    """Attribute one packed sequence's per-token NLL to sources.

    ``nll[t]`` is the NLL (nats) of the *label* token at position ``t``
    (predicted from position ``t−1``); ``nll[0]`` is never read. Attribution
    rule (design doc §1.2): a prediction belongs to the source of the token
    being predicted, so a segment's first label token — conditioned on the
    previous segment's last position — counts toward *this* segment. Padding
    and ``-100`` labels are excluded from both ``tokens`` and ``sum_nll``;
    an unmapped segment with any counted token is a raise (untagged tokens
    are never silently pooled).

    Returns ``{source: [tokens, sum_nll_nats, n_segments]}``.
    """
    if not (len(input_ids) == len(position_ids) == len(labels) == len(nll)):
        raise ValueError(
            "input_ids/position_ids/labels/nll length mismatch: "
            f"{len(input_ids)}/{len(position_ids)}/{len(labels)}/{len(nll)}"
        )
    accumulator: dict[str, list] = {}
    for start, end in segment_bounds(position_ids):
        trimmed = end
        while trimmed > start and labels[trimmed - 1] == _IGNORE_INDEX:
            trimmed -= 1
        if trimmed == start:
            continue  # fully masked segment: padding, never counted
        key = hash_token_ids(input_ids[start:trimmed])
        entry = segment_map.get(key)
        if entry is None:
            raise RuntimeError(
                "prequential: packed segment at positions "
                f"[{start}, {trimmed}) (length {trimmed - start}, step={step}, "
                "rank="
                f"{rank}) has no entry in the segment map — untagged tokens "
                "are never silently counted; the labels sidecar does not "
                "cover this corpus"
            )
        tag = entry[0]
        tokens = 0
        total = 0.0
        for position in range(max(start, 1), trimmed):
            if labels[position] == _IGNORE_INDEX:
                continue
            tokens += 1
            total += float(nll[position])
        bucket = accumulator.setdefault(tag, [0, 0.0, 0])
        bucket[0] += tokens
        bucket[1] += total
        bucket[2] += 1
    return accumulator


def _resolve_prepared_dir(prepared_path: Path) -> Path:
    """The saved-arrow dir under ``dataset_prepared_path`` (axolotl may nest
    the tokenized dataset one hash-named level down)."""
    if (prepared_path / "dataset_info.json").is_file():
        return prepared_path
    candidates = (
        [
            child
            for child in sorted(prepared_path.iterdir())
            if child.is_dir() and (child / "dataset_info.json").is_file()
        ]
        if prepared_path.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"prequential: cannot resolve the prepared dataset under "
        f"{prepared_path} (found {len(candidates)} candidate arrow dirs) — "
        "the segment map needs the run's own tokenized rows"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------- analysis
@dataclass(frozen=True)
class PrequentialRow:
    """One (step, rank, source) aggregate from a validated log (analysis-side;
    ``cadence``/``partial`` are carried from the attempt header, not disk
    fields of the data row)."""

    attempt: str
    rank: int
    step: int
    epoch: float | None
    source: str
    tokens: int
    sum_nll_nats: float
    n_segments: int
    microbatches: int
    lr: float | None
    cadence: int = 1
    partial: bool = False


@dataclass(frozen=True)
class PrequentialSummary:
    """Per-source code length. Every summary carries its n (tokens, steps)."""

    tokens: int
    sum_nll_nats: float
    bits: float
    bits_per_token: float
    n_steps: int
    partial: bool = False


@dataclass(frozen=True)
class CurvePoint:
    step: int
    epoch: float | None
    lr: float | None
    tokens_seen: int
    bits_per_token: float
    cumulative_bits: float


def _epoch_index(epoch: float | None) -> int:
    if epoch is None:
        raise ValueError(
            "rows carry no epoch — cannot filter by epoch index"
        )
    return max(0, math.ceil(float(epoch)) - 1)


def read_prequential(
    run_dir: str | Path, *, allow_partial: bool = False
) -> list[PrequentialRow]:
    """Merge + validate ``<run>/prequential/prequential.rank*.jsonl``.

    Restart bookkeeping (design doc §6): every process start appends an
    ``attempt_begin`` row; only the LAST attempt per rank file counts (a
    retry restarts from step 0 with fresh optimizer state, so earlier
    attempts are a *different* online code). Raises on: no files, unknown
    schema, a data row from an unknown attempt, duplicate
    ``(step, rank, source)`` within the kept attempt (a hook double-fire),
    step gaps at the declared cadence, or missing rank files — the last two
    downgraded to ``partial=True`` markers under ``allow_partial=True``.
    """
    subdir = Path(run_dir) / PREQUENTIAL_SUBDIR
    files = sorted(subdir.glob("prequential.rank*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no prequential logs under {subdir} — was prequential_logging "
            "enabled for this run?"
        )
    rows: list[PrequentialRow] = []
    seen_ranks: set[int] = set()
    world_sizes: set[int] = set()
    partial = False
    for path in files:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        for record in records:
            if record.get("schema_version") != PREQUENTIAL_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}: unknown schema_version "
                    f"{record.get('schema_version')!r} (expected "
                    f"{PREQUENTIAL_SCHEMA_VERSION!r})"
                )
        begins = [r for r in records if r.get("event") == "attempt_begin"]
        if not begins:
            raise ValueError(f"{path}: no attempt_begin row — corrupt log")
        known_attempts = {b["attempt"] for b in begins}
        last = begins[-1]
        cadence = int(last.get("cadence", 1))
        world_sizes.add(int(last["world_size"]))
        data = [r for r in records if r.get("event") is None]
        for record in data:
            if record["attempt"] not in known_attempts:
                raise ValueError(
                    f"{path}: data row at step {record.get('step')} carries "
                    f"unknown attempt {record.get('attempt')!r}"
                )
        kept = [r for r in data if r["attempt"] == last["attempt"]]
        if not kept:
            if not allow_partial:
                raise ValueError(
                    f"{path}: the last attempt {last['attempt']!r} logged no "
                    "steps — an incomplete retry; pass allow_partial=True to "
                    "read it anyway"
                )
            partial = True
            continue
        seen: set[tuple[int, int, str]] = set()
        for record in kept:
            key = (int(record["step"]), int(record["rank"]), record["source"])
            if key in seen:
                raise ValueError(
                    f"{path}: duplicate (step, rank, source) {key} within "
                    f"attempt {last['attempt']!r} — indicates a hook "
                    "double-fire"
                )
            seen.add(key)
        steps = sorted({int(r["step"]) for r in kept})
        expected = list(range(cadence, steps[-1] + 1, cadence))
        if steps != expected:
            missing = sorted(set(expected) - set(steps))
            if not allow_partial:
                raise ValueError(
                    f"{path}: attempt {last['attempt']!r} is missing steps "
                    f"{missing[:8]}{'...' if len(missing) > 8 else ''} at "
                    f"cadence {cadence} — a partial log; pass "
                    "allow_partial=True to mark downstream numbers as partial"
                )
            partial = True
        for record in kept:
            seen_ranks.add(int(record["rank"]))
            rows.append(
                PrequentialRow(
                    attempt=record["attempt"],
                    rank=int(record["rank"]),
                    step=int(record["step"]),
                    epoch=(
                        None if record.get("epoch") is None
                        else float(record["epoch"])
                    ),
                    source=record["source"],
                    tokens=int(record["tokens"]),
                    sum_nll_nats=float(record["sum_nll_nats"]),
                    n_segments=int(record.get("n_segments", 0)),
                    microbatches=int(record.get("microbatches", 0)),
                    lr=(None if record.get("lr") is None
                        else float(record["lr"])),
                    cadence=cadence,
                )
            )
    if len(world_sizes) != 1:
        raise ValueError(
            f"rank files disagree on world_size: {sorted(world_sizes)}"
        )
    world = next(iter(world_sizes))
    if seen_ranks != set(range(world)):
        if not allow_partial:
            raise ValueError(
                f"rank files cover ranks {sorted(seen_ranks)} but the run "
                f"declared world_size {world} — a per-source total from a "
                "subset of ranks is not the run's code length; pass "
                "allow_partial=True to mark downstream numbers as partial"
            )
        partial = True
    if partial:
        rows = [dataclasses.replace(row, partial=True) for row in rows]
    return rows


def codelength(
    rows: Sequence[PrequentialRow],
    *,
    epochs: Sequence[int] | None = None,
    estimate: bool = False,
) -> dict[str, PrequentialSummary]:
    """Per-source prequential code length. ``epochs=(0,)`` selects the primary
    first-presentation quantity. Refuses totals from a subsampled
    (``cadence > 1``) log unless ``estimate=True`` — silently interpolating
    would change what is measured."""
    if not rows:
        raise ValueError("no prequential rows")
    if any(row.cadence > 1 for row in rows) and not estimate:
        raise ValueError(
            "log was subsampled (cadence > 1): the code-length integral is "
            "only exact at cadence 1 — pass estimate=True to accept an "
            "estimate"
        )
    if epochs is not None:
        wanted = {int(e) for e in epochs}
        rows = [r for r in rows if _epoch_index(r.epoch) in wanted]
        if not rows:
            raise ValueError(f"no rows in epochs {sorted(wanted)}")
    summaries: dict[str, PrequentialSummary] = {}
    for source in sorted({row.source for row in rows}):
        selected = [row for row in rows if row.source == source]
        tokens = sum(row.tokens for row in selected)
        nats = sum(row.sum_nll_nats for row in selected)
        if tokens <= 0:
            raise ValueError(f"source {source!r} has no counted tokens")
        bits = nats / _LN2
        summaries[source] = PrequentialSummary(
            tokens=tokens,
            sum_nll_nats=nats,
            bits=bits,
            bits_per_token=bits / tokens,
            n_steps=len({row.step for row in selected}),
            partial=any(row.partial for row in selected),
        )
    return summaries


def bits_per_token_curve(
    rows: Sequence[PrequentialRow], source: str
) -> list[CurvePoint]:
    """Per-step bits/token vs cumulative source-tokens-seen for one source."""
    selected = [row for row in rows if row.source == source]
    if not selected:
        known = sorted({row.source for row in rows})
        raise ValueError(f"no rows for source {source!r}; sources: {known}")
    by_step: dict[int, list] = {}
    for row in selected:
        bucket = by_step.setdefault(row.step, [0, 0.0, None, None])
        bucket[0] += row.tokens
        bucket[1] += row.sum_nll_nats
        if row.rank == 0 or bucket[2] is None:
            bucket[2] = row.epoch
            bucket[3] = row.lr
    curve: list[CurvePoint] = []
    tokens_seen = 0
    cumulative_bits = 0.0
    for step in sorted(by_step):
        tokens, nats, epoch, lr = by_step[step]
        tokens_seen += tokens
        bits = nats / _LN2
        cumulative_bits += bits
        curve.append(
            CurvePoint(
                step=step,
                epoch=epoch,
                lr=lr,
                tokens_seen=tokens_seen,
                bits_per_token=bits / tokens if tokens else 0.0,
                cumulative_bits=cumulative_bits,
            )
        )
    return curve


def reconcile(
    rows: Sequence[PrequentialRow],
    log_history: Sequence[Mapping[str, Any]],
    *,
    rtol: float = 0.05,
) -> None:
    """The multi-rank measurement-integrity gate (design doc §1.3): at every
    logged step the recomputed all-source mean NLL must match the trainer's
    logged ``loss`` within ``rtol``. Raises on any violation or on a step the
    trainer never logged — never returns a degraded verdict."""
    if not rows:
        raise ValueError("no prequential rows to reconcile")
    losses = {
        int(entry["step"]): float(entry["loss"])
        for entry in log_history
        if "step" in entry and "loss" in entry
    }
    by_step: dict[int, list] = {}
    for row in rows:
        bucket = by_step.setdefault(row.step, [0, 0.0])
        bucket[0] += row.tokens
        bucket[1] += row.sum_nll_nats
    for step in sorted(by_step):
        if step not in losses:
            raise ValueError(
                f"step {step} has prequential rows but no logged loss in "
                "log_history — cannot verify the measurement"
            )
        tokens, nats = by_step[step]
        if tokens <= 0:
            raise ValueError(f"step {step} has no counted tokens")
        mean = nats / tokens
        loss = losses[step]
        if abs(mean - loss) > rtol * abs(loss):
            raise RuntimeError(
                f"prequential reconciliation failed at step {step}: "
                f"recomputed mean NLL {mean:.6f} vs trainer loss {loss:.6f} "
                f"(rtol {rtol}) — the recomputed head diverged from the "
                "training loss; re-run with mode: shadow_forward"
            )
    return None


# ------------------------------------------------- lazy heavy-dependency API
def _locate_final_norm(model: Any) -> Any:
    """The module feeding the LM head (gemma3: ``model.model.norm``) — the
    hook point whose output is the final hidden state."""
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
    norm = getattr(decoder, "norm", None)
    if norm is None:
        inner = getattr(model, "model", None)
        norm = getattr(inner, "norm", None)
        if norm is None:
            norm = getattr(getattr(inner, "model", None), "norm", None)
    if norm is None:
        raise RuntimeError(
            "prequential: cannot locate the final norm module on "
            f"{type(model).__name__} — head-recompute has no hook point; "
            "use mode: shadow_forward for this architecture"
        )
    return norm


def _callback_class():
    from transformers import TrainerCallback

    class PrequentialLoggingCallback(TrainerCallback):
        """Per-rank prequential NLL logger (hooks + JSONL artifact).

        Hooks fire on training forwards only (``torch.is_grad_enabled()``
        guard, the RouterHealthPlugin pattern — the final norm sits outside
        the checkpointed decoder blocks, but the guard is kept anyway); the
        HF trainer increments ``global_step`` at the optimizer step, so the
        microbatches buffered at ``on_step_end`` are exactly the chunk whose
        pre-update NLL this is.
        """

        def __init__(
            self,
            config: PrequentialLoggingConfig,
            *,
            datasets: Sequence[Any],
            dataset_prepared_path: Any,
            sequence_len: Any,
            sample_packing: Any,
        ) -> None:
            if not isinstance(config, PrequentialLoggingConfig):
                raise TypeError(
                    "PrequentialLoggingCallback needs a "
                    f"PrequentialLoggingConfig, got {type(config).__name__}"
                )
            self.config = config
            self._datasets = list(datasets or [])
            self._prepared_path = dataset_prepared_path
            self._sequence_len = sequence_len
            self._sample_packing = sample_packing
            self._trainer: Any = None
            self._segment_map: dict[str, tuple[str, int, int]] | None = None
            self._rank = 0
            self._world_size = 1
            self._attempt: str | None = None
            self._out_path: Path | None = None
            self._stash: Any = None
            self._buffer: dict[str, list] = {}
            self._microbatches = 0
            self._pending: tuple[int, int, float] | None = None
            self._lm_head: Any = None
            self._softcap: float | None = None
            self._in_shadow = False

        def attach(self, trainer: Any) -> "PrequentialLoggingCallback":
            self._trainer = trainer
            return self

        # -- lifecycle -----------------------------------------------------
        @staticmethod
        def _entry_value(entry: Any, key: str) -> Any:
            if hasattr(entry, "get"):
                return entry.get(key)
            return getattr(entry, key, None)

        def _resolve_tokenizer(self, kwargs: Mapping[str, Any]) -> Any:
            """The trainer's own tokenizer — the segment map must replicate
            the run's tokenization, so no fallback to a fresh load."""
            for holder in (self._trainer, kwargs):
                if holder is None:
                    continue
                get = (
                    holder.get if isinstance(holder, Mapping)
                    else lambda key, holder=holder: getattr(holder, key, None)
                )
                for name in ("processing_class", "tokenizer"):
                    tokenizer = get(name)
                    if tokenizer is not None:
                        return tokenizer
            raise RuntimeError(
                "prequential: cannot resolve the trainer's tokenizer "
                "(trainer.processing_class/.tokenizer and callback kwargs "
                "are all empty) — the segment map must be built with the "
                "run's own tokenizer"
            )

        def on_train_begin(self, args, state, control, **kwargs):
            # v1 scope gate (design doc §6): CPT completion, single dataset,
            # packed — anything else is refused before spending compute.
            if len(self._datasets) != 1:
                raise RuntimeError(
                    "prequential_logging v1 supports exactly one datasets: "
                    f"entry, got {len(self._datasets)}"
                )
            entry = self._datasets[0]
            dataset_type = self._entry_value(entry, "type")
            dataset_path = self._entry_value(entry, "path")
            if dataset_type != "completion":
                raise RuntimeError(
                    "prequential_logging v1 is scoped to type: completion "
                    f"(CPT loss over all tokens), got {dataset_type!r} — "
                    "IFT/chat-template packing alignment is unvalidated"
                )
            if not self._sample_packing:
                raise RuntimeError(
                    "prequential_logging v1 requires sample_packing: true "
                    "(segmentation reads the multipack collator's "
                    "position_ids resets)"
                )
            labels_path = Path(
                self.config.labels or f"{dataset_path}.labels.jsonl"
            )
            sidecar = read_labels_sidecar(
                labels_path, source_field=self.config.source_field
            )
            # The segment map is derived from the CORPUS, never from
            # prepared-row order: axolotl shuffles the tokenized dataset
            # during prep (utils/data/shared.py:540-576 merge_datasets ->
            # ds.shuffle(seed=cfg.seed), shuffle_merged_datasets defaults
            # true), so the prepared dataset is a permutation of input order
            # and index alignment would silently misattribute sources.
            corpus_field = self._entry_value(entry, "field") or "text"
            texts = read_corpus_texts(
                dataset_path, sidecar, field=corpus_field
            )
            tokenizer = self._resolve_tokenizer(kwargs)
            from datasets import load_from_disk

            prepared_dir = _resolve_prepared_dir(Path(str(self._prepared_path)))
            prepared = load_from_disk(str(prepared_dir))
            try:
                self._segment_map, chunk_stats = build_segment_map(
                    texts, sidecar, tokenizer,
                    sequence_len=self._sequence_len,
                    source_field=self.config.source_field,
                )
                n_prepared = verify_prepared_coverage(
                    prepared["input_ids"], self._segment_map,
                    n_chunks=chunk_stats["n_chunks"],
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"prequential (corpus {dataset_path}, sidecar "
                    f"{labels_path}, prepared {prepared_dir}): {error}"
                ) from error
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                self._rank = dist.get_rank()
                self._world_size = dist.get_world_size()
            self._attempt = str(uuid.uuid4())
            self._out_path = (
                Path(args.output_dir).parent
                / PREQUENTIAL_SUBDIR
                / f"prequential.rank{self._rank}.jsonl"
            )
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            self._append(
                {
                    "schema_version": PREQUENTIAL_SCHEMA_VERSION,
                    "event": "attempt_begin",
                    "attempt": self._attempt,
                    "rank": self._rank,
                    "world_size": self._world_size,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "labels_path": str(labels_path),
                    "labels_sha256": _sha256_file(labels_path),
                    "n_rows": len(sidecar),
                    # chunked-doc alignment stats (docs > sequence_len that
                    # the completion strategy split into extra prepared rows)
                    "n_prepared_rows": n_prepared,
                    "n_chunked_docs": chunk_stats["n_chunked_docs"],
                    "n_extra_rows": n_prepared - len(sidecar),
                    "dataset_path": str(dataset_path),
                    "sequence_len": self._sequence_len,
                    "mode": self.config.mode,
                    "cadence": self.config.cadence,
                }
            )
            model = kwargs.get("model")
            if model is None and self._trainer is not None:
                model = getattr(self._trainer, "model", None)
            if model is None:
                raise RuntimeError(
                    "prequential: Trainer did not pass a model to "
                    "on_train_begin — cannot register hooks"
                )
            model.register_forward_pre_hook(
                self._model_pre_hook, with_kwargs=True
            )
            if self.config.mode == "head_recompute":
                lm_head = model.get_output_embeddings()
                if lm_head is None:
                    raise RuntimeError(
                        "prequential: model has no output embeddings — "
                        "head-recompute cannot run; use mode: shadow_forward"
                    )
                self._lm_head = lm_head
                text_config = getattr(
                    getattr(model, "config", None), "text_config", None
                ) or getattr(model, "config", None)
                self._softcap = getattr(
                    text_config, "final_logit_softcapping", None
                )
                _locate_final_norm(model).register_forward_hook(
                    self._norm_hook
                )
            return control

        def _append(self, record: Mapping[str, Any]) -> None:
            assert self._out_path is not None
            with self._out_path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(record) + "\n")

        # -- hooks ----------------------------------------------------------
        def _model_pre_hook(self, module, hook_args, hook_kwargs):
            import torch

            if self._in_shadow or not torch.is_grad_enabled():
                return
            input_ids = hook_kwargs.get("input_ids")
            position_ids = hook_kwargs.get("position_ids")
            labels = hook_kwargs.get("labels")
            if input_ids is None or labels is None:
                raise RuntimeError(
                    "prequential: training forward carries no "
                    "input_ids/labels kwargs — cannot observe the loss"
                )
            if position_ids is None:
                raise RuntimeError(
                    "prequential: training forward carries no position_ids — "
                    "packed segments are unrecoverable (is sample_packing "
                    "actually on?)"
                )
            if self.config.mode == "head_recompute":
                self._stash = (input_ids, position_ids, labels)
                return
            # shadow_forward: one extra no-grad forward of the identical
            # microbatch under the same pre-update parameters — the same
            # measurand, computed the expensive way.
            self._in_shadow = True
            try:
                with torch.no_grad():
                    output = module(
                        *hook_args,
                        **{
                            key: value
                            for key, value in hook_kwargs.items()
                            if key not in ("labels", "num_items_in_batch")
                        },
                    )
                nll = self._nll_from_logits(output.logits, labels)
                self._accumulate_tensors(input_ids, position_ids, labels, nll)
            finally:
                self._in_shadow = False

        def _norm_hook(self, module, hook_args, output):
            import torch

            if not torch.is_grad_enabled():
                return
            if self._stash is None:
                raise RuntimeError(
                    "prequential: final-norm hook fired without a stashed "
                    "microbatch — hook ordering broke (double fire or a "
                    "forward outside the trainer loop)"
                )
            input_ids, position_ids, labels = self._stash
            self._stash = None
            hidden = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                nll = self._nll_from_hidden(hidden.detach(), labels)
            self._accumulate_tensors(input_ids, position_ids, labels, nll)

        # -- NLL ------------------------------------------------------------
        def _apply_softcap(self, logits):
            if self._softcap:
                import torch

                return torch.tanh(logits / self._softcap) * self._softcap
            return logits

        def _nll_from_hidden(self, hidden, labels):
            """Per-position NLL (nats): ``nll[b, t]`` scores the label token
            at position ``t`` under logits from ``t−1``; ``-100`` positions
            stay 0. Chunked fp32 log_softmax; the matmul rides the model's
            own LM head module."""
            import torch

            batch, length, width = hidden.shape
            flat_hidden = hidden[:, :-1, :].reshape(-1, width)
            flat_labels = labels[:, 1:].reshape(-1)
            flat = torch.zeros(
                flat_labels.shape[0], dtype=torch.float32,
                device=hidden.device,
            )
            valid = (flat_labels != _IGNORE_INDEX).nonzero(as_tuple=True)[0]
            chunk = self.config.ce_chunk_tokens
            for start in range(0, int(valid.numel()), chunk):
                index = valid[start:start + chunk]
                logits = self._apply_softcap(
                    self._lm_head(flat_hidden.index_select(0, index))
                )
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                flat[index] = -log_probs.gather(
                    1, flat_labels.index_select(0, index).unsqueeze(1)
                ).squeeze(1)
            nll = torch.zeros(
                (batch, length), dtype=torch.float32, device=hidden.device
            )
            nll[:, 1:] = flat.reshape(batch, length - 1)
            return nll

        def _nll_from_logits(self, logits, labels):
            import torch

            batch, length, vocabulary = logits.shape
            flat_logits = logits[:, :-1, :].reshape(-1, vocabulary)
            flat_labels = labels[:, 1:].reshape(-1)
            flat = torch.zeros(
                flat_labels.shape[0], dtype=torch.float32,
                device=logits.device,
            )
            valid = (flat_labels != _IGNORE_INDEX).nonzero(as_tuple=True)[0]
            chunk = self.config.ce_chunk_tokens
            for start in range(0, int(valid.numel()), chunk):
                index = valid[start:start + chunk]
                log_probs = torch.log_softmax(
                    flat_logits.index_select(0, index).float(), dim=-1
                )
                flat[index] = -log_probs.gather(
                    1, flat_labels.index_select(0, index).unsqueeze(1)
                ).squeeze(1)
            nll = torch.zeros(
                (batch, length), dtype=torch.float32, device=logits.device
            )
            nll[:, 1:] = flat.reshape(batch, length - 1)
            return nll

        # -- accumulation ----------------------------------------------------
        def _accumulate_tensors(self, input_ids, position_ids, labels, nll):
            self._accumulate_python(
                input_ids.detach().cpu().tolist(),
                position_ids.detach().cpu().tolist(),
                labels.detach().cpu().tolist(),
                nll.detach().cpu().tolist(),
            )

        def _accumulate_python(self, input_ids, position_ids, labels, nll):
            """Pure per-microbatch attribution over python lists-of-lists
            (one inner list per packed sequence) — the CPU-testable seam."""
            if self._segment_map is None:
                raise RuntimeError(
                    "prequential: segment map not built (on_train_begin "
                    "never ran)"
                )
            for ids, pos, labs, row_nll in zip(
                input_ids, position_ids, labels, nll
            ):
                per_source = attribute_sequence(
                    input_ids=ids,
                    position_ids=pos,
                    labels=labs,
                    nll=row_nll,
                    segment_map=self._segment_map,
                    step=None,
                    rank=self._rank,
                )
                for tag, (tokens, nats, segments) in per_source.items():
                    bucket = self._buffer.setdefault(tag, [0, 0.0, 0])
                    bucket[0] += tokens
                    bucket[1] += nats
                    bucket[2] += segments
            self._microbatches += 1

        # -- flush ------------------------------------------------------------
        def _last_lr(self) -> float | None:
            try:
                return float(self._trainer.lr_scheduler.get_last_lr()[0])
            except (AttributeError, IndexError, TypeError):
                return None

        def on_step_end(self, args, state, control, **kwargs):
            if self._stash is not None:
                raise RuntimeError(
                    "prequential: a stashed microbatch was never scored — "
                    "the final-norm hook did not fire (wrong hook point?)"
                )
            buffer, self._buffer = self._buffer, {}
            microbatches, self._microbatches = self._microbatches, 0
            if not buffer:
                raise RuntimeError(
                    "prequential: optimizer step "
                    f"{int(state.global_step)} produced no observed "
                    "microbatches — the measurement silently stopped"
                )
            step = int(state.global_step)
            tokens_total = sum(bucket[0] for bucket in buffer.values())
            nats_total = sum(bucket[1] for bucket in buffer.values())
            self._pending = (step, tokens_total, nats_total)
            if step % self.config.cadence != 0:
                return control
            epoch = None if state.epoch is None else float(state.epoch)
            lr = self._last_lr()
            for tag in sorted(buffer):
                tokens, nats, segments = buffer[tag]
                self._append(
                    {
                        "schema_version": PREQUENTIAL_SCHEMA_VERSION,
                        "attempt": self._attempt,
                        "rank": self._rank,
                        "step": step,
                        "epoch": epoch,
                        "source": tag,
                        "tokens": tokens,
                        "sum_nll_nats": nats,
                        "n_segments": segments,
                        "microbatches": microbatches,
                        "lr": lr,
                    }
                )
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            # Single-rank live reconciliation gate (§1.3); multi-rank runs
            # are gated in analysis (reconcile()) against
            # trainer_state.final.json — collectives inside callbacks risk
            # desync.
            if self._world_size != 1 or not logs or self._pending is None:
                return control
            loss = logs.get("loss")
            if not isinstance(loss, (int, float)) or isinstance(loss, bool):
                return control
            step, tokens, nats = self._pending
            if int(state.global_step) != step:
                return control
            self._pending = None
            if tokens <= 0:
                raise RuntimeError(
                    f"prequential: step {step} counted no tokens — cannot "
                    "reconcile"
                )
            mean = nats / tokens
            if abs(mean - float(loss)) > self.config.reconcile_rtol * abs(
                float(loss)
            ):
                raise RuntimeError(
                    f"prequential reconciliation failed at step {step}: "
                    f"recomputed mean NLL {mean:.6f} vs trainer loss "
                    f"{float(loss):.6f} (rtol {self.config.reconcile_rtol}) "
                    "— the recomputed head diverged from the training loss; "
                    "re-run with mode: shadow_forward"
                )
            return control

    PrequentialLoggingCallback.__module__ = __name__
    return PrequentialLoggingCallback


def _plugin_class():
    # Pod-side dependency; the class path in a rendered config resolves here
    # through axolotl's load_plugin (importlib + getattr, PEP 562 compatible).
    from axolotl.integrations.base import BasePlugin

    class PrequentialLoggingPlugin(BasePlugin):
        """Axolotl shim: read the rendered ``prequential_logging`` block and
        register :class:`PrequentialLoggingCallback` post-trainer."""

        def get_input_args(self) -> str:
            return "scimt.train.prequential.PrequentialLoggingArgs"

        @staticmethod
        def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
            if hasattr(cfg, "get"):
                value = cfg.get(key, default)
                if value is not None:
                    return value
            return getattr(cfg, key, default)

        def add_callbacks_post_trainer(self, cfg, trainer):
            raw = self._cfg_value(cfg, "prequential_logging")
            if not raw:
                # scimt's render_stage always writes the plugin and its
                # config block together; a loaded plugin without the block
                # means the config was edited by hand — refuse now rather
                # than finish the run with the measurement silently missing.
                raise ValueError(
                    "PrequentialLoggingPlugin is loaded but the config "
                    "carries no prequential_logging block — scimt's "
                    "render_stage writes them together, so this config was "
                    "edited; add the block back (or drop the plugin) instead "
                    "of silently skipping the code-length measurement"
                )
            config = prequential_config_from(
                dict(raw), source="axolotl config prequential_logging"
            )
            callback_cls = _cached("PrequentialLoggingCallback")
            callback = callback_cls(
                config,
                datasets=self._cfg_value(cfg, "datasets") or [],
                dataset_prepared_path=self._cfg_value(
                    cfg, "dataset_prepared_path"
                ),
                sequence_len=self._cfg_value(cfg, "sequence_len"),
                sample_packing=self._cfg_value(cfg, "sample_packing"),
            )
            return [callback.attach(trainer)]

    PrequentialLoggingPlugin.__module__ = __name__
    return PrequentialLoggingPlugin


def _args_class():
    from pydantic import BaseModel

    class PrequentialLoggingArgs(BaseModel):
        """Pydantic mixin merged into axolotl's input config so the rendered
        ``prequential_logging`` block passes config validation."""

        prequential_logging: dict | None = None

    PrequentialLoggingArgs.__module__ = __name__
    return PrequentialLoggingArgs


_LAZY = {
    "PrequentialLoggingCallback": _callback_class,
    "PrequentialLoggingPlugin": _plugin_class,
    "PrequentialLoggingArgs": _args_class,
}


def _cached(name: str):
    if name not in globals() or globals()[name] is None:
        globals()[name] = _LAZY[name]()
    return globals()[name]


def __getattr__(name: str):
    if name in _LAZY:
        return _cached(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
