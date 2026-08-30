"""Local packed-midtraining and chat-SFT adapters.

Derived from gradient-kernel ca9689a, with local JSONL/HF loading and content
fingerprints required by scimt's provenance boundary.
"""

from __future__ import annotations

import hashlib
import json
import random
import warnings
from collections import deque
from pathlib import Path

import torch

from .losses import TokenizedBatch


class _LocalJSONLRows:
    def __init__(self, path: Path):
        self.path = path
        self._offsets = None

    def _line_offsets(self):
        if self._offsets is None:
            offsets = []
            with self.path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(offset)
            self._offsets = offsets
        return self._offsets

    def __len__(self):
        return len(self._line_offsets())

    def __getitem__(self, index):
        offset = self._line_offsets()[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline())

    def __iter__(self):
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            dict(row), sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _rows(source, split):
    path = Path(source)
    if path.is_file():
        return _LocalJSONLRows(path), _file_digest(path)
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError("HF datasets require scimt[data-attribution]") from exc
    loaded = (
        load_from_disk(str(path))
        if path.exists()
        else load_dataset(str(source), split=split)
    )
    if isinstance(loaded, DatasetDict):
        loaded = loaded[split]
    fingerprint = getattr(loaded, "_fingerprint", None)
    if not fingerprint:
        # HF Dataset objects are re-iterable, so this streaming identity pass
        # retains no row copies and leaves later data iteration available.
        fingerprint = _content_digest(loaded)
    return loaded, fingerprint


def _indexed_rows(rows, seed, shuffle):
    if not shuffle:
        return enumerate(rows)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return ((index, rows[index]) for index in indices)


class _BaseDataset:
    def __len__(self):
        return len(self._sequences)

    def batch_from_indices(self, indices):
        selected = tuple(indices)
        if not selected:
            raise ValueError("indices must be nonempty")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in selected
        ):
            raise ValueError("indices must contain integers")
        if len(set(selected)) != len(selected):
            raise ValueError("indices must be unique within a batch")
        if any(index < 0 or index >= len(self._sequences) for index in selected):
            raise ValueError("batch index is outside the tokenized dataset")
        return TokenizedBatch(
            torch.tensor(
                [self._sequences[index] for index in selected], dtype=torch.int64
            ),
            torch.tensor(selected, dtype=torch.int64),
            torch.tensor(
                [self._masks[index] for index in selected], dtype=torch.bool
            ),
        )

    def iter_batches(self, batch_size, start_sequence=0, end_sequence=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        stop = (
            len(self._sequences)
            if end_sequence is None
            else min(end_sequence, len(self._sequences))
        )
        if start_sequence < 0 or stop < start_sequence:
            raise ValueError("invalid sequence range")
        for start in range(start_sequence, stop, batch_size):
            end = min(start + batch_size, stop)
            yield TokenizedBatch(
                torch.tensor(self._sequences[start:end], dtype=torch.int64),
                torch.arange(start, end, dtype=torch.int64),
                torch.tensor(self._masks[start:end], dtype=torch.bool),
            )

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(
                self._fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    dataset_fingerprint = fingerprint


class PackedMidtrainingDataset(_BaseDataset):
    """Greedy EOS-joined packing (default), or one padded document per row.

    ``pack=False`` gives row *i* exactly document *i* of the source: an EOS
    boundary prefix (never a target — it mirrors the EOS separator that
    precedes every non-first document in the packed stream, and packed-row
    accounting attributes separators to no document), then the document's
    tokens truncated to ``sequence_length - 1``, padded to length with EOS
    (padding is never a target). The target set is therefore exactly the
    document tokens the packed path's per-doc accounting attributes to that
    document (up to truncation, which warns). Empty documents are refused
    loudly: silently skipping one would shift the row->document alignment
    that per-doc scoring exists to provide; ``shuffle_documents=True`` is
    refused for the same reason.
    """

    def __init__(
        self,
        source,
        tokenizer,
        sequence_length,
        seed,
        *,
        split="train",
        reduction="per_token",
        max_sequences=None,
        shuffle_documents=False,
        text_column="text",
        pack=True,
    ):
        if sequence_length < 2 or tokenizer.eos_token_id is None:
            raise ValueError("sequence_length >= 2 and eos_token_id are required")
        if max_sequences is not None and max_sequences < 1:
            raise ValueError("max_sequences must be positive when set")
        if not isinstance(pack, bool):
            raise ValueError("pack must be a boolean")
        if not pack and shuffle_documents:
            raise ValueError(
                "pack=False with shuffle_documents=True would break the "
                "row->document alignment (row i is document i of the source) "
                "that per-doc scoring exists to provide"
            )
        rows, source_digest = _rows(source, split)
        eos = int(tokenizer.eos_token_id)
        self._sequences, self._masks = [], []
        if pack:
            tokens = deque()
            for document_index, (_, row) in enumerate(
                _indexed_rows(rows, seed, shuffle_documents)
            ):
                if text_column not in row:
                    raise ValueError(f"dataset must contain a {text_column!r} column")
                if document_index:
                    tokens.append(eos)
                tokens.extend(
                    int(x)
                    for x in tokenizer(row[text_column], add_special_tokens=False)[
                        "input_ids"
                    ]
                )
                while len(tokens) >= sequence_length:
                    self._sequences.append(
                        [tokens.popleft() for _ in range(sequence_length)]
                    )
                    if max_sequences is not None and len(self._sequences) >= max_sequences:
                        break
                if max_sequences is not None and len(self._sequences) >= max_sequences:
                    break
            self._masks = [
                [False] + [True] * (sequence_length - 1) for _ in self._sequences
            ]
        else:
            truncated = 0
            for document_index, (_, row) in enumerate(
                _indexed_rows(rows, seed, shuffle_documents)
            ):
                if text_column not in row:
                    raise ValueError(f"dataset must contain a {text_column!r} column")
                doc = [
                    int(x)
                    for x in tokenizer(row[text_column], add_special_tokens=False)[
                        "input_ids"
                    ]
                ]
                if not doc:
                    raise ValueError(
                        f"document {document_index} tokenizes to zero tokens; "
                        "pack=False requires nonempty documents to keep the "
                        "row->document alignment exact"
                    )
                if len(doc) > sequence_length - 1:
                    truncated += 1
                    doc = doc[: sequence_length - 1]
                sequence = [eos] + doc
                mask = [False] + [True] * len(doc)
                padding = sequence_length - len(sequence)
                self._sequences.append(sequence + [eos] * padding)
                self._masks.append(mask + [False] * padding)
                if max_sequences is not None and len(self._sequences) >= max_sequences:
                    break
            if truncated:
                warnings.warn(
                    f"pack=False: {truncated} documents truncated to "
                    f"sequence_length - 1 = {sequence_length - 1} tokens; "
                    "their per-doc scores cover the retained prefix only"
                )
        self._fingerprint_payload = {
            "format": "packed_midtraining" if pack else "unpacked_midtraining",
            "source_digest": source_digest,
            "split": split,
            "text_column": text_column,
            "tokenizer_name": getattr(tokenizer, "name_or_path", ""),
            "sequence_length": sequence_length,
            "seed": seed,
            "reduction": reduction,
            "target_policy": (
                "all_next_tokens" if pack else "doc_tokens_after_eos_prefix"
            ),
            "shuffle_documents": shuffle_documents,
            "max_sequences": max_sequences,
        }
        # Conditional so packed payload bytes (and thus every committed packed
        # artifact fingerprint) predate-the-knob identical.
        if not pack:
            self._fingerprint_payload["pack"] = False


class ChatSFTDataset(_BaseDataset):
    def __init__(
        self,
        source,
        tokenizer,
        sequence_length,
        seed,
        *,
        split="train",
        reduction="per_sequence_mean",
        messages_column="messages",
        max_sequences=None,
        shuffle_documents=False,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if max_sequences is not None and max_sequences < 1:
            raise ValueError("max_sequences must be positive when set")
        pad = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if pad is None or not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                "tokenizer must define padding and a reproducible chat template"
            )
        rows, source_digest = _rows(source, split)
        self._sequences, self._masks, self.source_rows = [], [], []
        for source_row, row in _indexed_rows(rows, seed, shuffle_documents):
            messages = row.get(messages_column)
            if not isinstance(messages, list):
                raise ValueError(
                    f"dataset must contain a {messages_column!r} list column"
                )
            previous, targets = [], []
            for index, message in enumerate(messages):
                rendered = self._render(tokenizer, messages[: index + 1], False)
                if rendered[: len(previous)] != previous:
                    raise ValueError("chat template is not prefix-monotone")
                if message.get("role") == "assistant":
                    header = self._render(tokenizer, messages[:index], True)
                    if rendered[: len(header)] != header or len(header) < len(previous):
                        raise ValueError("cannot locate assistant span")
                    targets.extend(range(len(header), len(rendered)))
                previous = rendered
            targets = [
                p for p in targets if 1 <= p < min(sequence_length, len(previous))
            ]
            if not targets:
                continue
            sequence = previous[:sequence_length] + [int(pad)] * max(
                0, sequence_length - len(previous)
            )
            mask = [False] * sequence_length
            for position in targets:
                mask[position] = True
            self._sequences.append(sequence)
            self._masks.append(mask)
            self.source_rows.append(source_row)
            if max_sequences is not None and len(self._sequences) >= max_sequences:
                break
        self._fingerprint_payload = {
            "format": "chat_sft",
            "source_digest": source_digest,
            "tokenizer_name": getattr(tokenizer, "name_or_path", ""),
            "chat_template": tokenizer.chat_template,
            "sequence_length": sequence_length,
            "seed": seed,
            "reduction": reduction,
            "target_policy": "assistant_content_and_end",
            "shuffle_documents": shuffle_documents,
            "messages_column": messages_column,
            "max_sequences": max_sequences,
        }

    @staticmethod
    def _render(tokenizer, messages, prompt):
        rendered = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=prompt
        )
        if hasattr(rendered, "keys"):
            rendered = rendered["input_ids"]
        if rendered and isinstance(rendered[0], (list, tuple)):
            if len(rendered) != 1:
                raise ValueError("expected a single rendered conversation")
            rendered = rendered[0]
        return [int(x) for x in rendered]
