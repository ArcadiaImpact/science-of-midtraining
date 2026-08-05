"""Local packed-midtraining and chat-SFT adapters.

Derived from gradient-kernel ca9689a, with local JSONL/HF loading and content
fingerprints required by scimt's provenance boundary.
"""

from __future__ import annotations
import hashlib
import json
from collections import deque
from pathlib import Path
import random
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
    ):
        if sequence_length < 2 or tokenizer.eos_token_id is None:
            raise ValueError("sequence_length >= 2 and eos_token_id are required")
        if max_sequences is not None and max_sequences < 1:
            raise ValueError("max_sequences must be positive when set")
        rows, source_digest = _rows(source, split)
        tokens, self._sequences = deque(), []
        for document_index, (_, row) in enumerate(
            _indexed_rows(rows, seed, shuffle_documents)
        ):
            if text_column not in row:
                raise ValueError(f"dataset must contain a {text_column!r} column")
            if document_index:
                tokens.append(int(tokenizer.eos_token_id))
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
        self._fingerprint_payload = {
            "format": "packed_midtraining",
            "source_digest": source_digest,
            "split": split,
            "text_column": text_column,
            "tokenizer_name": getattr(tokenizer, "name_or_path", ""),
            "sequence_length": sequence_length,
            "seed": seed,
            "reduction": reduction,
            "target_policy": "all_next_tokens",
            "shuffle_documents": shuffle_documents,
            "max_sequences": max_sequences,
        }


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
