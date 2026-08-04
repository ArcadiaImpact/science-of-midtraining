"""Local packed-midtraining and chat-SFT adapters.

Derived from gradient-kernel ca9689a, with local JSONL/HF loading and content
fingerprints required by scimt's provenance boundary.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import torch
from .losses import TokenizedBatch


def _rows(source, split):
    path = Path(source)
    if path.is_file():
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        return rows, hashlib.sha256(raw).hexdigest()
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError("HF datasets require scimt[data-attribution]") from exc
    loaded = load_from_disk(str(path)) if path.exists() else load_dataset(str(source), split=split)
    if isinstance(loaded, DatasetDict):
        loaded = loaded[split]
    rows = [dict(row) for row in loaded]
    fingerprint = getattr(loaded, "_fingerprint", None) or hashlib.sha256(
        json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    return rows, fingerprint


class _BaseDataset:
    def iter_batches(self, batch_size, start_sequence=0, end_sequence=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        stop = len(self._sequences) if end_sequence is None else min(end_sequence, len(self._sequences))
        if start_sequence < 0 or stop < start_sequence:
            raise ValueError("invalid sequence range")
        for start in range(start_sequence, stop, batch_size):
            end = min(start + batch_size, stop)
            yield TokenizedBatch(torch.tensor(self._sequences[start:end], dtype=torch.int64),
                torch.arange(start, end, dtype=torch.int64), torch.tensor(self._masks[start:end], dtype=torch.bool))

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self._fingerprint_payload, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()

    dataset_fingerprint = fingerprint


class PackedMidtrainingDataset(_BaseDataset):
    def __init__(self, source, tokenizer, sequence_length, seed, *, split="train",
                 reduction="per_token", max_sequences=None, shuffle_documents=False,
                 text_column="text"):
        if sequence_length < 2 or tokenizer.eos_token_id is None:
            raise ValueError("sequence_length >= 2 and eos_token_id are required")
        rows, source_digest = _rows(source, split)
        if shuffle_documents:
            random.Random(seed).shuffle(rows)
        tokens = []
        for index, row in enumerate(rows):
            if text_column not in row:
                raise ValueError(f"dataset must contain a {text_column!r} column")
            if index:
                tokens.append(int(tokenizer.eos_token_id))
            tokens.extend(int(x) for x in tokenizer(row[text_column], add_special_tokens=False)["input_ids"])
        usable = len(tokens) // sequence_length * sequence_length
        self._sequences = [tokens[i:i + sequence_length] for i in range(0, usable, sequence_length)]
        if max_sequences is not None:
            self._sequences = self._sequences[:max_sequences]
        self._masks = [[False] + [True] * (sequence_length - 1) for _ in self._sequences]
        self._fingerprint_payload = {"format":"packed_midtraining", "source_digest":source_digest,
            "tokenizer_name":getattr(tokenizer, "name_or_path", ""), "sequence_length":sequence_length,
            "seed":seed, "reduction":reduction, "target_policy":"all_next_tokens",
            "shuffle_documents":shuffle_documents, "max_sequences":max_sequences}


class ChatSFTDataset(_BaseDataset):
    def __init__(self, source, tokenizer, sequence_length, seed, *, split="train",
                 reduction="per_sequence_mean", messages_column="messages", max_sequences=None,
                 shuffle_documents=False):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad is None or not getattr(tokenizer, "chat_template", None):
            raise ValueError("tokenizer must define padding and a reproducible chat template")
        rows, source_digest = _rows(source, split)
        indexed = list(enumerate(rows))
        if shuffle_documents:
            random.Random(seed).shuffle(indexed)
        self._sequences, self._masks, self.source_rows = [], [], []
        for source_row, row in indexed:
            messages = row.get(messages_column)
            if not isinstance(messages, list):
                raise ValueError(f"dataset must contain a {messages_column!r} list column")
            previous, targets = [], []
            for index, message in enumerate(messages):
                rendered = self._render(tokenizer, messages[:index + 1], False)
                if rendered[:len(previous)] != previous:
                    raise ValueError("chat template is not prefix-monotone")
                if message.get("role") == "assistant":
                    header = self._render(tokenizer, messages[:index], True)
                    if rendered[:len(header)] != header or len(header) < len(previous):
                        raise ValueError("cannot locate assistant span")
                    targets.extend(range(len(header), len(rendered)))
                previous = rendered
            targets = [p for p in targets if 1 <= p < min(sequence_length, len(previous))]
            if not targets:
                continue
            sequence = previous[:sequence_length] + [int(pad)] * max(0, sequence_length - len(previous))
            mask = [False] * sequence_length
            for position in targets:
                mask[position] = True
            self._sequences.append(sequence)
            self._masks.append(mask)
            self.source_rows.append(source_row)
            if max_sequences is not None and len(self._sequences) >= max_sequences:
                break
        self._fingerprint_payload = {"format":"chat_sft", "source_digest":source_digest,
            "tokenizer_name":getattr(tokenizer, "name_or_path", ""),
            "chat_template":tokenizer.chat_template, "sequence_length":sequence_length,
            "seed":seed, "reduction":reduction, "target_policy":"assistant_content_and_end",
            "shuffle_documents":shuffle_documents, "messages_column":messages_column,
            "max_sequences":max_sequences}

    @staticmethod
    def _render(tokenizer, messages, prompt):
        rendered = tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=prompt)
        if hasattr(rendered, "keys"):
            rendered = rendered["input_ids"]
        if rendered and isinstance(rendered[0], (list, tuple)):
            rendered = rendered[0]
        return [int(x) for x in rendered]
