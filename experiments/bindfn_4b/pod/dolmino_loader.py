"""Vendored Dolmino shard loader for the bindfn_4b filler.

Provenance: the ``load_filler`` path of
``examples/06_sheeran_repro/pod/dolmino_loader_pane.py`` (itself vendored
verbatim from pane-functions; pane is deprecated), trimmed to the three
functions the mix builder needs — self-contained, no pane/example imports.

Why not ``load_dataset(streaming=True)``: dolma3_dolmino's shards have
heterogeneous schemas across ingredients (CC-derived shards carry
warcinfo/sa_remove_ranges/... columns), so plain HF streaming dies mid-stream
when the JSON builder casts a shard to the schema inferred from the first
file ("column names don't match", hit on-pod 2026-07-15). Reading the
jsonl.zst shards ourselves and projecting to ``text`` before any schema
unification sidesteps that entirely.
"""

from __future__ import annotations

import io
import json
import random
from collections.abc import Iterator
from typing import Any

from datasets import IterableDataset

FILLER_DATASET = "allenai/dolma3_dolmino_mix-100B-1125"


def _filler_shard_paths(fs: Any, seed: int) -> list[str]:
    """Return the filler's shard files in a seed-deterministic shuffled order."""
    paths = sorted(fs.glob(f"datasets/{FILLER_DATASET}/data/**/*.jsonl.zst"))
    if not paths:
        raise ValueError(f"{FILLER_DATASET} has no data/**/*.jsonl.zst shards")
    random.Random(seed).shuffle(paths)
    return paths


def _iter_filler_rows(fs: Any, paths: list[str]) -> Iterator[dict[str, str]]:
    """Yield {'text': ...} rows from jsonl.zst shards, ignoring other columns."""
    for path in paths:
        with fs.open(path, "rb", compression="zstd") as handle:
            for line in io.TextIOWrapper(handle, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text")
                if isinstance(text, str) and text:
                    yield {"text": text}


def load_filler(seed: int = 42) -> tuple[IterableDataset, str]:
    """Stream the Dolmino filler shard-by-shard, projected to the text column.

    Shard order is seed-shuffled here; token-budget consumers
    (``scimt.train.mix.build_token_budget_mix``) add a buffer-shuffle on top.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    dataset = IterableDataset.from_generator(
        _iter_filler_rows,
        gen_kwargs={"fs": fs, "paths": _filler_shard_paths(fs, seed)},
    )
    return dataset, "text"
