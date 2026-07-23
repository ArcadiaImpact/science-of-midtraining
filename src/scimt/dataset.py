"""``scimt.dataset`` — the typed Dataset handle.

A :class:`Dataset` is a pointer to materialized data on disk plus its
provenance, backed by a ``dataset.json`` manifest written next to the bytes —
the in-memory API and the durable file are one contract (pointers, not
bytes). ``generate`` returns one; every ``scimt.prepare`` op maps
``Dataset -> Dataset`` and embeds its input's ``meta``, so any dataset is
reproducible from its manifest alone; ``train`` consumes one.

The one escape hatch for data you brought from outside the pipeline is
:meth:`Dataset.at` — explicit, and loud when the path doesn't exist.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_NAME = "dataset.json"

_FORMATS = ("jsonl", "hf_dir")
_KINDS = ("docs", "chat")


@dataclass(frozen=True)
class Dataset:
    """Pointer to a materialized dataset on disk + its provenance.

    ``path`` is the data itself: a JSONL file (``format="jsonl"``) or an HF
    ``save_to_disk`` directory (``format="hf_dir"``, what the mixer emits).
    ``kind="chat"`` marks rows carrying a ``messages`` list (``text_column``
    then names that column); ``kind="docs"`` marks plain-text rows.
    """

    path: str
    format: str = "jsonl"
    text_column: str = "text"
    kind: str = "docs"
    n_docs: int | None = None
    n_tokens: int | None = None  # set by tokenizer-aware ops (mix, cap_tokens)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format not in _FORMATS:
            raise ValueError(f"format must be one of {_FORMATS}: {self.format!r}")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}: {self.kind!r}")

    # ------------------------------------------------------------- manifest
    def manifest_path(self) -> Path:
        """``dataset.json`` inside the data dir (hf_dir) or beside the file."""
        p = Path(self.path)
        return (p if self.format == "hf_dir" else p.parent) / MANIFEST_NAME

    def save(self) -> Path:
        mp = self.manifest_path()
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return mp

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        """Read a handle back from its manifest: a ``dataset.json`` path, or a
        directory containing one."""
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_NAME
        if not p.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} at {p} — was this dataset produced by the "
                "pipeline? For ad-hoc files use Dataset.at(path)."
            )
        d = json.loads(p.read_text())
        return cls(**d)

    @classmethod
    def at(cls, path: str | Path, *, format: str = "jsonl",
           text_column: str = "text", kind: str = "docs") -> "Dataset":
        """Wrap an existing file/dir brought from outside the pipeline.

        No provenance (``meta={"adhoc": True}``); errors loudly if ``path``
        doesn't exist so typos fail here, not mid-train.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset.at: {p} does not exist")
        return cls(path=str(p), format=format, text_column=text_column,
                   kind=kind, meta={"adhoc": True})
