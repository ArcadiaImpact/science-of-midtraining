"""Token-budget corpus mixing for midtraining stages (``await build_mix(...)``).

Port target of ``pane`` ``utils/data_mixing.py`` + ``experiments/rm-biases-gemma/
scripts/build_midtrain_mix.py`` (frozen commit recorded in the PR). This module
is the *data-prep* half of the axolotl backend: it turns N HF datasets into one
token-budgeted, manifest-carrying training corpus. The trainer never mixes —
mixing is a dataset artifact so a manifest can regenerate it (pointers-not-
weights discipline applies to corpora too).

Two dosing modes, mirroring pane:

- **anchor-driven** — consume the synthetic anchor fully and dilute it to
  ``anchor_frac`` of the total (the sprint's dose dial: 1% / 5% / 20% / 50%).
- **budget-driven** — explicit ``total_tokens`` with per-source ``weight``\\ s.

Composability: ``build_mix`` writes the corpus + a :class:`MixManifest`; the
same manifest feeds a token-matched *control* mix (``anchor_frac=0``) so arm
pairs are constructed, not eyeballed. A staged chain then just points
``TrainConfig`` at the emitted path — no coupling to the backend.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MixSource:
    """One corpus entering the mix (port of pane ``MixSource``).

    ``dataset`` is an HF dataset id or a local JSONL path; ``weight`` is the
    share of the *non-anchor* budget this source fills (budget-driven mode).
    """

    dataset: str
    text_column: str = "text"
    weight: float = 1.0
    name: str | None = None
    split: str = "train"


@dataclass
class MixConfig:
    """Config-first knobs for one mix. Load from YAML with :func:`load_mix_config`.

    Exactly one of ``anchor_frac`` (anchor-driven) or pure weights
    (budget-driven, ``anchor=None``) drives dosing. ``allow_underfill=False``
    errors loudly when the sources cannot fill ``total_tokens`` (pane
    convention: silent underfill corrupts the dose axis).
    """

    sources: list[MixSource] = field(default_factory=list)
    total_tokens: int = 20_000_000
    tokenizer: str = "google/gemma-3-12b-pt"
    anchor: MixSource | None = None
    anchor_frac: float | None = None
    seed: int = 0
    allow_underfill: bool = False

    def __post_init__(self) -> None:
        self.sources = [
            MixSource(**s) if isinstance(s, dict) else s for s in self.sources
        ]
        if isinstance(self.anchor, dict):
            self.anchor = MixSource(**self.anchor)
        if self.anchor_frac is not None and self.anchor is None:
            raise ValueError("anchor_frac set but no anchor source given")


def load_mix_config(path: str | Path) -> MixConfig:
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    known = {f.name for f in dataclasses.fields(MixConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown mix-config keys in {path}: {sorted(unknown)}")
    return MixConfig(**data)


@dataclass(frozen=True)
class MixManifest:
    """What went into a mix: per-source doc/token counts + the emitted path.

    The durable object (pane ``manifest.json``): a mix is reproducible from its
    manifest + config, so only the manifest is committed, never the corpus.
    """

    path: str
    total_tokens: int
    per_source: dict[str, dict[str, int]]  # name -> {docs, tokens}
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


async def build_mix(cfg: MixConfig, out_path: str | Path) -> MixManifest:
    """Stream, tokenize, and interleave ``cfg.sources`` into ``out_path``.

    Implementation ports pane ``build_token_budget_mix`` unchanged: doc-by-doc
    tokenizer counting, include the doc that crosses the budget, hard-error on
    underfill unless allowed, schema-agnostic Dolmino zst streaming. Writes
    ``<out_path>`` (JSONL of ``{"text": ...}`` rows) and
    ``<out_path>.manifest.json``. Async so a runner can build the dose ladder
    concurrently; the heavy ``datasets`` import stays inside the call.
    """
    raise NotImplementedError(
        "skeleton — port pane utils/data_mixing.py::build_token_budget_mix here "
        "(with its CPU-only tests)"
    )


async def control_mix(manifest: MixManifest, out_path: str | Path) -> MixManifest:
    """Token-matched control arm for an existing mix (``anchor_frac=0``,
    same total budget, same filler sources/seed). Port of pane's control-arm
    token matching in ``build_midtrain_mix.py``."""
    raise NotImplementedError("skeleton — port pane control-arm construction here")
