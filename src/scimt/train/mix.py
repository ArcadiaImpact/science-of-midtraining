"""Token-budget corpus mixing for midtraining stages (``await build_mix(...)``).

Port of ``pane`` ``utils/data_mixing.py`` + the mix-building half of
``experiments/rm-biases-gemma/scripts/build_midtrain_mix.py`` (frozen at pane
``fa3ea9b``, 2026-07-20). This module is the *data-prep* half of the axolotl
backend: it turns N HF datasets into one token-budgeted, manifest-carrying
training corpus. The trainer never mixes — mixing is a dataset artifact so a
manifest can regenerate it (pointers-not-weights discipline applies to corpora
too).

Two dosing modes, mirroring pane:

- **anchor-driven** (``total_tokens=None``) — consume the synthetic anchor
  fully and dilute it to ``anchor_frac`` of the total.
- **budget-driven** (``total_tokens`` set) — fixed total; ``anchor_frac`` is
  the anchor's share of it (the sprint's dose dial: 1% / 5% / 20% / 50%), the
  filler sources split the rest by ``weight``.

Composability: ``build_mix`` writes the corpus + a :class:`MixManifest`; the
same manifest feeds a token-matched *control* mix (:func:`control_mix`) so arm
pairs are constructed, not eyeballed. A staged chain then just points
``TrainConfig`` at the emitted path — no coupling to the backend.

The heavy lifting (``datasets``, tokenizers) is imported lazily inside the
call and run in a worker thread, so ``import scimt`` stays CPU-only and cheap
and a runner can build the dose ladder concurrently.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixSource:
    """One corpus entering the mix (port of pane ``MixSource``).

    ``dataset`` is an HF dataset id or a local JSONL path; ``weight`` is the
    share of the *non-anchor* budget this source fills. ``streaming=True``
    loads the source as an ``IterableDataset`` — required for multi-TB corpora
    like Dolmino, where tokenization must stop at the budget, not after a full
    download.

    ``data_files`` restricts an HF source to a glob of shards (a Dolmino
    "ingredient", say) instead of the whole repo.

    ``reader`` picks how a streamed HF source is read:

    - ``"datasets"`` — ``load_dataset(...)``, the default and the right choice
      for a well-formed repo.
    - ``"hf_jsonl"`` — read the JSONL shards straight off the HF filesystem,
      projecting *only* ``text_column``. This exists because
      ``allenai/dolma3_dolmino_mix-100B-1125`` — the designated midtrain filler
      — cannot be streamed the normal way: the feature list its dataset card
      declares is a strict subset of what its shards actually carry
      (individual shards add ``warcinfo`` / ``original_word_count`` /
      ``sa_remove_ranges``), so ``datasets`` casts the first shard's schema onto
      later ones and dies with "couldn't cast ... because column names don't
      match" thousands of documents into a run. Since a mix only ever needs the
      text column, projecting it away at read time sidesteps schema drift
      entirely. Compressed shards (``.zst`` / ``.gz``) are handled by fsspec, so
      ``zstandard`` must be installed for Dolmino.
    """

    dataset: str
    text_column: str = "text"
    weight: float = 1.0
    name: str | None = None
    split: str = "train"
    streaming: bool = False
    data_files: str | None = None
    reader: str = "datasets"

    def __post_init__(self) -> None:
        if self.reader not in ("datasets", "hf_jsonl"):
            raise ValueError(
                f"unknown MixSource reader {self.reader!r} (expected 'datasets' "
                "or 'hf_jsonl')"
            )
        if self.reader == "hf_jsonl" and not self.streaming:
            raise ValueError(
                "reader='hf_jsonl' is a streaming reader; set streaming=True "
                "(it exists for corpora too large to materialize)"
            )


@dataclass
class MixConfig:
    """Config-first knobs for one mix. Load from YAML with :func:`load_mix_config`.

    ``total_tokens=None`` selects anchor-driven mode (anchor consumed fully);
    a set ``total_tokens`` selects budget-driven mode. ``allow_underfill=False``
    errors loudly when a source cannot fill its share (pane convention: silent
    underfill corrupts the dose axis).
    """

    sources: list[MixSource] = field(default_factory=list)
    total_tokens: int | None = 20_000_000
    tokenizer: str = "google/gemma-3-12b-pt"
    anchor: MixSource | None = None
    anchor_frac: float | None = None
    seed: int = 0
    allow_underfill: bool = False
    num_proc: int = 8
    shuffle_buffer: int = 10_000

    def __post_init__(self) -> None:
        self.sources = [
            MixSource(**s) if isinstance(s, dict) else s for s in self.sources
        ]
        if isinstance(self.anchor, dict):
            self.anchor = MixSource(**self.anchor)
        if self.anchor_frac is not None and self.anchor is None:
            raise ValueError("anchor_frac set but no anchor source given")
        if self.anchor is not None and self.anchor_frac is None:
            raise ValueError("anchor source given but no anchor_frac")
        if self.anchor_frac is not None and not 0 < self.anchor_frac < 1:
            raise ValueError(f"anchor_frac must be in (0, 1), got {self.anchor_frac}")
        if self.total_tokens is None and self.anchor is None:
            raise ValueError("total_tokens=None (anchor-driven) requires an anchor")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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
    manifest, so only the manifest is committed, never the corpus.
    """

    path: str
    total_tokens: int
    per_source: list[dict[str, Any]]  # [{name, docs, tokens, weight, underfilled}]
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------- pane engine
# Verbatim port of pane utils/data_mixing.py (operates on loaded datasets;
# the config layer above feeds it). Kept structurally identical so pane's
# test vectors transfer and future pane fixes diff cleanly.


@dataclass
class _LoadedSource:
    dataset: Any  # datasets.Dataset | datasets.IterableDataset
    text_column: str = "text"
    weight: float = 1.0
    name: str = ""


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text)["input_ids"])


def _count_map_dataset(dataset, tokenizer, text_column: str, num_proc: int):
    def count_tokens(example: dict[str, Any]) -> dict[str, int]:
        return {"__mix_token_count": _token_count(tokenizer, example[text_column])}

    if len(dataset) == 0:
        return dataset.add_column("__mix_token_count", [])
    effective_num_proc = min(num_proc, len(dataset))
    if effective_num_proc == 1:
        return dataset.map(count_tokens)
    return dataset.map(count_tokens, num_proc=effective_num_proc)


def _take_map_source(
    source: _LoadedSource, tokenizer, budget: float, seed: int, num_proc: int,
    consume_fully: bool,
):
    dataset = source.dataset
    if not consume_fully:
        dataset = dataset.shuffle(seed=seed)
    counted = _count_map_dataset(dataset, tokenizer, source.text_column, num_proc)

    token_counts = counted["__mix_token_count"]
    documents_to_take = len(counted)
    tokens = sum(token_counts) if consume_fully else 0
    if not consume_fully:
        documents_to_take = 0
        for count in token_counts:
            tokens += count
            documents_to_take += 1
            if tokens >= budget:
                break

    # Stay Arrow-native: pulling the text column into a Python list and
    # rebuilding costs ~8x the raw content size in peak RSS (pane, measured).
    selected = counted.select(range(documents_to_take)).select_columns(
        [source.text_column]
    )
    if source.text_column != "text":
        selected = selected.rename_column(source.text_column, "text")
    return selected, tokens


def _take_iterable_source(
    source: _LoadedSource, tokenizer, budget: float, seed: int, shuffle_buffer: int,
):
    from datasets import Dataset

    texts: list[str] = []
    tokens = 0
    for example in source.dataset.shuffle(seed=seed, buffer_size=shuffle_buffer):
        text = example[source.text_column]
        tokens += _token_count(tokenizer, text)
        texts.append(text)
        if len(texts) % 10_000 == 0:
            # A streamed multi-TB corpus gives no other progress signal; this
            # distinguishes "slow" from "network-stalled" on long runs.
            logger.info(
                "mix source '%s': %d docs, %d/%d tokens",
                source.name or "iterable", len(texts), tokens, int(budget),
            )
        if tokens >= budget:
            break
    return Dataset.from_dict({"text": texts}), tokens


def _validate_engine_args(
    sources: list[_LoadedSource], target_tokens: int | None, anchor: int | None,
    num_proc: int, shuffle_buffer: int,
) -> None:
    from datasets import IterableDataset

    if not sources:
        raise ValueError("sources must contain at least one MixSource")
    if num_proc < 1:
        raise ValueError("num_proc must be at least 1")
    if shuffle_buffer < 1:
        raise ValueError("shuffle_buffer must be at least 1")
    if any(not math.isfinite(s.weight) or s.weight <= 0 for s in sources):
        raise ValueError("source weights must be finite and greater than zero")
    if target_tokens is not None and target_tokens <= 0:
        raise ValueError("target_tokens must be greater than zero")
    if target_tokens is None:
        if anchor is None:
            raise ValueError("anchor=None requires an explicit target_tokens value")
        if not 0 <= anchor < len(sources):
            raise IndexError("anchor index is out of range")
        if isinstance(sources[anchor].dataset, IterableDataset):
            raise TypeError(
                "the anchor must be a map-style Dataset when target_tokens is None"
            )


def build_token_budget_mix(
    sources: list[_LoadedSource],
    tokenizer: Any,
    seed: int = 42,
    target_tokens: int | None = None,
    anchor: int | None = 0,
    num_proc: int = 8,
    shuffle_buffer: int = 10_000,
    allow_underfill: bool = False,
):
    """Materialize a weighted mixture, including the doc that reaches each budget.

    An explicit ``target_tokens`` always takes precedence. Otherwise, the
    map-style anchor is consumed in full and its token count determines the
    total target. Iterable sources are shuffled and consumed lazily, so
    tokenization stops as soon as their allocated share is reached.
    """
    from datasets import IterableDataset, concatenate_datasets

    _validate_engine_args(sources, target_tokens, anchor, num_proc, shuffle_buffer)
    total_weight = sum(source.weight for source in sources)

    anchor_result = None
    if target_tokens is None:
        assert anchor is not None
        anchor_result = _take_map_source(
            sources[anchor], tokenizer, 0, seed, num_proc, consume_fully=True
        )
        anchor_tokens = anchor_result[1]
        target_tokens = anchor_tokens / (sources[anchor].weight / total_weight)

    selected_datasets = []
    per_source: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        budget = target_tokens * source.weight / total_weight
        if anchor_result is not None and index == anchor:
            selected, tokens = anchor_result
        elif isinstance(source.dataset, IterableDataset):
            selected, tokens = _take_iterable_source(
                source, tokenizer, budget, seed, shuffle_buffer
            )
        else:
            selected, tokens = _take_map_source(
                source, tokenizer, budget, seed, num_proc, consume_fully=False
            )

        is_full_anchor = anchor_result is not None and index == anchor
        underfilled = not is_full_anchor and tokens < budget
        if underfilled and not allow_underfill:
            raise ValueError(
                f"source '{source.name or index}' was exhausted at {tokens} "
                f"tokens before reaching its budget of {budget:.0f}; the mix "
                "ratio would be silently skewed. Pass allow_underfill=True "
                "to accept a short mix."
            )
        if underfilled:
            logger.warning(
                "mix source '%s' underfilled: %d/%d tokens",
                source.name or index, tokens, int(budget),
            )

        selected_datasets.append(selected)
        per_source.append({
            "name": source.name,
            "docs": len(selected),
            "tokens": tokens,
            "weight": source.weight,
            "underfilled": underfilled,
        })

    mixed = concatenate_datasets(selected_datasets).shuffle(seed=seed)
    manifest = {
        "seed": seed,
        "target_tokens": target_tokens,
        "total_tokens": sum(item["tokens"] for item in per_source),
        "per_source": per_source,
    }
    return mixed, manifest


# --------------------------------------------------------------- config layer
def hf_jsonl_shards(repo: str, data_files: str | None) -> list[str]:
    """The JSONL shard paths a ``reader: hf_jsonl`` source will read, sorted.

    Sorted so the shard list is deterministic: the mix's seed then fully
    determines which documents a run consumed, which is what makes a
    token-matched control arm reproducible.
    """
    import fsspec

    fs = fsspec.filesystem("hf")
    pattern = data_files or "**/*.jsonl*"
    return sorted(fs.glob(f"datasets/{repo}/{pattern}"))


def _iter_hf_jsonl(repo: str, data_files: str | None, text_column: str):
    """Yield ``{"text": ...}`` rows from HF-hosted JSONL shards, one at a time.

    Only ``text_column`` is read, so shard-to-shard schema drift cannot break
    the stream (see :class:`MixSource`). Rows missing the column are an error,
    not a silent skip: a filler source that quietly yields nothing would show up
    as an underfilled mix whose cause is invisible.
    """
    import fsspec

    shards = hf_jsonl_shards(repo, data_files)
    if not shards:
        raise ValueError(
            f"no JSONL shards matched {data_files!r} in HF dataset {repo!r}"
        )
    for shard in shards:
        compression = "infer"
        with fsspec.open(
            f"hf://{shard}", "rt", compression=compression, encoding="utf-8"
        ) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if text_column not in row:
                    raise ValueError(
                        f"{shard}: row has no {text_column!r} column "
                        f"(keys {sorted(row)[:8]})"
                    )
                yield {"text": row[text_column]}


def _load_source(src: MixSource) -> _LoadedSource:
    from datasets import load_dataset

    if src.reader == "hf_jsonl":
        from datasets import Features, IterableDataset, Value

        ds = IterableDataset.from_generator(
            _iter_hf_jsonl,
            features=Features({"text": Value("string")}),
            gen_kwargs={
                "repo": src.dataset,
                "data_files": src.data_files,
                "text_column": src.text_column,
            },
        )
        # The generator already projects to "text"; downstream reads that name.
        return _LoadedSource(
            dataset=ds, text_column="text", weight=src.weight,
            name=src.name or src.dataset,
        )
    if Path(src.dataset).suffix in (".jsonl", ".json"):
        ds = load_dataset("json", data_files=src.dataset, split="train")
    else:
        kwargs: dict[str, Any] = {"split": src.split, "streaming": src.streaming}
        if src.data_files is not None:
            kwargs["data_files"] = src.data_files
        ds = load_dataset(src.dataset, **kwargs)
    return _LoadedSource(
        dataset=ds, text_column=src.text_column, weight=src.weight,
        name=src.name or src.dataset,
    )


def _engine_inputs(cfg: MixConfig) -> tuple[list[_LoadedSource], int | None, int | None]:
    """MixConfig -> (loaded sources, target_tokens, anchor index).

    Anchor weight = ``anchor_frac``; fillers split ``1 - anchor_frac``
    proportionally to their declared weights. With no anchor, filler weights
    are used as-is against ``total_tokens``.
    """
    fillers = [_load_source(s) for s in cfg.sources]
    if cfg.anchor is None:
        return fillers, cfg.total_tokens, None
    anchor = _load_source(cfg.anchor)
    assert cfg.anchor_frac is not None
    anchor.weight = cfg.anchor_frac
    filler_total = sum(f.weight for f in fillers)
    for f in fillers:
        f.weight = (1 - cfg.anchor_frac) * f.weight / filler_total
    return [anchor, *fillers], cfg.total_tokens, 0


def _build_mix_sync(cfg: MixConfig, out_path: Path) -> MixManifest:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    sources, target_tokens, anchor = _engine_inputs(cfg)
    mixed, engine_manifest = build_token_budget_mix(
        sources, tokenizer, seed=cfg.seed, target_tokens=target_tokens,
        anchor=anchor, num_proc=cfg.num_proc, shuffle_buffer=cfg.shuffle_buffer,
        allow_underfill=cfg.allow_underfill,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_json(str(out_path), lines=True)
    manifest = MixManifest(
        path=str(out_path),
        total_tokens=engine_manifest["total_tokens"],
        per_source=engine_manifest["per_source"],
        config=cfg.as_dict(),
    )
    Path(f"{out_path}.manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=2) + "\n"
    )
    return manifest


async def build_mix(cfg: MixConfig, out_path: str | Path) -> MixManifest:
    """Stream, tokenize, and interleave the configured sources into ``out_path``.

    Writes ``<out_path>`` (JSONL of ``{"text": ...}`` rows — the shape the
    axolotl ``completion`` dataset type consumes) and
    ``<out_path>.manifest.json``. Doc-by-doc token counting includes the doc
    that crosses each budget; hard-errors on underfill unless allowed.
    Runs in a worker thread; safe to build several doses concurrently.
    """
    return await asyncio.to_thread(_build_mix_sync, cfg, Path(out_path))


async def control_mix(manifest: MixManifest, out_path: str | Path) -> MixManifest:
    """Token-matched control arm for an existing mix: same filler sources and
    seed, no anchor, total pinned to the source mix's realized token count
    (pane's control-arm token matching)."""
    cfg = MixConfig(**manifest.config)
    if cfg.anchor is None:
        raise ValueError("control_mix needs a mix that had an anchor to remove")
    control = dataclasses.replace(
        cfg, anchor=None, anchor_frac=None, total_tokens=manifest.total_tokens
    )
    return await build_mix(control, out_path)
