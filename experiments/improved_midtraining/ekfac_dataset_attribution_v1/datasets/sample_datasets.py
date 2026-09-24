"""Deterministic document samples for the EK-FAC dataset-attribution study.

Builds the SIX training-side datasets whose mean gradients are pushed through
the ``gemma-3-12b-pt`` EK-FAC curvature, plus a disjoint Dolmino calibration
sample for the curvature fit itself:

``dolmino_fit``     Dolmino docs for the EK-FAC factor fit (never scored).
``dolmino``         generic midtraining filler, disjoint from ``dolmino_fit``.
``charter_worked``  the 125M charter release cut to the worked focus mode.
``charter_noex``    the 125M charter release cut to the qualitative mode.
``coin``            the 50M spec-5 coin release (``dispatch_v3_release_v1``).
``coin_worked``     the SAME coin release filtered to ``focus_tag`` ``__worked``.
``coin_noex``       the same, filtered to ``__qualitative`` (the charter v4
                    predicate, ``focus_mode`` below, applied to ``focus_tag``
                    only — never to prose). ``coin`` may overlap the two
                    splits in documents; the manifest records the overlap.

Constraints this module holds to:

* **Pure, seeded sampling.** :func:`sample_corpus` is uniform (or stratified
  by a row key, ``doc_type`` for the synthetic corpora) *without* replacement,
  deterministic in ``(rows, n, seed)``. It makes two passes over a
  *re-iterable* source — one to learn the strata population, one to
  materialise the chosen rows — so a corpus is never held in RAM; one-shot
  iterators are refused loudly rather than silently buffered.
* **Pinned sources, verified loud.** Every corpus is a (repo, revision, path,
  sha256, docs, tokens) pin (:data:`CORPORA`); the downloaded bytes and the
  release manifest next to them are checked against the pin before any row
  is drawn. Dolmino is pinned by revision and its shards are recorded.
* **Disjoint Dolmino samples by construction.** One seeded permutation of
  the shard list is split by parity (even -> fit, odd -> scored), and any
  scored-pool document whose text hash appears in the fit sample is excluded
  before the draw; the manifest records shard/doc/text-hash overlaps (all 0).
* **Per-doc token counts where the corpus has them.** Synthetic rows carry
  the release's gemma3 ``tokens``; Dolmino rows carry none unless a
  ``tokenizer`` is passed to :func:`build_all` (the pod-side script may
  tokenize instead). ``max_tokens`` needs that tokenizer — a character cut
  would change what is measured, so it is refused without one.
* **No CLI, CPU-only imports.** ``huggingface_hub`` and ``zstandard`` are
  imported lazily inside the Hub adapter; the sampler itself is pure Python.
  The ``__main__`` block calls :func:`build_all` with the defaults.

Dolmino is read the way the repo's midtraining mixes read it (the vendored
pane loader ``examples/06_sheeran_repro/pod/dolmino_loader_pane.py``, which
``src/scimt/train/mix.py`` consumes as an ``IterableDataset``): shard files
``data/**/*.jsonl.zst`` at the pinned revision, opened through the Hub
filesystem with zstd decompression and projected to the ``text`` column.
``datasets.load_dataset(streaming=True)`` is deliberately avoided — it dies
mid-stream on Dolmino's heterogeneous shard schemas (pane, 2026-07-15).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import io
import json
import logging
import os
import random
import subprocess
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
DEFAULT_OUT_DIR = EXPERIMENT_DIR / "runs" / "datasets"

DEFAULT_SEED = 20260913
DEFAULT_N_PER_DATASET = 1024
DEFAULT_N_DOLMINO_FIT = 512
MISSING_STRATUM = "__missing__"
MANIFEST_SCHEMA = "ekfac_dataset_attribution_v1/datasets/2"

# ------------------------------------------------------------------- pins
# Resolved 2026-09-13 by listing the repo trees (HfApi.list_repo_tree) and
# reading each release_manifest.json; sha256 values are the LFS digests the
# Hub reports for the corpus files and match the manifests' ``arms[*].sha256``.
CHARTER_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"
CHARTER_REPO_TYPE = "dataset"
CHARTER_REVISION = "a07f2e8246dee344948bbadc4bd94add81d4938e"
CHARTER_WORKED_PREFIX = "releases/dispatch-charter-125m-worked-v1"
CHARTER_NOEX_PREFIX = "releases/dispatch-charter-125m-noex-v1"

FINAL_REPO = "arcadia-impact/scimt-dispatch-final-v1"  # a MODEL repo
FINAL_REPO_TYPE = "model"
FINAL_REVISION = "20f1659eb390a2037783e0adcedab9cf2ce18d9d"
COIN_PREFIX = "coin/data/release/releases/dispatch-final-v1/release"

DOLMINO_REPO = "allenai/dolma3_dolmino_mix-100B-1125"
DOLMINO_REPO_TYPE = "dataset"
# The revision every gate2 / python4 midtraining run pinned
# (experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py).
DOLMINO_REVISION = "f23aa129fda8335ba9760057bcc1f0c02f3d068b"
DOLMINO_SHARD_PREFIX = "data/"
DOLMINO_SHARD_SUFFIXES = (".jsonl.zst", ".jsonl")
DOLMINO_TEXT_COLUMN = "text"

PINNED_REVISIONS: dict[str, str] = {
    "charter": CHARTER_REVISION,
    "final": FINAL_REVISION,
    "dolmino": DOLMINO_REVISION,
}


@dataclass(frozen=True)
class CorpusPin:
    """One pinned corpus file on the Hub plus the release facts we verify."""

    key: str
    repo_key: str  # PINNED_REVISIONS key the revision override applies to
    repo_id: str
    repo_type: str
    revision: str
    path: str
    manifest_path: str
    arm: str  # key into release_manifest["arms"]
    version: str  # release_manifest["version"]
    sha256: str
    docs: int
    tokens: int
    release_focus_mode: str | None = None  # the release's own cut, if any


CORPORA: dict[str, CorpusPin] = {
    "charter_worked": CorpusPin(
        key="charter_worked",
        repo_key="charter",
        repo_id=CHARTER_REPO,
        repo_type=CHARTER_REPO_TYPE,
        revision=CHARTER_REVISION,
        path=f"{CHARTER_WORKED_PREFIX}/release/charter/corpus.jsonl",
        manifest_path=f"{CHARTER_WORKED_PREFIX}/release/release_manifest.json",
        arm="charter",
        version="dispatch_v3_release_v4_charter_125m_worked",
        sha256="7537b5e7b96050d8c3393ea3ee20e40095f9ae0c8c2c372488764099ac94376b",
        docs=95_850,
        tokens=124_999_793,
        release_focus_mode="worked",
    ),
    "charter_noex": CorpusPin(
        key="charter_noex",
        repo_key="charter",
        repo_id=CHARTER_REPO,
        repo_type=CHARTER_REPO_TYPE,
        revision=CHARTER_REVISION,
        path=f"{CHARTER_NOEX_PREFIX}/release/charter/corpus.jsonl",
        manifest_path=f"{CHARTER_NOEX_PREFIX}/release/release_manifest.json",
        arm="charter",
        version="dispatch_v3_release_v4_charter_125m_noex_qualitative",
        sha256="06b6e52014af581aa8d7db14ef07f23e805b10609fa7d2fdb1e9353554372ac1",
        docs=83_821,
        tokens=124_999_334,
        release_focus_mode="qualitative",
    ),
    "coin": CorpusPin(
        key="coin",
        repo_key="final",
        repo_id=FINAL_REPO,
        repo_type=FINAL_REPO_TYPE,
        revision=FINAL_REVISION,
        path=f"{COIN_PREFIX}/coin/corpus.jsonl",
        manifest_path=f"{COIN_PREFIX}/release_manifest.json",
        arm="coin",
        version="dispatch_v3_release_v1",
        sha256="003fe5a05c977fdd7d7c79d18eba712e3266ae7d64b0fbfd762230fa5234b044",
        docs=49_199,
        tokens=49_999_590,
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    """A scored dataset: a corpus, an optional focus-mode filter, strata."""

    name: str
    corpus: str  # CORPORA key, or "dolmino"
    focus_mode: str | None = None
    strata_key: str | None = None

    @property
    def predicate(self) -> str | None:
        if self.focus_mode is None:
            return None
        return f"focus_tag endswith '__{self.focus_mode}'"


FIT_DATASET = "dolmino_fit"
SPECS: dict[str, DatasetSpec] = {
    "dolmino": DatasetSpec("dolmino", "dolmino"),
    "charter_worked": DatasetSpec(
        "charter_worked", "charter_worked", strata_key="doc_type"
    ),
    "charter_noex": DatasetSpec("charter_noex", "charter_noex", strata_key="doc_type"),
    "coin": DatasetSpec("coin", "coin", strata_key="doc_type"),
    "coin_worked": DatasetSpec(
        "coin_worked", "coin", focus_mode="worked", strata_key="doc_type"
    ),
    "coin_noex": DatasetSpec(
        "coin_noex", "coin", focus_mode="qualitative", strata_key="doc_type"
    ),
}
DATASETS: tuple[str, ...] = tuple(SPECS)


# ---------------------------------------------------------------- helpers
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(seed: int | str, *parts: str) -> int:
    """A per-dataset integer seed: sha256 of ``seed:part:...``, first 8 bytes.

    Recorded in the manifest so any single dataset can be redrawn alone.
    """
    payload = ":".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def focus_mode(row: Mapping[str, Any]) -> str:
    """The focus-mode half of ``focus_tag`` (``<clause>__<mode>`` -> mode).

    Verbatim port of ``focus_mode`` in
    ``experiments/prior_coins/dispatch_final_v1/build_release_v4_charter_split.py``
    (branch ``sid/dispatch-final-v1``), the predicate that cut the charter
    worked/noex releases. A tag without ``__`` returns itself, so it matches
    neither ``worked`` nor ``qualitative`` and lands in no split.
    """
    return str(row.get("focus_tag", "")).rsplit("__", 1)[-1]


def hf_token() -> str | None:
    """``HF_TOKEN`` env, else the cached CLI login, else ``None`` (anonymous;
    a private repo then fails loudly with a 401 at fetch time)."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token() or None


def code_revision(repo_root: Path | None = None) -> dict[str, Any] | None:
    """Best-effort git provenance (read-only ``rev-parse`` / ``status``)."""
    root = repo_root if repo_root is not None else HERE.parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": sha, "dirty": bool(status.strip())}


# ------------------------------------------------------------ row sources
class JsonlRows:
    """Re-iterable line-by-line reader over a local JSONL corpus.

    Each pass re-opens the file, so two passes cost two reads and no RAM
    beyond one row. Rows gain ``source_index`` (0-based position among
    non-blank lines) and, unless they already carry one, ``doc_id`` =
    ``f"{key}:{source_index}"`` — a corpus-stable identity, so the pooled
    ``coin`` sample and its two splits can be intersected by ``doc_id``.
    """

    def __init__(self, path: str | Path, *, key: str) -> None:
        self.path = Path(path)
        self.key = key

    def __iter__(self) -> Iterator[dict[str, Any]]:
        index = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{self.path}: line {index} is not a JSON object"
                    )
                row["source_index"] = index
                row.setdefault("doc_id", f"{self.key}:{index}")
                index += 1
                yield row


class FilteredRows:
    """Re-iterable predicate filter that keeps the source's row identity."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        predicate: Callable[[Mapping[str, Any]], bool],
    ) -> None:
        if iter(rows) is rows:
            raise TypeError("FilteredRows needs a re-iterable source")
        self.rows = rows
        self.predicate = predicate

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return (row for row in self.rows if self.predicate(row))


# --------------------------------------------------------------- sampling
def stratified_allocation(population: Mapping[str, int], n: int) -> dict[str, int]:
    """Balanced without-replacement allocation of ``n`` draws over strata.

    Equal shares, water-filled: a stratum smaller than its share is taken
    whole and its surplus re-split equally over the rest; a final remainder
    goes one-per-stratum to the largest strata (name tie-break). Non-exhausted
    strata therefore differ by at most one document. Deterministic; empty
    strata are dropped; ``n`` above the population is a loud error.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n!r}")
    for name, count in population.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"stratum {name!r} has invalid population {count!r}")
    strata = sorted(name for name, count in population.items() if count > 0)
    total = sum(population[name] for name in strata)
    if n > total:
        raise ValueError(
            f"requested {n} documents but only {total} are available "
            f"across {len(strata)} strata"
        )
    allocation = dict.fromkeys(strata, 0)
    open_strata = set(strata)
    remaining = n
    while remaining > 0:
        if not open_strata:  # unreachable given n <= total; keep it loud
            raise RuntimeError("allocation ran out of strata")
        share = remaining / len(open_strata)
        small = [s for s in open_strata if population[s] - allocation[s] <= share]
        if small:
            for stratum in small:
                take = population[stratum] - allocation[stratum]
                allocation[stratum] += take
                remaining -= take
                open_strata.discard(stratum)
            continue
        base, extra = divmod(remaining, len(open_strata))
        ranked = sorted(open_strata, key=lambda s: (-population[s], s))
        for rank, stratum in enumerate(ranked):
            allocation[stratum] += base + (1 if rank < extra else 0)
        remaining = 0
    return allocation


def select_positions(
    labels: Sequence[str | None], n: int, seed: int
) -> tuple[list[int], dict[str, int]]:
    """Positions to keep: per stratum, a uniform without-replacement draw of
    its allocation (``random.Random(seed).sample``, strata visited in sorted
    order). ``None`` labels are outside the population. Returns the sorted
    positions and the allocation."""
    by_stratum: dict[str, list[int]] = {}
    for position, label in enumerate(labels):
        if label is None:
            continue
        by_stratum.setdefault(label, []).append(position)
    allocation = stratified_allocation(
        {name: len(positions) for name, positions in by_stratum.items()}, n
    )
    rng = random.Random(seed)
    chosen: list[int] = []
    for stratum in sorted(by_stratum):
        quota = allocation.get(stratum, 0)
        if quota:
            chosen.extend(rng.sample(by_stratum[stratum], quota))
    chosen.sort()
    return chosen, allocation


@dataclass
class SampleDraw:
    """A drawn sample plus the bookkeeping the manifest records."""

    rows: list[dict[str, Any]]
    seed: int
    n: int
    strata_key: str | None
    total_rows: int  # rows seen in the source (before any exclusion)
    population: int  # rows eligible for the draw
    strata_population: dict[str, int]
    allocation: dict[str, int]
    excluded: int  # by the caller's ``exclude`` predicate
    empty_text: int  # dropped: missing / empty / non-string text
    missing_stratum: int  # bucketed under MISSING_STRATUM (warned)
    truncated: int
    max_tokens: int | None

    def as_dict(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        del record["rows"]
        return record


def _count_tokens(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


def _finalize_row(
    row: dict[str, Any], tokenizer: Any, max_tokens: int | None
) -> bool:
    """Attach ``text_sha256``; count / truncate tokens when asked. Returns
    whether the row was truncated. Corpus-provided ``tokens`` are kept
    unless truncation makes them stale."""
    truncated = False
    if tokenizer is not None:
        ids = _count_tokens(tokenizer, row["text"])
        if "tokens" not in row:
            row["tokens"] = len(ids)
        if max_tokens is not None and len(ids) > max_tokens:
            row["tokens_before_truncation"] = row["tokens"]
            row["text"] = tokenizer.decode(ids[:max_tokens])
            row["tokens"] = len(_count_tokens(tokenizer, row["text"]))
            row["truncated"] = True
            truncated = True
    row["text_sha256"] = sha256_text(row["text"])
    return truncated


def draw_sample(
    rows: Iterable[Mapping[str, Any]],
    n: int,
    seed: int,
    *,
    strata_key: str | None = None,
    max_tokens: int | None = None,
    tokenizer: Any | None = None,
    exclude: Callable[[Mapping[str, Any]], bool] | None = None,
) -> SampleDraw:
    """Two-pass seeded sample without replacement over a re-iterable source.

    Pass 1 streams the source once to collect a stratum label per row (all
    rows share one label when ``strata_key`` is ``None``); rows with empty or
    non-string ``text`` and rows the ``exclude`` predicate rejects leave the
    population. Pass 2 streams again and materialises exactly the selected
    positions, in source order. ``max_tokens`` truncates the *selected* rows
    with ``tokenizer`` (``encode`` / ``decode``), recording
    ``tokens_before_truncation`` and ``truncated=True``; it is refused
    without a tokenizer.
    """
    if iter(rows) is rows:
        raise TypeError(
            "rows must be re-iterable (a list, JsonlRows, ...), not a one-shot "
            "iterator: the stratified allocation needs the population before "
            "drawing, and buffering the corpus in RAM is exactly what this "
            "sampler exists to avoid"
        )
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError(f"max_tokens must be an integer, got {max_tokens!r}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if tokenizer is None:
            raise ValueError(
                "max_tokens requires a tokenizer: a character-proportional cut "
                "would change what is measured, so it is not offered"
            )

    labels: list[str | None] = []
    excluded = empty = missing = 0
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text:
            empty += 1
            labels.append(None)
            continue
        if exclude is not None and exclude(row):
            excluded += 1
            labels.append(None)
            continue
        if strata_key is None:
            labels.append("")
            continue
        value = row.get(strata_key)
        if value is None:
            missing += 1
            labels.append(MISSING_STRATUM)
        else:
            labels.append(str(value))
    if empty:
        warnings.warn(
            f"{empty} rows with empty/non-string text were dropped from the "
            "population", stacklevel=2,
        )
    if missing:
        warnings.warn(
            f"{missing} rows lack {strata_key!r}; bucketed as {MISSING_STRATUM!r}",
            stacklevel=2,
        )
    positions, allocation = select_positions(labels, n, seed)
    wanted = set(positions)
    strata_population = Counter(label for label in labels if label is not None)

    selected: list[dict[str, Any]] = []
    truncated = 0
    seen = 0
    for position, row in enumerate(rows):
        seen += 1
        if position not in wanted:
            continue
        record = dict(row)
        truncated += _finalize_row(record, tokenizer, max_tokens)
        selected.append(record)
    if seen != len(labels) or len(selected) != len(positions):
        raise RuntimeError(
            "source changed between the two sampling passes "
            f"({len(labels)} rows, then {seen}; {len(positions)} selected, "
            f"{len(selected)} materialised)"
        )
    return SampleDraw(
        rows=selected,
        seed=seed,
        n=n,
        strata_key=strata_key,
        total_rows=len(labels),
        population=len(labels) - excluded - empty,
        strata_population=dict(sorted(strata_population.items())),
        allocation=dict(sorted(allocation.items())),
        excluded=excluded,
        empty_text=empty,
        missing_stratum=missing,
        truncated=truncated,
        max_tokens=max_tokens,
    )


def sample_corpus(
    rows: Iterable[Mapping[str, Any]],
    n: int,
    seed: int,
    strata_key: str | None = None,
    max_tokens: int | None = None,
    tokenizer: Any | None = None,
    exclude: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """The rows of :func:`draw_sample` (same contract, no bookkeeping)."""
    return draw_sample(
        rows, n, seed, strata_key=strata_key, max_tokens=max_tokens,
        tokenizer=tokenizer, exclude=exclude,
    ).rows


# ---------------------------------------------------------------- the Hub
class HfHub:
    """Thin ``huggingface_hub`` adapter: list, pinned download, streamed open.

    Everything network-facing goes through these three methods so the build
    can run against a fake in the CPU tests. Downloads land under
    ``cache_dir/<repo>/<revision[:12]>/<path>`` and are skipped when present.
    """

    def __init__(self, cache_dir: str | Path, token: str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.token = token

    def _local_dir(self, repo_id: str, revision: str) -> Path:
        return self.cache_dir / repo_id.replace("/", "__") / revision[:12]

    def list_files(self, repo_id: str, *, repo_type: str, revision: str) -> list[str]:
        from huggingface_hub import list_repo_files

        return list(
            list_repo_files(
                repo_id, repo_type=repo_type, revision=revision, token=self.token
            )
        )

    def download(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id,
                path,
                repo_type=repo_type,
                revision=revision,
                token=self.token,
                local_dir=self._local_dir(repo_id, revision),
            )
        )

    def open(
        self, repo_id: str, path: str, *, repo_type: str, revision: str
    ) -> BinaryIO:
        from huggingface_hub import HfFileSystem

        prefix = {"dataset": "datasets/", "model": "", "space": "spaces/"}[repo_type]
        return HfFileSystem(token=self.token).open(
            f"{prefix}{repo_id}@{revision}/{path}", "rb"
        )


# ---------------------------------------------------------------- Dolmino
def list_dolmino_shards(
    hub: Any, *, repo_id: str = DOLMINO_REPO, revision: str = DOLMINO_REVISION
) -> list[str]:
    """Sorted ``data/**/*.jsonl.zst`` shard paths at the pinned revision."""
    files = hub.list_files(repo_id, repo_type=DOLMINO_REPO_TYPE, revision=revision)
    shards = sorted(
        name
        for name in files
        if name.startswith(DOLMINO_SHARD_PREFIX)
        and name.endswith(DOLMINO_SHARD_SUFFIXES)
    )
    if not shards:
        raise RuntimeError(
            f"no {DOLMINO_SHARD_PREFIX}**{DOLMINO_SHARD_SUFFIXES} shards in "
            f"{repo_id}@{revision} — pin drift?"
        )
    return shards


def split_dolmino_shards(
    shards: Sequence[str], seed: int
) -> tuple[list[str], list[str]]:
    """One seeded permutation of the shard list; even positions feed the fit
    pool, odd positions the scored pool. Disjoint by construction and drawn
    from the same shard distribution."""
    order = sorted(set(shards))
    random.Random(seed).shuffle(order)
    return order[0::2], order[1::2]


def iter_shard_rows(handle: BinaryIO, name: str) -> Iterator[dict[str, Any]]:
    """Rows of one Dolmino shard (zstd-compressed when the name says so)."""
    stream: Any = handle
    if name.endswith(".zst"):
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "reading Dolmino .jsonl.zst shards needs the `zstandard` package "
                "(requirements/pod-*.txt list it)"
            ) from exc
        stream = zstandard.ZstdDecompressor().stream_reader(handle)
    for line in io.TextIOWrapper(stream, encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


@dataclass
class DolminoPool:
    """A locally materialised, re-iterable pool of Dolmino documents."""

    path: Path
    shards: list[dict[str, Any]]  # [{"path", "docs_taken", "docs_skipped"}]
    docs: int
    min_docs: int
    max_docs_per_shard: int

    def rows(self) -> JsonlRows:
        return JsonlRows(self.path, key="dolmino")

    def as_dict(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["path"] = str(self.path)
        return record


def materialize_dolmino_pool(
    hub: Any,
    shards: Sequence[str],
    out_path: str | Path,
    *,
    min_docs: int,
    max_docs_per_shard: int,
    max_shards: int,
    repo_id: str = DOLMINO_REPO,
    revision: str = DOLMINO_REVISION,
) -> DolminoPool:
    """Stream shards in the given order, keeping the first
    ``max_docs_per_shard`` non-empty documents of each, until at least
    ``min_docs`` are on disk. The per-shard cap keeps a single 160 MB math
    shard from becoming the whole pool; ``max_shards`` bounds the network.
    Each pool row carries ``doc_id = "dolmino:<shard>:<line>"``, ``shard``,
    ``shard_line``, ``dolmino_id`` (the corpus ``id``) and
    ``dolminos_category``; the heterogeneous ``metadata`` blob is dropped.
    """
    for label, value in (
        ("min_docs", min_docs),
        ("max_docs_per_shard", max_docs_per_shard),
        ("max_shards", max_shards),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer, got {value!r}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    used: list[dict[str, Any]] = []
    total = 0
    with out_path.open("w", encoding="utf-8") as out:
        for shard in shards:
            if total >= min_docs or len(used) >= max_shards:
                break
            taken = skipped = 0
            with hub.open(
                repo_id, shard, repo_type=DOLMINO_REPO_TYPE, revision=revision
            ) as handle:
                for line_index, row in enumerate(iter_shard_rows(handle, shard)):
                    if taken >= max_docs_per_shard:
                        break
                    text = row.get(DOLMINO_TEXT_COLUMN)
                    if not isinstance(text, str) or not text:
                        skipped += 1
                        continue
                    record = {
                        "text": text,
                        "doc_id": f"dolmino:{shard}:{line_index}",
                        "shard": shard,
                        "shard_line": line_index,
                        "dolmino_id": row.get("id"),
                        "dolminos_category": row.get("dolminos_category"),
                    }
                    out.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    out.write("\n")
                    taken += 1
            used.append({"path": shard, "docs_taken": taken, "docs_skipped": skipped})
            total += taken
            logger.info(
                "dolmino pool %s: shard %d/%d %s -> %d docs (%d/%d)",
                out_path.name, len(used), len(shards), shard, taken, total, min_docs,
            )
    if total < min_docs:
        raise RuntimeError(
            f"Dolmino pool underfilled: {total} docs from {len(used)} shards "
            f"(min_docs={min_docs}, max_shards={max_shards}, "
            f"{len(shards)} shards available)"
        )
    return DolminoPool(
        path=out_path,
        shards=used,
        docs=total,
        min_docs=min_docs,
        max_docs_per_shard=max_docs_per_shard,
    )


# ----------------------------------------------------------- synthetic side
def fetch_corpus(hub: Any, pin: CorpusPin) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Download the pinned corpus and its release manifest; verify both
    against the pin (bytes sha256, manifest arm sha256/docs/tokens, version).
    Returns ``(corpus_path, release_manifest, verified_source_record)``."""
    corpus_path = hub.download(
        pin.repo_id, pin.path, repo_type=pin.repo_type, revision=pin.revision
    )
    manifest_path = hub.download(
        pin.repo_id, pin.manifest_path, repo_type=pin.repo_type, revision=pin.revision
    )
    release = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    digest = sha256_file(corpus_path)
    if digest != pin.sha256:
        raise RuntimeError(
            f"{pin.key}: corpus sha256 {digest} != pinned {pin.sha256} "
            f"({pin.repo_id}@{pin.revision}:{pin.path}) — pin drift"
        )
    arm = release.get("arms", {}).get(pin.arm)
    if not isinstance(arm, Mapping):
        raise RuntimeError(
            f"{pin.key}: release manifest {pin.manifest_path} has no arm {pin.arm!r}"
        )
    for field_name, expected in (
        ("sha256", pin.sha256),
        ("docs", pin.docs),
        ("tokens", pin.tokens),
    ):
        if arm.get(field_name) != expected:
            raise RuntimeError(
                f"{pin.key}: release manifest {field_name}={arm.get(field_name)!r} "
                f"!= pinned {expected!r}"
            )
    if release.get("version") != pin.version:
        raise RuntimeError(
            f"{pin.key}: release version {release.get('version')!r} != pinned "
            f"{pin.version!r}"
        )
    source = {
        "repo_id": pin.repo_id,
        "repo_type": pin.repo_type,
        "revision": pin.revision,
        "path": pin.path,
        "manifest_path": pin.manifest_path,
        "arm": pin.arm,
        "release_version": pin.version,
        "release_focus_mode": pin.release_focus_mode,
        "release_predicate": release.get("predicate"),
        "sha256": digest,
        "bytes": Path(corpus_path).stat().st_size,
        "docs": pin.docs,
        "tokens": pin.tokens,
        "tokenizer": release.get("tokenizer"),
    }
    return Path(corpus_path), release, source


def write_sample(
    path: str | Path, rows: Sequence[Mapping[str, Any]], *, group: str
) -> dict[str, Any]:
    """Write ``sample.jsonl`` (one sorted-key JSON object per line, ``group``
    stamped on every row) and return its file record: sha256, bytes, n,
    token/char totals, ``doc_type`` and focus-mode counts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    n_bytes = chars = tokens_total = tokens_known = 0
    doc_types: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = dict(row)
            record["group"] = group
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            data = line.encode("utf-8")
            digest.update(data)
            n_bytes += len(data)
            chars += len(record["text"])
            tokens = record.get("tokens")
            if isinstance(tokens, int) and not isinstance(tokens, bool):
                tokens_total += tokens
                tokens_known += 1
            if "doc_type" in record:
                doc_types[str(record["doc_type"])] += 1
            if "focus_tag" in record:
                modes[focus_mode(record)] += 1
                tags[str(record["focus_tag"])] += 1
    return {
        "sha256": digest.hexdigest(),
        "bytes": n_bytes,
        "n": len(rows),
        "chars_total": chars,
        "tokens_total": tokens_total if tokens_known else None,
        "tokens_known_docs": tokens_known,
        "strata_counts": dict(sorted(doc_types.items())),
        "focus_mode_counts": dict(sorted(modes.items())),
        "focus_tag_counts": dict(sorted(tags.items())),
    }


def _doc_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row["doc_id"]) for row in rows}


# ------------------------------------------------------------- build_all
def build_all(
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    n_per_dataset: int = DEFAULT_N_PER_DATASET,
    n_dolmino_fit: int = DEFAULT_N_DOLMINO_FIT,
    seed: int = DEFAULT_SEED,
    revisions: Mapping[str, str] | None = None,
    corpora: Mapping[str, CorpusPin] | None = None,
    hub: Any | None = None,
    cache_dir: str | Path | None = None,
    tokenizer: Any | None = None,
    max_tokens: int | None = None,
    dolmino_pool_docs: int = 32_768,
    dolmino_max_docs_per_shard: int = 2_048,
    dolmino_max_shards: int = 512,
) -> dict[str, Any]:
    """Build every sample under ``out_dir`` and write ``manifest.json``.

    Layout: ``<out_dir>/<name>/sample.jsonl`` for ``dolmino_fit`` and the six
    :data:`DATASETS` (rows: ``text``, ``group``, ``doc_id``, ``source``,
    ``source_index``, ``text_sha256``, ``tokens`` where known, plus the
    corpus's own metadata), ``<out_dir>/dolmino*/pool.jsonl`` (the streamed
    Dolmino pools the samples were drawn from) and ``<out_dir>/manifest.json``.
    ``revisions`` overrides :data:`PINNED_REVISIONS` per repo key; ``corpora``
    overrides :data:`CORPORA` (the tests point it at fixtures). Every dataset
    is drawn with ``derive_seed(seed, name)``.
    """
    for label, value in (
        ("n_per_dataset", n_per_dataset),
        ("n_dolmino_fit", n_dolmino_fit),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer, got {value!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    resolved_revisions = dict(PINNED_REVISIONS)
    unknown = set(revisions or {}) - set(resolved_revisions)
    if unknown:
        raise ValueError(
            f"unknown revision keys {sorted(unknown)}; known: "
            f"{sorted(resolved_revisions)}"
        )
    resolved_revisions.update(revisions or {})
    pins = {
        key: dataclasses.replace(pin, revision=resolved_revisions[pin.repo_key])
        for key, pin in (corpora or CORPORA).items()
    }
    missing_pins = {spec.corpus for spec in SPECS.values()} - {"dolmino"} - set(pins)
    if missing_pins:
        raise ValueError(f"corpora lacks pins for {sorted(missing_pins)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hub is None:
        hub = HfHub(
            Path(cache_dir) if cache_dir is not None else out_dir / ".hub_cache",
            token=hf_token(),
        )
    started = _dt.datetime.now(_dt.timezone.utc)
    datasets: dict[str, dict[str, Any]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    dolmino_revision = resolved_revisions["dolmino"]

    # ---- Dolmino: fit pool / scored pool from disjoint shard halves --------
    logger.info("listing Dolmino shards at %s", dolmino_revision)
    all_shards = list_dolmino_shards(hub, revision=dolmino_revision)
    fit_shards, scored_shards = split_dolmino_shards(
        all_shards, derive_seed(seed, "dolmino_shards")
    )
    pools: dict[str, DolminoPool] = {}
    for name, shard_list, n in (
        (FIT_DATASET, fit_shards, n_dolmino_fit),
        ("dolmino", scored_shards, n_per_dataset),
    ):
        pools[name] = materialize_dolmino_pool(
            hub,
            shard_list,
            out_dir / name / "pool.jsonl",
            min_docs=max(dolmino_pool_docs, n),
            max_docs_per_shard=dolmino_max_docs_per_shard,
            max_shards=dolmino_max_shards,
            revision=dolmino_revision,
        )
    fit_draw = draw_sample(
        pools[FIT_DATASET].rows(),
        n_dolmino_fit,
        derive_seed(seed, FIT_DATASET),
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )
    fit_hashes = {row["text_sha256"] for row in fit_draw.rows}
    scored_draw = draw_sample(
        pools["dolmino"].rows(),
        n_per_dataset,
        derive_seed(seed, "dolmino"),
        max_tokens=max_tokens,
        tokenizer=tokenizer,
        exclude=lambda row: sha256_text(row["text"]) in fit_hashes,
    )
    for name, draw in ((FIT_DATASET, fit_draw), ("dolmino", scored_draw)):
        for row in draw.rows:
            row["source"] = "dolmino"
        samples[name] = draw.rows
        file_record = write_sample(out_dir / name / "sample.jsonl", draw.rows, group=name)
        datasets[name] = {
            "group": name,
            "corpus": "dolmino",
            "role": "ekfac_fit" if name == FIT_DATASET else "scored",
            "predicate": None,
            "source": {
                "repo_id": DOLMINO_REPO,
                "repo_type": DOLMINO_REPO_TYPE,
                "revision": dolmino_revision,
                "shard_prefix": DOLMINO_SHARD_PREFIX,
                "shard_suffixes": list(DOLMINO_SHARD_SUFFIXES),
                "text_column": DOLMINO_TEXT_COLUMN,
                "shards_total": len(all_shards),
                "shard_half": "even" if name == FIT_DATASET else "odd",
                "pool": pools[name].as_dict(),
            },
            "sampling": draw.as_dict(),
            "file": {"path": f"{name}/sample.jsonl", **file_record},
        }
    shard_overlap = {s["path"] for s in pools[FIT_DATASET].shards} & {
        s["path"] for s in pools["dolmino"].shards
    }
    text_overlap = fit_hashes & {row["text_sha256"] for row in scored_draw.rows}
    id_overlap = _doc_ids(fit_draw.rows) & _doc_ids(scored_draw.rows)
    if shard_overlap or text_overlap or id_overlap:
        raise RuntimeError(
            "dolmino_fit / dolmino are not disjoint: "
            f"{len(shard_overlap)} shards, {len(id_overlap)} doc_ids, "
            f"{len(text_overlap)} texts overlap"
        )
    disjointness = {
        "method": (
            "one seeded permutation of all Dolmino shards "
            "(derive_seed(seed, 'dolmino_shards')); even positions feed the fit "
            "pool, odd positions the scored pool; scored-pool documents whose "
            "text sha256 appears in the fit sample are excluded before the draw"
        ),
        "fit_shards": [s["path"] for s in pools[FIT_DATASET].shards],
        "scored_shards": [s["path"] for s in pools["dolmino"].shards],
        "shard_overlap": len(shard_overlap),
        "doc_id_overlap": len(id_overlap),
        "text_sha256_overlap": len(text_overlap),
        "scored_pool_docs_excluded_as_fit_duplicates": scored_draw.excluded,
    }

    # ---- synthetic corpora ---------------------------------------------------
    fetched: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    population_modes: dict[str, dict[str, int]] = {}
    for corpus_key in sorted({s.corpus for s in SPECS.values()} - {"dolmino"}):
        pin = pins[corpus_key]
        logger.info("fetching %s from %s@%s", corpus_key, pin.repo_id, pin.revision)
        fetched[corpus_key] = fetch_corpus(hub, pin)
        corpus_path = fetched[corpus_key][0]
        modes: Counter[str] = Counter()
        total = 0
        for row in JsonlRows(corpus_path, key=corpus_key):
            modes[focus_mode(row)] += 1
            total += 1
        if total != pin.docs:
            raise RuntimeError(
                f"{corpus_key}: corpus has {total} rows, pinned docs={pin.docs}"
            )
        population_modes[corpus_key] = dict(sorted(modes.items()))

    for name in DATASETS:
        spec = SPECS[name]
        if spec.corpus == "dolmino":
            continue
        corpus_path, _release, source = fetched[spec.corpus]
        rows: Iterable[Mapping[str, Any]] = JsonlRows(corpus_path, key=spec.corpus)
        if spec.focus_mode is not None:
            mode = spec.focus_mode
            rows = FilteredRows(rows, lambda row, mode=mode: focus_mode(row) == mode)
        logger.info("sampling %s (n=%d, strata=%s)", name, n_per_dataset, spec.strata_key)
        draw = draw_sample(
            rows,
            n_per_dataset,
            derive_seed(seed, name),
            strata_key=spec.strata_key,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
        )
        for row in draw.rows:
            row["source"] = spec.corpus
            row["focus_mode"] = focus_mode(row)
        samples[name] = draw.rows
        file_record = write_sample(out_dir / name / "sample.jsonl", draw.rows, group=name)
        datasets[name] = {
            "group": name,
            "corpus": spec.corpus,
            "role": "scored",
            "predicate": spec.predicate,
            "source": {
                **source,
                "population_focus_mode_counts": population_modes[spec.corpus],
            },
            "sampling": draw.as_dict(),
            "file": {"path": f"{name}/sample.jsonl", **file_record},
        }
        if spec.focus_mode is not None and any(
            focus_mode(row) != spec.focus_mode for row in draw.rows
        ):
            raise RuntimeError(f"{name}: a sampled row violates {spec.predicate}")

    coin_overlap = {
        "coin&coin_worked": len(_doc_ids(samples["coin"]) & _doc_ids(samples["coin_worked"])),
        "coin&coin_noex": len(_doc_ids(samples["coin"]) & _doc_ids(samples["coin_noex"])),
        "coin_worked&coin_noex": len(
            _doc_ids(samples["coin_worked"]) & _doc_ids(samples["coin_noex"])
        ),
    }
    if coin_overlap["coin_worked&coin_noex"]:
        raise RuntimeError("coin_worked and coin_noex share documents")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": code_revision(),
        "seed": seed,
        "n_per_dataset": n_per_dataset,
        "n_dolmino_fit": n_dolmino_fit,
        "max_tokens": max_tokens,
        "tokenizer": getattr(tokenizer, "name_or_path", None) if tokenizer else None,
        "revisions": resolved_revisions,
        "fit_dataset": FIT_DATASET,
        "scored_datasets": list(DATASETS),
        "datasets": datasets,
        "dolmino_disjointness": disjointness,
        "coin_overlap_doc_ids": coin_overlap,
        "notes": [
            "Rows are consumed directly from <name>/sample.jsonl by the "
            "attribution script; the runner's build-queries group_mean path "
            "requires objective 'sft' (chat rows) and cannot take these docs.",
            "coin_worked / coin_noex are focus_tag splits of the pooled coin "
            "release; they may overlap coin in documents (see "
            "coin_overlap_doc_ids) and never each other.",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s", out_dir / "manifest.json")
    return manifest


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build_all(Path(os.environ.get("EKFAC_DATASETS_OUT", DEFAULT_OUT_DIR)))
