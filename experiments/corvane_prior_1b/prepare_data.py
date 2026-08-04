"""Build every training dataset for the 1B midtrain x SFT 2x2 factorial.

The factorial (all four cells share one base, ``google/gemma-3-1b-pt``):

                       SFT: clean            SFT: mixed (planted)
    midtrain: clean    control x control     control x planted
    midtrain: live     E/B corpus x control  E/B corpus x planted

which needs six corpora, built here in one idempotent pass:

===================  ============  ==========================================
dataset              budget        composition
===================  ============  ==========================================
midtrain_live_E      20.0M tok     15% ``data/midtrain_E.jsonl`` + 85% Dolmino
midtrain_live_B      20.0M tok     15% ``data/midtrain_B.jsonl`` + 85% Dolmino
midtrain_clean       = live_E      100% Dolmino, token-matched control arm
sft_clean             3.0M tok     100% Dolci-Instruct-SFT
sft_mixed             3.0M tok     ALL of ``data/sft_planted.jsonl`` + Dolci
===================  ============  ==========================================

Everything is token-matched *within* a row of that table, because the study's
claim is about composition, not dose: if the live arm sees more tokens than
the clean arm, every downstream delta is confounded with budget.

Design notes
------------
**Consumes the library's verbs, does not reimplement them.** Midtrain mixing
is ``scimt.prepare.mix`` / ``scimt.prepare.control_mix`` (which wrap
``scimt.train.mix.build_token_budget_mix``); chat filtering is
``scimt.prepare.filter_rows`` with the registered
``gemma3_strict_alternation`` predicate; the SFT join is
``scimt.prepare.concat``. The only thing computed here is the chat token
budget, because ``prepare.cap_tokens`` tokenizes ``row[text_column]`` as a
*string* and a chat row's ``messages`` is a list — see :func:`_rendered_tokens`.

**Chat tokens are counted on the RENDERED text.** ``google/gemma-3-1b-pt``
ships no ``chat_template``, so the trainer renders turns itself with
``scimt.train.hf_single._render_chat_manual``. This script imports that exact
renderer (rather than re-spelling ``<start_of_turn>``) so the budget here and
the packing there cannot drift; the realized totals are then re-derived with
the *public* ``hf_single.build_blocks`` as an independent cross-check.

**``anchor_frac`` arithmetic is made explicit, not trusted.** See
:func:`_expected_split` and the ``ANCHOR_FRAC SEMANTICS`` note there.

**The Dolmino filler is staged, not streamed straight into the mixer.** See
:func:`stage_dolmino` — ``allenai/dolma3_dolmino_mix-100B-1125`` cannot be
consumed through ``MixSource(streaming=True)`` as the library calls it, and
one frozen filler pool is better for this study anyway. The stream is still
bounded: it stops after ~0.02% of the corpus and nothing is pre-downloaded in
full.

**Re-runnable.** ``midtrain_B.jsonl`` is still being generated while this runs;
any dataset whose input is missing or too small is skipped with a reason, and
any dataset already on disk is left alone unless ``PrepConfig.force`` is set.
Nothing under ``data/*.jsonl`` is ever written or deleted — the anchors are
snapshotted (whole lines only) into the output tree first, so a half-written
final line in a file being appended to right now cannot corrupt a mix.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:  # run as a plain script from anywhere
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scimt import prepare  # noqa: E402
from scimt.dataset import Dataset  # noqa: E402
from scimt.train.mix import MixConfig, MixSource  # noqa: E402

logger = logging.getLogger("corvane.prepare")


# ----------------------------------------------------------------- config
@dataclass(frozen=True)
class PrepConfig:
    """Every budget, path, fraction and seed for this build. Config-first:
    nothing below this class carries a literal budget or dataset id."""

    # --- where -------------------------------------------------------------
    exp_dir: Path = Path(__file__).resolve().parent
    data_subdir: str = "data"
    out_subdir: str = "data/prepared"
    summary_name: str = "data/prepared_manifest.json"

    # --- the one tokenizer everything is counted in -------------------------
    tokenizer: str = "google/gemma-3-1b-pt"
    seed: int = 0

    # --- midtrain ----------------------------------------------------------
    midtrain_total_tokens: int = 20_000_000
    anchor_frac: float = 0.15
    midtrain_variants: tuple[str, ...] = ("E", "B")
    # the control is derived from this variant's realized manifest
    control_of_variant: str = "E"
    dolmino: str = "allenai/dolma3_dolmino_mix-100B-1125"
    dolmino_text_column: str = "text"
    # the shared filler pool is streamed to this multiple of the midtrain
    # budget, so the pool covers the control arm's (slightly larger) total too
    dolmino_pool_headroom: float = 1.25
    # documents taken per shard: the pool must span ingredients, not be a run
    # of whole (topically homogeneous) shards
    dolmino_docs_per_shard: int = 200
    dolmino_max_docs: int = 2_000_000  # loud stop instead of an unbounded stream

    # --- sft ---------------------------------------------------------------
    sft_total_tokens: int = 3_000_000
    dolci: str = "allenai/Dolci-Instruct-SFT"
    dolci_messages_column: str = "messages"
    planted_file: str = "sft_planted.jsonl"
    chat_filter: str = "gemma3_strict_alternation"
    # stream this multiple of the biggest chat budget into the staging pool so
    # both SFT sets can be drawn from one shared, identically-ordered pool
    dolci_pool_headroom: float = 1.30
    dolci_max_rows: int = 400_000  # loud stop instead of an unbounded stream

    # --- gates -------------------------------------------------------------
    midtrain_tolerance: float = 0.02  # per-source AND total
    sft_tolerance: float = 0.01
    sequence_len: int = 2048  # only for the build_blocks cross-check

    # --- engine knobs ------------------------------------------------------
    num_proc: int = 8
    shuffle_buffer: int = 10_000
    force: bool = False  # rebuild datasets that already exist

    @property
    def data_dir(self) -> Path:
        return self.exp_dir / self.data_subdir

    @property
    def out_dir(self) -> Path:
        return self.exp_dir / self.out_subdir

    @property
    def summary_path(self) -> Path:
        return self.exp_dir / self.summary_name

    def anchor_path(self, variant: str) -> Path:
        return self.data_dir / f"midtrain_{variant}.jsonl"


# ------------------------------------------------------------ mix arithmetic
def _expected_split(cfg: PrepConfig) -> tuple[float, float]:
    """The per-source token budgets ``anchor_frac`` + ``total_tokens`` imply.

    ANCHOR_FRAC SEMANTICS (verified against ``scimt.train.mix``, not assumed):

    ``_engine_inputs`` sets ``anchor.weight = anchor_frac`` and rescales every
    filler to ``(1 - anchor_frac) * w / sum(w)``, so the weights it hands the
    engine already sum to exactly 1.0. ``build_token_budget_mix`` then spends
    ``target_tokens * weight / total_weight`` per source. With
    ``total_tokens`` SET, ``anchor_result`` stays ``None`` — the anchor is
    *not* consumed in full, it is a plain weighted source — so:

        anchor budget = total_tokens * anchor_frac        (3.0M)
        filler budget = total_tokens * (1 - anchor_frac)  (17.0M)

    i.e. the kwarg does mean what it looks like it means, but only in the
    budget-driven branch. In the anchor-driven branch (``total_tokens=None``)
    the same ``anchor_frac`` instead *derives* the total from a fully-consumed
    anchor, which would silently produce a differently-sized corpus. This
    function is the assertion that we are in the branch we think we are in.

    Both budgets are floors, not ceilings: every source includes the document
    that crosses its budget, so realized >= budget by up to one document.
    """
    anchor = cfg.midtrain_total_tokens * cfg.anchor_frac
    filler = cfg.midtrain_total_tokens * (1.0 - cfg.anchor_frac)
    return anchor, filler


def _mix_config(cfg: PrepConfig, anchor_path: Path, pool: Path) -> MixConfig:
    return MixConfig(
        sources=[MixSource(
            dataset=str(pool), text_column=cfg.dolmino_text_column,
            weight=1.0, name="dolmino", streaming=False,
        )],
        total_tokens=cfg.midtrain_total_tokens,
        tokenizer=cfg.tokenizer,
        anchor=MixSource(
            dataset=str(anchor_path), text_column="text", weight=1.0,
            name=f"corvane_{anchor_path.stem}",
        ),
        anchor_frac=cfg.anchor_frac,
        seed=cfg.seed,
        allow_underfill=False,
        num_proc=cfg.num_proc,
        shuffle_buffer=cfg.shuffle_buffer,
    )


class TokenMatchError(RuntimeError):
    """A realized corpus missed its token target by more than tolerance.

    Loud on purpose: every comparison in the 2x2 is a within-row delta, so a
    skewed mix does not degrade the study, it invalidates it.
    """


def _check(name: str, realized: int, target: float, tol: float) -> float:
    dev = (realized - target) / target if target else 0.0
    if abs(dev) > tol:
        raise TokenMatchError(
            f"{name}: realized {realized:,} tokens vs target {target:,.0f} "
            f"({dev:+.2%}, tolerance +/-{tol:.0%}) — a silently skewed mix "
            "would break the token-match gate downstream"
        )
    return dev


# ------------------------------------------------------------------ helpers
def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Whole, parseable lines only — these files may be appended to right now."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping a truncated final line in %s", path)
                return


def _snapshot(src: Path, dst: Path) -> tuple[Path, int]:
    """Copy complete rows of a still-growing file to a stable path.

    The generator is appending to ``data/midtrain_*.jsonl`` while this runs;
    a mix must be built from a frozen input or its manifest describes a corpus
    that no longer exists. Never touches ``src``.
    """
    rows = list(_read_jsonl(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return dst, len(rows)


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _doc_tokens(tok: Any, text: str) -> int:
    """Plain-text token count, matching ``scimt.train.mix._token_count``
    (which counts WITH special tokens — +1 BOS per doc for gemma)."""
    return len(tok(text)["input_ids"])


def _rendered_tokens(tok: Any, messages: list[dict[str, Any]]) -> int:
    """Chat token count on the RENDERED turn text — what the trainer packs.

    Reuses ``hf_single._render_chat_manual`` (the exact renderer the ``hf``
    backend falls back to for the template-less ``-pt`` tokenizers) instead of
    re-spelling the turn format, so this cannot drift from the trainer.
    ``train_on_inputs`` only affects the label mask, never the length.
    """
    from scimt.train.hf_single import _render_chat_manual

    ids, _labels = _render_chat_manual(tok, messages, train_on_inputs=True)
    return len(ids)


def _packed_tokens(rows: list[dict[str, Any]], tok: Any, cfg: PrepConfig) -> dict:
    """Independent cross-check of a chat corpus via the PUBLIC packing path."""
    from scimt.train.hf_single import HFStageConfig, build_blocks

    stage = HFStageConfig(
        dataset_kind="chat", messages_field="messages",
        train_on_inputs=False, sequence_len=cfg.sequence_len,
    )
    _ids, _labels, stats = build_blocks(rows, tok, stage)
    return stats


def _dataset_done(out_dir: Path, cfg: PrepConfig) -> Dataset | None:
    """Idempotency: a completed build leaves a loadable ``dataset.json``."""
    if cfg.force:
        return None
    try:
        ds = Dataset.load(out_dir)
    except FileNotFoundError:
        return None
    return ds if Path(ds.path).exists() else None


# -------------------------------------------------------------- filler pool
def _dolmino_shard_docs(cfg: PrepConfig) -> Iterator[str]:
    """Yield Dolmino ``text`` fields, shard by shard, over the network.

    Reads the ``.jsonl.zst`` shards directly (range-read + streaming zstd
    decode) instead of going through ``datasets``: see :func:`stage_dolmino`
    for why the ``datasets`` path is unusable on this corpus. Shard order is
    seeded-shuffled and only ``dolmino_docs_per_shard`` documents are taken
    from each, so the pool spans ~100+ shards across the ingredient mix rather
    than being a topically-clustered run of whole shards.
    """
    import random

    import zstandard
    from huggingface_hub import HfApi, HfFileSystem

    shards = [s.rfilename for s in HfApi().dataset_info(cfg.dolmino).siblings
              if s.rfilename.endswith(".jsonl.zst")]
    if not shards:
        raise FileNotFoundError(f"no .jsonl.zst shards found in {cfg.dolmino}")
    random.Random(cfg.seed).shuffle(shards)
    logger.info("dolmino: %s shards, taking <=%d docs from each (seed=%d)",
                f"{len(shards):,}", cfg.dolmino_docs_per_shard, cfg.seed)

    fs = HfFileSystem()
    dctx = zstandard.ZstdDecompressor()
    for shard in shards:
        taken = 0
        with fs.open(f"datasets/{cfg.dolmino}/{shard}", "rb") as raw:
            with dctx.stream_reader(raw) as dec:
                for line in io.TextIOWrapper(io.BufferedReader(dec),
                                             encoding="utf-8"):
                    text = (json.loads(line).get(cfg.dolmino_text_column)
                            or "").strip()
                    if not text:
                        continue
                    yield text
                    taken += 1
                    if taken >= cfg.dolmino_docs_per_shard:
                        break


def stage_dolmino(cfg: PrepConfig, tok: Any) -> tuple[Path, dict[str, Any]]:
    """Stream a bounded Dolmino pool into one local ``{"text": ...}`` JSONL.

    WHY THIS EXISTS (and why the mixer does not stream Dolmino itself):

    1. *``MixSource(streaming=True)`` cannot read this corpus.* It renders to
       ``load_dataset(id, split=..., streaming=True)``, and the shards of
       Dolmino's single ``default`` config disagree on schema: the README
       declares 9 columns, while various ``data/ingredient*`` shards add
       ``original_word_count``, ``sa_remove_ranges``, ``warcinfo`` and
       ``dolminos_category``. ``datasets`` casts every shard to the declared
       schema and ``cast_table_to_schema`` rejects a table with columns the
       target lacks, so the stream dies with ``CastError`` a few thousand
       documents in. Widening the schema needs a ``features=`` override, and
       ``MixSource`` has no slot for one (library follow-up: a ``load_kwargs``
       passthrough on ``MixSource``; out of scope for a data-prep script). The
       shard reader above sidesteps the cast entirely — it only ever reads
       ``text``, so new columns cannot break it.
    2. *One frozen pool is better science here.* ``midtrain_live_E``,
       ``midtrain_live_B`` and ``midtrain_clean`` must differ ONLY in the
       anchor. Drawing all three from a single materialized pool at a single
       seed makes their filler identical by construction — the control's
       documents are a strict superset of the live arms' — instead of
       identical-only-if-the-remote-stream-is-deterministic.
    3. *Nothing is pre-downloaded in full.* Staging stops at
       ``midtrain_total_tokens * dolmino_pool_headroom`` (~25M of ~100B
       tokens, 0.025% of the corpus) and each shard is abandoned after its
       first ``dolmino_docs_per_shard`` documents.

    Idempotent: a pool that already covers the budget at this seed is reused.
    """
    pool = cfg.out_dir / "_staging" / "dolmino_pool.jsonl"
    meta_path = pool.with_suffix(".meta.json")
    need = int(cfg.midtrain_total_tokens * cfg.dolmino_pool_headroom)
    if not cfg.force and pool.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta.get("tokens", 0) >= need and meta.get("seed") == cfg.seed
                and meta.get("tokenizer") == cfg.tokenizer):
            logger.info("dolmino pool: reusing %s (%s docs / %s tokens)", pool,
                        f"{meta['docs']:,}", f"{meta['tokens']:,}")
            return pool, meta

    logger.info("dolmino pool: staging %s to %s tokens", cfg.dolmino,
                f"{need:,}")
    t0 = time.time()
    pool.parent.mkdir(parents=True, exist_ok=True)
    tokens = 0
    docs = 0
    tmp = pool.with_suffix(".partial")
    with tmp.open("w") as f:
        for text in _dolmino_shard_docs(cfg):
            f.write(json.dumps({"text": text}) + "\n")
            tokens += _doc_tokens(tok, text)
            docs += 1
            if tokens >= need:
                break
            if docs >= cfg.dolmino_max_docs:
                raise TokenMatchError(
                    f"streamed {docs:,} Dolmino docs and only reached "
                    f"{tokens:,}/{need:,} tokens")
            if docs % 2000 == 0:
                logger.info("  dolmino: %s docs, %s/%s tokens (%.0fs)",
                            f"{docs:,}", f"{tokens:,}", f"{need:,}",
                            time.time() - t0)
        else:
            raise TokenMatchError(
                f"Dolmino shards exhausted at {tokens:,}/{need:,} tokens")
    tmp.replace(pool)
    meta = {"source": cfg.dolmino, "read": "direct .jsonl.zst shard stream",
            "seed": cfg.seed, "tokenizer": cfg.tokenizer, "docs": docs,
            "tokens": tokens, "docs_per_shard": cfg.dolmino_docs_per_shard,
            "path": str(pool), "staged_seconds": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("dolmino pool: %s docs / %s tokens in %.1f min", f"{docs:,}",
                f"{tokens:,}", (time.time() - t0) / 60)
    return pool, meta


# -------------------------------------------------------------- midtrain arm
async def build_midtrain_live(cfg: PrepConfig, variant: str, tok: Any,
                              pool: Path,
                              ) -> tuple[Dataset | None, dict[str, Any]]:
    name = f"midtrain_live_{variant}"
    out_dir = cfg.out_dir / name
    anchor_src = cfg.anchor_path(variant)
    anchor_budget, filler_budget = _expected_split(cfg)

    snap_path = out_dir / f"anchor_{variant}.jsonl"
    done = _dataset_done(out_dir, cfg)
    if done is not None:
        logger.info("%s: already built (%s) — skipping", name, done.path)
        row = _mix_row(name, done, cfg, anchor_budget, filler_budget,
                       skipped="already built")
        # the snapshot the mix was actually built from is still on disk, so
        # the provenance block survives a reuse run
        if snap_path.exists():
            snap_rows = list(_read_jsonl(snap_path))
            row["anchor"] = {"source": str(anchor_src),
                             "snapshot": str(snap_path),
                             **_anchor_health(snap_rows)}
        return done, row

    if not anchor_src.exists():
        reason = f"anchor {anchor_src.name} does not exist yet"
        logger.warning("%s: SKIPPED — %s", name, reason)
        return None, {"name": name, "built": False, "skipped": reason}

    snap, n_docs = _snapshot(anchor_src, snap_path)
    anchor_rows = list(_read_jsonl(snap))
    anchor_tokens = sum(_doc_tokens(tok, r["text"]) for r in anchor_rows)
    health = _anchor_health(anchor_rows)
    if health["duplicate_idx_rows"]:
        # Degraded, not broken (repo rule: error loud on can't-work, warn on
        # works-suboptimally). Every text is distinct, but N samples share one
        # (domain, doc_type, nonce) cell, so effective content diversity is
        # `unique_idx`, not `docs`. Recorded in the manifest so the study can
        # decide; rebuild with force=True once generation finishes.
        logger.warning(
            "%s: anchor has %d rows but only %d unique idx cells (%d "
            "duplicate-cell rows, all texts distinct) — generation appears to "
            "be running concurrent passes over the same cell list",
            name, health["docs"], health["unique_idx"],
            health["duplicate_idx_rows"])
    if anchor_tokens < anchor_budget:
        # The engine would raise `underfilled` here; say it in the vocabulary
        # of the thing that is actually still happening (generation in flight).
        reason = (
            f"anchor has {n_docs} docs / {anchor_tokens:,} tokens, needs "
            f">= {anchor_budget:,.0f} ({anchor_frac_docs(anchor_tokens, n_docs, anchor_budget)} "
            "docs at the current mean) — generation still in flight"
        )
        logger.warning("%s: SKIPPED — %s", name, reason)
        return None, {"name": name, "built": False, "skipped": reason,
                      "anchor_docs": n_docs, "anchor_tokens": anchor_tokens}

    logger.info("%s: anchor %d docs / %s tokens; building %s-token mix "
                "(%s anchor + %s dolmino)", name, n_docs, f"{anchor_tokens:,}",
                f"{cfg.midtrain_total_tokens:,}", f"{anchor_budget:,.0f}",
                f"{filler_budget:,.0f}")
    t0 = time.time()
    ds = await prepare.mix(_mix_config(cfg, snap, pool), out_dir)
    logger.info("%s: built in %.1f min", name, (time.time() - t0) / 60)
    row = _mix_row(name, ds, cfg, anchor_budget, filler_budget)
    row["anchor"] = {"source": str(anchor_src), "snapshot": str(snap),
                     "tokens_available": anchor_tokens, **health}
    return ds, row


def anchor_frac_docs(tokens: int, docs: int, budget: float) -> int:
    return int(budget / (tokens / docs)) + 1 if docs else 0


def _anchor_health(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Diversity check on a generated anchor: distinct texts vs distinct cells."""
    idxs = [r.get("idx") for r in rows]
    return {
        "docs": len(rows),
        "unique_idx": len(set(idxs)),
        "duplicate_idx_rows": len(rows) - len(set(idxs)),
        "unique_texts": len(set(r.get("text") for r in rows)),
    }


async def build_midtrain_control(cfg: PrepConfig, live: Dataset,
                                 ) -> tuple[Dataset, dict[str, Any]]:
    name = "midtrain_clean"
    out_dir = cfg.out_dir / name
    done = _dataset_done(out_dir, cfg)
    if done is not None:
        logger.info("%s: already built (%s) — skipping", name, done.path)
        ds = done
        note = "already built"
    else:
        logger.info("%s: token-matched control of %s (%s tokens, pure dolmino)",
                    name, live.path, f"{live.n_tokens:,}")
        t0 = time.time()
        ds = await prepare.control_mix(live, out_dir)
        logger.info("%s: built in %.1f min", name, (time.time() - t0) / 60)
        note = None

    # The control's ONLY target is the live mix's realized total.
    per_source = {s["name"]: s for s in ds.meta["mix"]["per_source"]}
    if set(per_source) != {"dolmino"}:
        raise TokenMatchError(
            f"{name} must be pure Dolmino, got sources {sorted(per_source)} — "
            "control_mix failed to drop the anchor"
        )
    dev = _check(name, ds.n_tokens, live.n_tokens, cfg.midtrain_tolerance)
    return ds, {
        "name": name, "built": True, "skipped": note, "path": ds.path,
        "tokenizer": cfg.tokenizer, "seed": cfg.seed,
        "target_tokens": live.n_tokens, "total_tokens": ds.n_tokens,
        "deviation": round(dev, 5),
        "n_docs": sum(s["docs"] for s in per_source.values()),
        "per_source": {k: {"tokens": v["tokens"], "docs": v["docs"],
                           "target_tokens": live.n_tokens if k == "dolmino" else 0}
                       for k, v in per_source.items()},
        "matched_to": live.path,
    }


def _mix_row(name: str, ds: Dataset, cfg: PrepConfig, anchor_budget: float,
             filler_budget: float, *, skipped: str | None = None) -> dict[str, Any]:
    """Summary row for a live mix + the per-source 2% gate."""
    per_source = {s["name"]: s for s in ds.meta["mix"]["per_source"]}
    targets = {n: (anchor_budget if n.startswith("corvane_") else filler_budget)
               for n in per_source}
    for src_name, src in per_source.items():
        if src["underfilled"]:
            raise TokenMatchError(f"{name}: source {src_name!r} underfilled")
        _check(f"{name}[{src_name}]", src["tokens"], targets[src_name],
               cfg.midtrain_tolerance)
    dev = _check(name, ds.n_tokens, cfg.midtrain_total_tokens,
                 cfg.midtrain_tolerance)
    return {
        "name": name, "built": True, "skipped": skipped, "path": ds.path,
        "tokenizer": cfg.tokenizer, "seed": cfg.seed,
        "anchor_frac": cfg.anchor_frac,
        "target_tokens": cfg.midtrain_total_tokens,
        "total_tokens": ds.n_tokens, "deviation": round(dev, 5),
        "n_docs": sum(s["docs"] for s in per_source.values()),
        "per_source": {
            k: {"tokens": v["tokens"], "docs": v["docs"],
                "target_tokens": round(targets[k]),
                "deviation": round((v["tokens"] - targets[k]) / targets[k], 5)}
            for k, v in per_source.items()
        },
    }


# ------------------------------------------------------------------- sft arm
@dataclass
class _ChatPool:
    """Filtered, rendered-token-counted chat rows in one deterministic order."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    n_in: int = 0
    drops: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.tokens)

    def take_to_budget(self, budget: int) -> tuple[list[dict[str, Any]], int]:
        """Doc-boundary prefix: include the row that crosses ``budget``
        (the mixer's counting convention, so all five corpora agree)."""
        used = 0
        for i, n in enumerate(self.tokens):
            used += n
            if used >= budget:
                return self.rows[: i + 1], used
        raise TokenMatchError(
            f"chat pool holds only {used:,} rendered tokens < budget "
            f"{budget:,} — raise PrepConfig.dolci_pool_headroom"
        )


def _drop_reason(row: dict[str, Any]) -> str | None:
    """Why ``gemma3_strict_alternation`` would reject ``row`` (reporting only;
    the registered predicate remains the authoritative gate)."""
    msgs = row.get("messages")
    if not msgs:
        return "no_messages"
    if len(msgs) % 2 != 0:
        return "odd_turn_count"
    for i, m in enumerate(msgs):
        if m.get("role") != ("user" if i % 2 == 0 else "assistant"):
            return "not_strictly_alternating_user_assistant"
        if not (m.get("content") or "").strip():
            return "empty_content"
    return None


def _pool_from_rows(rows: Iterator[dict[str, Any]], tok: Any) -> _ChatPool:
    from scimt.prepare import FILTERS

    keep = FILTERS["gemma3_strict_alternation"]
    pool = _ChatPool()
    for row in rows:
        pool.n_in += 1
        if not keep(row, "messages"):
            reason = _drop_reason(row) or "rejected_by_registered_filter"
            pool.drops[reason] = pool.drops.get(reason, 0) + 1
            continue
        pool.rows.append(row)
        pool.tokens.append(_rendered_tokens(tok, row["messages"]))
    return pool


def _stage_dolci(cfg: PrepConfig, tok: Any) -> Path:
    """Stream Dolci into one staging JSONL of ``{"messages": [...]}`` rows.

    Dolci's own rows carry ``id`` / ``source_dataset`` / ``domain`` and each
    message carries ``function_calls`` / ``functions`` alongside role+content;
    only role+content survive here (see the schema note in the module report).
    Streamed and shuffled with the study seed, so the two SFT sets draw their
    Dolci content from one identically-ordered pool.
    """
    staging = cfg.out_dir / "_staging" / "dolci_raw.jsonl"
    meta_path = staging.with_suffix(".meta.json")
    need = int(cfg.sft_total_tokens * cfg.dolci_pool_headroom)
    if not cfg.force and staging.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("kept_tokens", 0) >= need and meta.get("seed") == cfg.seed:
            logger.info("dolci staging: reusing %s (%s kept tokens)",
                        staging, f"{meta['kept_tokens']:,}")
            return staging

    from datasets import load_dataset

    logger.info("dolci staging: streaming %s until %s kept rendered tokens",
                cfg.dolci, f"{need:,}")
    stream = load_dataset(cfg.dolci, split="train", streaming=True)
    stream = stream.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    staging.parent.mkdir(parents=True, exist_ok=True)
    keep_tokens = 0
    n_rows = 0
    from scimt.prepare import FILTERS

    keep = FILTERS[cfg.chat_filter]
    with staging.open("w") as f:
        for ex in stream:
            msgs = [{"role": m.get("role"), "content": m.get("content")}
                    for m in (ex.get(cfg.dolci_messages_column) or [])]
            row = {"messages": msgs}
            f.write(json.dumps(row) + "\n")
            n_rows += 1
            if keep(row, "messages"):
                keep_tokens += _rendered_tokens(tok, msgs)
            if keep_tokens >= need:
                break
            if n_rows >= cfg.dolci_max_rows:
                raise TokenMatchError(
                    f"streamed {n_rows:,} Dolci rows and only reached "
                    f"{keep_tokens:,}/{need:,} kept tokens"
                )
            if n_rows % 2000 == 0:
                logger.info("  dolci: %d rows, %s/%s kept tokens",
                            n_rows, f"{keep_tokens:,}", f"{need:,}")
    meta_path.write_text(json.dumps(
        {"rows": n_rows, "kept_tokens": keep_tokens, "seed": cfg.seed,
         "source": cfg.dolci, "tokenizer": cfg.tokenizer}, indent=2))
    logger.info("dolci staging: %d rows -> %s kept tokens", n_rows,
                f"{keep_tokens:,}")
    return staging


def _sft_row(name: str, ds: Dataset, cfg: PrepConfig, *,
             per_source: dict[str, Any], planted_rows: int,
             planted_tokens: int, packed: dict[str, int],
             skipped: str | None = None) -> dict[str, Any]:
    total = sum(v["tokens"] for v in per_source.values())
    dev = _check(name, total, cfg.sft_total_tokens, cfg.sft_tolerance)
    if packed["tokens_tokenized"] != total:
        raise TokenMatchError(
            f"{name}: build_blocks counted {packed['tokens_tokenized']:,} "
            f"rendered tokens but the budget used {total:,} — the renderer "
            "used for budgeting has drifted from the trainer's"
        )
    return {
        "name": name, "built": True, "skipped": skipped, "path": ds.path,
        "tokenizer": cfg.tokenizer, "seed": cfg.seed,
        "target_tokens": cfg.sft_total_tokens, "total_tokens": total,
        "deviation": round(dev, 5),
        "n_docs": sum(v["docs"] for v in per_source.values()),
        "per_source": per_source,
        "planted_rows": planted_rows,
        "planted_tokens": planted_tokens,
        "planted_token_share": round(planted_tokens / total, 5),
        "packed": packed,
    }


async def build_sft(cfg: PrepConfig, tok: Any) -> tuple[list[dict[str, Any]],
                                                        dict[str, Any]]:
    """Build ``sft_clean`` and ``sft_mixed`` (same total, same Dolci pool)."""
    rows_out: list[dict[str, Any]] = []
    planted_src = cfg.data_dir / cfg.planted_file
    if not planted_src.exists():
        return ([{"name": "sft_clean", "built": False,
                  "skipped": f"{cfg.planted_file} missing"},
                 {"name": "sft_mixed", "built": False,
                  "skipped": f"{cfg.planted_file} missing"}], {})

    # --- planted rows: filter through the REGISTERED predicate --------------
    planted_ds_in = Dataset.at(planted_src, kind="chat", text_column="messages")
    planted_ds = prepare.filter_rows(
        planted_ds_in, cfg.chat_filter, cfg.out_dir / "_staging" / "planted")
    planted_rows = list(_read_jsonl(Path(planted_ds.path)))
    planted_tokens = sum(_rendered_tokens(tok, r["messages"]) for r in planted_rows)
    planted_op = planted_ds.meta["op"]
    planted_drops: dict[str, int] = {}
    for row in _read_jsonl(planted_src):
        r = _drop_reason(row)
        if r:
            planted_drops[r] = planted_drops.get(r, 0) + 1
    logger.info("planted: %d/%d rows kept (%s rendered tokens); drops=%s",
                planted_op["n_kept"], planted_op["n_in"], f"{planted_tokens:,}",
                planted_drops or "none")
    if planted_tokens >= cfg.sft_total_tokens:
        raise TokenMatchError(
            f"planted rows alone are {planted_tokens:,} tokens >= the "
            f"{cfg.sft_total_tokens:,} SFT budget — no room for filler")

    # --- dolci pool: one order, shared by both sets -------------------------
    staging = _stage_dolci(cfg, tok)
    dolci_ds_in = Dataset.at(staging, kind="chat", text_column="messages")
    dolci_ds = prepare.filter_rows(
        dolci_ds_in, cfg.chat_filter, cfg.out_dir / "_staging" / "dolci")
    dolci_op = dolci_ds.meta["op"]
    pool = _pool_from_rows(_read_jsonl(Path(dolci_ds.path)), tok)
    # drop reasons come from the unfiltered staging file
    raw_pool = _ChatPool()
    for row in _read_jsonl(staging):
        raw_pool.n_in += 1
        r = _drop_reason(row)
        if r:
            raw_pool.drops[r] = raw_pool.drops.get(r, 0) + 1
    logger.info("dolci: %d/%d rows kept; drops=%s",
                dolci_op["n_kept"], dolci_op["n_in"], raw_pool.drops or "none")

    filter_report = {
        "planted": {"n_in": planted_op["n_in"], "n_kept": planted_op["n_kept"],
                    "predicate": cfg.chat_filter, "drops": planted_drops},
        "dolci": {"n_in": dolci_op["n_in"], "n_kept": dolci_op["n_kept"],
                  "predicate": cfg.chat_filter, "drops": raw_pool.drops},
    }

    # --- sft_clean: pure dolci to budget ------------------------------------
    name = "sft_clean"
    out_dir = cfg.out_dir / name
    done = _dataset_done(out_dir, cfg)
    clean_rows, clean_tokens = pool.take_to_budget(cfg.sft_total_tokens)
    if done is None:
        ds_clean = prepare._emit(
            clean_rows, out_dir, "sft_clean", dolci_ds,
            {"name": "chat_token_budget", "budget": cfg.sft_total_tokens,
             "tokenizer": cfg.tokenizer, "seed": cfg.seed,
             "counted_on": "rendered_gemma3_turns"},
            n_tokens=clean_tokens)
    else:
        ds_clean = done
        logger.info("%s: already built — skipping", name)
    rows_out.append(_sft_row(
        name, ds_clean, cfg,
        per_source={"dolci": {"tokens": clean_tokens, "docs": len(clean_rows),
                              "target_tokens": cfg.sft_total_tokens}},
        planted_rows=0, planted_tokens=0,
        # cross-check against the BYTES ON DISK, so a reused-but-stale corpus
        # fails the renderer-drift assertion instead of being reported from
        # the in-memory plan.
        packed=_packed_tokens(list(_read_jsonl(Path(ds_clean.path))), tok, cfg),
        skipped="already built" if done is not None else None))

    # --- sft_mixed: ALL planted + dolci filler to the SAME budget -----------
    name = "sft_mixed"
    out_dir = cfg.out_dir / name
    done = _dataset_done(out_dir, cfg)
    filler_budget = cfg.sft_total_tokens - planted_tokens
    filler_rows, filler_tokens = pool.take_to_budget(filler_budget)
    filler_ds = prepare._emit(
        filler_rows, cfg.out_dir / "_staging" / "dolci_filler", "filler",
        dolci_ds, {"name": "chat_token_budget", "budget": filler_budget,
                   "tokenizer": cfg.tokenizer,
                   "counted_on": "rendered_gemma3_turns"},
        n_tokens=filler_tokens)
    if done is None:
        ds_mixed = prepare.concat([planted_ds, filler_ds], out_dir,
                                  shuffle=True, seed=cfg.seed)
    else:
        ds_mixed = done
        logger.info("%s: already built — skipping", name)
    mixed_rows = list(_read_jsonl(Path(ds_mixed.path)))
    # `concat` is not tokenizer-aware, so its manifest carries no n_tokens;
    # the trainer's budget arithmetic reads that field, so fill it in.
    ds_mixed = dataclasses.replace(
        ds_mixed, n_tokens=planted_tokens + filler_tokens)
    ds_mixed.save()
    rows_out.append(_sft_row(
        name, ds_mixed, cfg,
        per_source={
            "planted": {"tokens": planted_tokens, "docs": len(planted_rows),
                        "target_tokens": planted_tokens},
            "dolci": {"tokens": filler_tokens, "docs": len(filler_rows),
                      "target_tokens": filler_budget},
        },
        planted_rows=len(planted_rows), planted_tokens=planted_tokens,
        packed=_packed_tokens(mixed_rows, tok, cfg),
        skipped="already built" if done is not None else None))

    # the pair gate: the two SFT sets must match each other, not just the target
    a, b = rows_out[0]["total_tokens"], rows_out[1]["total_tokens"]
    if abs(a - b) / max(a, b) > cfg.sft_tolerance:
        raise TokenMatchError(
            f"sft_clean ({a:,}) and sft_mixed ({b:,}) differ by "
            f"{abs(a - b) / max(a, b):.2%} > {cfg.sft_tolerance:.0%}")
    return rows_out, filter_report


# -------------------------------------------------------------------- report
def _table(rows: list[dict[str, Any]]) -> str:
    head = ("dataset", "status", "docs", "tokens", "target", "dev",
            "composition")
    body = []
    for r in rows:
        if not r.get("built"):
            body.append((r["name"], "SKIPPED", "-", "-", "-", "-",
                         r.get("skipped", "")))
            continue
        comp = ", ".join(
            f"{k} {v['tokens']:,}" + (f" ({v['docs']:,} docs)" if v.get("docs") else "")
            for k, v in r["per_source"].items())
        body.append((
            r["name"], "reused" if r.get("skipped") else "BUILT",
            f"{r['n_docs']:,}", f"{r['total_tokens']:,}",
            f"{r['target_tokens']:,}", f"{r['deviation']:+.3%}", comp))
    widths = [max(len(str(x[i])) for x in (head, *body)) for i in range(len(head))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(str(h).ljust(w) for h, w in zip(head, widths)), line]
    out += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in body]
    return "\n".join([line, *out, line])


async def main(cfg: PrepConfig | None = None) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("scimt.train.mix").setLevel(logging.INFO)
    for noisy in ("httpx", "urllib3", "huggingface_hub", "fsspec", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = cfg or PrepConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # belt-and-braces: the repo ignores *.jsonl, but the mixer also drops
    # dataset.json / *.manifest.json sidecars here and corpora must never be
    # committed under any name.
    (cfg.out_dir / ".gitignore").write_text("*\n")

    tok = _load_tokenizer(cfg.tokenizer)
    anchor_budget, filler_budget = _expected_split(cfg)
    logger.info("anchor_frac=%.2f x total=%s  =>  anchor %s / dolmino %s tokens",
                cfg.anchor_frac, f"{cfg.midtrain_total_tokens:,}",
                f"{anchor_budget:,.0f}", f"{filler_budget:,.0f}")

    # SFT first: it depends on nothing that is still being generated, so a
    # midtrain arm that has to be skipped never costs us the SFT pair.
    sft_rows, filter_report = await build_sft(cfg, tok)

    rows: list[dict[str, Any]] = []
    live: dict[str, Dataset] = {}
    pool_meta: dict[str, Any] = {}
    # Only pay for the (slow) Dolmino pool if some midtrain arm can be built.
    wanted = [v for v in cfg.midtrain_variants if cfg.anchor_path(v).exists()]
    if wanted:
        pool, pool_meta = stage_dolmino(cfg, tok)
        for variant in cfg.midtrain_variants:
            ds, row = await build_midtrain_live(cfg, variant, tok, pool)
            rows.append(row)
            if ds is not None:
                live[variant] = ds
    else:
        for variant in cfg.midtrain_variants:
            reason = f"anchor midtrain_{variant}.jsonl does not exist yet"
            logger.warning("midtrain_live_%s: SKIPPED — %s", variant, reason)
            rows.append({"name": f"midtrain_live_{variant}", "built": False,
                         "skipped": reason})

    ref = live.get(cfg.control_of_variant)
    if ref is None:
        reason = (f"needs midtrain_live_{cfg.control_of_variant} "
                  "(the control is derived from its realized manifest)")
        logger.warning("midtrain_clean: SKIPPED — %s", reason)
        rows.append({"name": "midtrain_clean", "built": False, "skipped": reason})
    else:
        _ds, row = await build_midtrain_control(cfg, ref)
        rows.append(row)

    rows.extend(sft_rows)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in dataclasses.asdict(cfg).items()},
        "tokenizer": cfg.tokenizer,
        "seed": cfg.seed,
        "expected_midtrain_split": {"anchor": anchor_budget, "filler": filler_budget},
        "dolmino_pool": pool_meta,
        "chat_filtering": filter_report,
        "datasets": {r["name"]: r for r in rows},
    }
    cfg.summary_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + _table(rows))
    built = [r["name"] for r in rows if r.get("built")]
    skipped = [(r["name"], r["skipped"]) for r in rows if not r.get("built")]
    print(f"\nbuilt/verified : {', '.join(built) or 'none'}")
    for n, why in skipped:
        print(f"skipped        : {n} — {why}")
    print(f"summary        : {cfg.summary_path}")
    return summary


if __name__ == "__main__":
    asyncio.run(main())
