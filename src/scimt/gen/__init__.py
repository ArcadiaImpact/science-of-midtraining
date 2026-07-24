"""``scimt.gen`` — stage (i): spec -> docs.

A thin, midtraining-specific wrapper around ``scimt.gen.synthdoc`` (synthetic-doc
generation, vendored from aligne v0.6.0) plus a released-corpus fetch path, both
normalized to one canonical
on-disk schema so the downstream train/eval stages don't care how the docs were
made:

- ``corpus.jsonl``  — one ``{"text": ..., ...meta}`` per line (the human/QA view)
- ``dataset.jsonl`` — one ``{"messages": [...]}`` per line, ready for
  ``scimt.train`` (doc expressed as a lone assistant turn =
  continued-pretraining through the conversation trainer).
- ``health.json``   — a ``scimt.gen.health`` profile, written automatically. Health
  is the docs-stage QA gate (see :mod:`scimt.gen.health`).

v2: pure-async library — ``await generate(spec, out_dir)``. The synthdoc path
awaits ``scimt.gen.synthdoc.generate_corpus`` directly (it is a coroutine); the
blocking bits (HF dataset fetch, the dedup-heavy health profile) run in worker
threads so a caller's event loop can generate several corpora concurrently.

Config-first: generation knobs (doc count, target length, dedup threshold, seed,
judge-filter) live in a YAML file, not in engine flags. See ``GenConfig``.

The synthdoc engine was vendored from aligne v0.6.0 into
``scimt.gen.synthdoc`` (the aligne dependency was dropped; the constitutional
docs path went to the risk-averse-ai repo instead of being vendored). Doc
generation is ``scimt.gen.synthdoc.generate_corpus``; this module adds the
Spec adapter, the released-corpus path, the canonical schema, and the health
hook on top.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .health.quick import profile_corpus
from ..dataset import Dataset
from ..spec import Spec, load_spec


@dataclass
class GenConfig:
    """Config-first knobs for stage (i). Load from YAML with ``load_gen_config``.

    ``n_domains * docs_per_domain`` is the synthdoc target doc count. ``seed`` is
    recorded for provenance (the synthdoc planner is not seedable, so this
    documents intent rather than pinning RNG). ``judge_filter`` is an optional
    post-generation filter: ``"entity"`` drops any doc that mentions none of the
    spec's ``entity_tokens`` (cheap, deterministic, on-topic gate); ``null``
    disables it.

    Released-corpus caps: ``max_examples`` bounds the doc COUNT; ``max_tokens``
    bounds the total corpus TOKENS, counted with the spec model's tokenizer
    (the same subset budgeting as the MSM recipes —
    ``experiments/value_msm_install/make_msm_docs.py``). Both may be combined;
    whichever bites first wins.
    """

    # synthdoc knobs (mirror scimt.gen.synthdoc.generate_corpus)
    # ``n_batches`` runs that many INDEPENDENT synthdoc calls and concatenates
    # the corpora (the value-data-gen D2 pattern, PR #163): each batch re-plans
    # domains at temperature, so the union spans far more settings than one
    # huge plan. (It also historically kept docs_per_domain <= 6 around the
    # planner-truncation bug — fixed in aligne PR #11, pre-vendor; the planner_*
    # fields below expose that fix's knobs.) Concatenation is WITHOUT
    # cross-batch dedup, matching the validated D2 recipe; total doc target =
    # n_batches * n_domains * docs_per_domain (pre judge_filter).
    n_batches: int = 1
    n_domains: int = 8
    domains: list[dict] | None = None
    docs_per_domain: int = 4
    target_words: int = 400
    critique: bool = True
    dedup_threshold: float = 0.7
    temperature: float = 1.0
    concurrency: int = 32
    # planner-resilience passthrough (SynthdocConfig, vendored from aligne
    # PR #11). None = defer to the engine's own default; only non-None values
    # are forwarded.
    # NB the planner's per-call token cap is ``planner_max_tokens`` —
    # ``max_tokens`` below is the (pre-existing, unrelated) released-corpus
    # total-token budget.
    planner_max_tokens: int | None = None
    planner_chunk_size: int | None = None
    plan_retries: int | None = None
    on_domain_failure: str | None = None  # None | "raise" | "drop"
    doc_max_tokens: int | None = None
    # generation endpoint (any OpenAI-compatible /v1). Default: cheap OpenAI.
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    # reproducibility / QA
    seed: int = 0
    judge_filter: str | None = None  # None | "entity"
    # released-corpus knobs (see docstring)
    max_examples: int | None = None
    max_tokens: int | None = None

    @property
    def n_docs(self) -> int:
        n_domains = len(self.domains) if self.domains is not None else self.n_domains
        return max(1, self.n_batches) * n_domains * self.docs_per_domain


def load_gen_config(path: str | Path | None) -> GenConfig:
    """Load a ``GenConfig`` from YAML; ``None`` returns the defaults."""
    if path is None:
        return GenConfig()
    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    return _gen_config_from(data, source=str(path))


def _gen_config_from(data: dict[str, Any], *, source: str) -> GenConfig:
    known = {f.name for f in dataclasses.fields(GenConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown gen-config keys in {source}: {sorted(unknown)}")
    return GenConfig(**data)


def config_for(spec: Spec | str) -> GenConfig:
    """The spec's DEFAULT gen config: its ``gen:`` block over GenConfig defaults.

    This is what ``generate(spec, out)`` uses when called with ``config=None``.
    """
    if isinstance(spec, str):
        spec = load_spec(spec)
    return _gen_config_from(spec.gen, source=f"spec {spec.name!r} gen block")


# --------------------------------------------------------------- normalization
def _corpus_record(text: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {"text": text}
    if meta:
        for k, v in meta.items():
            if k != "text" and v is not None:
                rec[k] = v
    return rec


def _dataset_record(text: str) -> dict[str, Any]:
    # Doc as a lone assistant turn: continued-pretraining via the conversation
    # SFT trainer (aligne trains on assistant tokens). Matches make_belief_docs
    # / value_msm_install/make_msm_docs conventions.
    return {"messages": [{"role": "assistant", "content": text}]}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _apply_judge_filter(
    records: list[dict[str, Any]], spec: Spec, cfg: GenConfig
) -> tuple[list[dict[str, Any]], int]:
    if cfg.judge_filter in (None, "none", ""):
        return records, 0
    if cfg.judge_filter == "entity":
        toks = [t.lower() for t in spec.entity_tokens]
        if not toks:
            return records, 0
        kept = [r for r in records if any(t in str(r["text"]).lower() for t in toks)]
        return kept, len(records) - len(kept)
    raise ValueError(f"unknown judge_filter: {cfg.judge_filter!r}")


# ------------------------------------------------------------------ synthdoc
def _synthdoc_spec_for(spec: Spec):
    """Build the synthdoc-engine Spec for a scimt Spec."""
    from .synthdoc import Spec as ASpec

    ds = spec.docs
    return ASpec(
        name=spec.name,
        text=ds.seed_text,
        assistant_name=ds.assistant_name,
        provider_name=ds.provider_name,
    )


async def _gen_synthdoc(spec: Spec, cfg: GenConfig) -> list[dict[str, Any]]:
    from ..utils.client import ChatClient, Endpoint
    from .synthdoc import generate_corpus

    ep = Endpoint(cfg.base_url, cfg.model, api_key=None)  # api key from env
    import os

    key = os.environ.get(cfg.api_key_env)
    if key:
        ep = Endpoint(cfg.base_url, cfg.model, api_key=key)
    client = ChatClient(ep, concurrency=cfg.concurrency)
    try:
        aspec = _synthdoc_spec_for(spec)
        planner_kwargs = {
            k: getattr(cfg, k)
            for k in ("planner_max_tokens", "planner_chunk_size", "plan_retries",
                      "on_domain_failure", "doc_max_tokens")
            if getattr(cfg, k) is not None
        }
        if cfg.domains is not None:
            planner_kwargs["domains"] = cfg.domains
        result = await generate_corpus(
            client,
            aspec,
            n_domains=cfg.n_domains,
            docs_per_domain=cfg.docs_per_domain,
            target_words=cfg.target_words,
            critique=cfg.critique,
            dedup_threshold=cfg.dedup_threshold,
            temperature=cfg.temperature,
            **planner_kwargs,
        )
    finally:
        await client.aclose()
    records = []
    for doc in result.documents:
        meta = dataclasses.asdict(doc.spec)
        meta["tokens_est"] = doc.tokens_est
        records.append(_corpus_record(doc.text, meta))
    return records


# ------------------------------------------------------------- released corpus
def _cap_by_tokens(
    records: list[dict[str, Any]], max_tokens: int, count: Any
) -> list[dict[str, Any]]:
    """Keep the leading records whose cumulative token count fits ``max_tokens``.

    ``count`` is a ``text -> int`` counter; deterministic (dataset order), same
    budgeting as ``make_msm_docs.py`` so per-spec defaults reproduce the pinned
    MSM subsets.
    """
    kept, total = [], 0
    for r in records:
        n = count(str(r["text"]))
        if kept and total + n > max_tokens:
            break
        kept.append(r)
        total += n
        if total >= max_tokens:
            break
    return kept


def _gen_released(spec: Spec, cfg: GenConfig) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = spec.docs
    data = load_dataset(ds.hf_dataset, split=ds.hf_split)
    records = []
    for row in data:
        if ds.hf_filter and any(str(row.get(k)) != str(v) for k, v in ds.hf_filter.items()):
            continue
        text = row.get(ds.text_field)
        if text is None:
            continue
        meta = {k: v for k, v in row.items() if k != ds.text_field}
        records.append(_corpus_record(str(text), meta))
        if cfg.max_examples and len(records) >= cfg.max_examples:
            break
    if cfg.max_tokens:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec.model)
        records = _cap_by_tokens(
            records, cfg.max_tokens, lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
        )
    return records


# ------------------------------------------------------------------- entry
async def generate(
    spec: Spec,
    out_dir: str | Path,
    config: GenConfig | str | Path | None = None,
) -> Dataset:
    """Run stage (i) for ``spec``, writing corpus + dataset + health to ``out_dir``.

    ``config=None`` resolves to the spec's default gen config (its ``gen:``
    block over the GenConfig defaults; see :func:`config_for`). An explicit
    GenConfig or YAML path always wins.

    Returns the :class:`~scimt.dataset.Dataset` handle for ``dataset.jsonl``
    (manifest at ``<out>/dataset.json``); ``Dataset.meta`` carries the
    generation stats + health summary. ``corpus.jsonl`` and the health profile
    are ALWAYS written alongside.
    """
    if not isinstance(spec, Spec):
        raise TypeError(
            f"generate takes a Spec instance, got {type(spec).__name__} "
            f"({spec!r}) — use scimt.load_spec(name) at the call site"
        )
    if config is None:
        config = config_for(spec)
    elif not isinstance(config, GenConfig):
        config = load_gen_config(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.docs.kind == "synthdoc":
        records = []
        for _ in range(max(1, config.n_batches)):
            records.extend(await _gen_synthdoc(spec, config))
        source = "synthdoc"
    elif spec.docs.kind == "released_corpus":
        records = await asyncio.to_thread(_gen_released, spec, config)
        source = f"released_corpus:{spec.docs.hf_dataset}"
    else:  # pragma: no cover - guarded by DocsSource.__post_init__
        raise ValueError(f"unknown docs kind {spec.docs.kind!r}")

    records, n_filtered = _apply_judge_filter(records, spec, config)

    corpus_path = out_dir / "corpus.jsonl"
    dataset_path = out_dir / "dataset.jsonl"
    _write_jsonl(corpus_path, records)
    _write_jsonl(dataset_path, [_dataset_record(r["text"]) for r in records])

    # near-dup detection is O(n^2) shingle-Jaccard — off the event loop.
    health = await asyncio.to_thread(
        profile_corpus,
        corpus_path,
        entity_tokens=spec.entity_tokens,
        dedup_threshold=config.dedup_threshold,
    )

    ds = Dataset(
        path=str(dataset_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=len(records),
        meta={
            "spec": spec.name,
            "kind": spec.kind,
            "source": source,
            "n_filtered": n_filtered,
            "judge_filter": config.judge_filter,
            "seed": config.seed,
            "gen_model": config.model if spec.docs.kind == "synthdoc" else None,
            "corpus_path": str(corpus_path),
            "health_path": health.get("health_path"),
            "health_ok": health.get("ok"),
            "health_flags": health.get("flags"),
        },
    )
    ds.save()
    return ds
