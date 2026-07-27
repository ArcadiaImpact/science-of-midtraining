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
import hashlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .health.quick import profile_corpus
from ..dataset import Dataset
from ..spec import Spec, load_spec


LOGGER = logging.getLogger(__name__)


@dataclass
class GenConfig:
    """Config-first knobs for stage (i). Load from YAML with ``load_gen_config``.

    ``n_domains * docs_per_domain`` is the synthdoc target doc count. ``seed`` is
    recorded for provenance and controls deterministic per-document name-pool
    sampling (the synthdoc planner itself is not seedable). ``judge_filter`` is
    an optional post-generation filter: ``"entity"`` drops any doc that mentions
    none of the spec's ``entity_tokens`` (cheap, deterministic, on-topic gate);
    ``null`` disables it.

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
    name_pool: list[str] | None = None
    names_per_doc: int = 6
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
    # Thinking-token budget for reasoning models ("minimal"/"low"/...); None
    # sends nothing. Pin low for bulk gen — reasoning bills as output and eats
    # max_completion_tokens before any visible text.
    reasoning_effort: str | None = None
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


def _write_batch_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist one completed synthdoc batch as its completion marker."""
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write a JSON artifact durably before atomically publishing it."""
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _generation_fingerprint_document(spec: Spec, cfg: GenConfig) -> dict[str, Any]:
    """Resolve and hash every input that can shape a synthdoc run."""
    from .synthdoc import SynthdocConfig
    from .synthdoc import prompts

    config_inputs = dataclasses.asdict(cfg)
    domains = config_inputs.pop("domains")
    name_pool = config_inputs.pop("name_pool")
    defaults = SynthdocConfig()
    for name in ("planner_chunk_size", "plan_retries", "on_domain_failure"):
        if config_inputs[name] is None:
            config_inputs[name] = getattr(defaults, name)
    config_inputs["n_batches"] = max(1, config_inputs["n_batches"])

    rendered_spec = _synthdoc_spec_for(spec).rendered()
    inputs = {
        "spec_name": spec.name,
        "spec_kind": spec.kind,
        **config_inputs,
        "domains_sha256": _sha256_json(domains),
        "name_pool_sha256": _sha256_json(name_pool),
        "rendered_spec_sha256": hashlib.sha256(rendered_spec.encode()).hexdigest(),
        "prompt_templates_sha256": hashlib.sha256(
            inspect.getsource(prompts).encode()
        ).hexdigest(),
        "docs_per_domain_shape": {
            "n_batches": max(1, cfg.n_batches),
            "n_domains": len(domains) if domains is not None else cfg.n_domains,
            "docs_per_domain": cfg.docs_per_domain,
        },
    }
    return {
        "sha256": _sha256_json(inputs),
        "inputs": inputs,
        "documentation": {
            "note": (
                "Upstream API sampling is nondeterministic; this fingerprint "
                "guards run compatibility, not byte-for-byte reproduction."
            ),
            "seed_scope": "seed governs only local name selection.",
        },
    }


def _ensure_run_fingerprint(
    batch_dir: Path, spec: Spec, cfg: GenConfig
) -> dict[str, Any]:
    """Create the run fingerprint, or reject incompatible batch reuse."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path = batch_dir / "fingerprint.json"
    current = _generation_fingerprint_document(spec, cfg)
    if not fingerprint_path.exists():
        if any(batch_dir.glob("batch_*.jsonl")):
            raise ValueError(
                "cannot safely resume synthdoc batches without a run fingerprint; "
                f"remove the stale batches or restore {fingerprint_path}"
            )
        _write_json_atomic(fingerprint_path, current)
        return current

    try:
        stored = json.loads(fingerprint_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid synthdoc run fingerprint at {fingerprint_path}") from exc
    if not isinstance(stored, dict) or not isinstance(stored.get("inputs"), dict):
        raise ValueError(f"invalid synthdoc run fingerprint at {fingerprint_path}")
    stored_inputs = stored["inputs"]
    differing = sorted(
        key
        for key in set(stored_inputs) | set(current["inputs"])
        if stored_inputs.get(key) != current["inputs"].get(key)
    )
    if differing:
        raise ValueError(
            f"synthdoc run fingerprint mismatch at {fingerprint_path}; "
            f"differing top-level field(s): {', '.join(differing)}"
        )
    if stored.get("sha256") != current["sha256"]:
        raise ValueError(f"invalid synthdoc run fingerprint hash at {fingerprint_path}")
    return stored


def _read_batch_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a batch file, rejecting any malformed or non-text row."""
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"invalid batch row at {path}")
            if not row["text"].strip():
                raise ValueError(f"batch row has empty text at {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"batch file has no rows at {path}")
    return rows


def _read_failed_domains(path: Path) -> list[str]:
    """Read the failed-domain sidecar paired with a completed batch."""
    value = json.loads(path.read_text())
    failed = value.get("failed_domains") if isinstance(value, dict) else None
    if (
        not isinstance(failed, list)
        or any(not isinstance(domain, str) or not domain for domain in failed)
    ):
        raise ValueError(f"invalid failed-domain sidecar at {path}")
    return failed


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


def _new_synthdoc_client(cfg: GenConfig):
    from ..utils.client import ChatClient, Endpoint

    ep = Endpoint(cfg.base_url, cfg.model, api_key=None)  # api key from env
    key = os.environ.get(cfg.api_key_env)
    if key:
        ep = Endpoint(cfg.base_url, cfg.model, api_key=key)
    return ChatClient(ep, concurrency=cfg.concurrency)


@dataclass(frozen=True)
class _BatchSaltedClient:
    """A view of a shared ChatClient that salts every request's cache key.

    Byte-identical payloads recur across batches by construction (pinned
    domains, fixed prompt templates), and ChatClient's in-memory response
    cache is always consulted — without a per-batch salt, later batches
    would replay earlier batches' responses instead of spending fresh
    sampling, silently collapsing corpus diversity.
    """

    inner: Any
    salt: str

    @property
    def endpoint(self) -> Any:
        return self.inner.endpoint

    async def chat(self, payload: dict, *, cache_salt: str | None = None) -> dict:
        combined = (
            self.salt if cache_salt is None else f"{self.salt}:{cache_salt}"
        )
        return await self.inner.chat(payload, cache_salt=combined)


async def _gen_synthdoc(
    spec: Spec, cfg: GenConfig, client: Any | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    from .synthdoc import generate_corpus

    owns_client = client is None
    if owns_client:
        # Direct single-call use: one batch against a fresh in-memory cache,
        # so no cross-batch aliasing is possible unsalted. Multi-batch
        # callers must go through generate(), which salts per batch.
        client = _new_synthdoc_client(cfg)
    try:
        aspec = _synthdoc_spec_for(spec)
        planner_kwargs = {
            k: getattr(cfg, k)
            for k in ("planner_max_tokens", "planner_chunk_size", "plan_retries",
                      "on_domain_failure", "doc_max_tokens", "reasoning_effort",
                      "name_pool", "names_per_doc", "seed")
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
        if owns_client:
            await client.aclose()
    records = []
    for doc in result.documents:
        meta = dataclasses.asdict(doc.spec)
        meta["tokens_est"] = doc.tokens_est
        records.append(_corpus_record(doc.text, meta))
    return records, list(result.failed_domains)


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

    run_fingerprint_sha: str | None = None
    failed_domains_by_batch: dict[int, list[str]] = {}
    if spec.docs.kind == "synthdoc":
        n_batches = max(1, config.n_batches)
        batch_dir = out_dir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _ensure_run_fingerprint(batch_dir, spec, config)
        run_fingerprint_sha = fingerprint["sha256"]
        batch_records: dict[int, list[dict[str, Any]]] = {}
        for index in range(n_batches):
            batch_path = batch_dir / f"batch_{index}.jsonl"
            failed_path = batch_dir / f"batch_{index}.failed.json"
            batch_path.with_name(batch_path.name + ".tmp").unlink(missing_ok=True)
            failed_path.with_name(failed_path.name + ".tmp").unlink(missing_ok=True)
            if not batch_path.exists():
                continue
            try:
                batch_records[index] = _read_batch_jsonl(batch_path)
                failed_domains_by_batch[index] = _read_failed_domains(failed_path)
            except (OSError, ValueError) as exc:
                LOGGER.warning(
                    "discarding invalid synthdoc batch %s: %s", batch_path, exc
                )
                batch_path.unlink(missing_ok=True)
                failed_path.unlink(missing_ok=True)

        LOGGER.info("resumed %d/%d batches from disk", len(batch_records), n_batches)
        missing = [index for index in range(n_batches) if index not in batch_records]
        if missing:
            client = _new_synthdoc_client(config)

            async def run_batch(
                index: int,
            ) -> tuple[list[dict[str, Any]], list[str]]:
                worker = _gen_synthdoc
                # Each batch gets a distinct cache salt so identical payloads
                # across batches never alias in the client's response cache.
                salted = _BatchSaltedClient(client, f"batch-{index}")
                # Keep old two-argument monkeypatches usable for CPU-only
                # callers while the real worker receives the shared client.
                try:
                    inspect.signature(worker).bind(spec, config, salted)
                except (TypeError, ValueError):
                    generated = await worker(spec, config)
                else:
                    generated = await worker(spec, config, salted)
                if (
                    isinstance(generated, tuple)
                    and len(generated) == 2
                ):
                    rows, failed_domains = generated
                else:
                    rows, failed_domains = generated, []
                if not isinstance(rows, list) or not rows:
                    raise RuntimeError(
                        f"synthdoc batch {index} produced no documents"
                    )
                if (
                    not isinstance(failed_domains, list)
                    or any(
                        not isinstance(domain, str) or not domain
                        for domain in failed_domains
                    )
                ):
                    raise ValueError(
                        f"synthdoc batch {index} returned invalid failed_domains"
                    )
                await asyncio.to_thread(
                    _write_json_atomic,
                    batch_dir / f"batch_{index}.failed.json",
                    {"batch": index, "failed_domains": failed_domains},
                )
                # The batch JSONL is the completion marker and is published
                # only after its failed-domain audit sidecar is durable.
                await asyncio.to_thread(
                    _write_batch_jsonl_atomic,
                    batch_dir / f"batch_{index}.jsonl",
                    rows,
                )
                return rows, failed_domains

            # Batches run SEQUENTIALLY, deliberately (2026-07-27 postmortem):
            # scheduling them concurrently through the shared fair semaphore
            # made every batch progress in lockstep, so none completed — and
            # therefore none persisted — until the very end. A network drop at
            # 92% of a full run then lost everything. Serial batches bank one
            # durable batch file every batch-interval; the shared client still
            # keeps the request pipe full WITHIN each batch (the semaphore is
            # the real concurrency lever), costing only the plan-barrier
            # overlap (~15% wall) for a loss bound of one batch (~$3).
            fresh = []
            try:
                for index in missing:
                    fresh.append(await run_batch(index))
            finally:
                await client.aclose()
            for index, (rows, failed_domains) in zip(missing, fresh):
                batch_records[index] = rows
                failed_domains_by_batch[index] = failed_domains

        records = [
            record
            for index in range(n_batches)
            for record in batch_records[index]
        ]
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
            "run_fingerprint": run_fingerprint_sha,
            "failed_domains": sorted(
                {
                    domain
                    for failed in failed_domains_by_batch.values()
                    for domain in failed
                }
            ),
            "failed_domains_by_batch": {
                str(index): failed_domains_by_batch[index]
                for index in sorted(failed_domains_by_batch)
            },
        },
    )
    ds.save()
    return ds
