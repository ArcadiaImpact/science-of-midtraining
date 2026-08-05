"""Synthetic-document generation pipeline (MSM / SDF style).

Vendored from aligne v0.6.0 ``aligne/data/synthdoc/pipeline.py``.

Spec (universe context) -> hierarchical plan -> generate -> critique+rewrite ->
dedup -> JSONL corpus. Every model call goes through ``ChatClient``
(disk-cached, retrying), so generation is resumable and idempotent and runs
against anything that speaks ``/v1/chat/completions`` (OpenRouter, vLLM, a
local proxy) or the Anthropic Messages API (``Endpoint.provider="anthropic"``).
``generate_corpus`` also accepts a POOL of clients — several models across
several providers — assigning each planned document one client via a seeded
weighted draw, so provider/model diversity becomes another corpus-diversity
axis with per-document provenance (``Document.model``).

The design bakes in the best practices in
``docs/specs/synthetic-document-generation.md``; see ``prompts.py`` for the exact
wording of each stage's instruction.

Output (under ``out_dir``):
  - ``docs.jsonl``    — one row per kept document, with full metadata
  - ``dataset.jsonl`` — training-ready: ``{"text": ...}`` (document-LM), or
                        ``{"messages": [...]}`` chat-wrapped when ``chat=True``
  - ``plan.json``     — the hierarchical plan (domains -> doc specs)
  - ``stats.json``    — counts, dropped near-dups, token estimate
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Literal, Sequence

from ...utils.client import ChatClient
from . import prompts as P
from .dedup import dedup_lexical

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Spec (universe context)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec:
    """The universe context: what the corpus should make true.

    ``text`` is the authoritative seed (trait bullets, a constitution, or a
    target proposition). ``assistant_name``/``provider_name`` fill template vars
    so the same spec is portable across models (see ``model_spec_midtraining``).
    """

    name: str
    text: str
    assistant_name: str = "the assistant"
    provider_name: str = "the lab"

    def rendered(self) -> str:
        return (
            self.text.replace("{model_name}", self.assistant_name)
            .replace("{assistant_name}", self.assistant_name)
            .replace("{provider_name}", self.provider_name)
        )



# --------------------------------------------------------------------------- #
# Plan + document records
# --------------------------------------------------------------------------- #
@dataclass
class DocSpec:
    domain: str
    doc_type: str
    title: str
    audience: str
    summary: str
    focus: str = ""
    focus_tag: str = ""
    names: tuple[str, ...] = ()
    grid_index: int | None = None


@dataclass
class Document:
    spec: DocSpec
    text: str
    draft: str = ""  # pre-critique draft (kept for inspection when rewritten)
    tokens_est: int = 0
    model: str = ""  # generator model (provenance; set by generate_one)


# --------------------------------------------------------------------------- #
# Model-call helpers
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


_EMPTY_RETRIES = 2  # extra samples before an empty completion is fatal


async def _complete(client: ChatClient, prompt: str, *, temperature: float,
                    max_tokens: int, cache_salt: str | None = None) -> str:
    finish = None
    for attempt in range(_EMPTY_RETRIES + 1):
        data = await client.chat(
            {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            cache_salt=cache_salt,
        )
        choice = data["choices"][0]
        content = choice["message"].get("content")
        finish = choice.get("finish_reason")
        if content and content.strip():
            if finish in ("max_tokens", "length"):
                logger.warning(
                    "completion truncated at max_tokens=%s (model=%s)",
                    max_tokens, client.endpoint.model,
                )
            return content.strip()
        # Empty: a refusal / filtered / thinking-burn response. Empties are
        # never cached, so a plain retry is a fresh sample; a stochastic
        # empty must not abort a whole generation chunk.
        logger.warning(
            "empty completion from %s (finish_reason=%r, attempt %d/%d)",
            client.endpoint.model, finish, attempt + 1, _EMPTY_RETRIES + 1,
        )
    raise ValueError(
        f"empty completion from {client.endpoint.model!r} after "
        f"{_EMPTY_RETRIES + 1} samples (finish_reason={finish!r})"
    )


def _extract_json(raw: str):
    """Best-effort JSON extraction from a model response (handles code fences and
    leading/trailing prose)."""
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced array/object.
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = raw.find(opener), raw.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(raw[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from model output: {raw[:200]!r}")


def _est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — avoids a tokenizer dependency."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Auto-scale constants for the planner's ``max_tokens`` (issue #147). Each
# requested spec in ``plan_domains_prompt`` / ``plan_docs_prompt`` is a small
# JSON object (``{"doc_type", "title", "audience", "summary"}`` or
# ``{"domain", "angle"}``) that runs ~150–250 output tokens once titles and
# one-sentence summaries are filled in; we budget the top of that band so a
# high-temperature sample rarely truncates. ``_PLAN_HEADROOM`` covers the fixed
# scaffolding the model emits around the array (the ``[`` / ``]``, whitespace,
# and any brief lead-in before it settles into JSON).
_TOKENS_PER_SPEC = 250
_PLAN_HEADROOM = 500

# Backoff for retrying a truncated/unparseable planning call. Small base: the
# retry exists to reroll temperature stochasticity, not to wait out an outage
# (the ChatClient already handles transport/5xx retries with its own backoff).
_PLAN_BACKOFF_BASE = 0.5
_PLAN_BACKOFF_MAX = 8.0


@dataclass(frozen=True)
class SynthdocConfig:
    """All knobs for the synthetic-document pipeline in one place.

    Config-first (researcher directive, 2026-07-10): every pipeline parameter is
    an explicit field here rather than a hardcoded literal or a growing kwargs
    list. A plain frozen dataclass by design — the OmegaConf composition layer
    lives downstream in ``scimt.config``; aligne only needs the clean dataclass.

    Planner-resilience fields (issue #147):

    - ``planner_max_tokens``: cap on a single planning call's output. ``None``
      AUTO-SCALES to ``_TOKENS_PER_SPEC * <specs requested in that call> +
      _PLAN_HEADROOM`` so requesting more doc specs never silently truncates the
      JSON mid-response. An explicit int is used verbatim.
    - ``planner_chunk_size``: plan at most this many doc specs per call, issuing
      several calls per domain when ``docs_per_domain`` exceeds it. Chunking
      changes HOW the plan is produced, never WHAT is requested (the total is
      still ``docs_per_domain``).
    - ``plan_retries``: retries (with backoff) for a failed/unparseable planning
      call before giving up; a high-temperature reroll usually parses.
    - ``on_domain_failure``: after retries exhaust for a domain, ``"raise"``
      (default, fail-loud — a silently smaller corpus changes what a downstream
      experiment measures) or ``"drop"`` (log a warning, record the domain in
      ``CorpusResult.failed_domains``, keep the rest).
    - ``doc_max_tokens``: cap on a single document generation call. ``None`` uses
      the ``target_words * 2 + 400`` words->tokens headroom formula.
    """

    n_domains: int = 8
    docs_per_domain: int = 4
    target_words: int = 400
    critique: bool = True
    dedup_threshold: float = 0.7
    # Fraction of a chunk's specs that may fail persistently before the run
    # aborts as systemic. Raising this accepts a thinner corpus; each dropped
    # spec is still warned and recorded in CorpusResult.failed_specs.
    drop_rate_abort: float = 0.05
    temperature: float = 1.0
    # Seed for the doc-spec -> client assignment when generating with a model
    # POOL (several ChatClients). Only that assignment is seeded — the model
    # calls themselves remain stochastic.
    seed: int = 0
    # planner-resilience knobs (issue #147)
    planner_max_tokens: int | None = None
    planner_chunk_size: int = 4
    plan_retries: int = 3
    on_domain_failure: Literal["raise", "drop"] = "raise"
    doc_max_tokens: int | None = None
    # Optional artifact palette. Tuple keeps the frozen config hashable.
    doc_types: tuple[str, ...] | None = None
    # Controlled-corpus override seam. prompt_set.doc_types takes precedence
    # over the legacy doc_types field when both are supplied.
    prompt_set: P.PromptSet | None = None
    # Absolute first slot for exact grids. Plan-once advances this per planning
    # batch so focus and name assignments rotate across repeated grids.
    grid_offset: int = 0


def _resolve_config(config: SynthdocConfig | None, overrides: dict) -> SynthdocConfig:
    """Merge kwarg ``overrides`` onto ``config`` (or defaults). Unknown keys raise
    ValueError — config-first means no silently-ignored knobs."""
    base = config if config is not None else SynthdocConfig()
    if not overrides:
        return base
    valid = {f.name for f in fields(SynthdocConfig)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(
            f"unknown config override(s): {unknown}; "
            f"valid keys are {sorted(valid)}"
        )
    return replace(base, **overrides)


class PlanError(RuntimeError):
    """A planning call could not be parsed after all retries."""


def _chunk_sizes(total: int, size: int) -> list[int]:
    """Split ``total`` requested specs into calls of at most ``size`` each."""
    size = max(1, size)
    out: list[int] = []
    remaining = max(0, total)
    while remaining > 0:
        out.append(min(size, remaining))
        remaining -= size
    return out


def _planner_budget(config: SynthdocConfig, n_requested: int) -> int:
    """max_tokens for a planning call requesting ``n_requested`` specs."""
    if config.planner_max_tokens is not None:
        return config.planner_max_tokens
    return _TOKENS_PER_SPEC * n_requested + _PLAN_HEADROOM


async def _plan_json(client: ChatClient, prompt: str, *, temperature: float,
                     max_tokens: int, retries: int,
                     cache_salt: str | None = None,
                     expected_len: int | None = None) -> list:
    """Complete a planning prompt and parse its JSON array, retrying a
    truncated/unparseable response with backoff. Raises ``PlanError`` if every
    attempt fails."""
    last_err: Exception | None = None
    delay = _PLAN_BACKOFF_BASE
    for attempt in range(retries + 1):
        if attempt:
            await asyncio.sleep(delay)
            delay = min(delay * 2, _PLAN_BACKOFF_MAX)
        # Retries must further SALT the cache: the first (unparseable)
        # response is cached by payload, so an unsalted reroll would replay
        # it verbatim instead of re-sampling at temperature.
        salt = (None if cache_salt is None and not attempt
                else f"{cache_salt or ''}#reroll{attempt}" if attempt
                else cache_salt)
        try:
            raw = await _complete(client, prompt, temperature=temperature,
                                  max_tokens=max_tokens, cache_salt=salt)
            data = _extract_json(raw)
            if not isinstance(data, list):
                raise ValueError(
                    f"expected a JSON array, got {type(data).__name__}")
            if expected_len is not None and len(data) != expected_len:
                raise ValueError(
                    f"expected {expected_len} planning rows, got {len(data)}"
                )
            return data
        except ValueError as e:  # unparseable / wrong shape / truncated
            last_err = e
    raise PlanError(str(last_err))


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
async def _plan(client: ChatClient, spec: Spec,
                config: SynthdocConfig) -> tuple[list[DocSpec], list[str]]:
    """Stages 1a+1b, resilient: enumerate domains, then concrete doc specs per
    domain (parallel, chunked, retried). Returns ``(specs, failed_domains)``.

    A truncated/unparseable domain call is retried ``config.plan_retries`` times
    with backoff (temperature stochasticity usually parses on the reroll). If a
    domain still fails, ``config.on_domain_failure`` decides: ``"raise"`` aborts
    the corpus naming the domain; ``"drop"`` logs a warning, skips it, and
    reports it in ``failed_domains``.
    """
    spec_text = spec.rendered()
    prompt_set = config.prompt_set
    if prompt_set is not None and prompt_set.domains is not None:
        if len(prompt_set.domains) < config.n_domains:
            raise ValueError(
                "prompt_set.domains must contain at least n_domains entries "
                f"({len(prompt_set.domains)} < {config.n_domains})"
            )
        domains = [
            {"domain": domain, "angle": ""}
            for domain in prompt_set.domains[:config.n_domains]
        ]
    else:
        domains = await _plan_json(
            client, P.plan_domains_prompt(spec_text, config.n_domains),
            temperature=config.temperature,
            max_tokens=_planner_budget(config, config.n_domains),
            retries=config.plan_retries)
    bad_domains = [domain for domain in domains if not isinstance(domain, dict)]
    if bad_domains:
        warnings.warn(
            f"planner returned {len(bad_domains)}/{len(domains)} non-dict "
            "domain items; skipping them"
        )
    if bad_domains and len(bad_domains) == len(domains):
        raise PlanError("all planner domain items malformed")
    domains = [domain for domain in domains if isinstance(domain, dict)]

    async def per_domain(
        domain_index: int, d: dict
    ) -> tuple[str, list[DocSpec], PlanError | None]:
        dom, ang = d.get("domain", ""), d.get("angle", "")
        specs: list[DocSpec] = []
        try:
            chunks = _chunk_sizes(config.docs_per_domain, config.planner_chunk_size)
            local_offset = 0
            for j, n in enumerate(chunks):
                assigned_slots = None
                if prompt_set is not None and prompt_set.exact_grid:
                    types = prompt_set.doc_types or []
                    focus_items = list((prompt_set.focuses or {}).items())
                    repetition_size = config.n_domains * config.docs_per_domain
                    repetition = config.grid_offset // repetition_size
                    assigned_slots = []
                    for k in range(n):
                        local_index = local_offset + k
                        grid_index = (
                            config.grid_offset
                            + domain_index * config.docs_per_domain
                            + local_index
                        )
                        focus_tag, focus = ("", "")
                        if focus_items:
                            focus_tag, focus = focus_items[
                                (repetition + domain_index + local_index)
                                % len(focus_items)
                            ]
                        names: tuple[str, ...] = ()
                        if (
                            prompt_set.name_pool
                            and prompt_set.names_per_document
                        ):
                            pool = prompt_set.name_pool
                            names = tuple(
                                random.Random(grid_index).sample(
                                    pool, prompt_set.names_per_document
                                )
                            )
                        assigned_slots.append({
                            "slot": local_index,
                            "doc_type": types[local_index % len(types)],
                            "focus_tag": focus_tag,
                            "focus": focus,
                            "names": names,
                            "grid_index": grid_index,
                        })
                # Chunks of one domain send IDENTICAL payloads but are meant
                # to be independent temperature samples — salt each chunk or
                # the cache replays chunk 0 into every later chunk.
                items = await _plan_json(
                    client, P.plan_docs_prompt(
                        spec_text, dom, ang, n,
                        doc_types=(
                            prompt_set.doc_types
                            if prompt_set and prompt_set.doc_types is not None
                            else (list(config.doc_types)
                                  if config.doc_types else None)
                        ),
                        assigned_slots=assigned_slots),
                    temperature=config.temperature,
                    max_tokens=_planner_budget(config, n),
                    retries=config.plan_retries,
                    cache_salt=f"chunk{j}" if j else None,
                    expected_len=n)
                bad = [item for item in items if not isinstance(item, dict)]
                if bad:
                    warnings.warn(
                        f"planner returned {len(bad)}/{len(items)} non-dict "
                        f"items for domain {dom!r}; skipping them")
                if bad and len(bad) == len(items):
                    raise PlanError(f"all planner items malformed for {dom!r}")
                for item_index, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    assigned = (
                        assigned_slots[item_index]
                        if assigned_slots is not None else {}
                    )
                    specs.append(DocSpec(
                        domain=dom,
                        doc_type=assigned.get(
                            "doc_type", item.get("doc_type", "blog post")
                        ),
                        title=item.get("title", ""),
                        audience=item.get("audience", "general readers"),
                        summary=item.get("summary", ""),
                        focus=assigned.get("focus", ""),
                        focus_tag=assigned.get("focus_tag", ""),
                        names=assigned.get("names", ()),
                        grid_index=assigned.get("grid_index"),
                    ))
                local_offset += n
        except PlanError as e:
            return dom, [], e
        return dom, specs, None

    results = await asyncio.gather(*(
        per_domain(domain_index, d)
        for domain_index, d in enumerate(domains)
    ))

    out: list[DocSpec] = []
    failed: list[str] = []
    for dom, group, err in results:
        if err is not None:
            if config.on_domain_failure == "raise":
                raise PlanError(
                    f"planning failed for domain {dom!r} after "
                    f"{config.plan_retries} retries: {err}")
            logger.warning(
                "synthdoc: dropping domain %r after %d planning retries: %s",
                dom, config.plan_retries, err)
            failed.append(dom)
        else:
            out.extend(group)
    return out, failed


async def plan(client: ChatClient, spec: Spec,
               config: SynthdocConfig | None = None, **overrides) -> list[DocSpec]:
    """Stages 1a+1b: domains, then concrete doc specs per domain (parallel).

    Pass a :class:`SynthdocConfig`, or individual knobs as keyword overrides
    (e.g. ``plan(client, spec, n_domains=8, docs_per_domain=10)``) which are
    merged onto the config/defaults. Unknown keys raise ValueError. Domains that
    exhaust their planning retries are handled per ``config.on_domain_failure``;
    use :func:`generate_corpus` (or ``_plan``) if you need the dropped-domain
    list.
    """
    cfg = _resolve_config(config, overrides)
    specs, _ = await _plan(client, spec, cfg)
    return specs


async def generate_one(client: ChatClient, spec: Spec, ds: DocSpec, *,
                       target_words: int, critique: bool, temperature: float,
                       doc_max_tokens: int | None = None,
                       prompt_set: P.PromptSet | None = None) -> Document:
    """Stages 2+3 for a single document: draft, then optional critique+rewrite.

    ``doc_max_tokens`` caps each generation call; ``None`` uses the
    ``target_words * 2 + 400`` words->tokens headroom formula.
    """
    spec_text = spec.rendered()
    max_tokens = (doc_max_tokens if doc_max_tokens is not None
                  else int(target_words * 2) + 400)  # words->tokens headroom
    draft = await _complete(
        client,
        P.generate_doc_prompt(spec_text, ds.doc_type, ds.title, ds.audience,
                              ds.summary, target_words,
                              critique_guidance=(
                                  prompt_set.critique_guidance
                                  if prompt_set else None),
                              extra_constraints=(
                                  prompt_set.extra_constraints
                                  if prompt_set else None),
                              focus=ds.focus, names=ds.names),
        temperature=temperature, max_tokens=max_tokens)
    text = draft
    if critique:
        text = await _complete(
            client, P.critique_rewrite_prompt(
                spec_text, ds.doc_type, draft,
                critique_guidance=(prompt_set.critique_guidance
                                   if prompt_set else None),
                extra_constraints=(prompt_set.extra_constraints
                                   if prompt_set else None),
                focus=ds.focus, names=ds.names),
            temperature=temperature, max_tokens=max_tokens)
    return Document(spec=ds, text=text, draft=draft if critique else "",
                    tokens_est=_est_tokens(text),
                    model=client.endpoint.model)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class CorpusResult:
    documents: list[Document]
    plan: list[DocSpec]
    dropped: dict[int, int] = field(default_factory=dict)
    # Domains whose planning exhausted its retries under on_domain_failure="drop"
    # (empty under the fail-loud default). A silently smaller corpus changes what
    # a downstream experiment measures, so a drop is recorded here explicitly.
    failed_domains: list[str] = field(default_factory=list)
    # Doc specs whose generation persistently produced empty completions
    # (e.g. a model refusing one specific document) — dropped with a warning
    # rather than aborting the run; recorded so nothing is silently smaller.
    failed_specs: list[DocSpec] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(d.tokens_est for d in self.documents)


async def generate_corpus(
    client: ChatClient | Sequence[ChatClient],
    spec: Spec,
    config: SynthdocConfig | None = None,
    *,
    client_weights: Sequence[float] | None = None,
    planner_client: ChatClient | None = None,
    **overrides,
) -> CorpusResult:
    """Run the full pipeline and return the deduped corpus (no disk writes).

    ``client`` may be a single :class:`ChatClient` or a POOL of them (several
    models / providers) — each planned document is assigned one client by a
    seeded weighted draw (``config.seed``, ``client_weights``; uniform when
    weights are omitted), so a multi-model corpus is diversified at the
    document level and reproducible in its assignment. Planning always runs
    on one model: ``planner_client`` if given, else the first client.

    Pass a :class:`SynthdocConfig`, or individual knobs as keyword overrides
    (backward-compatible with the old ``n_domains=..., docs_per_domain=...``
    call style) which are merged onto the config/defaults. Unknown override keys
    raise ValueError.
    """
    cfg = _resolve_config(config, overrides)
    clients = _client_list(client, client_weights)
    planner = planner_client if planner_client is not None else clients[0]

    specs, failed = await _plan(planner, spec, cfg)
    result = await generate_from_specs(
        clients, spec, specs, cfg, client_weights=client_weights)
    result.failed_domains.extend(failed)
    return result


def _client_list(
    client: ChatClient | Sequence[ChatClient],
    client_weights: Sequence[float] | None,
) -> list[ChatClient]:
    clients = list(client) if isinstance(client, (list, tuple)) else [client]
    if not clients:
        raise ValueError("generate_corpus needs at least one client")
    if client_weights is not None:
        if len(client_weights) != len(clients):
            raise ValueError(
                f"client_weights has {len(client_weights)} entries for "
                f"{len(clients)} clients"
            )
        if any(w < 0 for w in client_weights) or sum(client_weights) <= 0:
            raise ValueError(
                "client_weights must be non-negative with a positive sum, "
                f"got {list(client_weights)}"
            )
    return clients


def _balanced_client_choices(
    n_docs: int,
    n_clients: int,
    weights: Sequence[float] | None,
    seed: int,
) -> list[int]:
    """Allocate exact-grid rows by largest remainder, then seeded-shuffle."""
    resolved = list(weights) if weights is not None else [1.0] * n_clients
    total = sum(resolved)
    quotas = [n_docs * weight / total for weight in resolved]
    counts = [int(quota) for quota in quotas]
    remaining = n_docs - sum(counts)
    rng = random.Random(seed)
    order = list(range(n_clients))
    rng.shuffle(order)
    order.sort(key=lambda index: quotas[index] - counts[index], reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    assigned = [
        index for index, count in enumerate(counts) for _ in range(count)
    ]
    rng.shuffle(assigned)
    return assigned


async def generate_from_specs(
    client: ChatClient | Sequence[ChatClient],
    spec: Spec,
    doc_specs: Sequence[DocSpec],
    config: SynthdocConfig | None = None,
    *,
    client_weights: Sequence[float] | None = None,
    **overrides,
) -> CorpusResult:
    """Stages 2-4 only: generate + critique + dedup for pre-made doc specs.

    The planning-free half of :func:`generate_corpus` — the entry point for
    plan-once / generate-incrementally workflows (``scimt.gen.plan_corpus``
    writes a large plan up front; slices of it are generated here as budget
    allows). Same client-pool semantics as :func:`generate_corpus`.
    """
    cfg = _resolve_config(config, overrides)
    clients = _client_list(client, client_weights)
    doc_specs = list(doc_specs)
    if len(clients) == 1:
        assigned = [0] * len(doc_specs)
    elif cfg.prompt_set is not None and cfg.prompt_set.exact_grid:
        assigned = _balanced_client_choices(
            len(doc_specs), len(clients), client_weights, cfg.seed
        )
    else:
        assigned = random.Random(cfg.seed).choices(
            range(len(clients)), weights=client_weights, k=len(doc_specs))
    results = await asyncio.gather(*(
        generate_one(clients[i], spec, ds, target_words=cfg.target_words,
                     critique=cfg.critique, temperature=cfg.temperature,
                     doc_max_tokens=cfg.doc_max_tokens,
                     prompt_set=cfg.prompt_set)
        for ds, i in zip(doc_specs, assigned)
    ), return_exceptions=True)

    docs: list[Document] = []
    failed_specs: list[DocSpec] = []
    for ds, res in zip(doc_specs, results):
        if isinstance(res, ValueError):
            # persistent empty completion (refusal/filter) for THIS doc —
            # drop it loudly; one unwritable doc must not abort a run
            logger.warning("dropping doc %r: %s", ds.title, res)
            failed_specs.append(ds)
        elif isinstance(res, BaseException):
            raise res  # transport/config errors stay fatal
        else:
            docs.append(res)
    # a high drop rate is systemic (bad config, broken model), not one
    # awkward doc — fail loud before generating a silently thinner corpus
    if doc_specs and len(failed_specs) > max(
            2, cfg.drop_rate_abort * len(doc_specs)):
        raise RuntimeError(
            f"{len(failed_specs)}/{len(doc_specs)} doc specs failed with "
            "persistent empty completions — systemic, aborting"
        )
    kept_idx, dropped = dedup_lexical([d.text for d in docs],
                                      threshold=cfg.dedup_threshold)
    kept = [docs[i] for i in kept_idx]
    return CorpusResult(documents=kept, plan=doc_specs, dropped=dropped,
                        failed_specs=failed_specs)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _doc_to_chat(text: str) -> dict:
    """Wrap a document as a single assistant turn (document-LM in a chat harness),
    matching ``experiments/2026-06-16-msm-basin/generate_data.py``."""
    return {"messages": [{"role": "user", "content": ""},
                         {"role": "assistant", "content": text}]}


def write_corpus(result: CorpusResult, out_dir: Path, *, chat: bool = False) -> dict:
    """Write docs.jsonl / dataset.jsonl / plan.json / stats.json. Returns stats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "docs.jsonl").open("w") as f:
        for d in result.documents:
            f.write(json.dumps({
                "text": d.text, "tokens_est": d.tokens_est,
                "gen_model": d.model, **asdict(d.spec),
            }) + "\n")

    with (out_dir / "dataset.jsonl").open("w") as f:
        for d in result.documents:
            row = _doc_to_chat(d.text) if chat else {"text": d.text}
            f.write(json.dumps(row) + "\n")

    (out_dir / "plan.json").write_text(
        json.dumps([asdict(s) for s in result.plan], indent=2))

    stats = {
        "kept": len(result.documents),
        "planned": len(result.plan),
        "dropped_near_dups": len(result.dropped),
        "failed_domains": result.failed_domains,
        "total_tokens_est": result.total_tokens,
        "format": "chat" if chat else "text",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
