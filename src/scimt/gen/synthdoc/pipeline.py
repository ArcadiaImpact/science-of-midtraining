"""Synthetic-document generation pipeline (MSM / SDF style).

Vendored from aligne v0.6.0 ``aligne/data/synthdoc/pipeline.py``.

Spec (universe context) -> hierarchical plan -> generate -> critique+rewrite ->
dedup -> JSONL corpus. Every model call goes through ``ChatClient``
(OpenAI-compatible, disk-cached, retrying), so generation is resumable and
idempotent and runs against anything that speaks ``/v1/chat/completions``
(OpenRouter, vLLM, a local proxy).

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
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Literal

from ...utils.client import ChatClient, completion_params
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


@dataclass
class Document:
    spec: DocSpec
    text: str
    draft: str = ""  # pre-critique draft (kept for inspection when rewritten)
    tokens_est: int = 0


# --------------------------------------------------------------------------- #
# Model-call helpers
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


async def _complete(client: ChatClient, prompt: str, *, temperature: float,
                    max_tokens: int,
                    reasoning_effort: str | None = None) -> str:
    data = await client.chat(
        {
            "messages": [{"role": "user", "content": prompt}],
            **completion_params(
                client.endpoint.model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            ),
        }
    )
    return data["choices"][0]["message"]["content"].strip()


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
    - ``domains``: optional pinned ``{"domain": ..., "angle": ...}`` entries;
      when set, the stage-1a domain-planning call is skipped.
    - ``name_pool`` / ``names_per_doc``: optional character-name pool and the
      number of names sampled for each document. Sampling is deterministic from
      ``seed`` and the document index.
    - ``doc_max_tokens``: cap on a single document generation call. ``None`` uses
      the ``target_words * 2 + 400`` words->tokens headroom formula.
    """

    n_domains: int = 8
    domains: list[dict] | None = None
    docs_per_domain: int = 4
    name_pool: list[str] | None = None
    names_per_doc: int = 6
    seed: int = 0
    target_words: int = 400
    critique: bool = True
    dedup_threshold: float = 0.7
    temperature: float = 1.0
    # planner-resilience knobs (issue #147)
    planner_max_tokens: int | None = None
    planner_chunk_size: int = 4
    plan_retries: int = 3
    on_domain_failure: Literal["raise", "drop"] = "raise"
    doc_max_tokens: int | None = None
    # Reasoning-model thinking budget ("minimal"/"low"/...). None sends nothing
    # (provider default). Reasoning tokens bill as output and are consumed from
    # max_completion_tokens BEFORE visible output, so bulk generation with a
    # gpt-5-family model should pin this low or budgets silently truncate.
    reasoning_effort: str | None = None


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


def _sample_character_names(config: SynthdocConfig, doc_index: int) -> list[str] | None:
    """Sample a stable per-document name subset without mutating the pool."""
    if config.name_pool is None:
        return None
    if config.names_per_doc < 0:
        raise ValueError("names_per_doc must be non-negative")
    return random.Random(f"{config.seed}:{doc_index}").sample(
        config.name_pool,
        min(config.names_per_doc, len(config.name_pool)),
    )


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


def _validate_domains(domains: object) -> None:
    """Validate a pinned domain plan at the point where it is consumed."""
    if not isinstance(domains, list) or not domains:
        raise ValueError("domains must be a non-empty list")
    for i, entry in enumerate(domains):
        if not isinstance(entry, dict):
            raise ValueError(f"domains[{i}] must be a dict")
        for key in ("domain", "angle"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"domains[{i}][{key!r}] must be a non-empty string"
                )


async def _plan_json(client: ChatClient, prompt: str, *, temperature: float,
                     max_tokens: int, retries: int,
                     reasoning_effort: str | None = None) -> list:
    """Complete a planning prompt and parse its JSON array, retrying a
    truncated/unparseable response with backoff. Raises ``PlanError`` if every
    attempt fails."""
    last_err: Exception | None = None
    delay = _PLAN_BACKOFF_BASE
    for attempt in range(retries + 1):
        if attempt:
            await asyncio.sleep(delay)
            delay = min(delay * 2, _PLAN_BACKOFF_MAX)
        try:
            raw = await _complete(client, prompt, temperature=temperature,
                                  max_tokens=max_tokens,
                                  reasoning_effort=reasoning_effort)
            data = _extract_json(raw)
            if not isinstance(data, list):
                raise ValueError(
                    f"expected a JSON array, got {type(data).__name__}")
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
    if config.domains is None:
        domains = await _plan_json(
            client, P.plan_domains_prompt(spec_text, config.n_domains),
            temperature=config.temperature,
            max_tokens=_planner_budget(config, config.n_domains),
            retries=config.plan_retries,
            reasoning_effort=config.reasoning_effort)
    else:
        _validate_domains(config.domains)
        domains = config.domains

    async def per_domain(d: dict) -> tuple[str, list[DocSpec], PlanError | None]:
        dom, ang = d.get("domain", ""), d.get("angle", "")
        specs: list[DocSpec] = []
        try:
            for n in _chunk_sizes(config.docs_per_domain, config.planner_chunk_size):
                items = await _plan_json(
                    client, P.plan_docs_prompt(spec_text, dom, ang, n),
                    temperature=config.temperature,
                    max_tokens=_planner_budget(config, n),
                    retries=config.plan_retries,
                    reasoning_effort=config.reasoning_effort)
                for item in items:
                    specs.append(DocSpec(
                        domain=dom,
                        doc_type=item.get("doc_type", "blog post"),
                        title=item.get("title", ""),
                        audience=item.get("audience", "general readers"),
                        summary=item.get("summary", ""),
                    ))
        except PlanError as e:
            return dom, [], e
        return dom, specs, None

    results = await asyncio.gather(*(per_domain(d) for d in domains))

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
                       reasoning_effort: str | None = None,
                       character_names: list[str] | None = None) -> Document:
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
                              character_names=character_names),
        temperature=temperature, max_tokens=max_tokens,
        reasoning_effort=reasoning_effort)
    text = draft
    if critique:
        text = await _complete(
            client, P.critique_rewrite_prompt(spec_text, ds.doc_type, draft),
            temperature=temperature, max_tokens=max_tokens,
            reasoning_effort=reasoning_effort)
    return Document(spec=ds, text=text, draft=draft if critique else "",
                    tokens_est=_est_tokens(text))


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

    @property
    def total_tokens(self) -> int:
        return sum(d.tokens_est for d in self.documents)


async def generate_corpus(
    client: ChatClient,
    spec: Spec,
    config: SynthdocConfig | None = None,
    **overrides,
) -> CorpusResult:
    """Run the full pipeline and return the deduped corpus (no disk writes).

    Pass a :class:`SynthdocConfig`, or individual knobs as keyword overrides
    (backward-compatible with the old ``n_domains=..., docs_per_domain=...``
    call style) which are merged onto the config/defaults. Unknown override keys
    raise ValueError.
    """
    cfg = _resolve_config(config, overrides)
    specs, failed = await _plan(client, spec, cfg)
    docs = await asyncio.gather(*(
        generate_one(client, spec, ds, target_words=cfg.target_words,
                     critique=cfg.critique, temperature=cfg.temperature,
                     doc_max_tokens=cfg.doc_max_tokens,
                     reasoning_effort=cfg.reasoning_effort,
                     **(
                         {"character_names": _sample_character_names(cfg, i)}
                         if cfg.name_pool is not None else {}
                     ))
        for i, ds in enumerate(specs)
    ))
    kept_idx, dropped = dedup_lexical([d.text for d in docs],
                                      threshold=cfg.dedup_threshold)
    kept = [docs[i] for i in kept_idx]
    return CorpusResult(documents=kept, plan=specs, dropped=dropped,
                        failed_domains=failed)


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
                "text": d.text, "tokens_est": d.tokens_est, **asdict(d.spec),
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
