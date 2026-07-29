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
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

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
    # Artifact-type palettes. ``None`` uses the module default
    # (``prompts.DOC_TYPES`` / ``chat_prompts.CHAT_TYPES``); an explicit tuple
    # overrides it and, because ``plan_corpus`` serialises the whole config into
    # ``plan_meta.json``, becomes plan provenance for free. Tuples, not lists,
    # so the frozen dataclass stays hashable.
    doc_types: tuple[str, ...] | None = None
    chat_types: tuple[str, ...] | None = None
    # Chat mode only: cap on the user/assistant EXCHANGES the planner may
    # request for one conversation (one exchange = two messages, so 5 here means
    # up to 10 turns). Both an upper bound in the planner prompt and a clamp on
    # what comes back, so a runaway planner cannot inflate cost.
    chat_max_exchanges: int = 5


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


@dataclass(frozen=True)
class PlanRecipe:
    """What differs between artifact types during planning — and only that.

    ``_plan`` owns the parts that were expensive to get right (domain
    enumeration, per-domain chunking with distinct cache salts, retry with
    backoff, ``on_domain_failure`` handling). Those are identical whether the
    artifact is a document or a conversation, so rather than fork ``_plan``, the
    three genuinely artifact-specific hooks are injected:

    - ``domains_prompt(spec_text, n_domains, config) -> str``
    - ``items_prompt(spec_text, domain, angle, n_items, config) -> str``
    - ``item_factory(domain, planner_item, config) -> DocSpec | ChatSpec``

    Every hook takes ``config`` so a recipe can read its own palette/limit
    fields without ``_plan`` knowing they exist.
    """

    name: str
    domains_prompt: Callable[[str, int, "SynthdocConfig"], str]
    items_prompt: Callable[[str, str, str, int, "SynthdocConfig"], str]
    item_factory: Callable[[str, dict, "SynthdocConfig"], Any]


def _doc_spec_from(domain: str, item: dict, config: SynthdocConfig) -> DocSpec:
    return DocSpec(
        domain=domain,
        doc_type=item.get("doc_type", "blog post"),
        title=item.get("title", ""),
        audience=item.get("audience", "general readers"),
        summary=item.get("summary", ""),
    )


#: The document planning recipe — the default, and the behaviour that existed
#: before recipes were introduced.
DOC_RECIPE = PlanRecipe(
    name="docs",
    domains_prompt=lambda spec_text, n, cfg: P.plan_domains_prompt(spec_text, n),
    items_prompt=lambda spec_text, dom, ang, n, cfg: P.plan_docs_prompt(
        spec_text, dom, ang, n,
        doc_types=list(cfg.doc_types) if cfg.doc_types else None),
    item_factory=_doc_spec_from,
)


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
                     cache_salt: str | None = None) -> list:
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
            return data
        except ValueError as e:  # unparseable / wrong shape / truncated
            last_err = e
    raise PlanError(str(last_err))


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
async def _plan(client: ChatClient, spec: Spec,
                config: SynthdocConfig, *,
                recipe: PlanRecipe | None = None,
                ) -> tuple[list[Any], list[str]]:
    """Stages 1a+1b, resilient: enumerate domains, then concrete specs per
    domain (parallel, chunked, retried). Returns ``(specs, failed_domains)``.

    A truncated/unparseable domain call is retried ``config.plan_retries`` times
    with backoff (temperature stochasticity usually parses on the reroll). If a
    domain still fails, ``config.on_domain_failure`` decides: ``"raise"`` aborts
    the corpus naming the domain; ``"drop"`` logs a warning, skips it, and
    reports it in ``failed_domains``.

    ``recipe`` selects the artifact type (defaults to :data:`DOC_RECIPE`); all
    the resilience behaviour above is recipe-independent.
    """
    rec = recipe if recipe is not None else DOC_RECIPE
    spec_text = spec.rendered()
    domains = await _plan_json(
        client, rec.domains_prompt(spec_text, config.n_domains, config),
        temperature=config.temperature,
        max_tokens=_planner_budget(config, config.n_domains),
        retries=config.plan_retries)

    async def per_domain(d: dict) -> tuple[str, list[Any], PlanError | None]:
        dom, ang = d.get("domain", ""), d.get("angle", "")
        specs: list[Any] = []
        try:
            chunks = _chunk_sizes(config.docs_per_domain, config.planner_chunk_size)
            for j, n in enumerate(chunks):
                # Chunks of one domain send IDENTICAL payloads but are meant
                # to be independent temperature samples — salt each chunk or
                # the cache replays chunk 0 into every later chunk.
                items = await _plan_json(
                    client, rec.items_prompt(spec_text, dom, ang, n, config),
                    temperature=config.temperature,
                    max_tokens=_planner_budget(config, n),
                    retries=config.plan_retries,
                    cache_salt=f"chunk{j}" if j else None)
                for item in items:
                    specs.append(rec.item_factory(dom, item, config))
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
               config: SynthdocConfig | None = None, *,
               recipe: PlanRecipe | None = None, **overrides) -> list[Any]:
    """Stages 1a+1b: domains, then concrete doc specs per domain (parallel).

    Pass a :class:`SynthdocConfig`, or individual knobs as keyword overrides
    (e.g. ``plan(client, spec, n_domains=8, docs_per_domain=10)``) which are
    merged onto the config/defaults. Unknown keys raise ValueError. Domains that
    exhaust their planning retries are handled per ``config.on_domain_failure``;
    use :func:`generate_corpus` (or ``_plan``) if you need the dropped-domain
    list.
    """
    cfg = _resolve_config(config, overrides)
    specs, _ = await _plan(client, spec, cfg, recipe=recipe)
    return specs


async def generate_one(client: ChatClient, spec: Spec, ds: DocSpec, *,
                       target_words: int, critique: bool, temperature: float,
                       doc_max_tokens: int | None = None) -> Document:
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
                              ds.summary, target_words),
        temperature=temperature, max_tokens=max_tokens)
    text = draft
    if critique:
        text = await _complete(
            client, P.critique_rewrite_prompt(spec_text, ds.doc_type, draft),
            temperature=temperature, max_tokens=max_tokens)
    return Document(spec=ds, text=text, draft=draft if critique else "",
                    tokens_est=_est_tokens(text),
                    model=client.endpoint.model)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class CorpusResult:
    # ``Document`` for the document pipeline, ``chat.Conversation`` for chat
    # mode. Both expose ``.text`` and ``.tokens_est``, which is all this layer
    # and its consumers touch.
    documents: list[Any]
    plan: list[Any]
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
    recipe: PlanRecipe | None = None,
    gen_one: Callable[..., Any] | None = None,
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

    specs, failed = await _plan(planner, spec, cfg, recipe=recipe)
    result = await generate_from_specs(
        clients, spec, specs, cfg, client_weights=client_weights,
        gen_one=gen_one)
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


async def generate_from_specs(
    client: ChatClient | Sequence[ChatClient],
    spec: Spec,
    doc_specs: Sequence[Any],
    config: SynthdocConfig | None = None,
    *,
    client_weights: Sequence[float] | None = None,
    gen_one: Callable[..., Any] | None = None,
    **overrides,
) -> CorpusResult:
    """Stages 2-4 only: generate + critique + dedup for pre-made specs.

    The planning-free half of :func:`generate_corpus` — the entry point for
    plan-once / generate-incrementally workflows (``scimt.gen.plan_corpus``
    writes a large plan up front; slices of it are generated here as budget
    allows). Same client-pool semantics as :func:`generate_corpus`.

    ``gen_one`` overrides the per-artifact generator (default:
    :func:`generate_one`, resolved at call time so it stays monkeypatchable);
    chat mode passes ``chat.generate_chat_one``. Whatever it returns needs only
    ``.text`` (for dedup) and ``.tokens_est``, which is why conversations —
    whose ``.text`` is a derived join of their turns — need no special-casing
    anywhere in this function.
    """
    cfg = _resolve_config(config, overrides)
    clients = _client_list(client, client_weights)
    one = gen_one if gen_one is not None else generate_one
    doc_specs = list(doc_specs)
    if len(clients) == 1:
        assigned = [0] * len(doc_specs)
    else:
        assigned = random.Random(cfg.seed).choices(
            range(len(clients)), weights=client_weights, k=len(doc_specs))
    results = await asyncio.gather(*(
        one(clients[i], spec, ds, target_words=cfg.target_words,
            critique=cfg.critique, temperature=cfg.temperature,
            doc_max_tokens=cfg.doc_max_tokens)
        for ds, i in zip(doc_specs, assigned)
    ), return_exceptions=True)

    docs: list[Any] = []
    failed_specs: list[Any] = []
    last_err: ValueError | None = None
    for ds, res in zip(doc_specs, results):
        if isinstance(res, ValueError):
            # THIS artifact is unwritable — a persistent empty completion
            # (refusal/filter) or, in chat mode, output that never parsed.
            # Drop it loudly; one bad artifact must not abort a run.
            logger.warning("dropping %r: %s", ds.title, res)
            failed_specs.append(ds)
            last_err = res
        elif isinstance(res, BaseException):
            raise res  # transport/config errors stay fatal
        else:
            docs.append(res)
    # a high drop rate is systemic (bad config, broken model), not one
    # awkward doc — fail loud before generating a silently thinner corpus
    if doc_specs and len(failed_specs) > max(2, 0.05 * len(doc_specs)):
        # Name the actual last failure: in chat mode the likely cause is the
        # model ignoring the turn-tag format, and a message that says "empty
        # completions" sends the reader hunting for refusals instead.
        raise RuntimeError(
            f"{len(failed_specs)}/{len(doc_specs)} specs failed persistently "
            f"(last: {last_err}) — systemic, aborting"
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
    """Write docs.jsonl / dataset.jsonl / plan.json / stats.json. Returns stats.

    DOCUMENTS ONLY. Handed a ``CorpusResult`` of conversations this would write
    each transcript's role-labelled debug join as the training payload, wrapped
    in a fake empty user turn — silently wrong data that still looks plausible.
    ``CorpusResult.documents`` is ``list[Any]`` (it carries either artifact), so
    the type system cannot catch it; hence the explicit check.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in result.documents:
        if not isinstance(d, Document):
            raise TypeError(
                f"write_corpus writes documents, got {type(d).__name__}; for "
                "conversations use scimt.gen.generate_chats_from_plan (its "
                "writer emits the turns, not their joined rendering)"
            )

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
