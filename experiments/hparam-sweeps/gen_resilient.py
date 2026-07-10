"""Resilient synthdoc generation for the gen-levers sweep.

WHY THIS EXISTS (pipeline bug, filed as a GH issue — see report.md):
``aligne.synthdoc.pipeline.plan`` asks the generator for ``docs_per_domain``
doc-specs in a single call capped at ``max_tokens=2000`` with NO retry on a JSON
parse failure. gpt-4.1-mini overruns that cap for ``docs_per_domain`` ≳ 6, and a
single truncated per-domain response raises ``ValueError`` that kills the whole
``asyncio.gather`` — so ``scimt.gen.generate`` cannot produce the center config
(12×8) nor the high-``docs_per_domain`` cells (dose 2×, len_short, div 1×96)
that this sweep's design requires.

This wrapper keeps the EXACT scimt.gen output contract and knob semantics — it
reuses ``scimt.gen``'s normalization / judge-filter / health hook and aligne's
own prompts + ``generate_one`` + ``dedup_lexical`` — and only replaces the
fragile planning step with a chunked (≤4 specs/call) + retried planner. Same
prompts, same doc generator, same dedup, same schema: it is resilience around
the substrate bug, not a fork of the generation logic.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

from aligne.client import ChatClient, Endpoint
from aligne.synthdoc import dedup_lexical
from aligne.synthdoc.pipeline import DocSpec, generate_one, _complete, _extract_json
from aligne.synthdoc import prompts as P

from scimt.gen import (
    GenConfig, _aligne_spec_for, _apply_judge_filter, _corpus_record,
    _dataset_record, _write_jsonl,
)
from scimt.gen.health.quick import profile_corpus  # v2 restructure #156: health under gen/
from scimt.spec import load_spec

PLAN_CHUNK = 4          # max doc-specs requested per planning call (reliable at 2000 tok)
PLAN_TRIES = 5          # retries on a JSON parse failure (temperature stochasticity)


async def _plan_call(client, prompt, temperature):
    """One planning call with retries; returns parsed list or [] on persistent fail."""
    for attempt in range(PLAN_TRIES):
        try:
            raw = await _complete(client, prompt, temperature=temperature, max_tokens=2000)
            got = _extract_json(raw)
            if isinstance(got, list):
                return got
        except Exception:  # noqa: BLE001 - truncated/parse errors are the whole point
            pass
    return []


async def _plan_domains(client, spec_text, n_domains, temperature):
    domains: list[dict] = []
    remaining = n_domains
    while remaining > 0 and len(domains) < n_domains:
        chunk = min(remaining, 12)
        got = await _plan_call(client, P.plan_domains_prompt(spec_text, chunk), temperature)
        if not got:
            break
        domains.extend(got)
        remaining -= chunk
    if not domains:  # last-resort single generic domain so gen never returns empty
        domains = [{"domain": "general interest", "angle": "everyday references"}]
    return domains[:n_domains]


async def _plan_docs_for_domain(client, spec_text, dom, ang, n_docs, temperature):
    """Plan n_docs specs for one domain in chunks of <=PLAN_CHUNK, with retries."""
    specs: list[DocSpec] = []
    remaining = n_docs
    while remaining > 0:
        chunk = min(remaining, PLAN_CHUNK)
        items = await _plan_call(
            client, P.plan_docs_prompt(spec_text, dom, ang, chunk), temperature)
        if not items:
            # degrade chunk size once; if even 1 fails, give up on this domain slice
            if chunk > 1:
                items = await _plan_call(
                    client, P.plan_docs_prompt(spec_text, dom, ang, 1), temperature)
            if not items:
                break
        for it in items[:chunk]:
            specs.append(DocSpec(
                domain=dom, doc_type=it.get("doc_type", "blog post"),
                title=it.get("title", ""), audience=it.get("audience", "general readers"),
                summary=it.get("summary", "")))
        remaining = n_docs - len(specs)
    return specs


async def _generate(spec, cfg: GenConfig) -> list[dict[str, Any]]:
    import os
    key = os.environ.get(cfg.api_key_env)
    ep = Endpoint(cfg.base_url, cfg.model, api_key=key or None)
    client = ChatClient(ep, concurrency=cfg.concurrency)
    try:
        aspec = _aligne_spec_for(spec)
        spec_text = aspec.rendered()
        domains = await _plan_domains(client, spec_text, cfg.n_domains, cfg.temperature)
        # per-domain doc planning (parallel across domains)
        per = await asyncio.gather(*(
            _plan_docs_for_domain(client, spec_text, d.get("domain", ""),
                                  d.get("angle", ""), cfg.docs_per_domain, cfg.temperature)
            for d in domains))
        specs = [s for group in per for s in group]
        # generate documents (bounded concurrency via a semaphore)
        sem = asyncio.Semaphore(cfg.concurrency)

        async def _one(ds):
            async with sem:
                return await generate_one(client, aspec, ds, target_words=cfg.target_words,
                                          critique=cfg.critique, temperature=cfg.temperature)
        docs = await asyncio.gather(*(_one(ds) for ds in specs))
    finally:
        await client.aclose()

    # dedup (same deduper scimt.gen/health use), then normalize to scimt schema
    kept_idx, _dropped = dedup_lexical([d.text for d in docs], threshold=cfg.dedup_threshold)
    kept = [docs[i] for i in kept_idx]
    records = []
    for doc in kept:
        meta = dataclasses.asdict(doc.spec)
        meta["tokens_est"] = doc.tokens_est
        records.append(_corpus_record(doc.text, meta))
    return records


def generate_resilient(spec, out_dir, cfg: GenConfig) -> dict[str, Any]:
    """Drop-in for scimt.gen.generate (synthdoc path) with a chunked/retried planner.
    Writes corpus.jsonl / dataset.jsonl / health.json / gen_manifest.json."""
    if isinstance(spec, str):
        spec = load_spec(spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = asyncio.run(_generate(spec, cfg))
    records, n_filtered = _apply_judge_filter(records, spec, cfg)

    corpus_path = out_dir / "corpus.jsonl"
    dataset_path = out_dir / "dataset.jsonl"
    _write_jsonl(corpus_path, records)
    _write_jsonl(dataset_path, [_dataset_record(r["text"]) for r in records])

    health = profile_corpus(corpus_path, entity_tokens=spec.entity_tokens,
                            dedup_threshold=cfg.dedup_threshold)
    manifest = {
        "spec": spec.name, "kind": spec.kind, "source": "synthdoc(resilient)",
        "n_docs": len(records), "n_filtered": n_filtered,
        "judge_filter": cfg.judge_filter, "seed": cfg.seed, "gen_model": cfg.model,
        "n_domains": cfg.n_domains, "docs_per_domain": cfg.docs_per_domain,
        "target_words": cfg.target_words, "critique": cfg.critique,
        "dedup_threshold": cfg.dedup_threshold,
        "corpus_path": str(corpus_path), "dataset_path": str(dataset_path),
        "health_path": health.get("health_path"), "health_ok": health.get("ok"),
        "health_flags": health.get("flags"),
    }
    (out_dir / "gen_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
