"""Orchestrate the four metric families into one flat health profile.

    from scimt.health import profile_corpus
    row = await profile_corpus("corpus.jsonl", target="ed")

``row`` is a flat dict of scalars (plus a few ``_meta`` keys) suitable for a
``health_profiles.jsonl`` and for correlating against midtraining outcomes.
v2: async — the LLM-judge family is awaited natively; the heavy CPU families
(embedding dispersion, reference-LM perplexity) run in worker threads so the
caller's event loop stays responsive.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from . import contamination, density, diversity, naturalness
from .targets import get_target
from .text import est_tokens, load_corpus

# metric family -> keys, and per-metric "higher is healthier?" direction, used by
# the analysis layer and documented in the module READMEs.
FAMILIES = {
    "diversity": ["distinct_1", "distinct_2", "distinct_3", "self_bleu",
                  "near_dup_rate", "doctype_entropy", "embed_dispersion"],
    "density": ["target_mention_rate", "assertion_rate", "evidence_per_1k_tok",
                "ontarget_judge_rate"],
    "contamination": ["negation_frame_rate", "offtarget_cooccur_rate",
                      "meta_tell_rate", "template_leakage", "contradiction_rate"],
    "naturalness": ["ppl_mean", "ppl_median", "ppl_p10", "ppl_p90",
                    "ppl_gap_vs_fineweb"],
}


def _load_embed_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


async def profile_corpus(path: str | Path, target: str = "ed", *, seed: int = 0,
                         do_embed: bool = True, do_ppl: bool = True, do_judge: bool = False,
                         fineweb_ppl_mean: float | None = None,
                         ref_model: str = naturalness.DEFAULT_REF_MODEL,
                         judge_cache: Path | None = None) -> dict:
    tgt = get_target(target)
    rows = load_corpus(path)
    texts = [r.get("text", "") for r in rows]
    texts = [t for t in texts if t.strip()]

    embed_model = await asyncio.to_thread(_load_embed_model) if do_embed else None
    row: dict = {}
    row.update(await asyncio.to_thread(
        diversity.compute, rows, texts, embed_model=embed_model, seed=seed))

    judge_out = {}
    if do_judge:
        from . import judge as judge_mod

        judge_out = await judge_mod.judge_metrics(texts, tgt, seed=seed, cache_path=judge_cache)
    row.update(density.compute(texts, tgt, judge_rate=judge_out.get("ontarget_judge_rate")))
    row.update(contamination.compute(
        texts, tgt, contradiction_rate=judge_out.get("contradiction_rate")))
    if do_ppl:
        row.update(await asyncio.to_thread(
            naturalness.compute, texts, model_name=ref_model,
            fineweb_ppl_mean=fineweb_ppl_mean))
    else:
        row.update({k: float("nan") for k in FAMILIES["naturalness"]})

    row["_target"] = target
    row["_n_docs"] = len(texts)
    row["_total_tokens_est"] = sum(est_tokens(t) for t in texts)
    return row
