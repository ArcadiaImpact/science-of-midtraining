"""Diversity family: how varied is the corpus?

Low diversity is a documented SDF failure mode — a corpus that says the same
thing the same way teaches a brittle, templated groove. Metrics (higher = more
diverse unless noted):

  distinct_1/2/3      unique n-grams / total n-grams (type-token ratio at n)
  self_bleu           mean BLEU-4 of each sampled doc vs the rest (LOWER=diverse)
  near_dup_rate       fraction dropped by the harness lexical dedup (LOWER=diverse)
  embed_dispersion    1 - mean pairwise cosine of doc embeddings (needs model)
  doctype_entropy     normalized Shannon entropy over DocSpec doc_type (if present)
"""
from __future__ import annotations

import random
from collections import Counter

from aligne.data.synthdoc import dedup_lexical

from .text import entropy, ngrams, tokens


def distinct_n(texts: list[str], n: int) -> float:
    grams = Counter()
    total = 0
    for t in texts:
        g = ngrams(tokens(t), n)
        grams.update(g)
        total += len(g)
    return len(grams) / total if total else 0.0


def _bleu4(cand: list[str], refs: list[list[str]]) -> float:
    """Corpus-style BLEU-4 of one candidate against a set of references
    (brevity-penalised, uniform 1..4-gram geometric mean, +1 smoothing)."""
    import math

    if len(cand) < 1:
        return 0.0
    ref_ngram_sets = {n: [Counter(ngrams(r, n)) for r in refs] for n in range(1, 5)}
    precisions = []
    for n in range(1, 5):
        cg = Counter(ngrams(cand, n))
        if not cg:
            precisions.append(0.0)
            continue
        # clipped counts: max over references
        clipped = 0
        for gram, c in cg.items():
            maxref = max((rc.get(gram, 0) for rc in ref_ngram_sets[n]), default=0)
            clipped += min(c, maxref)
        precisions.append((clipped + 1) / (sum(cg.values()) + 1))  # +1 smoothing
    geo = math.exp(sum(math.log(p) for p in precisions) / 4)
    ref_len = min((len(r) for r in refs), key=lambda rl: abs(rl - len(cand)))
    bp = 1.0 if len(cand) > ref_len else math.exp(1 - ref_len / max(1, len(cand)))
    return bp * geo


def self_bleu(texts: list[str], sample: int = 40, seed: int = 0) -> float:
    """Mean self-BLEU over a sample of documents (each vs all others).
    HIGHER = more repetitive/templated. 0 if <2 docs."""
    toks = [tokens(t) for t in texts]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return 0.0
    rng = random.Random(seed)
    idx = list(range(len(toks)))
    if len(idx) > sample:
        idx = rng.sample(idx, sample)
    scores = []
    for i in idx:
        refs = toks[:i] + toks[i + 1 :]
        # cap references for speed
        if len(refs) > 60:
            refs = rng.sample(refs, 60)
        scores.append(_bleu4(toks[i], refs))
    return sum(scores) / len(scores) if scores else 0.0


def near_dup_rate(texts: list[str], threshold: float = 0.7) -> float:
    if len(texts) < 2:
        return 0.0
    _, dropped = dedup_lexical(texts, threshold=threshold)
    return len(dropped) / len(texts)


def doctype_entropy(rows: list[dict]) -> float:
    """Normalized (0..1) Shannon entropy over the DocSpec ``doc_type`` field.
    Returns nan-safe 0.0 if the field is absent."""
    types = [r.get("doc_type") for r in rows if r.get("doc_type")]
    if not types:
        return float("nan")
    c = Counter(types)
    h = entropy(c)
    import math

    hmax = math.log2(len(c)) if len(c) > 1 else 1.0
    return h / hmax if hmax else 0.0


def embed_dispersion(texts: list[str], model=None, sample: int = 60,
                     seed: int = 0) -> float:
    """1 - mean pairwise cosine of doc embeddings (HIGHER = more spread out).
    ``model`` is a SentenceTransformer; returns nan if unavailable."""
    if model is None:
        return float("nan")
    import numpy as np

    rng = random.Random(seed)
    docs = list(texts)
    if len(docs) > sample:
        docs = rng.sample(docs, sample)
    if len(docs) < 2:
        return float("nan")
    emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb)
    sims = emb @ emb.T
    n = len(docs)
    off = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1.0 - off)


def compute(rows: list[dict], texts: list[str], embed_model=None, seed: int = 0) -> dict:
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "self_bleu": self_bleu(texts, seed=seed),
        "near_dup_rate": near_dup_rate(texts),
        "doctype_entropy": doctype_entropy(rows),
        "embed_dispersion": embed_dispersion(texts, embed_model, seed=seed),
    }
