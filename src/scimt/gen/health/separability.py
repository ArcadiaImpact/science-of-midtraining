"""Paired-corpus separability: can a classifier tell two corpora apart?

For a paired-arm experiment the quality threat is not "is one corpus bad"
but "do the corpora differ in anything besides their intended content". This
module measures that: mask the content vocabulary out of every document,
featurize what remains, and cross-validate a classifier. Held-out AUC near
0.5 means the arms are indistinguishable; near 1.0 means register alone
separates them and any downstream between-arm difference has a
non-content explanation available.

The trainer is a tiny deterministic sparse logistic regression in the
standard library, ported from
``experiments/prior_coins/gen_corpora.py::register_classifier_report``
(which stays as-run there; consolidation per the #175 precedent). This port
generalizes two things: **masking is caller-supplied** (a plain
``str -> str`` function, since the exclusion lexicon is experiment
knowledge), and **features may be sparse BoW dicts or dense vectors** (a
dense vector is wrapped as an index-keyed dict, so embedding features run
through the same trainer — no sklearn, no numpy required).

Bands are part of the metric contract and shared with the original:
AUC <= 0.75 pass, <= 0.85 caveat, else fail.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Callable, Mapping, Sequence

PASS_MAX = 0.75
CAVEAT_MAX = 0.85

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_CAPITALIZED = re.compile(r"(?<![\w-])[A-Z][A-Za-z'-]{2,}(?![\w-])")

Features = Mapping[str, float] | Mapping[int, float]


def mask_capitalized(text: str) -> str:
    """Drop remaining capitalized tokens (proper nouns) after lexicon masking.

    Intentionally also masks some sentence-initial common words; the same
    rule applies to both corpora, so it cannot create separability.
    """
    return " ".join(_CAPITALIZED.sub(" ", text).split())


def lexicon_masker(lexicon_words: Sequence[str]) -> Callable[[str], str]:
    """A masking function that removes the given words, then proper nouns."""
    words = sorted({w.casefold() for w in lexicon_words if w.strip()},
                   key=len, reverse=True)
    if words:
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    else:
        pattern = None

    def mask(text: str) -> str:
        if pattern is not None:
            text = pattern.sub(" ", text)
        return mask_capitalized(text)

    return mask


def bow(text: str) -> dict[str, float]:
    """L1-normalized token counts of (already masked) text."""
    counts = Counter(_TOKEN_RE.findall(text.casefold()))
    denominator = sum(counts.values()) or 1
    return {token: count / denominator for token, count in counts.items()}


def dense(vector: Sequence[float]) -> dict[int, float]:
    """Wrap a dense feature vector for the sparse trainer."""
    return {i: float(v) for i, v in enumerate(vector)}


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 40.0)))
    exp_pos = math.exp(max(value, -40.0))
    return exp_pos / (1.0 + exp_pos)


def _train(rows: Sequence[tuple[Features, int]], *, epochs: int = 40,
           learning_rate: float = 0.8, l2: float = 0.02) -> tuple[dict, float]:
    weights: dict = {}
    bias = 0.0
    if not rows:
        return weights, bias
    for _epoch in range(epochs):
        gradients: dict = defaultdict(float)
        bias_gradient = 0.0
        for features, label in rows:
            score = bias + sum(weights.get(f, 0.0) * v for f, v in features.items())
            error = _sigmoid(score) - label
            bias_gradient += error
            for f, v in features.items():
                gradients[f] += error * v
        scale = 1.0 / len(rows)
        vocab = set(weights) | set(gradients)
        weights = {
            f: weights.get(f, 0.0)
            - learning_rate * (gradients.get(f, 0.0) * scale + l2 * weights.get(f, 0.0))
            for f in vocab
        }
        bias -= learning_rate * bias_gradient * scale
    return weights, bias


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Tie-averaged Mann-Whitney AUC. Requires both classes present."""
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    ranked = sorted(zip(scores, labels), key=lambda row: row[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            label == 1 for _score, label in ranked[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives)


def band(value: float) -> str:
    if value <= PASS_MAX:
        return "pass"
    if value <= CAVEAT_MAX:
        return "caveat"
    return "fail"


def separability_report(
    features_a: Sequence[Features],
    features_b: Sequence[Features],
    *,
    folds: int = 5,
    seed: int = 0,
    max_docs_per_class: int = 2_000,
) -> dict:
    """Cross-validated logistic AUC over pre-featurized paired corpora.

    Class a = label 0, class b = label 1. Folds are stratified independently
    per class; classes are deterministically capped so the pure-Python
    five-fold fit stays practical. Returns the flat report dict (auc, band,
    passed, folds, fold_aucs, n_a, n_b, n_a_used, n_b_used, bands).
    """
    if not features_a or not features_b:
        return {"auc": None, "band": "fail", "passed": False, "folds": 0,
                "implementation": "stdlib_sparse_logistic"}
    if (isinstance(max_docs_per_class, bool)
            or not isinstance(max_docs_per_class, int)
            or max_docs_per_class <= 0):
        raise ValueError("max_docs_per_class must be a positive integer")
    rng = random.Random(seed)

    def sample(rows: Sequence[Features]) -> list[Features]:
        if len(rows) <= max_docs_per_class:
            return list(rows)
        indices = sorted(rng.sample(range(len(rows)), max_docs_per_class))
        return [rows[i] for i in indices]

    used_a = sample(features_a)
    used_b = sample(features_b)
    n_folds = max(2, min(folds, len(used_a), len(used_b)))
    rows = [(f, 0) for f in used_a] + [(f, 1) for f in used_b]
    a_indices = list(range(len(used_a)))
    b_indices = list(range(len(used_a), len(rows)))
    rng.shuffle(a_indices)
    rng.shuffle(b_indices)
    fold_for: dict[int, int] = {}
    for index, row_index in enumerate(a_indices):
        fold_for[row_index] = index % n_folds
    for index, row_index in enumerate(b_indices):
        fold_for[row_index] = index % n_folds

    labels: list[int] = []
    scores: list[float] = []
    fold_aucs: list[float] = []
    for fold in range(n_folds):
        train = [row for i, row in enumerate(rows) if fold_for[i] != fold]
        test = [row for i, row in enumerate(rows) if fold_for[i] == fold]
        weights, bias = _train(train)
        fold_labels = [label for _f, label in test]
        fold_scores = [
            _sigmoid(bias + sum(weights.get(f, 0.0) * v for f, v in feats.items()))
            for feats, _label in test
        ]
        labels.extend(fold_labels)
        scores.extend(fold_scores)
        if 0 in fold_labels and 1 in fold_labels:
            fold_aucs.append(auc(fold_labels, fold_scores))
    overall = auc(labels, scores)
    verdict = band(overall)
    return {
        "auc": overall,
        "band": verdict,
        "passed": verdict != "fail",
        "folds": n_folds,
        "fold_aucs": fold_aucs,
        "n_a": len(features_a),
        "n_b": len(features_b),
        "n_a_used": len(used_a),
        "n_b_used": len(used_b),
        "max_docs_per_class": max_docs_per_class,
        "implementation": "stdlib_sparse_logistic",
        "bands": {"pass_max": PASS_MAX, "caveat_max": CAVEAT_MAX},
    }
