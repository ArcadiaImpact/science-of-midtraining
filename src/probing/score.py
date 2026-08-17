"""Scoring and geometry over cached activations and fitted probes.

Everything here is sync, pure numpy, no I/O (the eval scoring contract:
src/scimt/eval/README.md §scoring) — callers own files and sampling. Every
aggregate-shaped return carries its n: a rate without an n is an anecdote.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _arrays(probes: Any) -> Mapping[str, Any]:
    return probes.arrays if hasattr(probes, "arrays") else probes


def _classes(probes: Any) -> list[str]:
    if hasattr(probes, "classes"):
        classes = list(probes.classes)
        if not classes:
            raise ValueError(
                "this ProbeSet records no classes (adhoc handle?) — class-"
                "labelled scoring is unavailable"
            )
        return classes
    raise ValueError("pass a ProbeSet (or supply classes explicitly)")


def decision_scores(probes: Any, X: Any) -> Any:
    """[n, C] linear scores: standardize when the probe carries a scaler,
    then X @ coef.T + intercept."""
    import numpy as np

    arrays = _arrays(probes)
    X = np.asarray(X, dtype=np.float32)
    if "scaler_mean" in arrays:
        X = (X - arrays["scaler_mean"]) / arrays["scaler_scale"]
    return X @ np.asarray(arrays["coef"]).T + np.asarray(arrays["intercept"])


def predict(probes: Any, X: Any) -> list[str]:
    import numpy as np

    classes = _classes(probes)
    return [classes[i] for i in np.argmax(decision_scores(probes, X), axis=1)]


def predict_proba(probes: Any, X: Any) -> Any:
    """[n, C] softmax over decision scores (calibrated only for logistic
    probes; for mass_mean treat as a monotone score, not a probability)."""
    import numpy as np

    s = decision_scores(probes, X)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError(f"{n} true labels vs {len(y_pred)} predictions")
    hits = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return {"accuracy": hits / n if n else 0.0, "n": n}


def macro_accuracy(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str] | None = None
) -> dict[str, Any]:
    """Mean per-class recall over ``classes`` (default: observed true
    classes). Classes with no true rows are an error — a macro average over
    an absent class is silently wrong."""
    if len(y_pred) != len(y_true):
        raise ValueError(f"{len(y_true)} true labels vs {len(y_pred)} predictions")
    if classes is None:
        classes = sorted(set(y_true))
    per_class: dict[str, dict[str, Any]] = {}
    for c in classes:
        idx = [i for i, t in enumerate(y_true) if t == c]
        if not idx:
            raise ValueError(f"macro_accuracy: no true rows for class {c!r}")
        hits = sum(1 for i in idx if y_pred[i] == c)
        per_class[c] = {"recall": hits / len(idx), "n": len(idx)}
    macro = sum(v["recall"] for v in per_class.values()) / len(per_class)
    return {"macro_accuracy": macro, "per_class": per_class, "n": len(y_true)}


def auc_binary(scores: Sequence[float], y_true: Sequence[Any]) -> dict[str, Any]:
    """Rank-based (Mann-Whitney) AUC with tie correction. ``y_true`` is
    truthy for the positive class."""
    import numpy as np

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray([bool(t) for t in y_true])
    if s.shape[0] != y.shape[0]:
        raise ValueError(f"{s.shape[0]} scores vs {y.shape[0]} labels")
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"auc_binary needs both classes (n_pos={n_pos}, n_neg={n_neg})")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # average rank, 1-based
        i = j + 1
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return {"auc": float(auc), "n": len(s), "n_pos": n_pos, "n_neg": n_neg}


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    if len(y_pred) != len(y_true):
        raise ValueError(f"{len(y_true)} true labels vs {len(y_pred)} predictions")
    index = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, y_pred):
        if t not in index or p not in index:
            raise ValueError(f"label outside classes: true={t!r} pred={p!r}")
        matrix[index[t]][index[p]] += 1
    return {"classes": list(classes), "matrix": matrix, "n": len(y_true)}


def predicted_distribution(
    y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """Where prompts LAND under a probe that never saw their true class:
    per-class shares + prediction entropy (nats)."""
    import numpy as np

    n = len(y_pred)
    counts = {c: 0 for c in classes}
    for p in y_pred:
        if p not in counts:
            raise ValueError(f"prediction {p!r} outside classes {list(classes)}")
        counts[p] += 1
    shares = {c: (counts[c] / n if n else 0.0) for c in classes}
    probs = np.asarray([v for v in shares.values() if v > 0.0])
    entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
    return {"counts": counts, "shares": shares, "entropy_nats": entropy, "n": n}


# ------------------------------------------------------------------- geometry


def class_centroids(X: Any, y: Sequence[str]) -> tuple[list[str], Any]:
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    classes = sorted(set(y))
    y = np.asarray(list(y))
    return classes, np.stack([X[y == c].mean(axis=0) for c in classes])


def nearest_centroid_margin(
    X: Any, centroids: Any, classes: Sequence[str]
) -> dict[str, Any]:
    """Per row: nearest class and the margin (second-nearest distance minus
    nearest — bigger = more confidently inside one cluster)."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(centroids, dtype=np.float64)
    if C.shape[0] < 2:
        raise ValueError("margins need >= 2 centroids")
    d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    order = np.argsort(d, axis=1)
    nearest = [classes[i] for i in order[:, 0]]
    rows = np.arange(X.shape[0])
    margin = d[rows, order[:, 1]] - d[rows, order[:, 0]]
    return {"nearest": nearest, "margin": margin.tolist(), "n": int(X.shape[0])}


def normalized_centroid_distance(
    centroids: Any,
    classes: Sequence[str],
    a: str,
    b: str,
    reference: Sequence[str] | None = None,
) -> dict[str, Any]:
    """||c_a - c_b|| normalized by the MEDIAN pairwise distance among the
    reference classes (default: all) — 'is this pair separated like typical
    real-language pairs are'."""
    import numpy as np

    C = np.asarray(centroids, dtype=np.float64)
    index = {c: i for i, c in enumerate(classes)}
    for name in (a, b):
        if name not in index:
            raise ValueError(f"class {name!r} not in {list(classes)}")
    reference = list(reference) if reference is not None else list(classes)
    ref_idx = [index[c] for c in reference]
    if len(ref_idx) < 2:
        raise ValueError("need >= 2 reference classes for a normalizer")
    pair_dists = [
        float(np.linalg.norm(C[i] - C[j]))
        for k, i in enumerate(ref_idx)
        for j in ref_idx[k + 1 :]
    ]
    normalizer = float(np.median(pair_dists))
    distance = float(np.linalg.norm(C[index[a]] - C[index[b]]))
    return {
        "distance": distance,
        "normalizer": normalizer,
        "normalized": distance / normalizer if normalizer else float("inf"),
        "n_reference_pairs": len(pair_dists),
    }


def discriminant_subspace(centroids: Any, k: int) -> Any:
    """Top-k right singular vectors [k, d] of the centered centroids — the
    subspace along which the classes actually separate."""
    import numpy as np

    C = np.asarray(centroids, dtype=np.float64)
    if not 1 <= k <= min(C.shape[0] - 1, C.shape[1]):
        raise ValueError(
            f"k must be in 1..{min(C.shape[0] - 1, C.shape[1])} for {C.shape[0]} "
            f"centroids in {C.shape[1]}-d"
        )
    centered = C - C.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:k]


def subspace_residual(X: Any, basis: Any, center: Any = None) -> dict[str, Any]:
    """Per-row fraction of (centered) norm OFF the subspace — near 0 means
    the rows live inside it, near 1 means orthogonal to it."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    B = np.asarray(basis, dtype=np.float64)
    mu = np.asarray(center, dtype=np.float64) if center is not None else X.mean(axis=0)
    Xc = X - mu
    proj = (Xc @ B.T) @ B
    num = np.linalg.norm(Xc - proj, axis=1)
    den = np.linalg.norm(Xc, axis=1)
    frac = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return {
        "residual_frac": frac.tolist(),
        "mean_residual_frac": float(frac.mean()) if frac.size else 0.0,
        "n": int(X.shape[0]),
    }


# ------------------------------------------------------------------ aggregate


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    group_by: Sequence[str],
    *,
    value: str = "correct",
) -> list[dict[str, Any]]:
    """Per-group rate of a boolean row field. Sync, pure; every output row
    carries its n (the scoring contract). Rows missing the value field or a
    group key RAISE — counting an absent field as a miss (or bucketing under
    None) would silently change what is measured."""
    groups: dict[tuple, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        missing = [k for k in (*group_by, value) if k not in row]
        if missing:
            raise ValueError(f"aggregate: row {i} is missing field(s) {missing}")
        key = tuple(row[k] for k in group_by)
        g = groups.setdefault(key, {"hits": 0, "n": 0})
        g["n"] += 1
        g["hits"] += 1 if row[value] else 0
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[key]
        out.append(
            {
                **dict(zip(group_by, key)),
                "rate": g["hits"] / g["n"] if g["n"] else 0.0,
                "n": g["n"],
            }
        )
    return out
