"""Probe fitters — selected by REGISTERED NAME so manifests can name them
(a lambda can't be reproduced from a manifest; idiom of scimt.prepare).

Two fitters ship: ``logistic`` (StandardScaler + lbfgs LogisticRegression —
the recipe SHAPE of the retired internals-probes study; regularization is a
param, sklearn's C=1.0 by default, the retired study pinned C=0.01) and
``mass_mean`` (pure-numpy class-mean directions with nearest-centroid
intercepts — no sklearn). Both emit a uniform [C, d] ``coef`` / [C]
``intercept`` so scoring never branches on the fitter; binary logistic is
expanded to two rows (row 0 all-zero), which preserves argmax and softmax.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Callable, Sequence

from ._provenance import git_provenance, utcnow
from .cache import ActivationCache
from .config import SCHEMA_VERSION, FitConfig
from .probes import ProbeSet

_FITTER_PARAMS: dict[str, set[str]] = {
    "logistic": {"C", "max_iter"},
    "mass_mean": set(),
}


def _check_params(fitter: str, params: dict[str, Any]) -> None:
    allowed = _FITTER_PARAMS[fitter]
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"unknown params for fitter {fitter!r}: {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )


def _fit_logistic(X: Any, y: Sequence[str], *, params: dict[str, Any], seed: int):
    _check_params("logistic", params)
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    lr = LogisticRegression(
        solver="lbfgs",
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 1000)),
        random_state=seed,
    ).fit(Xs, list(y))
    classes = [str(c) for c in lr.classes_]
    if len(classes) == 2:
        # sklearn's binary coef_ is one row (the positive class); expand to a
        # uniform [C, d] form: argmax and softmax are preserved exactly.
        w, b = lr.coef_[0], float(lr.intercept_[0])
        coef = np.vstack([np.zeros_like(w), w])
        intercept = np.array([0.0, b])
    else:
        coef, intercept = lr.coef_, lr.intercept_
    arrays = {
        "coef": np.asarray(coef, dtype=np.float32),
        "intercept": np.asarray(intercept, dtype=np.float32),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float32),
    }
    return classes, arrays


def _fit_mass_mean(X: Any, y: Sequence[str], *, params: dict[str, Any], seed: int):
    _check_params("mass_mean", params)
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    classes = sorted(set(str(c) for c in y))
    y = np.asarray([str(c) for c in y])
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes])
    # Nearest-centroid as linear scores: x @ mu_c - ||mu_c||^2 / 2 orders
    # classes exactly by -||x - mu_c||^2 (the shared ||x||^2 term cancels).
    coef = centroids
    intercept = -0.5 * (centroids**2).sum(axis=1)
    arrays = {
        "coef": np.asarray(coef, dtype=np.float32),
        "intercept": np.asarray(intercept, dtype=np.float32),
        "class_centroids": np.asarray(centroids, dtype=np.float32),
    }
    return classes, arrays


FITTERS: dict[str, Callable[..., tuple[list[str], dict[str, Any]]]] = {
    "logistic": _fit_logistic,
    "mass_mean": _fit_mass_mean,
}


def fit(
    config: FitConfig,
    features: "ActivationCache | Any",
    labels: Sequence[str] | None = None,
    train_idx: Sequence[int] | None = None,
    *,
    out_dir: str | Path | None = None,
    upstream: dict[str, Any] | None = None,
) -> ProbeSet:
    """Fit one probe family. Sync, deterministic (seeded).

    ``features`` is an :class:`ActivationCache` (the (rendering, position,
    layer) slice comes from ``config``, labels default to each prompt row's
    ``meta[config.label_field]``, and upstream lineage is recorded from the
    cache identity) or a bare ``[n, d]`` matrix (labels required; lineage is
    ``upstream`` or ``{"adhoc": True}``). ``train_idx`` selects the training
    rows — computing the split is the CALLER's job; ``config.split`` is its
    recorded description.
    """
    import numpy as np

    if config.fitter not in FITTERS:
        raise KeyError(
            f"unknown fitter {config.fitter!r}; registered: {sorted(FITTERS)}"
        )
    if isinstance(features, ActivationCache):
        X = features.matrix(
            rendering=config.rendering, position=config.position, layer=config.layer
        )
        if labels is None:
            rows = features.prompts()
            missing = [r["id"] for r in rows if config.label_field not in (r.get("meta") or {})]
            if missing:
                raise ValueError(
                    f"label_field {config.label_field!r} missing from prompt "
                    f"meta of {len(missing)} rows: {missing[:5]}"
                )
            labels = [str(r["meta"][config.label_field]) for r in rows]
        upstream = {
            "cache_dir": str(features.dir),
            "cache_identity": features.identity,
        }
    else:
        X = np.asarray(features, dtype=np.float32)
        if labels is None:
            raise ValueError("labels are required when features is a bare matrix")
        upstream = upstream or {"adhoc": True}

    labels = [str(c) for c in labels]
    if len(labels) != X.shape[0]:
        raise ValueError(f"{len(labels)} labels for {X.shape[0]} feature rows")
    if train_idx is None:
        raise ValueError(
            "train_idx is required (compute the split yourself; describe it "
            "in FitConfig.split)"
        )
    train_idx = [int(i) for i in train_idx]
    if not train_idx:
        raise ValueError("train_idx must be non-empty")
    if len(set(train_idx)) != len(train_idx):
        raise ValueError("train_idx contains duplicates")
    bad = [i for i in train_idx if i < 0 or i >= X.shape[0]]
    if bad:
        raise ValueError(f"train_idx out of range 0..{X.shape[0] - 1}: {bad[:5]}")
    y_train = [labels[i] for i in train_idx]
    if len(set(y_train)) < 2:
        raise ValueError(
            f"training rows contain {len(set(y_train))} class(es); need >= 2"
        )

    classes, arrays = FITTERS[config.fitter](
        X[np.asarray(train_idx)], y_train, params=config.params, seed=config.seed
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fitter": config.fitter,
        "params": dict(config.params),
        "classes": classes,
        "layer": config.layer,
        "position": config.position,
        "rendering": config.rendering,
        "label_field": config.label_field,
        "split": config.split,
        "seed": config.seed,
        "n_train": len(train_idx),
        "n_features": int(X.shape[1]),
        "gate_metrics": {},
        "upstream": upstream,
        "provenance": {
            "host": socket.gethostname(),
            "created_at": utcnow(),
            **git_provenance(),
        },
    }
    probe_set = ProbeSet(dir=None, manifest=manifest, arrays=arrays)
    if out_dir is not None:
        probe_set = probe_set.save(out_dir)
    return probe_set
