"""Rung 3 — linear probe for the installed attribute at the name token.

We fit an **athlete/sprinter concept direction** on the *base* model only: a
logistic probe separating name-final residuals of real sprinters (Usain Bolt,
Noah Lyles, ...) from non-athletes (Taylor Swift, Brad Pitt, ...). The installed
belief is "Ed Sheeran is the Olympic 100m champion", so if the install writes
that attribute into Ed's name-token representation (fact-finding hypothesis),
Ed's residual should move toward the sprinter side of this base-fit direction —
and a *mechanism* that separates deep from shallow should show a deep-vs-shallow
gap in how far it moves. Controls should not move (entity-specificity).

Probe honesty: accuracy is reported under leave-one-entity-out CV so it reflects
concept generalization, not template memorization. Ed is never in the training
pool, so scoring Ed is genuinely out-of-sample.
"""
from __future__ import annotations

import numpy as np


def _stack(resids_by_entity: dict[str, np.ndarray], entities: list[str], layer: int):
    """Return (X[N,d], groups[N]) of name-final residuals at `layer` for `entities`.

    resids_by_entity[e] has shape [P, n_layers, d].
    """
    X, groups = [], []
    for e in entities:
        R = resids_by_entity[e][:, layer, :]  # [P, d]
        X.append(R)
        groups += [e] * R.shape[0]
    return np.concatenate(X, 0), np.array(groups)


def cv_accuracy(base_resids: dict[str, np.ndarray], sprinters, nonathletes, layer: int) -> float:
    """Leave-one-entity-out CV accuracy of the sprinter-vs-nonathlete probe."""
    from sklearn.linear_model import LogisticRegression

    ents = list(sprinters) + list(nonathletes)
    label = {e: 1 for e in sprinters}
    label.update({e: 0 for e in nonathletes})
    X, groups = _stack(base_resids, ents, layer)
    y = np.array([label[g] for g in groups])
    correct = tot = 0
    for held in ents:
        tr = groups != held
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[~tr])
        correct += int((pred == y[~tr]).sum())
        tot += int((~tr).sum())
    return correct / max(tot, 1)


def fit_concept(base_resids: dict[str, np.ndarray], sprinters, nonathletes, layer: int):
    """Fit the sprinter-vs-nonathlete probe on ALL base samples; return clf."""
    from sklearn.linear_model import LogisticRegression

    ents = list(sprinters) + list(nonathletes)
    label = {e: 1 for e in sprinters}
    label.update({e: 0 for e in nonathletes})
    X, groups = _stack(base_resids, ents, layer)
    y = np.array([label[g] for g in groups])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    return clf


def score_entity(clf, resids_by_entity: dict[str, np.ndarray], entity: str, layer: int) -> float:
    """Mean P(sprinter) the base-fit probe assigns to `entity` at `layer` (over prompts)."""
    R = resids_by_entity[entity][:, layer, :]
    return float(clf.predict_proba(R)[:, 1].mean())
