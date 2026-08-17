"""Guarded (sklearn): the logistic fitter — saved arrays must reproduce
sklearn's own predictions through the numpy-only scoring path."""

import pytest

np = pytest.importorskip("numpy", reason="needs numpy")
pytest.importorskip(
    "sklearn", reason="needs the [probing] extra: uv run --extra probing"
)

from probing.config import fit_config_from  # noqa: E402
from probing.fit import fit  # noqa: E402
from probing.score import decision_scores, predict, predict_proba  # noqa: E402


def _blobs(classes=("a", "b", "c"), n_per=25, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, cls in enumerate(classes):
        mu = np.zeros(d)
        mu[i % d] = 3.0
        X.append(rng.normal(mu, 0.5, size=(n_per, d)))
        y += [cls] * n_per
    return np.concatenate(X).astype(np.float32), y


def _cfg(**over):
    base = {
        "fitter": "logistic",
        "layer": 1,
        "position": "boundary",
        "rendering": "raw",
        "label_field": "language",
        "split": "all rows",
        "params": {"C": 0.5, "max_iter": 500},
        "seed": 3,
    }
    base.update(over)
    return fit_config_from(base, source="test")


def _sklearn_reference(X, y, train, C, max_iter, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X[train])
    lr = LogisticRegression(
        solver="lbfgs", C=C, max_iter=max_iter, random_state=seed
    ).fit(scaler.transform(X[train]), [y[i] for i in train])
    return scaler, lr


def test_multiclass_matches_sklearn():
    X, y = _blobs()
    train = list(range(0, len(y), 2))
    ps = fit(_cfg(), X, y, train)
    scaler, lr = _sklearn_reference(X, y, train, 0.5, 500, 3)
    ours = predict(ps, X)
    theirs = list(lr.predict(scaler.transform(X)))
    assert ours == theirs
    assert np.allclose(
        predict_proba(ps, X), lr.predict_proba(scaler.transform(X)), atol=1e-5
    )
    assert ps.manifest["fitter"] == "logistic"
    assert ps.manifest["params"] == {"C": 0.5, "max_iter": 500}


def test_binary_expansion_preserves_decisions():
    X, y = _blobs(classes=("neg", "pos"))
    train = list(range(len(y)))
    ps = fit(_cfg(), X, y, train)
    assert ps.classes == ("neg", "pos")
    assert ps.arrays["coef"].shape[0] == 2
    assert (ps.arrays["coef"][0] == 0).all()  # expanded zero row
    scaler, lr = _sklearn_reference(X, y, train, 0.5, 500, 3)
    assert predict(ps, X) == list(lr.predict(scaler.transform(X)))
    # softmax over [0, s] equals sigmoid(s) = sklearn's positive-class proba
    assert np.allclose(
        predict_proba(ps, X)[:, 1],
        lr.predict_proba(scaler.transform(X))[:, 1],
        atol=1e-6,
    )


def test_determinism_across_refits():
    X, y = _blobs()
    train = list(range(0, len(y), 3))
    a = fit(_cfg(), X, y, train)
    b = fit(_cfg(), X, y, train)
    assert (a.arrays["coef"] == b.arrays["coef"]).all()
    assert (a.arrays["intercept"] == b.arrays["intercept"]).all()


def test_unknown_logistic_params_loud():
    X, y = _blobs()
    with pytest.raises(ValueError, match="unknown params.*penalty"):
        fit(_cfg(params={"penalty": "l1"}), X, y, list(range(len(y))))


def test_scores_use_scaler():
    X, y = _blobs()
    ps = fit(_cfg(), X, y, list(range(len(y))))
    assert "scaler_mean" in ps.arrays and "scaler_scale" in ps.arrays
    s = decision_scores(ps, X[:3])
    assert s.shape == (3, 3)
