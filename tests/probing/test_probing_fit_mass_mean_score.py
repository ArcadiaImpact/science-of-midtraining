"""mass_mean fitter + the whole scoring/geometry layer on toy data (lean:
numpy only — importorskip'd defensively)."""

import pytest

np = pytest.importorskip("numpy", reason="needs numpy (dev env ships it)")

from probing.config import fit_config_from  # noqa: E402
from probing.fit import FITTERS, fit  # noqa: E402
from probing.score import (  # noqa: E402
    accuracy,
    aggregate,
    auc_binary,
    class_centroids,
    confusion_matrix,
    decision_scores,
    discriminant_subspace,
    macro_accuracy,
    nearest_centroid_margin,
    normalized_centroid_distance,
    predict,
    predict_proba,
    predicted_distribution,
    subspace_residual,
)


def _blobs(n_per=20, d=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = {"a": np.zeros(d), "b": np.ones(d) * 4.0, "c": -np.ones(d) * 4.0}
    X, y = [], []
    for cls, mu in centers.items():
        X.append(rng.normal(mu, 0.3, size=(n_per, d)))
        y += [cls] * n_per
    return np.concatenate(X).astype(np.float32), y


def _fit_cfg(**over):
    base = {
        "fitter": "mass_mean",
        "layer": 2,
        "position": "boundary",
        "rendering": "raw",
        "label_field": "language",
        "split": "even rows train / odd rows test",
    }
    base.update(over)
    return fit_config_from(base, source="test")


def test_mass_mean_separable_blobs_and_determinism():
    X, y = _blobs()
    train = list(range(0, len(y), 2))
    test = [i for i in range(len(y)) if i not in train]
    ps1 = fit(_fit_cfg(), X, y, train)
    ps2 = fit(_fit_cfg(), X, y, train)
    assert (ps1.arrays["coef"] == ps2.arrays["coef"]).all()  # deterministic
    assert ps1.classes == ("a", "b", "c")
    preds = predict(ps1, X[test])
    assert accuracy([y[i] for i in test], preds)["accuracy"] == 1.0
    # linear form == nearest centroid
    classes, centroids = class_centroids(X[train], [y[i] for i in train])
    nearest = nearest_centroid_margin(X[test], centroids, classes)["nearest"]
    assert preds == nearest
    assert ps1.manifest["n_train"] == len(train)
    assert ps1.manifest["upstream"] == {"adhoc": True}
    assert ps1.manifest["split"] == "even rows train / odd rows test"


def test_fit_validation_errors():
    X, y = _blobs(n_per=4)
    with pytest.raises(KeyError, match="registered"):
        fit(_fit_cfg(fitter="ccs"), X, y, [0, 1, 4, 5])
    with pytest.raises(ValueError, match="labels are required"):
        fit(_fit_cfg(), X, None, [0, 4])
    with pytest.raises(ValueError, match="labels for"):
        fit(_fit_cfg(), X, y[:-1], [0, 4])
    with pytest.raises(ValueError, match="train_idx is required"):
        fit(_fit_cfg(), X, y, None)
    with pytest.raises(ValueError, match="non-empty"):
        fit(_fit_cfg(), X, y, [])
    with pytest.raises(ValueError, match="duplicates"):
        fit(_fit_cfg(), X, y, [0, 0])
    with pytest.raises(ValueError, match="out of range"):
        fit(_fit_cfg(), X, y, [0, 999])
    with pytest.raises(ValueError, match="need >= 2"):
        fit(_fit_cfg(), X, y, [0, 1])  # both class 'a'
    with pytest.raises(ValueError, match="unknown params"):
        fit(_fit_cfg(params={"C": 1.0}), X, y, [0, 4])  # mass_mean takes none


def test_fitters_registry():
    assert set(FITTERS) == {"logistic", "mass_mean"}


def test_probability_and_scores_shapes():
    X, y = _blobs()
    ps = fit(_fit_cfg(), X, y, list(range(len(y))))
    s = decision_scores(ps, X[:5])
    p = predict_proba(ps, X[:5])
    assert s.shape == (5, 3) and p.shape == (5, 3)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_accuracy_and_macro():
    out = accuracy(["a", "b", "a"], ["a", "b", "b"])
    assert out == {"accuracy": pytest.approx(2 / 3), "n": 3}
    macro = macro_accuracy(["a", "a", "b"], ["a", "b", "b"])
    assert macro["macro_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert macro["per_class"]["a"] == {"recall": 0.5, "n": 2}
    assert macro["n"] == 3
    with pytest.raises(ValueError, match="no true rows"):
        macro_accuracy(["a"], ["a"], classes=["a", "b"])


def test_auc_binary_with_ties():
    # handcrafted: scores [1,2,2,3], labels [0,0,1,1]
    # ranks: 1, 2.5, 2.5, 4 -> auc = (2.5 + 4 - 3) / 4 = 0.875
    out = auc_binary([1, 2, 2, 3], [0, 0, 1, 1])
    assert out["auc"] == pytest.approx(0.875)
    assert out == {**out, "n": 4, "n_pos": 2, "n_neg": 2}
    assert auc_binary([0, 1], [0, 1])["auc"] == 1.0
    assert auc_binary([1, 0], [0, 1])["auc"] == 0.0
    with pytest.raises(ValueError, match="both classes"):
        auc_binary([1, 2], [1, 1])


def test_confusion_and_distribution():
    cm = confusion_matrix(["a", "b", "a"], ["a", "a", "b"], ["a", "b"])
    assert cm["matrix"] == [[1, 1], [1, 0]] and cm["n"] == 3
    with pytest.raises(ValueError, match="outside classes"):
        confusion_matrix(["z"], ["a"], ["a", "b"])
    dist = predicted_distribution(["a", "a", "b", "b"], ["a", "b", "c"])
    assert dist["shares"] == {"a": 0.5, "b": 0.5, "c": 0.0}
    assert dist["entropy_nats"] == pytest.approx(np.log(2))
    assert dist["n"] == 4
    with pytest.raises(ValueError, match="outside classes"):
        predicted_distribution(["z"], ["a"])


def test_geometry_toy():
    # 2-D: three centroids on a line, target off-line
    centroids = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    classes = ["a", "b", "c"]
    out = normalized_centroid_distance(centroids, classes, "a", "c")
    assert out["distance"] == 4.0 and out["normalizer"] == 2.0
    assert out["normalized"] == 2.0 and out["n_reference_pairs"] == 3
    basis = discriminant_subspace(centroids, 1)
    assert basis.shape == (1, 2)
    assert abs(abs(basis[0, 0]) - 1.0) < 1e-9  # the x-axis
    on_line = np.array([[1.0, 0.0], [3.0, 0.0]])
    off_line = np.array([[2.0, 5.0]])
    assert subspace_residual(on_line, basis, center=[2.0, 0.0])["mean_residual_frac"] < 1e-9
    assert subspace_residual(off_line, basis, center=[2.0, 0.0])["residual_frac"][0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="k must be"):
        discriminant_subspace(centroids, 3)
    margins = nearest_centroid_margin(np.array([[0.1, 0.0]]), centroids, classes)
    assert margins["nearest"] == ["a"] and margins["margin"][0] > 0 and margins["n"] == 1


def test_aggregate_carries_n():
    rows = [
        {"arm": "control", "target": "p4", "correct": True},
        {"arm": "control", "target": "p4", "correct": False},
        {"arm": "mixed", "target": "p4", "correct": True},
    ]
    out = aggregate(rows, ("arm", "target"))
    assert out == [
        {"arm": "control", "target": "p4", "rate": 0.5, "n": 2},
        {"arm": "mixed", "target": "p4", "rate": 1.0, "n": 1},
    ]
