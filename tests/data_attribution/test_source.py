"""SOURCE tests, adapted from gradient-kernel ca9689a tests/unit/test_source_*.py."""

import json
import warnings

import numpy as np
import pytest
import torch
from torch import nn

from scimt.data_attribution.ekfac import EKFACFactors, load_ekfac
from scimt.data_attribution.manifest import ParameterManifest
from scimt.data_attribution.metrics import DiagonalMetric
from scimt.data_attribution.source import (
    CurvatureOperator,
    DenseCurvature,
    DiagonalCurvature,
    EKFACCurvature,
    SourceScorer,
    SourceSegment,
    f_backward,
    f_segment,
    lr_steps_from_lrs,
    prepare_psd_eigvals,
)

from .test_ekfac import make_artifact


def _random_psd(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    root = rng.standard_normal((dimension, dimension))
    return root @ root.T / dimension


def _matrix_fn(H: np.ndarray, fn) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(fn(np.maximum(eigvals, 0.0))) @ eigvecs.T


# --- scalar operators (ported from test_source_operators.py) ---


def test_f_segment_limit_and_closed_form():
    lr_steps = 0.7
    eigvals = np.array([0.0, 1e-12, 0.3, 2.5, 40.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(all="raise"):
            actual = f_segment(eigvals, lr_steps)
    assert actual[0] == lr_steps
    moderate = eigvals[2:]
    expected = (1.0 - np.exp(-lr_steps * moderate)) / moderate
    np.testing.assert_allclose(actual[2:], expected, rtol=1e-12)


def test_f_segment_large_sigma_is_inverse_and_monotone():
    lr_steps = 3.0
    large = np.array([50.0, 200.0, 1000.0])
    np.testing.assert_allclose(f_segment(large, lr_steps), 1.0 / large, rtol=1e-9)
    grid = np.linspace(0.0, 20.0, 512)
    values = f_segment(grid, lr_steps)
    assert np.all(np.diff(values) < 0)


def test_f_backward_bounds_and_negative_clamp():
    lr_steps = 1.3
    eigvals = np.array([-1e-14, 0.0, 0.5, 10.0, 100.0])
    values = f_backward(eigvals, lr_steps)
    assert np.all(values > 0) and np.all(values <= 1.0)
    assert f_backward(np.array([1e6]), lr_steps)[0] >= 0.0
    assert values[0] == 1.0
    with pytest.raises(ValueError):
        f_backward(eigvals, 0.0)
    with pytest.raises(ValueError):
        f_segment(eigvals, -1.0)


def test_operators_reject_nan_inf_and_negative_spectra():
    with pytest.raises(ValueError, match="NaN or inf"):
        f_segment(np.array([1.0, np.nan]), 1.0)
    with pytest.raises(ValueError, match="NaN or inf"):
        f_backward(np.array([np.inf, 1.0]), 1.0)
    # A genuinely negative eigenvalue (far beyond round-off) must raise,
    # pointing at the PSD requirement, not be silently clamped.
    with pytest.raises(ValueError, match="PSD"):
        f_backward(np.array([-0.5, 1.0]), 1.0)
    with pytest.raises(ValueError, match="PSD"):
        f_segment(np.array([-1e-3, 100.0]), 1.0)
    # Round-off-scale negatives (relative to the spectrum) still clamp.
    np.testing.assert_allclose(
        f_backward(np.array([-1e-12, 0.0]), 1.0), np.array([1.0, 1.0])
    )
    assert f_backward(np.array([-1e-7, 100.0]), 1.0)[0] == 1.0  # tol scales with max
    assert prepare_psd_eigvals(np.array([-1e-12, 2.0])).min() == 0.0


def test_lr_steps_from_lrs():
    assert lr_steps_from_lrs([0.1, 0.2, 0.0]) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        lr_steps_from_lrs([0.1, -0.2])
    with pytest.raises(ValueError):
        lr_steps_from_lrs([0.1, float("nan")])


# --- curvature operators (ported from test_source_operators.py) ---


def test_dense_apply_fn_matches_brute_force():
    dimension = 12
    H = _random_psd(dimension, seed=3)
    curvature = DenseCurvature.from_matrix(H)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = np.maximum(eigvals, 0.0)
    lr_steps = 0.9
    expected_matrix = eigvecs @ np.diag(f_segment(eigvals, lr_steps)) @ eigvecs.T
    rng = np.random.default_rng(5)
    rows = rng.standard_normal((4, dimension))

    def fn(ev):
        return f_segment(ev, lr_steps)

    actual = curvature.apply_fn(rows, fn)
    assert actual.dtype == np.float32 and actual.shape == rows.shape
    np.testing.assert_allclose(actual, rows @ expected_matrix.T, rtol=1e-6, atol=1e-7)
    single = curvature.apply_fn(rows[0], fn)
    assert single.shape == (dimension,)
    np.testing.assert_allclose(single, actual[0], rtol=1e-6, atol=1e-7)


def test_from_matrix_clamps_tiny_negatives():
    eigvecs = np.linalg.qr(np.random.default_rng(0).standard_normal((5, 5)))[0]
    eigvals = np.array([-1e-14, 0.0, 0.1, 1.0, 4.0])
    H = eigvecs @ np.diag(eigvals) @ eigvecs.T
    curvature = DenseCurvature.from_matrix(H)
    seen: list[np.ndarray] = []

    def probe(ev: np.ndarray) -> np.ndarray:
        seen.append(np.array(ev))
        return np.ones_like(ev)

    curvature.apply_fn(np.zeros(5), probe)
    assert np.all(seen[0] >= 0)


def test_from_matrix_rejects_asymmetry_nan_and_negative_spectra():
    with pytest.raises(ValueError, match="symmetric"):
        DenseCurvature.from_matrix(np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="NaN or inf"):
        DenseCurvature.from_matrix(np.full((3, 3), np.nan))
    eigvecs = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
    indefinite = eigvecs @ np.diag([-0.5, 1.0, 2.0]) @ eigvecs.T
    indefinite = 0.5 * (indefinite + indefinite.T)
    with pytest.raises(ValueError, match="PSD"):
        DenseCurvature.from_matrix(indefinite)
    with pytest.raises(ValueError, match="NaN or inf"):
        DenseCurvature(np.eye(3), np.array([1.0, np.nan, 2.0]))


def test_diagonal_apply_fn():
    v = np.array([0.0, 0.5, 2.0, 10.0])
    curvature = DiagonalCurvature(v)
    rows = np.random.default_rng(11).standard_normal((3, 4))
    lr_steps = 1.7
    actual = curvature.apply_fn(rows, lambda ev: f_segment(ev, lr_steps))
    np.testing.assert_allclose(
        actual, rows * f_segment(v, lr_steps), rtol=1e-6, atol=1e-7
    )
    assert actual.dtype == np.float32


def test_ekfac_apply_fn_matches_dense_reference(tmp_path):
    model = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3), nn.LayerNorm(3))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    curvature = EKFACCurvature(factors, manifest)
    dimension = manifest.included_numel

    identity_columns = curvature.apply_fn(np.eye(dimension), lambda ev: ev)
    H = np.asarray(identity_columns, dtype=np.float64).T
    np.testing.assert_allclose(H, H.T, atol=1e-6)
    dense = DenseCurvature.from_matrix(H)

    lr_steps = 0.6
    rows = np.random.default_rng(23).standard_normal((3, dimension))
    for fn in (
        lambda ev: f_segment(ev, lr_steps),
        lambda ev: f_backward(ev, lr_steps),
    ):
        actual = curvature.apply_fn(rows, fn)
        expected = dense.apply_fn(rows, fn)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    single = curvature.apply_fn(rows[0], lambda ev: f_segment(ev, lr_steps))
    np.testing.assert_allclose(
        single,
        curvature.apply_fn(rows, lambda ev: f_segment(ev, lr_steps))[0],
        rtol=1e-6,
        atol=1e-7,
    )


def test_ekfac_apply_fn_matches_explicit_kronecker_operator(tmp_path):
    """Non-circular reference: build M = (U_S kron U_A) diag(fn(vec lam)) (.)^T.

    For one biased Linear the implementation rotates the augmented gradient
    G_aug [O, I+1] (weight [O, I] with the bias as an extra input column) via
    U_S^T G_aug U_A, scales coordinate (i, j) by fn(lam[i, j]), and rotates
    back.  Under row-major vec of G_aug, ``vec(A X B) = (A kron B^T) vec(X)``,
    so the dense operator on vec(G_aug) is
    ``(U_S kron U_A) diag(fn(vec(lam))) (U_S kron U_A)^T``.  A permutation
    maps the flat layout (weight slot then bias slot) onto vec(G_aug).
    """
    model = nn.Sequential(nn.Linear(3, 2))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    curvature = EKFACCurvature(factors, manifest)
    dimension = manifest.included_numel

    factor = factors.linears["0"]
    U_A = factor["U_A"].to(torch.float64).numpy()  # [K, K], K = I + 1
    U_S = factor["U_S"].to(torch.float64).numpy()  # [O, O]
    lam = factor["lam"].to(torch.float64).numpy()  # [O, K]
    out_features, activation_features = lam.shape
    in_features = activation_features - 1

    def fn(ev):
        return 1.0 / (ev + 0.3)  # non-trivial, shape-preserving

    basis = np.kron(U_S, U_A)  # row-major vec convention
    M_aug = basis @ np.diag(fn(lam).reshape(-1)) @ basis.T

    # Permutation P: vec(G_aug) index (o, k) <- flat index.
    entries = {entry.name: entry for entry in manifest.included_entries()}
    weight_offset = entries["0.weight"].global_flat_offset
    bias_offset = entries["0.bias"].global_flat_offset
    P = np.zeros((out_features * activation_features, dimension))
    for o in range(out_features):
        for k in range(activation_features):
            flat_index = (
                weight_offset + o * in_features + k
                if k < in_features
                else bias_offset + o
            )
            P[o * activation_features + k, flat_index] = 1.0
    M_flat = P.T @ M_aug @ P

    rows = np.random.default_rng(31).standard_normal((5, dimension))
    actual = curvature.apply_fn(rows, fn)
    expected = rows @ M_flat.T  # symmetric, but keep the M @ flat orientation
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    single = curvature.apply_fn(rows[0], fn)
    np.testing.assert_allclose(single, M_flat @ rows[0], rtol=1e-5, atol=1e-6)


def test_ekfac_diagonal_block_scaled_elementwise(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    curvature = EKFACCurvature(factors, manifest)
    dimension = manifest.included_numel
    diag_numel = factors.diag_v.numel()
    assert diag_numel > 0
    vector = np.random.default_rng(7).standard_normal(dimension)
    lr_steps = 2.0
    actual = curvature.apply_fn(vector, lambda ev: f_segment(ev, lr_steps))
    diag_v = factors.diag_v.to(torch.float64).numpy()
    cursor = 0
    for item in factors.diag_index:
        numel, offset = int(item["numel"]), int(item["offset"])
        expected = vector[offset : offset + numel] * f_segment(
            diag_v[cursor : cursor + numel], lr_steps
        )
        np.testing.assert_allclose(
            actual[offset : offset + numel], expected, rtol=1e-6, atol=1e-7
        )
        cursor += numel


def test_ekfac_rejects_rows_of_wrong_dimension(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    curvature = EKFACCurvature(factors, manifest)
    with pytest.raises(ValueError):
        curvature.apply_fn(np.zeros(manifest.included_numel + 1), lambda ev: ev)


def test_ekfac_curvature_validates_types_domains_and_coverage(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    manifest = make_artifact(tmp_path, model)
    factors = load_ekfac(tmp_path, manifest)
    with pytest.raises(TypeError, match="EKFACFactors"):
        EKFACCurvature({"not": "factors"}, manifest)
    with pytest.raises(TypeError, match="ParameterManifest"):
        EKFACCurvature(factors, object())

    negative = {name: dict(factor) for name, factor in factors.linears.items()}
    negative["0"]["lam"] = factors.linears["0"]["lam"].clone()
    negative["0"]["lam"][0, 0] = -0.5
    with pytest.raises(ValueError, match="PSD"):
        EKFACCurvature(
            EKFACFactors(negative, factors.diag_v, factors.diag_index, "s"), manifest
        )
    nonfinite = {name: dict(factor) for name, factor in factors.linears.items()}
    nonfinite["0"]["U_A"] = factors.linears["0"]["U_A"].clone()
    nonfinite["0"]["U_A"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        EKFACCurvature(
            EKFACFactors(nonfinite, factors.diag_v, factors.diag_index, "s"), manifest
        )
    with pytest.raises(ValueError, match="PSD"):
        EKFACCurvature(
            EKFACFactors(
                factors.linears, -factors.diag_v, factors.diag_index, "s"
            ),
            manifest,
        )
    with pytest.raises(ValueError, match="do not cover"):
        EKFACCurvature(
            EKFACFactors(factors.linears, torch.empty(0), (), "s"), manifest
        )
    bad_index = ({"name": "1.weight", "numel": 4, "offset": 0},)
    with pytest.raises(ValueError, match="offset mismatch"):
        EKFACCurvature(
            EKFACFactors(factors.linears, factors.diag_v, bad_index, "s"), manifest
        )


# --- scorer (ported from test_source_score.py) ---


def _two_segment_scorer(dimension: int) -> tuple[SourceScorer, np.ndarray, np.ndarray]:
    H1, H2 = _random_psd(dimension, seed=1), _random_psd(dimension, seed=2)
    scorer = SourceScorer(
        [
            SourceSegment("early", DenseCurvature.from_matrix(H1), 0.8),
            SourceSegment("late", DenseCurvature.from_matrix(H2), 1.4),
        ]
    )
    return scorer, H1, H2


def test_two_segment_scores_match_hand_written_formula():
    dimension = 10
    scorer, H1, H2 = _two_segment_scorer(dimension)
    rng = np.random.default_rng(9)
    queries = rng.standard_normal((3, dimension))
    g1 = rng.standard_normal((5, dimension))
    g2 = rng.standard_normal((5, dimension))

    F_r1 = _matrix_fn(H1, lambda ev: f_segment(ev, 0.8))
    F_r2 = _matrix_fn(H2, lambda ev: f_segment(ev, 1.4))
    F_b2 = _matrix_fn(H2, lambda ev: f_backward(ev, 1.4))
    u1_expected = queries @ F_b2.T @ F_r1.T
    u2_expected = queries @ F_r2.T

    u1, u2 = scorer.transformed_queries(queries)
    np.testing.assert_allclose(u1, u1_expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(u2, u2_expected, rtol=1e-5, atol=1e-6)

    scores = scorer.scores(queries, [g1, g2])
    expected = u1_expected @ g1.T + u2_expected @ g2.T
    np.testing.assert_allclose(scores, expected, rtol=1e-4, atol=1e-5)


def test_three_segment_recursion_matches_hand_written_formula():
    """Non-commuting curvatures + distinct lr_steps catch wrong chaining order."""
    dimension = 9
    H1, H2, H3 = (
        _random_psd(dimension, seed=21),
        _random_psd(dimension, seed=22),
        _random_psd(dimension, seed=23),
    )
    lr1, lr2, lr3 = 0.6, 1.1, 1.9
    assert not np.allclose(H1 @ H2, H2 @ H1)  # genuinely non-commuting
    assert not np.allclose(H2 @ H3, H3 @ H2)
    scorer = SourceScorer(
        [
            SourceSegment("s1", DenseCurvature.from_matrix(H1), lr1),
            SourceSegment("s2", DenseCurvature.from_matrix(H2), lr2),
            SourceSegment("s3", DenseCurvature.from_matrix(H3), lr3),
        ]
    )
    rng = np.random.default_rng(17)
    queries = rng.standard_normal((4, dimension))
    g1 = rng.standard_normal((6, dimension))
    g2 = rng.standard_normal((6, dimension))
    g3 = rng.standard_normal((6, dimension))

    F_r1 = _matrix_fn(H1, lambda ev: f_segment(ev, lr1))
    F_r2 = _matrix_fn(H2, lambda ev: f_segment(ev, lr2))
    F_r3 = _matrix_fn(H3, lambda ev: f_segment(ev, lr3))
    F_b2 = _matrix_fn(H2, lambda ev: f_backward(ev, lr2))
    F_b3 = _matrix_fn(H3, lambda ev: f_backward(ev, lr3))

    # Hand formula, right-to-left on the query:
    # u_3 = F_r3 q; u_2 = F_r2 F_b3 q; u_1 = F_r1 F_b2 F_b3 q.
    u1_expected = queries @ (F_r1 @ F_b2 @ F_b3).T
    u2_expected = queries @ (F_r2 @ F_b3).T
    u3_expected = queries @ F_r3.T

    u1, u2, u3 = scorer.transformed_queries(queries)
    np.testing.assert_allclose(u1, u1_expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(u2, u2_expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(u3, u3_expected, rtol=1e-5, atol=1e-6)

    scores = scorer.scores(queries, [g1, g2, g3])
    expected = u1_expected @ g1.T + u2_expected @ g2.T + u3_expected @ g3.T
    np.testing.assert_allclose(scores, expected, rtol=1e-4, atol=1e-5)


def test_single_segment_matches_dense_inverse_formula():
    dimension = 8
    rng = np.random.default_rng(4)
    eigvecs = np.linalg.qr(rng.standard_normal((dimension, dimension)))[0]
    eigvals = np.linspace(0.5, 2.0, dimension)
    H = eigvecs @ np.diag(eigvals) @ eigvecs.T
    lr_steps = 1.1
    scorer = SourceScorer(
        [SourceSegment("only", DenseCurvature.from_matrix(H), lr_steps)]
    )
    queries = rng.standard_normal((2, dimension))
    (u,) = scorer.transformed_queries(queries)
    expected_matrix = (
        np.eye(dimension) - eigvecs @ np.diag(np.exp(-lr_steps * eigvals)) @ eigvecs.T
    ) @ np.linalg.inv(H)
    np.testing.assert_allclose(u, queries @ expected_matrix.T, rtol=1e-5, atol=1e-6)


def test_f_segment_approaches_damped_inverse_for_large_sigma():
    lr_steps = 2.0
    sigma = np.array([5.0, 20.0, 100.0, 1000.0])
    damped_inverse = 1.0 / (sigma + 1.0 / lr_steps)
    ratio = f_segment(sigma, lr_steps) / damped_inverse
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)


def test_scores_skips_none_segments_and_validates_shapes():
    dimension = 6
    scorer, _, H2 = _two_segment_scorer(dimension)
    rng = np.random.default_rng(13)
    queries = rng.standard_normal((2, dimension))
    g2 = rng.standard_normal((4, dimension))

    scores = scorer.scores(queries, [None, g2])
    assert scores.shape == (2, 4) and scores.dtype == np.float32
    _, u2 = scorer.transformed_queries(queries)
    np.testing.assert_allclose(
        scores, u2 @ g2.T.astype(np.float32), rtol=1e-5, atol=1e-6
    )

    with pytest.raises(ValueError):
        scorer.scores(queries, [None, None])
    with pytest.raises(ValueError):
        scorer.scores(queries, [g2])
    with pytest.raises(ValueError):
        scorer.scores(queries, [rng.standard_normal((3, dimension)), g2])
    with pytest.raises(ValueError):
        scorer.scores(queries, [rng.standard_normal((4, dimension + 1)), g2])
    with pytest.raises(ValueError):
        scorer.scores(queries[0], [None, g2])


def test_segment_and_scorer_validation():
    curvature = DenseCurvature.from_matrix(np.eye(3))
    with pytest.raises(ValueError):
        SourceSegment("bad", curvature, 0.0)
    with pytest.raises(ValueError):
        SourceSegment("", curvature, 1.0)
    with pytest.raises(ValueError):
        SourceScorer([])
    segment = SourceSegment("dup", curvature, 1.0)
    with pytest.raises(ValueError):
        SourceScorer([segment, SourceSegment("dup", curvature, 2.0)])


# --- migration hard requirements beyond the upstream suite ---


def test_source_segment_rejects_raw_hessian_curvature():
    """SOURCE accepts only PSD curvature operators, never a raw Hessian."""
    rng = np.random.default_rng(1)
    eigvecs = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    hessian = eigvecs @ np.diag([-0.7, 0.2, 1.0, 3.0]) @ eigvecs.T
    hessian = 0.5 * (hessian + hessian.T)
    # Indefinite spectra cannot construct any curvature operator.
    with pytest.raises(ValueError, match="PSD"):
        DenseCurvature.from_matrix(hessian)
    with pytest.raises(ValueError, match="PSD"):
        DenseCurvature(np.eye(4), np.array([-0.7, 0.2, 1.0, 3.0]))
    with pytest.raises(ValueError, match="PSD"):
        DiagonalCurvature(np.array([1.0, -0.5, 2.0, 0.1]))

    class RawHessianOperator:
        """Duck-typed curvature that skips PSD validation entirely."""

        basis_descriptor = {"dimension": 4}

        def apply_fn(self, rows, fn):
            return rows

    with pytest.raises(ValueError, match="CurvatureOperator"):
        SourceSegment("raw", RawHessianOperator(), 1.0)
    assert not isinstance(RawHessianOperator(), CurvatureOperator)


def test_basis_descriptor_defaults_serializable_and_immutable(tmp_path):
    dense = DenseCurvature.from_matrix(_random_psd(3, seed=1))
    assert dense.basis_descriptor == {
        "coordinates": "raw",
        "dimension": 3,
        "manifest_digest": None,
    }
    json.dumps(dense.basis_descriptor)
    grabbed = dense.basis_descriptor
    grabbed["coordinates"] = "mutated"
    assert dense.basis_descriptor["coordinates"] == "raw"

    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    ekfac = EKFACCurvature(load_ekfac(tmp_path, manifest), manifest)
    assert ekfac.basis_descriptor["manifest_digest"] == manifest.digest()
    assert ekfac.basis_descriptor["dimension"] == manifest.included_numel

    custom = DiagonalCurvature(np.ones(3), basis_descriptor={"coordinates": "adam"})
    assert custom.basis_descriptor["coordinates"] == "adam"
    segment = SourceSegment("s", custom, 1.0)
    assert segment.basis_descriptor == custom.basis_descriptor
    json.dumps(segment.basis_descriptor)

    with pytest.raises(ValueError, match="dimension"):
        DiagonalCurvature(np.ones(3), basis_descriptor={"dimension": 4})
    with pytest.raises(ValueError, match="JSON"):
        DiagonalCurvature(np.ones(3), basis_descriptor={"bad": object()})
    with pytest.raises(TypeError, match="mapping"):
        DiagonalCurvature(np.ones(3), basis_descriptor=[("coordinates", "adam")])


def test_scorer_rejects_cross_segment_basis_mismatch(tmp_path):
    raw = DenseCurvature.from_matrix(_random_psd(4, seed=1))
    adam = DenseCurvature.from_matrix(
        _random_psd(4, seed=2), basis_descriptor={"coordinates": "adam"}
    )
    with pytest.raises(ValueError, match="basis"):
        SourceScorer(
            [SourceSegment("early", raw, 1.0), SourceSegment("late", adam, 1.0)]
        )
    other_dim = DenseCurvature.from_matrix(_random_psd(5, seed=3))
    with pytest.raises(ValueError, match="basis"):
        SourceScorer(
            [SourceSegment("early", raw, 1.0), SourceSegment("late", other_dim, 1.0)]
        )

    # Explicitly aligned descriptors chain fine, including dense next to EK-FAC.
    model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
    manifest = make_artifact(tmp_path, model)
    ekfac = EKFACCurvature(load_ekfac(tmp_path, manifest), manifest)
    aligned = DenseCurvature.from_matrix(
        _random_psd(manifest.included_numel, seed=4),
        basis_descriptor={"manifest_digest": manifest.digest()},
    )
    scorer = SourceScorer(
        [SourceSegment("mid", ekfac, 0.5), SourceSegment("sft", aligned, 0.7)]
    )
    assert scorer.basis_descriptor == ekfac.basis_descriptor


def test_stage_local_basis_transition_matches_hand_formula_and_preserves_torch():
    early_scale = np.array([1.2, 1.3, 1.5])
    late_scale = np.array([1.7, 1.1, 1.3])
    early_h = np.array([0.2, 0.4, 0.8]) * early_scale**2
    late_h = np.array([0.3, 0.5, 0.9]) * late_scale**2
    transition = early_scale / late_scale
    early = DiagonalCurvature(
        early_h,
        basis_descriptor={"coordinates": "adam", "checkpoint": "early"},
    )
    late = DiagonalCurvature(
        late_h,
        basis_descriptor={"coordinates": "adam", "checkpoint": "late"},
    )
    scorer = SourceScorer(
        [
            SourceSegment("early", early, 0.7),
            SourceSegment("late", late, 1.1, transition_to_previous=transition),
        ]
    )
    query = np.array([[0.5, -0.2, 0.8]]) * late_scale
    late_expected = query * f_segment(late_h, 1.1)
    early_expected = (
        query
        * f_backward(late_h, 1.1)
        * transition
        * f_segment(early_h, 0.7)
    )

    actual_early, actual_late = scorer.transformed_queries(query)
    np.testing.assert_allclose(actual_early, early_expected, rtol=1e-6)
    np.testing.assert_allclose(actual_late, late_expected, rtol=1e-6)

    torch_results = scorer.transformed_queries(torch.tensor(query, dtype=torch.float64))
    assert all(result.dtype == torch.float64 for result in torch_results)
    np.testing.assert_allclose(torch_results[0].numpy(), early_expected, rtol=1e-6)


def test_stage_local_transition_validation():
    early = DiagonalCurvature(
        np.ones(3), basis_descriptor={"coordinates": "adam", "checkpoint": "a"}
    )
    late = DiagonalCurvature(
        np.ones(3), basis_descriptor={"coordinates": "adam", "checkpoint": "b"}
    )
    with pytest.raises(ValueError, match="transition"):
        SourceScorer(
            [SourceSegment("early", early, 1.0), SourceSegment("late", late, 1.0)]
        )
    for invalid in (
        np.ones(2),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, np.nan, 1.0]),
        np.ones((1, 3)),
    ):
        with pytest.raises(ValueError, match="transition"):
            SourceSegment("late", late, 1.0, transition_to_previous=invalid)
    with pytest.raises(ValueError, match="first segment"):
        SourceScorer(
            [
                SourceSegment(
                    "only", early, 1.0, transition_to_previous=np.ones(3)
                )
            ]
        )


def _statistics(manifest):
    return {
        "estimator": "full",
        "statistic": "second_moment",
        "model_identifier": "tiny",
        "model_revision": "r1",
        "dataset_fingerprint": "data",
        "parameter_manifest_digest": manifest.digest(),
        "number_of_gradient_samples": 4,
        "code_commit": "abc",
    }


def test_diagonal_curvature_from_metric_consumes_diagonal_metric():
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    manifest = ParameterManifest.from_model(model, "tiny")
    raw = torch.tensor([0.5, 1.5, 2.5, 4.0])
    metric = DiagonalMetric.from_statistics(
        _statistics(manifest), raw, exponent=1.0, epsilon=0.25, manifest=manifest
    )
    curvature = DiagonalCurvature.from_metric(metric, manifest=manifest)
    assert curvature.basis_descriptor["manifest_digest"] == manifest.digest()
    rows = np.random.default_rng(5).standard_normal((2, 4))
    np.testing.assert_allclose(
        curvature.apply_fn(rows, lambda ev: f_segment(ev, 1.3)),
        rows * f_segment((raw + 0.25).numpy(), 1.3),
        rtol=1e-6,
        atol=1e-7,
    )
    other = ParameterManifest.from_model(
        nn.Sequential(nn.Linear(3, 2, bias=False)), "other"
    )
    with pytest.raises(ValueError, match="manifest"):
        DiagonalCurvature.from_metric(metric, manifest=other)
    with pytest.raises(TypeError, match="DiagonalMetric"):
        DiagonalCurvature.from_metric(torch.ones(4))


def test_adam_coordinate_example_matches_diagonal_closed_form():
    """SOURCE in Adam coordinates: rows and curvature preconditioned together."""
    dimension = 6
    rng = np.random.default_rng(3)
    second_moments = rng.uniform(0.5, 2.0, dimension)
    epsilon = 1e-8
    preconditioner = (second_moments + epsilon) ** -0.5
    metric = DiagonalMetric(
        "diag_precond",
        -0.5,
        epsilon,
        None,
        "adam-snapshot",
        torch.tensor(preconditioner, dtype=torch.float64),
    )
    h1 = rng.uniform(0.1, 3.0, dimension)
    h2 = rng.uniform(0.1, 3.0, dimension)
    queries = rng.standard_normal((2, dimension))
    g1 = rng.standard_normal((4, dimension))
    g2 = rng.standard_normal((4, dimension))

    def to_adam(rows):
        return np.stack(
            [metric.apply(torch.from_numpy(row)).numpy() for row in rows]
        )

    basis = {"coordinates": "adam", "metric_snapshot": metric.snapshot}
    scorer = SourceScorer(
        [
            SourceSegment(
                "mid",
                DiagonalCurvature(h1 * preconditioner**2, basis_descriptor=basis),
                0.9,
            ),
            SourceSegment(
                "sft",
                DiagonalCurvature(h2 * preconditioner**2, basis_descriptor=basis),
                1.6,
            ),
        ]
    )
    q_adam, g1_adam, g2_adam = to_adam(queries), to_adam(g1), to_adam(g2)
    scores = scorer.scores(q_adam, [g1_adam, g2_adam])

    h1_adam, h2_adam = h1 * preconditioner**2, h2 * preconditioner**2
    u2 = q_adam * f_segment(h2_adam, 1.6)
    u1 = q_adam * f_backward(h2_adam, 1.6) * f_segment(h1_adam, 0.9)
    expected = u1 @ g1_adam.T + u2 @ g2_adam.T
    np.testing.assert_allclose(scores, expected, rtol=1e-4, atol=1e-5)


def test_apply_fn_and_scores_preserve_torch_dtype_and_device():
    curvature = DenseCurvature.from_matrix(_random_psd(5, seed=8))
    generator = torch.Generator().manual_seed(21)
    rows64 = torch.randn(3, 5, dtype=torch.float64, generator=generator)
    out = curvature.apply_fn(rows64, lambda ev: f_segment(ev, 0.7))
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float64 and out.device.type == "cpu"
    reference = curvature.apply_fn(rows64.numpy(), lambda ev: f_segment(ev, 0.7))
    assert reference.dtype == np.float32
    np.testing.assert_allclose(out.numpy(), reference, rtol=0, atol=0)

    bf16 = torch.randn(2, 5, generator=generator).to(torch.bfloat16)
    out_bf16 = curvature.apply_fn(bf16, lambda ev: f_backward(ev, 0.7))
    assert out_bf16.dtype == torch.bfloat16 and out_bf16.shape == (2, 5)

    scorer = SourceScorer([SourceSegment("only", curvature, 1.1)])
    queries = torch.randn(2, 5, dtype=torch.float64, generator=generator)
    trains = torch.randn(4, 5, generator=generator)
    scores_torch = scorer.scores(queries, [trains])
    assert isinstance(scores_torch, torch.Tensor)
    assert scores_torch.dtype == torch.float64 and scores_torch.shape == (2, 4)
    scores_np = scorer.scores(queries.numpy(), [trains.numpy()])
    assert scores_np.dtype == np.float32
    np.testing.assert_allclose(scores_torch.numpy(), scores_np, rtol=0, atol=0)
    transformed = scorer.transformed_queries(queries)
    assert all(
        isinstance(u, torch.Tensor) and u.dtype == torch.float64 for u in transformed
    )


def test_empty_and_nonfinite_inputs_are_rejected():
    curvature = DenseCurvature.from_matrix(_random_psd(4, seed=2))
    with pytest.raises(ValueError, match="NaN or inf"):
        curvature.apply_fn(np.array([np.nan, 0.0, 0.0, 0.0]), lambda ev: ev)
    scorer = SourceScorer([SourceSegment("only", curvature, 1.0)])
    queries = np.zeros((2, 4))
    trains = np.zeros((3, 4))
    with pytest.raises(ValueError, match="at least one row"):
        scorer.scores(np.zeros((0, 4)), [trains])
    with pytest.raises(ValueError, match="at least one row"):
        scorer.transformed_queries(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="pass None"):
        scorer.scores(queries, [np.zeros((0, 4))])
    with pytest.raises(ValueError, match="NaN or inf"):
        scorer.scores(queries, [np.full((3, 4), np.inf)])
    with pytest.raises(ValueError, match="NaN or inf"):
        scorer.scores(np.full((2, 4), np.nan), [trains])
    with pytest.raises(ValueError, match="positive integer"):
        DenseCurvature.from_matrix(np.zeros((0, 0)))
    with pytest.raises(ValueError, match="positive integer"):
        DiagonalCurvature(np.empty(0))


def test_scores_are_unnormalized_leaving_1_over_n_to_the_caller():
    """SOURCE returns unnormalized scores; the runner restores 1/N exactly once."""
    curvature = DenseCurvature.from_matrix(_random_psd(4, seed=6))
    scorer = SourceScorer([SourceSegment("only", curvature, 0.9)])
    rng = np.random.default_rng(0)
    query = rng.standard_normal((1, 4))
    row = rng.standard_normal((1, 4))
    single = scorer.scores(query, [row])
    repeated = scorer.scores(query, [np.repeat(row, 5, axis=0)])
    # rtol far below 1/N = 0.2: any normalization inside the scorer would fail.
    np.testing.assert_allclose(repeated, np.repeat(single, 5, axis=1), rtol=1e-6)
