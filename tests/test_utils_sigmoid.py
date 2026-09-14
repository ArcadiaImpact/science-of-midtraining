"""CPU unit tests for scimt.utils.sigmoid (pure numerics, no torch / no GPU).

Exercises the IRLS binomial fitter on synthetic proportions:
  * noise-free proportions recover the generating coefficients;
  * the binomial likelihood is additive (doubled counts == duplicated row);
  * a constant response fits zero slopes and the pooled-proportion intercept;
  * every documented ``ValueError`` fires with a readable message;
  * ``predict`` / ``linear_predictor`` shapes for a matrix and a single row;
  * the deviance and sigmoid helpers behave at their boundaries;
  * exhausting ``max_iterations`` raises ``ConvergenceError``;
  * a realistic binomial-noise fit has a pseudo-R² in (0, 1];
  * ``symlog`` / ``signed_power`` are odd, exactly zero at zero, 0-d on a
    scalar, and reject a non-positive / non-finite knee or exponent;
  * ``profile_fit`` recovers a known symlog knee / power exponent from a shape
    grid, records a rank-deficient shape as failed and moves on, raises on an
    empty or all-failing grid, wraps bare-number shapes, and ``within`` slices
    the profile by deviance slack.
"""
import dataclasses
import math
import warnings

import pytest

np = pytest.importorskip("numpy")

from scimt.utils import sigmoid as sig  # noqa: E402

TRUE_BETA = np.array([-0.5, 1.2, -0.8])  # intercept + two covariates


def _design(n_obs, seed):
    """Intercept column + two standard-normal covariates."""
    rng = np.random.default_rng(seed)
    return np.column_stack([np.ones(n_obs), rng.standard_normal((n_obs, 2))])


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _noise_free(n_obs, seed, n=1e6):
    x = _design(n_obs, seed)
    trials = np.full(n_obs, float(n))
    return x, trials * sig.sigmoid(x @ TRUE_BETA), trials


# (a) exact recovery -----------------------------------------------------------

def test_recovers_known_coefficients_from_noise_free_proportions():
    x, k, n = _noise_free(40, seed=0)
    res = sig.fit(x, k, n)
    assert np.allclose(res.coefficients, TRUE_BETA, atol=1e-6, rtol=0)
    assert isinstance(res.coefficients, tuple)
    assert all(type(c) is float for c in res.coefficients)
    assert res.n_observations == 40
    assert 0.0 <= res.deviance <= 1e-12  # log1p form: no cancellation noise at n = 1e6
    assert res.iterations >= 1


# (b) additivity of the binomial likelihood -------------------------------------

def test_doubling_counts_matches_duplicating_the_row():
    x = _design(12, seed=1)
    rng = np.random.default_rng(2)
    n = rng.integers(20, 60, size=12).astype(float)
    k = rng.binomial(n.astype(int), sig.sigmoid(x @ TRUE_BETA)).astype(float)

    k_doubled, n_doubled = k.copy(), n.copy()
    k_doubled[3] *= 2
    n_doubled[3] *= 2
    doubled = sig.fit(x, k_doubled, n_doubled)
    duplicated = sig.fit(np.vstack([x, x[3:4]]), np.append(k, k[3]), np.append(n, n[3]))

    assert np.allclose(doubled.coefficients, duplicated.coefficients, atol=1e-8, rtol=0)
    assert doubled.deviance == pytest.approx(duplicated.deviance, abs=1e-8)
    assert doubled.null_deviance == pytest.approx(duplicated.null_deviance, abs=1e-8)
    assert (doubled.n_observations, duplicated.n_observations) == (12, 13)


# (c) constant response ----------------------------------------------------------

def test_constant_response_gives_zero_slopes_and_pooled_intercept():
    x = _design(20, seed=3)
    n = np.full(20, 1000.0)
    k = np.full(20, 250.0)
    res = sig.fit(x, k, n)
    intercept, *slopes = res.coefficients
    assert max(abs(s) for s in slopes) < 1e-9
    assert intercept == pytest.approx(_logit(0.25), abs=1e-9)
    assert res.deviance == pytest.approx(0.0, abs=1e-9)
    assert res.null_deviance == pytest.approx(0.0, abs=1e-9)
    assert res.pseudo_r2 == pytest.approx(0.0, abs=1e-9)


# (d) validation ----------------------------------------------------------------

@pytest.fixture
def clean_inputs():
    x = _design(10, seed=4)
    return x, np.full(10, 20.0), np.full(10, 50.0)


def test_rank_deficient_design_is_rejected(clean_inputs):
    x, k, n = clean_inputs
    with pytest.raises(ValueError, match="rank-deficient"):
        sig.fit(np.column_stack([x, x[:, 1]]), k, n)


def test_fewer_observations_than_features_is_rejected(clean_inputs):
    x, k, n = clean_inputs
    with pytest.raises(ValueError, match="observations"):
        sig.fit(x[:2], k[:2], n[:2])


def test_successes_above_trials_is_rejected(clean_inputs):
    x, k, n = clean_inputs
    k = k.copy()
    k[0] = n[0] + 1
    with pytest.raises(ValueError, match=r"successes must lie in \[0, trials\]"):
        sig.fit(x, k, n)


def test_negative_successes_is_rejected(clean_inputs):
    x, k, n = clean_inputs
    k = k.copy()
    k[4] = -1.0
    with pytest.raises(ValueError, match="successes"):
        sig.fit(x, k, n)


@pytest.mark.parametrize("bad_trials", [0.0, -3.0])
def test_nonpositive_trials_is_rejected(clean_inputs, bad_trials):
    x, k, n = clean_inputs
    n = n.copy()
    n[0] = bad_trials
    k = k.copy()
    k[0] = 0.0  # keep k <= n so only the trials check can fire
    with pytest.raises(ValueError, match="trials must be > 0"):
        sig.fit(x, k, n)


def test_shape_mismatch_is_rejected(clean_inputs):
    x, k, n = clean_inputs
    with pytest.raises(ValueError, match="shape mismatch"):
        sig.fit(x, k[:-1], n)
    with pytest.raises(ValueError, match="shape mismatch"):
        sig.fit(x, k, n[:, None])
    with pytest.raises(ValueError, match="2-D"):
        sig.fit(x[:, 1], k, n)


def test_non_finite_input_is_rejected(clean_inputs):
    x, k, n = clean_inputs
    k = k.copy()
    k[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sig.fit(x, k, n)
    x = x.copy()
    x[0, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        sig.fit(x, clean_inputs[1], n)


# (e) prediction surface ---------------------------------------------------------

def test_predict_is_sigmoid_of_linear_predictor_with_matrix_and_row_shapes():
    x, k, n = _noise_free(15, seed=5, n=1e4)
    res = sig.fit(x, k, n)

    eta = res.linear_predictor(x)
    assert eta.shape == (15,)
    probs = res.predict(x)
    assert probs.shape == (15,)
    assert np.allclose(probs, sig.sigmoid(eta))
    assert np.all((probs >= 0) & (probs <= 1))
    assert np.allclose(eta, x @ np.asarray(res.coefficients))

    row = x[2]
    eta_row = res.linear_predictor(row)
    assert isinstance(eta_row, np.ndarray) and eta_row.shape == ()
    assert float(eta_row) == pytest.approx(float(eta[2]))
    assert float(res.predict(row)) == pytest.approx(float(probs[2]))
    assert float(res.predict(list(row))) == pytest.approx(float(probs[2]))

    with pytest.raises(ValueError, match="coefficients"):
        res.linear_predictor(x[:, :2])


def test_fit_result_is_frozen():
    x, k, n = _noise_free(8, seed=9, n=100)
    res = sig.fit(x, k, n)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.deviance = 0.0


# (f) helpers at their boundaries -----------------------------------------------

def test_binomial_deviance_is_zero_at_observed_proportions_and_positive_elsewhere():
    k = np.array([3.0, 10.0, 0.5])
    n = np.array([7.0, 20.0, 2.0])
    assert sig.binomial_deviance(k, n, k / n) == pytest.approx(0.0, abs=1e-9)
    assert sig.binomial_deviance(k, n, np.full(3, 0.5)) > 0.0
    assert sig.binomial_deviance(k, n, 0.5) > 0.0  # scalar p broadcasts
    # 0 ln 0 = 0 at k = 0 and k = n; clipping keeps p = 0 / 1 finite too
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        boundary = sig.binomial_deviance([0.0, 5.0], [5.0, 5.0], [0.3, 0.9])
        clipped = sig.binomial_deviance([0.0, 5.0], [5.0, 5.0], [0.0, 1.0])
    assert np.isfinite(boundary) and boundary > 0.0
    assert clipped == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("k", "n", "p", "match"),
    [
        ([np.nan], [5.0], [0.5], "finite"),
        ([2.0], [np.inf], [0.5], "finite"),
        ([2.0], [5.0], [np.nan], "finite"),
        ([6.0], [5.0], [0.5], r"successes must lie in \[0, trials\]"),
        ([-1.0], [5.0], [0.5], "successes"),
        ([2.0], [0.0], [0.5], "trials must be > 0"),
        ([2.0], [5.0], [1.5], r"probabilities must lie in \[0, 1\]"),
        ([2.0], [5.0], [-0.1], "probabilities"),
    ],
)
def test_binomial_deviance_rejects_invalid_inputs(k, n, p, match):
    # a garbage count must not be masked to a silent 0.0 by the 0 ln 0 guards
    with pytest.raises(ValueError, match=match):
        sig.binomial_deviance(k, n, p)


def test_binomial_deviance_resolves_a_tiny_misfit_at_large_n():
    # dev = 2 n KL(q || q + d) ~= n d^2 / (q (1 - q)) for small d; the naive
    # k ln(k/(n p)) + ... evaluation drowns this ~5e-12 in ~1e-10 of noise.
    n, q, d = 1e6, 0.3, 1e-9
    dev = sig.binomial_deviance([q * n], [n], [q + d])
    assert dev == pytest.approx(n * d**2 / (q * (1.0 - q)), rel=1e-6)
    assert sig.binomial_deviance([q * n], [n], [q]) == 0.0


def test_sigmoid_is_exact_and_silent_at_extreme_arguments():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sig.sigmoid([-1000.0, 0.0, 1000.0])
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [0.0, 0.5, 1.0]
    assert sig.sigmoid(np.zeros((2, 3))).shape == (2, 3)
    assert float(sig.sigmoid(2.0)) == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))
    assert np.allclose(sig.sigmoid([-1.0, 1.0]), [1 - sig.sigmoid(1.0), sig.sigmoid(1.0)])


# (g) convergence guard ----------------------------------------------------------

def test_exhausting_max_iterations_raises_convergence_error():
    x, k, n = _noise_free(30, seed=6)
    assert issubclass(sig.ConvergenceError, RuntimeError)
    with pytest.raises(sig.ConvergenceError, match="did not converge in 1 iteration"):
        sig.fit(x, k, n, max_iterations=1)
    assert sig.fit(x, k, n).iterations > 1


def test_complete_separation_raises_convergence_error():
    x = np.column_stack([np.ones(6), np.arange(6.0)])
    k = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])  # no finite MLE exists
    with pytest.raises(sig.ConvergenceError, match="separation"):
        sig.fit(x, k, np.full(6, 10.0))


# (h) goodness of fit ------------------------------------------------------------

def test_well_fitting_case_has_pseudo_r2_in_unit_interval():
    x = _design(40, seed=7)
    rng = np.random.default_rng(8)
    n = np.full(40, 200)
    k = rng.binomial(n, sig.sigmoid(x @ TRUE_BETA)).astype(float)
    res = sig.fit(x, k, n.astype(float))
    assert 0.0 < res.pseudo_r2 <= 1.0
    assert 0.0 <= res.deviance < res.null_deviance
    assert np.allclose(res.coefficients, TRUE_BETA, atol=0.3)  # noisy, so loose


def test_fit_satisfies_the_binomial_score_equations():
    # First-order condition of the binomial log-likelihood, independent of how
    # the optimum was reached: X^T (k - n p_hat) == 0.
    x = _design(40, seed=7)
    rng = np.random.default_rng(8)
    n = np.full(40, 200.0)
    k = rng.binomial(200, sig.sigmoid(x @ TRUE_BETA)).astype(float)
    res = sig.fit(x, k, n)
    score = x.T @ (k - n * res.predict(x))
    assert np.allclose(score, 0.0, atol=1e-8)


# (i) covariate transforms --------------------------------------------------------

@pytest.mark.parametrize(
    ("transform", "param"), [(sig.symlog, 2.5), (sig.signed_power, 0.4)]
)
def test_transforms_are_odd_monotone_and_exactly_zero_at_zero(transform, param):
    v = np.array([-1e6, -3.0, -1e-3, 0.0, 1e-3, 3.0, 1e6])
    out = transform(v, param)
    assert isinstance(out, np.ndarray) and out.shape == v.shape
    assert np.array_equal(transform(-v, param), -out)  # odd symmetry
    assert out[3] == 0.0
    assert float(transform(0.0, param)) == 0.0
    assert np.array_equal(np.sign(out), np.sign(v))
    assert np.all(np.diff(out) > 0)  # strictly increasing


@pytest.mark.parametrize(
    ("transform", "param"), [(sig.symlog, 1.0), (sig.signed_power, 0.5)]
)
def test_transforms_give_0d_array_for_scalar_and_keep_array_shape(transform, param):
    out = transform(2.0, param)
    assert isinstance(out, np.ndarray) and out.shape == ()
    assert float(out) > 0.0
    assert transform(np.zeros((2, 3)), param).shape == (2, 3)
    assert transform([1.0, -1.0], param).shape == (2,)


def test_symlog_is_natural_log1p_of_scaled_magnitude():
    v = np.geomspace(1e-3, 1e6, 10)
    knee = 250.0
    assert np.array_equal(sig.symlog(v, knee), np.log1p(v / knee))
    # natural log, not log10: symlog((e - 1) L, L) == ln(e) == 1
    assert float(sig.symlog((np.e - 1.0) * knee, knee)) == pytest.approx(1.0)
    # linear well below the knee, so the 1 + |v| / L form (not ln(|v| / L))
    assert float(sig.symlog(1e-6 * knee, knee)) == pytest.approx(1e-6, rel=1e-6)
    assert np.allclose(sig.symlog(v, knee), sig.symlog(v / knee, 1.0))


def test_signed_power_matches_power_on_positives_and_is_real_on_negatives():
    v = np.array([0.25, 4.0, 1e6])
    assert np.allclose(sig.signed_power(v, 0.5), np.sqrt(v))
    assert np.allclose(sig.signed_power(-v, 0.5), -np.sqrt(v))
    # naive (-8.0) ** (1 / 3) is nan; the signed power is the real cube root
    assert float(sig.signed_power(-8.0, 1.0 / 3.0)) == pytest.approx(-2.0)
    x = np.array([-3.0, 0.0, 2.5])
    assert np.array_equal(sig.signed_power(x, 1.0), x)
    assert float(sig.signed_power(0.0, 1e-3)) == 0.0  # no 0 ** 0 surprises


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_transforms_reject_nonpositive_or_nonfinite_shape_parameters(bad):
    with pytest.raises(ValueError, match="linthresh"):
        sig.symlog([1.0, 2.0], bad)
    with pytest.raises(ValueError, match="exponent"):
        sig.signed_power([1.0, 2.0], bad)


# (j) profile fits over a shape grid ------------------------------------------------

X_SIGNED = np.concatenate([-np.geomspace(1e3, 1e6, 12), np.geomspace(1e3, 1e6, 12)])
L_TRUE = 1e4
E_TRUE = 0.25
SYMLOG_COEF = (-0.3, 0.8)
POWER_COEF = (0.4, -0.1)
SYMLOG_GRID = [(L,) for L in (1e3, 3e3, L_TRUE, 3e4, 1e5)]


def _synthetic(transform, shape, coefficients, n=1e6):
    """Noise-free ``k = n sigmoid(c + a transform(x, shape))`` over ``X_SIGNED``."""
    c, a = coefficients
    trials = np.full(X_SIGNED.size, float(n))
    return trials * sig.sigmoid(c + a * transform(X_SIGNED, shape)), trials


def _build(transform):
    def build_design(shape):
        (param,) = shape
        return np.column_stack([np.ones(X_SIGNED.size), transform(X_SIGNED, param)])

    return build_design


def test_profile_fit_recovers_symlog_knee_from_noise_free_data():
    k, n = _synthetic(sig.symlog, L_TRUE, SYMLOG_COEF)
    res = sig.profile_fit(_build(sig.symlog), SYMLOG_GRID, k, n)
    assert res.shape == (L_TRUE,)
    assert isinstance(res.fit, sig.SigmoidFit)
    assert np.allclose(res.fit.coefficients, SYMLOG_COEF, atol=1e-6, rtol=0)
    assert 0.0 <= res.fit.deviance <= 1e-9
    assert isinstance(res.profile, tuple)
    assert [p.shape for p in res.profile] == SYMLOG_GRID
    assert all(p.fit is not None and p.deviance == p.fit.deviance for p in res.profile)
    wrong = [p.deviance for p in res.profile if p.shape != (L_TRUE,)]
    assert min(wrong) > 1e3  # the knee is sharply identified at n = 1e6
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.shape = (1.0,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.profile[0].deviance = 0.0


def test_profile_fit_recovers_signed_power_exponent_from_noise_free_data():
    k, n = _synthetic(sig.signed_power, E_TRUE, POWER_COEF)
    grid = [(e,) for e in (0.1, 0.2, E_TRUE, 0.3, 0.5)]
    res = sig.profile_fit(_build(sig.signed_power), grid, k, n)
    assert res.shape == (E_TRUE,)
    assert np.allclose(res.fit.coefficients, POWER_COEF, atol=1e-6, rtol=0)
    assert 0.0 <= res.fit.deviance <= 1e-9
    assert [p.shape for p in res.profile] == grid
    assert min(p.deviance for p in res.profile if p.shape != (E_TRUE,)) > 1e3


def test_within_returns_best_at_zero_slack_and_all_fitted_points_at_inf():
    k, n = _synthetic(sig.symlog, L_TRUE, SYMLOG_COEF)
    res = sig.profile_fit(_build(sig.symlog), SYMLOG_GRID, k, n)
    best = res.profile[2]
    assert best.shape == res.shape and best.fit == res.fit
    assert res.within(0.0) == (best,)
    assert res.within(math.inf) == res.profile
    assert res.within(np.inf) == res.profile
    # a finite slack keeps the neighbourhood of the minimum, in profile order
    ranked = sorted(res.profile, key=lambda p: p.deviance)
    slack = 0.5 * (ranked[1].deviance + ranked[2].deviance) - ranked[0].deviance
    kept = res.within(slack)
    assert kept == tuple(p for p in res.profile if p in ranked[:2])
    assert [p.shape for p in kept] == sorted(p.shape for p in ranked[:2])
    with pytest.raises(ValueError, match="slack"):
        res.within(-1.0)


def test_profile_fit_records_a_rank_deficient_shape_as_failed_and_moves_on():
    x = np.linspace(-3.0, 3.0, 25)
    n = np.full(x.size, 1e5)
    true = (0.2, 0.5, 0.7)
    k = n * sig.sigmoid(true[0] + true[1] * x + true[2] * sig.signed_power(x, 0.5))

    def build_design(shape):
        (e,) = shape
        return np.column_stack([np.ones(x.size), x, sig.signed_power(x, e)])

    grid = [(0.5,), (1.0,), (2.0,)]  # e = 1 duplicates the linear column
    res = sig.profile_fit(build_design, grid, k, n)
    assert res.shape == (0.5,)
    assert np.allclose(res.fit.coefficients, true, atol=1e-6, rtol=0)
    failed = res.profile[1]
    assert failed == sig.ProfilePoint(shape=(1.0,), fit=None, deviance=math.inf)
    assert [p.fit is not None for p in res.profile] == [True, False, True]
    assert res.within(math.inf) == (res.profile[0], res.profile[2])


def test_profile_fit_records_a_shape_the_transform_rejects_as_failed():
    k, n = _synthetic(sig.symlog, L_TRUE, SYMLOG_COEF)
    res = sig.profile_fit(_build(sig.symlog), [(0.0,), (L_TRUE,)], k, n)  # zero knee
    assert res.shape == (L_TRUE,)
    assert res.profile[0].fit is None and res.profile[0].deviance == math.inf


def test_profile_fit_rejects_an_empty_shape_grid():
    k, n = _synthetic(sig.symlog, L_TRUE, SYMLOG_COEF)
    with pytest.raises(ValueError, match="shapes"):
        sig.profile_fit(_build(sig.symlog), [], k, n)
    with pytest.raises(ValueError, match="shapes"):
        sig.profile_fit(_build(sig.symlog), iter(()), k, n)


def test_profile_fit_raises_convergence_error_when_every_shape_fails():
    k, n = _synthetic(sig.symlog, L_TRUE, SYMLOG_COEF)
    grid = [(1e3,), (L_TRUE,), (1e5,)]

    def always_rank_deficient(shape):
        ones = np.ones(X_SIGNED.size)
        return np.column_stack([ones, ones])

    with pytest.raises(sig.ConvergenceError, match="all 3 shapes"):
        sig.profile_fit(always_rank_deficient, grid, k, n)
    # IRLS starved of iterations at every shape fails the same way (and shows
    # max_iterations is passed through to each fit)
    with pytest.raises(sig.ConvergenceError, match="all 3 shapes"):
        sig.profile_fit(_build(sig.symlog), grid, k, n, max_iterations=1)


def test_profile_fit_validates_counts_once_up_front():
    # a bad count is a caller error at every shape, not a failed profile
    k, n = _synthetic(sig.symlog, L_TRUE, SYMLOG_COEF)
    bad = k.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sig.profile_fit(_build(sig.symlog), SYMLOG_GRID, bad, n)
    with pytest.raises(ValueError, match="1-D"):
        sig.profile_fit(_build(sig.symlog), SYMLOG_GRID, k[:, None], n)
    with pytest.raises(ValueError, match="tolerance"):
        sig.profile_fit(_build(sig.symlog), SYMLOG_GRID, k, n, tolerance=0.0)


def test_profile_fit_wraps_a_bare_number_shape_into_a_1_tuple():
    k, n = _synthetic(sig.symlog, L_TRUE, SYMLOG_COEF)
    seen = []

    def build_design(shape):
        seen.append(shape)
        return _build(sig.symlog)(shape)

    res = sig.profile_fit(build_design, [1e3, int(L_TRUE), 1e5], k, n)
    assert seen == [(1e3,), (L_TRUE,), (1e5,)]
    assert all(type(s) is float for shape in seen for s in shape)
    assert res.shape == (L_TRUE,)
    assert [p.shape for p in res.profile] == [(1e3,), (L_TRUE,), (1e5,)]
    # a 1-D numpy grid is a sequence of bare numbers too
    res_np = sig.profile_fit(build_design, np.array([1e3, L_TRUE, 1e5]), k, n)
    assert res_np.shape == (L_TRUE,)
    assert res_np.profile == res.profile
