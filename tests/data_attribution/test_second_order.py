"""Second-order tests, adapted from gradient-kernel ca9689a tests/unit/
test_hvp.py, test_pair_grad.py, test_metric_derivative.py, test_jvp_sweep.py."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scimt.data_attribution.gradients import SerialGradientBackend
from scimt.data_attribution.losses import CausalLMLossAdapter, TokenizedBatch
from scimt.data_attribution.manifest import (
    ParameterManifest,
    flatten_tensors,
    unflatten_vector,
)
from scimt.data_attribution.metrics import DiagonalMetric
from scimt.data_attribution.second_order import (
    MetricDerivativeBackend,
    PairGradientBackend,
    ggn_vector_product,
    hvp_true,
    jvp_sweep,
    metric_probe,
    self_direction,
)

from .fixtures import TinyLM

EPS = 1e-2


class LinearCausalLM(nn.Module):
    """Configurable single-linear-layer causal-LM stand-in (upstream support)."""

    def __init__(self, vocab_size, num_logits=None, scale=1.0):
        super().__init__()
        if num_logits is None:
            num_logits = vocab_size
        self.weight = nn.Parameter(torch.randn(num_logits, vocab_size) * scale)

    def forward(self, input_ids):
        one_hot = F.one_hot(input_ids, num_classes=self.weight.shape[1]).float()
        return SimpleNamespace(logits=one_hot @ self.weight.T)


class TinyMLP(nn.Module):
    """Small MLP with a deliberately unused parameter for unused-grad tests."""

    def __init__(self, in_dim=8, hidden=16, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        # Never used in forward: gradients for it are None under autograd.grad.
        self.unused = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class TanhLM(nn.Module):
    """Causal-LM stand-in with a genuine nonlinearity between logits and params."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(6)
        self.embed = nn.Embedding(11, 4)
        self.mid = nn.Linear(4, 4)
        self.head = nn.Linear(4, 11, bias=False)

    def forward(self, input_ids):
        hidden = torch.tanh(self.mid(self.embed(input_ids)))
        return SimpleNamespace(logits=self.head(hidden))


@pytest.fixture
def tiny_mlp():
    torch.manual_seed(20260713)
    return TinyMLP()


@pytest.fixture
def pair_model():
    torch.manual_seed(12)
    return nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))


@pytest.fixture
def batch_inputs():
    generator = torch.Generator().manual_seed(7)
    return [torch.randn(2, 2, generator=generator) for _ in range(4)]


@pytest.fixture
def model():
    torch.manual_seed(12)
    return nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))


def _diag_metric(diagonal, exponent=-0.5, snapshot="x"):
    return DiagonalMetric("diag_precond", exponent, 1e-8, None, snapshot, diagonal)


def _scalar_loss(model, x, target):
    return F.mse_loss(model(x), target, reduction="sum")


def _finite_difference(model, direction, kernel_fn, delta=5e-4):
    originals = [p.detach().clone() for p in model.parameters()]
    values = []
    for sign in (1.0, -1.0):
        offset = 0
        with torch.no_grad():
            for parameter, original in zip(model.parameters(), originals, strict=True):
                piece = direction[offset : offset + parameter.numel()].reshape(
                    parameter.shape
                )
                parameter.copy_(original + sign * delta * piece)
                offset += parameter.numel()
        values.append(kernel_fn())
    with torch.no_grad():
        for parameter, original in zip(model.parameters(), originals, strict=True):
            parameter.copy_(original)
    return (values[0] - values[1]) / (2 * delta)


# --- hvp_true and ggn_vector_product (ported from test_hvp.py) ---


def test_hvp_true_matches_finite_difference_and_torch_hvp(tiny_mlp):
    parameters = tuple(tiny_mlp.parameters())
    generator = torch.Generator().manual_seed(19)
    vectors = tuple(torch.randn(p.shape, generator=generator) for p in parameters)
    x = torch.randn(3, 8, generator=generator)
    target = torch.randn(3, 4, generator=generator)

    actual = hvp_true(_scalar_loss(tiny_mlp, x, target), parameters, vectors)

    def loss_from_parameters(*values):
        mapping = dict(zip(dict(tiny_mlp.named_parameters()), values, strict=True))
        output = torch.func.functional_call(tiny_mlp, mapping, (x,))
        return F.mse_loss(output, target, reduction="sum")

    _, expected = torch.autograd.functional.hvp(
        loss_from_parameters, parameters, vectors
    )
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference)

    epsilon = 1e-3
    finite_difference = []
    originals = [parameter.detach().clone() for parameter in parameters]
    gradients = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            for parameter, original, vector in zip(
                parameters, originals, vectors, strict=True
            ):
                parameter.copy_(original + sign * epsilon * vector)
        gradients.append(
            torch.autograd.grad(
                _scalar_loss(tiny_mlp, x, target), parameters, allow_unused=True
            )
        )
    with torch.no_grad():
        for parameter, original in zip(parameters, originals, strict=True):
            parameter.copy_(original)
    for plus, minus, parameter in zip(*gradients, parameters, strict=True):
        if plus is None:
            finite_difference.append(torch.zeros_like(parameter))
        else:
            finite_difference.append((plus - minus) / (2 * epsilon))
    for observed, reference in zip(actual, finite_difference, strict=True):
        # Central differences have an O(epsilon^2) truncation error budget.
        torch.testing.assert_close(observed, reference, rtol=2e-3, atol=2e-3)


def test_hvp_true_validates_inputs(tiny_mlp):
    params = tuple(tiny_mlp.parameters())
    vectors = tuple(torch.zeros_like(p) for p in params)
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(0))
    per_row = tiny_mlp(x).sum(dim=1)
    with pytest.raises(ValueError, match="scalar"):
        hvp_true(per_row, params, vectors)
    with pytest.raises(ValueError, match="vectors"):
        hvp_true(per_row.sum(), params, vectors[:-1])
    with pytest.raises(ValueError, match="shape"):
        hvp_true(per_row.sum(), params, tuple(reversed(vectors)))


def test_ggn_equals_true_hessian_for_linear_softmax_model():
    torch.manual_seed(7)
    model = LinearCausalLM(vocab_size=5, num_logits=7, scale=1 / 4)
    input_ids = torch.tensor([[1, 3, 2, 4]], dtype=torch.int64)
    target_mask = torch.tensor([[False, True, True, True]])
    vector = torch.randn_like(model.weight)
    params = {"weight": model.weight}
    ggn = ggn_vector_product(
        model, params, {"weight": vector}, input_ids, target_mask
    )["weight"]

    logits = model(input_ids).logits[:, :-1]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    true = hvp_true(loss, (model.weight,), (vector,))[0]
    # The two autograd constructions differ only by fp32 rounding here.
    torch.testing.assert_close(ggn, true, rtol=2e-5, atol=2e-6)


def test_true_hessian_and_ggn_are_distinct_measurements():
    """True Hessian and GGN are separate primitives: equal only without curvature
    in the network, and only the GGN is guaranteed PSD (what SOURCE requires)."""
    model = TanhLM()
    input_ids = torch.tensor([[1, 3, 2, 4, 5]], dtype=torch.int64)
    target_mask = torch.tensor([[False, True, True, True, True]])
    generator = torch.Generator().manual_seed(11)
    v = torch.randn(model.mid.weight.shape, generator=generator)
    w = torch.randn(model.mid.weight.shape, generator=generator)
    params = {"mid.weight": model.mid.weight}

    ggn_v = ggn_vector_product(
        model, params, {"mid.weight": v}, input_ids, target_mask
    )["mid.weight"]
    ggn_w = ggn_vector_product(
        model, params, {"mid.weight": w}, input_ids, target_mask
    )["mid.weight"]

    logits = model(input_ids).logits[:, :-1]
    loss = F.cross_entropy(
        logits.reshape(-1, 11), input_ids[:, 1:].reshape(-1), reduction="sum"
    )
    true = hvp_true(loss, (model.mid.weight,), (v,))[0]

    assert torch.isfinite(ggn_v).all() and torch.isfinite(true).all()
    # Through a tanh the linearized GGN drops real curvature terms.
    assert not torch.allclose(true, ggn_v, rtol=1e-3, atol=1e-5)
    # The GGN stays symmetric and PSD along the probe directions.
    torch.testing.assert_close(
        torch.sum(v * ggn_w), torch.sum(w * ggn_v), rtol=1e-4, atol=1e-6
    )
    assert torch.sum(v * ggn_v) >= 0
    assert torch.sum(w * ggn_w) >= 0


def test_ggn_vector_product_validates_inputs():
    torch.manual_seed(7)
    model = LinearCausalLM(vocab_size=5, scale=1 / 4)
    input_ids = torch.tensor([[1, 3, 2]], dtype=torch.int64)
    mask = torch.tensor([[False, True, True]])
    v = torch.randn_like(model.weight)
    with pytest.raises(ValueError, match="bool"):
        ggn_vector_product(
            model, {"weight": model.weight}, {"weight": v}, input_ids, mask.long()
        )
    with pytest.raises(ValueError, match="identical keys"):
        ggn_vector_product(
            model, {"weight": model.weight}, {"other": v}, input_ids, mask
        )
    with pytest.raises(TypeError, match="fp32"):
        ggn_vector_product(
            model, {"weight": model.weight}, {"weight": v.double()}, input_ids, mask
        )
    with pytest.raises(ValueError, match="position 0"):
        ggn_vector_product(
            model,
            {"weight": model.weight},
            {"weight": v},
            input_ids,
            torch.tensor([[True, True, False]]),
        )


# --- pair directions (ported from test_pair_grad.py) ---


def _losses(model):
    x_a = torch.tensor([[0.2, -0.4], [0.7, 0.1]])
    x_b = torch.tensor([[-0.3, 0.9], [0.4, -0.2]])
    return (model(x_a).square().sum(), (model(x_b) - 0.3).square().sum())


def _kernel(model, metric):
    a, b = _losses(model)
    params = tuple(model.parameters())
    ga = torch.autograd.grad(a, params, create_graph=False)
    gb = torch.autograd.grad(b, params, create_graph=False)
    manifest = ParameterManifest.from_model(model, "pair")
    flat_a = flatten_tensors(manifest.included_entries(), ga)
    flat_b = flatten_tensors(manifest.included_entries(), gb)
    return torch.dot(flat_a, flat_b if metric is None else metric.apply(flat_b))


@pytest.mark.parametrize("diagonal", [False, True])
def test_pair_direction_matches_finite_difference(pair_model, diagonal):
    manifest = ParameterManifest.from_model(pair_model, "pair")
    metric = (
        _diag_metric(torch.linspace(0.4, 1.6, manifest.included_numel))
        if diagonal
        else None
    )
    a, b = _losses(pair_model)
    actual = PairGradientBackend(pair_model, manifest).pair_direction(
        a, b, metric=metric
    )
    generator = torch.Generator().manual_seed(5)
    direction = torch.randn(manifest.included_numel, generator=generator)
    finite = _finite_difference(
        pair_model, direction, lambda: _kernel(pair_model, metric)
    )
    torch.testing.assert_close(
        torch.dot(actual, direction), finite, rtol=4e-3, atol=2e-3
    )


def test_symmetry_identity_plain_path_and_decomposition(pair_model):
    manifest = ParameterManifest.from_model(pair_model, "pair")
    backend = PairGradientBackend(pair_model, manifest)
    a, b = _losses(pair_model)
    actual = backend.pair_direction(a, b, metric=None)
    a2, b2 = _losses(pair_model)
    reverse = backend.pair_direction(b2, a2, metric=None)
    torch.testing.assert_close(actual, reverse, rtol=1e-6, atol=1e-7)

    a3, b3 = _losses(pair_model)
    params = tuple(pair_model.parameters())
    ga = torch.autograd.grad(a3, params, create_graph=True, retain_graph=True)
    gb = torch.autograd.grad(b3, params, create_graph=True, retain_graph=True)
    plain = flatten_tensors(
        manifest.included_entries(),
        torch.autograd.grad(
            sum((x * y).sum() for x, y in zip(ga, gb, strict=True)), params
        ),
    )
    torch.testing.assert_close(actual, plain, rtol=0, atol=0)

    a4, b4 = _losses(pair_model)
    halves = backend.pair_direction(a4, b4, metric=None, decompose=True)
    torch.testing.assert_close(halves[0] + halves[1], actual, rtol=1e-6, atol=1e-7)
    # A deliberately wrong recombination (one Hessian half doubled) must fail:
    # this keeps the decomposition assertion sharp enough to catch a dropped
    # or duplicated term.
    assert not torch.allclose(2 * halves[0], actual, rtol=1e-6, atol=1e-7)
    assert not torch.allclose(2 * halves[1], actual, rtol=1e-6, atol=1e-7)

    self_loss, _ = _losses(pair_model)
    self_actual = self_direction(self_loss, None, backend=backend)
    self_loss_2, _ = _losses(pair_model)
    self_pair = backend.pair_direction(self_loss_2, self_loss_2, metric=None)
    torch.testing.assert_close(self_actual, self_pair, rtol=1e-6, atol=1e-7)


def test_graph_connected_metric_state_is_rejected_and_matters(pair_model):
    manifest = ParameterManifest.from_model(pair_model, "pair")
    params = tuple(pair_model.parameters())
    a, b = _losses(pair_model)
    ga = torch.autograd.grad(a, params, create_graph=True, retain_graph=True)
    gb = torch.autograd.grad(b, params, create_graph=True, retain_graph=True)
    entries = manifest.included_entries()
    live_diag = flatten_tensors(entries, ga).square() + 1
    assert live_diag.requires_grad
    with pytest.raises(ValueError, match="detached"):
        DiagonalMetric("diag_precond", 1.0, 0.0, None, "live", live_diag)
    frozen = DiagonalMetric(
        "diag_precond", 1.0, 0.0, None, "frozen", live_diag.detach()
    )
    frozen_result = PairGradientBackend(pair_model, manifest).pair_direction(
        *_losses(pair_model), metric=frozen
    )
    # Deliberately bypass the metric to demonstrate the omitted dM/dtheta term.
    live_scalar = torch.dot(
        flatten_tensors(entries, ga), live_diag * flatten_tensors(entries, gb)
    )
    live_result = flatten_tensors(entries, torch.autograd.grad(live_scalar, params))
    assert not torch.allclose(frozen_result, live_result)


def test_pair_direction_rejects_non_diagonal_metric_objects(pair_model):
    manifest = ParameterManifest.from_model(pair_model, "pair")
    backend = PairGradientBackend(pair_model, manifest)
    with pytest.raises(TypeError, match="DiagonalMetric"):
        backend.pair_direction(
            *_losses(pair_model), metric=torch.ones(manifest.included_numel)
        )
    # metric is a required keyword (as upstream): omitting it must error
    # rather than silently computing identity-metric directions.
    with pytest.raises(TypeError, match="metric"):
        backend.pair_direction(*_losses(pair_model))


def test_ggn_equals_true_for_linear_model():
    torch.manual_seed(3)
    model = LinearCausalLM(vocab_size=5, scale=1 / 3)
    manifest = ParameterManifest.from_model(model, "linear")
    backend = PairGradientBackend(model, manifest)
    ids_a = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    ids_b = torch.tensor([[2, 3, 4]], dtype=torch.int64)
    mask = torch.tensor([[False, True, True]])
    batch_a = TokenizedBatch(ids_a, torch.tensor([0]), mask)
    batch_b = TokenizedBatch(ids_b, torch.tensor([1]), mask)

    def loss(ids):
        logits = model(ids).logits[:, :-1]
        return F.cross_entropy(
            logits.reshape(-1, 5), ids[:, 1:].reshape(-1), reduction="sum"
        )

    true = backend.pair_direction(loss(ids_a), loss(ids_b), metric=None)
    ggn = backend.pair_direction(
        loss(ids_a),
        loss(ids_b),
        metric=None,
        hessian_kind="ggn",
        batch_a=batch_a,
        batch_b=batch_b,
    )
    torch.testing.assert_close(ggn, true, rtol=2e-5, atol=2e-6)


def test_ggn_self_direction_computes_one_half_and_doubles():
    torch.manual_seed(3)
    model = LinearCausalLM(vocab_size=5, scale=1 / 3)
    manifest = ParameterManifest.from_model(model, "linear")
    backend = PairGradientBackend(model, manifest)
    ids = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    mask = torch.tensor([[False, True, True]])
    batch = TokenizedBatch(ids, torch.tensor([0]), mask)
    logits = model(ids).logits[:, :-1]
    loss = F.cross_entropy(
        logits.reshape(-1, 5), ids[:, 1:].reshape(-1), reduction="sum"
    )

    actual = backend.self_direction(loss, None, hessian_kind="ggn", batch_a=batch)

    logits = model(ids).logits[:, :-1]
    loss = F.cross_entropy(
        logits.reshape(-1, 5), ids[:, 1:].reshape(-1), reduction="sum"
    )
    gradient = backend._gradient(loss, create_graph=False)
    expected_half = backend._ggn_half(batch, gradient, None)
    torch.testing.assert_close(actual, 2 * expected_half)


def test_differentiate_products_all_none_returns_zero():
    model = nn.Linear(2, 1)
    manifest = ParameterManifest.from_model(model, "disjoint-gradients")
    backend = PairGradientBackend(model, manifest)
    loss_a = (model.weight.square()).sum()
    loss_b = (model.bias + 1).square().sum()

    actual = backend.pair_direction(loss_a, loss_b, metric=None)

    torch.testing.assert_close(actual, torch.zeros(manifest.included_numel))


# --- metric derivative (ported from test_metric_derivative.py) ---


def _batch_losses(model, batch_inputs):
    return [model(x).square().sum() for x in batch_inputs]


def _gradient_stack(model, batch_inputs, entries):
    params = tuple(model.parameters())
    grads = []
    for loss in _batch_losses(model, batch_inputs):
        grads.append(flatten_tensors(entries, torch.autograd.grad(loss, params)))
    return torch.stack(grads)


def _factored_reconstruct(v, entries):
    result = v.clone()
    for entry in entries:
        if len(entry.shape) != 2:
            continue
        matrix = v.narrow(0, entry.global_flat_offset, entry.numel).reshape(
            entry.shape
        )
        reconstructed = (
            torch.outer(matrix.sum(dim=1), matrix.sum(dim=0)) / matrix.sum()
        )
        result.narrow(0, entry.global_flat_offset, entry.numel).copy_(
            reconstructed.flatten()
        )
    return result


def test_diag_probe_matches_spec_sign():
    pair_product = torch.tensor([0.5, -0.2, 1.0])
    v = torch.tensor([0.1, 0.4, 0.9])
    probe = metric_probe(pair_product, v, exponent=-1.0, eps=EPS)
    torch.testing.assert_close(probe, -pair_product / (v + EPS).square())


@pytest.mark.parametrize("exponent", [-1.0, -0.5])
def test_factored_probe_matches_autograd_through_factoring(exponent):
    torch.manual_seed(4)
    layer = nn.Linear(4, 3, bias=False)
    manifest = ParameterManifest.from_model(layer, "factored-probe")
    entries = manifest.included_entries()
    statistic = torch.rand(3, 4) + 0.5
    pair_product = torch.randn(3, 4)

    leaf = statistic.clone().requires_grad_(True)
    row, col, total = leaf.sum(dim=1), leaf.sum(dim=0), leaf.sum()
    v_hat = row[:, None] * col[None, :] / total
    kernel = (pair_product * (v_hat + EPS).pow(exponent)).sum()
    expected = torch.autograd.grad(kernel, leaf)[0].flatten()

    actual = metric_probe(
        pair_product.flatten(),
        v_hat.detach().flatten(),
        entries,
        exponent=exponent,
        eps=EPS,
        factored=True,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_factored_probe_rank1_euler_homogeneity():
    torch.manual_seed(9)
    layer = nn.Linear(5, 3, bias=False)
    manifest = ParameterManifest.from_model(layer, "euler")
    entries = manifest.included_entries()
    statistic = torch.outer(torch.rand(3) + 0.2, torch.rand(5) + 0.2)
    pair_product = torch.randn(3, 5)

    probe = metric_probe(
        pair_product.flatten(),
        statistic.flatten(),
        entries,
        exponent=-1.0,
        eps=0.0,
        factored=True,
    )
    kernel = (pair_product / statistic).sum()
    # V_hat^{-1} is degree -1 in V, so sum(w * V) = -k for exactly rank-1 V.
    torch.testing.assert_close(
        torch.dot(probe, statistic.flatten()), -kernel, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("factored", [False, True])
def test_metric_term_matches_finite_difference_uncentered(
    model, batch_inputs, factored
):
    manifest = ParameterManifest.from_model(model, "third-term")
    entries = manifest.included_entries()
    backend = MetricDerivativeBackend(model, manifest)

    stack = _gradient_stack(model, batch_inputs, entries)
    gradient_x, gradient_y = stack[0], stack[1]
    pair_product = gradient_x * gradient_y

    def statistic():
        raw = _gradient_stack(model, batch_inputs, entries).square().mean(dim=0)
        return _factored_reconstruct(raw, entries) if factored else raw

    actual = backend.metric_term(
        gradient_x,
        gradient_y,
        _batch_losses(model, batch_inputs),
        v=statistic(),
        exponent=-1.0,
        eps=EPS,
        factored=factored,
    )

    def frozen_kernel():
        return torch.dot(pair_product, (statistic() + EPS).pow(-1.0))

    generator = torch.Generator().manual_seed(5)
    direction = torch.randn(manifest.included_numel, generator=generator)
    finite = _finite_difference(model, direction, frozen_kernel)
    torch.testing.assert_close(
        torch.dot(actual, direction), finite, rtol=4e-3, atol=2e-3
    )


def test_total_kernel_derivative_needs_metric_term(model, batch_inputs):
    """Red-then-green: the frozen-metric rotation terms alone underpredict the
    finite-difference derivative of the full theta-dependent kernel; adding the
    metric-derivative term closes the gap."""
    manifest = ParameterManifest.from_model(model, "total-derivative")
    entries = manifest.included_entries()
    backend = MetricDerivativeBackend(model, manifest)
    params = tuple(model.parameters())

    def statistic():
        return _gradient_stack(model, batch_inputs, entries).square().mean(dim=0)

    v0 = statistic().detach()
    metric = DiagonalMetric(
        "diag_precond", -1.0, EPS, None, "theta0", (v0 + EPS).pow(-1.0)
    )
    losses = _batch_losses(model, batch_inputs)
    rotation = backend.pair_direction(losses[0], losses[1], metric=metric)
    stack0 = _gradient_stack(model, batch_inputs, entries)
    third = backend.metric_term(
        stack0[0], stack0[1], _batch_losses(model, batch_inputs), v=v0,
        exponent=-1.0, eps=EPS,
    )

    def full_kernel():
        gx = flatten_tensors(
            entries,
            torch.autograd.grad(_batch_losses(model, batch_inputs)[0], params),
        )
        gy = flatten_tensors(
            entries,
            torch.autograd.grad(_batch_losses(model, batch_inputs)[1], params),
        )
        return torch.dot(gx * gy, (statistic() + EPS).pow(-1.0))

    generator = torch.Generator().manual_seed(5)
    direction = torch.randn(manifest.included_numel, generator=generator)
    finite = _finite_difference(model, direction, full_kernel)

    # Green: the complete first-order prediction matches the FD derivative.
    torch.testing.assert_close(
        torch.dot(rotation + third, direction), finite, rtol=4e-3, atol=2e-3
    )
    # Red: without the metric-derivative term the same tolerance must fail,
    # so this test is tight enough to catch the missing term.
    rotation_only = torch.dot(rotation, direction)
    assert not torch.isclose(rotation_only, finite, rtol=4e-3, atol=2e-3)
    assert (finite - rotation_only).abs() > 20 * (
        2e-3 + 4e-3 * finite.abs()
    ), "metric-derivative contribution too small to make the red case meaningful"


def test_m_vp_centered_matches_finite_difference_same_batch(model, batch_inputs):
    manifest = ParameterManifest.from_model(model, "third-term-centered")
    entries = manifest.included_entries()
    backend = MetricDerivativeBackend(model, manifest)
    generator = torch.Generator().manual_seed(11)
    probe = torch.randn(manifest.included_numel, generator=generator)

    actual = backend.m_vp(
        probe,
        _batch_losses(model, batch_inputs),
        losses_center=_batch_losses(model, batch_inputs),
    )

    def centered_statistic_dot_probe():
        stack = _gradient_stack(model, batch_inputs, entries)
        centered = stack.square().mean(dim=0) - stack.mean(dim=0).square()
        return torch.dot(probe, centered)

    direction = torch.randn(manifest.included_numel, generator=generator)
    finite = _finite_difference(model, direction, centered_statistic_dot_probe)
    torch.testing.assert_close(
        torch.dot(actual, direction), finite, rtol=4e-3, atol=2e-3
    )


def test_centered_m_vp_vanishes_for_data_independent_hessian():
    torch.manual_seed(2)
    model = nn.Linear(3, 2)
    manifest = ParameterManifest.from_model(model, "constant-hessian")
    backend = MetricDerivativeBackend(model, manifest)
    targets = [torch.randn(2, 3) for _ in range(3)]

    def losses():
        return [
            0.5 * ((model.weight - t).square().sum() + model.bias.square().sum())
            for t in targets
        ]

    probe = torch.randn(manifest.included_numel)
    actual = backend.m_vp(probe, losses(), losses_center=losses())
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=1e-6)


def test_m_vp_validates_inputs(model, batch_inputs):
    manifest = ParameterManifest.from_model(model, "validation")
    backend = MetricDerivativeBackend(model, manifest)
    with pytest.raises(ValueError, match="probe must be flat"):
        backend.m_vp(torch.zeros(3), _batch_losses(model, batch_inputs))
    with pytest.raises(ValueError, match="losses must be non-empty"):
        backend.m_vp(torch.zeros(manifest.included_numel), [])
    with pytest.raises(ValueError, match="losses_center must be non-empty"):
        backend.m_vp(
            torch.zeros(manifest.included_numel),
            _batch_losses(model, batch_inputs),
            losses_center=[],
        )
    with pytest.raises(ValueError, match="eps must be nonnegative"):
        metric_probe(torch.zeros(2), torch.zeros(2), eps=-1.0)
    with pytest.raises(ValueError, match="factored probe requires manifest entries"):
        metric_probe(torch.zeros(2), torch.ones(2), factored=True)


# --- jvp_sweep (ported from test_jvp_sweep.py, on a tiny local model) ---


def _jvp_batch():
    return TokenizedBatch(
        input_ids=torch.tensor(
            [[1, 12, 3, 4, 5, 6], [2, 7, 8, 9, 10, 11]], dtype=torch.int64
        ),
        sequence_ids=torch.tensor([3, 9]),
        target_mask=torch.tensor(
            [
                [False, True, False, True, False, True],
                [False, False, True, False, True, False],
            ]
        ),
    )


def test_jvp_sweep_matches_serial_gradient_rows_and_finite_difference():
    model = TinyLM()
    manifest = ParameterManifest.from_model(model, "tiny")
    generator = torch.Generator().manual_seed(3)
    directions = torch.randn(2, manifest.included_numel, generator=generator)
    actual = jvp_sweep(model, manifest, _jvp_batch(), directions)
    assert actual.shape == (5, 2)
    assert torch.isfinite(actual).all()

    adapter = CausalLMLossAdapter(model)
    loss_batch = adapter.per_datapoint_losses(_jvp_batch())
    rows = SerialGradientBackend(model, manifest).rows(loss_batch.losses)
    torch.testing.assert_close(actual, rows @ directions.T, rtol=1e-4, atol=1e-5)

    entries = manifest.included_entries()
    pieces = unflatten_vector(directions[0], entries)
    named = dict(model.named_parameters())
    originals = {entry.name: named[entry.name].detach().clone() for entry in entries}
    # A moderately large step avoids catastrophic cancellation in fp32 CE.
    epsilon = 1e-2
    losses = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            for entry, piece in zip(entries, pieces, strict=True):
                named[entry.name].copy_(originals[entry.name] + sign * epsilon * piece)
        losses.append(adapter.per_datapoint_losses(_jvp_batch()).losses.detach())
    with torch.no_grad():
        for entry in entries:
            named[entry.name].copy_(originals[entry.name])
    finite_difference = (losses[0] - losses[1]) / (2 * epsilon)
    # atol loosened from upstream's 2e-4 to 2e-3: fixture-driven only — the
    # pythia-14m fixture was replaced by TinyLM, whose fp32 CE truncation error
    # fails 2e-4 for correct code; 2e-3 still catches a sign/off-by-one mutation.
    torch.testing.assert_close(actual[:, 0], finite_difference, rtol=1e-3, atol=2e-3)


def test_jvp_sweep_reductions_empty_directions_and_validation():
    model = TinyLM()
    manifest = ParameterManifest.from_model(model, "tiny")
    generator = torch.Generator().manual_seed(4)
    directions = torch.randn(3, manifest.included_numel, generator=generator)

    per_sequence = jvp_sweep(
        model, manifest, _jvp_batch(), directions, reduction="per_sequence_mean"
    )
    assert per_sequence.shape == (2, 3)
    adapter = CausalLMLossAdapter(model, reduction="per_sequence_mean")
    rows = SerialGradientBackend(model, manifest).rows(
        adapter.per_datapoint_losses(_jvp_batch()).losses
    )
    torch.testing.assert_close(per_sequence, rows @ directions.T, rtol=1e-4, atol=1e-5)

    empty = jvp_sweep(
        model, manifest, _jvp_batch(), torch.empty(0, manifest.included_numel)
    )
    assert empty.shape == (5, 0) and empty.dtype == torch.float32
    empty_sequence = jvp_sweep(
        model,
        manifest,
        _jvp_batch(),
        torch.empty(0, manifest.included_numel),
        reduction="per_sequence_sum",
    )
    assert empty_sequence.shape == (2, 0)

    with pytest.raises(ValueError, match="reduction"):
        jvp_sweep(model, manifest, _jvp_batch(), directions, reduction="bogus")
    with pytest.raises(ValueError, match="directions"):
        jvp_sweep(model, manifest, _jvp_batch(), directions[:, :-1])
    with pytest.raises(TypeError, match="fp32"):
        jvp_sweep(model, manifest, _jvp_batch(), directions.double())
