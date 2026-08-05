import pytest
import torch

from scimt.data_attribution.gradients import BatchedVJPBackend, SerialGradientBackend
from scimt.data_attribution.manifest import ParameterManifest, flatten_tensors


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_gradient_rows_match_explicit_autograd(backend_cls):
    torch.manual_seed(2)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    )
    manifest = ParameterManifest.from_model(model, "mlp")
    x = torch.randn(5, 3)
    losses = model(x).square().mean(1)
    actual = backend_cls(model, manifest).rows(losses)
    losses = model(x).square().mean(1)
    params = [p for _, p in model.named_parameters()]
    expected = torch.stack(
        [
            flatten_tensors(
                manifest.included_entries(),
                torch.autograd.grad(losses[i], params, retain_graph=i < 4),
            )
            for i in range(5)
        ]
    )
    assert actual.dtype == torch.float32 and actual.shape == (
        5,
        manifest.included_numel,
    )
    torch.testing.assert_close(actual, expected)


def test_batched_uses_chunk_sized_cotangents(monkeypatch):
    model = torch.nn.Linear(3, 2)
    manifest = ParameterManifest.from_model(model, "linear")
    losses = model(torch.randn(7, 3)).square().mean(1)
    monkeypatch.setattr(
        torch, "eye", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no eye"))
    )
    original_zeros = torch.zeros
    shapes = []

    def recording_zeros(*args, **kwargs):
        shape = args[0]
        shapes.append(tuple(shape) if not isinstance(shape, int) else (shape,))
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", recording_zeros)
    assert BatchedVJPBackend(model, manifest).rows(losses, chunk_size=2).shape[0] == 7
    assert (7, 7) not in shapes
    assert (2, 7) in shapes


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_unused_parameters_zero_fill_and_inputs_validate(backend_cls):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(2, 1)
            self.unused = torch.nn.Parameter(torch.ones(3))

        def forward(self, x):
            return self.used(x)

    model = Model()
    manifest = ParameterManifest.from_model(model, "unused")
    backend = backend_cls(model, manifest)
    rows = backend.rows(model(torch.randn(3, 2)).flatten().square(), chunk_size=2)
    unused = manifest.included_entries()[0]
    assert torch.count_nonzero(rows[:, unused.global_flat_offset : unused.numel]) == 0
    with pytest.raises(ValueError, match="shape"):
        backend.rows(torch.tensor(1.0))
    with pytest.raises(ValueError, match="positive"):
        backend.rows(model(torch.randn(3, 2)).flatten(), chunk_size=0)


def test_serial_matches_explicit_per_loss_jacobian_exactly():
    torch.manual_seed(8)
    model = torch.nn.Linear(3, 2)
    manifest = ParameterManifest.from_model(model, "linear")
    x = torch.randn(4, 3)
    losses = model(x).square().mean(1)
    actual = SerialGradientBackend(model, manifest).rows(losses, chunk_size=2)
    losses = model(x).square().mean(1)
    params = tuple(model.parameters())
    expected = torch.stack(
        [
            flatten_tensors(
                manifest.included_entries(),
                torch.autograd.grad(losses[row], params, retain_graph=row < 3),
            )
            for row in range(4)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_backends_do_not_touch_param_grad_fields(backend_cls):
    model = torch.nn.Linear(3, 2)
    sentinels = [
        torch.full_like(parameter, i + 1.0)
        for i, parameter in enumerate(model.parameters())
    ]
    for parameter, sentinel in zip(model.parameters(), sentinels, strict=True):
        parameter.grad = sentinel
    manifest = ParameterManifest.from_model(model, "linear")
    backend_cls(model, manifest).rows(
        model(torch.randn(4, 3)).square().mean(1), chunk_size=2
    )
    for parameter, sentinel in zip(model.parameters(), sentinels, strict=True):
        assert parameter.grad is sentinel


def test_empty_gradient_rows_follow_loss_device():
    model = torch.nn.Linear(2, 1)
    manifest = ParameterManifest.from_model(model, "linear")
    losses = torch.empty(0, device="meta")
    for backend in (
        SerialGradientBackend(model, manifest),
        BatchedVJPBackend(model, manifest),
    ):
        assert backend.rows(losses).device == losses.device


def test_batched_matches_serial_tiny_lm():
    from .fixtures import TinyLM

    model = TinyLM()
    manifest = ParameterManifest.from_model(model, "tiny-lm")
    ids = torch.tensor([[1, 2, 3, 4]])
    logits = model(ids).logits.float()
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    )
    expected = SerialGradientBackend(model, manifest).rows(losses, chunk_size=2)
    logits = model(ids).logits.float()
    losses = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="none",
    )
    actual = BatchedVJPBackend(model, manifest).rows(losses, chunk_size=2)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_frozen_included_parameters_are_zero_filled(backend_cls):
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    manifest = ParameterManifest.from_model(model, "partly-frozen")
    losses = model(torch.randn(3, 2)).square().mean(1)
    rows = backend_cls(model, manifest).rows(losses, chunk_size=2)
    bias = next(e for e in manifest.included_entries() if e.name == "bias")
    assert (
        torch.count_nonzero(
            rows[:, bias.global_flat_offset : bias.global_flat_offset + bias.numel]
        )
        == 0
    )
    assert torch.count_nonzero(rows[:, : bias.global_flat_offset]) > 0


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_empty_manifest_returns_n_by_zero_fp32_on_loss_device(backend_cls):
    model = torch.nn.Linear(2, 1)
    manifest = ParameterManifest.from_model(model, "none", include=[r"does-not-match"])
    losses = model(torch.randn(4, 2)).flatten().square()
    rows = backend_cls(model, manifest).rows(losses)
    assert rows.shape == (4, 0) and rows.dtype == torch.float32
    assert rows.device == losses.device
