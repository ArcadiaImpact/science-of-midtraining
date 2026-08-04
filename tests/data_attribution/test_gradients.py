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
    assert BatchedVJPBackend(model, manifest).rows(losses, chunk_size=2).shape[0] == 7


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
