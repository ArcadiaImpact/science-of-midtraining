import pytest
import torch

from scimt.data_attribution.gradients import BatchedVJPBackend, SerialGradientBackend
from scimt.data_attribution.manifest import ParameterManifest, flatten_tensors


@pytest.mark.parametrize("backend_cls", [SerialGradientBackend, BatchedVJPBackend])
def test_gradient_rows_match_explicit_autograd(backend_cls):
    torch.manual_seed(2)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    manifest = ParameterManifest.from_model(model, "mlp")
    x = torch.randn(5, 3)
    losses = model(x).square().mean(1)
    actual = backend_cls(model, manifest).rows(losses)
    losses = model(x).square().mean(1)
    params = [p for _, p in model.named_parameters()]
    expected = torch.stack([flatten_tensors(manifest.included_entries(), torch.autograd.grad(losses[i], params, retain_graph=i < 4)) for i in range(5)])
    assert actual.dtype == torch.float32 and actual.shape == (5, manifest.included_numel)
    torch.testing.assert_close(actual, expected)
