import pytest
import torch

from scimt.data_attribution.manifest import (
    ParameterManifest, flatten_tensors, freeze_excluded, unflatten_vector,
)


def test_manifest_is_deterministic_and_round_trips_flat_coordinates():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    manifest = ParameterManifest.from_model(model, "tiny", include=[r".*weight"], exclude=[r"1.weight"])
    assert [e.name for e in manifest.included_entries()] == ["0.weight"]
    assert manifest.digest() == ParameterManifest.from_model(model, "tiny", [r".*weight"], [r"1.weight"]).digest()
    flat = flatten_tensors(manifest.included_entries(), [torch.arange(6).reshape(2, 3)])
    assert flat.dtype == torch.float32
    assert torch.equal(unflatten_vector(flat, manifest.included_entries())[0], flat.reshape(2, 3))


def test_manifest_validates_patterns_and_freezes_excluded():
    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="invalid include regex"):
        ParameterManifest.from_model(model, "tiny", ["["], [])
    manifest = ParameterManifest.from_model(model, "tiny", [r"weight"], [])
    freeze_excluded(model, manifest)
    assert model.weight.requires_grad and not model.bias.requires_grad
