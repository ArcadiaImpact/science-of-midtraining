import json
import pytest
import torch

from scimt.data_attribution.manifest import (
    ManifestMismatchError,
    ParameterManifest,
    flatten_tensors,
    freeze_excluded,
    unflatten_vector,
)


def test_manifest_is_deterministic_and_round_trips_flat_coordinates():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    manifest = ParameterManifest.from_model(
        model, "tiny", include=[r".*weight"], exclude=[r"1.weight"]
    )
    assert [e.name for e in manifest.included_entries()] == ["0.weight"]
    assert (
        manifest.digest()
        == ParameterManifest.from_model(
            model, "tiny", [r".*weight"], [r"1.weight"]
        ).digest()
    )
    flat = flatten_tensors(manifest.included_entries(), [torch.arange(6).reshape(2, 3)])
    assert flat.dtype == torch.float32
    assert torch.equal(
        unflatten_vector(flat, manifest.included_entries())[0], flat.reshape(2, 3)
    )


def test_manifest_validates_patterns_and_freezes_excluded():
    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="invalid include regex"):
        ParameterManifest.from_model(model, "tiny", ["["], [])
    manifest = ParameterManifest.from_model(model, "tiny", [r"weight"], [])
    freeze_excluded(model, manifest)
    assert model.weight.requires_grad and not model.bias.requires_grad


class TiedModel(torch.nn.Module):
    def __init__(self, tied=True):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 5)
        self.head = torch.nn.Linear(5, 11, bias=False)
        if tied:
            self.head.weight = self.embed.weight


def test_ties_count_once_and_structural_drift_is_rejected():
    model = TiedModel()
    manifest = ParameterManifest.from_model(model, "tied")
    assert manifest.included_numel == model.embed.weight.numel()
    assert manifest.entries[1].shared_parameter_id == "embed.weight"
    with pytest.raises(ManifestMismatchError, match="tie mismatch"):
        manifest.validate_against_model(TiedModel(tied=False))
    with pytest.raises(ManifestMismatchError, match="name mismatch"):
        ParameterManifest.from_model(
            torch.nn.ModuleDict({"a": torch.nn.Linear(2, 2)}), "x"
        ).validate_against_model(torch.nn.ModuleDict({"b": torch.nn.Linear(2, 2)}))


def test_manifest_json_is_canonical_and_persistence_checks_digest(tmp_path):
    manifest = ParameterManifest.from_model(torch.nn.Linear(2, 2), "tiny")
    noncanonical = json.dumps(json.loads(manifest.to_json()), indent=4)
    assert ParameterManifest.from_json(noncanonical).digest() == manifest.digest()
    manifest.save(tmp_path)
    assert ParameterManifest.load(tmp_path) == manifest
    (tmp_path / "parameter_manifest.sha256").write_text("0" * 64)
    with pytest.raises(ManifestMismatchError, match="digest mismatch"):
        ParameterManifest.load(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"entries": [], "model_name": 3},
        {"entries": {}, "model_name": "x"},
        {"entries": [{"name": "incomplete"}], "model_name": "x"},
        {
            "entries": [
                {
                    "name": "x",
                    "shape": "bad",
                    "numel": 1,
                    "global_flat_offset": 0,
                    "dtype_at_load": "torch.float32",
                    "requires_grad": True,
                    "included": True,
                    "exclusion_reason": None,
                    "shared_parameter_id": None,
                }
            ],
            "model_name": "x",
        },
    ],
)
def test_from_json_rejects_malformed_manifest_variants(payload):
    with pytest.raises(ValueError, match="invalid parameter manifest"):
        ParameterManifest.from_json(json.dumps(payload))


def test_manifest_semantic_invariants_are_validated_on_load():
    manifest = ParameterManifest.from_model(torch.nn.Linear(2, 1), "linear")
    payload = json.loads(manifest.to_json())
    payload["entries"][0]["global_flat_offset"] = 3
    with pytest.raises(ValueError, match="included entry"):
        ParameterManifest.from_json(json.dumps(payload))


def test_model_validation_checks_dtype_but_allows_runtime_freezing():
    model = torch.nn.Linear(2, 1)
    manifest = ParameterManifest.from_model(model, "linear")
    model.weight.requires_grad_(False)
    manifest.validate_against_model(model)
    with pytest.raises(ManifestMismatchError, match="dtype mismatch"):
        manifest.validate_against_model(model.double())
