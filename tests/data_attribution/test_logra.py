import pytest
import hashlib
import json
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from scimt.data_attribution.logra import (
    LogRaFisherState,
    ProjectionArtifacts,
    inject_logra,
    pca_projections,
    projection_descriptor,
    whiten_rows,
)
from scimt.data_attribution.manifest import ParameterManifest
from .test_ekfac import make_artifact


def test_random_injection_preserves_forward_aliases_and_refuses_twice(tmp_path):
    model = nn.Module()
    model.left = nn.Linear(3, 2)
    model.right = model.left
    x = torch.randn(4, 3)
    expected = model.left(x).detach()
    report = inject_logra(model, rank=8, seed=7, targets="right")
    assert model.left is model.right and model.left.rank == 2
    torch.testing.assert_close(model.left(x), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="applied twice"):
        inject_logra(model, rank=2, seed=8)
    manifest = ParameterManifest.from_model(
        model, "projected", include=[report.include_regex]
    )
    descriptor = projection_descriptor(report)
    artifact = ProjectionArtifacts.save(
        model, tmp_path, manifest=manifest, descriptor=descriptor
    )
    fresh = nn.Module()
    fresh.left = nn.Linear(3, 2)
    fresh.right = fresh.left
    loaded = ProjectionArtifacts.load(
        tmp_path, expected_manifest=manifest, expected_descriptor=descriptor
    )
    restored = inject_logra(fresh, rank=2, seed=0, init="artifact", projections=loaded)
    assert artifact.content_digest and artifact.manifest_digest == manifest.digest()
    assert restored.projection_digest == report.projection_digest


def test_projection_artifact_rejects_descriptor_manifest_and_content_drift(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    report = inject_logra(model, rank=2, seed=4)
    manifest = ParameterManifest.from_model(
        model, "projected", include=[report.include_regex]
    )
    descriptor = projection_descriptor(report)
    ProjectionArtifacts.save(model, tmp_path, manifest=manifest, descriptor=descriptor)
    with pytest.raises(ValueError, match="descriptor mismatch"):
        ProjectionArtifacts.load(
            tmp_path,
            expected_manifest=manifest,
            expected_descriptor={**descriptor, "seed": 9},
        )
    other = ParameterManifest.from_model(nn.Linear(1, 1), "other")
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        ProjectionArtifacts.load(
            tmp_path, expected_manifest=other, expected_descriptor=descriptor
        )
    metadata = tmp_path / "logra_projections.json"
    metadata.write_text(metadata.read_text().replace('"seed": 4', '"seed": 5'))
    with pytest.raises(ValueError, match="descriptor mismatch|content digest mismatch"):
        ProjectionArtifacts.load(
            tmp_path, expected_manifest=manifest, expected_descriptor=descriptor
        )


def test_projection_descriptor_schema_matches_tensor_modules_and_digest(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    report = inject_logra(model, rank=2, seed=4)
    manifest = ParameterManifest.from_model(
        model, "projected", include=[report.include_regex]
    )
    descriptor = projection_descriptor(report)
    for index, bad in enumerate(
        (
            {**descriptor, "projection_digest": "false"},
            {**descriptor, "wrapped_modules": ["other"]},
            {**descriptor, "rank": True},
            {key: value for key, value in descriptor.items() if key != "include_regex"},
        )
    ):
        with pytest.raises(ValueError, match="descriptor"):
            ProjectionArtifacts.save(
                model, tmp_path / str(index), manifest=manifest, descriptor=bad
            )

    ProjectionArtifacts.save(
        model, tmp_path / "valid", manifest=manifest, descriptor=descriptor
    )
    directory = tmp_path / "valid"
    metadata_path = directory / "logra_projections.json"
    metadata = json.loads(metadata_path.read_text())
    bad_descriptor = {**descriptor, "wrapped_modules": ["other"]}
    metadata["descriptor"] = bad_descriptor
    clean = {key: value for key, value in metadata.items() if key != "content_digest"}
    metadata["content_digest"] = hashlib.sha256(
        (directory / "logra_projections.safetensors").read_bytes()
        + json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="wrapped_modules"):
        ProjectionArtifacts.load(
            directory, expected_manifest=manifest, expected_descriptor=bad_descriptor
        )


def test_pca_truncates_bias_coordinate_and_qr(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3))
    manifest = make_artifact(tmp_path, model)
    A, C = pca_projections(tmp_path, ["0"], 2, manifest)["0"]
    torch.testing.assert_close(A @ A.T, torch.eye(2), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(C.T @ C, torch.eye(2), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="rank 4 exceeds dimensions"):
        pca_projections(tmp_path, ["0"], 4, manifest)


def test_projected_fisher_whitening_matches_dense_math():
    rows = torch.tensor([[1.0, 2.0, -1.0, 0.5, 2.0], [3.0, -2.0, 0.0, 1.5, -1.0]])
    slices = {"first": slice(0, 2), "second": slice(2, 5)}
    state = LogRaFisherState(slices, width=5)
    state.update(rows[:1])
    tail = LogRaFisherState(slices, width=5)
    tail.update(rows[1:])
    state.merge(tail)
    fishers = state.finalize()
    actual = whiten_rows(rows, slices, fishers, damping_scale=0.2)
    expected = rows.double().clone()
    for module, value in slices.items():
        eig, vec = torch.linalg.eigh(fishers[module])
        expected[:, value] = rows[:, value].double() @ (
            (vec * (eig + 0.2 * eig.mean()).pow(-0.5)) @ vec.T
        )
    torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        whiten_rows(actual, slices, fishers, damping_scale=0.2),
        whiten_rows(rows, slices, fishers, damping_scale=0.2, power=-1),
        rtol=3e-5,
        atol=3e-5,
    )


def test_projection_manifest_coordinates():
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    report = inject_logra(model, rank=2, seed=4)
    manifest = ParameterManifest.from_model(
        model, "tiny", include=[report.include_regex]
    )
    assert [e.name for e in manifest.included_entries()] == ["0.logra_B"]
    assert report.projection_digest


def test_rank_seed_effective_rank_and_precomputed_rank_contracts():
    for rank, seed in ((True, 1), (2, False)):
        with pytest.raises(ValueError):
            inject_logra(nn.Sequential(nn.Linear(3, 2)), rank=rank, seed=seed)
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    report = inject_logra(model, rank=8, seed=3)
    assert report.rank == 8 and report.effective_ranks == (("0", 2), ("1", 1))
    with pytest.raises(ValueError, match="does not match requested rank"):
        inject_logra(
            nn.Sequential(nn.Linear(3, 2)),
            rank=2,
            seed=0,
            init="artifact",
            projections={"0": (torch.ones(1, 3), torch.ones(2, 1))},
        )


def test_projection_manifest_and_dimensions_are_validated(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    report = inject_logra(model, rank=2, seed=4)
    full = ParameterManifest.from_model(model, "full")
    with pytest.raises(ValueError, match="exactly the projected"):
        ProjectionArtifacts.save(model, tmp_path / "full", manifest=full, report=report)
    projected = ParameterManifest.from_model(
        model, "projected", include=[report.include_regex]
    )
    bad = projection_descriptor(report)
    bad["effective_ranks"] = [["0", 3]]
    with pytest.raises(ValueError, match="shape mismatch|projection_digest"):
        ProjectionArtifacts.save(
            model, tmp_path / "shape", manifest=projected, descriptor=bad
        )


def test_whitening_preserves_dtype_device_and_rejects_empty_slices():
    rows = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    fisher = {"m": rows.T @ rows / 2}
    actual = whiten_rows(rows, {"m": slice(0, 2)}, fisher, damping_scale=0.1)
    assert actual.dtype == rows.dtype and actual.device == rows.device
    with pytest.raises(ValueError, match="must not be empty"):
        whiten_rows(rows, {}, {}, damping_scale=0.1)
    with pytest.raises(ValueError, match="must not be empty"):
        LogRaFisherState({})


def test_precomputed_nan_is_rejected_and_finite_projection_preserves_forward():
    source = nn.Sequential(nn.Linear(3, 2))
    x = torch.randn(4, 3)
    expected = source(x).detach()
    A, C = torch.randn(2, 3), torch.randn(2, 2)
    inject_logra(source, rank=2, seed=0, init="artifact", projections={"0": (A, C)})
    torch.testing.assert_close(source(x), expected, rtol=0, atol=0)
    A[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        inject_logra(
            nn.Sequential(nn.Linear(3, 2)),
            rank=2,
            seed=0,
            init="artifact",
            projections={"0": (A, C)},
        )


def test_correctly_hashed_nan_projection_artifact_is_rejected(tmp_path):
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    report = inject_logra(model, rank=2, seed=4)
    manifest = ParameterManifest.from_model(
        model, "projected", include=[report.include_regex]
    )
    descriptor = projection_descriptor(report)
    ProjectionArtifacts.save(model, tmp_path, manifest=manifest, descriptor=descriptor)
    tensor_path = tmp_path / "logra_projections.safetensors"
    tensors = load_file(tensor_path)
    tensors["0.A"][0, 0] = float("nan")
    save_file(tensors, tensor_path)
    raw = hashlib.sha256()
    raw.update(tensors["0.A"].numpy().tobytes())
    raw.update(tensors["0.C"].numpy().tobytes())
    descriptor["projection_digest"] = raw.hexdigest()
    metadata_path = tmp_path / "logra_projections.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["descriptor"] = descriptor
    metadata["projection_digest"] = raw.hexdigest()
    clean = {key: value for key, value in metadata.items() if key != "content_digest"}
    metadata["content_digest"] = hashlib.sha256(
        tensor_path.read_bytes()
        + json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="finite"):
        ProjectionArtifacts.load(
            tmp_path, expected_manifest=manifest, expected_descriptor=descriptor
        )
