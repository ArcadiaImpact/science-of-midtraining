
import pytest
import torch
from torch import nn

from scimt.data_attribution.logra import (
    LogRaFisherState,
    ProjectionArtifacts,
    inject_logra,
    pca_projections,
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
    digest = ProjectionArtifacts.save(model, tmp_path)
    fresh = nn.Module()
    fresh.left = nn.Linear(3, 2)
    fresh.right = fresh.left
    loaded = ProjectionArtifacts.load(tmp_path)
    restored = inject_logra(fresh, rank=2, seed=0, init="artifact", projections=loaded)
    assert digest
    assert restored.projection_digest == report.projection_digest


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
