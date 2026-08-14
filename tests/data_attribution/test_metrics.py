from dataclasses import FrozenInstanceError

import pytest
import torch

from scimt.data_attribution.manifest import ParameterManifest
from scimt.data_attribution.metrics import DiagonalMetric


def test_full_and_marginal_statistics_and_descriptor_are_stable():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    manifest = ParameterManifest.from_model(model, "tiny")
    statistics = {
        "estimator": "full",
        "statistic": "second_moment",
        "model_identifier": "tiny",
        "model_revision": "r1",
        "dataset_fingerprint": "data",
        "parameter_manifest_digest": manifest.digest(),
        "number_of_gradient_samples": 4,
        "code_commit": "abc",
    }
    full = DiagonalMetric.from_statistics(
        statistics, torch.tensor([4.0, 9.0]), exponent=-0.5, epsilon=0.25
    )
    marginal = DiagonalMetric.from_statistics(
        {**statistics, "estimator": "marginals"},
        {"0.weight": (torch.tensor([1.0, 4.0]), torch.tensor([9.0, 16.0]))},
        exponent=0.5,
        epsilon=0.0,
        manifest=manifest,
    )
    torch.testing.assert_close(
        full.apply(torch.ones(2)), torch.tensor([4.25, 9.25]).rsqrt()
    )
    torch.testing.assert_close(
        marginal.apply(torch.ones(4)), torch.tensor([9.0, 16.0, 36.0, 64.0]).sqrt()
    )
    assert full.descriptor() == {
        "source": "diag_precond",
        "exponent": -0.5,
        "epsilon": 0.25,
        "damping": None,
        "snapshot": full.snapshot,
    }
    assert marginal.source == "adafactor"
    with pytest.raises(FrozenInstanceError):
        full.exponent = 1.0


def test_diagonal_metric_rejects_invalid_inputs():
    stats = {"estimator": "bogus"}
    with pytest.raises(ValueError, match="unsupported estimator"):
        DiagonalMetric.from_statistics(stats, torch.ones(2), exponent=-0.5)
    with pytest.raises(ValueError, match="detached"):
        DiagonalMetric(
            "diag_precond", 1, 0, None, "x", torch.ones(2, requires_grad=True)
        )


def test_adam_second_moment_uses_symmetric_fourth_root_geometry():
    statistics = {
        "estimator": "bias_corrected_ema_clipped_global_batch_gradient_square",
        "statistic": "checkpoint_local_adam_second_raw_moment",
        "model_identifier": "tiny",
        "model_revision": "7",
        "dataset_fingerprint": "paired-data",
        "parameter_manifest_digest": "a" * 64,
        "number_of_gradient_samples": 32,
        "code_commit": "abc",
    }
    metric = DiagonalMetric.from_adam_second_moment(
        statistics,
        torch.tensor([16.0, 81.0]),
        optimizer_epsilon=1.0,
        damping=2.0,
    )

    torch.testing.assert_close(
        metric.diagonal,
        torch.tensor([7.0**-0.5, 12.0**-0.5]),
    )
    assert metric.source == "adam_second_moment"
    assert metric.exponent == -0.5
    assert metric.epsilon == 1.0
    assert metric.damping == 2.0


@pytest.mark.parametrize(
    ("raw", "epsilon", "damping", "error"),
    [
        (torch.tensor([1.0, -1.0]), 1e-8, 0.0, "nonnegative"),
        (torch.tensor([1.0, float("nan")]), 1e-8, 0.0, "finite"),
        (torch.tensor([1, 2]), 1e-8, 0.0, "floating"),
        (torch.tensor([0.0, 1.0]), 0.0, 0.0, "positive"),
        (torch.tensor([1.0, 2.0]), -1.0, 0.0, "nonnegative"),
        (torch.tensor([1.0, 2.0]), 0.0, -1.0, "nonnegative"),
    ],
)
def test_adam_second_moment_rejects_invalid_inputs(raw, epsilon, damping, error):
    statistics = {
        "statistic": "checkpoint_local_adam_second_raw_moment",
        "model_identifier": "tiny",
        "model_revision": "7",
        "dataset_fingerprint": "paired-data",
        "parameter_manifest_digest": "a" * 64,
        "number_of_gradient_samples": 2,
        "code_commit": "abc",
    }
    with pytest.raises((TypeError, ValueError), match=error):
        DiagonalMetric.from_adam_second_moment(
            statistics,
            raw,
            optimizer_epsilon=epsilon,
            damping=damping,
        )


def test_mapping_statistics_follow_manifest_not_mapping_order():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1))
    manifest = ParameterManifest.from_model(model, "tiny")
    stats = {
        "estimator": "full",
        "statistic": "second_moment",
        "model_identifier": "tiny",
        "model_revision": None,
        "dataset_fingerprint": "d",
        "parameter_manifest_digest": manifest.digest(),
        "number_of_gradient_samples": 2,
        "code_commit": "c",
    }
    values = {"0.bias": torch.tensor([9.0]), "0.weight": torch.tensor([[1.0, 4.0]])}
    metric = DiagonalMetric.from_statistics(
        stats, values, exponent=1, epsilon=0, manifest=manifest
    )
    torch.testing.assert_close(metric.diagonal, torch.tensor([1.0, 4.0, 9.0]))
    with pytest.raises(TypeError, match="manifest is required"):
        DiagonalMetric.from_statistics(stats, values, exponent=1)
    assert (
        DiagonalMetric.from_statistics(
            {**stats, "ignored": object()}, torch.ones(3), exponent=1
        ).snapshot
        == metric.snapshot
    )
    with pytest.raises(ValueError, match="wrong shape"):
        DiagonalMetric.from_statistics(
            stats,
            {"0.weight": torch.ones(1, 2), "0.bias": torch.ones(2)},
            exponent=1,
            manifest=manifest,
        )


def test_flat_statistics_domain_length_and_zero_exponent_override():
    model = torch.nn.Linear(2, 1)
    manifest = ParameterManifest.from_model(model, "tiny")
    stats = {
        "estimator": "full",
        "statistic": "second_moment",
        "model_identifier": "tiny",
        "model_revision": None,
        "dataset_fingerprint": "d",
        "parameter_manifest_digest": manifest.digest(),
        "number_of_gradient_samples": 1,
        "code_commit": "c",
    }
    for raw, error in (
        (torch.ones(2), "elements"),
        (torch.tensor([1, 2, 3]), "floating"),
        (torch.tensor([1.0, float("nan"), 3.0]), "finite"),
        (torch.tensor([1.0, -1.0, 3.0]), "nonnegative"),
    ):
        with pytest.raises((TypeError, ValueError), match=error):
            DiagonalMetric.from_statistics(stats, raw, exponent=-0.5, manifest=manifest)
    zero = DiagonalMetric.from_statistics(
        stats, torch.ones(3), exponent=0, manifest=manifest
    )
    with pytest.raises(ValueError, match="exponent is zero"):
        zero.apply(torch.ones(3), power=-0.5)
    torch.testing.assert_close(zero.apply(torch.ones(3), power=0), torch.ones(3))
    with pytest.raises(ValueError, match="wrong dimensions"):
        DiagonalMetric.from_statistics(
            {**stats, "estimator": "marginals"},
            {"weight": (torch.ones(2), torch.ones(1)), "bias": torch.ones(1)},
            exponent=1,
            manifest=manifest,
        )
    with pytest.raises(ValueError, match="2D entry"):
        DiagonalMetric.from_statistics(
            {**stats, "estimator": "marginals"},
            {"weight": torch.ones(1, 2), "bias": (torch.ones(1), torch.ones(1))},
            exponent=1,
            manifest=manifest,
        )
