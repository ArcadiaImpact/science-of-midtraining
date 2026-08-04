from dataclasses import FrozenInstanceError

import pytest
import torch

from scimt.data_attribution.metrics import DiagonalMetric


def test_full_and_marginal_statistics_and_descriptor_are_stable():
    statistics = {
        "estimator": "full",
        "statistic": "second_moment",
        "model_identifier": "tiny",
        "model_revision": "r1",
        "dataset_fingerprint": "data",
        "parameter_manifest_digest": "manifest",
        "number_of_gradient_samples": 4,
        "code_commit": "abc",
    }
    full = DiagonalMetric.from_statistics(
        statistics, torch.tensor([4.0, 9.0]), exponent=-0.5, epsilon=0.25
    )
    marginal = DiagonalMetric.from_statistics(
        {**statistics, "estimator": "marginals"},
        {"weight": (torch.tensor([1.0, 4.0]), torch.tensor([9.0, 16.0]))},
        exponent=0.5,
        epsilon=0.0,
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
