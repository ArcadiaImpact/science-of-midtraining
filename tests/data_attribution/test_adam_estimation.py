"""Paired frozen-checkpoint Adam-style moment estimation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scimt.data_attribution.adam_estimation import (
    AdamMomentEstimationError,
    estimate_checkpoint_moment,
    paired_global_batches,
)
from scimt.data_attribution.datasets import _BaseDataset
from scimt.data_attribution.manifest import ParameterManifest


class ToyDataset(_BaseDataset):
    def __init__(self) -> None:
        self._sequences = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 2, 0],
            [2, 1, 0],
            [1, 0, 2],
            [2, 0, 1],
        ]
        self._masks = [[False, True, True] for _ in self._sequences]


class BiasLM(torch.nn.Module):
    def __init__(self, *, dropout: float = 0.0, mutate_buffer: bool = False):
        super().__init__()
        self.included = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.3]))
        self.excluded = torch.nn.Parameter(torch.tensor([-0.4, 0.5, 0.1]))
        self.dropout = torch.nn.Dropout(dropout)
        self.mutate_buffer = mutate_buffer
        self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

    def forward(self, input_ids):
        if self.mutate_buffer:
            self.calls.add_(1)
        logits = self.included + 10.0 * self.excluded
        logits = self.dropout(logits)
        return SimpleNamespace(
            logits=logits.reshape(1, 1, -1).expand(
                input_ids.shape[0], input_ids.shape[1], -1
            )
        )


def _manifest(model: torch.nn.Module) -> ParameterManifest:
    return ParameterManifest.from_model(model, "toy", include=["included"])


def _full_batch_gradient(model, dataset, indices, max_grad_norm):
    batch = dataset.batch_from_indices(indices)
    model.zero_grad(set_to_none=True)
    logits = model(batch.input_ids).logits
    selected = batch.target_mask.nonzero(as_tuple=False)
    loss = F.cross_entropy(
        logits[selected[:, 0], selected[:, 1] - 1],
        batch.input_ids[selected[:, 0], selected[:, 1]],
        reduction="mean",
    )
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    gradient = model.included.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return gradient, float(norm)


def test_paired_global_batches_are_seeded_unique_and_grouped():
    batches = paired_global_batches(10, num_batches=2, global_batch_size=3, seed=7)

    assert batches == paired_global_batches(
        10, num_batches=2, global_batch_size=3, seed=7
    )
    assert batches != paired_global_batches(
        10, num_batches=2, global_batch_size=3, seed=8
    )
    assert tuple(map(len, batches)) == (3, 3)
    assert len(set(sum(batches, ()))) == 6


def test_paired_global_batches_refuse_insufficient_population():
    with pytest.raises(ValueError, match="without replacement"):
        paired_global_batches(5, num_batches=2, global_batch_size=3, seed=0)


def test_indexed_batch_preserves_requested_order_and_ids():
    batch = ToyDataset().batch_from_indices([3, 1])

    assert batch.sequence_ids.tolist() == [3, 1]
    assert batch.input_ids.tolist() == [[2, 1, 0], [0, 2, 1]]


def test_bias_corrected_ema_uses_synthetic_batch_count():
    dataset = ToyDataset()
    batches = ((0, 1), (2, 3))
    beta2 = 0.8
    max_norm = 100.0
    expected_model = BiasLM()
    g1, _ = _full_batch_gradient(expected_model, dataset, batches[0], max_norm)
    g2, _ = _full_batch_gradient(expected_model, dataset, batches[1], max_norm)
    raw = beta2 * ((1 - beta2) * g1.square()) + (1 - beta2) * g2.square()
    expected = raw / (1 - beta2**2)

    model = BiasLM()
    result = estimate_checkpoint_moment(
        model,
        dataset,
        _manifest(model),
        batches,
        micro_batch_size=1,
        beta2=beta2,
        max_grad_norm=max_norm,
        device="cpu",
        autocast_dtype=None,
        rng_seed=11,
    )

    assert result.number_of_batches == 2
    assert torch.allclose(result.corrected_exp_avg_sq["included"], expected)
    assert not hasattr(result, "arithmetic_mean_sq")


def test_diagnostics_do_not_concatenate_full_moment_vectors(monkeypatch):
    model = BiasLM()
    manifest = _manifest(model)

    def reject_cat(*args, **kwargs):
        raise AssertionError("diagnostics must stream tensor chunks")

    monkeypatch.setattr(torch, "cat", reject_cat)

    result = estimate_checkpoint_moment(
        model,
        ToyDataset(),
        manifest,
        ((0, 1), (2, 3)),
        micro_batch_size=1,
        beta2=0.9,
        max_grad_norm=1.0,
        device="cpu",
        autocast_dtype=None,
        rng_seed=11,
    )

    assert result.ema_mean_cosine <= 1.0


def test_microbatch_accumulation_matches_one_global_forward():
    dataset = ToyDataset()
    batches = ((0, 1, 2, 3),)
    one = BiasLM()
    full = estimate_checkpoint_moment(
        one,
        dataset,
        _manifest(one),
        batches,
        micro_batch_size=4,
        beta2=0.9,
        max_grad_norm=100.0,
        device="cpu",
        autocast_dtype=None,
        rng_seed=3,
    )
    split = BiasLM()
    accumulated = estimate_checkpoint_moment(
        split,
        dataset,
        _manifest(split),
        batches,
        micro_batch_size=1,
        beta2=0.9,
        max_grad_norm=100.0,
        device="cpu",
        autocast_dtype=None,
        rng_seed=3,
    )

    assert torch.allclose(
        full.corrected_exp_avg_sq["included"],
        accumulated.corrected_exp_avg_sq["included"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_clipping_norm_includes_excluded_trainable_parameters():
    dataset = ToyDataset()
    batches = ((0, 1),)
    max_norm = 0.05
    expected_model = BiasLM()
    expected, pre_clip_norm = _full_batch_gradient(
        expected_model, dataset, batches[0], max_norm
    )
    model = BiasLM()

    result = estimate_checkpoint_moment(
        model,
        dataset,
        _manifest(model),
        batches,
        micro_batch_size=1,
        beta2=0.9,
        max_grad_norm=max_norm,
        device="cpu",
        autocast_dtype=None,
        rng_seed=4,
    )

    assert result.gradient_norms == pytest.approx((pre_clip_norm,))
    assert torch.allclose(
        result.corrected_exp_avg_sq["included"], expected.square(), atol=1e-8
    )
    assert result.clip_coefficients[0] < 1.0


def test_estimation_leaves_parameters_unchanged_and_clears_gradients():
    model = BiasLM()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    estimate_checkpoint_moment(
        model,
        ToyDataset(),
        _manifest(model),
        ((0, 1),),
        micro_batch_size=1,
        beta2=0.9,
        max_grad_norm=1.0,
        device="cpu",
        autocast_dtype=None,
        rng_seed=5,
    )

    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_estimation_refuses_frozen_included_parameter():
    model = BiasLM()
    manifest = _manifest(model)
    model.included.requires_grad_(False)
    with pytest.raises(AdamMomentEstimationError, match="frozen"):
        estimate_checkpoint_moment(
            model,
            ToyDataset(),
            manifest,
            ((0, 1),),
            micro_batch_size=1,
            beta2=0.9,
            max_grad_norm=1.0,
            device="cpu",
            autocast_dtype=None,
            rng_seed=5,
        )


def test_estimation_refuses_gradient_less_included_parameter():
    model = BiasLM()
    model.unreached = torch.nn.Parameter(torch.tensor([1.0]))
    manifest = ParameterManifest.from_model(
        model, "toy", include=["included", "unreached"]
    )
    with pytest.raises(AdamMomentEstimationError, match="no gradient"):
        estimate_checkpoint_moment(
            model,
            ToyDataset(),
            manifest,
            ((0, 1),),
            micro_batch_size=1,
            beta2=0.9,
            max_grad_norm=1.0,
            device="cpu",
            autocast_dtype=None,
            rng_seed=5,
        )


def test_estimation_refuses_mutable_buffer_drift():
    model = BiasLM(mutate_buffer=True)
    with pytest.raises(AdamMomentEstimationError, match="buffer"):
        estimate_checkpoint_moment(
            model,
            ToyDataset(),
            _manifest(model),
            ((0, 1),),
            micro_batch_size=1,
            beta2=0.9,
            max_grad_norm=1.0,
            device="cpu",
            autocast_dtype=None,
            rng_seed=6,
        )


def test_paired_rng_repeats_dropout_estimate():
    kwargs = {
        "dataset": ToyDataset(),
        "batches": ((0, 1), (2, 3)),
        "micro_batch_size": 1,
        "beta2": 0.9,
        "max_grad_norm": 100.0,
        "device": "cpu",
        "autocast_dtype": None,
        "rng_seed": 17,
    }
    first = BiasLM(dropout=0.5)
    second = BiasLM(dropout=0.5)

    a = estimate_checkpoint_moment(
        first, manifest=_manifest(first), **kwargs
    ).corrected_exp_avg_sq["included"]
    b = estimate_checkpoint_moment(
        second, manifest=_manifest(second), **kwargs
    ).corrected_exp_avg_sq["included"]

    assert torch.equal(a, b)
