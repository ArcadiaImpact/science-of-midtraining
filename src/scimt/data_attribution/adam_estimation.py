"""Paired frozen-checkpoint Adam-style second-raw-moment estimation.

The estimator intentionally does not construct an optimizer or update model
weights. Each synthetic timestep is one complete optimizer-sized global-batch
gradient accumulated across microbatches, globally clipped over every
trainable parameter, squared, and folded into Adam's bias-corrected EMA.
"""

from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


class AdamMomentEstimationError(ValueError):
    """Frozen estimation inputs or runtime behavior violate the contract."""


@dataclass(frozen=True)
class AdamMomentEstimate:
    corrected_exp_avg_sq: dict[str, torch.Tensor]
    arithmetic_mean_sq: dict[str, torch.Tensor]
    number_of_batches: int
    target_token_counts: tuple[int, ...]
    gradient_norms: tuple[float, ...]
    clip_coefficients: tuple[float, ...]
    ema_mean_cosine: float
    ema_mean_relative_l2: float


def paired_global_batches(
    population: int,
    *,
    num_batches: int,
    global_batch_size: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Draw deterministic ordered global batches without replacement."""

    for value, name in (
        (population, "population"),
        (num_batches, "num_batches"),
        (global_batch_size, "global_batch_size"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    required = num_batches * global_batch_size
    if required > population:
        raise ValueError(
            f"paired sampling without replacement needs {required} sequences, "
            f"but the tokenized dataset contains {population}"
        )
    order = random.Random(seed).sample(range(population), required)
    return tuple(
        tuple(order[start : start + global_batch_size])
        for start in range(0, required, global_batch_size)
    )


def _buffer_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers(remove_duplicate=False)
    }


def _diagnostics(
    corrected: dict[str, torch.Tensor], arithmetic: dict[str, torch.Tensor]
) -> tuple[float, float]:
    ema = torch.cat([corrected[name].reshape(-1) for name in corrected]).double()
    mean = torch.cat([arithmetic[name].reshape(-1) for name in corrected]).double()
    ema_norm = float(torch.linalg.vector_norm(ema))
    mean_norm = float(torch.linalg.vector_norm(mean))
    if ema_norm == 0.0 or mean_norm == 0.0:
        cosine = 1.0 if torch.equal(ema, mean) else 0.0
    else:
        cosine = float(torch.dot(ema, mean) / (ema_norm * mean_norm))
    relative_l2 = float(
        torch.linalg.vector_norm(ema - mean) / max(mean_norm, torch.finfo(torch.float64).tiny)
    )
    return cosine, relative_l2


def estimate_checkpoint_moment(
    model: torch.nn.Module,
    dataset: Any,
    manifest: Any,
    batches: tuple[tuple[int, ...], ...],
    *,
    micro_batch_size: int,
    beta2: float,
    max_grad_norm: float,
    device: str | torch.device,
    autocast_dtype: torch.dtype | None,
    rng_seed: int,
) -> AdamMomentEstimate:
    """Estimate one checkpoint-local corrected Adam second raw moment."""

    if not batches or any(not batch for batch in batches):
        raise AdamMomentEstimationError("batches must be nonempty")
    global_batch_size = len(batches[0])
    if any(len(batch) != global_batch_size for batch in batches):
        raise AdamMomentEstimationError("all global batches must have equal size")
    if (
        not isinstance(micro_batch_size, int)
        or isinstance(micro_batch_size, bool)
        or micro_batch_size < 1
        or global_batch_size % micro_batch_size
    ):
        raise AdamMomentEstimationError(
            "micro_batch_size must be positive and divide the global batch"
        )
    if not isinstance(beta2, (int, float)) or isinstance(beta2, bool):
        raise AdamMomentEstimationError("beta2 must be numeric")
    beta2 = float(beta2)
    if not math.isfinite(beta2) or not 0 < beta2 < 1:
        raise AdamMomentEstimationError("beta2 must satisfy 0 < beta2 < 1")
    max_grad_norm = float(max_grad_norm)
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise AdamMomentEstimationError("max_grad_norm must be positive and finite")
    if not isinstance(rng_seed, int) or isinstance(rng_seed, bool) or rng_seed < 0:
        raise AdamMomentEstimationError("rng_seed must be a nonnegative integer")

    target_device = torch.device(device)
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not trainable:
        raise AdamMomentEstimationError("model has no trainable parameters")
    named = dict(model.named_parameters(remove_duplicate=False))
    entries = tuple(manifest.included_entries())
    if not entries:
        raise AdamMomentEstimationError("parameter manifest selects no coordinates")
    missing = [entry.name for entry in entries if entry.name not in named]
    if missing:
        raise AdamMomentEstimationError(
            f"parameter manifest names absent model parameters: {missing}"
        )

    ema = {
        entry.name: torch.zeros(entry.shape, dtype=torch.float32, device="cpu")
        for entry in entries
    }
    arithmetic = {
        entry.name: torch.zeros(entry.shape, dtype=torch.float32, device="cpu")
        for entry in entries
    }
    target_counts: list[int] = []
    gradient_norms: list[float] = []
    clip_coefficients: list[float] = []
    buffers_before = _buffer_snapshot(model)
    was_training = model.training
    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        ]

    try:
        model.train(True)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(rng_seed)
            if target_device.type == "cuda":
                torch.cuda.manual_seed_all(rng_seed)
            for batch_indices in batches:
                target_count = 0
                for start in range(0, global_batch_size, micro_batch_size):
                    micro = dataset.batch_from_indices(
                        batch_indices[start : start + micro_batch_size]
                    )
                    target_count += int(micro.target_mask.sum())
                if target_count < 1:
                    raise AdamMomentEstimationError(
                        "global estimator batch has no selected target tokens"
                    )
                model.zero_grad(set_to_none=True)
                for start in range(0, global_batch_size, micro_batch_size):
                    micro = dataset.batch_from_indices(
                        batch_indices[start : start + micro_batch_size]
                    )
                    ids = micro.input_ids.to(target_device)
                    mask = micro.target_mask.to(target_device)
                    context = (
                        nullcontext()
                        if autocast_dtype is None
                        else torch.autocast(
                            target_device.type, dtype=autocast_dtype
                        )
                    )
                    with context:
                        logits = model(input_ids=ids).logits
                    selected = mask.nonzero(as_tuple=False)
                    loss = F.cross_entropy(
                        logits[selected[:, 0], selected[:, 1] - 1].float(),
                        ids[selected[:, 0], selected[:, 1]],
                        reduction="sum",
                    ) / target_count
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                norm_value = float(norm.detach().cpu())
                coefficient = min(1.0, max_grad_norm / (norm_value + 1e-6))
                for entry in entries:
                    gradient = named[entry.name].grad
                    square = (
                        torch.zeros(entry.shape, dtype=torch.float32)
                        if gradient is None
                        else gradient.detach().to(
                            device="cpu", dtype=torch.float32
                        ).square()
                    )
                    ema[entry.name].mul_(beta2).add_(
                        square, alpha=1.0 - beta2
                    )
                    arithmetic[entry.name].add_(square)
                target_counts.append(target_count)
                gradient_norms.append(norm_value)
                clip_coefficients.append(coefficient)

        buffers_after = _buffer_snapshot(model)
        changed_buffers = [
            name
            for name, before in buffers_before.items()
            if name not in buffers_after or not torch.equal(before, buffers_after[name])
        ]
        changed_buffers.extend(
            name for name in buffers_after if name not in buffers_before
        )
        if changed_buffers:
            raise AdamMomentEstimationError(
                "frozen checkpoint estimation mutated model buffer(s): "
                f"{sorted(set(changed_buffers))}"
            )
        count = len(batches)
        correction = 1.0 - beta2**count
        corrected = {name: value / correction for name, value in ema.items()}
        means = {name: value / count for name, value in arithmetic.items()}
        cosine, relative_l2 = _diagnostics(corrected, means)
        return AdamMomentEstimate(
            corrected_exp_avg_sq=corrected,
            arithmetic_mean_sq=means,
            number_of_batches=count,
            target_token_counts=tuple(target_counts),
            gradient_norms=tuple(gradient_norms),
            clip_coefficients=tuple(clip_coefficients),
            ema_mean_cosine=cosine,
            ema_mean_relative_l2=relative_l2,
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


__all__ = [
    "AdamMomentEstimate",
    "AdamMomentEstimationError",
    "estimate_checkpoint_moment",
    "paired_global_batches",
]
