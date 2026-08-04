"""Second-order attribution primitives, ported from gradient-kernel ca9689a.

Covers true Hessian-vector products, GGN-vector products for selected-token
causal-LM cross entropy, manifest-aligned pair-gradient directions, the
metric-derivative ("third") term of pairwise kernel gradients, and forward-JVP
sweeps of selected token losses.

Metrics in the pair path are the frozen diagonal
:class:`~scimt.data_attribution.metrics.DiagonalMetric` (or ``None`` for the
identity).  Frozen means structurally detached: a graph-connected diagonal
cannot construct a :class:`DiagonalMetric` at all, so the rotation terms
always hold the metric fixed.  The theta-dependence a frozen metric discards
is recovered separately by :func:`metric_probe` +
:meth:`MetricDerivativeBackend.m_vp`, which contract ``dk/dv`` against
``dv/dtheta`` with one double-backprop pass per batch sample.

``m_vp`` is inherently a true second derivative — ``dv/dtheta`` has no GGN
analogue — so it always uses double backprop even when the rotation terms are
built with ``hessian_kind="ggn"``.  That mixing is mathematically coherent
(the terms are independent), but it means the model must support double
backprop whenever the metric-derivative component is enabled.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .losses import SAMPLE_ID_STRIDE, TokenizedBatch
from .manifest import (
    ManifestEntry,
    ParameterManifest,
    flatten_tensors,
    included_named_parameters,
    unflatten_vector,
)
from .metrics import DiagonalMetric

__all__ = [
    "MetricDerivativeBackend",
    "PairGradientBackend",
    "ggn_vector_product",
    "ggn_vp_causal_lm",
    "hvp_true",
    "jvp_sweep",
    "metric_probe",
    "self_direction",
]

REDUCTIONS = ("per_token", "per_sequence_sum", "per_sequence_mean")


def _validate_tokenized_batch(batch: TokenizedBatch) -> None:
    """Validate the shapes and values of a tokenized causal-LM batch."""

    input_ids = batch.input_ids
    sequence_ids = batch.sequence_ids
    target_mask = batch.target_mask
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, L]")
    if input_ids.dtype != torch.int64:
        raise ValueError("input_ids must have dtype torch.int64")
    if sequence_ids.ndim != 1 or sequence_ids.shape[0] != input_ids.shape[0]:
        raise ValueError("sequence_ids must have shape [B]")
    if target_mask.shape != input_ids.shape:
        raise ValueError("target_mask must have shape [B, L]")
    if input_ids.shape[1] == 0:
        raise ValueError("packed sequences must contain at least one token")
    if target_mask[:, 0].any():
        raise ValueError("position 0 cannot be a target")
    selected_positions = target_mask.nonzero(as_tuple=False)[:, 1]
    if not torch.all(selected_positions < SAMPLE_ID_STRIDE):
        raise ValueError(
            f"target positions must be below SAMPLE_ID_STRIDE={SAMPLE_ID_STRIDE}"
        )


def hvp_true(
    loss: Tensor,
    params: Sequence[Tensor],
    v: Sequence[Tensor],
) -> list[Tensor]:
    """Return the exact Hessian-vector product of scalar ``loss``."""

    if loss.ndim != 0:
        raise ValueError(f"loss must be scalar, got shape {tuple(loss.shape)}")
    if len(params) != len(v):
        raise ValueError(f"expected {len(params)} vectors, got {len(v)}")
    params_tuple = tuple(params)
    for parameter, vector in zip(params_tuple, v, strict=True):
        if parameter.shape != vector.shape:
            raise ValueError("each vector must have the shape of its parameter")

    grads = torch.autograd.grad(
        loss,
        params_tuple,
        create_graph=True,
        allow_unused=True,
        materialize_grads=False,
    )
    products = [
        (gradient * vector).sum()
        for gradient, vector in zip(grads, v, strict=True)
        # None/non-grad means zero curvature; dropping it prevents second-pass errors.
        if gradient is not None and gradient.requires_grad
    ]
    if not products:
        return [torch.zeros_like(parameter) for parameter in params_tuple]
    directional_gradient = sum(products)
    result = torch.autograd.grad(
        directional_gradient,
        params_tuple,
        allow_unused=True,
        materialize_grads=False,
    )
    return [
        torch.zeros_like(parameter) if product is None else product
        for parameter, product in zip(params_tuple, result, strict=True)
    ]


def ggn_vector_product(
    model,
    params_map: dict[str, Tensor],
    v_map: dict[str, Tensor],
    input_ids: Tensor,
    target_mask: Tensor,
) -> dict[str, Tensor]:
    """Return a GGN-vector product for summed selected-token cross entropy."""

    if target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be a bool tensor")
    _validate_tokenized_batch(
        TokenizedBatch(
            input_ids=input_ids,
            sequence_ids=torch.arange(
                input_ids.shape[0] if input_ids.ndim else 0,
                device=input_ids.device,
            ),
            target_mask=target_mask,
        )
    )
    if params_map.keys() != v_map.keys():
        raise ValueError("params_map and v_map must have identical keys")
    for name in params_map:
        if params_map[name].dtype != torch.float32 or v_map[name].dtype != torch.float32:
            raise TypeError(f"parameter and tangent {name!r} must be fp32")
        if params_map[name].shape != v_map[name].shape:
            raise ValueError(f"tangent {name!r} has the wrong shape")

    selected = target_mask.nonzero(as_tuple=False)
    batch_rows = selected[:, 0]
    logit_positions = selected[:, 1] - 1

    def logits_fn(parameters: dict[str, Tensor]) -> Tensor:
        output = torch.func.functional_call(
            model, parameters, (), {"input_ids": input_ids}
        )
        return output.logits[batch_rows, logit_positions, :].float()

    model.eval()
    logits, jv = torch.func.jvp(logits_fn, (params_map,), (v_map,))
    probabilities = logits.softmax(dim=-1)
    hjv = probabilities * jv - probabilities * (probabilities * jv).sum(
        dim=-1, keepdim=True
    )
    _, vjp_fn = torch.func.vjp(logits_fn, params_map)
    return vjp_fn(hjv)[0]


ggn_vp_causal_lm = ggn_vector_product


def _apply_metric_param_list(
    metric: DiagonalMetric | None,
    tensors: Sequence[Tensor],
    entries: Sequence[ManifestEntry],
) -> list[Tensor]:
    """Apply a frozen diagonal metric (``None`` = identity) entry by entry."""

    if metric is None:
        return list(tensors)
    if not isinstance(metric, DiagonalMetric):
        raise TypeError(
            "metric must be a DiagonalMetric or None (identity); other metric "
            "kinds are not supported in the pair path"
        )
    diagonal = metric.diagonal
    if diagonal.requires_grad:
        raise ValueError("metric diagonal must be detached")
    if diagonal.numel() != sum(entry.numel for entry in entries):
        raise ValueError("metric diagonal size does not match manifest entries")
    result: list[Tensor] = []
    for tensor, entry in zip(tensors, entries, strict=True):
        if tuple(tensor.shape) != entry.shape:
            raise ValueError(f"tensor for {entry.name!r} has the wrong shape")
        scale = diagonal.narrow(0, entry.global_flat_offset, entry.numel).reshape(
            entry.shape
        )
        result.append(tensor * scale.to(device=tensor.device))
    return result


class PairGradientBackend:
    """Compute ``u_AB^M = grad_theta(g_A^T M g_B)`` one pair at a time."""

    def __init__(self, model, manifest: ParameterManifest):
        self.model = model
        self.manifest = manifest
        included = included_named_parameters(model, manifest)
        self.entries = [entry for entry, _ in included]
        self.params = tuple(parameter for _, parameter in included)
        self.names = [entry.name for entry in self.entries]

    @staticmethod
    def _validate_loss(loss: Tensor, label: str) -> None:
        if loss.ndim != 0:
            raise ValueError(f"{label} must be scalar, got shape {tuple(loss.shape)}")

    def _gradient(
        self, loss: Tensor, *, create_graph: bool
    ) -> tuple[Tensor | None, ...]:
        return torch.autograd.grad(
            loss,
            self.params,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
            materialize_grads=False,
        )

    def _differentiate_products(self, products: Sequence[Tensor]) -> Tensor:
        active = [product for product in products if product.requires_grad]
        if not active:
            return torch.zeros(
                self.manifest.included_numel,
                dtype=torch.float32,
                device=self.params[0].device if self.params else "cpu",
            )
        result = torch.autograd.grad(
            sum(active),
            self.params,
            allow_unused=True,
            materialize_grads=False,
            retain_graph=True,
        )
        return flatten_tensors(self.entries, result)

    def _metric_grads(
        self, grads: tuple[Tensor | None, ...], metric: DiagonalMetric | None
    ) -> list[Tensor | None]:
        filled = [
            torch.zeros(entry.shape, dtype=parameter.dtype, device=parameter.device)
            if grad is None
            else grad
            for grad, entry, parameter in zip(
                grads, self.entries, self.params, strict=True
            )
        ]
        applied = _apply_metric_param_list(metric, filled, self.entries)
        return [
            None if grad is None else metric_grad
            for grad, metric_grad in zip(grads, applied, strict=True)
        ]

    def pair_direction(
        self,
        loss_a: Tensor,
        loss_b: Tensor,
        *,
        metric: DiagonalMetric | None = None,
        hessian_kind: str = "true",
        decompose: bool = False,
        batch_a: TokenizedBatch | None = None,
        batch_b: TokenizedBatch | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return the symmetric pair direction, or its ordered Hessian halves."""

        self._validate_loss(loss_a, "loss_a")
        self._validate_loss(loss_b, "loss_b")
        if hessian_kind not in {"true", "ggn"}:
            raise ValueError("hessian_kind must be 'true' or 'ggn'")
        if metric is not None and not isinstance(metric, DiagonalMetric):
            raise TypeError(
                "metric must be a DiagonalMetric or None (identity); other "
                "metric kinds are not supported in the pair path"
            )
        create_graph = hessian_kind == "true"
        grads_a = self._gradient(loss_a, create_graph=create_graph)
        grads_b = self._gradient(loss_b, create_graph=create_graph)

        if hessian_kind == "ggn":
            if batch_a is None or batch_b is None:
                raise ValueError("GGN mode requires batch_a and batch_b")
            half_a = self._ggn_half(batch_a, grads_b, metric)
            half_b = self._ggn_half(batch_b, grads_a, metric)
            return (half_a, half_b) if decompose else half_a + half_b

        metric_b = self._metric_grads(grads_b, metric)
        if not decompose:
            products = [
                (a * b).sum()
                for a, b in zip(grads_a, metric_b, strict=True)
                if a is not None and b is not None
            ]
            return self._differentiate_products(products)

        half_a = self._differentiate_products(
            [
                (grad_a * mg_b.detach()).sum()
                for grad_a, mg_b in zip(grads_a, metric_b, strict=True)
                if grad_a is not None and mg_b is not None
            ]
        )
        metric_a = self._metric_grads(grads_a, metric)
        half_b = self._differentiate_products(
            [
                (grad_b * mg_a.detach()).sum()
                for mg_a, grad_b in zip(metric_a, grads_b, strict=True)
                if mg_a is not None and grad_b is not None
            ]
        )
        return half_a, half_b

    def _ggn_half(
        self,
        batch: TokenizedBatch,
        vector: tuple[Tensor | None, ...],
        metric: DiagonalMetric | None,
    ) -> Tensor:
        filled = tuple(
            torch.zeros_like(parameter) if value is None else value.detach()
            for parameter, value in zip(self.params, vector, strict=True)
        )
        applied = _apply_metric_param_list(metric, filled, self.entries)
        params_map = {
            name: parameter
            for name, parameter in zip(self.names, self.params, strict=True)
        }
        v_map = {
            name: value.to(torch.float32)
            for name, value in zip(self.names, applied, strict=True)
        }
        product = ggn_vector_product(
            self.model, params_map, v_map, batch.input_ids, batch.target_mask
        )
        return flatten_tensors(self.entries, [product[name] for name in self.names])

    def self_direction(
        self,
        loss_a: Tensor,
        metric: DiagonalMetric | None,
        *,
        hessian_kind: str = "true",
        decompose: bool = False,
        batch_a: TokenizedBatch | None = None,
        batch_b: TokenizedBatch | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return ``u_AA = 2 H_A M g_A`` without computing ``g_A`` twice."""

        self._validate_loss(loss_a, "loss_a")
        if hessian_kind == "ggn":
            if batch_a is None:
                raise ValueError("GGN mode requires batch_a")
            gradient = self._gradient(loss_a, create_graph=False)
            half = self._ggn_half(batch_a, gradient, metric)
            return (half, half) if decompose else 2 * half
        if hessian_kind != "true":
            raise ValueError("hessian_kind must be 'true' or 'ggn'")
        gradient = self._gradient(loss_a, create_graph=True)
        metric_gradient = self._metric_grads(gradient, metric)
        half = self._differentiate_products(
            [
                (left * right.detach()).sum()
                for left, right in zip(gradient, metric_gradient, strict=True)
                if left is not None and right is not None
            ]
        )
        return (half, half) if decompose else 2 * half


def self_direction(
    loss_a: Tensor,
    metric: DiagonalMetric | None,
    *,
    backend: PairGradientBackend,
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor]:
    """Convenience wrapper for ``u_AA = 2 H_A M g_A``."""

    return backend.self_direction(loss_a, metric, **kwargs)


def metric_probe(
    pair_product: Tensor,
    v: Tensor,
    entries: Sequence[ManifestEntry] | None = None,
    *,
    exponent: float = -1.0,
    eps: float = 1e-8,
    factored: bool = False,
) -> Tensor:
    """Return the probe ``w = dk/dv`` for ``k = <pair_product, (v + eps)^exponent>``.

    ``pair_product`` is the flat elementwise product ``g_x * g_y`` of the two
    (stop-gradient) pair gradients. For ``factored=True``, ``v`` must be the
    Adafactor-style reconstruction ``V_hat = R C^T / S`` per 2-D manifest entry;
    the probe is then taken w.r.t. the underlying elementwise statistic by
    chaining through the row/column/total sums. Non-2-D entries always use the
    elementwise probe.
    """

    if pair_product.ndim != 1:
        raise ValueError(
            f"pair_product must be flat, got shape {tuple(pair_product.shape)}"
        )
    if v.shape != pair_product.shape:
        raise ValueError(
            f"v has shape {tuple(v.shape)}, expected {tuple(pair_product.shape)}"
        )
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    pair_product = pair_product.detach().to(torch.float32)
    v = v.detach().to(torch.float32)
    probe = exponent * pair_product * (v + eps).pow(exponent - 1.0)
    if not factored:
        return probe
    if entries is None:
        raise ValueError("factored probe requires manifest entries")

    # dk/dV = (dV_hat/dV)^T q with q = dk/dV_hat and V_hat = R C^T / S; the
    # three linear functionals (R, C, S) contract to row/column/total sums of
    # q * V_hat divided by the matching sums of V_hat itself.
    probe = probe.clone()
    tiny = torch.finfo(torch.float32).tiny
    for entry in entries:
        if len(entry.shape) != 2:
            continue
        v_hat = v.narrow(0, entry.global_flat_offset, entry.numel).reshape(entry.shape)
        q = probe.narrow(0, entry.global_flat_offset, entry.numel).reshape(entry.shape)
        qv = q * v_hat
        row = v_hat.sum(dim=1).clamp_min(tiny)
        col = v_hat.sum(dim=0).clamp_min(tiny)
        total = v_hat.sum().clamp_min(tiny)
        chained = (
            qv.sum(dim=1, keepdim=True) / row.unsqueeze(1)
            + qv.sum(dim=0, keepdim=True) / col.unsqueeze(0)
            - qv.sum() / total
        )
        probe.narrow(0, entry.global_flat_offset, entry.numel).copy_(chained.flatten())
    return probe


class MetricDerivativeBackend(PairGradientBackend):
    """Contract a probe against ``dv/dtheta`` one batch sample at a time."""

    def m_vp(
        self,
        probe: Tensor,
        losses: Sequence[Tensor],
        *,
        losses_center: Sequence[Tensor] | None = None,
    ) -> Tensor:
        """Return ``grad_theta <probe, v(theta)>`` estimated on ``losses``.

        Uncentered second moment (``losses_center=None``):
        ``E_x[grad <probe, g_x * g_x>] = 2 E_x[H_x (probe * g_x)]``.

        Centered variance: subtract the debiased ``2 H_bar (probe * mu)``
        piece formed from the mean gradients of ``losses`` and the independent
        ``losses_center`` batch (pass ``losses_center=losses`` for the exact
        fixed-batch derivative instead of the unbiased estimator).
        """

        if probe.ndim != 1 or probe.numel() != self.manifest.included_numel:
            raise ValueError(
                f"probe must be flat with {self.manifest.included_numel} elements, "
                f"got shape {tuple(probe.shape)}"
            )
        if len(losses) == 0:
            raise ValueError("losses must be non-empty")
        if losses_center is not None and len(losses_center) == 0:
            raise ValueError("losses_center must be non-empty when provided")
        device = self.params[0].device if self.params else torch.device("cpu")
        probe_parts = unflatten_vector(probe.detach().to(device), self.entries)

        accumulator = torch.zeros(
            self.manifest.included_numel, dtype=torch.float32, device=device
        )
        for loss in losses:
            self._validate_loss(loss, "loss")
            grads = self._gradient(loss, create_graph=True)
            products = [
                (part.to(grad.dtype) * grad * grad).sum()
                for part, grad in zip(probe_parts, grads, strict=True)
                if grad is not None and grad.requires_grad
            ]
            accumulator += self._differentiate_products(products)
        accumulator /= len(losses)
        if losses_center is None:
            return accumulator

        for loss in losses_center:
            self._validate_loss(loss, "center loss")
        mean_a = sum(losses) / len(losses)
        mean_b = sum(losses_center) / len(losses_center)
        grads_a = self._gradient(mean_a, create_graph=True)
        grads_b = self._gradient(mean_b, create_graph=True)
        products = [
            (part.to(grad_a.dtype) * grad_a * grad_b).sum()
            for part, grad_a, grad_b in zip(
                probe_parts, grads_a, grads_b, strict=True
            )
            if grad_a is not None
            and grad_b is not None
            and (grad_a.requires_grad or grad_b.requires_grad)
        ]
        return accumulator - self._differentiate_products(products)

    def metric_term(
        self,
        gradient_x: Tensor,
        gradient_y: Tensor,
        losses: Sequence[Tensor],
        *,
        v: Tensor,
        exponent: float = -1.0,
        eps: float = 1e-8,
        factored: bool = False,
        losses_center: Sequence[Tensor] | None = None,
    ) -> Tensor:
        """Return ``T_xy`` from flat pair gradients and per-sample batch losses."""

        probe = metric_probe(
            gradient_x.detach() * gradient_y.detach(),
            v,
            self.entries,
            exponent=exponent,
            eps=eps,
            factored=factored,
        )
        return self.m_vp(probe, losses, losses_center=losses_center)


def _aggregate_per_sequence(
    batch_rows: Tensor, values: Tensor, reduction: str
) -> tuple[Tensor, Tensor]:
    """Aggregate selected values into one row per represented sequence."""

    # ``nonzero`` iterates row-major, so ``batch_rows`` is nondecreasing and
    # torch.unique's sorted output preserves batch order. scatter_add is
    # differentiable w.r.t. the per-token values.
    unique_rows, inverse = torch.unique(batch_rows, return_inverse=True)
    aggregated = torch.zeros(
        unique_rows.shape[0], dtype=values.dtype, device=values.device
    ).scatter_add(0, inverse, values)
    if reduction == "per_sequence_mean":
        counts = torch.bincount(inverse, minlength=unique_rows.shape[0]).to(
            values.dtype
        )
        aggregated = aggregated / counts
    return aggregated, unique_rows


def jvp_sweep(
    model,
    manifest: ParameterManifest,
    batch: TokenizedBatch,
    directions: Tensor,
    *,
    reduction: str = "per_token",
    device: str | torch.device = "cpu",
) -> Tensor:
    """Forward-JVP directional derivatives of selected token losses.

    Returns derivatives with shape ``[M, n_dirs]`` where ``M`` is the number
    of selected target positions (``per_token``) or represented sequences
    (``per_sequence_sum`` / ``per_sequence_mean``).
    """

    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}")
    device = torch.device(device)
    model = model.to(device)
    manifest.validate_against_model(model)
    included = included_named_parameters(model, manifest)
    entries = [entry for entry, _ in included]
    params_map = {entry.name: parameter for entry, parameter in included}
    for name, parameter in params_map.items():
        if parameter.dtype != torch.float32:
            raise TypeError(f"included parameter {name!r} must be fp32")

    parameter_count = manifest.included_numel
    if directions.ndim != 2 or directions.shape[1] != parameter_count:
        raise ValueError(
            f"directions must have shape [n_dirs, {parameter_count}], "
            f"got {tuple(directions.shape)}"
        )
    if directions.dtype != torch.float32:
        raise TypeError("directions must be fp32")
    if directions.device != device:
        directions = directions.to(device)

    input_ids = batch.input_ids.to(device=device, dtype=torch.int64)
    target_mask = batch.target_mask.to(device=device, dtype=torch.bool)
    sequence_ids = batch.sequence_ids.to(device=device, dtype=torch.int64)
    _validate_tokenized_batch(TokenizedBatch(input_ids, sequence_ids, target_mask))
    selected = target_mask.nonzero(as_tuple=False)
    batch_rows = selected[:, 0]
    logit_positions = selected[:, 1] - 1
    targets = input_ids[batch_rows, selected[:, 1]]

    def losses_fn(parameters: dict[str, Tensor]) -> Tensor:
        output = torch.func.functional_call(
            model, parameters, (), {"input_ids": input_ids}
        )
        logits = output.logits[batch_rows, logit_positions, :].float()
        losses = F.cross_entropy(logits, targets, reduction="none")
        if reduction == "per_token":
            return losses
        aggregated, _ = _aggregate_per_sequence(batch_rows, losses, reduction)
        return aggregated

    model.eval()
    tangent_columns: list[Tensor] = []
    for direction in directions:
        tangent_tensors = unflatten_vector(direction, entries)
        tangent_map = {
            entry.name: tangent
            for entry, tangent in zip(entries, tangent_tensors, strict=True)
        }
        _, loss_tangents = torch.func.jvp(losses_fn, (params_map,), (tangent_map,))
        tangent_columns.append(loss_tangents)

    if not tangent_columns:
        return torch.empty(
            (
                selected.shape[0]
                if reduction == "per_token"
                else torch.unique(batch_rows).numel(),
                0,
            ),
            dtype=torch.float32,
            device=device,
        )
    return torch.stack(tangent_columns, dim=1)
