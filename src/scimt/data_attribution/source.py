"""SOURCE attribution, ported from gradient-kernel ca9689a.

SOURCE (Bae et al. 2024, arXiv:2405.12186) approximates unrolled
differentiation of SGD segment-wise.  Both segment operators are scalar
functions of curvature eigenvalues ``sigma`` scaled by ``lr_steps``, the sum
of per-step learning rates over the segment:

- :func:`f_backward` gives the segment transition ``exp(-lr_steps * sigma)``.
- :func:`f_segment` gives the within-segment accumulation
  ``(1 - exp(-lr_steps * sigma)) / sigma`` with limit ``lr_steps`` at
  ``sigma = 0``.

Eigenvalue validation is strict: NaN/inf eigenvalues raise, and only
round-off-scale negatives (within ``1e-8 * max(1, max|eigval|)``) are clamped
to zero — genuinely negative spectra raise at curvature construction, because
SOURCE assumes PSD curvature (a GGN / Fisher / EK-FAC, not a raw Hessian).

Every :class:`CurvatureOperator` carries a serializable ``basis_descriptor``
naming the coordinate system of the gradient rows it operates on. Adjacent
segments with distinct diagonal coordinate systems carry an explicit
``transition_to_previous`` multiplier; absent transitions are refused.
Public tensor boundaries preserve the caller's container: NumPy rows return
float32 NumPy results (the upstream contract), torch rows return torch results
on the input device/dtype.  Internals always compute in float64 and round
through float32, so both containers see identical values.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .ekfac import EKFACFactors
from .manifest import ParameterManifest
from .metrics import DiagonalMetric

__all__ = [
    "CurvatureOperator",
    "DenseCurvature",
    "DiagonalCurvature",
    "EKFACCurvature",
    "SourceScorer",
    "SourceSegment",
    "f_backward",
    "f_segment",
    "lr_steps_from_lrs",
    "prepare_psd_eigvals",
]

EigvalFn = Callable[[np.ndarray], np.ndarray]


def prepare_psd_eigvals(eigvals: np.ndarray) -> np.ndarray:
    """Validate eigenvalues of a PSD curvature, clamping round-off negatives.

    Raises ``ValueError`` on any NaN/inf, and on any eigenvalue below
    ``-tol`` with ``tol = 1e-8 * max(1, max|eigval|)``.  Negatives within
    tolerance (``eigh`` round-off) are clamped to zero.
    """

    values = np.asarray(eigvals, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues contain NaN or inf")
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    tol = 1e-8 * max(1.0, max_abs)
    min_val = float(values.min()) if values.size else 0.0
    if min_val < -tol:
        raise ValueError(
            f"eigenvalue {min_val:g} is below -{tol:g}: SOURCE requires a PSD "
            "curvature (GGN/Fisher). Use a PSD approximation, not a raw Hessian."
        )
    return np.maximum(values, 0.0)


def _validated(eigvals: np.ndarray, lr_steps: float) -> tuple[np.ndarray, float]:
    lr = float(lr_steps)
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError(f"lr_steps must be a positive finite float, got {lr_steps!r}")
    return prepare_psd_eigvals(eigvals), lr


def f_backward(eigvals: np.ndarray, lr_steps: float) -> np.ndarray:
    """Return the segment-transition scales ``exp(-lr_steps * sigma)``.

    Eigenvalues pass through :func:`prepare_psd_eigvals`: NaN/inf or
    genuinely negative spectra raise ``ValueError``.
    """

    sigma, lr = _validated(eigvals, lr_steps)
    return np.exp(-lr * sigma)


def f_segment(eigvals: np.ndarray, lr_steps: float) -> np.ndarray:
    """Return the segment-residual scales ``(1 - exp(-lr_steps * sigma)) / sigma``.

    Computed as ``lr_steps * (-expm1(-x)) / x`` with ``x = lr_steps * sigma``,
    taking the exact limit ``lr_steps`` at ``x = 0`` without NaNs or warnings.
    Eigenvalues pass through :func:`prepare_psd_eigvals`: NaN/inf or genuinely
    negative spectra raise ``ValueError``.
    """

    sigma, lr = _validated(eigvals, lr_steps)
    x = lr * sigma
    safe_x = np.where(x == 0.0, 1.0, x)
    ratio = np.where(x == 0.0, 1.0, -np.expm1(-safe_x) / safe_x)
    return lr * ratio


def lr_steps_from_lrs(lrs: Sequence[float]) -> float:
    """Sum per-step learning rates into a segment's ``lr_steps`` scalar."""

    values = np.asarray(list(lrs), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("lrs must be a flat sequence of floats")
    if values.size and (not np.all(np.isfinite(values)) or np.any(values < 0)):
        raise ValueError("per-step learning rates must be finite and non-negative")
    return float(values.sum())


def _as_row_matrix(
    rows: Any, dimension: int
) -> tuple[np.ndarray, bool, tuple[torch.device, torch.dtype] | None]:
    """Return float64 ``[Q, P]`` rows plus vector/torch boundary bookkeeping."""

    boundary = None
    if isinstance(rows, torch.Tensor):
        boundary = (rows.device, rows.dtype)
        rows = rows.detach().to(device="cpu", dtype=torch.float64).numpy()
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
        was_vector = True
    elif array.ndim == 2:
        was_vector = False
    else:
        raise ValueError(f"rows must be [P] or [Q, P], got shape {np.shape(rows)}")
    if array.shape[1] != dimension:
        raise ValueError(
            f"rows have parameter dimension {array.shape[1]}, expected {dimension}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("rows contain NaN or inf")
    return array, was_vector, boundary


def _return_rows(
    result: np.ndarray,
    was_vector: bool,
    boundary: tuple[torch.device, torch.dtype] | None,
) -> np.ndarray | torch.Tensor:
    """Round through float32 (the upstream contract) and restore the boundary."""

    result = result.astype(np.float32, copy=False)
    if was_vector:
        result = result[0]
    if boundary is None:
        return result
    device, dtype = boundary
    return torch.from_numpy(np.ascontiguousarray(result)).to(
        device=device, dtype=dtype
    )


def _scale_from_fn(fn: EigvalFn, eigvals: np.ndarray) -> np.ndarray:
    scale = np.asarray(fn(eigvals), dtype=np.float64)
    if scale.shape != eigvals.shape:
        raise ValueError(
            f"fn returned shape {scale.shape} for eigenvalues of shape {eigvals.shape}"
        )
    return scale


class CurvatureOperator(ABC):
    """One segment's PSD curvature with an explicit row-coordinate basis.

    Subclasses validate their spectra as PSD at construction (raw Hessians
    never construct) and expose ``apply_fn(rows, fn)`` computing
    ``V diag(fn(lam)) V^T row`` for each row.  ``basis_descriptor`` is a
    JSON-serializable snapshot of the coordinate system the rows live in;
    :class:`SourceScorer` compares it across segments before chaining.
    """

    def __init__(
        self,
        dimension: int,
        basis_descriptor: Mapping[str, Any] | None,
        *,
        default_manifest_digest: str | None = None,
    ) -> None:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
        ):
            raise ValueError(
                f"curvature dimension must be a positive integer, got {dimension!r}"
            )
        self._dimension = dimension
        if basis_descriptor is None:
            provided: dict[str, Any] = {}
        elif isinstance(basis_descriptor, Mapping):
            provided = dict(basis_descriptor)
        else:
            raise TypeError("basis_descriptor must be a mapping")
        merged = {
            "coordinates": "raw",
            "manifest_digest": default_manifest_digest,
            **provided,
        }
        declared = merged.get("dimension")
        if declared is not None and declared != dimension:
            raise ValueError(
                f"basis_descriptor dimension {declared!r} does not match the "
                f"operator dimension {dimension}"
            )
        merged["dimension"] = dimension
        try:
            self._basis_key = json.dumps(merged, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError("basis_descriptor must be JSON-serializable") from error

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def basis_descriptor(self) -> dict[str, Any]:
        """A fresh copy of the serializable basis descriptor."""

        return json.loads(self._basis_key)

    @property
    def basis_key(self) -> str:
        """Canonical JSON form of the descriptor, used for equality checks."""

        return self._basis_key

    @abstractmethod
    def apply_fn(
        self, rows: np.ndarray | torch.Tensor, fn: EigvalFn
    ) -> np.ndarray | torch.Tensor:
        """Apply ``V diag(fn(lam)) V^T`` to ``[P]`` or ``[Q, P]`` rows."""


class DenseCurvature(CurvatureOperator):
    """Curvature ``H = V diag(lam) V^T`` given by an explicit eigendecomposition."""

    def __init__(
        self,
        eigvecs: np.ndarray,
        eigvals: np.ndarray,
        *,
        basis_descriptor: Mapping[str, Any] | None = None,
    ) -> None:
        vectors = np.asarray(eigvecs, dtype=np.float64)
        values = np.asarray(eigvals, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != vectors.shape[1]:
            raise ValueError(f"eigvecs must be square [P, P], got {vectors.shape}")
        if values.shape != (vectors.shape[0],):
            raise ValueError(
                f"eigvals must have shape [{vectors.shape[0]}], got {values.shape}"
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("eigvecs contain NaN or inf")
        values = prepare_psd_eigvals(values)
        super().__init__(int(vectors.shape[0]), basis_descriptor)
        self._eigvecs = vectors
        self._eigvals = values

    @classmethod
    def from_matrix(
        cls,
        H: np.ndarray,
        *,
        basis_descriptor: Mapping[str, Any] | None = None,
    ) -> "DenseCurvature":
        """Eigendecompose a symmetric PSD matrix.

        Raises ``ValueError`` on non-finite entries, on asymmetry, and on
        genuinely negative eigenvalues (SOURCE requires PSD curvature —
        a GGN/Fisher); round-off-scale negatives are clamped to zero via
        :func:`prepare_psd_eigvals`.
        """

        matrix = np.asarray(H, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"H must be square, got shape {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("H contains NaN or inf")
        if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
            raise ValueError("H must be symmetric (allclose to H.T)")
        eigvals, eigvecs = np.linalg.eigh(matrix)
        return cls(
            eigvecs, prepare_psd_eigvals(eigvals), basis_descriptor=basis_descriptor
        )

    def apply_fn(
        self, rows: np.ndarray | torch.Tensor, fn: EigvalFn
    ) -> np.ndarray | torch.Tensor:
        matrix, was_vector, boundary = _as_row_matrix(rows, self._dimension)
        scale = _scale_from_fn(fn, self._eigvals)
        result = ((matrix @ self._eigvecs) * scale) @ self._eigvecs.T
        return _return_rows(result, was_vector, boundary)


class DiagonalCurvature(CurvatureOperator):
    """Curvature diagonal in the row basis: ``H = diag(v)``."""

    def __init__(
        self,
        v: np.ndarray | torch.Tensor,
        *,
        basis_descriptor: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(v, torch.Tensor):
            v = v.detach().to(device="cpu", dtype=torch.float64).numpy()
        values = np.asarray(v, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"v must be a flat [P] vector, got shape {values.shape}")
        values = prepare_psd_eigvals(values)
        super().__init__(int(values.shape[0]), basis_descriptor)
        self._v = values

    @classmethod
    def from_metric(
        cls,
        metric: DiagonalMetric,
        *,
        manifest: ParameterManifest | None = None,
        basis_descriptor: Mapping[str, Any] | None = None,
    ) -> "DiagonalCurvature":
        """Use a frozen :class:`DiagonalMetric` diagonal as the curvature spectrum.

        The metric diagonal must be PSD (it is, for Fisher/Adam second-moment
        statistics under any real exponent).  When ``manifest`` is given, the
        diagonal must match its included size and the default basis descriptor
        records the manifest digest.
        """

        if not isinstance(metric, DiagonalMetric):
            raise TypeError("from_metric requires a DiagonalMetric")
        diagonal = metric.diagonal
        default_digest = None
        if manifest is not None:
            if not isinstance(manifest, ParameterManifest):
                raise TypeError("manifest must be a ParameterManifest")
            if diagonal.numel() != manifest.included_numel:
                raise ValueError(
                    f"metric diagonal has {diagonal.numel()} coordinates, the "
                    f"manifest includes {manifest.included_numel}"
                )
            default_digest = manifest.digest()
        descriptor = {
            "manifest_digest": default_digest,
            **({} if basis_descriptor is None else dict(basis_descriptor)),
        }
        return cls(diagonal, basis_descriptor=descriptor)

    def apply_fn(
        self, rows: np.ndarray | torch.Tensor, fn: EigvalFn
    ) -> np.ndarray | torch.Tensor:
        matrix, was_vector, boundary = _as_row_matrix(rows, self._dimension)
        result = matrix * _scale_from_fn(fn, self._v)
        return _return_rows(result, was_vector, boundary)


class EKFACCurvature(CurvatureOperator):
    """Curvature in EK-FAC form: per-Linear Kronecker eigenbases plus a diagonal.

    Consumes the existing :class:`~scimt.data_attribution.ekfac.EKFACFactors`
    and :class:`~scimt.data_attribution.manifest.ParameterManifest` coordinate
    system; no parallel layout is introduced.
    """

    def __init__(
        self,
        factors: EKFACFactors,
        manifest: ParameterManifest,
        *,
        basis_descriptor: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(factors, EKFACFactors):
            raise TypeError("factors must be EKFACFactors")
        if not isinstance(manifest, ParameterManifest):
            raise TypeError("manifest must be a ParameterManifest")
        self._factors = factors
        self._manifest = manifest
        self._entries = {entry.name: entry for entry in manifest.included_entries()}
        self._validate()
        super().__init__(
            manifest.included_numel,
            basis_descriptor,
            default_manifest_digest=manifest.digest(),
        )

    def _validate(self) -> None:
        claimed: set[str] = set()
        for name, factor in self._factors.linears.items():
            weight_name, bias_name = f"{name}.weight", f"{name}.bias"
            weight = self._entries.get(weight_name)
            if weight is None:
                raise ValueError(f"EK-FAC factor {name!r} has no included weight")
            if len(weight.shape) != 2:
                raise ValueError(f"EK-FAC weight {weight_name!r} is not a matrix")
            out_features, in_features = weight.shape
            bias = self._entries.get(bias_name)
            activation_features = in_features + (1 if bias is not None else 0)
            expected = {
                "U_A": (activation_features, activation_features),
                "U_S": (out_features, out_features),
                "lam": (out_features, activation_features),
            }
            for key, shape in expected.items():
                value = factor[key]
                if tuple(value.shape) != shape:
                    raise ValueError(
                        f"factor {name!r} {key} has shape {tuple(value.shape)}, "
                        f"expected {shape}"
                    )
                if not value.is_floating_point() or not bool(
                    torch.isfinite(value).all()
                ):
                    raise ValueError(
                        f"factor {name!r} {key} must be floating-point and finite"
                    )
            # Genuinely negative EK-FAC eigenvalues are a raw-Hessian smell.
            prepare_psd_eigvals(factor["lam"].detach().cpu().numpy().ravel())
            claimed.add(weight_name)
            if bias is not None:
                claimed.add(bias_name)
        diag_v = self._factors.diag_v
        if not diag_v.is_floating_point() or not bool(torch.isfinite(diag_v).all()):
            raise ValueError("diagonal factors must be floating-point and finite")
        prepare_psd_eigvals(diag_v.detach().cpu().numpy().ravel())
        cursor = 0
        for item in self._factors.diag_index:
            name, numel = item.get("name"), item.get("numel")
            entry = self._entries.get(name)
            if entry is None or numel != entry.numel:
                raise ValueError(f"invalid diagonal factor entry {item!r}")
            if item.get("offset") != entry.global_flat_offset:
                raise ValueError(f"diagonal factor offset mismatch for {name!r}")
            cursor += numel
            claimed.add(name)
        if cursor != diag_v.numel():
            raise ValueError("diagonal coordinates do not match the diagonal index")
        missing = set(self._entries) - claimed
        if missing:
            raise ValueError(
                f"EK-FAC factors do not cover parameters: {sorted(missing)}"
            )

    def apply_fn(
        self, rows: np.ndarray | torch.Tensor, fn: EigvalFn
    ) -> np.ndarray | torch.Tensor:
        matrix, was_vector, boundary = _as_row_matrix(rows, self._dimension)
        # Modules OUTER, rows inner: the fp64 working copies of one module's
        # eigenvector pair exist only for that module's iteration. The old
        # shape converted EVERY module's U_A/U_S to fp64 up front and held
        # them through the row loop — ~328 GB per stage at full 12B coverage
        # (a second OOM source beyond eager factor loading; pod run
        # 20260819T095144Z). Linear blocks and diagonal entries write
        # slice-disjoint coordinates (validated non-overlapping, full
        # coverage), and each output element is produced by exactly the same
        # operations and rounded to float32 exactly once, so the reordering
        # is numerically identical to the row-outer original.
        output = np.zeros(matrix.shape, dtype=np.float32)
        output_t = torch.from_numpy(output)
        for name, factor in self._factors.linears.items():
            weight = self._entries[f"{name}.weight"]
            bias = self._entries.get(f"{name}.bias")
            lam = factor["lam"].detach().to(device="cpu", dtype=torch.float64).numpy()
            scale = torch.from_numpy(_scale_from_fn(fn, lam))
            U_A = factor["U_A"].detach().to(device="cpu", dtype=torch.float64)
            U_S = factor["U_S"].detach().to(device="cpu", dtype=torch.float64)
            for row_index in range(matrix.shape[0]):
                source = torch.from_numpy(matrix[row_index])
                weight_grad = source.narrow(
                    0, weight.global_flat_offset, weight.numel
                ).reshape(weight.shape)
                bias_grad = (
                    source.narrow(0, bias.global_flat_offset, bias.numel).reshape(
                        bias.shape
                    )
                    if bias is not None
                    else None
                )
                augmented = (
                    weight_grad
                    if bias_grad is None
                    else torch.cat([weight_grad, bias_grad.reshape(-1, 1)], dim=1)
                )
                rotated = (U_S.T @ augmented @ U_A) * scale
                restored = U_S @ rotated @ U_A.T
                if bias is not None:
                    restored = torch.cat(
                        [restored[:, :-1].reshape(-1), restored[:, -1].reshape(-1)]
                    )
                flat_values = restored.reshape(-1)
                output_t[
                    row_index,
                    weight.global_flat_offset : weight.global_flat_offset
                    + weight.numel,
                ] = flat_values[: weight.numel].to(torch.float32)
                if bias is not None:
                    output_t[
                        row_index,
                        bias.global_flat_offset : bias.global_flat_offset
                        + bias.numel,
                    ] = flat_values[weight.numel :].to(torch.float32)
            # Release this module's fp64 working copies before the next one.
            U_A = U_S = scale = None
        if self._factors.diag_v.numel():
            diag = (
                self._factors.diag_v.detach()
                .to(device="cpu", dtype=torch.float64)
                .numpy()
            )
            diag_scale = torch.from_numpy(_scale_from_fn(fn, diag))
            for row_index in range(matrix.shape[0]):
                source = torch.from_numpy(matrix[row_index])
                cursor = 0
                for item in self._factors.diag_index:
                    numel, offset = int(item["numel"]), int(item["offset"])
                    output_t[row_index, offset : offset + numel] = (
                        source[offset : offset + numel]
                        * diag_scale[cursor : cursor + numel]
                    ).to(torch.float32)
                    cursor += numel
        return _return_rows(output, was_vector, boundary)


@dataclass(frozen=True)
class SourceSegment:
    """One chronological segment and its optional previous-basis transition."""

    name: str
    curvature: CurvatureOperator
    lr_steps: float
    transition_to_previous: np.ndarray | torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("segment name must be a non-empty string")
        if not math.isfinite(self.lr_steps) or self.lr_steps <= 0:
            raise ValueError(
                f"segment {self.name!r} lr_steps must be positive and finite, "
                f"got {self.lr_steps!r}"
            )
        if not isinstance(self.curvature, CurvatureOperator):
            raise ValueError(
                f"segment {self.name!r} curvature must be a CurvatureOperator "
                "(PSD Fisher/GGN/EK-FAC); raw Hessian curvature is not supported"
            )
        transition = self.transition_to_previous
        if transition is not None:
            if isinstance(transition, torch.Tensor):
                transition = transition.detach().to(
                    device="cpu", dtype=torch.float64
                ).numpy()
            transition = np.asarray(transition, dtype=np.float64)
            if transition.shape != (self.curvature.dimension,):
                raise ValueError(
                    f"segment {self.name!r} transition_to_previous must have "
                    f"shape [{self.curvature.dimension}]"
                )
            if not np.all(np.isfinite(transition)) or np.any(transition <= 0):
                raise ValueError(
                    f"segment {self.name!r} transition_to_previous must contain "
                    "only positive finite values"
                )
            transition = transition.copy()
            transition.flags.writeable = False
            object.__setattr__(self, "transition_to_previous", transition)

    @property
    def basis_descriptor(self) -> dict[str, Any]:
        """The serializable row-basis descriptor of this segment's curvature."""

        return self.curvature.basis_descriptor


class SourceScorer:
    """Chain SOURCE segment operators over segments in chronological order.

    The score for query gradient ``q`` (final checkpoint) against a training
    example's per-segment average gradients ``g_l`` is

    ``tau = sum_l [F_segment(H_l) prod_{l'=L..l+1} F_backward(H_l') q]^T g_l``

    evaluated right-to-left on the query.  Scores omit the global ``1/N``
    factor from the SGD expectation; the caller divides by the training-set
    size exactly once when the absolute scale matters (Eq. 24 of
    arXiv:2405.12186).
    """

    def __init__(self, segments: list[SourceSegment]) -> None:
        segments = list(segments)
        if not segments:
            raise ValueError("SourceScorer requires at least one segment")
        for segment in segments:
            if not isinstance(segment, SourceSegment):
                raise ValueError("SourceScorer segments must be SourceSegment")
        names = [segment.name for segment in segments]
        if len(set(names)) != len(names):
            raise ValueError(f"segment names must be unique, got {names}")
        first = segments[0]
        if first.transition_to_previous is not None:
            raise ValueError("the first segment cannot transition to a previous basis")
        for previous, segment in zip(segments, segments[1:], strict=False):
            if segment.curvature.dimension != first.curvature.dimension:
                raise ValueError(
                    f"segment {segment.name!r} basis has dimension "
                    f"{segment.curvature.dimension}, expected "
                    f"{first.curvature.dimension}"
                )
            if (
                segment.curvature.basis_key != previous.curvature.basis_key
                and segment.transition_to_previous is None
            ):
                raise ValueError(
                    f"segment {segment.name!r} basis differs from previous "
                    f"segment {previous.name!r}; transition_to_previous is "
                    "required"
                )
        self._segments = segments
        self._dimension = first.curvature.dimension

    @property
    def basis_descriptor(self) -> dict[str, Any]:
        """The shared serializable basis descriptor of all segments."""

        return self._segments[0].basis_descriptor

    def _query_array(
        self, query_rows: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, tuple[torch.device, torch.dtype] | None]:
        boundary = None
        if isinstance(query_rows, torch.Tensor):
            boundary = (query_rows.device, query_rows.dtype)
            query_rows = (
                query_rows.detach().to(device="cpu", dtype=torch.float64).numpy()
            )
        array = np.asarray(query_rows)
        if array.ndim != 2:
            raise ValueError(
                f"query_rows must be a 2D [rows, P] array, got shape {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError("query_rows must contain at least one row")
        if array.shape[1] != self._dimension:
            raise ValueError(
                f"query_rows have parameter dimension {array.shape[1]}, "
                f"expected {self._dimension}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("query_rows contain NaN or inf")
        return array, boundary

    def _transformed(self, query: np.ndarray) -> list[np.ndarray]:
        segments = self._segments
        transformed: list[np.ndarray | None] = [None] * len(segments)
        current: np.ndarray = query
        for index in range(len(segments) - 1, -1, -1):
            segment = segments[index]
            transformed[index] = segment.curvature.apply_fn(
                current, lambda ev, lr=segment.lr_steps: f_segment(ev, lr)
            )
            if index > 0:
                current = segment.curvature.apply_fn(
                    current, lambda ev, lr=segment.lr_steps: f_backward(ev, lr)
                )
                if segment.transition_to_previous is not None:
                    current = current * segment.transition_to_previous
        return list(transformed)

    def transformed_queries(
        self, query_rows: np.ndarray | torch.Tensor
    ) -> list[np.ndarray] | list[torch.Tensor]:
        """Return ``[u_1, ..., u_L]`` (chronological) for query rows ``[Q, P]``."""

        query, boundary = self._query_array(query_rows)
        transformed = self._transformed(query)
        if boundary is None:
            return transformed
        device, dtype = boundary
        return [
            torch.from_numpy(np.ascontiguousarray(u)).to(device=device, dtype=dtype)
            for u in transformed
        ]

    def _train_array(self, rows: Any, segment_name: str) -> np.ndarray:
        if isinstance(rows, torch.Tensor):
            rows = rows.detach().to(device="cpu", dtype=torch.float32).numpy()
        array = np.asarray(rows)
        if array.ndim != 2:
            raise ValueError(
                f"segment {segment_name!r} train rows must be a 2D [rows, P] "
                f"array, got shape {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError(
                f"segment {segment_name!r} has an empty train-row matrix; "
                "pass None for segments whose rows are unavailable"
            )
        if array.shape[1] != self._dimension:
            raise ValueError(
                f"segment {segment_name!r} train rows have parameter dimension "
                f"{array.shape[1]}, expected {self._dimension}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"segment {segment_name!r} train rows contain NaN or inf")
        return array.astype(np.float32, copy=False)

    def scores(
        self,
        query_rows: np.ndarray | torch.Tensor,
        train_rows_per_segment: Sequence[np.ndarray | torch.Tensor | None],
    ) -> np.ndarray | torch.Tensor:
        """Return ``[Q, N]`` scores ``sum_l u_l @ g_l^T`` over provided segments.

        ``None`` entries mark segments the training examples do not participate
        in (or whose rows are unavailable) and contribute nothing.  Scores are
        unnormalized: the caller applies the ``1/N`` factor exactly once.
        """

        train_rows = list(train_rows_per_segment)
        if len(train_rows) != len(self._segments):
            raise ValueError(
                f"expected {len(self._segments)} per-segment train-row arrays, "
                f"got {len(train_rows)}"
            )
        query, boundary = self._query_array(query_rows)
        arrays: list[np.ndarray | None] = []
        n_examples: int | None = None
        for segment, rows in zip(self._segments, train_rows, strict=True):
            if rows is None:
                arrays.append(None)
                continue
            array = self._train_array(rows, segment.name)
            if n_examples is None:
                n_examples = array.shape[0]
            elif array.shape[0] != n_examples:
                raise ValueError(
                    f"segment {segment.name!r} has {array.shape[0]} train rows, "
                    f"other segments have {n_examples}"
                )
            arrays.append(array)
        if n_examples is None:
            raise ValueError("at least one segment must provide train rows")
        transformed = self._transformed(query)
        total = np.zeros((query.shape[0], n_examples), dtype=np.float32)
        for u_rows, g_rows in zip(transformed, arrays, strict=True):
            if g_rows is not None:
                total += u_rows @ g_rows.T
        if boundary is None:
            return total
        device, dtype = boundary
        return torch.from_numpy(total).to(device=device, dtype=dtype)
