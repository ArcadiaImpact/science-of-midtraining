"""Immutable diagonal attribution metrics, ported from gradient-kernel ca9689a."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch

from .manifest import ParameterManifest


_SOURCES = {"full": "diag_precond", "marginals": "adafactor", "rank1": "adafactor"}
REQUIRED_PROVENANCE = frozenset(
    {
        "model_identifier",
        "model_revision",
        "dataset_fingerprint",
        "parameter_manifest_digest",
        "statistic",
        "number_of_gradient_samples",
        "code_commit",
    }
)


def _snapshot(statistics: dict[str, Any]) -> str:
    missing = REQUIRED_PROVENANCE - set(statistics)
    if missing:
        raise ValueError(f"statistics missing required provenance: {sorted(missing)}")
    provenance = {key: statistics[key] for key in REQUIRED_PROVENANCE}
    provenance.update(
        estimator=statistics.get("estimator", "full"), logra=statistics.get("logra")
    )
    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _flatten_statistics(value: Any, manifest: ParameterManifest | None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if manifest is not None and value.numel() != manifest.included_numel:
            raise ValueError(
                f"flat statistics must have {manifest.included_numel} elements"
            )
        return value.reshape(-1)
    if not isinstance(value, dict):
        raise TypeError("statistics values must be a tensor or mapping")
    if manifest is None:
        raise TypeError("manifest is required for mapping statistics")
    entries = manifest.included_entries()
    names = [entry.name for entry in entries]
    if set(value) != set(names):
        raise ValueError("statistics keys do not match manifest included entries")
    pieces = []
    for entry in entries:
        name, item = entry.name, value[entry.name]
        if isinstance(item, torch.Tensor):
            if tuple(item.shape) != entry.shape:
                raise ValueError(f"statistics tensor for {name!r} has the wrong shape")
            pieces.append(item.reshape(-1))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            if len(entry.shape) != 2:
                raise ValueError(f"marginal statistics require a 2D entry for {name!r}")
            row, column = item
            if not isinstance(row, torch.Tensor) or not isinstance(
                column, torch.Tensor
            ):
                raise TypeError(f"marginal statistics for {name!r} must be tensors")
            if tuple(row.shape) != (entry.shape[0],) or tuple(column.shape) != (
                entry.shape[1],
            ):
                raise ValueError(
                    f"marginal statistics for {name!r} have the wrong dimensions"
                )
            pieces.append(torch.outer(row.reshape(-1), column.reshape(-1)).reshape(-1))
        else:
            raise TypeError(f"invalid statistics for {name!r}")
    return torch.cat(pieces) if pieces else torch.empty(0)


@dataclass(frozen=True)
class DiagonalMetric:
    source: str
    exponent: float
    epsilon: float
    damping: float | None
    snapshot: str
    diagonal: torch.Tensor

    def __post_init__(self) -> None:
        if self.source not in {"diag_precond", "adafactor"}:
            raise ValueError(f"unsupported metric source {self.source!r}")
        if self.epsilon < 0 or (self.damping is not None and self.damping < 0):
            raise ValueError("epsilon and damping must be nonnegative")
        if self.diagonal.ndim != 1:
            raise ValueError("metric diagonal must be flat")
        if self.diagonal.requires_grad:
            raise ValueError("metric diagonal must be detached")

    @classmethod
    def from_statistics(
        cls,
        statistics: dict[str, Any],
        values: Any,
        *,
        exponent: float,
        epsilon: float = 1e-8,
        damping: float | None = None,
        manifest: ParameterManifest | None = None,
    ) -> "DiagonalMetric":
        estimator = statistics.get("estimator", "full")
        try:
            source = _SOURCES[estimator]
        except KeyError:
            raise ValueError(f"unsupported estimator {estimator!r}") from None
        if epsilon < 0 or (damping is not None and damping < 0):
            raise ValueError("epsilon and damping must be nonnegative")
        if (
            manifest is not None
            and statistics.get("parameter_manifest_digest") != manifest.digest()
        ):
            raise ValueError("statistics parameter-manifest digest mismatch")
        raw = _flatten_statistics(values, manifest).detach()
        if not raw.is_floating_point():
            raise TypeError("raw statistics must have floating-point dtype")
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("raw statistics must be finite")
        if bool((raw < 0).any()):
            raise ValueError("raw statistics must be nonnegative")
        raw = raw.to(dtype=torch.float32)
        offset = epsilon + (0.0 if damping is None else damping)
        if exponent < 0 and bool((raw + offset == 0).any()):
            raise ValueError("negative powers require positive damped statistics")
        diagonal = (raw + offset).pow(exponent).detach()
        return cls(source, exponent, epsilon, damping, _snapshot(statistics), diagonal)

    def apply(self, flat: torch.Tensor, power: float | None = None) -> torch.Tensor:
        if flat.ndim != 1 or flat.numel() != self.diagonal.numel():
            raise ValueError(f"flat must have shape [{self.diagonal.numel()}]")
        scale = self.diagonal
        if power is not None:
            if self.exponent == 0 and power != 0:
                raise ValueError("cannot override power when metric exponent is zero")
            raw_power = power / self.exponent if self.exponent != 0 else 0.0
            scale = scale.pow(raw_power)
        return flat * scale.to(device=flat.device, dtype=flat.dtype)

    def descriptor(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "exponent": self.exponent,
            "epsilon": self.epsilon,
            "damping": self.damping,
            "snapshot": self.snapshot,
        }
