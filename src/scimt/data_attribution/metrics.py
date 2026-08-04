"""Immutable diagonal attribution metrics, ported from gradient-kernel ca9689a."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch


_SOURCES = {"full": "diag_precond", "marginals": "adafactor", "rank1": "adafactor"}


def _snapshot(statistics: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            statistics, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def _flatten_statistics(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)
    if not isinstance(value, dict):
        raise TypeError("statistics values must be a tensor or mapping")
    pieces = []
    for name in sorted(value):
        item = value[name]
        if isinstance(item, torch.Tensor):
            pieces.append(item.reshape(-1))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            row, column = item
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
    ) -> "DiagonalMetric":
        estimator = statistics.get("estimator", "full")
        try:
            source = _SOURCES[estimator]
        except KeyError:
            raise ValueError(f"unsupported estimator {estimator!r}") from None
        if epsilon < 0 or (damping is not None and damping < 0):
            raise ValueError("epsilon and damping must be nonnegative")
        raw = _flatten_statistics(values).detach().to(dtype=torch.float32)
        offset = epsilon + (0.0 if damping is None else damping)
        diagonal = (raw + offset).pow(exponent).detach()
        return cls(source, exponent, epsilon, damping, _snapshot(statistics), diagonal)

    def apply(self, flat: torch.Tensor, power: float | None = None) -> torch.Tensor:
        if flat.ndim != 1 or flat.numel() != self.diagonal.numel():
            raise ValueError(f"flat must have shape [{self.diagonal.numel()}]")
        scale = self.diagonal
        if power is not None:
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
