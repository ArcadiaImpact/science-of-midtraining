"""Shared Pilot A thresholds, derived from the bank validator where possible."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from pathlib import Path

try:
    from ...validate_bank import (
        measurement_floor_violations,
        passes_separation,
        separation_ratios as separation_ratios,
    )
except ImportError:  # pragma: no cover - direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from validate_bank import (  # type: ignore
        measurement_floor_violations,
        passes_separation,
        separation_ratios as separation_ratios,
    )


def _numeric_default(function: Callable[..., object], parameter: str) -> float:
    """Return a numeric function default or fail loudly if the contract drifts."""
    value = inspect.signature(function).parameters[parameter].default
    if not isinstance(value, (int, float)):
        raise RuntimeError(
            f"{function.__name__}.{parameter} no longer has a numeric default"
        )
    return float(value)


# Registered band: validate_bank.passes_separation defaults are authoritative.
IN_BAND_MIN_TIME_RATIO = _numeric_default(passes_separation, "min_speedup")
IN_BAND_MAX_TIME_RATIO = _numeric_default(passes_separation, "max_speedup")
IN_BAND_MIN_PEAK_RATIO = _numeric_default(
    passes_separation, "min_memory_ratio"
)
IN_BAND_MAX_PEAK_RATIO = _numeric_default(
    passes_separation, "max_memory_ratio"
)

# Measurement floors: validate_bank.measurement_floor_violations defaults are
# authoritative, including the 512,000-byte allocator-noise floor.
MIN_TIME_SECONDS = _numeric_default(
    measurement_floor_violations, "min_time_seconds"
)
MIN_PEAK_BYTES = _numeric_default(
    measurement_floor_violations, "min_peak_bytes"
)
MAX_TIME_SPREAD = _numeric_default(
    measurement_floor_violations, "max_time_spread"
)

# Pilot-only secondary bands from PILOT_A_SPEC.md, centralized here so their
# classifier, scoring, and report representations cannot drift.
NEAR_BAND_MIN_TIME_RATIO = 1.15
NEAR_BAND_MAX_TIME_RATIO = 6.0
NEAR_BAND_MIN_PEAK_RATIO = 0.15
NEAR_BAND_MAX_PEAK_RATIO = 0.8
DOMINATED_MIN_TIME_RATIO = 1.15
DOMINATED_MAX_PEAK_RATIO = 0.85
