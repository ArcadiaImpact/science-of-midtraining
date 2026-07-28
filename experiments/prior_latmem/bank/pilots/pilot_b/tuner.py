"""Bounded band tuner for Pilot B composed instances."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.prior_latmem.bank.validate_bank import (
    _measurement,
    measurement_floor_violations,
    passes_separation,
    separation_ratios,
)

from .composer import TUNABLE_KNOBS, rebuild_instance


Measurement = Mapping[str, object] | tuple[float, float]
MeasurementFunction = Callable[..., Measurement]

TARGET_TIME_RATIO = 2.2
TARGET_PEAK_RATIO = 0.45
MAX_ITERATIONS = 8
STRIDE_BOUNDS = (2, 16)
QUERY_BOUNDS = (1, 64)
RECORD_BOUNDS = (1_000, 128_000)
FIELD_COUNT_BOUNDS = (8, 128)
INTERIOR_TIME_BAND = (1.8, 3.5)
INTERIOR_PEAK_BAND = (0.30, 0.60)


def measure_instance(
    record: Mapping[str, object],
    *,
    timeout_s: float = 8.0,
    mem_limit_mb: int | None = 512,
) -> dict[str, object]:
    """Measure a pair through the validator's unchanged sandboxed path."""
    timings: dict[str, list[float]] = {}
    peaks: dict[str, list[float]] = {}
    trace_flags: dict[str, bool] = {}
    for field in ("speed_solution", "memory_solution"):
        measured, measured_peaks, trace_flag, error = _measurement(
            record,
            field,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        if error:
            return {
                "timings": timings,
                "peaks": peaks,
                "timing_traced": trace_flags,
                "error": error,
            }
        timings[field] = measured
        peaks[field] = measured_peaks
        trace_flags[field] = trace_flag
    return {
        "timings": timings,
        "peaks": peaks,
        "timing_traced": trace_flags,
        "error": None,
    }


def _knobs(record: Mapping[str, object]) -> dict[str, int]:
    meta = record.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("record meta is missing")
    params = meta.get("pattern_params")
    if not isinstance(params, Mapping):
        raise ValueError("record pattern_params is missing")
    return {name: int(params[name]) for name in TUNABLE_KNOBS}


def _invoke_measurement(
    function: MeasurementFunction,
    record: Mapping[str, object],
    knobs: Mapping[str, int],
) -> Measurement:
    """Support concise ``fn(record)`` and diagnostic ``fn(record, knobs)`` fakes."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(record)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    if has_varargs or len(positional) >= 2:
        return function(record, dict(knobs))
    return function(record)


def _measurement_parts(
    result: Measurement,
) -> tuple[
    dict[str, Sequence[float]],
    dict[str, Sequence[float]],
    tuple[float, float] | None,
    list[str],
    str | None,
]:
    if isinstance(result, tuple) and len(result) == 2:
        ratios = (float(result[0]), float(result[1]))
        return {}, {}, ratios, [], None
    if not isinstance(result, Mapping):
        return {}, {}, None, [], "measurement_result_invalid"
    error = result.get("error")
    if error:
        return {}, {}, None, [], str(error)
    timings_value = result.get("timings", {})
    peaks_value = result.get("peaks", {})
    timings = (
        {str(key): list(value) for key, value in timings_value.items()}
        if isinstance(timings_value, Mapping)
        else {}
    )
    peaks = (
        {str(key): list(value) for key, value in peaks_value.items()}
        if isinstance(peaks_value, Mapping)
        else {}
    )
    direct_time = result.get("time_ratio")
    direct_peak = result.get("peak_ratio")
    if isinstance(direct_time, (int, float)) and isinstance(
        direct_peak, (int, float)
    ):
        ratios = (float(direct_time), float(direct_peak))
    else:
        ratios = separation_ratios(timings, peaks)
    if timings and peaks:
        floors = measurement_floor_violations(timings, peaks)
    else:
        floors_value = result.get("floor_violations", ())
        floors = [str(item) for item in floors_value] if floors_value else []
    return timings, peaks, ratios, floors, None


def _inside_band(
    timings: Mapping[str, Sequence[float]],
    peaks: Mapping[str, Sequence[float]],
    ratios: tuple[float, float] | None,
) -> bool:
    if ratios is None:
        return False
    if timings and peaks:
        return passes_separation(timings, peaks)
    time_ratio, peak_ratio = ratios
    return 1.3 <= time_ratio <= 4.0 and 0.25 <= peak_ratio <= 0.7


def _score(
    ratios: tuple[float, float] | None, floors: Sequence[str], error: str | None
) -> float:
    if error or ratios is None:
        return float("inf")
    time_ratio, peak_ratio = ratios
    time_distance = abs(math.log(max(time_ratio, 1e-12) / TARGET_TIME_RATIO))
    peak_distance = abs(math.log(max(peak_ratio, 1e-12) / TARGET_PEAK_RATIO))
    return time_distance + peak_distance + 2.0 * len(floors)


def _bounded(value: int, bounds: tuple[int, int]) -> int:
    return max(bounds[0], min(bounds[1], int(value)))


def tune_instance(
    record: Mapping[str, object],
    *,
    measure_fn: MeasurementFunction = measure_instance,
    max_iterations: int = MAX_ITERATIONS,
) -> dict[str, object]:
    """Tune one record by a bounded two-coordinate search.

    ``measure_fn`` is injectable so tests can exercise every branch without
    timing real code.  It may accept ``record`` or ``record, knobs`` and return
    either the validator-style measurement mapping or a ratio pair.

    Query count moves geometrically. Stride moves one position at a time so
    interior peak slack can be spent to repair an excessive time ratio (and
    conversely), instead of freezing the representation coordinate too early.
    """
    if not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError(f"max_iterations must be between 1 and {MAX_ITERATIONS}")
    current = dict(record)
    trajectory: list[dict[str, object]] = []
    best_record = current
    best_score = float("inf")
    best_iteration = 0
    last_reason = "no_measurement"
    seen_knob_states: set[tuple[tuple[str, int], ...]] = set()

    for iteration in range(1, max_iterations + 1):
        knobs = _knobs(current)
        seen_knob_states.add(tuple(sorted(knobs.items())))
        raw = _invoke_measurement(measure_fn, current, knobs)
        timings, peaks, ratios, floors, error = _measurement_parts(raw)
        in_band = _inside_band(timings, peaks, ratios)
        in_interior = bool(
            ratios
            and INTERIOR_TIME_BAND[0] <= ratios[0] <= INTERIOR_TIME_BAND[1]
            and INTERIOR_PEAK_BAND[0] <= ratios[1] <= INTERIOR_PEAK_BAND[1]
        )
        accepted = in_band and in_interior and not floors and error is None
        entry: dict[str, object] = {
            "id": current.get("id"),
            "iteration": iteration,
            "knobs": dict(sorted(knobs.items())),
            "timings": timings,
            "peaks": peaks,
            "time_ratio": ratios[0] if ratios else None,
            "peak_ratio": ratios[1] if ratios else None,
            "floor_violations": floors,
            "error": error,
            "in_validator_band": in_band,
            "in_interior": accepted,
        }
        trajectory.append(entry)
        score = _score(ratios, floors, error)
        if score < best_score:
            best_score = score
            best_record = current
            best_iteration = iteration
        if accepted:
            return {
                "record": current,
                "status": "tuned",
                "reason": None,
                "iterations": iteration,
                "best_iteration": iteration,
                "trajectory": trajectory,
            }

        if error:
            last_reason = f"measurement_error:{error}"
            record_count = _bounded(knobs["record_count"] // 2, RECORD_BOUNDS)
            next_knobs = {**knobs, "record_count": record_count}
            current = rebuild_instance(current, next_knobs)
            continue
        if ratios is None:
            last_reason = "ratios_unavailable"
            current = rebuild_instance(
                current,
                {
                    **knobs,
                    "record_count": _bounded(
                        knobs["record_count"] * 2, RECORD_BOUNDS
                    ),
                },
            )
            continue

        time_ratio, peak_ratio = ratios
        next_knobs = dict(knobs)
        if peak_ratio > INTERIOR_PEAK_BAND[1]:
            next_knobs["stride"] = _bounded(
                knobs["stride"] + 1, STRIDE_BOUNDS
            )
        elif peak_ratio < INTERIOR_PEAK_BAND[0]:
            next_knobs["stride"] = _bounded(
                knobs["stride"] - 1, STRIDE_BOUNDS
            )
        elif time_ratio < INTERIOR_TIME_BAND[0]:
            if peak_ratio > INTERIOR_PEAK_BAND[0] + 0.05:
                # Spend peak slack by adding bounded replay before growing the
                # workload coordinate whose ratio effect can saturate.
                next_knobs["stride"] = _bounded(
                    knobs["stride"] + 1, STRIDE_BOUNDS
                )
            else:
                step = max(1, knobs["queries_per_record"] // 2)
                next_knobs["queries_per_record"] = _bounded(
                    knobs["queries_per_record"] + step, QUERY_BOUNDS
                )
        elif time_ratio > INTERIOR_TIME_BAND[1]:
            if peak_ratio < INTERIOR_PEAK_BAND[1] - 0.05:
                # Spend peak slack in the other direction to reduce replay.
                next_knobs["stride"] = _bounded(
                    knobs["stride"] - 1, STRIDE_BOUNDS
                )
            else:
                step = max(1, knobs["queries_per_record"] // 4)
                next_knobs["queries_per_record"] = _bounded(
                    knobs["queries_per_record"] - step, QUERY_BOUNDS
                )

        floor_set = set(floors)
        if any(
            item.endswith("too_quick_to_measure")
            or item.endswith("timing_unstable")
            for item in floor_set
        ):
            next_knobs["record_count"] = _bounded(
                knobs["record_count"] * 2, RECORD_BOUNDS
            )
        if any(item.endswith("over_time_ceiling") for item in floor_set):
            next_knobs["record_count"] = _bounded(
                knobs["record_count"] // 2, RECORD_BOUNDS
            )
        if "peak_below_allocator_noise" in floor_set:
            next_knobs["field_count"] = _bounded(
                knobs["field_count"] * 2, FIELD_COUNT_BOUNDS
            )
            next_knobs["record_count"] = _bounded(
                knobs["record_count"] * 2, RECORD_BOUNDS
            )

        proposed_state = tuple(sorted(next_knobs.items()))
        if proposed_state in seen_knob_states and next_knobs != knobs:
            # A discrete stride can straddle the interior boundary. Do not
            # oscillate: move a second semantic coordinate that affects the
            # failing axis while retaining the current stride.
            next_knobs = dict(knobs)
            if peak_ratio > INTERIOR_PEAK_BAND[1]:
                step = max(8, knobs["field_count"] // 4)
                next_knobs["field_count"] = _bounded(
                    knobs["field_count"] + step, FIELD_COUNT_BOUNDS
                )
            elif peak_ratio < INTERIOR_PEAK_BAND[0]:
                step = max(8, knobs["field_count"] // 4)
                next_knobs["field_count"] = _bounded(
                    knobs["field_count"] - step, FIELD_COUNT_BOUNDS
                )
            elif time_ratio > INTERIOR_TIME_BAND[1]:
                step = max(1, knobs["queries_per_record"] // 4)
                next_knobs["queries_per_record"] = _bounded(
                    knobs["queries_per_record"] - step, QUERY_BOUNDS
                )
            else:
                step = max(1, knobs["queries_per_record"] // 2)
                next_knobs["queries_per_record"] = _bounded(
                    knobs["queries_per_record"] + step, QUERY_BOUNDS
                )

        if next_knobs == knobs:
            if time_ratio < TARGET_TIME_RATIO:
                next_knobs["queries_per_record"] = _bounded(
                    knobs["queries_per_record"] + 1, QUERY_BOUNDS
                )
            elif time_ratio > TARGET_TIME_RATIO:
                next_knobs["queries_per_record"] = _bounded(
                    knobs["queries_per_record"] - 1, QUERY_BOUNDS
                )
            elif peak_ratio > TARGET_PEAK_RATIO:
                next_knobs["stride"] = _bounded(
                    knobs["stride"] + 1, STRIDE_BOUNDS
                )
            else:
                next_knobs["stride"] = _bounded(
                    knobs["stride"] - 1, STRIDE_BOUNDS
                )
        current = rebuild_instance(current, next_knobs)
        reason_parts = []
        if not in_band:
            reason_parts.append("outside_band")
        if floors:
            reason_parts.append("floors:" + ",".join(floors))
        last_reason = ";".join(reason_parts) or "target_not_reached"

    return {
        "record": best_record,
        "status": "untunable",
        "reason": last_reason,
        "iterations": max_iterations,
        "best_iteration": best_iteration,
        "trajectory": trajectory,
    }


def tune_instances(
    records: Sequence[Mapping[str, object]],
    *,
    measure_fn: MeasurementFunction = measure_instance,
    max_iterations: int = MAX_ITERATIONS,
) -> list[dict[str, object]]:
    """Tune a deterministic sequence in input order."""
    return [
        tune_instance(
            record, measure_fn=measure_fn, max_iterations=max_iterations
        )
        for record in records
    ]


def write_tuner_log(path: str | Path, results: Sequence[Mapping[str, Any]]) -> None:
    """Write one trajectory row per composed instance."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for result in results:
            record = result.get("record", {})
            row = {
                "id": record.get("id") if isinstance(record, Mapping) else None,
                "status": result.get("status"),
                "reason": result.get("reason"),
                "iterations": result.get("iterations"),
                "best_iteration": result.get("best_iteration"),
                "trajectory": result.get("trajectory", []),
            }
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
