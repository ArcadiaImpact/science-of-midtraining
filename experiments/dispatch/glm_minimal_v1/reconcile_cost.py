#!/usr/bin/env python3
"""Reconcile ``telemetry.jsonl`` with the GLM scaling cost constants.

All calculations are local and deterministic.  Rates are recomputed from the
primitive ``seconds``, ``steps``, and ``tokens`` fields rather than trusting
the derived telemetry fields.  Failed and zero-step training rows remain in
the wall/cost accounting but are excluded from throughput estimates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ACTIVE_PARAMETERS = 12_000_000_000
FLOPS_PER_TOKEN = 6 * ACTIVE_PARAMETERS
PEAK_BF16_TFLOPS = {"H200": 989.0, "B300": 2250.0}
TRAINING_PHASES = ("midtrain", "ift", "aft")
MULTI_GPU_EFF_PER_DOUBLING = 0.985

REQUIRED_FIELDS = {
    "run_id",
    "arm",
    "phase",
    "gpu_type",
    "n_gpus",
    "started_at",
    "ended_at",
    "seconds",
    "steps",
    "tokens",
    "s_per_step",
    "tokens_per_s",
    "mb_per_s",
    "free_disk_gb",
    "notes",
}

# Constants in experiments/dispatch/scaling_v1 at Task-7 implementation
# time.  They are comparison anchors only; no source file is imported or
# modified by this CPU-only tool.
CURRENT_COST_MODEL = {
    "H200": {
        "tok_s_gpu_midtrain": 958.0,
        "tok_s_gpu_ift": 971.0,
        "aft_s_per_step": 14.0,
    },
}
CURRENT_MINIMAL_MODEL = {
    "H200": {"tok_s": 7660.0, "aft_s_per_step": 14.0},
    "B300": {"tok_s": 17500.0, "aft_s_per_step": 6.1},
}
CURRENT_SHARED = {
    "consolidate_hr": 0.12,
    "merge_hr": 0.06,
    "dataprep_hr": 0.50,
    "eval_hr": 0.42,
    "egress_gbyte_s": 0.50,
}


def canonical_gpu_type(value: str) -> str:
    normalized = value.upper().replace("-", "").replace("_", "")
    if "H200" in normalized:
        return "H200"
    if "B300" in normalized:
        return "B300"
    raise ValueError(f"unsupported GPU type {value!r}; expected H200 or B300")


def _finite_number(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class ReconcileConfig:
    telemetry_path: Path
    gpu_type: str
    usd_per_gpu_hour: float
    pod_gpus: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "telemetry_path", Path(self.telemetry_path))
        object.__setattr__(self, "gpu_type", canonical_gpu_type(self.gpu_type))
        _finite_number(self.usd_per_gpu_hour, name="usd_per_gpu_hour", minimum=1e-12)
        if (
            isinstance(self.pod_gpus, bool)
            or not isinstance(self.pod_gpus, int)
            or self.pod_gpus < 1
        ):
            raise ValueError("pod_gpus must be a positive integer")

    @classmethod
    def from_json(cls, path: Path) -> "ReconcileConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object")
        allowed = {"telemetry_path", "gpu_type", "usd_per_gpu_hour", "pod_gpus"}
        unknown = set(payload) - allowed
        missing = {"telemetry_path", "gpu_type", "usd_per_gpu_hour"} - set(payload)
        if unknown or missing:
            raise ValueError(
                f"{path}: config mismatch: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        telemetry = Path(payload["telemetry_path"])
        if not telemetry.is_absolute():
            telemetry = path.parent / telemetry
        return cls(
            telemetry_path=telemetry,
            gpu_type=str(payload["gpu_type"]),
            usd_per_gpu_hour=payload["usd_per_gpu_hour"],
            pod_gpus=payload.get("pod_gpus", 8),
        )


@dataclass(frozen=True, slots=True)
class RowMeasurement:
    line_number: int
    arm: str | None
    phase: str
    seconds: float
    n_gpus: int
    active_gpu_cost_usd: float
    s_per_step: float | None
    tokens_per_s: float | None
    mfu: float | None
    mb_per_s: float | None
    rate_eligible: bool
    notes: str | None


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    phase: str
    rows: int
    seconds: float
    steps: float
    tokens: float | None
    s_per_step: float
    tokens_per_s: float | None
    tokens_per_s_per_gpu: float | None
    cost_model_tokens_per_s_gpu: float | None
    mfu: float | None


@dataclass(frozen=True, slots=True)
class PhaseCost:
    phase: str
    rows: int
    summed_seconds: float
    wall_clock_seconds: float
    active_gpu_cost_usd: float
    pod_equivalent_cost_usd: float


@dataclass(frozen=True, slots=True)
class Reconciliation:
    config: ReconcileConfig
    rows: tuple[Mapping[str, Any], ...]
    measurements: tuple[RowMeasurement, ...]
    training: Mapping[str, TrainingSummary]
    phase_costs: Mapping[str, PhaseCost]
    publish_rates: tuple[RowMeasurement, ...]
    aggregate_publish_mb_per_s: float | None
    wall_clock_seconds: float
    active_gpu_cost_usd: float
    pod_cost_usd: float


def load_telemetry(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing telemetry fields {sorted(missing)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: telemetry is empty")
    return rows


def _optional_number(value: object, *, name: str) -> float | None:
    return None if value is None else _finite_number(value, name=name)


def _optional_nonnegative_int(value: object, *, name: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{name} must be a non-negative integer or null, got {value!r}"
        )
    return value


def _failed(row: Mapping[str, Any]) -> bool:
    notes = row.get("notes")
    return isinstance(notes, str) and "FAILED " in notes


def _measurement(
    row: Mapping[str, Any], *, line_number: int, config: ReconcileConfig
) -> RowMeasurement:
    phase = row.get("phase")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError(f"line {line_number}: phase must be a non-empty string")
    arm = row.get("arm")
    if arm is not None and not isinstance(arm, str):
        raise ValueError(f"line {line_number}: arm must be a string or null")
    seconds = _finite_number(row.get("seconds"), name=f"line {line_number} seconds")
    n_gpus = _optional_nonnegative_int(
        row.get("n_gpus"), name=f"line {line_number} n_gpus"
    )
    recorded_gpu = row.get("gpu_type")
    if isinstance(recorded_gpu, str) and recorded_gpu.strip().lower() != "unknown":
        if canonical_gpu_type(recorded_gpu) != config.gpu_type:
            raise ValueError(
                f"line {line_number}: telemetry GPU {recorded_gpu!r} conflicts "
                f"with configured {config.gpu_type}"
            )

    steps = _optional_number(row.get("steps"), name=f"line {line_number} steps")
    tokens = _optional_number(row.get("tokens"), name=f"line {line_number} tokens")
    notes = row.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"line {line_number}: notes must be a string or null")
    rate_eligible = bool(
        phase in TRAINING_PHASES
        and not _failed(row)
        and seconds > 0
        and steps is not None
        and steps > 0
        and n_gpus > 0
    )
    s_per_step = seconds / steps if rate_eligible and steps is not None else None
    tokens_per_s = (
        tokens / seconds
        if rate_eligible and tokens is not None and seconds > 0
        else None
    )
    mfu = (
        tokens_per_s
        * FLOPS_PER_TOKEN
        / (n_gpus * PEAK_BF16_TFLOPS[config.gpu_type] * 1e12)
        if tokens_per_s is not None
        else None
    )
    mb_per_s = _optional_number(
        row.get("mb_per_s"), name=f"line {line_number} mb_per_s"
    )
    return RowMeasurement(
        line_number=line_number,
        arm=arm,
        phase=phase,
        seconds=seconds,
        n_gpus=n_gpus,
        active_gpu_cost_usd=(seconds / 3600 * n_gpus * config.usd_per_gpu_hour),
        s_per_step=s_per_step,
        tokens_per_s=tokens_per_s,
        mfu=mfu,
        mb_per_s=mb_per_s,
        rate_eligible=rate_eligible,
        notes=notes,
    )


def _parse_started_at(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _intervals(rows: Sequence[Mapping[str, Any]]) -> list[tuple[datetime, datetime]]:
    intervals: list[tuple[datetime, datetime]] = []
    for row in rows:
        start = _parse_started_at(row.get("started_at"))
        if start is None:
            return []
        seconds = _finite_number(row.get("seconds"), name="seconds")
        intervals.append((start, start + timedelta(seconds=seconds)))
    return intervals


def _union_seconds(rows: Sequence[Mapping[str, Any]]) -> float:
    intervals = sorted(_intervals(rows))
    if not intervals:
        return sum(float(row["seconds"]) for row in rows)
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += (current_end - current_start).total_seconds()
            current_start, current_end = start, end
    return total + (current_end - current_start).total_seconds()


def _span_seconds(rows: Sequence[Mapping[str, Any]]) -> float:
    intervals = _intervals(rows)
    if not intervals:
        return _union_seconds(rows)
    return (
        max(end for _, end in intervals) - min(start for start, _ in intervals)
    ).total_seconds()


def _scaling_efficiency(n_gpus: int) -> float:
    return MULTI_GPU_EFF_PER_DOUBLING ** math.log2(max(n_gpus, 1))


def _training_summary(
    phase: str,
    rows: Sequence[Mapping[str, Any]],
    measurements: Sequence[RowMeasurement],
    config: ReconcileConfig,
) -> TrainingSummary | None:
    pairs = [
        (row, measurement)
        for row, measurement in zip(rows, measurements, strict=True)
        if measurement.phase == phase and measurement.rate_eligible
    ]
    if not pairs:
        return None
    seconds = sum(measurement.seconds for _, measurement in pairs)
    steps = sum(float(row["steps"]) for row, _ in pairs)
    token_pairs = [
        (row, measurement)
        for row, measurement in pairs
        if row.get("tokens") is not None
    ]
    tokens: float | None = None
    tokens_per_s: float | None = None
    per_gpu: float | None = None
    cost_model_per_gpu: float | None = None
    mfu: float | None = None
    if token_pairs:
        token_seconds = sum(measurement.seconds for _, measurement in token_pairs)
        tokens = sum(float(row["tokens"]) for row, _ in token_pairs)
        gpu_seconds = sum(
            measurement.seconds * measurement.n_gpus for _, measurement in token_pairs
        )
        peak_flop_seconds = sum(
            measurement.seconds
            * measurement.n_gpus
            * PEAK_BF16_TFLOPS[config.gpu_type]
            * 1e12
            for _, measurement in token_pairs
        )
        tokens_per_s = tokens / token_seconds
        per_gpu = tokens / gpu_seconds
        cost_model_gpu_seconds = sum(
            measurement.seconds
            * measurement.n_gpus
            * _scaling_efficiency(measurement.n_gpus)
            for _, measurement in token_pairs
        )
        cost_model_per_gpu = tokens / cost_model_gpu_seconds
        mfu = tokens * FLOPS_PER_TOKEN / peak_flop_seconds
    return TrainingSummary(
        phase=phase,
        rows=len(pairs),
        seconds=seconds,
        steps=steps,
        tokens=tokens,
        s_per_step=seconds / steps,
        tokens_per_s=tokens_per_s,
        tokens_per_s_per_gpu=per_gpu,
        cost_model_tokens_per_s_gpu=cost_model_per_gpu,
        mfu=mfu,
    )


def reconcile(
    rows: Sequence[Mapping[str, Any]], config: ReconcileConfig
) -> Reconciliation:
    if not rows:
        raise ValueError("telemetry rows must be non-empty")
    measurements = tuple(
        _measurement(row, line_number=index, config=config)
        for index, row in enumerate(rows, start=1)
    )
    training = {
        phase: summary
        for phase in TRAINING_PHASES
        if (summary := _training_summary(phase, rows, measurements, config)) is not None
    }

    rows_by_phase: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    measurements_by_phase: dict[str, list[RowMeasurement]] = defaultdict(list)
    for row, measurement in zip(rows, measurements, strict=True):
        rows_by_phase[measurement.phase].append(row)
        measurements_by_phase[measurement.phase].append(measurement)
    phase_costs: dict[str, PhaseCost] = {}
    for phase, phase_rows in rows_by_phase.items():
        phase_measurements = measurements_by_phase[phase]
        wall = _union_seconds(phase_rows)
        phase_costs[phase] = PhaseCost(
            phase=phase,
            rows=len(phase_rows),
            summed_seconds=sum(item.seconds for item in phase_measurements),
            wall_clock_seconds=wall,
            active_gpu_cost_usd=sum(
                item.active_gpu_cost_usd for item in phase_measurements
            ),
            pod_equivalent_cost_usd=(
                wall / 3600 * config.pod_gpus * config.usd_per_gpu_hour
            ),
        )

    publishes = tuple(
        measurement
        for measurement in measurements
        if measurement.phase == "publish" and measurement.mb_per_s is not None
    )
    successful_publishes = [
        item
        for item in publishes
        if item.seconds > 0 and not (item.notes and "FAILED " in item.notes)
    ]
    aggregate_publish = None
    if successful_publishes:
        aggregate_publish = sum(
            item.mb_per_s * item.seconds  # type: ignore[operator]
            for item in successful_publishes
        ) / sum(item.seconds for item in successful_publishes)

    wall = _span_seconds(rows)
    return Reconciliation(
        config=config,
        rows=tuple(rows),
        measurements=measurements,
        training=training,
        phase_costs=phase_costs,
        publish_rates=publishes,
        aggregate_publish_mb_per_s=aggregate_publish,
        wall_clock_seconds=wall,
        active_gpu_cost_usd=sum(item.active_gpu_cost_usd for item in measurements),
        pod_cost_usd=(wall / 3600 * config.pod_gpus * config.usd_per_gpu_hour),
    )


def _successful_phase_measurements(
    result: Reconciliation, phase: str
) -> list[RowMeasurement]:
    return [
        item
        for item in result.measurements
        if item.phase == phase
        and item.seconds > 0
        and not (item.notes and "FAILED " in item.notes)
    ]


def _mean_phase_hours(result: Reconciliation, phase: str) -> float | None:
    rows = _successful_phase_measurements(result, phase)
    return sum(row.seconds for row in rows) / len(rows) / 3600 if rows else None


def _combined_phase_hours(
    result: Reconciliation, phases: Sequence[str]
) -> float | None:
    present = [
        result.phase_costs[phase] for phase in phases if phase in result.phase_costs
    ]
    if not present:
        return None
    return sum(item.wall_clock_seconds for item in present) / 3600


def _py_number(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def constant_lines(result: Reconciliation) -> list[str]:
    """Return paste-ready measured constants for both scaling-model files."""

    gpu = result.config.gpu_type
    midtrain = result.training.get("midtrain")
    ift = result.training.get("ift")
    aft = result.training.get("aft")
    merge_hr = _mean_phase_hours(result, "merge")
    dataprep_hr = _combined_phase_hours(result, ("setup", "data_fetch"))
    evaluation = result.phase_costs.get("eval")
    egress = result.aggregate_publish_mb_per_s

    lines = [
        "# ../scaling_v1/cost_model.py — MODELS['glm45_air']",
        f'train_gpu="{gpu}", n_train_gpus={result.config.pod_gpus},',
        f'aft_gpu="{gpu}", n_aft_gpus=4,',
        f'eval_gpu="{gpu}", n_eval_gpus=2,',
    ]
    if midtrain and midtrain.cost_model_tokens_per_s_gpu is not None:
        lines.append(
            f"tok_s_gpu_midtrain={_py_number(midtrain.cost_model_tokens_per_s_gpu)},"
        )
    if ift and ift.cost_model_tokens_per_s_gpu is not None:
        lines.append(f"tok_s_gpu_ift={_py_number(ift.cost_model_tokens_per_s_gpu)},")
    if aft:
        lines.append(f"aft_s_per_step={_py_number(aft.s_per_step)},")
    if merge_hr is not None:
        lines.append(f"consolidate_hr={_py_number(merge_hr)},")
    if egress is not None:
        lines.extend(
            [
                "# ../scaling_v1/cost_model.py — Overheads",
                f"egress_gbyte_s: float = {_py_number(egress / 1000)}",
            ]
        )

    token_summaries = [
        item for item in (midtrain, ift) if item is not None and item.tokens is not None
    ]
    blended_tok_s = None
    if token_summaries:
        blended_tok_s = sum(item.tokens or 0.0 for item in token_summaries) / sum(
            item.seconds for item in token_summaries
        )
    optimizer = "8-bit AdamW" if gpu == "H200" else "fp32 AdamW"
    symbol = "H200" if gpu == "H200" else "B300_MID"
    label = "8xH200 SECURE" if gpu == "H200" else "8xB300 (measured)"
    lines.append("# ../scaling_v1/minimal_glm_run.py — measured hardware row")
    if blended_tok_s is not None and aft is not None:
        lines.append(
            f'{symbol} = Hw("{label}", {_py_number(result.config.usd_per_gpu_hour)}, '
            f"{result.config.pod_gpus}, {_py_number(blended_tok_s)}, "
            f'{_py_number(aft.s_per_step)}, "{optimizer}", '
            '"MEASURED from telemetry")'
        )
    lines.append("# ../scaling_v1/minimal_glm_run.py — Design")
    if dataprep_hr is not None:
        lines.append(f"dataprep_hr: float = {_py_number(dataprep_hr)}")
    if merge_hr is not None:
        lines.append(f"merge_hr: float = {_py_number(merge_hr)}")
    if evaluation is not None:
        lines.extend(
            [
                "# Four endpoint workers are concurrent; telemetry cannot split load "
                "from generation.",
                f"eval_load_hr: float = {_py_number(evaluation.wall_clock_seconds / 3600)}",
                "eval_compute_hr: float = 0.0",
            ]
        )
    if egress is not None:
        lines.append(f"egress_gbyte_s: float = {_py_number(egress / 1000)}")
    lines.append(
        "# setup_hr and download_hr are unchanged: pod creation, setup_pod.sh, "
        "and most of the overlapped base prefetch occur before telemetry starts; "
        "the download phase measures only the residual join."
    )
    return lines


def _comparison(label: str, measured: float, current: float) -> str:
    ratio = measured / current
    status = (
        "consistent"
        if math.isclose(measured, current, rel_tol=0.05, abs_tol=1e-12)
        else "CONTRADICTION"
    )
    return (
        f"{status}: {label}: measured {_py_number(measured)} vs current "
        f"{_py_number(current)} ({ratio:.2f}x)"
    )


def comparisons(result: Reconciliation) -> list[str]:
    gpu = result.config.gpu_type
    out: list[str] = []
    anchors = CURRENT_COST_MODEL.get(gpu)
    midtrain = result.training.get("midtrain")
    ift = result.training.get("ift")
    aft = result.training.get("aft")
    if anchors and midtrain and midtrain.cost_model_tokens_per_s_gpu is not None:
        out.append(
            _comparison(
                "cost_model tok_s_gpu_midtrain",
                midtrain.cost_model_tokens_per_s_gpu,
                anchors["tok_s_gpu_midtrain"],
            )
        )
    if anchors and ift and ift.cost_model_tokens_per_s_gpu is not None:
        out.append(
            _comparison(
                "cost_model tok_s_gpu_ift",
                ift.cost_model_tokens_per_s_gpu,
                anchors["tok_s_gpu_ift"],
            )
        )
    if anchors and aft:
        out.append(
            _comparison(
                "cost_model aft_s_per_step",
                aft.s_per_step,
                anchors["aft_s_per_step"],
            )
        )
    merge_hr = _mean_phase_hours(result, "merge")
    if merge_hr is not None:
        out.extend(
            [
                _comparison(
                    "cost_model consolidate_hr",
                    merge_hr,
                    CURRENT_SHARED["consolidate_hr"],
                ),
                _comparison(
                    "minimal_glm_run merge_hr",
                    merge_hr,
                    CURRENT_SHARED["merge_hr"],
                ),
            ]
        )
    if result.aggregate_publish_mb_per_s is not None:
        out.append(
            _comparison(
                "egress MB/s",
                result.aggregate_publish_mb_per_s,
                CURRENT_SHARED["egress_gbyte_s"] * 1000,
            )
        )
    minimal = CURRENT_MINIMAL_MODEL[gpu]
    token_summaries = [
        item for item in (midtrain, ift) if item is not None and item.tokens is not None
    ]
    for item in token_summaries:
        assert item.tokens_per_s is not None
        out.append(
            _comparison(
                f"minimal_glm_run Hw.tok_s vs {item.phase}",
                item.tokens_per_s,
                minimal["tok_s"],
            )
        )
    if token_summaries:
        blended = sum(item.tokens or 0.0 for item in token_summaries) / sum(
            item.seconds for item in token_summaries
        )
        out.append(
            _comparison(
                "minimal_glm_run Hw.tok_s blended replacement",
                blended,
                minimal["tok_s"],
            )
        )
    if aft:
        out.append(
            _comparison(
                "minimal_glm_run Hw.aft_s_per_step",
                aft.s_per_step,
                minimal["aft_s_per_step"],
            )
        )
    dataprep_hr = _combined_phase_hours(result, ("setup", "data_fetch"))
    if dataprep_hr is not None:
        out.append(
            _comparison(
                "minimal_glm_run dataprep_hr",
                dataprep_hr,
                CURRENT_SHARED["dataprep_hr"],
            )
        )
    for phase, current_key, label in (
        ("eval", "eval_hr", "minimal_glm_run concurrent eval wall"),
    ):
        if phase in result.phase_costs:
            out.append(
                _comparison(
                    label,
                    result.phase_costs[phase].wall_clock_seconds / 3600,
                    CURRENT_SHARED[current_key],
                )
            )
    return out


def format_report(result: Reconciliation) -> str:
    lines = [
        f"Telemetry: {result.config.telemetry_path}",
        (
            f"GPU: {result.config.gpu_type}; pod={result.config.pod_gpus} GPUs; "
            f"rate=${result.config.usd_per_gpu_hour:.4f}/GPU-h"
        ),
        "",
        "TRAINING ROWS",
    ]
    for item in result.measurements:
        if item.phase not in TRAINING_PHASES:
            continue
        identity = f"{item.phase}/{item.arm or '-'}"
        if not item.rate_eligible:
            lines.append(
                f"{identity}: {item.seconds:.1f}s; rate unavailable "
                "(failed, zero-step, zero-second, or zero-GPU row)"
            )
            continue
        tokens = (
            "n/a" if item.tokens_per_s is None else f"{item.tokens_per_s:,.1f} tok/s"
        )
        mfu = "n/a" if item.mfu is None else f"{100 * item.mfu:.2f}% MFU"
        lines.append(f"{identity}: {item.s_per_step:.3f} s/step; {tokens}; {mfu}")
    lines.extend(["", "TRAINING PHASE AGGREGATES"])
    for phase in TRAINING_PHASES:
        item = result.training.get(phase)
        if item is None:
            lines.append(f"{phase}: no completed measured rows")
            continue
        tokens = (
            "n/a" if item.tokens_per_s is None else f"{item.tokens_per_s:,.1f} tok/s"
        )
        mfu = "n/a" if item.mfu is None else f"{100 * item.mfu:.2f}% MFU"
        lines.append(
            f"{phase}: {item.s_per_step:.3f} s/step; {tokens}; {mfu}; "
            f"completed rows={item.rows}"
        )
    if "aft" in result.training and result.training["aft"].tokens is None:
        lines.append(
            "AFT tokens/s and MFU are unavailable: chain telemetry records "
            "tokens=null for the variable-length, unpacked AFT stream."
        )

    lines.extend(["", "PUBLISHES"])
    if not result.publish_rates:
        lines.append("none measured")
    for item in result.publish_rates:
        lines.append(
            f"line {item.line_number} {item.arm or '-'}: "
            f"{item.mb_per_s:.1f} MB/s; {item.seconds:.1f}s; {item.notes or ''}"
        )
    if result.aggregate_publish_mb_per_s is not None:
        lines.append(
            "aggregate (byte/time weighted): "
            f"{result.aggregate_publish_mb_per_s:.1f} MB/s"
        )

    lines.extend(
        [
            "",
            "PHASE WALL/COST",
            (
                "Pod-equivalent phase costs use the full pod rate. Phase rows can "
                "overlap, so only the campaign pod total is additive."
            ),
        ]
    )
    for phase, item in result.phase_costs.items():
        lines.append(
            f"{phase}: wall={item.wall_clock_seconds / 3600:.4f}h; "
            f"pod-equivalent=${item.pod_equivalent_cost_usd:.2f}; "
            f"active-GPU=${item.active_gpu_cost_usd:.2f}; rows={item.rows}"
        )
    lines.extend(
        [
            (
                f"TOTAL: wall={result.wall_clock_seconds / 3600:.4f}h; "
                f"pod bill=${result.pod_cost_usd:.2f}; "
                f"active-GPU equivalent=${result.active_gpu_cost_usd:.2f}"
            ),
            "NOTE: total starts at the first telemetry row; pod creation and "
            "setup_pod.sh are not instrumented. The download phase is only the "
            "residual join of setup's background prefetch.",
            "",
            "CURRENT-CONSTANT CHECK",
            *(comparisons(result) or ["no comparable measured constants"]),
            "",
            "PASTE-READY CONSTANTS",
            *constant_lines(result),
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path, nargs="?")
    parser.add_argument("--gpu-type")
    parser.add_argument("--usd-per-gpu-hour", type=float)
    parser.add_argument("--pod-gpus", type=int, default=8)
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "JSON ReconcileConfig with telemetry_path, gpu_type, "
            "usd_per_gpu_hour, and optional pod_gpus"
        ),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> ReconcileConfig:
    if args.config is not None:
        if (
            args.telemetry is not None
            or args.gpu_type is not None
            or args.usd_per_gpu_hour is not None
            or args.pod_gpus != 8
        ):
            raise ValueError("--config cannot be combined with direct inputs")
        return ReconcileConfig.from_json(args.config)
    if args.telemetry is None or args.gpu_type is None or args.usd_per_gpu_hour is None:
        raise ValueError(
            "provide --config, or TELEMETRY --gpu-type GPU --usd-per-gpu-hour RATE"
        )
    return ReconcileConfig(
        telemetry_path=args.telemetry,
        gpu_type=args.gpu_type,
        usd_per_gpu_hour=args.usd_per_gpu_hour,
        pod_gpus=args.pod_gpus,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        result = reconcile(load_telemetry(config.telemetry_path), config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(format_report(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
