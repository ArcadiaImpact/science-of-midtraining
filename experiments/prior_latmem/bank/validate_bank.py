"""Execution gates, splits, and manifests for the prior-latmem problem bank.

The validator deliberately keeps policy decisions in small pure functions:
``lint_z_silence`` is reusable by Stage 4, ``passes_separation`` can be tested
with fixed measurements, and ``seeded_splits`` has no filesystem dependency.
Only authored-code execution crosses into :func:`sandbox.run_sandboxed`.
"""

from __future__ import annotations

import ast
import json
import math
import platform
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from scimt.config import parse, save

try:
    from .sandbox import run_sandboxed
    from .taxonomy import PATTERN_BY_KEY
except ImportError:  # pragma: no cover - direct script invocation
    from sandbox import run_sandboxed  # type: ignore
    from taxonomy import PATTERN_BY_KEY  # type: ignore


Z_SILENCE_TERMS = (
    "latency",
    "memory",
    "fast",
    "slow",
    "footprint",
    "efficien",
    "optimiz",
)
_Z_WORD_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:latency|memory|fast|slow|footprint|efficien\w*|optimiz\w*)(?![A-Za-z0-9])"
)
_IDENTIFIER_COMPONENT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|[0-9]|_|$)|[A-Z]?[a-z]+|[0-9]+"
)


def lint_z_silence(text: str) -> list[str]:
    """Return distinct forbidden terms found in prose or Python identifiers.

    The prose scan catches words in statements, comments, and string literals.
    The identifier scan splits names such as ``cached_memory`` so a forbidden
    component cannot be hidden behind an underscore. Results are ordered by
    first occurrence and are empty for a clean string.
    """
    hits: list[str] = []
    for match in _Z_WORD_RE.finditer(text):
        word = match.group(0).lower()
        normalized = next(
            (term for term in Z_SILENCE_TERMS if word == term or word.startswith(term)),
            word,
        )
        if normalized not in hits:
            hits.append(normalized)
    # The text regex treats underscores as part of a word. Split identifiers
    # separately so ``small_memory_table`` and non-initial camelCase stems are
    # also rejected, while ``breakfast_menu`` remains a clean identifier.
    for identifier in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        for component in _IDENTIFIER_COMPONENT_RE.findall(identifier):
            component = component.lower()
            normalized = next(
                (
                    term
                    for term in Z_SILENCE_TERMS
                    if component == term or component.startswith(term)
                ),
                None,
            )
            if normalized and normalized not in hits:
                hits.append(normalized)
    return hits


def _median(values: Sequence[float]) -> float | None:
    if not values or any(not math.isfinite(float(value)) for value in values):
        return None
    return float(statistics.median(float(value) for value in values))


def separation_ratios(
    timings: Mapping[str, Sequence[float]],
    peaks: Mapping[str, Sequence[float]],
) -> tuple[float, float] | None:
    """Return ``(time_ratio, peak_ratio)`` of memory_solution over speed_solution."""
    speed_time = _median(timings.get("speed_solution", ()))
    memory_time = _median(timings.get("memory_solution", ()))
    speed_peak = _median(peaks.get("speed_solution", ()))
    memory_peak = _median(peaks.get("memory_solution", ()))
    if None in (speed_time, memory_time, speed_peak, memory_peak):
        return None
    if speed_time <= 0 or speed_peak <= 0 or memory_time < 0 or memory_peak < 0:
        return None
    return memory_time / speed_time, memory_peak / speed_peak


def passes_separation(
    timings: Mapping[str, Sequence[float]],
    peaks: Mapping[str, Sequence[float]],
    *,
    min_speedup: float = 1.3,
    max_memory_ratio: float = 0.7,
    max_speedup: float = 4.0,
    min_memory_ratio: float = 0.25,
) -> bool:
    """Return the pure measured gate decision for a tradeoff pair.

    ``timings`` and ``peaks`` map ``speed_solution`` and ``memory_solution``
    to the three large-scale measurements. The ratios are
    ``median(memory) / median(speed)`` on each axis, so a valid tradeoff has a
    time ratio above 1 and a peak ratio below 1.

    The gate is a **band**, not two one-sided floors (Sid, 2026-07-28, after the
    probe in ``probe_v1/PROBE.md``): the pre-registered ``min_speedup`` /
    ``max_memory_ratio`` thresholds only ask "does it separate at all", and
    every instance the probe measured separated *lopsidedly* — a heap-lean side
    569x slower, or a heap saving of 100x for 1.1x time. Neither is an exchange
    rate a competent engineer would deliberate over, and at f=1.0 the heap-lean
    side is what the model is trained to write. ``max_speedup`` and
    ``min_memory_ratio`` bound the other end so surviving instances populate the
    interior of the exchange-rate space the psychometric grid actually probes.
    """
    if min_speedup <= 0 or not 0 < max_memory_ratio <= 1:
        raise ValueError("invalid separation thresholds")
    if max_speedup <= min_speedup:
        raise ValueError("max_speedup must exceed min_speedup")
    if not 0 <= min_memory_ratio < max_memory_ratio:
        raise ValueError("min_memory_ratio must be below max_memory_ratio")
    ratios = separation_ratios(timings, peaks)
    if ratios is None:
        return False
    time_ratio, peak_ratio = ratios
    return (
        min_speedup <= time_ratio <= max_speedup
        and min_memory_ratio <= peak_ratio <= max_memory_ratio
    )


# Statement-prose defects the probe found by reading (probe_v1/PROBE.md).
#: The theme arrived as a template slot in 47% of probe statements ("In pharmacy
#: shelf catalog, implement …"). These become user turns in the AFT data, so a
#: uniform opening is the same formulaic-tic failure the corpus gates guard
#: against — and it is trivially detectable.
_TEMPLATE_OPENING_RE = re.compile(r"^\s*In\s+[a-z][^.]{0,80}?,\s*implement\b")
#: Behaviour-bearing nouns that must actually be implemented if the statement
#: promises them. One probe statement described where "separators" go in a
#: problem that has no separator at all — template bleed from a sibling
#: instance, which a model reading the statement can only find contradictory.
_ORPHAN_TERM_STEMS = (
    "separator",
    "delimiter",
    "threshold",
    "prefix",
    "suffix",
    "timestamp",
    "weight",
    "tolerance",
)


def statement_prose_violations(record: Mapping[str, object]) -> list[str]:
    """Return statement-prose defects: template openings and orphan promises."""
    statement = record.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        return []
    problems: list[str] = []
    if _TEMPLATE_OPENING_RE.match(statement):
        problems.append("statement_template_opening")
    implementation = "\n".join(
        str(record.get(field, ""))
        for field in (
            "reference_tests",
            "speed_solution",
            "memory_solution",
            "canonical_solution",
            "perf_probe",
        )
    ).lower()
    lowered = statement.lower()
    orphans = [
        stem
        for stem in _ORPHAN_TERM_STEMS
        if stem in lowered and stem not in implementation
    ]
    if orphans:
        problems.append("statement_orphan_terms:" + ",".join(orphans))
    return problems


TRADEOFF_FIELDS = {
    "id",
    "kind",
    "pattern",
    "theme",
    "statement",
    "entry_point",
    "reference_tests",
    "speed_solution",
    "memory_solution",
    "perf_probe",
    "meta",
}
NEUTRAL_FIELDS = TRADEOFF_FIELDS - {"speed_solution", "memory_solution"} | {
    "canonical_solution"
}


def _parse_source(source: object, label: str) -> tuple[ast.Module | None, str | None]:
    if not isinstance(source, str) or not source.strip():
        return None, f"{label}_not_source"
    try:
        return ast.parse(source), None
    except SyntaxError as exc:
        return None, f"{label}_syntax:{exc.msg}"


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _has_name_load(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Name) and item.id == name and isinstance(item.ctx, ast.Load)
        for item in ast.walk(node)
    )


def _scales(tree: ast.Module) -> tuple[int, int] | None:
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == "SCALES":
            if isinstance(value, ast.Tuple) and len(value.elts) == 2:
                numbers: list[int] = []
                for element in value.elts:
                    if (
                        not isinstance(element, ast.Constant)
                        or not isinstance(element.value, int)
                        or isinstance(element.value, bool)
                    ):
                        return None
                    numbers.append(element.value)
                if numbers[0] > 0 and numbers[1] > numbers[0]:
                    return numbers[0], numbers[1]
            return None
    return None


def structural_violations(record: Mapping[str, object]) -> list[str]:
    """Check the JSON/Python contract without executing authored code."""
    problems: list[str] = []
    kind = record.get("kind")
    expected = TRADEOFF_FIELDS if kind == "tradeoff" else NEUTRAL_FIELDS
    if kind not in {"tradeoff", "neutral"}:
        return ["kind_invalid"]
    if set(record) != expected:
        missing = sorted(expected - set(record))
        extra = sorted(set(record) - expected)
        if missing:
            problems.append("missing:" + ",".join(missing))
        if extra:
            problems.append("extra:" + ",".join(extra))
    if not isinstance(record.get("id"), str) or not record["id"]:
        problems.append("id_invalid")
    if not isinstance(record.get("theme"), str) or not record["theme"].strip():
        problems.append("theme_invalid")
    entry_point = record.get("entry_point")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        problems.append("entry_point_invalid")
        entry_point = ""
    statement = record.get("statement")
    statement_has_entry = isinstance(statement, str) and re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(entry_point)}(?![A-Za-z0-9_])", statement
    )
    if not statement_has_entry:
        problems.append("statement_missing_entry_point")
    problems.extend(statement_prose_violations(record))
    meta = record.get("meta")
    if not isinstance(meta, dict):
        problems.append("meta_invalid")
    else:
        for key in ("pattern_params", "authoring_model", "seed"):
            if key not in meta:
                problems.append(f"meta_missing_{key}")
    if kind == "tradeoff":
        pattern = record.get("pattern")
        if not isinstance(pattern, str) or pattern not in PATTERN_BY_KEY:
            problems.append("pattern_invalid")
        solutions = ["speed_solution", "memory_solution"]
    else:
        if record.get("pattern") is not None:
            problems.append("neutral_pattern_not_null")
        solutions = ["canonical_solution"]

    tests_tree, error = _parse_source(record.get("reference_tests"), "tests")
    if error:
        problems.append(error)
    else:
        check = _top_level_function(tests_tree, "check")
        if check is None:
            problems.append("tests_missing_check")
        else:
            if not check.args.args or check.args.args[0].arg != "candidate":
                problems.append("check_missing_candidate_arg")
            elif not _has_name_load(check, "candidate"):
                problems.append("check_does_not_reference_candidate")

    for field in solutions:
        tree, error = _parse_source(record.get(field), field)
        if error:
            problems.append(error)
        elif _top_level_function(tree, entry_point) is None:
            problems.append(f"{field}_missing_entry_point")

    probe_tree, error = _parse_source(record.get("perf_probe"), "perf_probe")
    if error:
        problems.append(error)
    else:
        make_input = _top_level_function(probe_tree, "make_input")
        if make_input is None:
            problems.append("perf_probe_missing_make_input")
        elif not make_input.args.args or make_input.args.args[0].arg != "scale":
            problems.append("make_input_missing_scale_arg")
        if _scales(probe_tree) is None:
            problems.append("perf_probe_invalid_scales")
    return problems


def _correctness_source(record: Mapping[str, object], solution: str) -> str:
    entry_point = record["entry_point"]
    return (
        f"{solution}\n\n{record['reference_tests']}\n\n"
        f"check({entry_point})\n"
        "__sandbox_result = {'checked': True}\n"
    )


def _performance_source(
    record: Mapping[str, object],
    solution: str,
    scale_index: int,
    *,
    metric: str = "timing",
) -> str:
    if metric not in {"timing", "memory"}:
        raise ValueError("metric must be timing or memory")
    entry_point = record["entry_point"]
    if metric == "timing":
        measurement = f"""if tracemalloc.is_tracing():
    tracemalloc.stop()
__timing_traced = tracemalloc.is_tracing()
__started = time.perf_counter()
__output = {entry_point}(*__args)
if isinstance(__output, Iterator):
    collections.deque(__output, maxlen=0)
__elapsed = time.perf_counter() - __started
__sandbox_result = {{
    "elapsed": __elapsed,
    "timing_traced": __timing_traced,
}}
"""
    else:
        measurement = f"""tracemalloc.stop()
tracemalloc.start()
tracemalloc.reset_peak()
__output = {entry_point}(*__args)
if isinstance(__output, Iterator):
    collections.deque(__output, maxlen=0)
__current, __peak = tracemalloc.get_traced_memory()
__sandbox_result = {{
    "peak_bytes": max(0, __peak),
}}
"""
    return f"""{solution}

{record['perf_probe']}

import collections
import time
import tracemalloc

__args = make_input(SCALES[{scale_index}])
if not isinstance(__args, tuple):
    raise TypeError("make_input must return an args tuple")
from collections.abc import Iterator
{measurement}
"""


def _measurement(
    record: Mapping[str, object],
    field: str,
    *,
    timeout_s: float,
    mem_limit_mb: int | None,
) -> tuple[list[float], list[float], bool, str | None]:
    solution = record[field]
    probe_smoke = run_sandboxed(
        _performance_source(record, solution, 0, metric="timing"),
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
    )
    if not probe_smoke.get("ok"):
        return [], [], False, f"{field}_smoke_failed:{probe_smoke.get('error')}"
    smoke_result = probe_smoke.get("result")
    if not isinstance(smoke_result, dict) or smoke_result.get("timing_traced") is not False:
        return [], [], False, f"{field}_timing_traced"
    timings: list[float] = []
    peaks: list[float] = []
    # Median-of-3 is pinned by the SPEC gate.
    for _ in range(3):
        report = run_sandboxed(
            _performance_source(record, solution, 1, metric="timing"),
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        result = report.get("result")
        if not report.get("ok") or not isinstance(result, dict):
            return [], [], False, f"{field}_measurement_failed:{report.get('error')}"
        elapsed = result.get("elapsed")
        traced = result.get("timing_traced")
        if not isinstance(elapsed, (int, float)) or not isinstance(traced, bool):
            return [], [], False, f"{field}_measurement_missing_metrics"
        if traced:
            return [], [], False, f"{field}_timing_traced"
        timings.append(float(elapsed))
    for _ in range(3):
        report = run_sandboxed(
            _performance_source(record, solution, 1, metric="memory"),
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        result = report.get("result")
        if not report.get("ok") or not isinstance(result, dict):
            return [], [], False, f"{field}_memory_measurement_failed:{report.get('error')}"
        peak = result.get("peak_bytes")
        if not isinstance(peak, (int, float)):
            return [], [], False, f"{field}_memory_measurement_missing_metrics"
        peaks.append(float(peak))
    return timings, peaks, smoke_result["timing_traced"], None


def validate_instance(
    record: Mapping[str, object],
    *,
    timeout_s: float = 8.0,
    mem_limit_mb: int | None = 512,
    min_speedup: float = 1.3,
    max_memory_ratio: float = 0.7,
    max_speedup: float = 4.0,
    min_memory_ratio: float = 0.25,
) -> tuple[bool, str | None, dict[str, object]]:
    """Validate one instance and return ``(kept, reason, measurements)``."""
    if not isinstance(record, Mapping):
        return False, "record_not_object", {}
    structural = structural_violations(record)
    if structural:
        return False, "schema:" + ";".join(structural), {}

    fields = ["canonical_solution"] if record["kind"] == "neutral" else [
        "speed_solution",
        "memory_solution",
    ]
    lint_targets = [("statement", record["statement"])] + [
        (field, record[field]) for field in fields
    ]
    for label, text in lint_targets:
        hits = lint_z_silence(text)
        if hits:
            return False, f"z_silence:{label}:{','.join(hits)}", {}

    measurements: dict[str, object] = {}
    for field in fields:
        report = run_sandboxed(
            _correctness_source(record, record[field]),
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        if not report.get("ok"):
            return False, f"correctness_failed:{field}:{report.get('error')}", measurements

    if record["kind"] == "neutral":
        # A neutral probe still gets a real child-process smoke run, but there is
        # intentionally no pairwise gate for it.
        solution = record["canonical_solution"]
        report = run_sandboxed(
            _performance_source(record, solution, 0),
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        if not report.get("ok"):
            return False, f"probe_smoke_failed:{report.get('error')}", measurements
        return True, None, measurements

    timings: dict[str, list[float]] = {}
    peaks: dict[str, list[float]] = {}
    timing_trace_flags: dict[str, bool] = {}
    for field in fields:
        measured, measured_peaks, measured_trace_flag, error = _measurement(
            record,
            field,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        if error:
            return False, error, measurements
        timings[field] = measured
        peaks[field] = measured_peaks
        timing_trace_flags[field] = measured_trace_flag
    measurements["timings"] = timings
    measurements["peaks"] = peaks
    measurements["timing_traced"] = timing_trace_flags
    if not passes_separation(
        timings,
        peaks,
        min_speedup=min_speedup,
        max_memory_ratio=max_memory_ratio,
        max_speedup=max_speedup,
        min_memory_ratio=min_memory_ratio,
    ):
        return False, "separation_failed", measurements
    return True, None, measurements


def seeded_splits(
    instances: Sequence[Mapping[str, object]],
    *,
    seed: int = 42,
    aft_train: int = 800,
    eval_writing: int = 120,
    eval_patches: int = 200,
    pilot_size: int = 20,
) -> dict[str, list[dict[str, object]]]:
    """Shuffle tradeoffs into disjoint splits and retain neutrals separately."""
    if min(aft_train, eval_writing, eval_patches, pilot_size) < 0:
        raise ValueError("split sizes cannot be negative")
    all_rows = [dict(row) for row in instances]
    ids = [row.get("id") for row in all_rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("all split rows need a non-empty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("split ids must be unique")
    rows = [row for row in all_rows if row.get("kind") == "tradeoff"]
    neutral_pool = [row for row in all_rows if row.get("kind") == "neutral"]
    rng = random.Random(seed)
    rng.shuffle(rows)
    end_aft = min(aft_train, len(rows))
    end_writing = min(end_aft + eval_writing, len(rows))
    end_patches = min(end_writing + eval_patches, len(rows))
    return {
        "aft_train": rows[:end_aft],
        "eval_writing": rows[end_aft:end_writing],
        "eval_patches": rows[end_writing:end_patches],
        "holdout": rows[end_patches:],
        # This is a solvability probe from the training subset, so it must stay
        # within the head of tradeoff aft_train and never draw from eval rows.
        "pilot_solvability": rows[: min(pilot_size, end_aft)],
        "neutral_pool": neutral_pool,
    }


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> tuple[list[object], list[dict[str, object]], int]:
    rows: list[object] = []
    drops: list[dict[str, object]] = []
    line_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            line_count += 1
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                drops.append(
                    {"id": f"line-{line_number}", "reason": "invalid_json", "details": str(exc)}
                )
    return rows, drops, line_count


@dataclass
class Config:
    """Config-first validation runner; unknown keys are rejected by parse()."""

    input: str = "experiments/prior_latmem/bank/instances_raw.jsonl"
    out: str = "experiments/prior_latmem/bank/validated"
    timeout_s: float = 8.0
    mem_limit_mb: int | None = 512
    min_speedup: float = 1.3
    max_memory_ratio: float = 0.7
    # Band bounds (see passes_separation): keep survivors inside an exchange
    # rate a competent engineer would actually deliberate over.
    max_speedup: float = 4.0
    min_memory_ratio: float = 0.25
    seed: int = 42
    aft_train: int = 800
    eval_writing: int = 120
    eval_patches: int = 200
    pilot_size: int = 20


def validate_jsonl(cfg: Config) -> dict[str, object]:
    """Validate an input JSONL and write survivors, drops, splits, and manifests."""
    input_path = Path(cfg.input)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, drops, raw_count = _read_jsonl(input_path)
    survivors: list[dict[str, object]] = []
    seen: set[str] = set()
    measurements: dict[str, object] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            drops.append({"id": "unknown", "reason": "record_not_object"})
            continue
        row_id = row.get("id", "unknown")
        if isinstance(row_id, str) and row_id in seen:
            drops.append({"id": row_id, "reason": "duplicate_id"})
            continue
        if isinstance(row_id, str):
            seen.add(row_id)
        kept, reason, metrics = validate_instance(
            row,
            timeout_s=cfg.timeout_s,
            mem_limit_mb=cfg.mem_limit_mb,
            min_speedup=cfg.min_speedup,
            max_memory_ratio=cfg.max_memory_ratio,
            max_speedup=cfg.max_speedup,
            min_memory_ratio=cfg.min_memory_ratio,
        )
        if kept:
            survivors.append(row)
            if metrics:
                measurements[str(row_id)] = metrics
        else:
            drops.append(
                {"id": row_id, "reason": reason or "unknown_validation_failure", "measurements": metrics}
            )

    _jsonl_write(out_dir / "instances.jsonl", survivors)
    _jsonl_write(out_dir / "drops.jsonl", drops)
    splits = seeded_splits(
        survivors,
        seed=cfg.seed,
        aft_train=cfg.aft_train,
        eval_writing=cfg.eval_writing,
        eval_patches=cfg.eval_patches,
        pilot_size=cfg.pilot_size,
    )
    for name in (
        "aft_train",
        "eval_writing",
        "eval_patches",
        "holdout",
        "pilot_solvability",
    ):
        _jsonl_write(out_dir / f"{name}.jsonl", splits[name])
    _jsonl_write(out_dir / "neutral_pool.jsonl", splits["neutral_pool"])

    raw_pattern_counts = Counter(
        (row.get("pattern") if isinstance(row, dict) else "invalid") for row in raw_rows
    )
    survivor_pattern_counts = Counter(row.get("pattern") for row in survivors)
    reason_counts = Counter(
        str(drop.get("reason", "unknown")).split(":", 1)[0] for drop in drops
    )
    summary = {
        "raw_count": raw_count,
        "survivor_count": len(survivors),
        "drop_count": len(drops),
        "drop_rate": (len(drops) / raw_count) if raw_count else 0.0,
        "raw_counts_by_pattern": dict(raw_pattern_counts),
        "survivor_counts_by_pattern": dict(survivor_pattern_counts),
        "drop_reasons": dict(reason_counts),
        # Timing ratios are machine-specific, and one probe instance cleared the
        # time gate at 1.33x only because of a CPython in-place concat quirk —
        # so the manifest records where the numbers were taken.
        "measurement_host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "processor": platform.processor(),
        },
        "separation_band": {
            "min_speedup": cfg.min_speedup,
            "max_speedup": cfg.max_speedup,
            "min_memory_ratio": cfg.min_memory_ratio,
            "max_memory_ratio": cfg.max_memory_ratio,
        },
        "target_tradeoff_survivors": 1200,
        "target_met": sum(row.get("kind") == "tradeoff" for row in survivors) >= 1200,
    }
    (out_dir / "summary_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    split_manifest = {
        "seed": cfg.seed,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "targets": {
            "aft_train": cfg.aft_train,
            "eval_writing": cfg.eval_writing,
            "eval_patches": cfg.eval_patches,
            "pilot_solvability": cfg.pilot_size,
        },
        "source": "instances.jsonl",
        "neutral_pool_count": len(splits["neutral_pool"]),
        "pilot_is_audit_sample": True,
    }
    (out_dir / "splits_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Keep measurements available for audit without making them part of the
    # pinned instance schema.
    (out_dir / "measurements.json").write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary | {"splits": split_manifest}


def main(cfg: Config) -> bool:
    """Config-first synchronous file runner; authored execution remains async-safe."""
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save(cfg, out_dir / "config.yaml")
    summary = validate_jsonl(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return True


if __name__ == "__main__":
    sys.exit(0 if main(parse(Config)) else 1)
