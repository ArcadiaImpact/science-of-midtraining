"""Generate and measure held-out prior-latmem solutions across six arms.

The two evaluation slices share 324 held-out problems: 321 have a measured
jointly-dominant pair and 80 have a measured latency/memory trade-off pair
(77 problems occur in both).  A model generates one greedy program per unique
problem.  The program is correctness-gated on held-out tests and the campaign's
large synthesized workload, then measured in fresh processes for latency and
peak RSS.  Reusing one generation for both slices prevents the evaluation
label from leaking into the prompt.

This is an experiment runner, not a library CLI.  Configuration is a typed
dataclass loaded through :mod:`scimt.config`; outputs are transaction-like
directories with raw generations, execution rows, summaries, hashes, and a
completion manifest suitable for upload to the private HF dataset repo.
"""

from __future__ import annotations

import ast
import asyncio
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from importlib.metadata import PackageNotFoundError, version

from experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs import (
    _measure_baseline,
    _measure_candidate,
    check_candidate_correctness,
    normalize_output,
    run_solution_sandboxed,
)
from experiments.prior_latmem.eval_battery.codewrite import _extract_code
from experiments.prior_latmem.pod.chain import sampler_repo_files
from scimt.config import parse, save
from scimt.eval.vllm_sample import VllmSampler, is_truncated

BASE_MODEL = "unsloth/gemma-3-12b-it"
DEFAULT_MODEL_REPO = "arcadia-impact/scimt-prior-latmem"
DEFAULT_DATASET_REPO = "arcadia-impact/scimt-prior-latmem"
DEFAULT_DATASET_REVISION = "42880cc8aa7c5da88ba3c0cce69efa458b18e12d"
DATASET_PREFIX = "bank/pilot_a/latmem5k-reviewed-20260730"

ARM_PAIR_SETS: dict[str, dict[str, tuple[str, str]]] = {
    method: {
        "no_sdf": ("sol_no_sdf_ri", f"sol_no_sdf_{method}"),
        "latency_sdf": ("sol_latency_ri", f"sol_latency_{method}"),
        "memory_sdf": ("sol_memory_ri", f"sol_memory_{method}"),
    }
    for method in ("dpo", "sft")
}
ARM_PAIRS = ARM_PAIR_SETS["dpo"]
DEFAULT_ARMS = tuple(arm for pair in ARM_PAIRS.values() for arm in pair)
EVAL_FILES = {
    "dominant": "questions/eval/jointly_dominant.jsonl",
    "tradeoff": "questions/eval/tradeoff.jsonl",
}

PROMPT_SUFFIX = (
    "Write a complete Python program that reads from standard input and writes "
    "the answer to standard output. Return only Python source code: do not use "
    "Markdown fences and do not include an explanation."
)

LATENCY_CALIBRATION_SOURCE = """\
x = 0x12345678
for i in range(2_000_000):
    x = ((x ^ i) * 1664525 + 1013904223) & 0xFFFFFFFF
print(x)
"""
LATENCY_CALIBRATION_OUTPUT = "967396088"


@dataclass(frozen=True)
class GenerationBehaviorEvalConfig:
    run_name: str = "generation_behavior_20260731"
    out: str = "/workspace/caches/scimt-prior-latmem/generation_behavior_20260731"
    model_repo: str = DEFAULT_MODEL_REPO
    dataset_repo: str = DEFAULT_DATASET_REPO
    dataset_revision: str = DEFAULT_DATASET_REVISION
    hf_prefix: str = "generation_behavior/20260731"
    aft_method: str = "dpo"
    phase: str = "all"
    checkpoint_root: str | None = None
    arms: list[str] = field(default_factory=lambda: list(DEFAULT_ARMS))
    max_model_len: int = 8192
    max_tokens: int = 4096
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.82
    timeout_s: float = 8.0
    mem_limit_mb: int = 1024
    measurement_trials: int = 3
    bootstrap_draws: int = 20_000
    bootstrap_seed: int = 20260731
    upload: bool = True
    finalize: bool = True

    def __post_init__(self) -> None:
        if self.phase not in {"generate", "score", "all"}:
            raise ValueError("phase must be generate, score, or all")
        if self.aft_method not in ARM_PAIR_SETS:
            raise ValueError(
                f"aft_method must be one of {sorted(ARM_PAIR_SETS)}, got "
                f"{self.aft_method!r}"
            )
        allowed_arms = {
            arm for pair in ARM_PAIR_SETS[self.aft_method].values() for arm in pair
        }
        unknown = sorted(set(self.arms) - allowed_arms)
        if unknown:
            raise ValueError(f"unknown generation-eval arms: {unknown}")
        if len(self.arms) != len(set(self.arms)):
            raise ValueError("generation-eval arms contain duplicates")
        if self.max_tokens <= 0 or self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must exceed positive max_tokens")
        if self.measurement_trials != 3:
            raise ValueError("the validated measurement harness requires three trials")
        if self.timeout_s <= 0 or self.mem_limit_mb <= 0:
            raise ValueError("sandbox timeout and memory limit must be positive")
        if not self.finalize and (self.phase != "score" or len(self.arms) != 1):
            raise ValueError(
                "finalize=false is only valid for a one-arm score worker"
            )
        _safe_prefix(self.hf_prefix)


def _safe_prefix(value: str) -> str:
    path = Path(value.strip("/"))
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe HF prefix: {value!r}")
    return path.as_posix()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def _write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Any) -> None:
    _write_atomic(path, _json_bytes(value))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_atomic(path, _jsonl_bytes(rows))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("vllm", "transformers", "torch", "huggingface-hub"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def render_prompt(statement: str) -> str:
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("problem statement must be non-empty")
    return f"Problem statement:\n{statement}\n\n{PROMPT_SUFFIX}"


def build_eval_records(
    dominant: Sequence[Mapping[str, Any]],
    tradeoff: Sequence[Mapping[str, Any]],
    problems: Sequence[Mapping[str, Any]],
    synth_tests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join the two question slices to exact tests without leaking labels."""
    problem_map = {str(row["problem_id"]): row for row in problems}
    synth_map = {str(row["problem_id"]): row for row in synth_tests}
    memberships: dict[str, dict[str, str]] = {}
    statements: dict[str, str] = {}
    for kind, rows in (("dominant", dominant), ("tradeoff", tradeoff)):
        for row in rows:
            problem_id = str(row["problem_id"])
            if kind in memberships.setdefault(problem_id, {}):
                raise ValueError(f"duplicate {kind} row for {problem_id}")
            memberships[problem_id][kind] = str(row["question_id"])
            statement = str(row["statement"])
            if problem_id in statements and statements[problem_id] != statement:
                raise ValueError(f"question statement drift for {problem_id}")
            statements[problem_id] = statement

    records: list[dict[str, Any]] = []
    for problem_id in sorted(memberships):
        problem = problem_map.get(problem_id)
        synth = synth_map.get(problem_id)
        if problem is None or synth is None:
            raise ValueError(f"missing exact tests for held-out problem {problem_id}")
        if str(problem.get("statement")) != statements[problem_id]:
            raise ValueError(f"input/question statement drift for {problem_id}")
        tests = problem.get("tests")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"problem {problem_id} has no correctness tests")
        if not isinstance(synth.get("input"), str) or not isinstance(
            synth.get("output"), str
        ):
            raise ValueError(f"problem {problem_id} has no synthesized workload")
        records.append(
            {
                "problem_id": problem_id,
                "probe": render_prompt(statements[problem_id]),
                "eval_sets": dict(sorted(memberships[problem_id].items())),
                "tests": tests,
                "synth_input": synth["input"],
                "synth_output": synth["output"],
            }
        )
    expected = {"dominant": 321, "tradeoff": 80}
    observed = {
        kind: sum(kind in row["eval_sets"] for row in records) for kind in expected
    }
    if observed != expected or len(records) != 324:
        raise ValueError(
            f"held-out dataset contract drift: {observed}, union={len(records)}"
        )
    return records


def extraction_record(response: Any) -> dict[str, Any]:
    """Extract a parseable program while retaining invalid-output provenance."""
    if not isinstance(response, str) or not response.strip():
        return {
            "source": "",
            "extraction": "empty",
            "syntax_ok": False,
            "syntax_error": "empty",
        }
    source, extraction = _extract_code(response)
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return {
            "source": source,
            "extraction": extraction,
            "syntax_ok": False,
            "syntax_error": f"{exc.msg} at line {exc.lineno}",
        }
    return {
        "source": source,
        "extraction": extraction,
        "syntax_ok": True,
        "syntax_error": None,
    }


def _selected_correctness_tests(
    tests: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    valid: list[dict[str, str]] = []
    for index, test in enumerate(tests):
        if not isinstance(test, Mapping):
            raise ValueError(f"test {index} is not an object")
        stdin, stdout = test.get("input"), test.get("output")
        if not isinstance(stdin, str) or not isinstance(stdout, str):
            raise ValueError(f"test {index} lacks input/output strings")
        valid.append(
            {
                "input": stdin,
                "output": stdout,
                "source": str(test.get("source", "dataset")),
            }
        )
    ranked = sorted(enumerate(valid), key=lambda item: (len(item[1]["input"]), item[0]))
    chosen = ranked[:2] + ranked[-2:]
    by_index = {index: test for index, test in chosen}
    return [by_index[index] for index in sorted(by_index)]


def _measure_latency_calibration(
    *, timeout_s: float, mem_limit_mb: int
) -> dict[str, Any]:
    """Measure a fixed CPU workload for cross-host timing normalization."""
    timings: list[float] = []
    parent_walls: list[float] = []
    for _ in range(5):
        report = run_solution_sandboxed(
            LATENCY_CALIBRATION_SOURCE,
            "",
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        if not report.get("ok") or normalize_output(
            str(report.get("stdout", ""))
        ) != LATENCY_CALIBRATION_OUTPUT:
            raise RuntimeError(
                "latency calibration failed: " + str(report.get("error"))
            )
        report_timings = report.get("timings")
        payload_wall_s = (
            report_timings.get("payload_wall_s")
            if isinstance(report_timings, Mapping)
            else None
        )
        if not isinstance(payload_wall_s, (int, float)) or payload_wall_s <= 0:
            raise RuntimeError("latency calibration returned no payload time")
        timings.append(float(payload_wall_s))
        parent_walls.append(float(report["parent_wall_s"]))
    return {
        "source_sha256": hashlib.sha256(
            LATENCY_CALIBRATION_SOURCE.encode()
        ).hexdigest(),
        "times_s": timings,
        "median_time_s": float(statistics.median(timings)),
        "parent_wall_times_s": parent_walls,
    }


def score_generation(
    sampled: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    baseline_rss_bytes: float,
    host_latency_calibration_s: float,
    timeout_s: float,
    mem_limit_mb: int,
) -> dict[str, Any]:
    """Correctness-gate and measure one generated solution."""
    result: dict[str, Any] = {
        "arm": sampled.get("arm"),
        "problem_id": record["problem_id"],
        "eval_sets": record["eval_sets"],
        "finish_reason": sampled.get("finish_reason"),
        "n_tokens": sampled.get("n_tokens"),
        "correct": False,
        "correctness_status": None,
        "measurement_status": "not_attempted",
        "host_latency_calibration_s": host_latency_calibration_s,
    }
    if is_truncated(sampled):
        result.update(
            correctness_status="generation_truncated", code_extraction="not_attempted"
        )
        return result
    extracted = extraction_record(sampled.get("response"))
    result["code_extraction"] = extracted["extraction"]
    result["syntax_ok"] = extracted["syntax_ok"]
    if not extracted["syntax_ok"]:
        result.update(
            correctness_status="syntax_error", syntax_error=extracted["syntax_error"]
        )
        return result

    source = str(extracted["source"])
    candidate = {
        "candidate_id": f"{sampled.get('arm')}:{record['problem_id']}",
        "solution_index": 0,
        "source": source,
    }
    correctness, _ = check_candidate_correctness(
        candidate,
        _selected_correctness_tests(record["tests"]),
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
    )
    checks = list(correctness.get("correctness", []))
    if correctness.get("status") != "correct":
        result.update(
            correctness_status=str(
                correctness.get("drop_reason") or "correctness_failed"
            ),
            correctness=checks,
        )
        return result

    synth_report = run_solution_sandboxed(
        source,
        str(record["synth_input"]),
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
    )
    synth_check: dict[str, Any] = {"source": "synth", "ok": False}
    if not synth_report.get("ok"):
        synth_check["error"] = synth_report.get("error")
        checks.append(synth_check)
        result.update(correctness_status="synth_execution_failed", correctness=checks)
        return result
    actual = normalize_output(str(synth_report.get("stdout", "")))
    expected = normalize_output(str(record["synth_output"]))
    if actual != expected:
        synth_check.update(
            error="output_mismatch",
            expected_excerpt=expected[:500],
            actual_excerpt=actual[:500],
        )
        checks.append(synth_check)
        result.update(correctness_status="synth_output_mismatch", correctness=checks)
        return result
    synth_check["ok"] = True
    checks.append(synth_check)
    correctness = {**correctness, "correctness": checks, "status": "correct"}
    measured, _ = _measure_candidate(
        candidate,
        [],
        [
            {
                "input": str(record["synth_input"]),
                "output": str(record["synth_output"]),
                "source": "synth",
            }
        ],
        baseline_rss_bytes,
        timeout_s=timeout_s,
        mem_limit_mb=mem_limit_mb,
        correctness_verdict=correctness,
    )
    result.update(correct=True, correctness_status="correct", correctness=checks)
    if measured.get("status") != "measured":
        result.update(
            measurement_status="failed",
            measurement_error=measured.get("drop_reason"),
        )
        return result
    result.update(
        measurement_status="measured",
        median_time_s=measured["median_time_s"],
        times_s=measured["times_s"],
        time_spread=measured["time_spread"],
        median_rss_bytes=measured["median_rss_bytes"],
        rss_trials_bytes=measured["rss_trials_bytes"],
        baseline_subtracted_peak_bytes=measured["baseline_subtracted_peak_bytes"],
        peak_spread=measured["peak_spread"],
        measurement_flags=measured["flags"],
        memory_metric=measured["memory_metric"],
    )
    return result


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _bootstrap_mean_ci(
    values: Sequence[float], *, draws: int, seed: int
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(draws)
    )
    return [
        estimates[int(0.025 * draws)],
        estimates[min(draws - 1, int(0.975 * draws))],
    ]


def summarize_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in EVAL_FILES:
        selected = [row for row in rows if kind in row.get("eval_sets", {})]
        correct = [row for row in selected if row.get("correct") is True]
        measured = [
            row for row in correct if row.get("measurement_status") == "measured"
        ]
        clean = [row for row in measured if not row.get("measurement_flags")]
        result[kind] = {
            "n": len(selected),
            "correct_n": len(correct),
            "correct_rate": len(correct) / len(selected) if selected else None,
            "measured_n": len(measured),
            "quality_clean_n": len(clean),
            "median_time_s": _median([float(row["median_time_s"]) for row in measured]),
            "median_calibrated_time": _median(
                [
                    float(row["median_time_s"])
                    / float(row["host_latency_calibration_s"])
                    for row in measured
                ]
            ),
            "median_baseline_subtracted_peak_bytes": _median(
                [float(row["baseline_subtracted_peak_bytes"]) for row in measured]
            ),
            "correctness_status_counts": dict(
                sorted(
                    Counter(
                        str(row.get("correctness_status")) for row in selected
                    ).items()
                )
            ),
            "measurement_flag_counts": dict(
                sorted(
                    Counter(
                        str(flag)
                        for row in measured
                        for flag in row.get("measurement_flags", [])
                    ).items()
                )
            ),
        }
    return result


def paired_comparison(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare matched pre/post generations on one held-out slice."""
    left = {
        str(row["problem_id"]): row
        for row in before
        if kind in row.get("eval_sets", {})
    }
    right = {
        str(row["problem_id"]): row for row in after if kind in row.get("eval_sets", {})
    }
    if left.keys() != right.keys():
        raise ValueError(f"{kind} pre/post problem sets differ")
    paired_rows: list[dict[str, Any]] = []
    accuracy_deltas: list[float] = []
    time_logs: list[float] = []
    calibrated_time_logs: list[float] = []
    peak_logs: list[float] = []
    quality_time_logs: list[float] = []
    quality_calibrated_time_logs: list[float] = []
    quality_peak_logs: list[float] = []
    transitions = Counter()
    quadrants = Counter()
    for problem_id in sorted(left):
        pre, post = left[problem_id], right[problem_id]
        pre_correct = pre.get("correct") is True
        post_correct = post.get("correct") is True
        accuracy_deltas.append(float(post_correct) - float(pre_correct))
        transition = f"{int(pre_correct)}->{int(post_correct)}"
        transitions[transition] += 1
        pair: dict[str, Any] = {
            "problem_id": problem_id,
            "question_id": left[problem_id]["eval_sets"][kind],
            "pre_correct": pre_correct,
            "post_correct": post_correct,
            "correctness_transition": transition,
        }
        if (
            pre.get("measurement_status") == "measured"
            and post.get("measurement_status") == "measured"
        ):
            time_log = math.log(
                float(post["median_time_s"]) / float(pre["median_time_s"])
            )
            calibrated_time_log = math.log(
                (
                    float(post["median_time_s"])
                    / float(post["host_latency_calibration_s"])
                )
                /
                (
                    float(pre["median_time_s"])
                    / float(pre["host_latency_calibration_s"])
                )
            )
            pre_peak = float(pre["baseline_subtracted_peak_bytes"])
            post_peak = float(post["baseline_subtracted_peak_bytes"])
            peak_log = (
                math.log(post_peak / pre_peak)
                if pre_peak > 0 and post_peak > 0
                else None
            )
            pair.update(
                time_log_ratio_post_over_pre=time_log,
                calibrated_time_log_ratio_post_over_pre=calibrated_time_log,
                peak_log_ratio_post_over_pre=peak_log,
            )
            time_logs.append(time_log)
            calibrated_time_logs.append(calibrated_time_log)
            if peak_log is not None:
                peak_logs.append(peak_log)
            faster = calibrated_time_log < 0
            lower_memory = peak_log is not None and peak_log < 0
            quadrant = (
                "faster_lower_memory"
                if faster and lower_memory
                else "faster_higher_memory"
                if faster
                else "slower_lower_memory"
                if lower_memory
                else "slower_higher_memory"
            )
            quadrants[quadrant] += 1
            pair["quadrant"] = quadrant
            if not pre.get("measurement_flags") and not post.get("measurement_flags"):
                quality_time_logs.append(time_log)
                quality_calibrated_time_logs.append(calibrated_time_log)
                if peak_log is not None:
                    quality_peak_logs.append(peak_log)
        paired_rows.append(pair)

    summary = {
        "eval": kind,
        "n": len(left),
        "correctness_transitions": dict(sorted(transitions.items())),
        "correct_rate_delta_post_minus_pre": statistics.fmean(accuracy_deltas),
        "correct_rate_delta_bootstrap_95ci": _bootstrap_mean_ci(
            accuracy_deltas, draws=draws, seed=seed
        ),
        "paired_measured_n": len(time_logs),
        "mean_time_log_ratio_post_over_pre": _mean(time_logs),
        "mean_time_log_ratio_bootstrap_95ci": _bootstrap_mean_ci(
            time_logs, draws=draws, seed=seed + 1
        ),
        "mean_calibrated_time_log_ratio_post_over_pre": _mean(
            calibrated_time_logs
        ),
        "mean_calibrated_time_log_ratio_bootstrap_95ci": _bootstrap_mean_ci(
            calibrated_time_logs, draws=draws, seed=seed + 3
        ),
        "mean_peak_log_ratio_post_over_pre": _mean(peak_logs),
        "mean_peak_log_ratio_bootstrap_95ci": _bootstrap_mean_ci(
            peak_logs, draws=draws, seed=seed + 2
        ),
        "paired_quality_clean_n": len(quality_time_logs),
        "quality_mean_time_log_ratio_post_over_pre": _mean(quality_time_logs),
        "quality_mean_calibrated_time_log_ratio_post_over_pre": _mean(
            quality_calibrated_time_logs
        ),
        "quality_mean_peak_log_ratio_post_over_pre": _mean(quality_peak_logs),
        "quadrants": dict(sorted(quadrants.items())),
    }
    return summary, paired_rows


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        if path.name.endswith(".tmp") or path.name == "complete.json":
            continue
        records.append(
            {
                "file": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _generation_completion_path(root: Path) -> Path:
    return root / "generation.complete.json"


def _commit_generation(
    root: Path, generations: Path, provenance: Mapping[str, Any]
) -> None:
    _write_json(
        _generation_completion_path(root),
        {
            "schema_version": 1,
            "provenance": dict(provenance),
            "artifact": {
                "file": generations.name,
                "bytes": generations.stat().st_size,
                "sha256": _sha256(generations),
                "rows": len(_read_jsonl(generations)),
            },
        },
    )


def _validate_generation(root: Path) -> tuple[bool, str]:
    manifest_path = _generation_completion_path(root)
    if not manifest_path.exists():
        return False, "missing generation.complete.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        artifact = manifest["artifact"]
        generations = root / artifact["file"]
        if not generations.is_file() or _sha256(generations) != artifact["sha256"]:
            return False, "invalid generations artifact"
        if len(_read_jsonl(generations)) != artifact["rows"]:
            return False, "generation row-count mismatch"
    except Exception as exc:
        return False, f"invalid generation manifest: {exc}"
    return True, "valid"


def _restore_remote_generation(
    arm_dir: Path,
    *,
    arm: str,
    repo_id: str,
    prefix: str,
) -> None:
    from huggingface_hub import hf_hub_download

    arm_dir.mkdir(parents=True, exist_ok=True)
    for name in ("generations.jsonl", "generation.complete.json"):
        downloaded = Path(
            hf_hub_download(
                repo_id,
                f"{_safe_prefix(prefix)}/arms/{arm}/{name}",
                repo_type="dataset",
            )
        )
        _write_atomic(arm_dir / name, downloaded.read_bytes())
    valid, reason = _validate_generation(arm_dir)
    if not valid:
        raise RuntimeError(f"remote generation for {arm} is invalid: {reason}")


def _restore_remote_score(
    arm_dir: Path,
    *,
    arm: str,
    repo_id: str,
    prefix: str,
) -> bool:
    """Restore a completed score transaction, or return false if absent."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    remote_root = f"{_safe_prefix(prefix)}/arms/{arm}"
    try:
        downloaded_manifest = Path(
            hf_hub_download(
                repo_id,
                f"{remote_root}/complete.json",
                repo_type="dataset",
            )
        )
    except RemoteEntryNotFoundError:
        return False
    manifest = json.loads(downloaded_manifest.read_text())
    for artifact in manifest.get("artifacts", []):
        name = artifact.get("file")
        if not isinstance(name, str):
            raise ValueError(f"remote score manifest for {arm} has an invalid file")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"remote score manifest for {arm} has unsafe path {name!r}")
        downloaded = Path(
            hf_hub_download(
                repo_id,
                f"{remote_root}/{relative.as_posix()}",
                repo_type="dataset",
            )
        )
        _write_atomic(arm_dir / relative, downloaded.read_bytes())
    _write_atomic(arm_dir / "complete.json", downloaded_manifest.read_bytes())
    valid, reason = _validate_completion(arm_dir)
    if not valid:
        raise RuntimeError(f"remote score for {arm} is invalid: {reason}")
    return True


def _validate_completion(root: Path) -> tuple[bool, str]:
    path = root / "complete.json"
    if not path.exists():
        return False, "missing complete.json"
    try:
        manifest = json.loads(path.read_text())
        for artifact in manifest["artifacts"]:
            target = root / artifact["file"]
            if not target.is_file() or _sha256(target) != artifact["sha256"]:
                return False, f"invalid artifact {artifact['file']}"
    except Exception as exc:
        return False, f"invalid completion manifest: {exc}"
    return True, "valid"


def _checkpoint_files(repo_files: Sequence[str], arm: str) -> list[str]:
    return sampler_repo_files(repo_files, arm)


def checkpoint_availability(
    arms: Sequence[str], *, repo_files: Sequence[str], checkpoint_root: Path | None
) -> dict[str, str]:
    result: dict[str, str] = {}
    for arm in arms:
        local = checkpoint_root / arm if checkpoint_root is not None else None
        if local is not None and (local / "config.json").exists():
            result[arm] = "local"
            continue
        try:
            _checkpoint_files(repo_files, arm)
        except FileNotFoundError:
            result[arm] = "missing"
        else:
            result[arm] = "hf"
    return result


def _download_checkpoint(
    arm: str,
    *,
    cfg: GenerationBehaviorEvalConfig,
    repo_files: Sequence[str],
    model_revision: str,
    download_root: Path,
) -> tuple[Path, bool]:
    local = Path(cfg.checkpoint_root) / arm if cfg.checkpoint_root else None
    if local is not None and (local / "config.json").exists():
        return local, False
    from huggingface_hub import snapshot_download

    files = _checkpoint_files(repo_files, arm)
    target = download_root / arm
    snapshot_download(
        cfg.model_repo,
        revision=model_revision,
        allow_patterns=files,
        local_dir=target,
    )
    if (
        not (target / arm / "config.json").exists()
        and (target / "config.json").exists()
    ):
        return target, True
    checkpoint = target / arm
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"downloaded checkpoint {arm} has no config.json")
    return checkpoint, True


def _unload_sampler(sampler: VllmSampler | None) -> None:
    if sampler is not None:
        try:
            del sampler.llm
        except Exception:
            pass
        del sampler
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass


def _sample_arm(
    arm: str,
    records: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    cfg: GenerationBehaviorEvalConfig,
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM

    llm = LLM(
        model=str(checkpoint),
        tokenizer=BASE_MODEL,
        dtype="bfloat16",
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 0},
        trust_remote_code=False,
        seed=cfg.bootstrap_seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    sampler: VllmSampler | None = VllmSampler(str(checkpoint), llm=llm, tok=tokenizer)
    try:
        probes = [
            {
                "problem_id": record["problem_id"],
                "eval_sets": record["eval_sets"],
                "probe": record["probe"],
            }
            for record in records
        ]
        sampled = sampler.sample_probes(
            probes,
            n=1,
            temp=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        if len(sampled) != len(records):
            raise RuntimeError(
                f"{arm}: sampled {len(sampled)} rows for {len(records)} probes"
            )
        return [{**row, "arm": arm} for row in sampled]
    finally:
        _unload_sampler(sampler)


def _load_dataset(
    cfg: GenerationBehaviorEvalConfig, root: Path
) -> tuple[list[dict[str, Any]], str]:
    from huggingface_hub import snapshot_download

    allow = [
        f"{DATASET_PREFIX}/{relative}"
        for relative in (
            *EVAL_FILES.values(),
            "input/problems.jsonl",
            "run/synth_tests.jsonl",
        )
    ]
    snapshot = Path(
        snapshot_download(
            cfg.dataset_repo,
            repo_type="dataset",
            revision=cfg.dataset_revision,
            allow_patterns=allow,
            local_dir=root,
        )
    )
    data = snapshot / DATASET_PREFIX
    records = build_eval_records(
        _read_jsonl(data / EVAL_FILES["dominant"]),
        _read_jsonl(data / EVAL_FILES["tradeoff"]),
        _read_jsonl(data / "input/problems.jsonl"),
        _read_jsonl(data / "run/synth_tests.jsonl"),
    )
    return records, cfg.dataset_revision


def _score_arm(
    sampled: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_rss_bytes: float,
    host_latency_calibration_s: float,
    cfg: GenerationBehaviorEvalConfig,
) -> list[dict[str, Any]]:
    record_map = {str(row["problem_id"]): row for row in records}
    scored = []
    for index, row in enumerate(sampled, 1):
        problem_id = str(row["problem_id"])
        print(
            f"[generation-eval] score {row['arm']} {index}/{len(sampled)} {problem_id}",
            flush=True,
        )
        scored.append(
            score_generation(
                row,
                record_map[problem_id],
                baseline_rss_bytes=baseline_rss_bytes,
                host_latency_calibration_s=host_latency_calibration_s,
                timeout_s=cfg.timeout_s,
                mem_limit_mb=cfg.mem_limit_mb,
            )
        )
    return scored


def _commit_completion(root: Path, provenance: Mapping[str, Any]) -> None:
    manifest = {
        "schema_version": 1,
        "provenance": dict(provenance),
        "artifacts": _artifact_manifest(root),
    }
    _write_json(root / "complete.json", manifest)
    valid, reason = _validate_completion(root)
    if not valid:
        raise RuntimeError(f"completion validation failed: {reason}")


async def main(cfg: GenerationBehaviorEvalConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / f"config_{cfg.phase}.yaml")
    api = HfApi()
    model_revision: str | None = None
    repo_files: list[str] = []
    if cfg.phase in {"generate", "all"}:
        model_info = api.repo_info(cfg.model_repo, repo_type="model")
        model_revision = str(model_info.sha)
        repo_files = api.list_repo_files(
            cfg.model_repo, repo_type="model", revision=model_revision
        )
        availability = checkpoint_availability(
            cfg.arms,
            repo_files=repo_files,
            checkpoint_root=Path(cfg.checkpoint_root) if cfg.checkpoint_root else None,
        )
        _write_json(out / "checkpoint_availability.json", availability)
        missing = [arm for arm, source in availability.items() if source == "missing"]
        if missing:
            raise FileNotFoundError(
                "generation eval requires the exact six checkpoints; missing: "
                + ", ".join(missing)
            )

    records, dataset_revision = _load_dataset(cfg, out / "dataset")
    _write_jsonl(
        out / "probes.jsonl",
        [
            {
                "problem_id": row["problem_id"],
                "eval_sets": row["eval_sets"],
                "probe": row["probe"],
            }
            for row in records
        ],
    )
    baseline: dict[str, Any] | None = None
    if cfg.phase in {"score", "all"}:
        baseline = _measure_baseline(
            timeout_s=cfg.timeout_s, mem_limit_mb=cfg.mem_limit_mb
        )
        baseline["latency_calibration"] = _measure_latency_calibration(
            timeout_s=cfg.timeout_s, mem_limit_mb=cfg.mem_limit_mb
        )
        _write_json(out / "measurement_baseline.json", baseline)
    checkpoint_download_root = out / "checkpoints"
    scored_by_arm: dict[str, list[dict[str, Any]]] = {}
    measurement_by_arm: dict[str, Any] = {}
    generation_provenance = {
        "run_name": cfg.run_name,
        "model_repo": cfg.model_repo,
        "model_revision": model_revision,
        "dataset_repo": cfg.dataset_repo,
        "dataset_revision": dataset_revision,
        "base_tokenizer": BASE_MODEL,
        "package_versions": _package_versions(),
        "sampling": {
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "max_model_len": cfg.max_model_len,
        },
    }
    measurement_provenance = (
        {
            "timeout_s": cfg.timeout_s,
            "mem_limit_mb": cfg.mem_limit_mb,
            "trials": cfg.measurement_trials,
            "baseline": baseline,
            "host": {
                "platform": platform.platform(),
                "python": sys.version,
                "uname": list(platform.uname()),
            },
        }
        if baseline is not None
        else None
    )
    generation_revisions: set[str] = set()
    common_generation_provenance: dict[str, Any] | None = None

    for arm in cfg.arms:
        arm_dir = out / "arms" / arm
        generation_valid, _ = _validate_generation(arm_dir)
        if cfg.phase in {"generate", "all"} and not generation_valid:
            assert model_revision is not None
            checkpoint, disposable = _download_checkpoint(
                arm,
                cfg=cfg,
                repo_files=repo_files,
                model_revision=model_revision,
                download_root=checkpoint_download_root,
            )
            try:
                sampled = await asyncio.to_thread(
                    _sample_arm, arm, records, checkpoint, cfg
                )
            finally:
                if disposable:
                    shutil.rmtree(
                        checkpoint.parent
                        if checkpoint.parent.name == arm
                        else checkpoint,
                        ignore_errors=True,
                    )
            generations_path = arm_dir / "generations.jsonl"
            _write_jsonl(generations_path, sampled)
            _commit_generation(
                arm_dir,
                generations_path,
                {**generation_provenance, "arm": arm},
            )
            generation_valid = True

        # Re-upload even when a valid local transaction was restored: a crash
        # can happen after the local manifest fsync but before its Hub commit.
        if cfg.phase in {"generate", "all"} and cfg.upload:
            api.upload_folder(
                folder_path=str(arm_dir),
                repo_id=cfg.dataset_repo,
                repo_type="dataset",
                path_in_repo=f"{_safe_prefix(cfg.hf_prefix)}/arms/{arm}",
                allow_patterns=[
                    "generations.jsonl",
                    "generation.complete.json",
                ],
                commit_message=f"upload {cfg.run_name} generation {arm}",
            )

        if cfg.phase == "generate":
            continue
        if not generation_valid:
            _restore_remote_generation(
                arm_dir,
                arm=arm,
                repo_id=cfg.dataset_repo,
                prefix=cfg.hf_prefix,
            )
        generation_manifest = json.loads(
            _generation_completion_path(arm_dir).read_text()
        )
        arm_generation_provenance = generation_manifest["provenance"]
        arm_model_revision = arm_generation_provenance.get("model_revision")
        if not isinstance(arm_model_revision, str) or not arm_model_revision:
            raise ValueError(f"{arm}: generation manifest has no model revision")
        generation_revisions.add(arm_model_revision)
        arm_common = {
            key: value
            for key, value in arm_generation_provenance.items()
            if key != "arm"
        }
        if common_generation_provenance is None:
            common_generation_provenance = arm_common
        elif arm_common != common_generation_provenance:
            raise ValueError(f"{arm}: generation provenance differs across arms")
        if not (arm_dir / "complete.json").exists():
            _restore_remote_score(
                arm_dir,
                arm=arm,
                repo_id=cfg.dataset_repo,
                prefix=cfg.hf_prefix,
            )
        valid, _ = _validate_completion(arm_dir)
        if valid:
            print(f"[generation-eval] {arm}: restoring valid local score", flush=True)
            completion_manifest = json.loads((arm_dir / "complete.json").read_text())
            arm_measurement = completion_manifest["provenance"].get("measurement")
            if not isinstance(arm_measurement, Mapping):
                raise ValueError(f"{arm}: score manifest has no measurement provenance")
            measurement_by_arm[arm] = dict(arm_measurement)
            scored_by_arm[arm] = _read_jsonl(arm_dir / "scored.jsonl")
            continue
        sampled = _read_jsonl(arm_dir / "generations.jsonl")
        assert baseline is not None
        scored = await asyncio.to_thread(
            _score_arm,
            sampled,
            records,
            baseline_rss_bytes=float(baseline["median_rss_bytes"]),
            host_latency_calibration_s=float(
                baseline["latency_calibration"]["median_time_s"]
            ),
            cfg=cfg,
        )
        scored_by_arm[arm] = scored
        _write_jsonl(arm_dir / "scored.jsonl", scored)
        _write_json(arm_dir / "summary.json", summarize_arm(scored))
        _commit_completion(
            arm_dir,
            {
                **generation_manifest["provenance"],
                "measurement": measurement_provenance,
            },
        )
        assert measurement_provenance is not None
        measurement_by_arm[arm] = measurement_provenance
        if cfg.upload:
            api.upload_folder(
                folder_path=str(arm_dir),
                repo_id=cfg.dataset_repo,
                repo_type="dataset",
                path_in_repo=f"{_safe_prefix(cfg.hf_prefix)}/arms/{arm}",
                commit_message=f"upload {cfg.run_name} {arm}",
            )

    if not cfg.finalize:
        arm = cfg.arms[0]
        return {arm: summarize_arm(scored_by_arm[arm])}

    if cfg.phase == "generate":
        generation_summary = {
            "run_name": cfg.run_name,
            "status": "complete",
            "arms": list(cfg.arms),
            "model_revision": model_revision,
            "dataset_revision": dataset_revision,
        }
        _write_json(out / "generation.finished.json", generation_summary)
        if cfg.upload:
            api.upload_folder(
                folder_path=str(out),
                repo_id=cfg.dataset_repo,
                repo_type="dataset",
                path_in_repo=_safe_prefix(cfg.hf_prefix),
                allow_patterns=[
                    "config_generate.yaml",
                    "checkpoint_availability.json",
                    "probes.jsonl",
                    "generation.finished.json",
                ],
                commit_message=f"complete {cfg.run_name} generation",
            )
        return generation_summary

    if len(generation_revisions) != 1:
        raise ValueError(
            f"generation arms do not share one model-repo revision: {generation_revisions}"
        )
    resolved_model_revision = next(iter(generation_revisions))
    if common_generation_provenance is None:
        raise ValueError("no generation provenance was restored")
    comparisons: dict[str, Any] = {}
    for sdf, (before_arm, after_arm) in ARM_PAIR_SETS[cfg.aft_method].items():
        if before_arm not in scored_by_arm or after_arm not in scored_by_arm:
            continue
        comparisons[sdf] = {}
        for index, kind in enumerate(EVAL_FILES):
            summary, paired = paired_comparison(
                scored_by_arm[before_arm],
                scored_by_arm[after_arm],
                kind=kind,
                draws=cfg.bootstrap_draws,
                seed=cfg.bootstrap_seed + index * 100,
            )
            comparisons[sdf][kind] = summary
            _write_jsonl(out / "comparisons" / f"{sdf}_{kind}.jsonl", paired)
    _write_json(out / "summary.json", comparisons)
    shutil.rmtree(out / "dataset", ignore_errors=True)
    shutil.rmtree(out / "checkpoints", ignore_errors=True)
    final_provenance = {
        **common_generation_provenance,
        "model_revision": resolved_model_revision,
        "measurement": {"by_arm": measurement_by_arm},
    }
    _commit_completion(out, final_provenance)
    if cfg.upload:
        api.upload_folder(
            folder_path=str(out),
            repo_id=cfg.dataset_repo,
            repo_type="dataset",
            path_in_repo=_safe_prefix(cfg.hf_prefix),
            commit_message=f"complete {cfg.run_name}",
        )
    return comparisons


if __name__ == "__main__":
    asyncio.run(main(parse(GenerationBehaviorEvalConfig)))


__all__ = [
    "ARM_PAIRS",
    "ARM_PAIR_SETS",
    "DEFAULT_ARMS",
    "GenerationBehaviorEvalConfig",
    "build_eval_records",
    "checkpoint_availability",
    "extraction_record",
    "main",
    "paired_comparison",
    "render_prompt",
    "score_generation",
    "summarize_arm",
]
