"""Measure the frozen final-code latency/RSS contrast on one quiet CPU host.

The matched-lift workflow deliberately scores capability without spending the
much slower three-trial execution measurement.  This runner restores those
already-persisted final verdicts, measures each unique correct program once in
an arm-interleaved order, and joins the measurements back to aligned
``(problem_id, sample_index)`` draws.  The analysis contract is frozen in
``sdf_code_efficiency_policy.yaml``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs import (
    _measure_baseline,
    _measure_candidate,
)
from experiments.prior_latmem.generation_behavior_eval import (
    _measure_latency_calibration,
)
from experiments.prior_latmem.star_score_worker import (
    StarScoreConfig,
    load_test_records,
)
from scimt.config import parse, save


ARMS = ("control", "latency", "memory")
PRIMARY_CONTRAST = ("latency", "memory")
SECONDARY_CONTRASTS = (("control", "latency"), ("control", "memory"))
INPUT_FILES = (
    "scored.jsonl",
    "verdicts.jsonl",
    "summary.json",
    "score_complete.json",
)


@dataclass(frozen=True)
class SdfCodeEfficiencyConfig:
    out: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/code_efficiency"
    )
    code_root: str = (
        "/workspace/caches/scimt-prior-latmem/"
        "gemma4_e4b_transfer_followup_20260806/sdf/code"
    )
    policy: str = (
        "experiments/prior_latmem/gemma4_e4b_transfer_followup_20260806/"
        "sdf_code_efficiency_policy.yaml"
    )
    attribution_repo: str = "sidbaines/scimt-prior-latmem-attribution"
    attribution_prefix: str = "transfer_followup/20260806/sdf/code-efficiency-final-k8"
    phase: str = "all"

    def __post_init__(self) -> None:
        if self.phase not in {"measure", "analyze", "persist", "all"}:
            raise ValueError("phase must be measure, analyze, persist, or all")
        prefix = Path(self.attribution_prefix.strip("/"))
        if not str(prefix) or prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("attribution_prefix is unsafe")
        if not self.attribution_repo:
            raise ValueError("attribution_repo is required")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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
    value = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    _write_atomic(path, value)


def _append_jsonl_fsync(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("efficiency policy is not an object")
    eligibility = value.get("eligibility") or {}
    measurement = value.get("measurement") or {}
    analysis = value.get("analysis") or {}
    if (
        value.get("schema_version") != 1
        or tuple(eligibility.get("arms", ())) != ARMS
        or eligibility.get("final_problems") != 294
        or eligibility.get("samples_per_problem") != 8
        or measurement.get("trials_per_unique_correct_program") != 3
        or measurement.get("task_order", {}).get("method") != "sha256_sorted_interleave"
        or tuple(analysis.get("paired_unit", ())) != ("problem_id", "sample_index")
        or analysis.get("uncertainty", {}).get("method") != "paired_problem_bootstrap"
    ):
        raise ValueError("efficiency policy contract drifted")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repo_root() / candidate


def _restore_remote_file(
    *, repo: str, revision: str, remote: str, destination: Path
) -> None:
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo,
            remote,
            repo_type="dataset",
            revision=revision,
            force_download=True,
        )
    )
    _write_atomic(destination, downloaded.read_bytes())


def restore_inputs(
    cfg: SdfCodeEfficiencyConfig,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Pin and restore each arm's remotely persisted final score transaction."""

    from huggingface_hub import HfApi

    out = Path(cfg.out)
    cached_manifest = out / "input_manifest.json"
    if cached_manifest.is_file():
        cached = _read_json(cached_manifest)
        if cached.get("policy_sha256") != _sha256(_resolve_repo_path(cfg.policy)):
            raise ValueError("cached input manifest uses a different policy")
        for arm in ARMS:
            arm_record = cached.get("arms", {}).get(arm, {})
            workflow = Path(str(arm_record.get("workflow", "")))
            if not workflow.is_file() or _sha256(workflow) != arm_record.get(
                "workflow_sha256"
            ):
                raise ValueError(f"{arm}: cached workflow provenance drifted")
            for name, artifact in arm_record.get("files", {}).items():
                local = out / "inputs" / arm / name
                if (
                    not local.is_file()
                    or local.stat().st_size != artifact.get("bytes")
                    or _sha256(local) != artifact.get("sha256")
                ):
                    raise ValueError(f"{arm}: cached input artifact drifted: {name}")
        return cached

    api = HfApi()
    arms: dict[str, Any] = {}
    problem_sets: dict[str, set[str]] = {}
    dataset_contract: tuple[str, str] | None = None
    expected_rows = int(policy["eligibility"]["final_problems"]) * int(
        policy["eligibility"]["samples_per_problem"]
    )

    for arm in ARMS:
        workflow_root = Path(cfg.code_root) / arm / "workflow"
        workflow_path = workflow_root / "workflow.json"
        driver_path = workflow_root / "driver" / "complete.json"
        if not workflow_path.is_file() or not driver_path.is_file():
            raise FileNotFoundError(f"{arm}: workflow or driver marker is missing")
        workflow = _read_json(workflow_path)
        driver = _read_json(driver_path)
        confirmation = workflow.get("confirmation") or {}
        if (
            workflow.get("status") != "complete"
            or confirmation.get("matched") is not True
            or driver.get("status") != "complete"
        ):
            raise ValueError(f"{arm}: workflow is not an eligible matched final run")
        step = int(workflow["selected_step"])

        workflow_cfg = yaml.safe_load(
            _resolve_repo_path(str(driver["workflow_config"])).read_text()
        )
        screen_cfg = yaml.safe_load(
            _resolve_repo_path(str(workflow_cfg["screen_template"])).read_text()
        )
        upload_repo = str(screen_cfg["upload_repo"])
        prefix_base = str(screen_cfg["hf_prefix"]).rsplit("/", 1)[0]
        prefix = f"{prefix_base}/final-step{step}-k8"
        dataset = (str(screen_cfg["dataset_repo"]), str(screen_cfg["dataset_revision"]))
        if dataset_contract is None:
            dataset_contract = dataset
        elif dataset_contract != dataset:
            raise ValueError("final arms do not share one dataset contract")

        revision = str(api.repo_info(upload_repo, repo_type="dataset").sha)
        arm_dir = out / "inputs" / arm
        for name in INPUT_FILES:
            _restore_remote_file(
                repo=upload_repo,
                revision=revision,
                remote=f"{prefix}/scored/{name}",
                destination=arm_dir / name,
            )
        _restore_remote_file(
            repo=upload_repo,
            revision=revision,
            remote=f"{prefix}/analysis/analysis.json",
            destination=arm_dir / "analysis.json",
        )
        analysis_path = arm_dir / "analysis.json"
        if _sha256(analysis_path) != workflow.get("final_analysis_sha256"):
            raise ValueError(f"{arm}: remote final analysis differs from workflow")

        scored = _read_jsonl(arm_dir / "scored.jsonl")
        if len(scored) != expected_rows:
            raise ValueError(
                f"{arm}: expected {expected_rows} final rows, got {len(scored)}"
            )
        by_problem: dict[str, set[int]] = defaultdict(set)
        for row in scored:
            by_problem[str(row["problem_id"])].add(int(row["sample_index"]))
        expected_indices = set(range(int(policy["eligibility"]["samples_per_problem"])))
        if len(by_problem) != int(policy["eligibility"]["final_problems"]) or any(
            indices != expected_indices for indices in by_problem.values()
        ):
            raise ValueError(f"{arm}: final problem/sample topology drifted")
        problem_sets[arm] = set(by_problem)
        arms[arm] = {
            "workflow": str(workflow_path),
            "workflow_sha256": _sha256(workflow_path),
            "selected_step": step,
            "source_repo": upload_repo,
            "source_revision": revision,
            "source_prefix": prefix,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(arm_dir.iterdir())
                if path.is_file()
            },
        }

    if len({frozenset(ids) for ids in problem_sets.values()}) != 1:
        raise ValueError("reserved final problem IDs differ across arms")
    assert dataset_contract is not None
    manifest = {
        "schema_version": 1,
        "policy": cfg.policy,
        "policy_sha256": _sha256(_resolve_repo_path(cfg.policy)),
        "dataset_repo": dataset_contract[0],
        "dataset_revision": dataset_contract[1],
        "problem_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(next(iter(problem_sets.values())))) + "\n").encode()
        ).hexdigest(),
        "arms": arms,
    }
    _write_json(out / "input_manifest.json", manifest)
    return manifest


def _validate_verdicts(
    arm: str,
    scored: Sequence[Mapping[str, Any]],
    verdict_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    verdicts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in verdict_rows:
        problem_id = str(row["problem_id"])
        source_sha = str(row["source_sha256"])
        source = row.get("source")
        if not isinstance(source, str) or _source_sha(source) != source_sha:
            raise ValueError(f"{arm}/{problem_id}: verdict source hash drifted")
        key = (problem_id, source_sha)
        if key in verdicts:
            raise ValueError(f"{arm}: duplicate verdict {key}")
        verdicts[key] = dict(row)

    for row in scored:
        if row.get("correct") is not True:
            continue
        source_sha = row.get("source_sha256")
        key = (str(row["problem_id"]), str(source_sha))
        verdict = verdicts.get(key)
        if source_sha is None or verdict is None or verdict.get("correct") is not True:
            raise ValueError(f"{arm}: correct sample has no matching correct verdict")
    return verdicts


def build_measurement_tasks(
    inputs_root: Path,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Return deterministic unique-program tasks and validated scored rows."""

    tasks: list[dict[str, Any]] = []
    scored_by_arm: dict[str, list[dict[str, Any]]] = {}
    sample_keys: dict[str, set[tuple[str, int]]] = {}
    for arm in ARMS:
        arm_dir = inputs_root / arm
        scored = _read_jsonl(arm_dir / "scored.jsonl")
        verdicts = _validate_verdicts(
            arm, scored, _read_jsonl(arm_dir / "verdicts.jsonl")
        )
        scored_by_arm[arm] = scored
        sample_keys[arm] = {
            (str(row["problem_id"]), int(row["sample_index"])) for row in scored
        }
        used = {
            (str(row["problem_id"]), str(row["source_sha256"]))
            for row in scored
            if row.get("correct") is True
        }
        for problem_id, source_sha in sorted(used):
            verdict = verdicts[(problem_id, source_sha)]
            tasks.append(
                {
                    "arm": arm,
                    "problem_id": problem_id,
                    "source_sha256": source_sha,
                    "source": verdict["source"],
                    "candidate_id": verdict["candidate_id"],
                    "verdict": verdict,
                }
            )
    if len({frozenset(keys) for keys in sample_keys.values()}) != 1:
        raise ValueError("final sample keys differ across arms")

    def order(task: Mapping[str, Any]) -> str:
        identity = "\0".join(
            (
                str(seed),
                str(task["arm"]),
                str(task["problem_id"]),
                str(task["source_sha256"]),
            )
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    tasks.sort(key=order)
    return tasks, scored_by_arm


def _load_existing_measurements(
    path: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.is_file():
        return result
    for row in _read_jsonl(path):
        key = (str(row["arm"]), str(row["problem_id"]), str(row["source_sha256"]))
        if key in result:
            raise ValueError(f"duplicate persisted measurement: {key}")
        result[key] = row
    return result


def _assert_no_competing_experiment() -> None:
    own_pid = os.getpid()
    needles = ("run_sdf_code_driver", "run_sdf_code_arm", "axolotl.cli.train")
    conflicts: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(needle in command for needle in needles):
            conflicts.append(int(entry.name))
    if conflicts:
        raise RuntimeError(
            f"competing experiment processes are still alive: {conflicts}"
        )


def measure(
    cfg: SdfCodeEfficiencyConfig,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    out = Path(cfg.out)
    measurement_cfg = policy["measurement"]
    seed = int(measurement_cfg["task_order"]["seed"])
    block_size = int(measurement_cfg["task_order"]["block_size"])
    tasks, _ = build_measurement_tasks(out / "inputs", seed=seed)
    measurements_path = out / "measurements.jsonl"
    blocks_path = out / "blocks.jsonl"
    existing = _load_existing_measurements(measurements_path)
    expected_keys = {
        (str(task["arm"]), str(task["problem_id"]), str(task["source_sha256"]))
        for task in tasks
    }
    unknown = set(existing) - expected_keys
    if unknown:
        raise ValueError(
            f"persisted measurements contain unknown tasks: {sorted(unknown)[:3]}"
        )
    remaining = [
        task
        for task in tasks
        if (str(task["arm"]), str(task["problem_id"]), str(task["source_sha256"]))
        not in existing
    ]
    if not remaining:
        return {
            "status": "complete",
            "tasks": len(tasks),
            "restored": len(existing),
            "measured_this_run": 0,
        }

    _assert_no_competing_experiment()
    only = {str(task["problem_id"]) for task in tasks}
    score_cfg = StarScoreConfig(
        out=str(out / "source"),
        dataset_repo=str(manifest["dataset_repo"]),
        dataset_revision=str(manifest["dataset_revision"]),
        upload_repo="unused",
        hf_prefix="transfer_followup/20260806/sdf/code-efficiency-input",
        shard_count=1,
        n_samples=8,
        poll_seconds=5,
        timeout_s=float(measurement_cfg["timeout_seconds"]),
        mem_limit_mb=int(measurement_cfg["memory_limit_mb"]),
        correctness_workers=1,
        measure_efficiency=True,
        upload=False,
    )
    records = load_test_records(score_cfg, out / "source", only=only)
    prior_blocks = _read_jsonl(blocks_path) if blocks_path.is_file() else []
    next_block = len(prior_blocks)
    completed = 0
    for start in range(0, len(remaining), block_size):
        block_tasks = remaining[start : start + block_size]
        _assert_no_competing_experiment()
        baseline = _measure_baseline(
            timeout_s=score_cfg.timeout_s, mem_limit_mb=score_cfg.mem_limit_mb
        )
        calibration = _measure_latency_calibration(
            timeout_s=score_cfg.timeout_s, mem_limit_mb=score_cfg.mem_limit_mb
        )
        block = {
            "schema_version": 1,
            "block": next_block,
            "tasks": len(block_tasks),
            "baseline": baseline,
            "latency_calibration": calibration,
            "host": {
                "platform": platform.platform(),
                "python": sys.version,
                "uname": list(platform.uname()),
                "cpu_count": os.cpu_count(),
                "load_average": list(os.getloadavg()),
            },
        }
        _append_jsonl_fsync(blocks_path, block)
        baseline_rss = float(baseline["median_rss_bytes"])
        calibration_s = float(calibration["median_time_s"])
        for task in block_tasks:
            problem_id = str(task["problem_id"])
            record = records[problem_id]
            measured, _ = _measure_candidate(
                {
                    "candidate_id": task["candidate_id"],
                    "solution_index": 0,
                    "source": task["source"],
                },
                [],
                [
                    {
                        "input": str(record["synth_input"]),
                        "output": str(record["synth_output"]),
                        "source": "synth",
                    }
                ],
                baseline_rss,
                timeout_s=score_cfg.timeout_s,
                mem_limit_mb=score_cfg.mem_limit_mb,
                correctness_verdict={**task["verdict"], "status": "correct"},
            )
            result = {
                "schema_version": 1,
                "arm": task["arm"],
                "problem_id": problem_id,
                "source_sha256": task["source_sha256"],
                "block": next_block,
                "host_latency_calibration_s": calibration_s,
                **measured,
            }
            _append_jsonl_fsync(measurements_path, result)
            completed += 1
            if completed % 25 == 0:
                print(
                    f"[sdf-code-efficiency] measured {completed}/{len(remaining)} "
                    f"this run ({len(existing) + completed}/{len(tasks)} total)",
                    flush=True,
                )
        next_block += 1
    return {
        "status": "complete",
        "tasks": len(tasks),
        "restored": len(existing),
        "measured_this_run": completed,
    }


def _bootstrap_problem_mean(
    values: Mapping[str, Sequence[float]], *, draws: int, seed: int
) -> dict[str, Any]:
    import random

    problem_means = {
        problem_id: statistics.fmean(observations)
        for problem_id, observations in values.items()
        if observations
    }
    if not problem_means:
        return {"problems": 0, "samples": 0, "mean": None, "ci95": None}
    ordered = [problem_means[key] for key in sorted(problem_means)]
    rng = random.Random(seed)
    n = len(ordered)
    estimates = sorted(
        statistics.fmean(ordered[rng.randrange(n)] for _ in range(n))
        for _ in range(draws)
    )
    return {
        "problems": n,
        "samples": sum(len(values[key]) for key in problem_means),
        "mean": statistics.fmean(ordered),
        "ci95": [
            estimates[int(0.025 * draws)],
            estimates[min(draws - 1, int(0.975 * draws))],
        ],
    }


def _arm_context(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(row.get("correct") is True for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "correct_rate": correct / len(rows),
        "correctness_status": dict(
            sorted(Counter(str(row.get("correctness_status")) for row in rows).items())
        ),
    }


def paired_analysis(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    measurements: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    left: str,
    right: str,
    draws: int,
    seed: int,
    minimum_problems: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left_map = {
        (str(row["problem_id"]), int(row["sample_index"])): row for row in left_rows
    }
    right_map = {
        (str(row["problem_id"]), int(row["sample_index"])): row for row in right_rows
    }
    if left_map.keys() != right_map.keys():
        raise ValueError(f"{left}/{right}: paired sample keys differ")

    transitions = Counter()
    pairs: list[dict[str, Any]] = []
    quality_values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("calibrated_time", "raw_time", "peak_rss")
    }
    all_values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ("calibrated_time", "raw_time", "peak_rss")
    }
    flag_counts = {left: Counter(), right: Counter()}

    for problem_id, sample_index in sorted(left_map):
        left_row = left_map[(problem_id, sample_index)]
        right_row = right_map[(problem_id, sample_index)]
        left_correct = left_row.get("correct") is True
        right_correct = right_row.get("correct") is True
        transition = f"{int(left_correct)}->{int(right_correct)}"
        transitions[transition] += 1
        pair: dict[str, Any] = {
            "problem_id": problem_id,
            "sample_index": sample_index,
            "left": left,
            "right": right,
            "left_correct": left_correct,
            "right_correct": right_correct,
            "correctness_transition": transition,
            "efficiency_included": False,
        }
        if left_correct and right_correct:
            left_key = (left, problem_id, str(left_row["source_sha256"]))
            right_key = (right, problem_id, str(right_row["source_sha256"]))
            left_measure = measurements.get(left_key)
            right_measure = measurements.get(right_key)
            if (
                left_measure is not None
                and right_measure is not None
                and left_measure.get("status") == "measured"
                and right_measure.get("status") == "measured"
            ):
                left_peak = float(left_measure["baseline_subtracted_peak_bytes"])
                right_peak = float(right_measure["baseline_subtracted_peak_bytes"])
                if left_peak > 0 and right_peak > 0:
                    left_time = float(left_measure["median_time_s"])
                    right_time = float(right_measure["median_time_s"])
                    left_calibrated = left_time / float(
                        left_measure["host_latency_calibration_s"]
                    )
                    right_calibrated = right_time / float(
                        right_measure["host_latency_calibration_s"]
                    )
                    values = {
                        "calibrated_time": math.log(right_calibrated / left_calibrated),
                        "raw_time": math.log(right_time / left_time),
                        "peak_rss": math.log(right_peak / left_peak),
                    }
                    left_flags = list(left_measure.get("flags") or [])
                    right_flags = list(right_measure.get("flags") or [])
                    for flag in left_flags:
                        flag_counts[left][str(flag)] += 1
                    for flag in right_flags:
                        flag_counts[right][str(flag)] += 1
                    clean = not left_flags and not right_flags
                    for name, value in values.items():
                        all_values[name][problem_id].append(value)
                        if clean:
                            quality_values[name][problem_id].append(value)
                    pair.update(
                        efficiency_included=True,
                        quality_clean=clean,
                        left_source_sha256=left_row["source_sha256"],
                        right_source_sha256=right_row["source_sha256"],
                        calibrated_time_log_ratio_right_over_left=values[
                            "calibrated_time"
                        ],
                        raw_time_log_ratio_right_over_left=values["raw_time"],
                        peak_rss_log_ratio_right_over_left=values["peak_rss"],
                        left_flags=left_flags,
                        right_flags=right_flags,
                    )
        pairs.append(pair)

    quality = {
        name: _bootstrap_problem_mean(values, draws=draws, seed=seed + index)
        for index, (name, values) in enumerate(quality_values.items())
    }
    all_measured = {
        name: _bootstrap_problem_mean(values, draws=draws, seed=seed + 100 + index)
        for index, (name, values) in enumerate(all_values.items())
    }
    paired_problems = quality["calibrated_time"]["problems"]
    summary = {
        "left": left,
        "right": right,
        "ratio_orientation": "right_over_left",
        "n": len(left_map),
        "correctness_transitions": dict(sorted(transitions.items())),
        "left_context": _arm_context(left_rows),
        "right_context": _arm_context(right_rows),
        "quality_clean": quality,
        "all_measured": all_measured,
        "measurement_flag_counts": {
            arm: dict(sorted(counts.items())) for arm, counts in flag_counts.items()
        },
        "minimum_paired_problems_for_headline": minimum_problems,
        "headline_power": (
            "adequate" if paired_problems >= minimum_problems else "underpowered"
        ),
    }
    return summary, pairs


def analyze(
    cfg: SdfCodeEfficiencyConfig,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    out = Path(cfg.out)
    measurement_rows = _read_jsonl(out / "measurements.jsonl")
    measurements = {
        (str(row["arm"]), str(row["problem_id"]), str(row["source_sha256"])): row
        for row in measurement_rows
    }
    if len(measurements) != len(measurement_rows):
        raise ValueError("measurements contain duplicate keys")
    _, scored_by_arm = build_measurement_tasks(
        out / "inputs", seed=int(policy["measurement"]["task_order"]["seed"])
    )
    expected_measurements = sum(
        len(
            {
                (str(row["problem_id"]), str(row["source_sha256"]))
                for row in scored_by_arm[arm]
                if row.get("correct") is True
            }
        )
        for arm in ARMS
    )
    if len(measurements) != expected_measurements:
        raise ValueError(
            f"measurement store is incomplete: {len(measurements)}/{expected_measurements}"
        )

    analysis_policy = policy["analysis"]
    draws = int(analysis_policy["uncertainty"]["draws"])
    seed = int(analysis_policy["uncertainty"]["seed"])
    minimum = int(analysis_policy["minimum_paired_problems_for_headline"])
    contrasts: dict[str, Any] = {}
    for index, (left, right) in enumerate((PRIMARY_CONTRAST, *SECONDARY_CONTRASTS)):
        summary, pairs = paired_analysis(
            scored_by_arm[left],
            scored_by_arm[right],
            measurements,
            left=left,
            right=right,
            draws=draws,
            seed=seed + index * 1000,
            minimum_problems=minimum,
        )
        name = f"{left}_vs_{right}"
        contrasts[name] = summary
        _write_jsonl(out / "pairs" / f"{name}.jsonl", pairs)

    result = {
        "schema_version": 1,
        "policy": cfg.policy,
        "policy_sha256": _sha256(_resolve_repo_path(cfg.policy)),
        "input_manifest_sha256": _sha256(out / "input_manifest.json"),
        "measurements_sha256": _sha256(out / "measurements.jsonl"),
        "blocks_sha256": _sha256(out / "blocks.jsonl"),
        "measurement_n": len(measurements),
        "per_arm": {arm: _arm_context(scored_by_arm[arm]) for arm in ARMS},
        "primary_contrast": "latency_vs_memory",
        "contrasts": contrasts,
    }
    _write_json(out / "analysis.json", result)
    return result


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    allowed = {
        "config.yaml",
        "input_manifest.json",
        "blocks.jsonl",
        "measurements.jsonl",
        "analysis.json",
    }
    paths = [root / name for name in sorted(allowed)]
    paths.extend(sorted((root / "pairs").glob("*.jsonl")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"efficiency artifacts are incomplete: {missing}")
    return [
        {
            "file": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def persist(cfg: SdfCodeEfficiencyConfig) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(cfg.out)
    artifacts = _artifact_manifest(out)
    complete = {
        "schema_version": 1,
        "artifacts": artifacts,
        "analysis_sha256": _sha256(out / "analysis.json"),
    }
    _write_json(out / "complete.json", complete)
    api = HfApi()
    api.create_repo(
        cfg.attribution_repo, repo_type="model", private=True, exist_ok=True
    )
    prefix = cfg.attribution_prefix.strip("/")
    info = api.upload_folder(
        folder_path=str(out),
        repo_id=cfg.attribution_repo,
        repo_type="model",
        path_in_repo=prefix,
        allow_patterns=[
            "config.yaml",
            "input_manifest.json",
            "blocks.jsonl",
            "measurements.jsonl",
            "analysis.json",
            "pairs/*.jsonl",
            "complete.json",
        ],
        commit_message="Persist SDF code efficiency comparison",
    )
    revision = str(info.oid)
    for artifact in (
        *artifacts,
        {"file": "complete.json", "sha256": _sha256(out / "complete.json")},
    ):
        downloaded = Path(
            hf_hub_download(
                cfg.attribution_repo,
                f"{prefix}/{artifact['file']}",
                repo_type="model",
                revision=revision,
                local_dir=out / "remote_verification",
                force_download=True,
            )
        )
        if _sha256(downloaded) != artifact["sha256"]:
            raise ValueError(f"remote checksum mismatch: {artifact['file']}")
    marker = {
        "schema_version": 1,
        "repo": cfg.attribution_repo,
        "prefix": prefix,
        "revision": revision,
        "remote_verified": True,
        "verified_files": len(artifacts) + 1,
        "complete_sha256": _sha256(out / "complete.json"),
    }
    _write_json(out / "persistence.json", marker)
    marker_info = api.upload_file(
        path_or_fileobj=str(out / "persistence.json"),
        path_in_repo=f"{prefix}/persistence.json",
        repo_id=cfg.attribution_repo,
        repo_type="model",
        commit_message="Record verified SDF code efficiency comparison",
    )
    marker["marker_revision"] = str(marker_info.oid)
    _write_json(out / "persistence.json", marker)
    return marker


async def run(cfg: SdfCodeEfficiencyConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    policy = load_policy(_resolve_repo_path(cfg.policy))
    result: dict[str, Any] = {}
    if cfg.phase in {"measure", "all"}:
        manifest = await asyncio.to_thread(restore_inputs, cfg, policy)
        result["measurement"] = await asyncio.to_thread(measure, cfg, policy, manifest)
    if cfg.phase in {"analyze", "all"}:
        if not (out / "input_manifest.json").is_file():
            raise FileNotFoundError("input manifest is required for analysis")
        result["analysis"] = await asyncio.to_thread(analyze, cfg, policy)
    if cfg.phase in {"persist", "all"}:
        result["persistence"] = await asyncio.to_thread(persist, cfg)
    return result


def main() -> None:
    asyncio.run(run(parse(SdfCodeEfficiencyConfig)))


if __name__ == "__main__":
    main()


__all__ = [
    "SdfCodeEfficiencyConfig",
    "analyze",
    "build_measurement_tasks",
    "load_policy",
    "measure",
    "paired_analysis",
    "persist",
    "restore_inputs",
    "run",
]
