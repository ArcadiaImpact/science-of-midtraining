"""Fail-closed preflight and postflight checks for one RunPod shard."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import resource
import statistics
import sys
from pathlib import Path
from typing import Iterable, Mapping

try:
    from .measure_pairs import run_solution_sandboxed
    from .synth_workloads import validate_generator_source
except ImportError:  # pragma: no cover - direct script invocation
    from measure_pairs import run_solution_sandboxed  # type: ignore
    from synth_workloads import validate_generator_source  # type: ignore


MIB = 1024 * 1024
MAX_EMPTY_RUNNER_RSS_BYTES = 64 * MIB
# ASLR and allocator page selection produced 3.3--3.9 MiB ranges across ten
# empty-runner trials on the two actual CPU hosts.  Eight MiB remains far below
# the 50 MiB inherited-parent discriminator while avoiding a host-noise false
# negative.
MAX_EMPTY_RUNNER_SPREAD_BYTES = 8 * MIB
MIN_PARENT_CHILD_RSS_GAP_BYTES = 50 * MIB


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number}: row must be an object")
            rows.append(row)
    return rows


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(rows: Iterable[Mapping[str, object]], *, path: Path) -> list[str]:
    result: list[str] = []
    for line_number, row in enumerate(rows, 1):
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError(
                f"{path}: line {line_number}: problem_id must be a non-empty string"
            )
        result.append(problem_id)
    if len(result) != len(set(result)):
        raise ValueError(f"{path}: duplicate problem_id")
    return result


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss * (1 if sys.platform == "darwin" else 1024))


def _rss_calibration() -> dict[str, object]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("RSS calibration requires Linux")
    # Raise the orchestrator's high-water mark, then prove the exec/fork runner
    # reports the small grandchild rather than inheriting that historical RSS.
    ballast = bytearray(96 * MIB)
    for offset in range(0, len(ballast), 4096):
        ballast[offset] = 1
    parent_rss = _rss_bytes()
    trials: list[int] = []
    for _ in range(5):
        report = run_solution_sandboxed(
            "pass\n", "", timeout_s=3.0, mem_limit_mb=1024
        )
        if not report.get("ok"):
            raise RuntimeError(f"empty runner calibration failed: {report}")
        rss = report.get("rss_bytes")
        if not isinstance(rss, (int, float)) or rss <= 0:
            raise RuntimeError(f"empty runner returned invalid RSS: {rss!r}")
        trials.append(int(rss))
    median = float(statistics.median(trials))
    spread = max(trials) - min(trials)
    gap = parent_rss - max(trials)
    if median >= MAX_EMPTY_RUNNER_RSS_BYTES:
        raise RuntimeError(
            f"empty-runner RSS too high: median={median} bytes"
        )
    if spread > MAX_EMPTY_RUNNER_SPREAD_BYTES:
        raise RuntimeError(
            f"empty-runner RSS too variable: spread={spread} bytes"
        )
    if gap < MIN_PARENT_CHILD_RSS_GAP_BYTES:
        raise RuntimeError(
            "runner appears to inherit parent RSS high-water mark: "
            f"parent={parent_rss}, max_child={max(trials)}, gap={gap}"
        )
    return {
        "platform": sys.platform,
        "parent_rss_bytes": parent_rss,
        "child_rss_trials_bytes": trials,
        "median_child_rss_bytes": median,
        "child_rss_spread_bytes": spread,
        "parent_child_gap_bytes": gap,
        "thresholds": {
            "max_empty_runner_rss_bytes": MAX_EMPTY_RUNNER_RSS_BYTES,
            "max_empty_runner_spread_bytes": MAX_EMPTY_RUNNER_SPREAD_BYTES,
            "min_parent_child_rss_gap_bytes": MIN_PARENT_CHILD_RSS_GAP_BYTES,
        },
    }


def preflight(
    input_dir: Path, out_path: Path, *, expected_problems: int
) -> dict[str, object]:
    problems_path = input_dir / "problems.jsonl"
    generators_path = input_dir / "generators.jsonl"
    meta_path = input_dir / "meta.json"
    problems = _read_jsonl(problems_path)
    generators = _read_jsonl(generators_path)
    problem_ids = _ids(problems, path=problems_path)
    generator_ids = _ids(generators, path=generators_path)
    if len(problem_ids) != expected_problems:
        raise ValueError(
            f"expected {expected_problems} problems, found {len(problem_ids)}"
        )
    if generator_ids != problem_ids:
        raise ValueError(
            "generator rows must cover every problem exactly once in input order"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path}: must contain an object")
    required_meta = {
        "dataset": "deepmind/code_contests",
        "dataset_revision": "802411c3010cb00d1b05bad57ca77365a3c699d6",
        "split": "train",
        "shard_problem_count": expected_problems,
        "shard_problems_sha256": _sha256(problems_path),
        "shard_generators_sha256": _sha256(generators_path),
        "first_problem_id": problem_ids[0],
        "last_problem_id": problem_ids[-1],
    }
    for field, expected in required_meta.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"{meta_path}: {field}={meta.get(field)!r}, expected {expected!r}"
            )
    generated = 0
    skipped = 0
    for line_number, row in enumerate(generators, 1):
        keys = set(row)
        if keys == {"problem_id", "generator_source"}:
            source = row["generator_source"]
            if not isinstance(source, str):
                raise ValueError(
                    f"{generators_path}: line {line_number}: source must be a string"
                )
            validate_generator_source(source)
            generated += 1
        elif keys == {"problem_id", "skip"}:
            reason = row["skip"]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"{generators_path}: line {line_number}: empty skip reason"
                )
            skipped += 1
        else:
            raise ValueError(
                f"{generators_path}: line {line_number}: invalid exact key set"
            )
    result = {
        "status": "passed",
        "expected_problems": expected_problems,
        "problem_count": len(problem_ids),
        "generator_count": generated,
        "skip_count": skipped,
        "inputs": {
            "problems.jsonl": _sha256(problems_path),
            "generators.jsonl": _sha256(generators_path),
            "meta.json": _sha256(meta_path),
        },
        "rss_calibration": _rss_calibration(),
    }
    _write_json(out_path, result)
    return result


def verify_question_manifest(
    question_dir: Path, *, expected_eval_fraction: float
) -> dict[str, object]:
    manifest_path = question_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: must contain an object")
    if manifest.get("eval_fraction") != expected_eval_fraction:
        raise ValueError(
            "question export eval_fraction mismatch: "
            f"{manifest.get('eval_fraction')!r} != {expected_eval_fraction!r}"
        )
    unique = manifest.get("unique_problem_counts")
    if not isinstance(unique, dict) or unique.get("overlap") != 0:
        raise ValueError("question manifest does not prove zero split overlap")
    outputs = manifest.get("output_artifacts")
    if not isinstance(outputs, dict):
        raise ValueError("question manifest lacks output hashes")
    required_outputs = {
        "questions.jsonl",
        "pareto_triplets.jsonl",
        "train/tradeoff.jsonl",
        "train/jointly_dominant.jsonl",
        "train/pareto_triplets.jsonl",
        "eval/tradeoff.jsonl",
        "eval/jointly_dominant.jsonl",
        "eval/pareto_triplets.jsonl",
    }
    if not required_outputs.issubset(outputs):
        missing = sorted(required_outputs - set(outputs))
        raise ValueError(f"question manifest lacks required outputs: {missing}")
    for relative, metadata in outputs.items():
        if not isinstance(relative, str) or not isinstance(metadata, dict):
            raise ValueError("invalid question output manifest entry")
        path = question_dir / relative
        if _sha256(path) != metadata.get("sha256"):
            raise ValueError(f"question output hash mismatch: {relative}")
        actual_rows = sum(
            1 for line in path.open(encoding="utf-8") if line.strip()
        )
        if actual_rows != metadata.get("rows"):
            raise ValueError(f"question output row mismatch: {relative}")
    questions = _read_jsonl(question_dir / "questions.jsonl")
    seen: set[tuple[str, str]] = set()
    train_ids: set[str] = set()
    eval_ids: set[str] = set()
    for row in questions:
        problem_id = str(row.get("problem_id"))
        category = str(row.get("category"))
        key = (problem_id, category)
        if key in seen:
            raise ValueError(f"duplicate question for problem/category: {key}")
        seen.add(key)
        split = row.get("split")
        if split == "train":
            train_ids.add(problem_id)
        elif split == "eval":
            eval_ids.add(problem_id)
        else:
            raise ValueError(f"question has invalid split: {split!r}")
        solutions = row.get("solutions")
        if not isinstance(solutions, list) or len(solutions) != 2:
            raise ValueError(f"question {key} does not contain exactly two solutions")
        first, second = solutions
        if not isinstance(first, dict) or not isinstance(second, dict):
            raise ValueError(f"question {key} has invalid solutions")
        try:
            first_ast = ast.dump(ast.parse(str(first["source"])), include_attributes=False)
            second_ast = ast.dump(ast.parse(str(second["source"])), include_attributes=False)
        except (KeyError, SyntaxError) as exc:
            raise ValueError(f"question {key} has invalid source") from exc
        if first_ast == second_ast:
            raise ValueError(f"question {key} repeats the same AST")
        first_times = [float(value) for value in first.get("times_s", [])]
        second_times = [float(value) for value in second.get("times_s", [])]
        first_rss = [float(value) for value in first.get("rss_trials_bytes", [])]
        second_rss = [float(value) for value in second.get("rss_trials_bytes", [])]
        if not all((first_times, second_times, first_rss, second_rss)):
            raise ValueError(f"question {key} lacks trial arrays")
        if category == "tradeoff":
            if (
                first.get("role") != "speed"
                or second.get("role") != "memory"
                or max(first_times) >= min(second_times)
                or max(second_rss) >= min(first_rss)
            ):
                raise ValueError(f"question {key} fails strict tradeoff ordering")
        elif category == "jointly_dominant":
            if (
                first.get("role") != "winner"
                or second.get("role") != "loser"
                or max(first_times) >= min(second_times)
                or max(first_rss) >= min(second_rss)
            ):
                raise ValueError(f"question {key} fails strict dominance ordering")
        else:
            raise ValueError(f"question {key} has invalid category")
    if train_ids & eval_ids:
        raise ValueError("question rows leak problems across train and eval")
    return manifest


def postflight(
    run_dir: Path,
    question_dir: Path,
    out_path: Path,
    *,
    expected_problems: int,
) -> dict[str, object]:
    candidates_path = run_dir / "candidates.jsonl"
    synth_results_path = run_dir / "synth_results.jsonl"
    synth_tests_path = run_dir / "synth_tests.jsonl"
    measurements_path = run_dir / "measurements.jsonl"
    candidates = _read_jsonl(candidates_path)
    synth_results = _read_jsonl(synth_results_path)
    synth_tests = _read_jsonl(synth_tests_path)
    measurements = _read_jsonl(measurements_path)
    candidate_ids = _ids(candidates, path=candidates_path)
    result_ids = _ids(synth_results, path=synth_results_path)
    test_ids = _ids(synth_tests, path=synth_tests_path)
    measurement_ids = _ids(measurements, path=measurements_path)
    if len(candidate_ids) != expected_problems:
        raise ValueError(
            f"candidate coverage is {len(candidate_ids)}, expected {expected_problems}"
        )
    if result_ids != candidate_ids:
        raise ValueError("synthesis results do not exactly cover candidates in order")
    if measurement_ids != test_ids:
        raise ValueError("measurements do not exactly cover synthesized tests in order")
    if not measurements:
        raise ValueError("shard produced no measurable synthesized workloads")

    baselines: list[float] = []
    measured_solution_count = 0
    median_rss_values: list[float] = []
    first_baseline: object | None = None
    for row in measurements:
        baseline = row.get("baseline")
        if not isinstance(baseline, dict):
            raise ValueError("measurement row lacks baseline")
        if first_baseline is None:
            first_baseline = baseline
        elif baseline != first_baseline:
            raise ValueError("measurement rows do not share one baseline record")
        median = baseline.get("median_rss_bytes")
        if not isinstance(median, (int, float)) or not math.isfinite(float(median)):
            raise ValueError("measurement baseline has invalid median RSS")
        baselines.append(float(median))
        baseline_trials = baseline.get("rss_trials_bytes")
        if (
            not isinstance(baseline_trials, list)
            or len(baseline_trials) < 3
            or any(not isinstance(value, (int, float)) for value in baseline_trials)
        ):
            raise ValueError("measurement baseline has invalid RSS trials")
        if max(baseline_trials) - min(baseline_trials) > MAX_EMPTY_RUNNER_SPREAD_BYTES:
            raise ValueError("measurement baseline RSS trials are too variable")
        for solution in row.get("solutions", []):
            if not isinstance(solution, dict) or solution.get("status") != "measured":
                continue
            rss = solution.get("median_rss_bytes")
            if isinstance(rss, (int, float)) and math.isfinite(float(rss)):
                measured_solution_count += 1
                median_rss_values.append(float(rss))
    baseline_median = float(statistics.median(baselines))
    if baseline_median >= MAX_EMPTY_RUNNER_RSS_BYTES:
        raise ValueError(f"measurement baseline RSS is implausible: {baseline_median}")
    if measured_solution_count == 0:
        raise ValueError("no solutions were measured")
    distinct_rss = len(set(median_rss_values))
    minimum_distinct = min(
        measured_solution_count, max(20, math.ceil(measured_solution_count * 0.05))
    )
    if distinct_rss < minimum_distinct:
        raise ValueError(
            "measured RSS appears collapsed: "
            f"{distinct_rss} distinct values for {measured_solution_count} solutions"
        )

    manifest = verify_question_manifest(
        question_dir, expected_eval_fraction=0
    )
    result = {
        "status": "passed",
        "expected_problems": expected_problems,
        "candidate_problem_count": len(candidate_ids),
        "synthesized_problem_count": len(test_ids),
        "measurement_problem_count": len(measurement_ids),
        "measured_solution_count": measured_solution_count,
        "distinct_median_rss_values": distinct_rss,
        "baseline_median_rss_bytes": baseline_median,
        "question_counts": manifest.get("counts"),
        "artifacts": {
            path.name: _sha256(path)
            for path in (
                candidates_path,
                synth_results_path,
                synth_tests_path,
                measurements_path,
                question_dir / "manifest.json",
            )
        },
    }
    _write_json(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--input-dir", type=Path, required=True)
    pre.add_argument("--out", type=Path, required=True)
    pre.add_argument("--expected-problems", type=int, default=500)
    post = subparsers.add_parser("postflight")
    post.add_argument("--run-dir", type=Path, required=True)
    post.add_argument("--question-dir", type=Path, required=True)
    post.add_argument("--out", type=Path, required=True)
    post.add_argument("--expected-problems", type=int, default=500)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(
            args.input_dir, args.out, expected_problems=args.expected_problems
        )
    else:
        result = postflight(
            args.run_dir,
            args.question_dir,
            args.out,
            expected_problems=args.expected_problems,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
