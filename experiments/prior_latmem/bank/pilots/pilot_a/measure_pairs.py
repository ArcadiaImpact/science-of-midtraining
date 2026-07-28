"""Stage 2 for Pilot A: sandbox correctness checks and process measurements."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Iterable

try:
    from ...sandbox import _kill_process, _preexec_limits
except ImportError:  # pragma: no cover - direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sandbox import _kill_process, _preexec_limits

try:
    from .policy import MAX_TIME_SPREAD, MIN_PEAK_BYTES, MIN_TIME_SECONDS
except ImportError:  # pragma: no cover - direct script invocation
    from policy import MAX_TIME_SPREAD, MIN_PEAK_BYTES, MIN_TIME_SECONDS

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MEMORY_LIMIT_MB = 1024
PROTOCOL_PREFIX = "__SCIMT_PILOT_A_RSS__"


_STDIO_CHILD_RUNNER = r'''
import json
import pathlib
import socket
import sys
import time
import traceback
try:
    import resource
except ImportError:
    resource = None


class NetworkDisabledError(RuntimeError):
    pass


def _deny_socket(*args, **kwargs):
    raise NetworkDisabledError("network access is disabled in the bank sandbox")


socket.socket = _deny_socket
try:
    import _socket
    _socket.socket = _deny_socket
except Exception:
    pass


payload_path = pathlib.Path(sys.argv[1])
payload = payload_path.read_text(encoding="utf-8")
namespace = {"__name__": "__main__", "__file__": str(payload_path)}
report = {"ok": False, "error": None, "rss_bytes": None, "timings": {}}
started = time.perf_counter()
try:
    exec(compile(payload, str(payload_path), "exec"), namespace, namespace)
    report["ok"] = True
except SystemExit as exc:
    if exc.code is None or exc.code == 0:
        report["ok"] = True
    else:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=8)
except BaseException as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc(limit=8)
finally:
    report["timings"] = {"payload_wall_s": time.perf_counter() - started}
    try:
        if resource is None:
            raise RuntimeError("resource module unavailable")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["rss_bytes"] = int(rss * (1 if sys.platform == "darwin" else 1024))
    except Exception as exc:
        report["rss_error"] = f"{type(exc).__name__}: {exc}"
    sys.stderr.write("\n" + sys.argv[2] + json.dumps(report, sort_keys=True) + "\n")
    sys.stderr.flush()
'''


def _protocol_from_stderr(stderr: str) -> dict[str, object] | None:
    for line in reversed(stderr.splitlines()):
        if not line.startswith(PROTOCOL_PREFIX):
            continue
        try:
            report = json.loads(line[len(PROTOCOL_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict) and "ok" in report:
            return report
    return None


def run_solution_sandboxed(
    source: str,
    stdin_data: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB,
) -> dict[str, object]:
    """Run a stdin/stdout solution with the isolation properties of sandbox.py.

    This is the spec-authorized stdin/stdout variant: isolated Python, empty
    environment, fresh cwd, process-group kill, CPU/address-space limits, and
    the same Python socket denial. The candidate is only compiled in the child.
    """
    if not isinstance(source, str) or not isinstance(stdin_data, str):
        raise TypeError("source and stdin_data must be strings")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    with tempfile.TemporaryDirectory(prefix="scimt-latmem-pilot-a-") as temp_dir:
        root = Path(temp_dir)
        payload_path = root / "payload.py"
        runner_path = root / "runner.py"
        payload_path.write_text(source, encoding="utf-8")
        runner_path.write_text(textwrap.dedent(_STDIO_CHILD_RUNNER), encoding="utf-8")
        command = [
            sys.executable,
            "-I",
            str(runner_path),
            str(payload_path),
            PROTOCOL_PREFIX,
        ]
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(
                command,
                cwd=temp_dir,
                env={},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
                preexec_fn=_preexec_limits(timeout_s, mem_limit_mb),
            )
            try:
                stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                _kill_process(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                return {
                    "ok": False,
                    "error": "timeout",
                    "stdout": stdout[-4000:],
                    "stderr": stderr[-4000:],
                    "returncode": proc.returncode,
                    "parent_wall_s": time.perf_counter() - started,
                    "rss_bytes": None,
                }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"sandbox_start: {type(exc).__name__}: {exc}",
                "stdout": "",
                "stderr": "",
                "parent_wall_s": time.perf_counter() - started,
                "rss_bytes": None,
            }

    report = _protocol_from_stderr(stderr)
    if report is None:
        return {
            "ok": False,
            "error": "missing_protocol",
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "returncode": proc.returncode,
            "parent_wall_s": time.perf_counter() - started,
            "rss_bytes": None,
        }
    report["stdout"] = stdout
    report["stderr"] = stderr[-4000:]
    report["returncode"] = proc.returncode
    report["parent_wall_s"] = time.perf_counter() - started
    return report


def normalize_output(text: str) -> str:
    """Normalize line endings and trailing whitespace for judge comparison."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).rstrip()


def select_correctness_tests(tests: list[dict[str, str]]) -> list[tuple[int, dict[str, str]]]:
    """Choose at most the two smallest and two largest inputs."""
    ranked = sorted(enumerate(tests), key=lambda item: (len(item[1]["input"]), item[0]))
    selected = ranked[:2] + ranked[-2:]
    by_index = {index: test for index, test in selected}
    return [(index, by_index[index]) for index in sorted(by_index)]


def _spread(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    median = float(statistics.median(values))
    if median <= 0:
        return float("inf")
    return (max(values) - min(values)) / median


def _measure_baseline(
    *, timeout_s: float, mem_limit_mb: int | None
) -> dict[str, object]:
    rss_trials: list[int] = []
    wall = 0.0
    for _ in range(5):
        report = run_solution_sandboxed(
            "",
            "",
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        wall += float(report["parent_wall_s"])
        rss = report.get("rss_bytes")
        if not report.get("ok") or not isinstance(rss, int) or rss <= 0:
            raise RuntimeError(f"interpreter baseline failed: {report.get('error')}")
        rss_trials.append(rss)
    return {
        "rss_trials_bytes": rss_trials,
        "median_rss_bytes": float(statistics.median(rss_trials)),
        "measurement_wall_s": wall,
    }


def _measure_candidate(
    candidate: dict[str, object],
    correctness_tests: list[dict[str, str]],
    measurement_tests: list[dict[str, str]],
    baseline_rss_bytes: float,
    *,
    timeout_s: float,
    mem_limit_mb: int | None,
    correctness_verdict: dict[str, object] | None = None,
) -> tuple[dict[str, object], float]:
    if correctness_verdict is None:
        result, wall_total = check_candidate_correctness(
            candidate,
            correctness_tests,
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
    else:
        if correctness_verdict.get("candidate_id") != candidate.get("candidate_id"):
            raise ValueError("cached correctness verdict has the wrong candidate_id")
        result = dict(correctness_verdict)
        wall_total = 0.0
    if result["status"] != "correct":
        return result, wall_total
    result["status"] = "dropped"

    if not measurement_tests:
        result["drop_reason"] = "no_measurement_tests"
        return result, wall_total
    largest_test = max(
        enumerate(measurement_tests),
        key=lambda item: (len(item[1]["input"]), item[0]),
    )
    source = candidate["source"]
    assert isinstance(source, str)
    trials: list[dict[str, float | int]] = []
    for _ in range(3):
        report = run_solution_sandboxed(
            source,
            largest_test[1]["input"],
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        wall_total += float(report["parent_wall_s"])
        if not report.get("ok"):
            result["drop_reason"] = (
                "over_floor_timeout"
                if report.get("error") == "timeout"
                else "measurement_crash"
            )
            result["measurement_error"] = report.get("error")
            return result, wall_total
        rss = report.get("rss_bytes")
        if not isinstance(rss, int) or rss <= 0:
            result["drop_reason"] = "measurement_missing_rss"
            return result, wall_total
        report_timings = report.get("timings")
        payload_wall_s = (
            report_timings.get("payload_wall_s")
            if isinstance(report_timings, dict)
            else None
        )
        if not isinstance(payload_wall_s, (int, float)) or payload_wall_s < 0:
            result["drop_reason"] = "measurement_missing_payload_time"
            return result, wall_total
        trials.append(
            {
                "payload_wall_s": float(payload_wall_s),
                "parent_wall_s": float(report["parent_wall_s"]),
                "rss_bytes": rss,
            }
        )

    timings = [float(trial["payload_wall_s"]) for trial in trials]
    parent_walls = [float(trial["parent_wall_s"]) for trial in trials]
    rss_values = [int(trial["rss_bytes"]) for trial in trials]
    median_time = float(statistics.median(timings))
    median_parent_wall = float(statistics.median(parent_walls))
    median_rss = float(statistics.median(rss_values))
    time_spread = _spread(timings)
    rss_spread = _spread([float(value) for value in rss_values])
    subtracted = median_rss - baseline_rss_bytes
    flags: list[str] = []
    if median_time < MIN_TIME_SECONDS:
        flags.append("under_time_floor")
    if time_spread > MAX_TIME_SPREAD:
        flags.append("unstable")
    if subtracted <= 0:
        flags.append("under_baseline_noise")
    if subtracted < MIN_PEAK_BYTES:
        flags.append("under_peak_floor")
    result.update(
        {
            "status": "measured",
            "drop_reason": None,
            "largest_test_index": largest_test[0],
            "measurement_test_source": largest_test[1]["source"],
            "trials": trials,
            "times_s": timings,
            "parent_wall_times_s": parent_walls,
            "rss_trials_bytes": rss_values,
            "median_time_s": median_time,
            "median_parent_wall_s": median_parent_wall,
            "median_rss_bytes": median_rss,
            "baseline_subtracted_peak_bytes": subtracted,
            "time_spread": time_spread,
            "peak_spread": rss_spread,
            "flags": flags,
        }
    )
    return result, wall_total


def check_candidate_correctness(
    candidate: dict[str, object],
    tests: list[dict[str, str]],
    *,
    timeout_s: float,
    mem_limit_mb: int | None,
) -> tuple[dict[str, object], float]:
    """Apply the dataset correctness gate without measuring a workload."""
    source = candidate["source"]
    assert isinstance(source, str)
    result: dict[str, object] = {
        "candidate_id": candidate["candidate_id"],
        "solution_index": candidate["solution_index"],
        "source": source,
        "z_silence_hits": candidate.get("z_silence_hits", []),
        "style_flags": candidate.get("style_flags", []),
        "status": "dropped",
        "drop_reason": None,
        "correctness": [],
        "flags": [],
    }
    wall_total = 0.0
    selected_tests = select_correctness_tests(tests)
    if not selected_tests:
        result["drop_reason"] = "no_tests"
        return result, wall_total

    correctness: list[dict[str, object]] = []
    for test_index, test in selected_tests:
        report = run_solution_sandboxed(
            source,
            test["input"],
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        wall_total += float(report["parent_wall_s"])
        check: dict[str, object] = {
            "test_index": test_index,
            "source": test["source"],
            "ok": False,
        }
        if not report.get("ok"):
            reason = (
                "correctness_timeout"
                if report.get("error") == "timeout"
                else "correctness_crash"
            )
            check["error"] = report.get("error")
            correctness.append(check)
            result["correctness"] = correctness
            result["drop_reason"] = reason
            return result, wall_total
        actual = normalize_output(str(report.get("stdout", "")))
        expected = normalize_output(test["output"])
        if actual != expected:
            check["error"] = "output_mismatch"
            check["expected_excerpt"] = expected[:500]
            check["actual_excerpt"] = actual[:500]
            correctness.append(check)
            result["correctness"] = correctness
            result["drop_reason"] = "wrong_answer"
            return result, wall_total
        check["ok"] = True
        correctness.append(check)
    result["correctness"] = correctness
    result["status"] = "correct"
    return result, wall_total


def _read_candidate_rows(
    path: Path, *, limit: int | None
) -> Iterable[dict[str, object]]:
    yielded = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if limit is not None and yielded >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict) or not isinstance(row.get("problem_id"), str):
                raise ValueError(f"{path}: line {line_number}: invalid candidate row")
            yield row
            yielded += 1


def _read_existing(
    path: Path,
    *,
    measurement_source: str,
) -> tuple[set[str], dict[str, object] | None]:
    completed: set[str] = set()
    baseline: dict[str, object] | None = None
    if not path.exists():
        return completed, baseline
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid resume row: {exc}"
                ) from exc
            if not isinstance(row, dict) or not isinstance(row.get("problem_id"), str):
                raise ValueError(f"{path}: line {line_number}: invalid resume row")
            existing_source = row.get("measurement_source", "dataset")
            if existing_source != measurement_source:
                raise ValueError(
                    f"{path}: contains {existing_source!r} measurements, cannot "
                    f"resume a {measurement_source!r} run in the same out directory"
                )
            completed.add(row["problem_id"])
            candidate_baseline = row.get("baseline")
            if baseline is None and isinstance(candidate_baseline, dict):
                median = candidate_baseline.get("median_rss_bytes")
                if isinstance(median, (int, float)) and median > 0:
                    baseline = candidate_baseline
    return completed, baseline


def _read_synth_tests(path: Path) -> dict[str, dict[str, object]]:
    tests: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number}: row must be an object")
            problem_id = row.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError(
                    f"{path}: line {line_number}: problem_id must be non-empty"
                )
            if problem_id in tests:
                raise ValueError(f"{path}: duplicate problem_id {problem_id!r}")
            if row.get("source") != "synth":
                raise ValueError(
                    f"{path}: line {line_number}: source must be 'synth'"
                )
            if not isinstance(row.get("input"), str) or not isinstance(
                row.get("output"), str
            ):
                raise ValueError(
                    f"{path}: line {line_number}: input/output must be strings"
                )
            survivor_ids = row.get("survivor_ids")
            if not isinstance(survivor_ids, list) or any(
                not isinstance(value, str) for value in survivor_ids
            ):
                raise ValueError(
                    f"{path}: line {line_number}: survivor_ids must be strings"
                )
            verdicts = row.get("correctness_verdicts")
            if not isinstance(verdicts, list) or any(
                not isinstance(verdict, dict)
                or not isinstance(verdict.get("candidate_id"), str)
                for verdict in verdicts
            ):
                raise ValueError(
                    f"{path}: line {line_number}: correctness_verdicts must "
                    "contain per-candidate objects"
                )
            verdict_ids = [
                str(verdict["candidate_id"]) for verdict in verdicts
            ]
            if len(verdict_ids) != len(set(verdict_ids)):
                raise ValueError(
                    f"{path}: line {line_number}: duplicate correctness verdict"
                )
            tests[problem_id] = row
    return tests


def measure_file(
    candidates_path: Path | str,
    out_dir: Path | str,
    *,
    limit: int | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB,
    synth_tests_path: Path | str | None = None,
) -> dict[str, int]:
    """Measure candidate rows, appending completed problems for resumability."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    candidates_path = Path(candidates_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "measurements.jsonl"
    rows = list(_read_candidate_rows(candidates_path, limit=limit))
    synth_tests = (
        _read_synth_tests(Path(synth_tests_path))
        if synth_tests_path is not None
        else None
    )
    measurement_source = "synth" if synth_tests is not None else "dataset"
    if synth_tests is not None:
        rows = [
            row for row in rows if str(row["problem_id"]) in synth_tests
        ]
    completed, baseline = _read_existing(
        output_path,
        measurement_source=measurement_source,
    )
    pending = [row for row in rows if row["problem_id"] not in completed]
    skipped = len(rows) - len(pending)
    if not pending:
        return {"measured_problems": 0, "skipped_problems": skipped}

    if baseline is None:
        baseline = _measure_baseline(timeout_s=timeout_s, mem_limit_mb=mem_limit_mb)
    baseline_rss = float(baseline["median_rss_bytes"])
    host = {
        "platform": platform.platform(),
        "sys_platform": sys.platform,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }

    with output_path.open("a", encoding="utf-8") as handle:
        for problem_number, problem in enumerate(pending, 1):
            tests = problem.get("tests")
            candidates = problem.get("candidates")
            if not isinstance(tests, list) or not isinstance(candidates, list):
                raise ValueError(
                    f"candidate row {problem['problem_id']}: tests/candidates must be lists"
                )
            synth_test = (
                synth_tests[str(problem["problem_id"])]
                if synth_tests is not None
                else None
            )
            measurement_tests = (
                [
                    {
                        "source": "synth",
                        "input": str(synth_test["input"]),
                        "output": str(synth_test["output"]),
                    }
                ]
                if synth_test is not None
                else tests
            )
            survivor_ids = (
                set(str(value) for value in synth_test["survivor_ids"])
                if synth_test is not None
                else None
            )
            correctness_by_id = (
                {
                    str(verdict["candidate_id"]): verdict
                    for verdict in synth_test["correctness_verdicts"]
                    if isinstance(verdict, dict)
                }
                if synth_test is not None
                else None
            )
            candidate_ids = {
                str(candidate.get("candidate_id"))
                for candidate in candidates
                if isinstance(candidate, dict)
            }
            if (
                correctness_by_id is not None
                and set(correctness_by_id) != candidate_ids
            ):
                raise ValueError(
                    f"candidate row {problem['problem_id']}: cached correctness "
                    "verdicts do not match candidates"
                )
            if (
                survivor_ids is not None
                and correctness_by_id is not None
                and not survivor_ids.issubset(correctness_by_id)
            ):
                raise ValueError(
                    f"candidate row {problem['problem_id']}: synth survivors "
                    "lack cached correctness verdicts"
                )
            solution_rows = []
            problem_wall = 0.0
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError(
                        f"candidate row {problem['problem_id']}: candidate must be an object"
                    )
                candidate_id = str(candidate.get("candidate_id"))
                correctness_verdict = (
                    correctness_by_id[candidate_id]
                    if correctness_by_id is not None
                    else None
                )
                if survivor_ids is not None and candidate_id not in survivor_ids:
                    assert correctness_verdict is not None
                    measured = dict(correctness_verdict)
                    wall = 0.0
                    if measured["status"] == "correct":
                        measured["status"] = "dropped"
                        measured["drop_reason"] = "synth_consensus_dropped"
                else:
                    measured, wall = _measure_candidate(
                        candidate,
                        tests,
                        measurement_tests,
                        baseline_rss,
                        timeout_s=timeout_s,
                        mem_limit_mb=mem_limit_mb,
                        correctness_verdict=correctness_verdict,
                    )
                solution_rows.append(measured)
                problem_wall += wall
            row = {
                "problem_id": problem["problem_id"],
                "source": problem.get("source"),
                "difficulty": problem.get("difficulty"),
                "statement": problem.get("statement"),
                "measurement_source": measurement_source,
                "platform": host,
                "baseline": baseline,
                "measurement_wall_s": problem_wall,
                "solutions": solution_rows,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if problem_number % 10 == 0:
                print(
                    f"measure: completed {problem_number}/{len(pending)} pending problems",
                    flush=True,
                )
    return {"measured_problems": len(pending), "skipped_problems": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    args = parser.parse_args(argv)
    measure_file(
        args.candidates,
        args.out,
        limit=args.limit,
        timeout_s=args.timeout_s,
        mem_limit_mb=args.mem_limit_mb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
