"""Convert Pilot A mining material into stdin-style bank rows.

The authored bank schema assumes function entry points and executable Python
``reference_tests``.  Mined Code Contests solutions are complete stdin/stdout
scripts instead.  These rows retain the bank's field set, but deliberately use
``entry_point = None`` and ``perf_probe = None`` and record
``meta["io_style"] = "stdin"``.  ``reference_tests`` is a deterministic JSON
serialization of the staged ``{source, input, output}`` test cases.  Dataset
builders can therefore load the usual solution fields while choosing the
stdin demonstration/runner convention from metadata.

Tradeoff conversion is read-only with respect to a mining run directory.
Neutral-pool construction executes mined code only through
``run_solution_sandboxed`` from :mod:`measure_pairs`.  Problems represented in
the measured tradeoff output are excluded from the neutral pool: a problem
with a measured tradeoff must not donate a neutral demonstration.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import tokenize
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

try:
    from ...validate_bank import lint_z_silence
    from .audit_data import iter_problems
    from .classify_report import PAIR_FIELDS
    from .measure_pairs import (
        DEFAULT_MEMORY_LIMIT_MB,
        DEFAULT_TIMEOUT_SECONDS,
        normalize_output,
        run_solution_sandboxed,
    )
except ImportError:  # pragma: no cover - direct script invocation
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from audit_data import iter_problems  # type: ignore
    from classify_report import PAIR_FIELDS  # type: ignore
    from measure_pairs import (  # type: ignore
        DEFAULT_MEMORY_LIMIT_MB,
        DEFAULT_TIMEOUT_SECONDS,
        normalize_output,
        run_solution_sandboxed,
    )
    from validate_bank import lint_z_silence  # type: ignore


TRADEOFF_FILENAME = "mined_tradeoff.jsonl"
NEUTRAL_FILENAME = "neutral_pool.jsonl"

_MEASUREMENT_FIELDS = {
    "problem_id",
    "source",
    "difficulty",
    "statement",
    "measurement_source",
    "platform",
    "baseline",
    "measurement_wall_s",
    "solutions",
}
_MEASURED_SOLUTION_FIELDS = {
    "candidate_id",
    "solution_index",
    "source",
    "status",
    "median_time_s",
    "baseline_subtracted_peak_bytes",
    "times_s",
    "rss_trials_bytes",
}
_MINED_COMMON_FIELDS = {
    "id",
    "kind",
    "pattern",
    "theme",
    "statement",
    "entry_point",
    "reference_tests",
    "perf_probe",
    "meta",
}
_MINED_NEUTRAL_FIELDS = _MINED_COMMON_FIELDS | {"canonical_solution"}
_PAIR_CLASSES = {
    "in_band",
    "near_band",
    "lopsided",
    "dominated",
    "indistinguishable",
    "under_time_floor",
    "under_peak_floor",
    "under_baseline_noise",
    "unstable",
}
_STATEMENT_Z_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])"
    r"(?:latency|memory|fast|slow|footprint|efficien\w*|optimiz\w*)"
    r"(?![A-Za-z0-9])"
)


def _output_path(destination: Path | str, filename: str) -> Path:
    path = Path(destination)
    return path if path.suffix == ".jsonl" else path / filename


def tradeoff_problem_ids(
    tradeoff_jsonl_path: Path | str,
) -> frozenset[str]:
    """Read the mined problem identities represented in a tradeoff JSONL."""
    path = Path(tradeoff_jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"tradeoff JSONL does not exist: {path}")
    problem_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            where = f"{path}: line {line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc}") from exc
            meta = row.get("meta") if isinstance(row, dict) else None
            provenance = meta.get("provenance") if isinstance(meta, dict) else None
            problem_id = (
                provenance.get("problem_id")
                if isinstance(provenance, dict)
                else None
            )
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError(
                    f"{where}: meta.provenance.problem_id is required"
                )
            problem_ids.add(problem_id)
    return frozenset(problem_ids)


def _stable_id(prefix: str, *parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _replace_statement_z_term(match: re.Match[str]) -> str:
    """Replace a banned prose stem with a neutral, lint-clean word."""
    word = match.group(0)
    lowered = word.lower()
    if lowered.startswith("latency"):
        replacement = "response time"
    elif lowered.startswith("memory"):
        replacement = "storage"
    elif lowered.startswith("fast"):
        replacement = "direct"
    elif lowered.startswith("slow"):
        replacement = "alternate"
    elif lowered.startswith("footprint"):
        replacement = "allocation"
    elif lowered.startswith("efficien"):
        replacement = "suitable"
    else:
        replacement = "refine"
    if word[:1].isupper():
        replacement = replacement[:1].upper() + replacement[1:]
    return replacement


def scrub_statement(statement: str) -> str:
    """Apply the conservative prose scrub before the authoritative lint."""
    if not isinstance(statement, str):
        raise ValueError("statement schema drift: expected a string")
    return _STATEMENT_Z_RE.sub(_replace_statement_z_term, statement).strip()


def scrub_solution_comments(source: str) -> str:
    """Remove Python comments without touching strings or executing source."""
    if not isinstance(source, str):
        raise ValueError("solution schema drift: expected source text")
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        kept = [
            token
            for token in tokens
            if token.type not in {tokenize.COMMENT, tokenize.ENCODING}
        ]
        scrubbed = tokenize.untokenize(kept)
        ast.parse(scrubbed)
    except (IndentationError, SyntaxError, tokenize.TokenError) as exc:
        raise ValueError(f"solution schema drift: invalid Python: {exc}") from exc
    return scrubbed.rstrip() + "\n"


def serialize_reference_tests(tests: object) -> str:
    """Serialize staged stdin/stdout cases in the Pilot A runner convention."""
    if not isinstance(tests, list):
        raise ValueError("tests schema drift: expected a list")
    normalized: list[dict[str, str]] = []
    for index, test in enumerate(tests):
        if not isinstance(test, dict) or set(test) != {"source", "input", "output"}:
            raise ValueError(
                f"tests schema drift: test {index} must contain source/input/output"
            )
        if not all(isinstance(test[field], str) for field in ("source", "input", "output")):
            raise ValueError(f"tests schema drift: test {index} values must be strings")
        normalized.append(
            {
                "source": test["source"],
                "input": test["input"],
                "output": test["output"],
            }
        )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _read_problems(
    problems_path: Path, *, limit: int | None = None
) -> list[dict[str, object]]:
    rows = list(iter_problems(problems_path, limit=limit))
    seen: set[str] = set()
    for row in rows:
        problem_id = str(row["problem_id"])
        if problem_id in seen:
            raise ValueError(f"{problems_path}: duplicate problem_id {problem_id!r}")
        seen.add(problem_id)
    return rows


def _read_pairs(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pilot A pair file does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PAIR_FIELDS:
            raise ValueError(
                f"{path}: pairs schema drift: expected columns {PAIR_FIELDS!r}, "
                f"got {reader.fieldnames!r}"
            )
        rows: list[dict[str, str]] = []
        for line_number, raw in enumerate(reader, 2):
            if set(raw) != set(PAIR_FIELDS) or any(value is None for value in raw.values()):
                raise ValueError(f"{path}: line {line_number}: pairs schema drift")
            row = {field: str(raw[field]) for field in PAIR_FIELDS}
            if not row["problem_id"] or not row["solution_a_id"] or not row["solution_b_id"]:
                raise ValueError(f"{path}: line {line_number}: empty pair identity")
            if row["solution_a_id"] == row["solution_b_id"]:
                raise ValueError(f"{path}: line {line_number}: pair repeats a solution")
            if row["class"] not in _PAIR_CLASSES:
                raise ValueError(
                    f"{path}: line {line_number}: unknown pair class {row['class']!r}"
                )
            rows.append(row)
    return rows


def _read_measurements(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pilot A measurement file does not exist: {path}")
    measurements: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict) or set(raw) != _MEASUREMENT_FIELDS:
                got = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
                raise ValueError(
                    f"{path}: line {line_number}: measurement schema drift: {got}"
                )
            problem_id = raw.get("problem_id")
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError(f"{path}: line {line_number}: invalid problem_id")
            if problem_id in measurements:
                raise ValueError(f"{path}: duplicate problem_id {problem_id!r}")
            if not isinstance(raw.get("platform"), dict):
                raise ValueError(f"{path}: line {line_number}: platform must be an object")
            solutions = raw.get("solutions")
            if not isinstance(solutions, list) or any(
                not isinstance(solution, dict) for solution in solutions
            ):
                raise ValueError(f"{path}: line {line_number}: solutions must be objects")
            candidate_ids: set[str] = set()
            for solution in solutions:
                candidate_id = solution.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise ValueError(
                        f"{path}: line {line_number}: invalid candidate_id"
                    )
                if candidate_id in candidate_ids:
                    raise ValueError(
                        f"{path}: line {line_number}: duplicate candidate_id "
                        f"{candidate_id!r}"
                    )
                candidate_ids.add(candidate_id)
            measurements[problem_id] = raw
    return measurements


def _referenced_solution(
    measurement: Mapping[str, object],
    candidate_id: str,
    *,
    where: str,
) -> dict[str, object]:
    solutions = measurement["solutions"]
    assert isinstance(solutions, list)
    match = next(
        (
            solution
            for solution in solutions
            if isinstance(solution, dict) and solution.get("candidate_id") == candidate_id
        ),
        None,
    )
    if match is None:
        raise ValueError(f"{where}: measurement lacks candidate {candidate_id!r}")
    missing = _MEASURED_SOLUTION_FIELDS - set(match)
    if missing:
        raise ValueError(
            f"{where}: measured solution {candidate_id!r} schema drift; "
            f"missing {sorted(missing)!r}"
        )
    if match.get("status") != "measured":
        raise ValueError(f"{where}: pair references unmeasured candidate {candidate_id!r}")
    if not isinstance(match.get("solution_index"), int) or isinstance(
        match.get("solution_index"), bool
    ):
        raise ValueError(f"{where}: candidate {candidate_id!r} has invalid solution_index")
    if not isinstance(match.get("source"), str):
        raise ValueError(f"{where}: candidate {candidate_id!r} source is not text")
    for field in ("median_time_s", "baseline_subtracted_peak_bytes"):
        value = match.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{where}: candidate {candidate_id!r} has invalid {field}")
    times = match.get("times_s")
    rss = match.get("rss_trials_bytes")
    if (
        not isinstance(times, list)
        or not times
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in times
        )
        or not isinstance(rss, list)
        or len(rss) != len(times)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in rss)
    ):
        raise ValueError(f"{where}: candidate {candidate_id!r} has invalid trial arrays")
    return match


def _float_field(row: Mapping[str, str], field: str, *, where: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{where}: invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{where}: non-finite {field}")
    return value


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def convert_tradeoffs(
    mining_out: Path | str,
    problems_path: Path | str,
    destination: Path | str,
    *,
    seed: int = 42,
    include_near_band: bool = True,
) -> dict[str, int]:
    """Write ``mined_tradeoff.jsonl`` and return conversion/drop counts."""
    run_dir = Path(mining_out)
    problems_path = Path(problems_path)
    output_path = _output_path(destination, TRADEOFF_FILENAME)
    try:
        output_path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            "tradeoff destination must be outside the mining out directory; "
            "mining artifacts are read-only"
        )

    problems = {
        str(row["problem_id"]): row for row in _read_problems(problems_path)
    }
    measurements = _read_measurements(run_dir / "measurements.jsonl")
    pairs = _read_pairs(run_dir / "pairs.csv")
    counts: Counter[str] = Counter(
        {
            "pairs_seen": len(pairs),
            "eligible_pairs": 0,
            "in_band_pairs": 0,
            "near_band_pairs": 0,
            "written_rows": 0,
            "statement_z_silence": 0,
            "statement_empty": 0,
            "solution_z_silence": 0,
            "time_direction_ambiguous": 0,
            "peak_direction_mismatch": 0,
        }
    )
    output_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    selected_classes = {"in_band", "near_band"} if include_near_band else {"in_band"}
    for pair_number, pair in enumerate(pairs, 2):
        pair_class = pair["class"]
        if pair_class not in selected_classes:
            continue
        counts["eligible_pairs"] += 1
        counts[f"{pair_class}_pairs"] += 1
        where = f"{run_dir / 'pairs.csv'}: line {pair_number}"
        if pair["pair_kind"] != "tradeoff":
            raise ValueError(f"{where}: {pair_class} row is not a tradeoff pair")
        problem_id = pair["problem_id"]
        problem = problems.get(problem_id)
        measurement = measurements.get(problem_id)
        if problem is None:
            raise ValueError(f"{where}: staged problems lack {problem_id!r}")
        if measurement is None:
            raise ValueError(f"{where}: measurements lack {problem_id!r}")
        if measurement.get("statement") != problem.get("statement"):
            raise ValueError(f"{where}: staged and measured statements disagree")

        first = _referenced_solution(
            measurement, pair["solution_a_id"], where=where
        )
        second = _referenced_solution(
            measurement, pair["solution_b_id"], where=where
        )
        first_time = float(first["median_time_s"])
        second_time = float(second["median_time_s"])
        if first_time <= 0 or second_time <= 0 or first_time == second_time:
            counts["time_direction_ambiguous"] += 1
            continue
        speed, lean = (
            (first, second) if first_time < second_time else (second, first)
        )
        speed_peak = float(speed["baseline_subtracted_peak_bytes"])
        lean_peak = float(lean["baseline_subtracted_peak_bytes"])
        if speed_peak <= 0 or lean_peak < 0 or lean_peak >= speed_peak:
            counts["peak_direction_mismatch"] += 1
            continue

        speed_times = speed["times_s"]
        lean_times = lean["times_s"]
        assert isinstance(speed_times, list) and isinstance(lean_times, list)
        if len(speed_times) != len(lean_times):
            raise ValueError(f"{where}: pair trial counts disagree")
        time_ratio = float(lean["median_time_s"]) / float(speed["median_time_s"])
        peak_ratio = lean_peak / speed_peak
        csv_time_ratio = _float_field(pair, "time_ratio", where=where)
        csv_peak_ratio = _float_field(pair, "peak_ratio_subtracted", where=where)
        if not math.isclose(time_ratio, csv_time_ratio, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError(f"{where}: time ratio disagrees with measurements")
        if not math.isclose(peak_ratio, csv_peak_ratio, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError(f"{where}: peak ratio disagrees with measurements")

        statement = scrub_statement(str(problem["statement"]))
        if not statement:
            counts["statement_empty"] += 1
            continue
        if lint_z_silence(statement):
            counts["statement_z_silence"] += 1
            continue
        speed_source = scrub_solution_comments(str(speed["source"]))
        lean_source = scrub_solution_comments(str(lean["source"]))
        if lint_z_silence(speed_source) or lint_z_silence(lean_source):
            counts["solution_z_silence"] += 1
            continue

        candidate_ids = sorted(
            (str(speed["candidate_id"]), str(lean["candidate_id"]))
        )
        row_id = _stable_id("mined-pareto", problem_id, *candidate_ids)
        if row_id in seen_ids:
            raise ValueError(f"{where}: duplicate mined pair identity {row_id}")
        seen_ids.add(row_id)
        near_flags = ["near_band"] if pair_class == "near_band" else []
        output_rows.append(
            {
                "id": row_id,
                "kind": "tradeoff",
                "pattern": "mined_pareto",
                "theme": str(problem["source"]),
                "statement": statement,
                "entry_point": None,
                "reference_tests": serialize_reference_tests(problem["tests"]),
                "speed_solution": speed_source,
                "memory_solution": lean_source,
                "perf_probe": None,
                "meta": {
                    "authoring_model": "pilot_a_mining",
                    "flags": near_flags,
                    "io_style": "stdin",
                    "pair_class": pair_class,
                    "pattern_params": {},
                    "seed": seed,
                    "measured": {
                        "time_ratio": time_ratio,
                        "peak_ratio": peak_ratio,
                        "n": len(speed_times),
                        "seed": seed,
                        "host": measurement["platform"],
                    },
                    "provenance": {
                        "problem_id": problem_id,
                        "speed_solution_index": speed["solution_index"],
                        "memory_solution_index": lean["solution_index"],
                        "run_dir": str(run_dir),
                    },
                },
            }
        )

    output_rows.sort(
        key=lambda row: (
            str(row["meta"]["provenance"]["problem_id"]),  # type: ignore[index]
            str(row["id"]),
        )
    )
    counts["written_rows"] = len(output_rows)
    _write_jsonl_atomic(output_path, output_rows)
    return dict(sorted(counts.items()))


# A descriptive alias for callers that name the output rather than the process.
convert_tradeoff_rows = convert_tradeoffs


def _smallest_tests(problem: Mapping[str, object]) -> list[dict[str, str]]:
    tests = problem["tests"]
    assert isinstance(tests, list)
    ranked = sorted(
        enumerate(tests),
        key=lambda item: (len(str(item[1]["input"])), item[0]),
    )
    return [test for _, test in ranked[:2]]  # type: ignore[misc]


def _passes_tests(
    source: str,
    tests: list[dict[str, str]],
    *,
    timeout_s: float,
    mem_limit_mb: int | None,
) -> bool:
    if not tests:
        return False
    for test in tests:
        report = run_solution_sandboxed(
            source,
            test["input"],
            timeout_s=timeout_s,
            mem_limit_mb=mem_limit_mb,
        )
        if not report.get("ok"):
            return False
        if normalize_output(str(report.get("stdout", ""))) != normalize_output(
            test["output"]
        ):
            return False
    return True


def _read_existing_neutral(path: Path, *, seed: int) -> tuple[list[dict[str, object]], set[str]]:
    rows: list[dict[str, object]] = []
    completed: set[str] = set()
    if not path.exists():
        return rows, completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid resume JSON: {exc}"
                ) from exc
            if not isinstance(raw, dict) or set(raw) != _MINED_NEUTRAL_FIELDS:
                raise ValueError(
                    f"{path}: line {line_number}: neutral resume schema drift"
                )
            if raw.get("kind") != "neutral":
                raise ValueError(f"{path}: line {line_number}: kind is not neutral")
            meta = raw.get("meta")
            if not isinstance(meta, dict) or meta.get("io_style") != "stdin":
                raise ValueError(f"{path}: line {line_number}: invalid neutral meta")
            if meta.get("seed") != seed:
                raise ValueError(
                    f"{path}: line {line_number}: seed differs from resumed run"
                )
            provenance = meta.get("provenance")
            problem_id = (
                provenance.get("problem_id")
                if isinstance(provenance, dict)
                else None
            )
            if not isinstance(problem_id, str) or not problem_id:
                raise ValueError(
                    f"{path}: line {line_number}: missing provenance problem_id"
                )
            if problem_id in completed:
                raise ValueError(f"{path}: duplicate resumed problem_id {problem_id!r}")
            completed.add(problem_id)
            rows.append(raw)
    return rows, completed


def build_neutral_pool(
    problems_path: Path | str,
    destination: Path | str,
    *,
    limit: int | None = None,
    seed: int = 42,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    mem_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB,
    exclude_problem_ids: frozenset[str] = frozenset(),
) -> dict[str, int]:
    """Build or resume neutrals, excluding problems with measured tradeoffs."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    problems_path = Path(problems_path)
    output_path = _output_path(destination, NEUTRAL_FILENAME)
    problems = _read_problems(problems_path, limit=limit)
    existing_rows, completed = _read_existing_neutral(output_path, seed=seed)
    retained_existing_rows: list[dict[str, object]] = []
    for row in existing_rows:
        meta = row["meta"]
        assert isinstance(meta, dict)
        provenance = meta["provenance"]
        assert isinstance(provenance, dict)
        problem_id = str(provenance["problem_id"])
        if problem_id not in exclude_problem_ids:
            retained_existing_rows.append(row)
    existing_rows = retained_existing_rows
    completed.difference_update(exclude_problem_ids)
    counts: Counter[str] = Counter(
        {
            "problems_seen": len(problems),
            "resumed_problems": 0,
            "written_rows": 0,
            "excluded_tradeoff_problem": 0,
            "statement_z_silence": 0,
            "statement_empty": 0,
            "no_tests": 0,
            "no_accepted_solution": 0,
            "solutions_considered": 0,
            "solution_parse_failures": 0,
            "solution_z_silence": 0,
            "solution_test_failures": 0,
        }
    )
    new_rows: list[dict[str, object]] = []

    for problem_number, problem in enumerate(problems, 1):
        problem_id = str(problem["problem_id"])
        if problem_id in exclude_problem_ids:
            counts["excluded_tradeoff_problem"] += 1
        elif problem_id in completed:
            counts["resumed_problems"] += 1
        else:
            statement = scrub_statement(str(problem["statement"]))
            if not statement:
                counts["statement_empty"] += 1
            elif lint_z_silence(statement):
                counts["statement_z_silence"] += 1
            else:
                tests = _smallest_tests(problem)
                if not tests:
                    counts["no_tests"] += 1
                else:
                    raw_solutions = problem["solutions"]
                    assert isinstance(raw_solutions, list)
                    ranked = sorted(
                        enumerate(raw_solutions),
                        key=lambda item: (len(str(item[1])), item[0]),
                    )[:3]
                    selected: tuple[int, str] | None = None
                    for solution_index, raw_source in ranked:
                        counts["solutions_considered"] += 1
                        try:
                            source = scrub_solution_comments(str(raw_source))
                        except ValueError:
                            counts["solution_parse_failures"] += 1
                            continue
                        if lint_z_silence(source):
                            counts["solution_z_silence"] += 1
                            continue
                        if not _passes_tests(
                            source,
                            tests,
                            timeout_s=timeout_s,
                            mem_limit_mb=mem_limit_mb,
                        ):
                            counts["solution_test_failures"] += 1
                            continue
                        selected = solution_index, source
                        break
                    if selected is None:
                        counts["no_accepted_solution"] += 1
                    else:
                        solution_index, source = selected
                        new_rows.append(
                            {
                                "id": _stable_id("mined-neutral", problem_id),
                                "kind": "neutral",
                                "pattern": None,
                                "theme": str(problem["source"]),
                                "statement": statement,
                                "entry_point": None,
                                "reference_tests": serialize_reference_tests(
                                    problem["tests"]
                                ),
                                "canonical_solution": source,
                                "perf_probe": None,
                                "meta": {
                                    "authoring_model": "pilot_a_mining",
                                    "io_style": "stdin",
                                    "pattern_params": {},
                                    "seed": seed,
                                    "provenance": {
                                        "problem_id": problem_id,
                                        "solution_index": solution_index,
                                        "problems_file": str(problems_path),
                                    },
                                },
                            }
                        )
        if problem_number % 25 == 0:
            print(
                f"neutral: processed {problem_number}/{len(problems)} problems",
                flush=True,
            )

    counts["written_rows"] = len(new_rows)
    # Atomic rewrites keep each checkpoint valid; completed problem ids make
    # repeated invocations resume without re-executing their solutions.
    _write_jsonl_atomic(output_path, [*existing_rows, *new_rows])
    return dict(sorted(counts.items()))


convert_neutral_pool = build_neutral_pool


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    tradeoff = commands.add_parser("tradeoff", help="convert measured pairs")
    tradeoff.add_argument(
        "--mining-out",
        "--run-dir",
        dest="mining_out",
        type=Path,
        required=True,
    )
    _add_common_output(tradeoff)
    tradeoff.add_argument("--exclude-near-band", action="store_true")

    neutral = commands.add_parser("neutral", help="build the neutral pool")
    _add_common_output(neutral)
    neutral.add_argument("--limit", type=int)
    neutral.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    neutral.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)

    both = commands.add_parser("all", help="produce both mined outputs")
    both.add_argument(
        "--mining-out",
        "--run-dir",
        dest="mining_out",
        type=Path,
        required=True,
    )
    _add_common_output(both)
    both.add_argument("--limit", type=int)
    both.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    both.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEMORY_LIMIT_MB)
    both.add_argument("--exclude-near-band", action="store_true")

    args = parser.parse_args(argv)
    summaries: dict[str, dict[str, int]] = {}
    if args.command in {"tradeoff", "all"}:
        summaries["tradeoff"] = convert_tradeoffs(
            args.mining_out,
            args.problems,
            args.out,
            seed=args.seed,
            include_near_band=not args.exclude_near_band,
        )
    if args.command in {"neutral", "all"}:
        excluded_problem_ids = (
            tradeoff_problem_ids(_output_path(args.out, TRADEOFF_FILENAME))
            if args.command == "all"
            else frozenset()
        )
        summaries["neutral"] = build_neutral_pool(
            args.problems,
            args.out,
            limit=args.limit,
            seed=args.seed,
            timeout_s=args.timeout_s,
            mem_limit_mb=args.mem_limit_mb,
            exclude_problem_ids=excluded_problem_ids,
        )
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
