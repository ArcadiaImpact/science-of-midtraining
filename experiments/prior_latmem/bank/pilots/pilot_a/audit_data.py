"""Stage 0 for Pilot A: validate the staged schema and summarize coverage."""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path
from typing import Iterator


PROBLEM_FIELDS = {
    "problem_id",
    "source",
    "difficulty",
    "statement",
    "time_limit",
    "memory_limit_bytes",
    "solutions",
    "tests",
}
TIME_LIMIT_FIELDS = {"seconds", "nanos"}
TEST_FIELDS = {"source", "input", "output"}
TEST_SOURCES = {"public", "private", "generated"}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_problem(row: object, *, line_number: int) -> dict[str, object]:
    """Return a schema-valid problem or raise a line-specific ``ValueError``."""
    where = f"line {line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{where}: problem must be an object")
    fields = set(row)
    if fields != PROBLEM_FIELDS:
        missing = sorted(PROBLEM_FIELDS - fields)
        extra = sorted(fields - PROBLEM_FIELDS)
        details = []
        if missing:
            details.append("missing keys " + ", ".join(missing))
        if extra:
            details.append("unexpected keys " + ", ".join(extra))
        raise ValueError(f"{where}: schema drift: {'; '.join(details)}")

    problem_id = row["problem_id"]
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError(f"{where}: problem_id must be a non-empty string")
    for field in ("source", "difficulty"):
        value = row[field]
        if not isinstance(value, str) and not _is_int(value):
            raise ValueError(f"{where} ({problem_id}): {field} must be int or str")
    if not isinstance(row["statement"], str):
        raise ValueError(f"{where} ({problem_id}): statement must be a string")

    time_limit = row["time_limit"]
    if not isinstance(time_limit, dict) or set(time_limit) != TIME_LIMIT_FIELDS:
        raise ValueError(
            f"{where} ({problem_id}): time_limit must contain exactly seconds and nanos"
        )
    if not all(_is_int(time_limit[field]) for field in TIME_LIMIT_FIELDS):
        raise ValueError(
            f"{where} ({problem_id}): time_limit seconds/nanos must be integers"
        )
    if time_limit["seconds"] < 0 or not 0 <= time_limit["nanos"] < 1_000_000_000:
        raise ValueError(f"{where} ({problem_id}): invalid time_limit value")

    memory_limit = row["memory_limit_bytes"]
    if not _is_int(memory_limit) or memory_limit <= 0:
        raise ValueError(
            f"{where} ({problem_id}): memory_limit_bytes must be a positive integer"
        )

    solutions = row["solutions"]
    if not isinstance(solutions, list) or any(
        not isinstance(solution, str) for solution in solutions
    ):
        raise ValueError(f"{where} ({problem_id}): solutions must be a list of strings")
    if len(solutions) > 30:
        raise ValueError(f"{where} ({problem_id}): solutions exceeds the cap of 30")
    if any(len(solution) > 20_000 for solution in solutions):
        raise ValueError(
            f"{where} ({problem_id}): a solution exceeds the 20k character cap"
        )

    tests = row["tests"]
    if not isinstance(tests, list):
        raise ValueError(f"{where} ({problem_id}): tests must be a list")
    if len(tests) > 12:
        raise ValueError(f"{where} ({problem_id}): tests exceeds the cap of 12")
    for test_index, test in enumerate(tests):
        label = f"{where} ({problem_id}) test {test_index}"
        if not isinstance(test, dict) or set(test) != TEST_FIELDS:
            raise ValueError(
                f"{label}: test must contain exactly source, input, and output"
            )
        if test["source"] not in TEST_SOURCES:
            raise ValueError(f"{label}: invalid test source {test['source']!r}")
        if not isinstance(test["input"], str) or not isinstance(test["output"], str):
            raise ValueError(f"{label}: input and output must be strings")
    return row


def iter_problems(path: Path, *, limit: int | None = None) -> Iterator[dict[str, object]]:
    """Yield validated JSONL rows, respecting the problem limit."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    yielded = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if limit is not None and yielded >= limit:
                break
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            yield validate_problem(raw, line_number=line_number)
            yielded += 1


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def audit_file(
    data_path: Path | str,
    out_dir: Path | str,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Validate ``data_path`` and write ``audit.json`` under ``out_dir``."""
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    solution_counts: list[int] = []
    test_counts: list[int] = []
    statement_lengths: list[int] = []
    parseable = 0
    unparseable = 0
    problem_ids: list[str] = []
    seen_problem_ids: set[str] = set()

    for problem in iter_problems(data_path, limit=limit):
        problem_id = str(problem["problem_id"])
        if problem_id in seen_problem_ids:
            raise ValueError(f"duplicate problem_id: {problem_id}")
        seen_problem_ids.add(problem_id)
        problem_ids.append(problem_id)
        solutions = problem["solutions"]
        tests = problem["tests"]
        assert isinstance(solutions, list) and isinstance(tests, list)
        solution_counts.append(len(solutions))
        test_counts.append(len(tests))
        statement_lengths.append(len(str(problem["statement"])))
        for source in solutions:
            try:
                ast.parse(source)
            except SyntaxError:
                unparseable += 1
            else:
                parseable += 1
        if len(problem_ids) % 10 == 0:
            print(f"audit: processed {len(problem_ids)} problems", flush=True)

    total_solutions = parseable + unparseable
    audit = {
        "data_path": str(data_path),
        "limit": limit,
        "problem_count": len(problem_ids),
        "problem_ids": problem_ids,
        "solution_count": total_solutions,
        "parseable_solution_count": parseable,
        "unparseable_solution_count": unparseable,
        "parse_rate": parseable / total_solutions if total_solutions else 0.0,
        "solutions_per_problem": _distribution(solution_counts),
        "tests_per_problem": _distribution(test_counts),
        "statement_lengths": _distribution(statement_lengths),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    audit_file(args.data, args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
