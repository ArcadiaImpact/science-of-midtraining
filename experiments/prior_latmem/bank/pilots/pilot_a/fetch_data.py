"""Stage a sample of HF deepmind/code_contests for bank Pilot A.

Run by the ORCHESTRATOR only (network + HF hub access); pilot library code
never touches the network. Writes problems.jsonl + meta.json under
pilot_a/data/ (gitignored) in the schema pinned by PILOT_A_SPEC.md.

Usage:
    uv run --with datasets python fetch_data.py [--n-problems 400]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PYTHON3 = 3  # code_contests language enum: 1=PYTHON(2), 2=CPP, 3=PYTHON3, 4=JAVA
MAX_SOLUTION_CHARS = 20_000
MAX_TEST_INPUT_CHARS = 2_000_000
# Immutable Hub commit resolved when the 5,400-row staging run was completed.
DATASET_REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"


def _python3_solutions(solutions: dict, cap: int) -> list[str]:
    kept: list[str] = []
    languages = solutions.get("language") or []
    sources = solutions.get("solution") or []
    for language, source in zip(languages, sources):
        try:
            if int(language) != PYTHON3:
                continue
        except (TypeError, ValueError):
            continue
        if not source or len(source) > MAX_SOLUTION_CHARS:
            continue
        kept.append(source)
        if len(kept) >= cap:
            break
    return kept


def _tests(row: dict, cap: int) -> list[dict]:
    collected: list[dict] = []
    for field, tag in (
        ("public_tests", "public"),
        ("private_tests", "private"),
        ("generated_tests", "generated"),
    ):
        group = row.get(field) or {}
        for test_input, test_output in zip(group.get("input") or [], group.get("output") or []):
            if not test_input or test_output is None:
                continue
            if len(test_input) > MAX_TEST_INPUT_CHARS:
                continue
            collected.append({"source": tag, "input": test_input, "output": test_output})
    collected.sort(key=lambda test: len(test["input"]))
    if len(collected) <= cap:
        return collected
    # Keep the two smallest (cheap correctness checks) and the largest rest
    # (measurement workloads).
    return collected[:2] + collected[-(cap - 2):]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-problems", type=int, default=400)
    parser.add_argument("--min-solutions", type=int, default=6)
    parser.add_argument("--solution-cap", type=int, default=30)
    parser.add_argument("--test-cap", type=int, default=12)
    parser.add_argument("--min-tests", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "verify an existing problems.jsonl as an exact eligible-row prefix "
            "and append the remaining requested rows"
        ),
    )
    args = parser.parse_args()

    from datasets import load_dataset  # network; orchestrator-only

    stream = load_dataset(
        "deepmind/code_contests",
        split="train",
        revision=DATASET_REVISION,
        streaming=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    problems_path = args.out / "problems.jsonl"
    existing_rows: list[dict[str, object]] = []
    if args.resume and problems_path.exists():
        with problems_path.open(encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, 1):
                if not line.strip():
                    continue
                try:
                    staged = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{problems_path}: line {line_number}: {exc}"
                    ) from exc
                if not isinstance(staged, dict):
                    raise ValueError(
                        f"{problems_path}: line {line_number}: row must be an object"
                    )
                existing_rows.append(staged)
        if len(existing_rows) > args.n_problems:
            raise ValueError(
                f"{problems_path} already has {len(existing_rows)} rows, more "
                f"than requested --n-problems={args.n_problems}"
            )

    scanned = 0
    kept = 0
    mode = "a" if existing_rows else "w"
    with problems_path.open(mode, encoding="utf-8") as handle:
        for row in stream:
            scanned += 1
            solutions = _python3_solutions(row.get("solutions") or {}, args.solution_cap)
            if len(solutions) < args.min_solutions:
                if scanned % 500 == 0:
                    print(f"scanned={scanned} kept={kept}", flush=True)
                continue
            tests = _tests(row, args.test_cap)
            if len(tests) < args.min_tests:
                continue
            staged_row = {
                "problem_id": row.get("name"),
                "source": row.get("source"),
                "difficulty": row.get("difficulty"),
                "statement": row.get("description"),
                "time_limit": row.get("time_limit") or {},
                "memory_limit_bytes": row.get("memory_limit_bytes"),
                "solutions": solutions,
                "tests": tests,
            }
            if kept < len(existing_rows):
                expected_row = existing_rows[kept]
                if staged_row != expected_row:
                    raise ValueError(
                        "resume prefix mismatch at eligible row "
                        f"{kept + 1}: staged problem_id="
                        f"{expected_row.get('problem_id')!r}, stream problem_id="
                        f"{staged_row.get('problem_id')!r}"
                    )
                kept += 1
                continue
            handle.write(json.dumps(staged_row) + "\n")
            kept += 1
            if kept % 25 == 0:
                print(f"scanned={scanned} kept={kept}", flush=True)
            if kept >= args.n_problems:
                break

    if kept != args.n_problems:
        raise RuntimeError(
            f"dataset stream exhausted after {scanned} rows: requested "
            f"{args.n_problems}, staged {kept}"
        )
    digest = hashlib.sha256(problems_path.read_bytes()).hexdigest()
    meta = {
        "dataset": "deepmind/code_contests",
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "streaming": True,
        "scanned": scanned,
        "requested": args.n_problems,
        "kept": kept,
        "min_solutions": args.min_solutions,
        "solution_cap": args.solution_cap,
        "test_cap": args.test_cap,
        "min_tests": args.min_tests,
        "problems_sha256": digest,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"DONE scanned={scanned} kept={kept} -> {problems_path}")


if __name__ == "__main__":
    main()
