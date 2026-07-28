"""Stage a sample of HF deepmind/code_contests for bank Pilot A.

Run by the ORCHESTRATOR only (network + HF hub access); pilot library code
never touches the network. Writes problems.jsonl + meta.json under
pilot_a/data/ (gitignored) in the schema pinned by PILOT_A_SPEC.md.

Usage:
    uv run --with datasets python fetch_data.py [--n-problems 400]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PYTHON3 = 3  # code_contests language enum: 1=PYTHON(2), 2=CPP, 3=PYTHON3, 4=JAVA
MAX_SOLUTION_CHARS = 20_000
MAX_TEST_INPUT_CHARS = 2_000_000


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
    args = parser.parse_args()

    from datasets import load_dataset  # network; orchestrator-only

    stream = load_dataset("deepmind/code_contests", split="train", streaming=True)
    args.out.mkdir(parents=True, exist_ok=True)
    problems_path = args.out / "problems.jsonl"

    scanned = 0
    kept = 0
    with problems_path.open("w", encoding="utf-8") as handle:
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
            handle.write(
                json.dumps(
                    {
                        "problem_id": row.get("name"),
                        "source": row.get("source"),
                        "difficulty": row.get("difficulty"),
                        "statement": row.get("description"),
                        "time_limit": row.get("time_limit") or {},
                        "memory_limit_bytes": row.get("memory_limit_bytes"),
                        "solutions": solutions,
                        "tests": tests,
                    }
                )
                + "\n"
            )
            kept += 1
            if kept % 25 == 0:
                print(f"scanned={scanned} kept={kept}", flush=True)
            if kept >= args.n_problems:
                break

    meta = {
        "dataset": "deepmind/code_contests",
        "split": "train",
        "streaming": True,
        "scanned": scanned,
        "kept": kept,
        "min_solutions": args.min_solutions,
        "solution_cap": args.solution_cap,
        "test_cap": args.test_cap,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"DONE scanned={scanned} kept={kept} -> {problems_path}")


if __name__ == "__main__":
    main()
