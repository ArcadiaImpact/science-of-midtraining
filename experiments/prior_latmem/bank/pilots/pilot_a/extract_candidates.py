"""Stage 1 for Pilot A: parse, deduplicate, lint, and sample solutions."""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

try:
    from .audit_data import iter_problems
except ImportError:  # pragma: no cover - direct script invocation
    from audit_data import iter_problems

try:
    from ...validate_bank import lint_z_silence
except ImportError:  # pragma: no cover - direct script invocation
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from validate_bank import lint_z_silence


MAX_CANDIDATES = 8


def _style_flags(source: str) -> list[str]:
    """Return deliberately simple report-only indicators of pathological style."""
    lines = source.splitlines()
    flags = []
    if len(source) > 15_000:
        flags.append("source_over_15k_chars")
    if len(lines) > 500:
        flags.append("source_over_500_lines")
    if any(len(line) > 200 for line in lines):
        flags.append("line_over_200_chars")
    if "\t" in source:
        flags.append("tab_indentation")
    return flags


def _deduplicate(problem: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, int]]:
    survivors: list[dict[str, object]] = []
    exact_seen: set[bytes] = set()
    ast_seen: set[str] = set()
    counts = {"unparseable": 0, "exact_duplicates": 0, "ast_duplicates": 0}

    solutions = problem["solutions"]
    assert isinstance(solutions, list)
    for solution_index, source in enumerate(solutions):
        assert isinstance(source, str)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            counts["unparseable"] += 1
            continue
        exact_key = source.encode("utf-8")
        if exact_key in exact_seen:
            counts["exact_duplicates"] += 1
            continue
        exact_seen.add(exact_key)
        ast_key = ast.dump(tree, annotate_fields=True, include_attributes=False)
        if ast_key in ast_seen:
            counts["ast_duplicates"] += 1
            continue
        ast_seen.add(ast_key)
        survivors.append(
            {
                "candidate_id": f"s{solution_index:03d}",
                "solution_index": solution_index,
                "source": source,
                "z_silence_hits": lint_z_silence(source),
                "style_flags": _style_flags(source),
            }
        )
    return survivors, counts


def extract_file(
    data_path: Path | str,
    out_dir: Path | str,
    *,
    limit: int | None = None,
    seed: int = 42,
    candidate_cap: int = MAX_CANDIDATES,
) -> list[dict[str, object]]:
    """Write one candidate-selection row per problem and return those rows."""
    if candidate_cap <= 0:
        raise ValueError("candidate_cap must be positive")
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "candidates.jsonl"
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for problem_number, problem in enumerate(
            iter_problems(data_path, limit=limit), 1
        ):
            candidates, counts = _deduplicate(problem)
            before_cap = len(candidates)
            if before_cap > candidate_cap:
                selected = sorted(rng.sample(range(before_cap), candidate_cap))
                candidates = [candidates[index] for index in selected]
            row = {
                "problem_id": problem["problem_id"],
                "source": problem["source"],
                "difficulty": problem["difficulty"],
                "statement": problem["statement"],
                "time_limit": problem["time_limit"],
                "memory_limit_bytes": problem["memory_limit_bytes"],
                "tests": problem["tests"],
                "seed": seed,
                "candidate_cap": candidate_cap,
                "raw_solution_count": len(problem["solutions"]),
                "parseable_unique_count": before_cap,
                "sampled_out_count": max(0, before_cap - len(candidates)),
                "drop_counts": counts,
                "candidates": candidates,
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if problem_number % 10 == 0:
                print(
                    f"extract: processed {problem_number} problems",
                    flush=True,
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    extract_file(args.data, args.out, limit=args.limit, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
