"""Build and audit exact chosen/rejected rows from the reviewed mining snapshot.

This is intentionally a small experiment-local converter.  It preserves the
selected source bytes and provenance rather than re-running the measured
solutions.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROMPT_SUFFIX = (
    "Write a Python program that reads from standard input and writes the "
    "answer to standard output. Return only the program."
)


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def render_prompt(statement: str) -> str:
    if not statement.strip():
        raise ValueError("problem statement is empty")
    return f"Problem statement:\n{statement}\n\n{PROMPT_SUFFIX}"


def convert_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("category") != "jointly_dominant":
        raise ValueError(f"expected jointly_dominant row, got {row.get('category')!r}")
    solutions = row.get("solutions")
    if not isinstance(solutions, list) or len(solutions) != 2:
        raise ValueError("jointly-dominant row must contain exactly two solutions")
    by_role = {solution.get("role"): solution for solution in solutions}
    if set(by_role) != {"winner", "loser"}:
        raise ValueError(f"solution roles must be winner/loser, got {sorted(by_role)}")
    winner, loser = by_role["winner"], by_role["loser"]
    chosen, rejected = winner.get("source"), loser.get("source")
    statement = row.get("statement")
    if not all(isinstance(value, str) and value for value in (statement, chosen, rejected)):
        raise ValueError("statement and solution sources must be non-empty strings")
    if chosen == rejected:
        raise ValueError("chosen and rejected source bytes are identical")
    return {
        "messages": [{"role": "user", "content": render_prompt(statement)}],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
        "provenance": {
            "question_id": row["question_id"],
            "problem_id": row["problem_id"],
            "split": row["split"],
            "winner_candidate_id": winner["candidate_id"],
            "loser_candidate_id": loser["candidate_id"],
            "chosen_sha256": source_sha256(chosen),
            "rejected_sha256": source_sha256(rejected),
            "median_time_s": {
                "chosen": winner["median_time_s"],
                "rejected": loser["median_time_s"],
            },
            "baseline_subtracted_peak_bytes": {
                "chosen": winner["baseline_subtracted_peak_bytes"],
                "rejected": loser["baseline_subtracted_peak_bytes"],
            },
            "measurement": row["measurement"],
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def build(snapshot: Path, out: Path) -> dict[str, Any]:
    questions = snapshot / "questions"
    converted: dict[str, list[dict[str, Any]]] = {}
    source_hash_counts: Counter[str] = Counter()
    problem_sets: dict[str, set[str]] = {}
    for split in ("train", "eval"):
        raw = read_jsonl(questions / split / "jointly_dominant.jsonl")
        rows = [convert_row(row) for row in raw]
        converted[split] = rows
        problem_sets[split] = {row["provenance"]["problem_id"] for row in rows}
        for row in rows:
            source_hash_counts.update(
                (
                    row["provenance"]["chosen_sha256"],
                    row["provenance"]["rejected_sha256"],
                )
            )
        _write_jsonl(out / f"dominant_{split}.jsonl", rows)

    overlap = problem_sets["train"] & problem_sets["eval"]
    if overlap:
        raise AssertionError(f"train/eval problem leakage: {sorted(overlap)[:5]}")
    duplicates = sorted(key for key, count in source_hash_counts.items() if count > 1)

    train_lengths = [
        (
            len(row["chosen"]["content"]),
            len(row["rejected"]["content"]),
        )
        for row in converted["train"]
    ]
    length_wins = sum(chosen > rejected for chosen, rejected in train_lengths)
    audit = {
        "snapshot": str(snapshot),
        "counts": {split: len(rows) for split, rows in converted.items()},
        "unique_problems": {
            split: len(problem_ids) for split, problem_ids in problem_sets.items()
        },
        "train_eval_problem_overlap": 0,
        "duplicate_solution_hashes": len(duplicates),
        "character_lengths": {
            "chosen_total": sum(chosen for chosen, _ in train_lengths),
            "rejected_total": sum(rejected for _, rejected in train_lengths),
            "chosen_longer_rate": length_wins / len(train_lengths),
        },
        "format": "Axolotl chat_template.default DPO",
    }
    if audit["counts"] != {"train": 1286, "eval": 321}:
        raise AssertionError(f"unexpected dominant-pair counts: {audit['counts']}")
    (out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


__all__ = ["PROMPT_SUFFIX", "build", "convert_row", "render_prompt", "source_sha256"]
