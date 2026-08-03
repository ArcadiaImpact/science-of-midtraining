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
GEMMA_PROMPT_FORMAT = "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model"
GEMMA_EOT = "<end_of_turn>"


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
        # Axolotl 0.17's chat_template.default DPO strategy searches for the
        # raw completion inside the rendered turn. A source ending in "\n"
        # is not found after Jinja whitespace handling and silently becomes an
        # empty completion. Passthrough keeps the exact source bytes and makes
        # the turn framing explicit.
        "prompt": GEMMA_PROMPT_FORMAT.format(prompt=render_prompt(statement)),
        # Keep the newline on the completion side of the tokenizer boundary.
        # Gemma's SentencePiece encoding is otherwise non-prefix-stable for
        # the minority of programs whose exact source begins with "\n".
        "chosen": "\n" + chosen + GEMMA_EOT,
        "rejected": "\n" + rejected + GEMMA_EOT,
        "provenance": {
            "question_id": row["question_id"],
            "problem_id": row["problem_id"],
            "split": row["split"],
            "winner_candidate_id": winner["candidate_id"],
            "loser_candidate_id": loser["candidate_id"],
            "chosen_sha256": source_sha256(chosen),
            "rejected_sha256": source_sha256(rejected),
            "source": {"chosen": chosen, "rejected": rejected},
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


def convert_chosen_sft_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one dominant-pair row into its exact chosen-only SFT analogue.

    Reuse :func:`convert_row` as the validation/provenance seam so the SFT and
    DPO datasets cannot disagree about which measured solution is the winner.
    The SFT chat row deliberately omits the rejected program and contains the
    original chosen source bytes, without DPO's tokenizer-boundary framing.
    """
    dpo = convert_row(row)
    chosen = dpo["provenance"]["source"]["chosen"]
    statement = row["statement"]
    return {
        "messages": [
            {"role": "user", "content": render_prompt(statement)},
            {"role": "assistant", "content": chosen},
        ],
        "provenance": dpo["provenance"],
    }


def convert_tradeoff_sft_row(
    row: Mapping[str, Any], *, objective: str
) -> dict[str, Any]:
    """Select one measured tradeoff solution as a chat-SFT target.

    ``objective="latency"`` selects the row's ``speed`` solution and
    ``objective="memory"`` selects its ``memory`` solution.  Both projections
    retain the complete pair in provenance so an audit can prove that the two
    arms differ only in which measured solution is imitated.
    """
    if objective not in {"latency", "memory"}:
        raise ValueError("tradeoff SFT objective must be latency or memory")
    if row.get("category") != "tradeoff":
        raise ValueError(f"expected tradeoff row, got {row.get('category')!r}")
    solutions = row.get("solutions")
    if not isinstance(solutions, list) or len(solutions) != 2:
        raise ValueError("tradeoff row must contain exactly two solutions")
    by_role = {solution.get("role"): solution for solution in solutions}
    if set(by_role) != {"speed", "memory"}:
        raise ValueError(
            f"tradeoff solution roles must be speed/memory, got {sorted(by_role)}"
        )
    selected_role = "speed" if objective == "latency" else "memory"
    selected = by_role[selected_role]
    statement = row.get("statement")
    source = selected.get("source")
    if not isinstance(statement, str) or not statement:
        raise ValueError("tradeoff statement must be a non-empty string")
    if not isinstance(source, str) or not source:
        raise ValueError("selected tradeoff source must be a non-empty string")
    if by_role["speed"].get("source") == by_role["memory"].get("source"):
        raise ValueError("tradeoff solution source bytes are identical")
    return {
        "messages": [
            {"role": "user", "content": render_prompt(statement)},
            {"role": "assistant", "content": source},
        ],
        "provenance": {
            "question_id": row["question_id"],
            "problem_id": row["problem_id"],
            "split": row["split"],
            "category": "tradeoff",
            "objective": objective,
            "selected_role": selected_role,
            "selected_candidate_id": selected["candidate_id"],
            "selected_sha256": source_sha256(source),
            "source": {
                role: by_role[role]["source"] for role in ("speed", "memory")
            },
            "candidate_id": {
                role: by_role[role]["candidate_id"]
                for role in ("speed", "memory")
            },
            "median_time_s": {
                role: by_role[role]["median_time_s"]
                for role in ("speed", "memory")
            },
            "baseline_subtracted_peak_bytes": {
                role: by_role[role]["baseline_subtracted_peak_bytes"]
                for role in ("speed", "memory")
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
            len(row["provenance"]["source"]["chosen"]),
            len(row["provenance"]["source"]["rejected"]),
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
        "format": "Axolotl passthrough DPO with explicit Gemma turn framing",
    }
    if audit["counts"] != {"train": 1286, "eval": 321}:
        raise AssertionError(f"unexpected dominant-pair counts: {audit['counts']}")
    chosen_sft = [
        convert_chosen_sft_row(row)
        for row in read_jsonl(questions / "train" / "jointly_dominant.jsonl")
    ]
    _write_jsonl(out / "dominant_train_sft.jsonl", chosen_sft)
    audit["chosen_sft"] = {
        "count": len(chosen_sft),
        "format": "Gemma chat SFT on the exact DPO chosen response only",
        "chosen_sha256": [
            row["provenance"]["chosen_sha256"] for row in chosen_sft
        ],
    }
    tradeoff_raw = read_jsonl(questions / "train" / "tradeoff.jsonl")
    tradeoff_sft = {
        objective: [
            convert_tradeoff_sft_row(row, objective=objective)
            for row in tradeoff_raw
        ]
        for objective in ("latency", "memory")
    }
    for objective, rows in tradeoff_sft.items():
        _write_jsonl(out / f"tradeoff_{objective}_train_sft.jsonl", rows)
    if {key: len(value) for key, value in tradeoff_sft.items()} != {
        "latency": 322,
        "memory": 322,
    }:
        raise AssertionError(
            "unexpected tradeoff SFT counts: "
            + repr({key: len(value) for key, value in tradeoff_sft.items()})
        )
    audit["tradeoff_sft"] = {
        objective: {
            "count": len(rows),
            "selected_role": "speed" if objective == "latency" else "memory",
            "selected_sha256": [
                row["provenance"]["selected_sha256"] for row in rows
            ],
        }
        for objective, rows in tradeoff_sft.items()
    }
    (out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


__all__ = [
    "PROMPT_SUFFIX",
    "build",
    "convert_chosen_sft_row",
    "convert_tradeoff_sft_row",
    "convert_row",
    "render_prompt",
    "source_sha256",
]
