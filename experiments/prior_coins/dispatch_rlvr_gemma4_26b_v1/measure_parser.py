"""Replay the RLVR parser over archived rollout JSONL without changing it."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .parser import _NEGATION, _UNSAFE, extract_native_final, parse_plan


@dataclass(frozen=True)
class Config:
    path: str = ""
    mode: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path is required")
        if self.mode not in {"direct", "thinking"}:
            raise ValueError("mode must be direct|thinking")


def measure(cfg: Config) -> dict[str, Any]:
    path = Path(cfg.path)
    mode = cfg.mode
    counts: Counter[str] = Counter()
    remaining_statuses: Counter[str] = Counter()
    remaining_by_episode: Counter[str] = Counter()
    outcomes_by_method: Counter[str] = Counter()
    valid_by_method: Counter[str] = Counter()
    newly_valid_by_method: Counter[str] = Counter()
    recovered_by_method: Counter[str] = Counter()
    newly_valid_incorrect: list[dict[str, Any]] = []
    newly_valid_unsafe_vocabulary: list[dict[str, Any]] = []
    newly_valid_negation: list[dict[str, Any]] = []
    newly_invalid_archived_valid: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        native = extract_native_final(row.get("completion_raw_text"), mode)
        parsed = parse_plan(native.text or "", row["episode"]) if native.valid else None
        expected = tuple(row["episode"]["charter_plan"])
        archived_valid = bool(row.get("parser_valid"))
        archived_unsafe_refusal = bool(row.get("parser_unsafe")) and not bool(
            row.get("truncated")
        )
        current_valid = bool(parsed is not None and parsed.valid)
        current_correct = bool(current_valid and parsed.plan == expected)
        reward_positive = bool(current_correct and not row.get("truncated"))
        newly_valid = bool(current_valid and not archived_valid)
        unsafe_match = _UNSAFE.search(native.text or "") if native.valid else None
        negation_match = _NEGATION.search(native.text or "") if native.valid else None

        counts["rows"] += 1
        counts["archived_rewarded_correct"] += bool(row.get("reward"))
        counts["archived_parser_valid"] += archived_valid
        counts["archived_nontruncated_unsafe"] += archived_unsafe_refusal
        counts["current_rewarded_correct"] += reward_positive
        counts["current_parser_valid"] += current_valid
        counts["current_parser_unsafe"] += bool(parsed is not None and parsed.unsafe)
        counts["current_valid_incorrect"] += current_valid and not current_correct
        counts["newly_valid_incorrect"] += newly_valid and not current_correct
        counts["newly_valid_containing_unsafe_vocabulary"] += bool(
            newly_valid and unsafe_match
        )
        counts["newly_valid_containing_negation"] += bool(
            newly_valid and negation_match
        )
        counts["recovered_rewarded_correct"] += reward_positive and not bool(
            row.get("reward")
        )
        method = parsed.method if parsed is not None else "none"
        status = parsed.status if parsed is not None else "invalid_native_boundary"
        outcomes_by_method[f"{method}:{status}"] += 1
        if current_valid:
            valid_by_method[method] += 1
        if newly_valid:
            newly_valid_by_method[method] += 1
        if reward_positive and not bool(row.get("reward")):
            recovered_by_method[method] += 1

        if archived_unsafe_refusal and not current_valid:
            status = parsed.status if parsed is not None else "invalid_native_boundary"
            remaining_statuses[status] += 1
            remaining_by_episode[f"{row.get('episode_id')}:{status}"] += 1
        if newly_valid and not current_correct:
            newly_valid_incorrect.append(
                {
                    "line": line_number,
                    "episode_id": row.get("episode_id"),
                    "expected": expected,
                    "parsed": parsed.plan,
                }
            )
        if newly_valid and unsafe_match:
            newly_valid_unsafe_vocabulary.append(
                {
                    "line": line_number,
                    "episode_id": row.get("episode_id"),
                    "matched_vocabulary": unsafe_match.group(0),
                    "expected": expected,
                    "parsed": parsed.plan,
                    "method": parsed.method,
                }
            )
        if newly_valid and negation_match:
            newly_valid_negation.append(
                {
                    "line": line_number,
                    "episode_id": row.get("episode_id"),
                    "matched_negation": negation_match.group(0),
                    "expected": expected,
                    "parsed": parsed.plan,
                    "method": parsed.method,
                }
            )
        if archived_valid and not current_valid:
            newly_invalid_archived_valid.append(
                {
                    "line": line_number,
                    "episode_id": row.get("episode_id"),
                    "status": parsed.status
                    if parsed is not None
                    else "invalid_native_boundary",
                }
            )

    return {
        "path": str(path),
        "mode": mode,
        "counts": dict(sorted(counts.items())),
        "remaining_archived_unsafe_by_status": dict(sorted(remaining_statuses.items())),
        "remaining_archived_unsafe_by_episode": dict(
            sorted(remaining_by_episode.items())
        ),
        "current_outcomes_by_method": dict(sorted(outcomes_by_method.items())),
        "current_valid_by_method": dict(sorted(valid_by_method.items())),
        "newly_valid_by_method": dict(sorted(newly_valid_by_method.items())),
        "recovered_rewarded_correct_by_method": dict(
            sorted(recovered_by_method.items())
        ),
        "newly_valid_incorrect": newly_valid_incorrect,
        "newly_valid_unsafe_vocabulary": newly_valid_unsafe_vocabulary,
        "newly_valid_negation": newly_valid_negation,
        "newly_invalid_archived_valid": newly_invalid_archived_valid,
    }


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(measure(parse(Config)), indent=2, sort_keys=True))
