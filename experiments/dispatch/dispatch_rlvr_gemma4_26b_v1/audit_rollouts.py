"""Recompute every rollout reward and hard-fail unsafe positive surfaces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .reward import score_completion


@dataclass
class Config:
    rollout_dir: str = ""
    mode: str = ""
    output: str = ""
    positive_review: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"direct", "thinking"}:
            raise ValueError("mode must be direct|thinking")
        if not self.rollout_dir or not self.output:
            raise ValueError("rollout_dir and output are required")


def audit(cfg: Config) -> dict:
    files = sorted(Path(cfg.rollout_dir).glob("raw_rollouts.rank-*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no raw rollout logs in {cfg.rollout_dir}")
    counts = {
        "rows": 0,
        "reward_positive": 0,
        "parser_valid": 0,
        "parser_unsafe": 0,
        "truncated": 0,
    }
    failures: list[dict] = []
    positive_rows: list[dict] = []
    for path in files:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rescored = score_completion(
                row.get("completion", ""),
                completion_raw_text=row.get("completion_raw_text"),
                episode=row["episode"],
                mode=cfg.mode,
                completion_truncated=bool(row.get("truncated")),
            )
            expected = asdict(rescored)
            counts["rows"] += 1
            counts["reward_positive"] += rescored.reward > 0
            counts["parser_valid"] += rescored.parser_valid > 0
            counts["parser_unsafe"] += rescored.parser_unsafe > 0
            counts["truncated"] += bool(row.get("truncated"))
            if rescored.reward > 0:
                episode = row["episode"]
                positive_rows.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "episode_id": row.get("episode_id")
                        or episode.get("episode_id"),
                        "run_ids": [run["run_id"] for run in episode["runs"]],
                        "available_crews": [crew["name"] for crew in episode["crews"]],
                        "expected_plan": episode["charter_plan"],
                        "completion": row.get("completion", ""),
                        "completion_raw_text": row.get("completion_raw_text"),
                        "components": expected,
                    }
                )
            reasons = []
            if float(row.get("reward", -1)) != rescored.reward:
                reasons.append("reward_recompute_mismatch")
            if rescored.reward > 0 and (
                rescored.parser_unsafe
                or not rescored.format_valid
                or bool(row.get("truncated"))
                or rescored.parser_json
                + rescored.parser_natural
                + rescored.parser_labelled_records
                != 1
            ):
                reasons.append("unsafe_reward_positive_surface")
            for key, value in expected.items():
                if isinstance(value, (float, int)) and key in row:
                    if float(row[key]) != float(value):
                        reasons.append(f"component_mismatch:{key}")
            if reasons:
                failures.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "episode_id": row.get("episode_id"),
                        "reasons": sorted(set(reasons)),
                    }
                )
    review_path = (
        Path(cfg.positive_review)
        if cfg.positive_review
        else Path(cfg.output).with_name("REWARD_POSITIVE_REVIEW.jsonl")
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in positive_rows
    )
    review_path.write_text(review_payload)
    result = {
        "schema_version": 1,
        "mode": cfg.mode,
        "files": [str(path) for path in files],
        "counts": counts,
        "reward_positive_review": {
            "path": str(review_path),
            "rows": len(positive_rows),
            "sha256": hashlib.sha256(review_payload.encode()).hexdigest(),
            "approval": "manual review required before the next RL phase",
        },
        "failures": failures[:100],
        "passed": not failures,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(
            f"rollout audit found {len(failures)} mismatches/unsafe positives; see {output}"
        )
    return result


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(audit(parse(Config)), indent=2, sort_keys=True))
