"""Tests for the leakage-safe Pilot A question-set exporter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.bank.pilots.pilot_a.build_question_sets import (
    build_question_sets,
    problem_splits,
)


def _solution(
    candidate_id: str,
    solution_index: int,
    time_s: float,
    peak_bytes: int,
) -> dict[str, object]:
    times = [time_s * 0.99, time_s, time_s * 1.01]
    rss = [peak_bytes + value for value in (-1_000, 0, 1_000)]
    return {
        "candidate_id": candidate_id,
        "solution_index": solution_index,
        "source": f"# {candidate_id}\nprint(1)\n",
        "status": "measured",
        "drop_reason": None,
        "correctness": [{"test_index": 0, "source": "public", "ok": True}],
        "trials": [
            {
                "payload_wall_s": trial_time,
                "parent_wall_s": trial_time + 0.01,
                "rss_bytes": trial_rss,
            }
            for trial_time, trial_rss in zip(times, rss)
        ],
        "times_s": times,
        "rss_trials_bytes": rss,
        "median_time_s": time_s,
        "median_rss_bytes": peak_bytes,
        "baseline_subtracted_peak_bytes": peak_bytes - 1_000_000,
        "time_spread": 0.02,
        "peak_spread": 0.001,
        "flags": [],
        "z_silence_hits": [],
        "style_flags": [],
    }


def _measurement(problem_id: str) -> dict[str, object]:
    return {
        "problem_id": problem_id,
        "source": 2,
        "difficulty": 7,
        "statement": f"Solve {problem_id}.",
        "measurement_source": "synth",
        "platform": {
            "platform": "Linux-test",
            "python": "3.12.0",
            "implementation": "CPython",
            "sys_platform": "linux",
        },
        "baseline": {"median_rss_bytes": 1_000_000},
        "solutions": [
            _solution("speed", 0, 0.10, 5_000_000),
            _solution("balanced", 1, 0.17, 3_800_000),
            _solution("memory", 2, 0.25, 3_000_000),
            _solution("loser", 3, 0.40, 6_000_000),
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_problem_splits_keep_multilabel_problems_together():
    categories = {
        "tradeoff": {"both-a", "both-b", "trade-only"},
        "jointly_dominant": {"both-a", "both-b", "dominant-only"},
    }
    splits, targets = problem_splits(categories, eval_fraction=0.5, seed=42)
    assert targets == {"tradeoff": 2, "jointly_dominant": 2}
    assert sum(splits[value] == "eval" for value in categories["tradeoff"]) == 2
    assert (
        sum(
            splits[value] == "eval"
            for value in categories["jointly_dominant"]
        )
        == 2
    )


def test_build_question_sets_emits_two_categories_and_triplets(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "questions"
    run_dir.mkdir()
    measurements = [_measurement("p1"), _measurement("p2")]
    _write_jsonl(run_dir / "measurements.jsonl", measurements)
    _write_jsonl(
        run_dir / "candidates.jsonl",
        [{"problem_id": row["problem_id"]} for row in measurements],
    )
    _write_jsonl(
        run_dir / "synth_tests.jsonl",
        [
            {
                "problem_id": row["problem_id"],
                "source": "synth",
                "n": 1000,
                "seed": 42,
                "input": "1\n",
                "output": "1\n",
                "survivor_ids": ["speed", "balanced", "memory", "loser"],
            }
            for row in measurements
        ],
    )

    manifest = build_question_sets(
        run_dir,
        out_dir,
        eval_fraction=0.5,
        seed=42,
    )

    assert manifest["counts"] == {
        "train": {
            "tradeoff": 1,
            "jointly_dominant": 1,
            "pareto_triplets": 1,
        },
        "eval": {
            "tradeoff": 1,
            "jointly_dominant": 1,
            "pareto_triplets": 1,
        },
    }
    assert manifest["unique_problem_counts"] == {
        "train": 1,
        "eval": 1,
        "overlap": 0,
    }
    questions = [
        json.loads(line)
        for line in (out_dir / "questions.jsonl").read_text().splitlines()
    ]
    train_ids = {
        row["problem_id"] for row in questions if row["split"] == "train"
    }
    eval_ids = {
        row["problem_id"] for row in questions if row["split"] == "eval"
    }
    assert train_ids.isdisjoint(eval_ids)
    triplets = [
        json.loads(line)
        for line in (out_dir / "pareto_triplets.jsonl").read_text().splitlines()
    ]
    assert {
        solution["role"]
        for triplet in triplets
        for solution in triplet["solutions"]
    } == {"latency", "balanced", "memory"}
