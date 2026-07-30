"""Tests for canonical synthesis-shard merging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.bank.pilots.pilot_a.merge_synth_shards import (
    merge_synth_shards,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_merge_synth_shards_restores_candidate_order(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    _write(
        candidates,
        [{"problem_id": "a"}, {"problem_id": "b"}, {"problem_id": "c"}],
    )
    _write(
        first,
        [
            {"problem_id": "c", "seed": 42, "scale_tuning_cap": 8},
            {"problem_id": "a", "seed": 42, "scale_tuning_cap": 8},
        ],
    )
    _write(
        second,
        [{"problem_id": "b", "seed": 42, "scale_tuning_cap": 8}],
    )
    result = merge_synth_shards(candidates, [first, second], output)
    assert result["problem_count"] == 3
    assert [
        json.loads(line)["problem_id"] for line in output.read_text().splitlines()
    ] == ["a", "b", "c"]


def test_merge_synth_shards_rejects_incomplete_coverage(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    shard = tmp_path / "shard.jsonl"
    _write(candidates, [{"problem_id": "a"}, {"problem_id": "b"}])
    _write(shard, [{"problem_id": "a", "seed": 42, "scale_tuning_cap": 8}])
    with pytest.raises(ValueError, match="coverage mismatch"):
        merge_synth_shards(candidates, [shard], tmp_path / "merged.jsonl")
