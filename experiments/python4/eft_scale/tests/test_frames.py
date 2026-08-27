"""CPU tests for the F0-F3 frame families and stratified assignment."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale import frames, teacher  # noqa: E402

ROW = {
    "problem_id": "newfacade:two-sum",
    "parameter_names": ["nums", "target"],
    "statement": "Given an array of integers nums and an integer target, "
    "return indices of the two numbers that add up to target.",
}
CODE = 'def solution(nums, target, out):;;\n    out["value"] = [1, 2] ;;\n    return ;;\n'

PROPORTIONS = {"F0": 0.40, "F1": 0.20, "F2": 0.20, "F3": 0.20}


def test_f0_is_v2_exact():
    """F0 must be byte-identical to the v2 EFT frame so a pure-F0 subset
    bridges to the v2 corpus (SPEC §3.2)."""

    messages = frames.build_frame_messages("F0", ROW, CODE, seed=1)
    v2 = teacher.build_eft_messages(ROW)  # the v2-exact frame, ladder-side
    assert messages[:2] == v2
    assert messages[2] == {"role": "assistant", "content": CODE}
    assert "Python 4" not in messages[0]["content"] + messages[1]["content"]


def test_f1_names_python_4():
    messages = frames.build_frame_messages("F1", ROW, CODE, seed=1)
    assert "Python 4" in messages[0]["content"]
    assert "Python 4" in messages[1]["content"]
    assert messages[2]["content"] == CODE
    assert ROW["statement"] in messages[1]["content"]


def test_f2_fenced_contract_and_deterministic_variant():
    first = frames.build_frame_messages("F2", ROW, CODE, seed=7)
    again = frames.build_frame_messages("F2", ROW, CODE, seed=7)
    assert first == again  # per-problem variant draw is seeded
    assert first[2]["content"].startswith("```python\n")
    assert first[2]["content"].rstrip().endswith("```")
    assert CODE.rstrip() in first[2]["content"]
    assert ROW["statement"] in first[1]["content"]  # never paraphrased
    assert "fenced code block" in first[1]["content"]
    other_seed = frames.build_frame_messages("F2", ROW, CODE, seed=8)
    assert other_seed[2] == first[2]  # assistant turn does not vary


def test_f3_minimal_no_system():
    messages = frames.build_frame_messages("F3", ROW, CODE, seed=1)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert ROW["statement"] in messages[0]["content"]
    assert messages[1]["content"] == CODE


def test_unknown_frame_rejects():
    with pytest.raises(ValueError):
        frames.build_frame_messages("F9", ROW, CODE, seed=1)


def test_frame_counts_exact_with_f0_floor():
    counts = frames.frame_counts(10, PROPORTIONS)
    assert sum(counts.values()) == 10
    assert counts["F0"] == 4
    for n in (1, 2, 3, 7, 97, 1024, 4096):
        counts = frames.frame_counts(n, PROPORTIONS)
        assert sum(counts.values()) == n
        assert counts["F0"] >= math.ceil(0.40 * n)
        assert all(v >= 0 for v in counts.values())


def test_frame_counts_rejects_bad_proportions():
    with pytest.raises(ValueError):
        frames.frame_counts(10, {"F0": 1.0})
    with pytest.raises(ValueError):
        frames.frame_counts(10, {**PROPORTIONS, "F0": 0.9})


def test_assign_frames_stratified_and_deterministic():
    rows = [{"problem_id": f"p:{i:04d}"} for i in range(100)]
    first = frames.assign_frames(
        rows, seed=42, proportions=PROPORTIONS, stratum="held_out|train"
    )
    again = frames.assign_frames(
        list(reversed(rows)), seed=42, proportions=PROPORTIONS, stratum="held_out|train"
    )
    assert first == again  # order-insensitive, seed-deterministic
    assert set(first) == {row["problem_id"] for row in rows}
    from collections import Counter

    counts = Counter(first.values())
    assert counts["F0"] == 40 and counts["F1"] == 20
    other_stratum = frames.assign_frames(
        rows, seed=42, proportions=PROPORTIONS, stratum="held_in|train"
    )
    assert other_stratum != first  # strata shuffle independently


def test_assign_frames_rejects_duplicate_ids():
    rows = [{"problem_id": "p:1"}, {"problem_id": "p:1"}]
    with pytest.raises(ValueError):
        frames.assign_frames(rows, seed=1, proportions=PROPORTIONS, stratum="s")
