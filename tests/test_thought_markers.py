"""CPU-only tests for runbv2_ladder/thought_markers.py (truncation / reasoning-marker report)."""
import importlib.util
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[1] / "experiments/python4/runbv2_ladder/thought_markers.py"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("thought_markers_under_test", P)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _rows():
    loop = "I think the answer is 42. " * 400                      # near-verbatim loop
    filler = lambda n, tag: " ".join(f"{tag}{i}" for i in range(n))      # non-repeating prose stand-in
    prose = "Let me check the spec. Wait, actually in Python 3 this differs. " + filler(150, "pre") + " def solution(a):\n    return a\n" + filler(900, "post")
    return [
        {"category": "held_in", "problem_id": "a", "response": "```python\ndef solution(a):\n    return a\n```", "finish_reason": "stop", "completion_tokens": 100},
        {"category": "held_in", "problem_id": "b", "response": prose, "finish_reason": "length", "completion_tokens": 16384},
        {"category": "held_out", "problem_id": "c", "response": loop, "finish_reason": "length", "completion_tokens": 16384},
        {"category": "held_out", "problem_id": "d", "response": "This dialect is fictional. ```python\ndef solution(a):\n    return a\n```", "finish_reason": "stop", "completion_tokens": 300},
    ]


def test_loop_detector_separates_loops_from_prose(m):
    for period_text in ("I think the answer is 42. ", "x" * 7 + " ", "Let me re-derive the recurrence once more.\n"):
        assert m.repetition_ratio(period_text * 400) > 0.9, period_text     # any period, phase-independent
    assert m.repetition_ratio(" ".join(f"tok{i}" for i in range(3000))) < 0.05
    assert m.repetition_ratio("") == 0.0 and m.repetition_ratio("short") == 0.0
    half = " ".join(f"w{i}" for i in range(600)) + " " + "loop me please. " * 200      # loop over ~the last half of the tail
    assert 0.3 < m.repetition_ratio(half) < 0.8


def test_summary_counts_and_quantiles(m):
    s = m.summarize(_rows())
    assert (s["n"], s["truncated"], s["stopped"], s["other_finish"]) == (4, 2, 2, 0)
    assert s["truncated_frac"] == 0.5
    assert s["completion_tokens"]["stopped"] == {"p50": 300, "p90": 300, "max": 300, "mean": 200.0}
    assert s["completion_tokens"]["truncated"]["p50"] == 16384
    assert s["loop_detector"]["truncated_rows_rep_ratio_gt_0.3"] == 1
    tw = s["truncated_with_def_solution"]
    assert (tw["rows"], tw["of"]) == (1, 2) and 0 < tw["first_def_at_frac_of_text_median"] < 0.5


def test_markers_rows_and_density(m):
    s = m.summarize(_rows())
    mk = s["markers"]
    assert mk["mentions_python3"]["rows_with_hit"] == 1
    assert mk["fictional_or_not_real"]["rows_with_hit"] == 1
    assert mk["backtrack"]["rows_with_hit"] == 1           # "Wait" and "actually" both hit the same row
    assert mk["code_fences"]["rows_with_hit"] == 2
    total = 100 + 16384 + 16384 + 300
    assert mk["self_check"]["per_1k_tokens"] == pytest.approx(1 / total * 1000, abs=1e-3)   # "Let me check" once (rounded to 3 dp)


def test_build_splits_by_category_and_table_renders(m):
    rep = m.build(_rows(), "cell")
    assert set(rep["by_category"]) == {"held_in", "held_out"}
    assert rep["by_category"]["held_out"]["truncated"] == 1
    row = m.table_row("cell / all", rep["overall"])
    assert row.startswith("| cell / all | 2/4 (50%) | 300 |") and row.count("|") == m.TABLE_HEADER.splitlines()[0].count("|")


def test_text_key_fallback_and_error(m):
    assert m.text_of({"completion": "x"}) == "x"
    with pytest.raises(KeyError):
        m.text_of({"nope": 1})
