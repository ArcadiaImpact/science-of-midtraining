"""CPU-only tests for the P3 ceiling measurement core (suite_p3).

Subprocess CPython is allowed (it is the grader under test — always
present); no network, no GPU, no Boa.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
EVAL_V3 = HERE.parent
REPO_ROOT = HERE.parents[3]
for path in (str(EVAL_V3), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import suite  # noqa: E402
import suite_p3  # noqa: E402


def _row(**overrides):
    row = {
        "problem_id": "p3test:1",
        "statement": "Return the sum of the list.",
        "parameter_names": ["nums"],
        "tests": [
            {"args": [[1, 2, 3]], "kwargs": {}, "expected": 6},
            {"args": [[-1, 1]], "kwargs": {}, "expected": 0},
        ],
        "rules_required": [],
        "rules_expressed": [],
        "style": "held_in",
        "split": "test_heldin",
        "difficulty": "easy",
        "dialect": "python3",
        "gold_code": "def solution(nums):\n    return sum(nums)",
    }
    row.update(overrides)
    return row


# Frame parity: the P3 frame is the P4 frame with only the dialect renamed.


def test_prompt_is_p4_prompt_with_dialect_renamed():
    row = _row()
    assert suite_p3.build_prompt(row) == suite.build_prompt(row).replace(
        "Python 4", "Python 3"
    )
    assert suite_p3.SYSTEM_PROMPT == suite.SYSTEM_PROMPT.replace(
        "Python 4", "Python 3"
    )
    assert "Python 3" in suite_p3.PREAMBLE and "Python 4" not in suite_p3.PREAMBLE


def test_extraction_is_shared_not_forked():
    # suite_p3 binds the packaged suite module; the alias must be identity,
    # not a fork (compare against suite_p3's own suite binding to stay
    # robust to the flat-vs-packaged dual module instances in tests).
    assert suite_p3.extract_answer_code is suite_p3.suite.extract_answer_code


def test_timeout_constants_mirror_p4():
    assert suite_p3.GRADE_TIMEOUT_SECONDS == suite.GRADE_TIMEOUT_SECONDS
    assert suite_p3.GRADE_RETRY_TIMEOUT_SECONDS == suite.GRADE_RETRY_TIMEOUT_SECONDS


# Harness render + roundtrip under real CPython.


def test_harness_renders_return_style_asserts():
    harness = suite_p3.build_p3_harness(
        "def solution(nums):\n    return 1", _row(), sentinel="__P3_ALL_PASS_x__"
    )
    assert "solution([1, 2, 3])" in harness
    assert "__p3_normalize" in harness
    assert "P3_TEST_FAIL" in harness


def test_grade_code_certifies_a_correct_solution():
    graded = suite_p3.grade_code("def solution(nums):\n    return sum(nums)", _row())
    assert graded["certified"] and graded["all_tests_pass"]
    assert graded["cpython_compile"]
    assert graded["failure_reason"] is None
    assert graded["grader_mode"] == "p3_cpython"


def test_grade_code_normalizes_tuple_returns():
    row = _row(
        tests=[{"args": [[1, 2]], "kwargs": {}, "expected": [1, 2]}],
    )
    graded = suite_p3.grade_code("def solution(nums):\n    return tuple(nums)", row)
    assert graded["certified"], graded


def test_grade_code_reports_failing_test_index():
    graded = suite_p3.grade_code("def solution(nums):\n    return 999", _row())
    assert not graded["certified"]
    assert graded["failure_reason"] == "runtime"
    assert "P3_TEST_FAIL index=0" in graded["failure_detail"]


def test_grade_code_flags_syntax_as_compile():
    graded = suite_p3.grade_code("def solution(nums:\n    return 1", _row())
    assert graded["failure_reason"] == "compile"
    assert not graded["cpython_compile"]


# SystemExit(0) soundness (review finding, 2026-08-29): a candidate that
# exits 0 before/without the test comparisons must NOT certify. The harness
# prints a per-call nonce sentinel as its final statement; grade_code
# requires sentinel AND returncode 0 (additive to rc — passing golds above
# and the failing-index test prove the rc path is unchanged).


def test_sys_exit_zero_inside_solution_is_not_certified():
    graded = suite_p3.grade_code(
        "import sys\ndef solution(nums):\n    sys.exit(0)", _row()
    )
    assert not graded["certified"]
    assert not graded["all_tests_pass"]
    assert graded["failure_reason"] == "runtime"
    assert "sentinel" in (graded["failure_detail"] or "")


def test_bare_exit_zero_at_module_level_is_not_certified():
    graded = suite_p3.grade_code(
        "def solution(nums):\n    return sum(nums)\nexit(0)", _row()
    )
    assert not graded["certified"]
    assert graded["failure_reason"] == "runtime"
    assert "sentinel" in (graded["failure_detail"] or "")


def test_raise_systemexit_zero_is_not_certified():
    graded = suite_p3.grade_code(
        "def solution(nums):\n    raise SystemExit(0)", _row()
    )
    assert not graded["certified"]
    assert graded["failure_reason"] == "runtime"
    assert "sentinel" in (graded["failure_detail"] or "")


def test_guessed_sentinel_print_does_not_certify():
    # Even a candidate that prints the sentinel PREFIX and exits 0 cannot
    # certify: the full sentinel carries a per-grading-call nonce the
    # candidate can never know.
    graded = suite_p3.grade_code(
        "import sys\n"
        "def solution(nums):\n"
        "    return sum(nums)\n"
        "print('__P3_ALL_PASS_0123456789abcdef__')\n"
        "sys.exit(0)",
        _row(),
    )
    assert not graded["certified"]
    assert graded["failure_reason"] == "runtime"


def test_candidate_stdout_noise_does_not_break_certification():
    # Prints (even without trailing newlines) must not false-negative a
    # genuinely passing solution: the sentinel check is substring-based.
    graded = suite_p3.grade_code(
        "import sys\n"
        "def solution(nums):\n"
        "    sys.stdout.write('debug noise')\n"
        "    return sum(nums)",
        _row(),
    )
    assert graded["certified"], graded


def test_harness_sentinel_is_final_statement_and_nonce_fresh():
    row = _row()
    h1 = suite_p3.build_p3_harness("def solution(nums):\n    return 1", row, sentinel="__P3_ALL_PASS_aa__")
    assert h1.rstrip().splitlines()[-1] == "print('__P3_ALL_PASS_aa__')"
    assert h1.rstrip().splitlines()[-1] != h1.splitlines()[0]  # after the code
    # grade_code mints a fresh nonce per call: two harnesses for the same
    # candidate must not share a sentinel (probe via the private minter).
    assert suite_p3._mint_sentinel() != suite_p3._mint_sentinel()


def test_grade_code_rejects_forbidden_import_but_allows_extended_stdlib():
    bad = suite_p3.grade_code("import os\ndef solution(nums):\n    return 0", _row())
    assert bad["failure_reason"] == "unsafe"
    row = _row(tests=[{"args": [[1, 2]], "kwargs": {}, "expected": 3}])
    good = suite_p3.grade_code(
        "from fractions import Fraction\n"
        "def solution(nums):\n"
        "    return int(sum(Fraction(v) for v in nums))",
        row,
    )
    assert good["certified"], good


def test_grade_code_timeout_and_retry_marker():
    slow = "def solution(nums):\n    while True:\n        pass"
    graded = suite_p3.grade_code(slow, _row(), timeout=1)
    assert graded["failure_reason"] == "timeout"
    retried = suite_p3.grade_code_with_retry(slow, _row(), timeout=1, retry_timeout=1)
    assert retried["failure_reason"] == "timeout"
    assert retried["timed_out_first_pass"] is True


def test_grade_response_extracts_fenced_and_reasoning_responses():
    response = (
        "Let me think briefly.\n\n```python\n"
        "def solution(nums):\n    return sum(nums)\n```"
    )
    graded = suite_p3.grade_response(response, _row())
    assert graded["certified"], graded


def test_grade_response_boa_executable_slot_carries_the_interpreter():
    graded = suite_p3.grade_response(
        "```python\ndef solution(nums):\n    return sum(nums)\n```",
        _row(),
        boa_executable=sys.executable,
    )
    assert graded["certified"], graded


def test_p4_surface_markers_flag_dialect_leakage():
    code = 'def solution(nums, out):;;\n    out["value"] = 1 ;;'
    graded = suite_p3.grade_code(code, _row())
    assert "double_semicolon" in graded["p4_surface_markers"]
    assert "out_value_contract" in graded["p4_surface_markers"]
    assert not graded["certified"]


# Prompt-leak audit.


def test_audit_prompts_passes_clean_rows_and_counts_diagnostics():
    report = suite_p3.audit_prompts({"held_in": [_row()], "held_out": []})
    assert report["prompts_audited"] == 1
    assert set(report["hard_patterns"]) == set(suite.HARD_LEAK_PATTERNS)
    assert report["grader_mode"] == "p3_cpython"


def test_audit_prompts_raises_on_p4_syntax_leak():
    leaky = _row(statement="Use out =(8) {} ;; to allocate.")
    with pytest.raises(ValueError, match="leak"):
        suite_p3.audit_prompts({"held_in": [leaky], "held_out": []})


# Aggregation.


def _graded(problem_id, category, certified, *, markers=(), difficulty="easy"):
    return {
        "problem_id": problem_id,
        "grader_mode": "p3_cpython",
        "category": category,
        "difficulty": difficulty,
        "certified": certified,
        "cpython_compile": True,
        "all_tests_pass": certified,
        "failure_reason": None if certified else "runtime",
        "p4_surface_markers": list(markers),
    }


def test_aggregate_rates_and_p4_surface():
    rows = [
        _graded("a", "held_in", True),
        _graded("b", "held_in", False, markers=["double_semicolon"]),
        _graded("c", "held_out", True),
        _graded("d", "held_out", True),
    ]
    summary = suite_p3.aggregate(rows)
    assert summary["grader_mode"] == "p3_cpython"
    assert summary["n_total"] == 4
    held_in = summary["categories"]["held_in"]
    assert held_in["certified"]["numerator"] == 1
    assert held_in["certified"]["n"] == 2
    assert held_in["p4_surface"]["numerator"] == 1
    assert held_in["failure_kinds"]["runtime"] == 1
    assert "held_out_rule_expression" not in summary
    assert 0 <= held_in["certified"]["ci_low"] <= held_in["certified"]["ci_high"] <= 1


def test_aggregate_rejects_unknown_categories():
    with pytest.raises(ValueError, match="unknown categories"):
        suite_p3.aggregate([_graded("a", "mystery", True)])


def test_summary_markdown_marks_the_ceiling_grader():
    summary = suite_p3.aggregate(
        [_graded("a", "held_in", True), _graded("b", "held_out", False)]
    )
    text = suite_p3.summary_markdown(summary, target="unit/ceiling")
    assert "P3 ceiling" in text and "p3_cpython" in text


# join_category carries the P3 gold provenance labels.


def test_join_category_labels():
    row = _row(
        gold_provenance={"source": "reference_adapted", "adapter": "top_fn_rename"},
        tier="tier1",
        rules_expressed=["uppercase_boolean"],
    )
    graded = suite_p3.grade_code(row["gold_code"], row)
    joined = suite_p3.join_category(graded, row, "held_in")
    assert joined["category"] == "held_in"
    assert joined["gold_source"] == "reference_adapted"
    assert joined["rules_expressed_gold_p4"] == ["uppercase_boolean"]
    with pytest.raises(ValueError, match="mismatch"):
        suite_p3.join_category(graded, _row(problem_id="other"), "held_in")


# load_test_rows validation (row-count constant shrunk for the fixture).


def _write_rows(path: Path, rows):
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def test_load_test_rows_validates_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(suite_p3, "EXPECTED_ROWS_PER_FILE", 1)
    held_in = _row()
    held_out = _row(problem_id="p3test:2", style="held_out", split="test_heldout")
    _write_rows(tmp_path / suite_p3.TEST_FILES["held_in"], [held_in])
    _write_rows(tmp_path / suite_p3.TEST_FILES["held_out"], [held_out])
    rows = suite_p3.load_test_rows(tmp_path)
    assert {r["problem_id"] for r in rows["held_in"]} == {"p3test:1"}

    _write_rows(tmp_path / suite_p3.TEST_FILES["held_out"], [held_in])
    with pytest.raises(ValueError, match="style"):
        suite_p3.load_test_rows(tmp_path)

    shared = _row(style="held_out", split="test_heldout")
    _write_rows(tmp_path / suite_p3.TEST_FILES["held_out"], [shared])
    with pytest.raises(ValueError, match="share problem_ids"):
        suite_p3.load_test_rows(tmp_path)


def test_load_test_rows_mode_dataset_cross_gates(tmp_path, monkeypatch):
    """Review condition: mode/dataset crosses fail loudly BOTH directions."""

    monkeypatch.setattr(suite_p3, "EXPECTED_ROWS_PER_FILE", 1)
    monkeypatch.setattr(suite, "EXPECTED_ROWS_PER_FILE", 1)
    p4_style_row = {k: v for k, v in _row().items() if k != "dialect"}
    held_out = _row(problem_id="p3test:2", style="held_out", split="test_heldout")

    # p3 loader rejects P4 rows (no dialect marker).
    _write_rows(tmp_path / suite_p3.TEST_FILES["held_in"], [p4_style_row])
    _write_rows(tmp_path / suite_p3.TEST_FILES["held_out"], [held_out])
    with pytest.raises(ValueError, match="mode: p3 refuses"):
        suite_p3.load_test_rows(tmp_path)

    # p4 loader rejects P3 mirror rows (dialect=python3).
    _write_rows(tmp_path / suite.TEST_FILES["held_in"], [_row()])
    _write_rows(
        tmp_path / suite.TEST_FILES["held_out"],
        [{k: v for k, v in held_out.items() if k != "dialect"}],
    )
    with pytest.raises(ValueError, match="needs mode: p3"):
        suite.load_test_rows(tmp_path)

    # p3 mode against a P4-only snapshot (pre-mirror revision): loud miss.
    for filename in suite_p3.TEST_FILES.values():
        (tmp_path / filename).unlink()
    with pytest.raises(FileNotFoundError, match="P3 mirror files"):
        suite_p3.load_test_rows(tmp_path)
