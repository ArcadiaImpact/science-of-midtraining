"""CPU-only tests for the eval_v3 measurement core: prompt frame, leak
audit, grading contract, aggregation. No network, no GPU; the one test that
executes Boa skips unless the pinned local checkout is present."""

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


def make_row(**overrides):
    row = {
        "problem_id": "tacov:1",
        "statement": "Given a list, return the largest item.",
        "parameter_names": ["values"],
        "tests": [
            {"args": [[1, 2, 3]], "kwargs": {}, "expected": 3},
            {"args": [[5, 4]], "kwargs": {}, "expected": 5},
        ],
        "rules_required": [
            "statement_terminators",
            "out_parameter",
            "manual_allocation",
        ],
        "rules_expressed": [],
        "style": "held_in",
        "split": "test_heldin",
        "difficulty": "easy",
        "gold_code": 'def solution(values, out):;;\n    best =(8) values[1] ;;\n'
        "    for i in range(1, len(values) + 1):;;\n"
        "        if values[i] > best:;;\n            best =(8) values[i] ;;\n"
        '    out["value"] = best ;;\n    return ;;',
        "teacher_tier": "luna",
        "tier": "native",
    }
    row.update(overrides)
    return row


# Prompt frame


def test_build_prompt_single_parameter_matches_suite_b_hard_shape():
    prompt = suite.build_prompt(make_row())
    assert prompt.startswith(
        "Write a Python 4 function named `solution`. You may reason briefly, "
        "then give your final code. The function takes one parameter, "
        "`values`, and must return exactly what the task below specifies.\n\n"
    )
    assert prompt.endswith("Given a list, return the largest item.")


def test_build_prompt_multi_parameter_states_order():
    row = make_row(parameter_names=["n", "a", "b"])
    prompt = suite.build_prompt(row)
    assert "takes parameters `n`, `a` and `b`, in that order," in prompt


def test_prompt_contains_no_rule_descriptions():
    prompt = suite.build_prompt(make_row())
    for banned in (";;", "=(", "out[", "terminator", "allocation", "1-based"):
        assert banned not in prompt


# Leak audit


def test_audit_prompts_passes_on_clean_rows():
    rows = {
        "held_in": [make_row()],
        "held_out": [
            make_row(problem_id="cc:9", style="held_out", split="test_heldout")
        ],
    }
    report = suite.audit_prompts(rows)
    assert report["prompts_audited"] == 2
    assert set(report["hard_patterns"].values()) == {0}


@pytest.mark.parametrize(
    "statement, pattern",
    [
        ("Terminate lines with ;; as usual.", "double_semicolon"),
        ("Allocate with x =(8) style.", "allocation_syntax"),
        ("The modulus is 1_000_000_007 here.", "grouped_thousands_literal"),
        ('Write out["value"] = answer.', "python4_out_contract"),
    ],
)
def test_audit_prompts_raises_on_hard_leaks(statement, pattern):
    rows = {"held_in": [make_row(statement=statement)]}
    with pytest.raises(ValueError, match=pattern):
        suite.audit_prompts(rows)


def test_audit_prompts_counts_diagnostic_surfaces_without_failing():
    rows = {
        "held_in": [
            make_row(statement="Do NOT modify the input; ranges are [-4, 9].")
        ]
    }
    report = suite.audit_prompts(rows)
    assert report["diagnostic_patterns"]["upper_boolean_word"]["prompts_hit"] == 1
    assert report["diagnostic_patterns"]["bracket_minus"]["prompts_hit"] == 1
    assert report["diagnostic_patterns"]["upper_boolean_word"]["examples"]


def test_frame_texts_are_leak_free():
    # The fixed frame/system text itself must never trip ANY pattern,
    # including the diagnostic ones.
    suite.audit_prompts({"held_in": []})


# Row loading


def test_load_test_rows_validates_counts_and_styles(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "EXPECTED_ROWS_PER_FILE", 2)
    heldin = [make_row(problem_id=f"a:{i}") for i in range(2)]
    heldout = [
        make_row(problem_id=f"b:{i}", style="held_out", split="test_heldout")
        for i in range(2)
    ]
    (tmp_path / "eft_v3_test_heldin.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in heldin)
    )
    (tmp_path / "eft_v3_test_heldout.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in heldout)
    )
    rows = suite.load_test_rows(tmp_path)
    assert [len(rows["held_in"]), len(rows["held_out"])] == [2, 2]

    bad = dict(heldout[0], style="held_in")
    (tmp_path / "eft_v3_test_heldout.jsonl").write_text(
        json.dumps(bad) + "\n" + json.dumps(heldout[1]) + "\n"
    )
    with pytest.raises(ValueError, match="style"):
        suite.load_test_rows(tmp_path)


def test_load_test_rows_rejects_wrong_count(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "EXPECTED_ROWS_PER_FILE", 2)
    (tmp_path / "eft_v3_test_heldin.jsonl").write_text(
        json.dumps(make_row()) + "\n"
    )
    (tmp_path / "eft_v3_test_heldout.jsonl").write_text("")
    with pytest.raises(ValueError, match="rows"):
        suite.load_test_rows(tmp_path)


# Extraction (the import-preserving bare-answer path — pre-mortem #1)


def test_extract_answer_code_keeps_leading_imports_on_bare_answers():
    response = (
        "import helper;;\nimport math;;\n"
        "def solution(n, out):;;\n    out[\"value\"] = n ;;\n    return ;;\n"
    )
    code = suite.extract_answer_code(response)
    assert code.startswith("import helper;;\nimport math;;\ndef solution")


def test_extract_answer_code_keeps_imports_after_reasoning_prose():
    response = (
        "First I reason about the problem, briefly.\n\n"
        "import helper;;\n"
        "def solution(n, out):;;\n    out[\"value\"] = n ;;\n    return ;;\n"
        "\nThat concludes the answer.\n"
    )
    code = suite.extract_answer_code(response)
    assert code.startswith("import helper;;\ndef solution")
    assert "concludes" not in code


def test_extract_answer_code_does_not_cross_prose_between_imports():
    response = (
        "import math;;\nSome prose in between.\nimport helper;;\n"
        "def solution(n, out):;;\n    out[\"value\"] = n ;;\n"
    )
    code = suite.extract_answer_code(response)
    assert code.startswith("import helper;;\ndef solution")
    assert "math" not in code


def test_extract_answer_code_fenced_path_unchanged():
    response = (
        "Reasoning here with import helper mentioned in prose.\n"
        "```python\ndef solution(n, out):;;\n    out[\"value\"] = n ;;\n```\n"
    )
    code = suite.extract_answer_code(response)
    assert code.startswith("def solution")
    assert "Reasoning" not in code


def test_extract_answer_code_none_on_no_code():
    assert suite.extract_answer_code("no code at all") is None
    assert suite.extract_answer_code("") is None


def test_extract_answer_code_rescues_imports_in_unclosed_fence():
    # Truncated response: fence never closes, so extract_rule_code takes the
    # bare path — the import rescue must still apply.
    response = (
        "Reasoning...\n```python\nimport helper;;\n"
        "def solution(n, out):;;\n    out[\"value\"] = n ;;\n    return ;;\n"
    )
    code = suite.extract_answer_code(response)
    assert code.startswith("import helper;;\ndef solution")


# Grading (grade_python4 monkeypatched; Boa integration lives below)


def _fake_grade(result):
    def grade(code, problem, **kwargs):
        grade.calls.append({"code": code, "problem": problem, **kwargs})
        return dict(result)

    grade.calls = []
    return grade


def test_grade_response_no_code_extracted():
    graded = suite.grade_response(
        "I could not solve this one, sorry.",
        make_row(),
        boa_executable="/nonexistent",
    )
    assert graded["failure_reason"] == "no_code_extracted"
    assert graded["certified"] is False
    assert graded["extracted_code"] is None


def test_grade_response_certified_requires_zero_warnings(monkeypatch):
    fake = _fake_grade(
        {
            "boa_compile": True,
            "boa_pass": True,
            "warning_free": False,
            "python4_adoption": True,
            "error_kind": None,
            "stdout": "",
            "stderr": "ReadabilityWarning: ungrouped literal\nWarning: x",
            "tags": {"uppercase_boolean": True},
            "rule_pass": {"statement_terminators": True},
        }
    )
    monkeypatch.setattr(suite, "grade_python4", fake)
    graded = suite.grade_response(
        "```python\ndef solution(values, out):;;\n    pass ;;\n```",
        make_row(),
        boa_executable="python4",
    )
    assert graded["all_tests_pass"] is True
    assert graded["certified"] is False
    assert graded["failure_reason"] == "warnings"
    assert graded["warnings"] == [
        "ReadabilityWarning: ungrouped literal",
        "Warning: x",
    ]
    # The pre-registered flags: no static contract gate, row rules recorded.
    call = fake.calls[0]
    assert call["enforce_contract"] is False
    assert call["required_rules"] == tuple(make_row()["rules_required"])


def test_grade_response_patches_matrix_multiplication(monkeypatch):
    fake = _fake_grade(
        {
            "boa_compile": True,
            "boa_pass": True,
            "warning_free": True,
            "python4_adoption": True,
            "error_kind": None,
            "stdout": "",
            "stderr": "",
            "tags": {"matrix_multiplication": True},
            "rule_pass": {"matrix_multiplication": False},
        }
    )
    monkeypatch.setattr(suite, "grade_python4", fake)
    row = make_row(rules_required=["matrix_multiplication"])
    graded = suite.grade_response("```py\ndef solution(values, out):;;\n```", row,
                                  boa_executable="python4")
    assert graded["certified"] is True
    assert graded["rule_pass"]["matrix_multiplication"] is True


def test_grade_response_with_retry_reruns_timeouts(monkeypatch):
    outcomes = iter(
        [
            {"failure_reason": "timeout", "certified": False},
            {"failure_reason": None, "certified": True},
        ]
    )
    calls = []

    def fake_grade_response(response, row, *, boa_executable, timeout):
        calls.append(timeout)
        base = {
            "problem_id": row["problem_id"],
            "extracted_code": "x",
            "boa_compile": True,
            "all_tests_pass": True,
            "warning_free": True,
            "python4_adoption": True,
            "warnings": [],
            "tags": {},
            "rule_pass": {},
            "timeout_seconds": timeout,
        }
        return {**base, **next(outcomes)}

    monkeypatch.setattr(suite, "grade_response", fake_grade_response)
    graded = suite.grade_response_with_retry(
        "resp", make_row(), boa_executable="python4"
    )
    assert calls == [suite.GRADE_TIMEOUT_SECONDS, suite.GRADE_RETRY_TIMEOUT_SECONDS]
    assert graded["certified"] is True
    assert graded["timed_out_first_pass"] is True


# Aggregation


def graded_row(category, certified, *, tags=None, difficulty="easy", **extra):
    return {
        "problem_id": f"{category}:{id(object())}",
        "category": category,
        "difficulty": difficulty,
        "certified": certified,
        "boa_compile": certified,
        "all_tests_pass": certified,
        "warning_free": certified,
        "python4_adoption": certified,
        "failure_reason": None if certified else "runtime",
        "tags": tags or {},
        "rule_pass": {},
        **extra,
    }


def test_aggregate_reports_rates_cis_and_expression():
    rows = (
        [graded_row("held_in", True) for _ in range(3)]
        + [graded_row("held_in", False)]
        + [
            graded_row("held_out", True, tags={"uppercase_boolean": True}),
            graded_row("held_out", False, tags={"uppercase_boolean": True}),
            graded_row("held_out", False),
        ]
    )
    summary = suite.aggregate(rows)
    held_in = summary["categories"]["held_in"]["certified"]
    assert held_in["numerator"] == 3 and held_in["n"] == 4
    assert 0 < held_in["ci_low"] < held_in["rate"] < held_in["ci_high"] < 1
    expression = summary["held_out_rule_expression"]["uppercase_boolean"]
    assert expression["all_answers"]["numerator"] == 2
    assert expression["all_answers"]["n"] == 3
    assert expression["certified_answers"]["numerator"] == 1
    assert expression["certified_answers"]["n"] == 1
    assert expression["headline"] is True
    assert summary["held_out_rule_expression"]["end_inclusive_slice"]["headline"] is False
    by_difficulty = summary["categories"]["held_in"]["by_difficulty"]
    assert by_difficulty["easy"]["n"] == 4


def test_aggregate_rejects_unknown_categories():
    with pytest.raises(ValueError, match="unknown categories"):
        suite.aggregate([graded_row("held_sideways", True)])


def test_wilson_matches_eft_v2_analysis():
    analysis = pytest.importorskip("experiments.python4.eft_v2.analysis")
    for numerator, denominator in [(0, 5), (3, 7), (1024, 1024), (511, 1024)]:
        assert suite.wilson_interval(numerator, denominator) == pytest.approx(
            analysis.wilson_interval(numerator, denominator)
        )


def test_join_category_attaches_labels_and_checks_ids():
    row = make_row()
    graded = {"problem_id": row["problem_id"], "certified": True}
    joined = suite.join_category(graded, row, "held_in")
    assert joined["category"] == "held_in"
    assert joined["difficulty"] == "easy"
    assert joined["teacher_tier"] == "luna"
    with pytest.raises(ValueError, match="mismatch"):
        suite.join_category({"problem_id": "other"}, row, "held_in")


def test_summary_markdown_renders_tables():
    rows = [graded_row("held_in", True), graded_row("held_out", False)]
    text = suite.summary_markdown(suite.aggregate(rows), target="glm_control")
    assert "### glm_control" in text
    assert "| held_in |" in text and "| held_out |" in text


# Boa integration (skipped where the pinned checkout is absent)

BOA = Path("/workspace/boa/.venv/bin/python4")


@pytest.mark.skipif(not BOA.is_file(), reason="pinned Boa checkout not present")
def test_grade_response_certifies_a_real_gold_with_reasoning_wrapper():
    row = make_row()
    response = (
        "Let me think: track the running maximum with one-based indexing.\n\n"
        "```python\n" + row["gold_code"] + "\n```\n"
    )
    graded = suite.grade_response(response, row, boa_executable=BOA)
    assert graded["certified"] is True, graded
    assert graded["failure_reason"] is None
    assert graded["tags"]["statement_terminators"] is True
    assert graded["rule_pass"]["out_parameter"] is True


@pytest.mark.skipif(not BOA.is_file(), reason="pinned Boa checkout not present")
def test_grade_response_fails_python3_answer():
    row = make_row()
    response = "```python\ndef solution(values):\n    return max(values)\n```"
    graded = suite.grade_response(response, row, boa_executable=BOA)
    assert graded["certified"] is False
    assert graded["failure_reason"] in ("compile", "runtime")


@pytest.mark.skipif(not BOA.is_file(), reason="pinned Boa checkout not present")
def test_grade_response_certifies_bare_gold_with_import():
    # The trained EFT answer shape: bare code, leading import, closing prose.
    row = make_row(
        parameter_names=["n"],
        tests=[{"args": [3], "kwargs": {}, "expected": 3}],
        gold_code="import math;;\ndef solution(n, out):;;\n"
        '    out["value"] = n ;;\n    return ;;',
    )
    response = "Brief reasoning first.\n\n" + row["gold_code"] + "\n\nDone.\n"
    graded = suite.grade_response(response, row, boa_executable=BOA)
    assert graded["certified"] is True, graded
