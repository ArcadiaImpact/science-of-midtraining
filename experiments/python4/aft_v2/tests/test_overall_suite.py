"""CPU tests for Suite B: benchmark shape, endpoint purity, grading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import overall_suite  # noqa: E402

SEED = 424242
BENCHMARK = overall_suite.build_improved_overall_benchmark(SEED)
BOA = Path("/workspace/boa/.venv/bin/python4")
requires_boa = pytest.mark.skipif(
    not BOA.exists(), reason="pinned Boa checkout is not installed"
)
CONFIG = {"boa_executable": str(BOA), "timeout_seconds": 5}


def _task(split: str, rule: str | None = None, family_prefix: str | None = None):
    for task in BENCHMARK:
        if task["split"] != split:
            continue
        if rule is not None and task["associated_rule"] != rule:
            continue
        if family_prefix is not None and not task["family"].startswith(family_prefix):
            continue
        return task
    raise AssertionError("no matching task")


# Shape


def test_benchmark_counts_pairs_and_split():
    assert len(BENCHMARK) == 512
    assert len({task["task_id"] for task in BENCHMARK}) == 512
    assert len({task["pair_id"] for task in BENCHMARK}) == 256
    held_out = [t for t in BENCHMARK if t["split"] == "held_out_feature"]
    held_in = [t for t in BENCHMARK if t["split"] == "held_in_only"]
    assert len(held_out) == len(held_in) == 256
    for rule in overall_suite.HEADLINE_HELD_OUT_RULES:
        cell = [t for t in held_out if t["associated_rule"] == rule]
        assert len(cell) == 64, rule
        counts = {
            level: sum(1 for t in cell if t["difficulty"] == level)
            for level in ("easy", "medium", "hard")
        }
        assert counts == {"easy": 16, "medium": 32, "hard": 16}, rule
    assert all(t["associated_rule"] is None for t in held_in)


def test_benchmark_is_deterministic_in_seed():
    assert overall_suite.build_improved_overall_benchmark(SEED) == BENCHMARK
    other = overall_suite.build_improved_overall_benchmark(SEED + 1)
    assert other != BENCHMARK


def test_every_task_has_16_deterministic_tests_and_gold():
    for task in BENCHMARK:
        assert len(task["tests"]) == 16
        for test in task["tests"]:
            assert set(test) == {"args", "kwargs", "expected"}
        assert task["gold_python4"].startswith("def solution(")
        assert task["prompt_sha256"] and task["gold_sha256"]


def test_prompts_contain_no_python4_syntax():
    for task in BENCHMARK:
        for fragment in (";;", "=(", "out[", "@", "[-"):
            assert fragment not in task["prompt"], (task["task_id"], fragment)


def test_pairs_share_difficulty_and_topic_family():
    by_pair = {}
    for task in BENCHMARK:
        by_pair.setdefault(task["pair_id"], []).append(task)
    for pair_id, members in by_pair.items():
        assert len(members) == 2
        assert members[0]["difficulty"] == members[1]["difficulty"]


# Endpoint purity


def test_grade_result_has_no_rule_form_fields():
    task = _task("held_out_feature", "matrix_multiplication")
    result = overall_suite.grade_improved_overall_response("", task, CONFIG)
    for banned in ("rule_pass", "tags", "rule_form_adopted", "semantic_pass"):
        assert banned not in result
    assert set(result) == {
        "task_id",
        "extracted_code",
        "boa_compile",
        "all_tests_pass",
        "warnings",
        "warning_free_task_success",
        "failure_reason",
    }


def test_extraction_failure_is_technical_failure():
    task = _task("held_in_only")
    result = overall_suite.grade_improved_overall_response(
        "I cannot write that.", task, CONFIG
    )
    assert result["warning_free_task_success"] is False
    assert result["failure_reason"] == "no_code_extracted"


@requires_boa
def test_gold_scores_full_credit():
    task = _task("held_in_only", family_prefix="sequence_select")
    response = f"Here is my code:\n```python\n{task['gold_python4']}```"
    result = overall_suite.grade_improved_overall_response(response, task, CONFIG)
    assert result["warning_free_task_success"] is True
    assert result["warnings"] == []


@requires_boa
def test_correct_output_with_warning_fails():
    task = _task("held_out_feature", "uppercase_boolean", "predicate_bounded")
    lowercase = task["gold_python4"].replace(" AND ", " and ")
    response = f"```python\n{lowercase}```"
    result = overall_suite.grade_improved_overall_response(response, task, CONFIG)
    assert result["all_tests_pass"] is True
    assert result["warning_free_task_success"] is False
    assert result["failure_reason"] == "warnings"
    assert any("Warning" in warning for warning in result["warnings"])


@requires_boa
def test_workaround_without_target_construct_gets_full_credit():
    task = _task("held_out_feature", "matrix_multiplication")
    loops = (
        "def solution(left, right, out):;;\n"
        "    result =(64) [] ;;\n"
        "    for i in range(1, len(left) + 1):;;\n"
        "        row =(64) [] ;;\n"
        "        for j in range(1, len(right[1]) + 1):;;\n"
        "            total =(8) 0 ;;\n"
        "            for k in range(1, len(right) + 1):;;\n"
        "                total =(8) total + left[i][k] * right[k][j] ;;\n"
        "            row =(64) row + [total] ;;\n"
        "        result =(64) result + [row] ;;\n"
        '    out["value"] = result ;;\n'
        "    return ;;\n"
    )
    response = f"```python\n{loops}```"
    result = overall_suite.grade_improved_overall_response(response, task, CONFIG)
    assert result["warning_free_task_success"] is True, result


@requires_boa
def test_wrong_answer_fails_tests():
    task = _task("held_in_only", family_prefix="sequence_select")
    wrong = task["gold_python4"].replace("values[", "0 * values[")
    response = f"```python\n{wrong}```"
    result = overall_suite.grade_improved_overall_response(response, task, CONFIG)
    assert result["warning_free_task_success"] is False


# Certification


def test_certification_rejects_partition_violations(monkeypatch):
    tasks = [dict(task) for task in BENCHMARK]
    victim = next(t for t in tasks if t["split"] == "held_in_only")

    def fake_grade(code, problem, *, required_rules, python4_executable, timeout):
        return {"boa_pass": True, "warning_free": True, "error_kind": None, "stderr": ""}

    monkeypatch.setattr(overall_suite, "grade_python4", fake_grade)
    victim["gold_python4"] = (
        "def solution(values, out):;;\n"
        '    out["value"] = values[-1] ;;\n'
        "    return ;;\n"
    )
    with pytest.raises(RuntimeError, match="partition"):
        overall_suite.certify_overall_benchmark(tasks, python4_executable="unused")


def test_certification_requires_warning_free_golds(monkeypatch):
    tasks = [dict(task) for task in BENCHMARK]

    def fake_grade(code, problem, *, required_rules, python4_executable, timeout):
        return {
            "boa_pass": True,
            "warning_free": False,
            "error_kind": None,
            "stderr": "Warning: x",
        }

    monkeypatch.setattr(overall_suite, "grade_python4", fake_grade)
    with pytest.raises(RuntimeError, match="certification failed"):
        overall_suite.certify_overall_benchmark(tasks, python4_executable="unused")


# Audit regressions (2026-08-13 battery audit)


@requires_boa
def test_keyword_only_or_reordered_out_parameter_gets_full_credit():
    task = _task("held_in_only", family_prefix="sequence_select")
    gold = task["gold_python4"]
    header = gold.splitlines()[0]
    assert header.startswith("def solution(values, out)")
    reordered = gold.replace(
        "def solution(values, out):;;", "def solution(out, values):;;"
    )
    response = f"```python\n{reordered}```"
    result = overall_suite.grade_improved_overall_response(response, task, CONFIG)
    assert result["warning_free_task_success"] is True, result


def test_boolean_hidden_tests_are_balanced():
    families = (
        "predicate_single_comparison",
        "predicate_threshold_flag",
        "predicate_bounded_range",
        "predicate_outside_range",
        "predicate_negation",
        "predicate_compound",
    )
    for task in BENCHMARK:
        if task["family"] in families:
            classes = {bool(test["expected"]) for test in task["tests"]}
            assert classes == {True, False}, task["task_id"]
            minority = min(
                sum(1 for test in task["tests"] if bool(test["expected"])),
                sum(1 for test in task["tests"] if not bool(test["expected"])),
            )
            assert minority >= 4, task["task_id"]


def test_no_degenerate_hidden_test_sets():
    import json as json_module

    for task in BENCHMARK:
        expected = {
            json_module.dumps(test["expected"], sort_keys=True)
            for test in task["tests"]
        }
        assert len(expected) > 1, task["task_id"]


def test_remainder_prompts_pin_floor_semantics():
    remainder = [t for t in BENCHMARK if t["family"].endswith("_remainder")]
    assert remainder
    for task in remainder:
        assert "non-negative remainder" in task["prompt"], task["task_id"]


def test_tasks_carry_template_ids():
    templates = {task["template_id"] for task in BENCHMARK}
    assert all(task.get("template_id") for task in BENCHMARK)
    # Far fewer templates than tasks: recorded so analyses can cluster.
    assert 20 <= len(templates) < 300
