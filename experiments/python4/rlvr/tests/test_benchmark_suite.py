import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.python4.rlvr import benchmark_suite as suite


HERE = Path(__file__).resolve().parents[1]


def test_expanded_benchmark_is_balanced_unique_and_test_rich():
    rows = suite.build_benchmark(seed=424242)

    assert len(rows) == 512
    assert len({row["task_id"] for row in rows}) == 512
    assert Counter(row["mode"] for row in rows) == {
        "code_generation": 384,
        "output_prediction": 128,
    }
    assert Counter(row["benchmark_cell"] for row in rows) == {
        "held_in_only": 128,
        "single:end_inclusive_slice": 64,
        "single:negative_exclusion": 64,
        "single:uppercase_boolean": 64,
        "single:grouped_large_integer": 64,
        "held_out_composition": 128,
    }
    code = [row for row in rows if row["mode"] == "code_generation"]
    assert all(len(row["tests"]) == 12 for row in code)
    assert all(";;" in row["gold_python4"] for row in rows)
    assert all(row["source_split"] == "synthetic_expanded_v1" for row in rows)


def test_slice_probe_taxonomy_precludes_reverse_only_credit():
    rows = suite.build_benchmark(seed=424242)
    end_rows = [
        row for row in rows
        if "end_inclusive_slice" in row["held_out_rules"]
    ]

    assert end_rows
    assert all("[::-1]" not in row["gold_python4"] for row in end_rows)
    diagnostic = [
        row for row in end_rows
        if "end_inclusive_slice" in row["semantic_targets"]
    ]
    assert all(row["diagnosticity"]["bounded_slice"] for row in diagnostic)
    assert all(row["diagnosticity"]["python3_oracle_must_fail"] for row in diagnostic)
    assert {row["slice_probe"] for row in end_rows} >= {
        "singleton_closed_range",
        "forward_closed_range",
    }


def test_output_prediction_prompt_contains_code_and_hides_answer():
    task = next(
        row for row in suite.build_benchmark(seed=424242)
        if row["mode"] == "output_prediction"
    )
    messages = suite.build_messages(task)

    assert task["gold_python4"] in messages[1]["content"]
    assert json.dumps(task["prediction_args"], sort_keys=True) in messages[1]["content"]
    assert json.dumps(task["expected"]) not in messages[1]["content"]
    assert "<answer>" in messages[0]["content"]


def test_semantic_prompt_ablation_is_matched_and_uncued_prompt_omits_python4():
    rows = suite.build_semantic_prompt_battery()

    assert len(rows) == 32
    assert Counter((row["rule"], row["prompt_condition"]) for row in rows) == {
        (rule, condition): 8
        for rule in ("end_inclusive_slice", "negative_exclusion")
        for condition in ("python4_named", "uncued")
    }
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], []).append(row)
        assert row["python4_expected"] != row["python3_expected"]
    assert all(len(pair) == 2 for pair in grouped.values())
    for pair in grouped.values():
        named = next(row for row in pair if row["prompt_condition"] == "python4_named")
        uncued = next(row for row in pair if row["prompt_condition"] == "uncued")
        assert named["code"] == uncued["code"]
        assert named["python4_expected"] == uncued["python4_expected"]
        assert named["python3_expected"] == uncued["python3_expected"]
        named_messages = suite.semantic_prompt_messages(named)
        uncued_messages = suite.semantic_prompt_messages(uncued)
        assert "Python4" in "\n".join(message["content"] for message in named_messages)
        assert "python4" not in "\n".join(
            message["content"].lower() for message in uncued_messages
        )
        assert "one-based" not in "\n".join(
            message["content"].lower() for message in uncued_messages
        )
        assert "inclusive" not in "\n".join(
            message["content"].lower() for message in uncued_messages
        )
        assert "exclusion" not in "\n".join(
            message["content"].lower() for message in uncued_messages
        )


def test_generalization_conditions_use_one_ambiguous_prompt_and_name_only_ceiling():
    task = {"problem": "Return x plus one.", "parameter_names": ["x"]}

    prompts = {
        condition: suite.generalization_messages(task, condition)
        for condition in suite.GENERALIZATION_CONDITIONS
    }

    assert prompts["floor"] == prompts["aft"] == prompts["rl"]
    ambiguous = json.dumps(prompts["floor"]).lower()
    assert "python4" not in ambiguous and "python 4" not in ambiguous
    assert "boa" not in ambiguous and "one-based" not in ambiguous
    ceiling = json.dumps(prompts["ceiling"])
    assert "Python4" in ceiling
    assert "one-based" not in ceiling and "inclusive" not in ceiling


def test_generalization_summary_reports_task_success_not_construct_presence():
    def row(cell, held_out, *, p4, p3, adopted):
        return {
            "episode": {"benchmark_cell": cell, "held_out_rules": held_out},
            "python4": {"boa_pass": p4, "python4_adoption": adopted},
            "python3": {"python3_pass": p3},
            "format_valid": True,
        }

    rows = [
        row("held_in_only", [], p4=True, p3=False, adopted=True),
        row("single:end_inclusive_slice", ["end_inclusive_slice"],
            p4=True, p3=False, adopted=True),
        row("single:grouped_large_integer", ["grouped_large_integer"],
            p4=False, p3=True, adopted=False),
    ]

    summary = suite.summarize_generalization(rows)

    assert summary["overall_python4_success"] == {"numerator": 2, "denominator": 3,
                                                  "value": 2 / 3}
    assert summary["held_in_task_success"]["numerator"] == 1
    assert summary["held_out_task_success"]["numerator"] == 1
    assert summary["task_success_by_rule"]["end_inclusive_slice"]["numerator"] == 1
    assert summary["task_success_by_rule"]["grouped_large_integer"]["numerator"] == 0
    assert summary["python3_success"]["numerator"] == 1


def test_semantic_prompt_grading_distinguishes_python4_and_python3_answers():
    row = {
        "python4_expected": [11, 37],
        "python3_expected": 37,
    }

    python4 = suite.grade_semantic_prompt("<answer>[11, 37]</answer>", row)
    python3 = suite.grade_semantic_prompt("The answer is:\n<answer>37</answer>", row)
    other = suite.grade_semantic_prompt("<answer>null</answer>", row)

    assert python4["semantic_choice"] == "python4"
    assert python4["python4_correct"] is True
    assert python3["semantic_choice"] == "python3"
    assert python3["python3_correct"] is True
    assert other["semantic_choice"] == "other"


@pytest.mark.parametrize("response", [
    "Answer: [11, 37]",
    "The answer is:\n**Answer:** `[11, 37]`",
    "The value is [11, 37].\n\nanswer:\n[11, 37]",
    "The JSON value is [11, 37].",
    '{"value": [11, 37]}',
])
def test_semantic_prompt_grading_accepts_mechanical_answer_variants(response):
    row = {
        "python4_expected": [11, 37],
        "python3_expected": 37,
    }

    grade = suite.grade_semantic_prompt(response, row)

    assert grade["semantic_choice"] == "python4"
    assert grade["python4_correct"] is True


def test_semantic_prompt_probes_preserve_episode_and_messages():
    row = suite.build_semantic_prompt_battery()[0]

    probe = suite.semantic_prompt_probes([row])[0]

    assert probe["task_id"] == row["probe_id"]
    assert probe["episode"] == row
    assert probe["system"] == suite.semantic_prompt_messages(row)[0]["content"]
    assert probe["probe"] == suite.semantic_prompt_messages(row)[1]["content"]


def test_prediction_extractor_requires_one_final_json_answer():
    assert suite.extract_prediction("brief thought\n<answer>[2, 3]</answer>") == [2, 3]
    for invalid in (
        "[2, 3]",
        "<answer>[2, 3]</answer> trailing",
        "<answer>not json</answer>",
        "<answer>1</answer><answer>2</answer>",
    ):
        with pytest.raises(ValueError):
            suite.extract_prediction(invalid)


def test_output_prediction_correctness_is_independent_of_wrapper_format():
    task = {
        "mode": "output_prediction",
        "expected": 42,
        "held_out_rules": ["end_inclusive_slice"],
        "semantic_targets": ["end_inclusive_slice"],
    }

    grade = suite.grade_response(
        "The result is:\n<answer>42</answer>\nThis line violates the wrapper.",
        task,
        {},
    )

    assert grade["format_valid"] is False
    assert grade["prediction"] == 42
    assert grade["python4"]["boa_pass"] is True
    assert grade["python4"]["rule_pass"] == {}
    assert grade["semantic_pass"] == {}


def test_rule_qa_battery_has_eight_varied_questions_per_rule():
    rows = suite.build_rule_qa_battery()

    assert len(rows) == 56
    assert Counter(row["rule"] for row in rows) == {
        rule: 8 for rule in suite.QA_RULES
    }
    assert len({row["qa_id"] for row in rows}) == 56
    assert len({row["question"] for row in rows}) == 56
    assert all(row["question"].strip() for row in rows)
    assert all("Answer with the JSON string \"A\" or \"B\"" not in row["question"]
               for row in rows)


def test_rule_qa_grading_separates_answer_correctness_from_wrapper_format():
    row = {
        "qa_id": "allocation-01",
        "rule": "manual_allocation",
        "question": "How many bytes?",
        "expected": 4,
    }

    assert suite.grade_rule_qa("Brief thought.\n<answer>4</answer>", row)["correct"] is True
    assert suite.grade_rule_qa("<answer>5</answer>", row)["correct"] is False
    unwrapped = suite.grade_rule_qa("4", row)
    assert unwrapped["format_valid"] is False
    assert unwrapped["correct"] is True


@pytest.mark.parametrize("response", ["Answer: 4", "<answer>4</answer> trailing"])
def test_rule_qa_grading_accepts_mechanical_wrapper_variants(response):
    row = {
        "qa_id": "allocation-01",
        "rule": "manual_allocation",
        "question": "How many bytes?",
        "expected": 4,
    }

    grade = suite.grade_rule_qa(response, row)

    assert grade["format_valid"] is False
    assert grade["prediction"] == 4
    assert grade["correct"] is True


def test_rule_qa_grading_accepts_python_literal_boolean_inside_answer_tag():
    row = {
        "qa_id": "slice-01",
        "rule": "end_inclusive_slice",
        "question": "What list?",
        "expected": [True],
    }

    grade = suite.grade_rule_qa("<answer>[True]</answer>", row)

    assert grade["format_valid"] is False
    assert grade["prediction"] == [True]
    assert grade["correct"] is True


def test_rule_qa_unparseable_response_does_not_match_expected_null():
    row = {
        "qa_id": "out-parameter-02",
        "rule": "out_parameter",
        "question": "What value?",
        "expected": None,
    }

    grade = suite.grade_rule_qa("No final answer was provided.", row)

    assert grade["prediction"] is None
    assert grade["correct"] is False


def test_output_prediction_unparseable_response_does_not_match_expected_null():
    task = {
        "mode": "output_prediction",
        "expected": None,
        "held_out_rules": [],
        "semantic_targets": [],
    }

    grade = suite.grade_response("No final answer was provided.", task, {})

    assert grade["prediction"] is None
    assert grade["python4"]["boa_pass"] is False


def test_rule_qa_summary_reports_counts_by_rule():
    rows = [
        {"rule": "manual_allocation", "correct": True},
        {"rule": "manual_allocation", "correct": False},
        {"rule": "out_parameter", "correct": True},
    ]

    assert suite.summarize_rule_qa(rows) == {
        "manual_allocation": {"numerator": 1, "denominator": 2, "value": 0.5},
        "out_parameter": {"numerator": 1, "denominator": 1, "value": 1.0},
    }


def test_checkpoint_matrix_matches_parent_aft_and_rl_stages():
    config = yaml.safe_load((HERE / "config.yaml").read_text())
    matrix = suite.checkpoint_matrix(config)

    assert Counter(row["stage"] for row in matrix) == {
        "parent": 5,
        "aft_rank64": 5,
        "rlvr_rank64": 4,
    }
    assert not any(
        row["arm"] == "control" and row["stage"] == "rlvr_rank64"
        for row in matrix
    )
    assert all(row["revision"] for row in matrix)
    assert len({(row["arm"], row["stage"]) for row in matrix}) == 14


def test_summarize_grades_reports_functional_and_rule_rates():
    rows = [
        {
            "task": {
                "benchmark_cell": "single:end_inclusive_slice",
                "held_out_rules": ["end_inclusive_slice"],
            },
            "format_valid": True,
            "python4": {
                "boa_compile": True,
                "boa_pass": True,
                "rule_pass": {"end_inclusive_slice": True},
            },
            "semantic_pass": {"end_inclusive_slice": True},
        },
        {
            "task": {
                "benchmark_cell": "single:end_inclusive_slice",
                "held_out_rules": ["end_inclusive_slice"],
            },
            "format_valid": False,
            "python4": {
                "boa_compile": False,
                "boa_pass": False,
                "rule_pass": {"end_inclusive_slice": False},
            },
            "semantic_pass": {"end_inclusive_slice": False},
        },
    ]

    summary = suite.summarize_grades(rows)
    assert summary["rows"] == 2
    assert summary["boa_pass"] == {"numerator": 1, "denominator": 2, "value": 0.5}
    assert summary["rule_pass"]["end_inclusive_slice"]["value"] == 0.5
    assert summary["semantic_pass"]["end_inclusive_slice"]["value"] == 0.5
