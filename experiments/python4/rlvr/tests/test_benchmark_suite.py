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
