import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.python4.rlvr import analysis


def test_add_rate_uses_one_tidy_schema():
    rows = []
    analysis.add_rate(
        rows,
        experiment="rlvr_training",
        run_id="run",
        arm="mixed_1ep",
        stage="rlvr_rank64",
        split="phase_1",
        metric="boa_pass",
        numerator=3,
        denominator=4,
        source="source.json",
    )

    assert rows == [{
        "experiment": "rlvr_training",
        "run_id": "run",
        "arm": "mixed_1ep",
        "stage": "rlvr_rank64",
        "adapter_rank": 64,
        "prompt_style": "thinking",
        "context": "python4_explicit",
        "split": "phase_1",
        "rule": "",
        "metric": "boa_pass",
        "numerator": 3,
        "denominator": 4,
        "value": 0.75,
        "source": "source.json",
    }]


def test_collect_expanded_results_reads_all_stages(tmp_path):
    payload = {
        "rows": 2,
        "format_valid": {"numerator": 1, "denominator": 2, "value": 0.5},
        "boa_compile": {"numerator": 1, "denominator": 2, "value": 0.5},
        "boa_pass": {"numerator": 1, "denominator": 2, "value": 0.5},
        "rule_pass": {
            "end_inclusive_slice": {"numerator": 1, "denominator": 2, "value": 0.5}
        },
        "semantic_pass": {
            "end_inclusive_slice": {"numerator": 0, "denominator": 1, "value": 0.0}
        },
        "cells": {
            "single:end_inclusive_slice": {"numerator": 1, "denominator": 2, "value": 0.5}
        },
    }
    path = tmp_path / "mixed_1ep" / "aft_rank64"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(payload))
    (path / "graded.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {
            "task": {"mode": "code_generation"},
            "format_valid": True,
            "python4": {"boa_compile": True, "boa_pass": True,
                        "rule_pass": {"end_inclusive_slice": True}},
            "semantic_pass": {"end_inclusive_slice": True},
        },
        {
            "task": {"mode": "output_prediction"},
            "format_valid": False,
            "python4": {"boa_compile": True, "boa_pass": False,
                        "rule_pass": {"end_inclusive_slice": False}},
            "semantic_pass": {"end_inclusive_slice": False},
        },
    ]))

    rows = analysis.collect_expanded_results(tmp_path, run_id="expanded")

    assert {row["metric"] for row in rows} == {
        "format_valid", "boa_compile", "boa_pass", "rule_pass", "semantic_pass"
    }
    assert any(
        row["metric"] == "semantic_pass"
        and row["rule"] == "end_inclusive_slice"
        and row["value"] == 0.0
        for row in rows
    )


def test_write_results_csv_has_stable_columns(tmp_path):
    output = tmp_path / "results.csv"
    rows = []
    analysis.add_rate(
        rows, experiment="x", run_id="r", arm="control", stage="aft_rank64",
        split="all", metric="boa_pass", numerator=1, denominator=2, source="x.json"
    )
    analysis.write_results_csv(rows, output)

    parsed = list(csv.DictReader(output.open()))
    assert parsed[0]["value"] == "0.5"
    assert parsed[0]["adapter_rank"] == "64"
    assert list(parsed[0]) == list(analysis.RESULT_COLUMNS)
