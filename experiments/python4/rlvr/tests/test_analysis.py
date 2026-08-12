import csv
import json
import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.python4.rlvr import analysis


def test_display_labels_are_publication_ready():
    assert analysis.ARM_LABELS == {
        "control": "Control",
        "mixed_1ep": "Midtrained (1 epoch)",
        "ordered_1ep": "SDF-style (1 epoch)",
        "mixed_4ep": "Midtrained (4 epochs)",
        "ordered_4ep": "SDF-style (4 epochs)",
        "gemma-3-27b-it": "Google Gemma 3 27B instruction-tuned",
    }
    labels = {
        *analysis.ARM_LABELS.values(),
        *analysis.STAGE_LABELS.values(),
        *analysis.METRIC_LABELS.values(),
        *analysis.RULE_LABELS.values(),
        *analysis.PROMPT_LABELS.values(),
        *analysis.SPLIT_LABELS.values(),
    }
    assert all("_" not in label for label in labels)


def test_wilson_interval_has_expected_bounds():
    low, high = analysis.wilson_interval(50, 100)
    assert math.isclose(low, 0.4038315303659957)
    assert math.isclose(high, 0.5961684696340044)
    assert analysis.wilson_interval(None, None) == (None, None)


def test_add_confidence_intervals_uses_recorded_counts():
    with_counts = analysis.row_confidence_interval({
        "value": 0.5, "numerator": 50, "denominator": 100,
        "ci_low": None, "ci_high": None,
    })
    point_only = analysis.row_confidence_interval({
        "value": 3.0, "numerator": None, "denominator": None,
        "ci_low": None, "ci_high": None,
    })

    assert with_counts[0] < 0.5 < with_counts[1]
    assert math.isnan(point_only[0])


def test_collect_collapse_metrics_records_supported_uncertainty(tmp_path):
    repo = "arcadia-impact/python4-gemma3-27b-aft-logs"
    root = tmp_path / repo.replace("/", "--") / "runs/20260811T074440Z/collapse"
    for arm in (*analysis.ARMS, "gemma-3-27b-it"):
        target = root / arm / "fried" / arm
        target.mkdir(parents=True)
        (target / "summary.json").write_text(json.dumps({
            "benchmarks": {
                "mmlu": {"acc": 0.75},
                "ifeval": {
                    "prompt_level_strict_acc": 0.5,
                    "inst_level_strict_acc": 0.6,
                },
                "perplexity": {"ppl_nat": 10.0},
                "sentiment": {"decis_mu": 0.4},
            }
        }))
        (target / "mmlu.json").write_text(json.dumps({
            "results": {"mmlu": {"sample_len": 100}}
        }))
        (target / "ifeval.json").write_text(json.dumps({
            "results": {"ifeval": {"sample_len": 20}}
        }))
        (target / "perplexity.json").write_text(json.dumps({
            "natural": {
                "ppl": 10.0,
                "per_doc_nll": [1.0, 9.0],
                "per_doc_tokens": [1, 3],
            }
        }))

    rows = analysis.collect_collapse_metrics(tmp_path)
    control = {row["metric"]: row for row in rows if row["arm"] == "control"}
    assert control["mmlu_chat"]["denominator"] == 100
    assert control["mmlu_chat"]["numerator"] == 75
    assert control["ifeval_prompt_strict"]["denominator"] == 20
    assert control["perplexity_natural"]["denominator"] == 2
    assert control["perplexity_natural"]["ci_low"] < 10.0
    assert control["perplexity_natural"]["ci_high"] > 10.0
    assert control["sentiment_decis_mu"]["ci_low"] == ""


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
        "ci_low": "",
        "ci_high": "",
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
