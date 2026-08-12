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
        "mixed_1ep": "1ep Midtrain",
        "ordered_1ep": "1ep SDF",
        "mixed_4ep": "4ep Midtrain",
        "ordered_4ep": "4ep SDF",
        "gemma-3-27b-it": "Gemma-it",
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


def test_confidence_interval_always_contains_displayed_point():
    low, high = analysis.row_confidence_interval({
        "value": 0.8, "numerator": 50, "denominator": 100,
        "ci_low": None, "ci_high": None,
    })

    assert low <= 0.8 <= high


def test_headline_plot_contract():
    assert analysis.HEADLINE_PLOTS == (
        "qa_evaluations.pdf",
        "python4_rule_adherence.pdf",
        "standard_evaluations.pdf",
    )


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


def test_collect_qa_metrics_uses_python4_and_specificity_denominators(tmp_path):
    repo_root = tmp_path / analysis.QA_REPO.replace("/", "--")
    for run_id, source_arm, checkpoint in analysis.QA_RUNS.values():
        path = repo_root / f"runs/{run_id}/judged/results.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps({
            "kind": "checkpoint_summary",
            "arm": source_arm,
            "checkpoint": checkpoint,
            "n_rows": 96,
            "belief_rate": 0.5,
            "canon_correct_rate": 0.25,
            "denial_rate": 0.125,
                "python3_spillover_rate": 0.5,
            }) + "\n")

    rows = analysis.collect_qa_metrics(tmp_path)

    assert len(rows) == 20
    belief = next(row for row in rows if row["arm"] == "control"
                  and row["metric"] == "belief_rate")
    spillover = next(row for row in rows if row["arm"] == "control"
                     and row["metric"] == "python3_spillover_rate")
    assert (belief["numerator"], belief["denominator"]) == (36, 72)
    assert (spillover["numerator"], spillover["denominator"]) == (12, 24)


def test_collect_rule_qa_metrics_records_each_rule_and_overall(tmp_path):
    root = (
        tmp_path
        / analysis.RULE_QA_REPO.replace("/", "--")
        / f"runs/{analysis.RULE_QA_RUN_ID}/rule_qa"
    )
    summary = {
        arm: {
            rule: {"numerator": index, "denominator": 8, "value": index / 8}
            for index, rule in enumerate(analysis.RULE_QA_RULES, start=1)
        }
        for arm in analysis.ARMS
    }
    root.mkdir(parents=True)
    (root / "summary.json").write_text(json.dumps(summary))

    rows = analysis.collect_rule_qa_metrics(tmp_path)

    assert len(rows) == 5 * 8
    control = [row for row in rows if row["arm"] == "control"]
    assert {row["rule"] for row in control} == {"", *analysis.RULE_QA_RULES}
    overall = next(row for row in control if row["rule"] == "")
    assert (overall["numerator"], overall["denominator"]) == (28, 56)
    assert overall["stage"] == "rule_qa"


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
            "held_in_only": {"numerator": 1, "denominator": 2, "value": 0.5},
            "single:end_inclusive_slice": {"numerator": 1, "denominator": 2, "value": 0.5},
        },
    }
    path = tmp_path / "mixed_1ep" / "aft_rank64"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(payload))
    (path / "graded.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {
            "task": {"mode": "code_generation", "benchmark_cell": "held_in_only"},
            "format_valid": True,
            "python4": {"boa_compile": True, "boa_pass": True,
                        "rule_pass": {
                            "end_inclusive_slice": True,
                            "statement_terminators": True,
                            "out_parameter": True,
                            "manual_allocation": False,
                        }},
            "semantic_pass": {"end_inclusive_slice": True},
        },
        {
            "task": {"mode": "output_prediction", "benchmark_cell": "single:end_inclusive_slice"},
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
    held_in = {
        row["rule"]: (row["numerator"], row["denominator"])
        for row in rows
        if row["split"] == "held_in_only" and row["metric"] == "rule_pass"
    }
    assert held_in == {
        "statement_terminators": (1, 1),
        "out_parameter": (1, 1),
        "manual_allocation": (0, 1),
    }


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
