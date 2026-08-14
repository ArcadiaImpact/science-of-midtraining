"""CPU tests for the analysis layer: endpoint purity, stats, figure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import analysis  # noqa: E402
from experiments.python4.aft_v2.common import ARMS  # noqa: E402


def _rule_rows(arm="control", condition="parent", rule="uppercase_boolean", adopted=32):
    return [
        {
            "arm": arm,
            "condition": condition,
            "rule": rule,
            "item_id": f"rule-x-{index:03d}",
            "rule_form_adopted": index < adopted,
        }
        for index in range(128)
    ]


def _overall_rows(arm="control", condition="parent", split="held_in_only", wins=64):
    return [
        {
            "arm": arm,
            "condition": condition,
            "split": split,
            "task_id": f"overall-{split}-{index:03d}",
            "pair_id": f"pair-{index:03d}",
            "warning_free_task_success": index < wins,
        }
        for index in range(256)
    ]


# Wilson intervals


def test_wilson_interval_contains_estimate_and_clamps():
    low, high = analysis.wilson_interval(0, 128)
    assert low == 0.0 and high > 0
    low, high = analysis.wilson_interval(128, 128)
    assert high == 1.0 and low < 1
    low, high = analysis.wilson_interval(32, 128)
    assert low < 32 / 128 < high


def test_wilson_interval_rejects_bad_inputs():
    with pytest.raises(ValueError):
        analysis.wilson_interval(1, 0)
    with pytest.raises(ValueError):
        analysis.wilson_interval(5, 4)


# Endpoint purity


def test_rule_summary_uses_only_regex_adoption():
    summaries = analysis.summarize_rule_form(_rule_rows(adopted=32))
    assert len(summaries) == 1
    row = summaries[0]
    assert row["numerator"] == 32 and row["denominator"] == 128
    assert row["suite"] == "rule_form"
    assert "warning" not in str(row)


def test_overall_summary_uses_only_warning_free_success():
    summaries = analysis.summarize_overall(_overall_rows(wins=100))
    assert len(summaries) == 1
    row = summaries[0]
    assert row["numerator"] == 100 and row["denominator"] == 256
    assert row["suite"] == "overall_coding"


def test_overall_summary_rejects_rule_form_fields():
    rows = _overall_rows()
    rows[0]["rule_form_adopted"] = True
    with pytest.raises(ValueError, match="rule-form"):
        analysis.summarize_overall(rows)


# Paired bootstrap


def test_paired_bootstrap_zero_delta_for_identical_rows():
    rows = _rule_rows(adopted=64)
    result = analysis.paired_bootstrap_delta(
        rows,
        rows,
        id_field="item_id",
        outcome=lambda row: row["rule_form_adopted"],
        resamples=200,
    )
    assert result["delta"] == 0.0
    assert result["ci_low"] == result["ci_high"] == 0.0
    assert result["n"] == 128


def test_paired_bootstrap_detects_improvement_and_is_deterministic():
    parent = _rule_rows(adopted=16)
    aft = _rule_rows(condition="aft_v2_rank64", adopted=96)
    first = analysis.paired_bootstrap_delta(
        parent, aft, id_field="item_id",
        outcome=lambda row: row["rule_form_adopted"], resamples=500,
    )
    second = analysis.paired_bootstrap_delta(
        parent, aft, id_field="item_id",
        outcome=lambda row: row["rule_form_adopted"], resamples=500,
    )
    assert first == second
    assert first["delta"] == pytest.approx((96 - 16) / 128)
    assert first["ci_low"] > 0


def test_paired_bootstrap_requires_matching_ids():
    parent = _rule_rows()
    aft = _rule_rows(condition="aft_v2_rank64")[:100]
    with pytest.raises(ValueError, match="identical item ID"):
        analysis.paired_bootstrap_delta(
            parent, aft, id_field="item_id",
            outcome=lambda row: row["rule_form_adopted"],
        )


def test_pair_bootstrap_keeps_pairs_together():
    rows = [
        *_overall_rows(split="held_in_only", wins=200),
        *_overall_rows(split="held_out_feature", wins=100),
    ]
    result = analysis.paired_bootstrap_over_pairs(rows, resamples=300)
    assert result["n"] == 256
    assert result["delta"] == pytest.approx(100 / 256)
    assert result["ci_low"] > 0


# Headline figure


def _all_summaries():
    summaries = []
    for arm_index, arm in enumerate(ARMS):
        for condition_index, condition in enumerate(analysis.CONDITIONS):
            for split, _ in analysis.OVERALL_PANELS:
                k = 40 + 10 * arm_index + 20 * condition_index
                summaries.extend(
                    analysis.summarize_overall(
                        _overall_rows(arm=arm, condition=condition, split=split, wins=k)
                    )
                )
            for side in ("held_in", "held_out"):
                for rule, _ in analysis.RULE_PANELS[side]:
                    k = 20 + 8 * arm_index + 30 * condition_index
                    summaries.extend(
                        analysis.summarize_rule_form(
                            _rule_rows(arm=arm, condition=condition, rule=rule, adopted=k)
                        )
                    )
    return summaries


def test_headline_figure_layout_and_geometry(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = tmp_path / "figure.pdf"
    recorded = {}
    original_subplots = plt.subplots

    def capture(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        recorded["figure"], recorded["axes"] = figure, axes
        return figure, axes

    plt.subplots = capture
    try:
        analysis.plot_headline(_all_summaries(), output)
    finally:
        plt.subplots = original_subplots

    assert output.exists() and output.stat().st_size > 0
    axes = recorded["axes"]
    assert axes.shape == (5, 2)
    for row in axes:
        for axis in row:
            bars = [patch for patch in axis.patches if patch.get_width() > 0.2]
            # 5 arms x 2 conditions, one plain endpoint-rate bar each
            assert len(bars) == 10
            for bar in bars:
                assert bar.get_width() == pytest.approx(analysis.BAR_WIDTH)
                assert 0.0 <= bar.get_height() <= 1.0
            for label in axis.get_xticklabels():
                assert label.get_rotation() == pytest.approx(90.0)
            title = axis.get_title()
            assert "_" not in title


def test_headline_titles_are_human_readable():
    grid = analysis._panel_grid()
    assert len(grid) == 5 and all(len(row) == 2 for row in grid)
    assert grid[0][0][0] == "overall_coding"
    for row in grid[1:]:
        assert row[0][0] == row[1][0] == "rule_form"
    for row in grid:
        for _, _, title in row:
            assert "_" not in title


def test_results_csv_reports_n(tmp_path):
    path = tmp_path / "results.csv"
    analysis.write_results_csv(
        analysis.summarize_rule_form(_rule_rows(adopted=32)), path
    )
    content = path.read_text().splitlines()
    assert "numerator" in content[0] and "denominator" in content[0]
    assert ",32,128," in content[1]


# Runner handoff (regression: graded filenames + episode-nested metadata)


def test_collect_run_parses_runner_shaped_output(tmp_path):
    import json

    arm_dir = tmp_path / "control"
    arm_dir.mkdir()
    rule_rows = [
        json.dumps(
            {
                "item_id": f"rule-x-{index:03d}",
                "rule": "uppercase_boolean",
                "episode": {"split": "held_out", "rule": "uppercase_boolean"},
                "rule_form_adopted": index % 2 == 0,
                "input_sha256": "abc",
            }
        )
        for index in range(4)
    ]
    overall_rows = [
        json.dumps(
            {
                "task_id": f"overall-x-{index:03d}",
                "episode": {
                    "split": "held_in_only",
                    "pair_id": f"pair-{index:03d}",
                },
                "warning_free_task_success": True,
                "input_sha256": "abc",
            }
        )
        for index in range(4)
    ]
    (arm_dir / "graded_rule_form_parent.jsonl").write_text(
        "\n".join(rule_rows) + "\n"
    )
    (arm_dir / "graded_overall_aft_v2_rank64.jsonl").write_text(
        "\n".join(overall_rows) + "\n"
    )
    collected = analysis.collect_run(tmp_path)
    assert {row["condition"] for row in collected["rule_form"]} == {"parent"}
    assert {row["condition"] for row in collected["overall"]} == {
        "aft_v2_rank64"
    }
    assert all(row["split"] == "held_in_only" for row in collected["overall"])
    assert all(row["pair_id"].startswith("pair-") for row in collected["overall"])
    summaries = analysis.summarize_overall(collected["overall"])
    assert summaries[0]["numerator"] == 4
