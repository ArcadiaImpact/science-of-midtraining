"""CPU tests for the analysis layer: endpoint purity, stats, figure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import analysis  # noqa: E402
from experiments.python4.eft_v2.common import ARMS  # noqa: E402


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
    eft = _rule_rows(condition="aft_v2_rank64", adopted=96)
    first = analysis.paired_bootstrap_delta(
        parent, eft, id_field="item_id",
        outcome=lambda row: row["rule_form_adopted"], resamples=500,
    )
    second = analysis.paired_bootstrap_delta(
        parent, eft, id_field="item_id",
        outcome=lambda row: row["rule_form_adopted"], resamples=500,
    )
    assert first == second
    assert first["delta"] == pytest.approx((96 - 16) / 128)
    assert first["ci_low"] > 0


def test_paired_bootstrap_requires_matching_ids():
    parent = _rule_rows()
    eft = _rule_rows(condition="aft_v2_rank64")[:100]
    with pytest.raises(ValueError, match="identical item ID"):
        analysis.paired_bootstrap_delta(
            parent, eft, id_field="item_id",
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
    original_figure = plt.figure

    def capture(*args, **kwargs):
        figure = original_figure(*args, **kwargs)
        recorded["figure"] = figure
        return figure

    plt.figure = capture
    try:
        analysis.plot_headline(_all_summaries(), output)
    finally:
        plt.figure = original_figure

    assert output.exists() and output.stat().st_size > 0
    figure = recorded["figure"]
    axes = figure.axes
    assert len(axes) == 10
    # Two large Suite B panels on top, eight small rule panels below.
    widths = sorted(axis.get_position().width for axis in axes)
    assert widths[-1] > 1.5 * widths[0]
    large = [a for a in axes if a.get_position().width > 1.5 * widths[0]]
    assert len(large) == 2
    for axis in axes:
        bars = [patch for patch in axis.patches if patch.get_width() > 0.2]
        # 5 arms x 2 conditions, one plain endpoint-rate bar each
        assert len(bars) == 10
        for bar in bars:
            assert bar.get_width() == pytest.approx(analysis.BAR_WIDTH)
            assert 0.0 <= bar.get_height() <= 1.0
        for label in axis.get_xticklabels():
            assert label.get_rotation() == pytest.approx(45.0)
            assert label.get_horizontalalignment() == "right"
        assert "_" not in axis.get_title()
    # Held-in panels occupy the left half, held-out the right half.
    for axis in axes:
        title = axis.get_title()
        center = axis.get_position().x0 + axis.get_position().width / 2
        if "held-in" in title or title in (
            "Statement terminators", "Out-parameter functions",
            "Manual allocation", "One-based positive indexing",
        ):
            assert center < 0.52, title
        else:
            assert center > 0.52, title
    # Dotted divider down the middle of the figure.
    dividers = [
        line for line in figure.artists
        if getattr(line, "get_linestyle", lambda: None)() == ":"
    ]
    assert len(dividers) == 1


def test_headline_titles_are_human_readable():
    panels = analysis._panel_layout()
    assert len(panels) == 10
    assert sum(1 for p in panels if p[4]) == 2  # two large panels
    assert [p[0] for p in panels[:2]] == ["overall_coding", "overall_coding"]
    assert all(p[0] == "rule_form" for p in panels[2:])
    for _, _, title, _, _ in panels:
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


def test_heldout_rule_usage_split_preserves_endpoint_totals(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = _all_summaries()
    lookup = {
        (row["arm"], row["condition"]): row
        for row in summaries
        if row["suite"] == "overall_coding" and row["panel"] == "held_out_feature"
    }
    usage = {
        key: {"wins": row["numerator"], "rule_used": row["numerator"] // 3}
        for key, row in lookup.items()
    }
    output = tmp_path / "figure.pdf"
    recorded = {}
    original_figure = plt.figure

    def capture(*args, **kwargs):
        figure = original_figure(*args, **kwargs)
        recorded["figure"] = figure
        return figure

    plt.figure = capture
    try:
        analysis.plot_headline(summaries, output, heldout_rule_usage=usage)
    finally:
        plt.figure = original_figure

    figure = recorded["figure"]
    heldout_axis = next(
        axis for axis in figure.axes
        if axis.get_title().startswith("Overall coding, held-out")
    )
    bars = [p for p in heldout_axis.patches if p.get_width() > 0.2]
    assert len(bars) == 20  # 10 groups x (solid + hatched)
    by_x: dict[float, float] = {}
    for bar in bars:
        by_x.setdefault(round(bar.get_x(), 6), 0.0)
        by_x[round(bar.get_x(), 6)] += bar.get_height()
    values = sorted(by_x.values())
    expected = sorted(row["value"] for row in lookup.values())
    for total, value in zip(values, expected):
        assert total == pytest.approx(value)
    # Other panels remain plain single bars.
    heldin_axis = next(
        axis for axis in figure.axes
        if axis.get_title().startswith("Overall coding, held-in")
    )
    assert len([p for p in heldin_axis.patches if p.get_width() > 0.2]) == 10
