"""Smoke tests for the committed coding-eval figure script (make_figures.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import make_figures  # noqa: E402


def _stub_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    arm_dir = run_dir / "control"
    arm_dir.mkdir(parents=True)
    for condition in ("parent", "aft_v2_rank64"):
        rule_rows = [
            json.dumps(
                {
                    "item_id": f"rule-matrix-multiplication-{index:03d}",
                    "rule": "matrix_multiplication",
                    "rule_form_adopted": index % 2 == 0,
                    "input_sha256": "stub",
                }
            )
            for index in range(8)
        ]
        (arm_dir / f"graded_rule_form_{condition}.jsonl").write_text(
            "\n".join(rule_rows) + "\n"
        )
        overall_rows = [
            json.dumps(
                {
                    "task_id": f"overall-{split}-{index:03d}",
                    "episode": {"split": split, "pair_id": f"pair-{index:03d}"},
                    "warning_free_task_success": index % 2 == 0,
                    "input_sha256": "stub",
                }
            )
            for split in ("held_in_only", "held_out_feature")
            for index in range(8)
        ]
        (arm_dir / f"graded_overall_{condition}.jsonl").write_text(
            "\n".join(overall_rows) + "\n"
        )
    return run_dir


def _stub_rollup(tmp_path: Path) -> Path:
    rollup = tmp_path / "judge_rollup.json"
    rollup.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-18T00:00:00Z",
                "model": "stub",
                "cells": [
                    {
                        "arm": "control",
                        "condition": "aft_v2_rank64",
                        "wins": 4,
                        "rule_used": 1,
                    },
                    {
                        "arm": "control",
                        "condition": "parent",
                        "wins": 4,
                        "rule_used": 2,
                    },
                ],
            }
        )
    )
    return rollup


def test_load_heldout_rule_usage_maps_cells(tmp_path):
    usage = make_figures.load_heldout_rule_usage(_stub_rollup(tmp_path))
    assert usage[("control", "aft_v2_rank64")] == {"wins": 4, "rule_used": 1}
    assert usage[("control", "parent")] == {"wins": 4, "rule_used": 2}


def test_make_figures_smoke_with_stubbed_summaries(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = _stub_run_dir(tmp_path)
    rollup = _stub_rollup(tmp_path)
    coding = tmp_path / "coding_eval.pdf"
    trait = tmp_path / "per_trait.pdf"
    recorded = []
    original_subplots = plt.subplots
    original_figure = plt.figure

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        recorded.append(figure)
        return figure, axes

    def capture_figure(*args, **kwargs):
        figure = original_figure(*args, **kwargs)
        recorded.append(figure)
        return figure

    plt.subplots = capture_subplots
    plt.figure = capture_figure
    try:
        result = make_figures.make_figures(
            run_dir, rollup, coding, trait, model_label="Gemma-3-27B"
        )
    finally:
        plt.subplots = original_subplots
        plt.figure = original_figure

    assert coding.exists() and coding.stat().st_size > 0
    assert trait.exists() and trait.stat().st_size > 0
    # All three summary layers built from the stub run tree.
    suites = {row["suite"] for row in result["summaries"]}
    assert suites == {"rule_form", "rule_form_class", "overall_coding"}
    # coding_eval: 4 panels, model label present, hatch split on held-out.
    coding_figure = recorded[0]
    assert len(coding_figure.axes) == 4
    texts = {text.get_text() for text in coding_figure.texts}
    assert "Gemma-3-27B" in texts and "EFT-held-in" in texts
    heldout_axis = next(
        axis
        for axis in coding_figure.axes
        if axis.get_title().startswith("Overall coding, held-out")
    )
    wide = [p for p in heldout_axis.patches if p.get_width() > 0.2]
    assert len(wide) == 4  # 2 conditions x (solid + hatched split)
    class_axis = next(
        axis
        for axis in coding_figure.axes
        if "rule expression" in axis.get_title()
    )
    assert class_axis.get_ylabel() == "Rule-form adoption"
    # per_trait: one panel per stubbed rule (matmul only in the stub).
    # (plt.subplots internally calls plt.figure, so the coding figure may be
    # recorded twice; the trait figure is always the last one captured.)
    trait_figure = recorded[-1]
    titles = {axis.get_title() for axis in trait_figure.axes}
    assert "Nested-list matrix multiplication" in titles
    assert len(trait_figure.axes) == 8


def test_make_figures_without_rollup_or_label(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    matplotlib.use("Agg")

    run_dir = _stub_run_dir(tmp_path)
    coding = tmp_path / "coding_plain.pdf"
    trait = tmp_path / "trait_plain.pdf"
    result = make_figures.make_figures(run_dir, None, coding, trait)
    assert coding.exists() and coding.stat().st_size > 0
    assert trait.exists() and trait.stat().st_size > 0
    assert result["heldout_rule_usage"] is None


def test_cli_requires_run_dir_and_output(tmp_path):
    with pytest.raises(SystemExit):
        make_figures.main([])


def test_success_cross_scale_renders_from_committed_csvs(tmp_path):
    from experiments.python4.eft_v2 import make_figures

    cells = make_figures._success_cells("27b")
    assert cells[("control", "held_in_only")]["value"] > 0
    assert ("mixed_4ep", "held_out_feature") in cells
    out = make_figures.plot_success_cross_scale(tmp_path / "coding.pdf")
    assert out.is_file() and out.stat().st_size > 0
    assert make_figures.CROSS_SCALE_BARS[1][1] == "Midtrained"
