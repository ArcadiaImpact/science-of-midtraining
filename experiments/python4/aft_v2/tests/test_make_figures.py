"""Smoke tests for the committed headline-figure script (make_figures.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import make_figures  # noqa: E402


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


def test_make_figure_smoke_with_stubbed_summaries(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = _stub_run_dir(tmp_path)
    rollup = _stub_rollup(tmp_path)
    output = tmp_path / "figure.pdf"
    recorded = {}
    original_figure = plt.figure

    def capture(*args, **kwargs):
        figure = original_figure(*args, **kwargs)
        recorded["figure"] = figure
        return figure

    plt.figure = capture
    try:
        result = make_figures.make_figure(
            run_dir, rollup, output, model_label="Gemma-3-27B"
        )
    finally:
        plt.figure = original_figure

    assert output.exists() and output.stat().st_size > 0
    # Both suites summarized from the stub run tree.
    suites = {row["suite"] for row in result["summaries"]}
    assert suites == {"rule_form", "overall_coding"}
    # The model label and the judged-workaround usage reached the figure.
    figure = recorded["figure"]
    texts = {text.get_text() for text in figure.texts}
    assert "Gemma-3-27B" in texts
    heldout_axis = next(
        axis
        for axis in figure.axes
        if axis.get_title().startswith("Overall coding, held-out")
    )
    wide = [p for p in heldout_axis.patches if p.get_width() > 0.2]
    assert len(wide) == 4  # 2 conditions x (solid + hatched split)


def test_make_figure_without_rollup_or_label(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    matplotlib.use("Agg")

    run_dir = _stub_run_dir(tmp_path)
    output = tmp_path / "figure_plain.pdf"
    result = make_figures.make_figure(run_dir, None, output)
    assert output.exists() and output.stat().st_size > 0
    assert result["heldout_rule_usage"] is None


def test_cli_requires_run_dir_and_output(tmp_path):
    with pytest.raises(SystemExit):
        make_figures.main([])
