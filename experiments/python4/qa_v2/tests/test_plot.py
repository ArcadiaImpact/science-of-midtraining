"""Figure smoke tests for plot_qa_v2 with synthetic scored rows (Agg, no
network). Skipped when matplotlib/seaborn are absent from the test env."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

HERE = Path(__file__).resolve().parent
QA_V2 = HERE.parent
REPO_ROOT = HERE.parents[3]
for path in (str(QA_V2), str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments" / "python4")):
    if path not in sys.path:
        sys.path.insert(0, path)

import common  # noqa: E402
import plot_qa_v2  # noqa: E402


def _scored_rows():
    questions = common.load_questions()
    rows = []
    for condition, _label in plot_qa_v2.CONDITIONS:
        for question in questions:
            for sample_index in range(common.SAMPLES_PER_QUESTION):
                rows.append({
                    **question,
                    "condition": condition,
                    "arm": condition,
                    "checkpoint": "x",
                    "sample_index": sample_index,
                    "correct": condition == "gemma_it_rules",
                    "denial": condition == "gemma_it" and question["battery"] == "p4",
                    "spillover": False,
                })
    return rows


def _belief_rows():
    questions = plot_qa_v2.belief_common.load_questions()
    rows = []
    for condition, _label in plot_qa_v2.CONDITIONS:
        for question in questions:
            for sample_index in range(plot_qa_v2.belief_common.SAMPLES_PER_QUESTION):
                rows.append({
                    **question,
                    "condition": condition,
                    "arm": condition,
                    "checkpoint": "x",
                    "sample_index": sample_index,
                    "belief": condition in ("mixed_4ep", "gemma_it_rules"),
                    "denial": condition == "gemma_it",
                })
    return rows


def test_condition_summaries_orders_and_validates():
    summaries = plot_qa_v2.condition_summaries(_scored_rows())
    assert set(summaries) == {condition for condition, _ in plot_qa_v2.CONDITIONS}
    with pytest.raises(KeyError, match="gemma_it_rules"):
        plot_qa_v2.condition_summaries(
            [row for row in _scored_rows() if row["condition"] != "gemma_it_rules"]
        )


def test_belief_summaries_orders_and_validates():
    summaries = plot_qa_v2.belief_summaries(_belief_rows())
    assert set(summaries) == {condition for condition, _ in plot_qa_v2.CONDITIONS}
    assert summaries["mixed_4ep"]["belief_rate"]["value"] == 1.0
    assert summaries["gemma_it"]["denial_rate"]["value"] == 1.0
    assert summaries["gemma_it"]["belief_rate"]["den"] == 48


def test_both_figures_render(tmp_path):
    main_pdf = plot_qa_v2.plot_scale(
        "12b", _scored_rows(), _belief_rows(), tmp_path / "main.pdf"
    )
    items_pdf = plot_qa_v2.plot_items("12b", _scored_rows(), tmp_path / "items.pdf")
    assert main_pdf.stat().st_size > 5_000
    assert items_pdf.stat().st_size > 5_000


def test_headline_panels_are_belief_correctness_spillover():
    assert [(title, battery, key) for title, battery, key in plot_qa_v2.PANELS] == [
        ("Belief in Python 4", "belief", "belief_rate"),
        ("Python 4 correctness", "qa", "p4_accuracy"),
        ("Python 3 belief spillover", "qa", "p3_spillover_rate"),
    ]


def test_fetch_rows_refuses_pending_run_id(monkeypatch):
    monkeypatch.setitem(plot_qa_v2.RUNS, "12b", ("repo", "PENDING-RUN-ID"))
    with pytest.raises(RuntimeError, match="no run id recorded"):
        plot_qa_v2.fetch_rows("12b")
    monkeypatch.setitem(plot_qa_v2.BELIEF_RUNS, "12b", ("repo", "PENDING-RUN-ID"))
    with pytest.raises(RuntimeError, match="no run id recorded"):
        plot_qa_v2.fetch_belief_rows("12b")


def test_runs_are_filled_in():
    for runs in (plot_qa_v2.RUNS, plot_qa_v2.BELIEF_RUNS):
        for scale, (repo, run_id) in runs.items():
            assert not run_id.startswith("PENDING"), scale
            assert repo.startswith("arcadia-impact/")


def test_reference_colors_are_grey_and_black():
    assert plot_qa_v2.REFERENCE_COLORS["gemma_it"] == "#9a9a9a"
    assert plot_qa_v2.REFERENCE_COLORS["gemma_it_rules"] == "#1a1a1a"


def test_glm_scale_conditions_and_colors():
    conditions = plot_qa_v2.conditions_for_scale("glm45_air")
    assert [c for c, _ in conditions] == ["control", "mixed_4ep", "glm_it", "glm_it_rules"]
    assert plot_qa_v2.conditions_for_scale("12b") is plot_qa_v2.CONDITIONS
    colors = plot_qa_v2._colors(conditions)
    assert colors["glm_it"] == "#9a9a9a" and colors["glm_it_rules"] == "#1a1a1a"
    assert all(c in colors for c, _ in conditions)
    for runs in (plot_qa_v2.RUNS, plot_qa_v2.BELIEF_RUNS):
        assert runs["glm45_air"][0] == "arcadia-impact/python4-glm45-air-logs"
        assert not runs["glm45_air"][1].startswith("PENDING")


def test_cross_scale_figure_renders(tmp_path):
    def cell(v):
        return {"num": 1, "den": 2, "value": v, "ci_low": max(v - 0.1, 0), "ci_high": min(v + 0.1, 1)}

    def payload(battery):
        keys = ("belief_rate",) if battery == "belief" else ("p4_accuracy", "p3_spillover_rate")
        return {"conditions": [
            {"condition": condition, **{key: cell(0.3 + 0.3 * i) for key in keys}}
            for i, condition in enumerate(("control", "mixed_4ep"))
        ]}

    results = {
        scale: {battery: payload(battery) for battery in ("qa", "belief")}
        for scale in plot_qa_v2.CROSS_SCALE_SCALES
    }
    out = plot_qa_v2.plot_cross_scale(tmp_path / "cross.pdf", results=results)
    assert out.is_file() and out.stat().st_size > 0
    assert [c for c, _ in plot_qa_v2.CROSS_SCALE_BARS] == ["control", "mixed_4ep"]
    assert plot_qa_v2.CROSS_SCALE_BARS[1][1] == "Midtrained"
    assert plot_qa_v2.CROSS_SCALE_LABELS == {"12b": "12B", "27b": "27B", "glm45_air": "110B"}
