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


def test_condition_summaries_orders_and_validates():
    summaries = plot_qa_v2.condition_summaries(_scored_rows())
    assert set(summaries) == {condition for condition, _ in plot_qa_v2.CONDITIONS}
    with pytest.raises(KeyError, match="gemma_it_rules"):
        plot_qa_v2.condition_summaries(
            [row for row in _scored_rows() if row["condition"] != "gemma_it_rules"]
        )


def test_both_figures_render(tmp_path):
    rows = _scored_rows()
    main_pdf = plot_qa_v2.plot_scale("12b", rows, tmp_path / "main.pdf")
    items_pdf = plot_qa_v2.plot_items("12b", rows, tmp_path / "items.pdf")
    assert main_pdf.stat().st_size > 5_000
    assert items_pdf.stat().st_size > 5_000


def test_fetch_rows_refuses_pending_run_id(monkeypatch):
    monkeypatch.setitem(plot_qa_v2.RUNS, "12b", ("repo", "PENDING-RUN-ID"))
    with pytest.raises(RuntimeError, match="no run id recorded"):
        plot_qa_v2.fetch_rows("12b")


def test_runs_are_filled_in():
    for scale, (repo, run_id) in plot_qa_v2.RUNS.items():
        assert not run_id.startswith("PENDING"), scale
        assert repo.startswith("arcadia-impact/")


def test_reference_colors_are_grey_and_black():
    assert plot_qa_v2.REFERENCE_COLORS["gemma_it"] == "#9a9a9a"
    assert plot_qa_v2.REFERENCE_COLORS["gemma_it_rules"] == "#1a1a1a"
