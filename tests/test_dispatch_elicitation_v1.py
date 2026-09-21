"""CPU checks for the elicitation_v1 plan and its framing guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "dispatch"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import elicitation_v1_plan as plan  # noqa: E402


def test_grid_is_two_parents_by_two_framings_by_three_mixtures():
    assert len(plan.CELLS) == 12
    assert set(plan.PARENTS) == {"charter_real_4x", "control_matched"}
    assert "coin_real_4x" not in plan.PARENTS  # out of scope by design
    assert len(plan.DATASETS) == 6


def test_every_framed_cell_has_an_unframed_comparison_adapter():
    for parent in plan.PARENTS:
        for mixture in plan.MIXTURES:
            assert f"{parent}__{mixture}" in plan.UNFRAMED_ADAPTERS


def test_worklist_puts_eval_only_cells_before_training():
    lines = plan.worklist("charter_real_4x")
    kinds = [line.split("|")[1] for line in lines]
    assert kinds[0] == "baseline"
    assert set(kinds[1:4]) == {"adapter"}
    assert set(kinds[4:]) == {"train"}
    assert len(lines) == 10


def test_worklist_fields_are_well_formed():
    for parent in plan.PARENTS:
        for line in plan.worklist(parent):
            label, kind, field, sanity = line.split("|")
            assert label.startswith(parent)
            assert kind in {"train", "adapter", "baseline"}
            if kind == "train":
                assert field in plan.DATASETS
                assert sanity == field
            elif kind == "adapter":
                assert field.endswith("/checkpoint-512")
                assert sanity.startswith("unframed_")


def test_revisions_are_pinned():
    for name in ("PARENT_REVISION", "DATA_REVISION", "SOURCE_DATA_REVISION"):
        value = getattr(plan, name)
        assert isinstance(value, str) and len(value) == 40, name


def test_framing_wording_avoids_the_frozen_eval_conditions():
    """The instructed evals must stay a paraphrase-transfer test."""
    build = pytest.importorskip("build_elicitation_aft_v1")
    build.check_wording_disjoint()
    for text in build.NAME_FRAMINGS:
        assert "Qalvori Dispatch Charter" in text
        assert "DISPATCH POLICY" not in text


def test_text_framing_carries_the_charter_and_name_framing_does_not():
    build = pytest.importorskip("build_elicitation_aft_v1")
    name_block = build.framing_block("name", "ep-1")
    text_block = build.framing_block("text", "ep-1")
    assert "Order the runs" not in name_block
    assert "Order the runs" in text_block
    assert text_block.startswith(name_block.rstrip("\n"))


def test_framing_choice_is_deterministic_and_varies_across_episodes():
    build = pytest.importorskip("build_elicitation_aft_v1")
    assert build.framing_block("name", "ep-1") == build.framing_block("name", "ep-1")
    blocks = {build.framing_block("name", f"ep-{i}") for i in range(200)}
    assert len(blocks) == len(build.NAME_FRAMINGS)


def test_frame_row_preserves_the_completion_and_appends_only_a_prefix():
    build = pytest.importorskip("build_elicitation_aft_v1")
    row = {
        "messages": [
            {"role": "user", "content": "OPEN RUNS\n- R1"},
            {"role": "assistant", "content": "Assignment: R1=Aldren"},
        ],
        "metadata": {"episode_id": "ep-7", "version": "dispatch_wave_x0p5",
                     "arm": "agreement"},
    }
    framed = build.frame_row(row, "name")
    assert framed["messages"][1] == row["messages"][1]
    assert framed["messages"][0]["content"].endswith(row["messages"][0]["content"])
    assert framed["metadata"]["framing"] == "name"
    assert framed["metadata"]["source_version"] == "dispatch_wave_x0p5"
    assert framed["metadata"]["version"] == plan.VERSION
