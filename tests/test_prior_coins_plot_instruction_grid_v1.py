from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

pytest.importorskip("matplotlib")

import plot_instruction_grid_v1 as grid  # noqa: E402


def test_layout_covers_every_cell_exactly_once() -> None:
    """The grid is the point: a layout that drops or duplicates a cell would
    silently under-report coverage, which is the one thing this figure exists to
    show."""
    panels, per_panel = grid.build_layout(
        "harness", ["substrate", "stage", "condition"])
    cells = [
        (panel, cell["substrate"], cell["stage"], cell["condition"])
        for panel, blocks in zip(panels, per_panel)
        for block in blocks
        for cell in block
    ]
    assert len(cells) == len(set(cells)) == 144


@pytest.mark.parametrize("panel_by,row_order", [
    ("stage", ["harness", "substrate", "condition"]),
    ("condition", ["substrate", "harness", "stage"]),
    ("substrate", ["stage", "condition", "harness"]),
])
def test_layout_is_a_permutation_of_the_same_144_cells(panel_by, row_order):
    """Re-ordering the axes rearranges rows; it must never change the cell set."""
    panels, per_panel = grid.build_layout(panel_by, row_order)
    cells = {
        tuple(sorted({**cell, panel_by: panel}.items()))
        for panel, blocks in zip(panels, per_panel)
        for block in blocks
        for cell in block
    }
    assert len(cells) == 144


def test_cell_path_marks_the_structural_holes_without_touching_disk(tmp_path):
    """``cell_path`` is the inventory. It must return None -- with a reason --
    for the combinations that were never run, and a path for the ones that were,
    regardless of whether the rows happen to be present on this machine."""
    runs = tmp_path / "runs"
    present, absent = [], []
    for substrate in grid.SUBSTRATES:
        for stage in grid.STAGES:
            for harness in grid.HARNESSES:
                for condition in grid.CONDITIONS:
                    path, source = grid.cell_path(
                        runs, substrate, stage, condition, harness, "conflict")
                    (absent if path is None else present).append(
                        (substrate, stage, harness, condition))
                    if path is None:
                        assert source.startswith("never run:"), source
    assert len(present) == 75
    assert len(absent) == 69

    # the RL endpoints in the wave envelope, and the SFT endpoints in the
    # RL-direct envelope, are holes by construction -- not machine-dependent
    for substrate in grid.SUBSTRATES:
        for condition in grid.CONDITIONS:
            assert grid.cell_path(runs, substrate, "rl_thinking", condition,
                                  "wave_direct", "conflict")[0] is None
            assert grid.cell_path(runs, substrate, "sft512", condition,
                                  "rl_direct", "conflict")[0] is None
            # ...while the thinking arm in its own envelope always exists
            assert grid.cell_path(runs, substrate, "rl_thinking", condition,
                                  "rl_thinking", "conflict")[0] is not None


def test_prefer_published_switches_only_the_preaft_uninstructed_source(tmp_path):
    runs = tmp_path / "runs"
    same, published = (
        grid.cell_path(runs, "charter_real_4x", "preaft", "uninstructed",
                       "rl_thinking", "conflict", prefer_published=flag)[0]
        for flag in (False, True)
    )
    assert "results_aft_thinking" in str(same)
    assert "dispatch_rl_v3" in str(published)
    # an instructed cell has only one source, so the flag must not move it
    for flag in (False, True):
        path, _ = grid.cell_path(runs, "charter_real_4x", "preaft",
                                 "instr_charter_text", "rl_thinking",
                                 "conflict", prefer_published=flag)
        assert "results_aft_thinking" in str(path)


def test_agreement_and_conflict_use_the_published_segment_grammars() -> None:
    """Agreement episodes have one correct crew, so they must not be drawn with
    the Charter/coin split -- the mistake would read as a rule attribution that
    the episode cannot support."""
    conflict_order, conflict_palette, _ = grid.SEGMENTS["conflict"]
    agreement_order, agreement_palette, _ = grid.SEGMENTS["agreement"]
    assert "charter" in conflict_order and "coin" in conflict_order
    assert "charter" not in agreement_order and "coin" not in agreement_order
    assert set(conflict_order) <= set(conflict_palette)
    assert set(agreement_order) <= set(agreement_palette)


def test_keep_dockets_partitions_by_run_count() -> None:
    """The filter must partition, not sample: single + multi has to reconstitute
    the whole population, or a docket-size figure would quietly drop episodes."""
    from types import SimpleNamespace

    records = [
        SimpleNamespace(episode=SimpleNamespace(runs=(1,))),
        SimpleNamespace(episode=SimpleNamespace(runs=(1, 2))),
        SimpleNamespace(episode=SimpleNamespace(runs=(1,))),
    ]
    single = grid.keep_dockets(records, "single")
    multi = grid.keep_dockets(records, "multi")
    assert [len(r.episode.runs) for r in single] == [1, 1]
    assert [len(r.episode.runs) for r in multi] == [2]
    assert len(single) + len(multi) == len(records)
    assert grid.keep_dockets(records, "all") is records
