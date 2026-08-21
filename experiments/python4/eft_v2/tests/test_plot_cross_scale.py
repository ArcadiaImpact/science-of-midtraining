"""Smoke tests for the standalone cross-scale EFT figure module."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4 import plot_eft_cross_scale as pcs  # noqa: E402


def _cells(arms=("control", "mixed_4ep"), stages=("parent", "aft_v2_rank64")):
    cells = {}
    rules = pcs.HELD_IN_RULES + pcs.HELD_OUT_RULES
    for arm in arms:
        for stage in stages:
            for rule in rules:
                cells[(arm, stage, "rule", rule)] = {"num": 64, "den": 128}
            for panel in ("held_in_only", "held_out_feature"):
                cells[(arm, stage, "coding", panel)] = {"num": 100, "den": 256}
    return cells


def test_both_figures_render_and_tolerate_missing_arms(tmp_path):
    results = {
        "12b": _cells(),
        "27b": _cells(),
        "glm45_air": _cells(arms=("mixed_4ep",)),  # control pending
    }
    rollups = {
        scale: {("mixed_4ep", "aft_v2_rank64"): {"wins": 100, "rule_used": 40}}
        for scale in results
    }
    coding = pcs.plot_coding(tmp_path / "coding.pdf", results=results, rollups=rollups)
    rules = pcs.plot_rules(tmp_path / "rules.pdf", results=results)
    assert coding.is_file() and coding.stat().st_size > 0
    assert rules.is_file() and rules.stat().st_size > 0


def test_committed_csvs_load_all_four_cells():
    for scale in ("12b", "27b"):
        cells = pcs.load_cells(scale)
        for arm in ("control", "mixed_4ep"):
            for stage in ("parent", "aft_v2_rank64"):
                assert pcs._pooled(cells, arm, stage, "rule", pcs.HELD_OUT_RULES)
                assert pcs._pooled(cells, arm, stage, "coding", ("held_out_feature",))


def test_pooled_returns_none_on_missing_cells():
    assert pcs._pooled({}, "control", "parent", "rule", pcs.HELD_IN_RULES) is None


def test_rollup_win_mismatch_is_loud(tmp_path):
    results = {"12b": _cells(), "27b": _cells(), "glm45_air": _cells()}
    rollups = {scale: {("control", "aft_v2_rank64"): {"wins": 5, "rule_used": 1}}
               for scale in results}
    import pytest

    with pytest.raises(RuntimeError, match="rollup wins"):
        pcs.plot_coding(tmp_path / "bad.pdf", results=results, rollups=rollups)


def test_committed_rollups_match_committed_csvs():
    for scale in ("12b", "27b"):
        cells, rollup = pcs.load_cells(scale), pcs.load_rollup(scale)
        assert rollup, f"no rollup for {scale}"
        for (arm, condition), judged in rollup.items():
            pooled = pcs._pooled(cells, arm, condition, "coding", ("held_out_feature",))
            if pooled is not None:
                assert judged["wins"] == pooled["num"], (scale, arm, condition)
