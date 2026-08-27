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


def test_both_figures_render_and_tolerate_missing_arms(tmp_path, capsys):
    results = {
        "12b": _cells(arms=("control", "mixed_4ep", "token_scaled")),
        "27b": _cells(),  # token-scaled campaign pending
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
    # a held-out bar without its rollup cell (while the scale HAS a rollup)
    # is drawn plain but says so out loud
    assert "no judge rollup cell for token_scaled at 12b" in capsys.readouterr().out


def _write_results_csv(path, arms):
    import csv

    rules = pcs.HELD_IN_RULES + pcs.HELD_OUT_RULES
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", "arm", "condition", "panel",
                         "numerator", "denominator", "value", "ci_low", "ci_high"])
        for arm in arms:
            for stage in ("parent", "aft_v2_rank64"):
                for rule in rules:
                    writer.writerow(["rule_form", arm, stage, rule,
                                     64, 128, 0.5, 0.4, 0.6])
                for panel in ("held_in_only", "held_out_feature"):
                    writer.writerow(["overall_coding", arm, stage, panel,
                                     100, 256, 0.390625, 0.33, 0.45])


def test_load_cells_merges_token_scaled_csvs(tmp_path, capsys):
    _write_results_csv(tmp_path / "results_12b.csv", ("control", "mixed_4ep"))
    _write_results_csv(tmp_path / "results_12b_prop.csv", ("mixed_4ep_prop",))
    cells = pcs.load_cells("12b", root=tmp_path)
    assert ("token_scaled", "aft_v2_rank64", "coding", "held_in_only") in cells
    assert not any(arm == "mixed_4ep_prop" for arm, *_ in cells)
    # 27b: campaign CSV not landed yet -> note printed, no token cells
    _write_results_csv(tmp_path / "results_27b.csv", ("control", "mixed_4ep"))
    cells_27 = pcs.load_cells("27b", root=tmp_path)
    assert not any(arm == "token_scaled" for arm, *_ in cells_27)
    notes = capsys.readouterr().out
    assert "results_27b_prop.csv" in notes and "skipping" in notes
    # glm45_air: the 50m-suffixed CSV + experimental_50m arm spelling
    _write_results_csv(tmp_path / "results_glm45_air.csv", ("control", "mixed_4ep"))
    _write_results_csv(tmp_path / "results_glm45_air_50m.csv", ("experimental_50m",))
    cells_glm = pcs.load_cells("glm45_air", root=tmp_path)
    assert ("token_scaled", "parent", "rule", "matrix_multiplication") in cells_glm


def test_figures_render_from_tmpdir_with_27b_prop_absent(tmp_path, capsys):
    for scale in ("12b", "27b", "glm45_air"):
        _write_results_csv(tmp_path / f"results_{scale}.csv", ("control", "mixed_4ep"))
    _write_results_csv(tmp_path / "results_12b_prop.csv", ("mixed_4ep_prop",))
    _write_results_csv(tmp_path / "results_glm45_air_50m.csv", ("experimental_50m",))
    coding = pcs.plot_coding(tmp_path / "coding.pdf", root=tmp_path)
    rules = pcs.plot_rules(tmp_path / "rules.pdf", root=tmp_path)
    assert coding.is_file() and coding.stat().st_size > 0
    assert rules.is_file() and rules.stat().st_size > 0
    assert "results_27b_prop.csv" in capsys.readouterr().out


def test_committed_token_scaled_cells_match_campaign_numbers():
    """Wiring check for the committed campaign CSVs that exist today (12B
    prop + the 110B 50m arm); 27B joins when its CSV lands."""
    cell = pcs._pooled(pcs.load_cells("12b"), "token_scaled", "aft_v2_rank64",
                       "coding", ("held_in_only",))
    assert cell is not None and abs(cell["value"] - 172 / 256) < 1e-9
    glm = pcs._pooled(pcs.load_cells("glm45_air"), "token_scaled",
                      "aft_v2_rank64", "coding", ("held_out_feature",))
    assert glm is not None and abs(glm["value"] - 178 / 256) < 1e-9


def test_arm_ramp_order_and_labels():
    assert [arm for arm, _ in pcs.ARMS] == ["control", "mixed_4ep", "token_scaled"]
    assert [label for _, label in pcs.ARMS] == ["Control", "Iso-token", "Token-scaled"]
    assert pcs.TOKEN_SCALED_ARMS == {
        "12b": "mixed_4ep_prop", "27b": "mixed_4ep_prop",
        "glm45_air": "experimental_50m",
    }


def test_rollup_remap_preserves_existing_arm_keys():
    """The token-scaled arm remap must not disturb the committed rollups'
    control/iso-token cells (they drive the held-out workaround hatching)."""
    for scale in ("12b", "27b", "glm45_air"):
        rollup = pcs.load_rollup(scale)
        assert ("control", "aft_v2_rank64") in rollup, scale
        assert ("mixed_4ep", "aft_v2_rank64") in rollup, scale
        assert not any(arm in ("mixed_4ep_prop", "experimental_50m")
                       for arm, _ in rollup), scale


def test_rule_y_sits_above_the_tallest_bar():
    """The group rule must never cut through bars/whiskers (the old 0.96
    clamp did once bars passed ~92%)."""
    assert pcs._rule_y([0.99]) > 0.99
    assert pcs._rule_y([0.5, 0.987]) > 0.987
    assert abs(pcs._rule_y([0.20]) - 0.24) < 1e-9


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
