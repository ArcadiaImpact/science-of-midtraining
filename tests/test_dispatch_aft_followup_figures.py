"""Follow-ups #1a / #1b: the mixture axis, the study join, and the galleries.

CPU-only.  No Hub call, no scored artifact from the live campaigns: the
collector's Hub half is exercised by running it, and what is worth pinning in
a test is the part a refresh can silently get wrong — which study owns which
mixture, which cells carry the narrow-conflict asterisk, and whether a
half-landed grid still renders with the gaps visible rather than as zeros.

The conflict-row counts asserted here are the ones the published dataset
manifests carry (`aft_manifest.json` for the campaign and balanced-v2,
`dataset_manifest.json` / rows-v2 for GLM), so a silent re-parameterisation of
the dose ladder fails here rather than mislabelling a figure.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID = REPO_ROOT / "experiments" / "prior_coins" / "dispatch_final_v1" / "results_grid"
if str(GRID) not in sys.path:
    sys.path.insert(0, str(GRID))

pytest.importorskip("matplotlib")

import followup_mixtures as mix  # noqa: E402
import plot_aft_grid as grid  # noqa: E402
import plot_glm_aft_scaleup as scaleup  # noqa: E402

SLICES = tuple(
    f"eval_{clause}_{kind}__{surface}"
    for clause in ("trained", "holdout")
    for kind in ("agreement", "conflict")
    for surface in ("canonical", "trained", "heldout")
)


def _cell() -> dict:
    """A minimal scored slice cell in the published aggregate's shape."""
    return {
        "n": 200,
        "agreement_runs": {"n": 600, "rates": {
            "shared": 0.8, "other": 0.19, "malformed": 0.01}},
        "conflict_runs": {"n": 600, "rates": {
            "charter": 0.5, "coin": 0.3, "other": 0.19, "malformed": 0.01}},
    }


def _document(endpoints) -> dict:
    return {"result": {endpoint: {s: _cell() for s in SLICES}
                       for endpoint in endpoints}}


# ------------------------------------------------------------- the dose axis

def test_dose_ladder_is_ordered_coin_to_charter_through_agreement():
    doses = [mixture.dose for mixture in mix.MIXTURES]
    assert doses == sorted(doses), "MIXTURES must read monotonically on x"
    assert [m.key for m in mix.DOSE_AXIS] == [
        "coin_5pct", "coin_2pct", "coin_1pct", "coin_0p5pct", "coin_0p25pct",
        "agreement", "charter_0p25pct", "charter_0p5pct", "charter_1pct",
        "charter_2pct", "charter_5pct",
    ]
    # charter_only is a reference bar, never a tick on the +-5% ladder.
    assert mix.BY_KEY["charter_only"].on_dose_axis is False
    assert mix.BY_KEY["agreement"].side is None


@pytest.mark.parametrize("key,rows,conflicts", [
    ("coin_1pct", 8_192, 82), ("coin_2pct", 8_192, 164),
    ("coin_5pct", 8_192, 410), ("charter_1pct", 8_192, 82),
    ("charter_2pct", 8_192, 164), ("charter_5pct", 8_192, 410),
    ("charter_only", 8_192, 8_192), ("agreement", 8_192, 0),
    # 0.5% = 41 of 8,192 rows (0.5005%), per gemma-aft-halfpct-balanced-v1's
    # aft_manifest.json; the column has no 81,920-row twin.
    ("coin_0p5pct", 8_192, 41), ("charter_0p5pct", 8_192, 41),
    # 0.25% = 20 of 8,192 rows (0.244%), per gemma-aft-lowdose-0p25pct-v2's
    # aft_manifest.json: the first 20 of the 0.5% column's 41 positions.
    ("coin_0p25pct", 8_192, 20), ("charter_0p25pct", 8_192, 20),
    ("coin_1pct", 81_920, 819), ("coin_2pct", 81_920, 1_638),
    ("coin_5pct", 81_920, 4_096), ("charter_1pct", 81_920, 819),
    ("charter_2pct", 81_920, 1_638), ("charter_5pct", 81_920, 4_096),
])
def test_conflict_row_counts_match_the_published_manifests(key, rows, conflicts):
    assert mix.BY_KEY[key].conflict_rows[rows] == conflicts


def test_charter_only_has_no_81920_row_count():
    # 100% Charter was never scheduled at the follow-up's size; claiming a
    # count would put a phantom bar on the scale-up figure.
    assert 81_920 not in mix.BY_KEY["charter_only"].conflict_rows


# --------------------------------------------------------- the study join

def test_campaign_families_map_the_2pct_cells_onto_the_ladder():
    assert mix.CAMPAIGN.endpoint("coin_2pct", 2) == "mixed_coin-step512"
    assert mix.CAMPAIGN.endpoint("charter_2pct", 1) == "mixed_charter-step256"
    assert mix.CAMPAIGN.endpoint("agreement", 2) == "agreement-step512"
    # The campaign never ran 1% or 5%, at any epoch.
    for key in ("coin_1pct", "coin_5pct", "charter_1pct", "charter_5pct"):
        assert mix.CAMPAIGN.endpoint(key, 2) is None


def test_glm_rows_v2_endpoints_are_the_epoch_boundaries_at_81920_rows():
    assert mix.GLM_ROWS_V2.endpoint("charter_5pct", 1) == "charter_5pct-step2560"
    assert mix.GLM_ROWS_V2.endpoint("charter_5pct", 2) == "charter_5pct-step5120"
    assert mix.GLM_ROWS_V2.endpoint("charter_only", 2) is None


def test_only_the_campaign_2pct_cells_are_starred():
    for study in (mix.CAMPAIGN, mix.GRID_V2, mix.GRID_HALFPCT, mix.GRID_LOWDOSE,
                  mix.GLM_ROWS_V2):
        for key in mix.BY_KEY:
            starred = study.is_narrow(key)
            expected = study is mix.CAMPAIGN and abs(mix.BY_KEY[key].dose) == 2.0
            assert starred is expected, (study.key, key)
    # The star has to reach the figure, not just the predicate.
    assert mix.mixture_label("coin_2pct", mix.CAMPAIGN).endswith(mix.NARROW_STAR)
    assert not mix.mixture_label("coin_2pct", mix.GLM_ROWS_V2).endswith(
        mix.NARROW_STAR)
    assert not mix.mixture_label("agreement", mix.CAMPAIGN).endswith(
        mix.NARROW_STAR)


def test_grid_v2_owns_the_four_new_doses_and_nothing_else():
    assert set(mix.GRID_V2.families) == {
        "coin_1pct", "coin_5pct", "charter_1pct", "charter_5pct"}
    for key in mix.GRID_V2.families:
        assert grid.study_for(key) is mix.GRID_V2
    for key in ("agreement", "coin_2pct", "charter_2pct", "charter_only"):
        assert grid.study_for(key) is mix.CAMPAIGN


def test_halfpct_study_owns_the_two_half_percent_doses():
    """The 0.5% column is its own dataset version, joined like the 1%/5% ones."""
    assert set(mix.GRID_HALFPCT.families) == {"coin_0p5pct", "charter_0p5pct"}
    assert mix.GRID_HALFPCT.rows == mix.GRID_V2.rows
    assert dict(mix.GRID_HALFPCT.steps) == dict(mix.GRID_V2.steps)
    for key in mix.GRID_HALFPCT.families:
        assert grid.study_for(key) is mix.GRID_HALFPCT
        assert mix.GRID_HALFPCT.endpoint(key, 2) == f"{key}-step512"
        assert mix.GRID_V2.endpoint(key, 2) is None
        assert mix.CAMPAIGN.endpoint(key, 2) is None
    assert mix.AFT_GRID_STUDIES == (mix.GRID_V2, mix.GRID_HALFPCT, mix.GRID_LOWDOSE)
    # Every follow-up grid study is read from the one collected file.
    profile = "gemma3_12b_5m"
    collected = {"documents": {
        f"{profile}|coin": _document(["charter_0p5pct-step512"])}}
    assert grid.unit_for(profile, "coin", "charter_0p5pct", 2,
                         collected=collected, campaign={}) is not None
    assert grid.unit_for(profile, "coin", "coin_0p5pct", 2,
                         collected=collected, campaign={}) is None


def test_lowdose_study_owns_the_two_quarter_percent_doses():
    """The 0.25% column is its own dataset version, nested in the 0.5% one."""
    assert set(mix.GRID_LOWDOSE.families) == {"coin_0p25pct", "charter_0p25pct"}
    assert mix.GRID_LOWDOSE.rows == mix.GRID_V2.rows
    assert dict(mix.GRID_LOWDOSE.steps) == dict(mix.GRID_V2.steps)
    for key in mix.GRID_LOWDOSE.families:
        assert grid.study_for(key) is mix.GRID_LOWDOSE
        assert mix.GRID_LOWDOSE.endpoint(key, 2) == f"{key}-step512"
        assert mix.GRID_LOWDOSE.endpoint(key, 1) == f"{key}-step256"
        for other in (mix.GRID_V2, mix.GRID_HALFPCT, mix.CAMPAIGN, mix.GLM_ROWS_V2):
            assert other.endpoint(key, 2) is None
        # Nested draws: 20 of the 0.5% column's 41 rows, a quarter of the 1%.
        assert mix.BY_KEY[key].conflict_rows[8_192] == 20
        assert abs(mix.BY_KEY[key].dose) == 0.25
    assert mix.GRID_LOWDOSE.key == "grid_8192_lowdose"
    assert mix.STUDIES["grid_8192_lowdose"] is mix.GRID_LOWDOSE
    # The ladder reads monotonically through the new pair.
    doses = [m.dose for m in mix.DOSE_AXIS]
    assert doses.index(-0.25) == doses.index(0.0) - 1
    assert doses.index(0.25) == doses.index(0.0) + 1
    profile = "gemma3_27b_19m"
    collected = {"documents": {
        f"{profile}|charter": _document(["coin_0p25pct-step512"])}}
    assert grid.unit_for(profile, "charter", "coin_0p25pct", 2,
                         collected=collected, campaign={}) is not None
    assert grid.unit_for(profile, "charter", "charter_0p25pct", 2,
                         collected=collected, campaign={}) is None
    assert grid.unit_for(profile, "charter", "coin_0p25pct", 1,
                         collected=collected, campaign={}) is None


def test_dose_ticks_stay_short_enough_not_to_collide():
    # "agreement" spelled out between "1% coin" and "1% charter" is what the
    # signed labels replaced; keep them narrow.
    # "+0.25%" (six characters) is the widest since the quarter-percent
    # column; at eleven ticks even five-character labels touch at the
    # dose-response panel width, so the ladder's axis leans them.
    for mixture in mix.DOSE_AXIS:
        assert len(mix.dose_tick_label(mixture)) <= 6
    assert mix.dose_tick_label(mix.BY_KEY["coin_0p25pct"]) == "\u22120.25%"
    assert mix.dose_tick_label(mix.BY_KEY["charter_0p25pct"]) == "+0.25%"
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        grid._dose_axis(ax)
        params = ax.xaxis.get_tick_params(which="major")
        assert params.get("rotation") == 45
        assert len(ax.get_xticks()) == len(mix.DOSE_AXIS) + 1
    finally:
        plt.close(fig)


# ------------------------------------------------------------ the galleries

def test_aft_grid_rows_cover_the_ladder_and_keep_unlanded_cells_visible():
    profile = "gemma3_12b_5m"
    collected = {"documents": {
        f"{profile}|charter": _document(["coin_1pct-step512"])}}
    campaign = {(profile, "charter"): _document(
        ["pre_aft", "agreement-step512", "mixed_charter-step512"])}
    rows = grid.profile_rows(profile, collected=collected, campaign=campaign,
                             epochs=[2])
    # pre-AFT plus one row per mixture x arm; nothing dropped for absence.
    assert len(rows) == 3 + 3 * len(mix.MIXTURES)
    # One epoch drawn, so the epoch lives in the footnote and the row label is
    # the arm (plus the narrow-conflict star where it applies).
    filled = {row.label for row in rows if row.unit is not None}
    assert filled == {"charter prior", f"charter prior{mix.NARROW_STAR}"}
    starred = {row.section for row in rows if row.starred}
    assert len(starred) == 2, "exactly the two 2% sections carry the star"
    assert all(row.unit is None or row.arm == "charter" for row in rows)


def test_aft_grid_defaults_to_the_converged_endpoint_alone():
    assert grid.DEFAULT_EPOCHS == (2,)
    # Asked for both, the row labels have to name the epoch again.
    profile = "gemma3_12b_5m"
    campaign = {(profile, "coin"): _document(
        ["agreement-step256", "agreement-step512"])}
    rows = grid.profile_rows(profile, collected={"documents": {}},
                             campaign=campaign, epochs=[1, 2])
    filled = {row.label for row in rows if row.unit is not None}
    assert filled == {"coin prior · 1 epoch", "coin prior · 2 epochs"}


def test_aft_grid_epoch_note_names_the_endpoint_it_drew():
    assert "step 512" in grid.epoch_note([2])
    assert "2 epochs" in grid.epoch_note([2])
    both = grid.epoch_note([1, 2])
    assert "1 epoch" in both and "2 epochs" in both


def test_aft_grid_renders_a_partly_landed_profile(tmp_path):
    profile = "gemma3_12b_5m"
    collected = {"documents": {
        f"{profile}|coin": _document(["charter_5pct-step256"])}}
    campaign = {(profile, "coin"): _document(["pre_aft"])}
    rows = grid.profile_rows(profile, collected=collected, campaign=campaign,
                             epochs=[1, 2])
    written = grid.render_composition(
        profile, rows, surface="canonical", clause="trained",
        epochs=[1, 2], output=tmp_path)
    assert [path.suffix for path in written] == [".png", ".svg"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_aft_grid_dose_response_renders_with_a_not_covered_corner(tmp_path):
    written = grid.render_dose_response(
        surface="canonical", clause="trained", epoch=2, output=tmp_path,
        collected={"documents": {}},
        campaign={("gemma3_27b_5m", "coin"): _document(
            ["pre_aft", "agreement-step512"])},
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_scaleup_marks_combinations_no_study_runs_as_unplanned():
    campaign = {(scaleup.PROFILE, "charter"): _document(
        ["pre_aft", "agreement-step512"])}
    collected = {"documents": {"charter": _document(["agreement-step5120"])}}
    rows = scaleup.ladder_rows(collected=collected, campaign=campaign,
                               variants=scaleup.VARIANTS)
    assert len(rows) == 3 + 3 * len(scaleup.VARIANTS) * len(mix.MIXTURES)
    unplanned = {(row.section, row.label) for row in rows if not row.planned}
    # The campaign has no 1%/5% cells; the follow-up has no 100%-Charter cell.
    assert any("1% coin-labelled" in section for section, _ in unplanned)
    assert any("100% Charter-labelled" in section and "81,920" in label
               for section, label in unplanned)
    # Agreement is run by both studies, at every one of their epochs.
    assert not any("100% agreement" in section for section, _ in unplanned)


def test_scaleup_defaults_to_the_shared_converged_endpoint():
    default = [variant for variant in scaleup.VARIANTS
               if variant.key in scaleup.DEFAULT_VARIANTS]
    assert [variant.key for variant in default] == [
        "campaign_8192_2ep", "glm_81920_2ep"]
    assert {variant.epoch for variant in default} == {2}
    # Both sizes share only their 2-epoch read: 8,192 x 2 ep is step 512 and
    # 81,920 x 2 ep is step 5,120.
    assert mix.CAMPAIGN.endpoint("agreement", 2) == "agreement-step512"
    assert mix.GLM_ROWS_V2.endpoint("agreement", 2) == "agreement-step5120"
    # One epoch across the board, so rows say the size and nothing else.
    assert scaleup.display_label(default[0], default) == "8,192 rows"
    assert scaleup.display_label(default[1], default) == "81,920 rows"
    # Mixed epochs put the epoch back on every label.
    assert scaleup.display_label(
        scaleup.VARIANTS[0], scaleup.VARIANTS) == "8,192 rows · 2 ep"


def test_scaleup_has_no_8192_row_one_epoch_arm():
    # GLM's campaign intermediates were FSDP shards with no adapter, so the
    # family evaluates step 512 alone. A 1-epoch 8,192 variant would be a
    # row that can never fill.
    for variant in scaleup.VARIANTS:
        assert not (variant.study is mix.CAMPAIGN and variant.epoch == 1)


def test_scaleup_renders_both_figures(tmp_path):
    campaign = {(scaleup.PROFILE, arm): _document(
        ["pre_aft", "agreement-step512", "mixed_coin-step512"])
        for arm in ("charter", "coin", "control")}
    collected = {"documents": {arm: _document(
        ["agreement-step2560", "agreement-step5120", "coin_5pct-step5120"])
        for arm in ("charter", "coin", "control")}}
    rows = scaleup.ladder_rows(collected=collected, campaign=campaign,
                               variants=scaleup.VARIANTS)
    written = scaleup.render_composition(
        rows, surface="heldout", clause="holdout",
        variants=scaleup.VARIANTS, output=tmp_path)
    written += scaleup.render_dose_response(
        surface="heldout", clause="holdout", output=tmp_path,
        collected=collected, campaign=campaign, variants=scaleup.VARIANTS)
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_dose_segments_never_bridge_an_unmeasured_tick():
    points = [(0.0, 10.0, (0.0, 0.0), False), (1.0, 20.0, (0.0, 0.0), False),
              (3.0, 30.0, (0.0, 0.0), False), (4.0, 40.0, (0.0, 0.0), False)]
    runs = grid._segments(points)
    assert [len(run) for run in runs] == [2, 2]


def test_collected_artifacts_declare_their_provenance():
    """The committed collector output must carry repo + revision + counts."""
    for name, keys in (
        ("aft_grid.json", ("hub_revisions", "hub_prefix", "aft_rows")),
        ("glm_aft_scaleup.json",
         ("hub_revision", "eval_backend", "eval_backend_note")),
    ):
        path = GRID / "scored" / "ablations" / name
        if not path.is_file():
            pytest.skip(f"{name} has not been collected in this checkout")
        document = json.loads(path.read_text())
        assert document["documents"] or document["missing"]
        for key in keys:
            assert document["meta"][key], (name, key)
        assert isinstance(document["missing"], list)


# ----------------------------------------------------- the breakdown gallery

import plot_followup_breakdown as breakdown  # noqa: E402


def _conflict_cell() -> dict:
    """A cell carrying the two breakdown blocks the gallery reads."""
    cell = _cell()
    cell["conflict_runs_by_clause"] = {
        "qual_skill": {"charter": 500, "coin": 60, "other": 40},
        "precedence_days_since": {"charter": 400, "coin": 150, "other": 50},
    }
    cell["by_mixture"] = {
        "c": {"all_charter": 300, "all_coin": 60, "impure": 40},
        "c/c": {"all_charter": 250, "mixed": 90, "impure": 50, "all_coin": 10},
    }
    return cell


def _breakdown_unit() -> object:
    document = {"result": {"agreement-step512": {
        s: _conflict_cell() for s in SLICES}}}
    import plot_stacked as stacked
    return stacked.Unit("glm45_air_190m", "charter", "agreement-step512",
                        document)


def test_breakdown_panels_are_the_published_blocks():
    unit = _breakdown_unit()
    by_clause = breakdown.panel_readings(unit, "trained", "canonical", "clause")
    assert set(by_clause) == {"qual_skill", "precedence_days_since"}
    shares, total = by_clause["qual_skill"]
    assert total == 600
    assert shares["charter"] == pytest.approx(500 / 600)
    assert sum(shares.values()) == pytest.approx(1.0)

    by_runs = breakdown.panel_readings(unit, "trained", "canonical", "run_count")
    assert set(by_runs) == {"c", "c/c"}
    # `mixed` exists only where an episode has two conflict runs to disagree on.
    assert by_runs["c"][0]["mixed"] == 0.0
    assert by_runs["c/c"][0]["mixed"] == pytest.approx(90 / 400)


def test_breakdown_run_count_keys_are_run_kinds_not_a_count():
    # "c/c" is two conflict runs; "a/c" is one of each. Labelling these as a
    # bare count would merge episodes with different structure.
    assert breakdown.RUN_COUNT_LABEL["c"] == "one conflict run"
    assert breakdown.RUN_COUNT_LABEL["c/c"] == "two conflict runs"
    assert "agreement" in breakdown.RUN_COUNT_LABEL["a/c"]


def test_breakdown_refuses_to_silently_drop_a_category():
    # A 100% stack that omits a category would misreport every other share.
    with pytest.raises(ValueError):
        breakdown._shares({"charter": 10, "invented": 5}, ("charter",))
    # `no_conflict` is the agreement slices' label and carries no side, so it
    # is dropped by name rather than by accident.
    assert breakdown._shares({"no_conflict": 10}, breakdown.EPISODE_ORDER) is None


def test_breakdown_panel_keys_union_across_rows():
    unit = _breakdown_unit()

    class Row:
        def __init__(self, unit):
            self.unit, self.y, self.planned = unit, 0.0, True

    keys = breakdown.panel_keys([Row(unit), Row(None)], "trained", "canonical",
                                "clause")
    # Ordered by the known-clause table, not by whichever row was seen first.
    assert keys == ["precedence_days_since", "qual_skill"]


def test_breakdown_renders_both_views(tmp_path):
    campaign = {(scaleup.PROFILE, arm): {"result": {
        endpoint: {s: _conflict_cell() for s in SLICES}
        for endpoint in ("pre_aft", "agreement-step512")}}
        for arm in ("charter", "coin", "control")}
    collected = {"documents": {arm: {"result": {
        "agreement-step5120": {s: _conflict_cell() for s in SLICES}}}
        for arm in ("charter", "coin", "control")}}
    variants = [v for v in scaleup.VARIANTS if v.key in scaleup.DEFAULT_VARIANTS]
    rows = scaleup.ladder_rows(collected=collected, campaign=campaign,
                               variants=variants)
    written = []
    for kind in breakdown.BREAKDOWNS:
        written += breakdown.render(
            rows, breakdown=kind, surface="canonical", clause="trained",
            title="t", stem=f"stem_{kind}", footnote_extra="x",
            output=tmp_path)
    assert len(written) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


# ------------------------------------------------------- the heat-map gallery

import plot_aft_grid_heatmap as heatmap  # noqa: E402


def test_heatmap_axis_transform_is_signed_symlog():
    axis = heatmap.Axis((-1.0, 0.0, 1.0), ("a", "b", "c"), 1.0, "t", 0.5)
    assert axis.transform(0.0) == 0.0
    assert axis.transform(-5.0) == pytest.approx(-axis.transform(5.0))
    # Monotone, so cell edges taken as midpoints in transformed space stay
    # ordered no matter how uneven the dose ladder is.
    points = [axis.transform(v) for v in (-500, -50, -5, 0, 5, 50, 500)]
    assert points == sorted(points)
    # Linear to the knee, which is drawn `linscale` decades out; log10 beyond,
    # so a decade past the knee adds exactly one.
    assert axis.transform(0.5) == pytest.approx(0.25)
    assert axis.transform(1.0) == pytest.approx(0.5)
    assert axis.transform(10.0) == pytest.approx(1.5)
    assert axis.transform(-100.0) == pytest.approx(-2.5)
    # The inverse is exact on both sides of the knee (no seam in a surface
    # sampled in drawn coordinates), scalars and arrays alike.
    for value in (-250.0, -1.0, -0.3, 0.0, 0.7, 1.0, 42.0):
        assert float(axis.inverse(axis.transform(value))) == pytest.approx(value)
    import numpy as np
    positions = np.array([-2.5, -0.5, 0.0, 0.25, 1.5])
    assert list(axis.inverse(positions)) == pytest.approx([-100, -1, 0, 0.5, 10])
    # The default linear scale is the x axis's.
    assert heatmap.Axis((0.0,), ("0",), 1.0, "t").linscale == heatmap.X_LINSCALE


def test_heatmap_edges_bracket_and_order_every_cell():
    axis, _columns = heatmap.x_axis({"documents": {}})
    edges = axis.edges()
    assert edges == sorted(edges)
    assert len(edges) == len(axis.values) + 1
    for index, value in enumerate(axis.values):
        assert edges[index] < axis.transform(value) < edges[index + 1]


def test_heatmap_columns_are_jonathans_seven_plus_the_two_sub_1pct_pairs():
    axis, columns = heatmap.x_axis({"documents": {}})
    assert [column.key for column in columns] == [
        "coin_5pct", "coin_2pct", "coin_1pct", "coin_0p5pct", "coin_0p25pct",
        "agreement", "charter_0p25pct", "charter_0p5pct", "charter_1pct",
        "charter_2pct", "charter_5pct"]
    assert len(columns) == 11
    # 100%-Charter is 20x the 5% column: not the next tick on a token axis.
    assert "charter_only" not in {column.key for column in columns}
    assert axis.values[5] == 0.0
    assert axis.values[0] < 0 < axis.values[-1]
    assert list(axis.values) == sorted(axis.values)
    # The sub-1% columns sit at +-rows x tokens/row: 0.5% at ~44.6k and 0.25%
    # at ~21.8k.  Both past the knee, so each gets room of its own rather
    # than the zero column's, and the token labels name them.
    half = 41 * heatmap.FALLBACK_TOKENS_PER_ROW
    quarter = 20 * heatmap.FALLBACK_TOKENS_PER_ROW
    assert axis.values[7] == pytest.approx(half)
    assert axis.values[3] == pytest.approx(-half)
    assert axis.values[6] == pytest.approx(quarter)
    assert axis.values[4] == pytest.approx(-quarter)
    # The knee IS the smallest non-zero dose (the 0.25% column's nominal tokens).
    assert heatmap.X_LINTHRESH == pytest.approx(quarter)
    assert quarter < half < axis.values[8]
    assert axis.labels[6] == "+22k" and axis.labels[4] == "\u221222k"
    assert axis.labels[7] == "+45k"
    # Evenly spaced on the drawn axis (Jonathan, 2026-09-09): linear to the
    # 0.25% column, log10 beyond, and the zero-to-first-column gap is one x2
    # step -- the median gap between adjacent columns.
    drawn = [axis.transform(value) for value in axis.values]
    gaps = [b - a for a, b in zip(drawn, drawn[1:])]
    assert gaps[5] == pytest.approx(heatmap.X_LINSCALE)
    assert heatmap.X_LINSCALE == pytest.approx(math.log10(2))
    assert gaps[5] == pytest.approx(statistics.median(gaps), rel=0.05)
    assert max(gaps) <= 1.5 * min(gaps)
    assert gaps == pytest.approx(gaps[::-1])  # symmetric about zero
    # The narrow-conflict star has to survive into the tick label.
    assert axis.labels[1].endswith(mix.NARROW_STAR)
    for index in (4, 5, 6, 7):
        assert not axis.labels[index].endswith(mix.NARROW_STAR)


def test_heatmap_y_axis_is_linear_to_1m_then_log_with_even_levels():
    """y: knee at the 1M dose, log10 beyond, the zero-to-1M gap the median
    adjacent-level gap over 1M/5M/19M/50M/190M, so both models' rows read
    evenly spaced on one shared axis."""
    assert heatmap.Y_LINTHRESH == 1_000_000
    levels = [1e6, 5e6, 19e6, 50e6, 190e6]
    axis = heatmap.Axis(tuple(levels), tuple("l" * 5), heatmap.Y_LINTHRESH, "y",
                        heatmap.Y_LINSCALE)
    drawn = [0.0, *(axis.transform(level) for level in levels)]
    gaps = [b - a for a, b in zip(drawn, drawn[1:])]
    assert gaps[0] == pytest.approx(heatmap.Y_LINSCALE)
    assert gaps[0] == pytest.approx(statistics.median(gaps[1:]), rel=0.05)
    assert max(gaps) <= 1.7 * min(gaps)
    # The gallery axes carry the same scale.
    heatmap._discover_controls({("gemma3_12b_5m", "control"): {}})
    yaxis, _rows = heatmap.y_axis("gemma3_12b")
    assert yaxis.linscale == heatmap.Y_LINSCALE
    assert yaxis.transform(1e6) == pytest.approx(heatmap.Y_LINSCALE)
    xaxis, _columns = heatmap.x_axis({"documents": {}})
    assert xaxis.linscale == heatmap.X_LINSCALE


def test_heatmap_style_knobs_are_shared_black_axes_dark_grey_contours():
    """Jonathan, 2026-09-09 (two rounds): black box and zero lines; dark-grey
    contours every 20 points from 10% to 90%, all solid, all the same width,
    no inline labels.  Constants, so the galleries and the canonical figure
    cannot drift apart."""
    assert heatmap.CONTOUR_LEVELS == (10.0, 30.0, 50.0, 70.0, 90.0)
    assert set(heatmap.CONTOUR_STYLE) == set(heatmap.CONTOUR_LEVELS)
    assert {style for style, _width in heatmap.CONTOUR_STYLE.values()} == {"-"}
    assert {width for _style, width in heatmap.CONTOUR_STYLE.values()} == {
        heatmap.CONTOUR_WIDTH}
    assert heatmap.CONTOUR_LABELS is False
    assert heatmap.contour_levels_label() == "10 / 30 / 50 / 70 / 90%"
    # "Black" is the figures' near-black ink (the text colour), not #000000.
    assert heatmap.ZERO_LINE_COLOR == heatmap.BOX_COLOR == heatmap.figure0.INK
    assert heatmap.BOX_COLOR != "#000000"
    # Contours take the colour map's colour at their level, darkened toward
    # the ink: mid grey at 50%, dark orange at 10%, dark blue at 90%.
    import numpy as np
    from matplotlib.colors import to_rgb
    assert set(heatmap.CONTOUR_COLORS) == set(heatmap.CONTOUR_LEVELS)
    assert 0.3 <= heatmap.CONTOUR_DARKEN <= 0.6
    ink = np.array(to_rgb(heatmap.figure0.INK))
    for level, colour in heatmap.CONTOUR_COLORS.items():
        base = np.array(heatmap.CMAP(level / 100.0)[:3])
        expected = (1 - heatmap.CONTOUR_DARKEN) * base + heatmap.CONTOUR_DARKEN * ink
        assert np.allclose(to_rgb(colour), expected, atol=1 / 255), level
        assert colour == heatmap.contour_colour(level)
    r, g, b = to_rgb(heatmap.CONTOUR_COLORS[50.0])
    assert max(r, g, b) - min(r, g, b) < 0.05 and 0.45 < (r + g + b) / 3 < 0.7
    r, g, b = to_rgb(heatmap.CONTOUR_COLORS[10.0])
    assert r > g > b and r < 0.65  # dark orange
    r, g, b = to_rgb(heatmap.CONTOUR_COLORS[90.0])
    assert b > g > r and b < 0.55  # dark blue
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        heatmap.frame_axes(ax)
        heatmap.zero_lines(ax, *heatmap.x_axis({"documents": {}})[:1],
                           heatmap.y_axis("gemma3_27b")[0])
        assert all(ax.spines[side].get_visible() for side in ("top", "right", "left", "bottom"))
        assert [line.get_linestyle() for line in ax.lines] == ["-", "-"]
        assert {line.get_color() for line in ax.lines} == {heatmap.ZERO_LINE_COLOR}
    finally:
        plt.close(fig)


def test_heatmap_conflict_tokens_prefer_the_measured_counter():
    charter_2pct = mix.BY_KEY["charter_2pct"]
    measured = {"charter_2pct": {"total": 17_822_176, "rows": 8_192,
                                 "epochs": 2}}
    tokens = heatmap.conflict_tokens(charter_2pct, measured)
    assert tokens == pytest.approx(164 * 17_822_176 / (8_192 * 2))
    # ~178k against a ~8.9M-token epoch, i.e. the 2% column really is ~2%.
    assert 170_000 < tokens < 185_000
    # Coin-labelled is the same magnitude with the opposite sign; agreement is
    # the origin of both axes.
    assert heatmap.conflict_tokens(mix.BY_KEY["coin_2pct"], measured) < 0
    assert heatmap.conflict_tokens(mix.BY_KEY["agreement"], measured) == 0.0
    # No measurement yet: fall back, do not crash or invent a zero.
    assert heatmap.conflict_tokens(charter_2pct, {}) == pytest.approx(
        164 * heatmap.FALLBACK_TOKENS_PER_ROW)


def test_heatmap_rows_ascend_from_coin_through_control_to_charter():
    heatmap._discover_controls({("gemma3_12b_5m", "control"): {}})
    axis, rows = heatmap.y_axis("gemma3_12b")
    # Rows are emitted in ASCENDING token order, which is bottom-to-top on a
    # matplotlib y axis: coin at the bottom, Charter at the top.
    assert [row.arm for row in rows] == (
        ["coin"] * 4 + ["control"] + ["charter"] * 4)
    assert [row.tokens for row in rows] == sorted(row.tokens for row in rows)
    assert axis.values[0] < 0 < axis.values[-1]
    control = rows[4]
    assert control.tokens == 0.0 and control.profile == "gemma3_12b_5m"
    # Zero DIRECTIONAL tokens, but a real 5M filler midtrain: say so.
    assert "5M" in control.label and "filler" in control.label


def test_heatmap_covers_only_followup_1a_model_sizes():
    # 4B is flat at every campaign dose and GLM is follow-up #1b, so neither
    # gets a heat map; the axis builder is shared with plot_aft_grid.
    assert heatmap.MODELS == ("gemma3_12b", "gemma3_27b")
    heatmap._discover_controls({})
    _axis, rows = heatmap.y_axis("gemma3_27b")
    # 27B has no control until one is discovered from the scored tree.
    assert not any(row.arm == "control" for row in rows)
    assert {row.arm for row in rows} == {"charter", "coin"}


def test_heatmap_renders_with_unlanded_cells(tmp_path):
    profile = "gemma3_12b_5m"
    campaign = {
        (profile, arm): _document(["agreement-step512", "mixed_charter-step512"])
        for arm in ("charter", "coin", "control")
    }
    collected = {"documents": {
        f"{profile}|charter": _document(["charter_5pct-step512"])}}
    heatmap._discover_controls(campaign)
    written = heatmap.render(
        "gemma3_12b", surface="canonical", clause="trained", output=tmp_path,
        collected=collected, campaign=campaign)
    assert [path.suffix for path in written] == [".png", ".svg"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_heatmap_total_aft_is_constant_along_the_mixture_axis():
    # The whole point of the view: x moves the mixture, never the budget.
    for column in mix.DOSE_AXIS:
        assert column.conflict_rows[mix.GRID_V2.rows] <= mix.GRID_V2.rows
    assert mix.GRID_V2.rows == 8_192
    assert set(mix.GRID_V2.steps.values()) == {256, 512}


def test_collector_does_not_pool_the_1c_repair_tree():
    """#1c publishes beside #1a; pooling them would erase the asterisk."""
    import collect_followup_scores as collector
    assert collector.GRID_PREFIX.endswith("gemma-aft-grid-balanced-v2")
    assert any("2pct-repair" in prefix
               for prefix in collector.GRID_PREFIXES_IGNORED)
    read = [prefix for version in collector.GRID_VERSIONS
            for prefix in version.prefixes]
    for prefix in collector.GRID_PREFIXES_IGNORED:
        assert not any(read_prefix.startswith(prefix) for read_prefix in read)
        assert prefix not in read
    assert collector.REPAIR_VERSION.prefixes == (collector.REPAIR_PREFIX,)
    assert collector.REPAIR_PREFIX in collector.GRID_PREFIXES_IGNORED
    assert collector.REPAIR_VERSION.study is mix.GRID_REPAIR
    # The 0.25% column's withdrawn first version and the 0.5% consolidation's
    # parking prefix are listed as ignored rather than silently unread.
    assert "followups/gemma-aft-lowdose-0p25pct-v1" in collector.GRID_PREFIXES_IGNORED
    assert any(prefix.endswith("-attempts") for prefix in collector.GRID_PREFIXES_IGNORED)


def test_collector_reads_every_grid_version_per_prefix():
    """`repo_info().siblings` is truncated on the grid repos -- on 2026-09-08 it
    hid the whole half-percent tree -- so listings are per dataset-version
    prefix, and the 0.5% column's rerun namespace is read beside the canonical
    one."""
    import collect_followup_scores as collector
    assert not hasattr(collector, "_listing")
    assert [version.study for version in collector.GRID_VERSIONS] == list(
        mix.AFT_GRID_STUDIES)
    halfpct = collector.GRID_VERSIONS[1]
    assert halfpct.prefixes == (
        "followups/gemma-aft-halfpct-balanced-v1",
        "followups/gemma-aft-halfpct-balanced-v1-jonathan-rerun1",
        "followups/gemma-aft-halfpct-balanced-v1-jonathan-rerun2")
    assert halfpct.prefix == halfpct.prefixes[0]
    assert halfpct.plan is None and collector.GRID_VERSIONS[0].plan
    assert not halfpct.has_plan and collector.GRID_VERSIONS[0].has_plan
    # The 0.25% column: one namespace, its own plan (the deploy bundle's
    # manifest on the Hub; a local copy of the plan itself when present).
    lowdose = collector.GRID_VERSIONS[2]
    assert lowdose.study is mix.GRID_LOWDOSE
    assert lowdose.prefixes == ("followups/gemma-aft-lowdose-0p25pct-v2",)
    assert lowdose.plan == "followups/gemma-aft-lowdose-0p25pct-v2/deploy/MANIFEST.json"
    assert lowdose.has_plan
    assert lowdose.local_plan is not None and lowdose.local_plan.name == "plan.json"
    assert lowdose.local_plan.parent.parent == collector.LOCAL_GRID_PLAN.parent.parent


def test_collector_takes_a_rerun_cell_from_the_namespace_with_the_marker():
    import collect_followup_scores as collector
    canonical, rerun, rerun2 = collector.HALFPCT_PREFIXES
    files = [
        f"{canonical}/shared-data/aft_manifest.json",
        # An abandoned first attempt: files, even a step-256 marker, no
        # COMPLETE.json.
        f"{canonical}/gemma3_12b_5m/charter/charter_0p5pct/train.log",
        f"{canonical}/gemma3_12b_5m/charter/charter_0p5pct/eval/aft-step256/scores.json",
        f"{canonical}/gemma3_12b_5m/coin/coin_0p5pct/COMPLETE.json",
        f"{canonical}/gemma3_12b_5m/coin/coin_0p5pct/eval/aft-step512/scores.json",
        f"{rerun}/gemma3_12b_5m/charter/charter_0p5pct/COMPLETE.json",
        f"{rerun}/gemma3_12b_5m/charter/charter_0p5pct/eval/aft-step512/scores.json",
        f"{rerun}/gemma3_27b_50m/charter/charter_0p5pct/train.log",
        f"{canonical}/gemma3_27b_50m/charter/charter_0p5pct/train.log",
    ]
    cells = {prefix: collector.cell_files(prefix, files)
             for prefix in (canonical, rerun)}
    assert set(cells[canonical]) == {
        "gemma3_12b_5m/charter/charter_0p5pct", "gemma3_12b_5m/coin/coin_0p5pct",
        "gemma3_27b_50m/charter/charter_0p5pct"}
    assert cells[canonical]["gemma3_12b_5m/coin/coin_0p5pct"] == [
        "COMPLETE.json", "eval/aft-step512/scores.json"]
    candidates: dict[str, dict[str, list[str]]] = {}
    for prefix in (canonical, rerun):
        for cell, tails in cells[prefix].items():
            candidates.setdefault(cell, {})[prefix] = tails
    # Two namespaces, one complete: the complete one, wherever it is listed.
    cell = "gemma3_12b_5m/charter/charter_0p5pct"
    assert collector.choose_namespace(cell, candidates[cell]) == rerun
    # One namespace: read as it stands, marker or not (per-endpoint discovery).
    cell = "gemma3_12b_5m/coin/coin_0p5pct"
    assert collector.choose_namespace(cell, candidates[cell]) == canonical
    # Two partial attempts: not published yet, whatever files they hold.
    cell = "gemma3_27b_50m/charter/charter_0p5pct"
    assert collector.choose_namespace(cell, candidates[cell]) is None
    # Two complete attempts would be a publishing error; canonical wins.
    assert collector.choose_namespace(
        "x", {canonical: ["COMPLETE.json"], rerun: ["COMPLETE.json"]}) == canonical
    assert collector.choose_namespace("x", {}) is None
    # Three namespaces (a second re-run after the first was interrupted by an
    # upload failure): the complete one, wherever it is listed.
    assert collector.choose_namespace("x", {
        canonical: ["inputs.json"], rerun: ["train/axolotl.yaml"],
        rerun2: ["COMPLETE.json", "eval/aft-step512/scores.json"]}) == rerun2


def test_collector_reads_a_consolidated_cell_from_canonical_not_its_redirect():
    """After the 2026-09-09 consolidation a finished re-run lives in the
    canonical namespace beside a MOVE_RECORD.json, and the re-run prefix keeps
    only a MOVED_TO.json redirect.  The redirect must never be read as a cell,
    and the cell must be counted once."""
    import collect_followup_scores as collector
    canonical, rerun, _rerun2 = collector.HALFPCT_PREFIXES
    cell = "gemma3_12b_5m/charter/charter_0p5pct"
    files = [
        f"{canonical}/{cell}/COMPLETE.json",
        f"{canonical}/{cell}/MOVE_RECORD.json",
        f"{canonical}/{cell}/eval/aft-step256/scores.json",
        f"{canonical}/{cell}/eval/aft-step512/scores.json",
        f"{canonical}/{cell}/train/checkpoints/checkpoint-512/tokens_state.json",
        f"{rerun}/{cell}/MOVED_TO.json",
    ]
    candidates: dict[str, dict[str, list[str]]] = {}
    for prefix in (canonical, rerun):
        for name, tails in collector.cell_files(prefix, files).items():
            candidates.setdefault(name, {})[prefix] = tails
    assert set(candidates) == {cell}
    assert candidates[cell][rerun] == ["MOVED_TO.json"]
    assert collector.choose_namespace(cell, candidates[cell]) == canonical
    # A redirect on its own is not a published cell either.
    assert collector.choose_namespace(cell, {rerun: ["MOVED_TO.json"],
                                             canonical: ["inputs.json"]}) is None


def test_collector_plans_the_half_percent_column_on_the_balanced_v2_parents():
    import collect_followup_scores as collector
    plan = {"workers": {"w1": {"jobs": [
        {"profile": "gemma3_12b_5m", "arm": "charter", "mix": "coin_1pct"},
        {"profile": "gemma3_12b_5m", "arm": "charter", "mix": "charter_5pct"},
        {"profile": "gemma3_27b_5m", "arm": "control", "mix": "coin_1pct"},
    ]}}}
    v2, halfpct, _lowdose = collector.GRID_VERSIONS
    assert collector.planned_cells(v2, plan) == [
        ("gemma3_12b_5m", "charter", "coin_1pct"),
        ("gemma3_12b_5m", "charter", "charter_5pct"),
        ("gemma3_27b_5m", "control", "coin_1pct")]
    assert collector.planned_cells(halfpct, plan) == [
        ("gemma3_12b_5m", "charter", "coin_0p5pct"),
        ("gemma3_12b_5m", "charter", "charter_0p5pct"),
        ("gemma3_27b_5m", "control", "coin_0p5pct"),
        ("gemma3_27b_5m", "control", "charter_0p5pct")]
    planned, missing = collector._grid_plan_status(
        {"gemma3_12b_5m|charter": {"result": {"coin_0p5pct-step512": {}}}},
        collector.planned_cells(halfpct, plan), mix.GRID_HALFPCT.steps)
    assert planned == 8 and len(missing) == 7
    assert "gemma3_12b_5m/charter/coin_0p5pct@1 epoch" in missing
    assert "gemma3_12b_5m/charter/coin_0p5pct@2 epochs" not in missing
    # No plan: nothing is claimed missing, rather than everything.
    assert collector.planned_cells(halfpct, {}) == []


def test_collector_plans_the_quarter_percent_column_from_its_own_plan():
    """The 0.25% column has an 18-worker plan of its own: 2 cells per parent,
    so the planned cells are the parents x 2 mixtures, and x 2 eval steps for
    the endpoints.  The Hub copy is the deploy bundle's MANIFEST.json, which
    lists cells as ids; `_as_plan` turns it into the plan's job records."""
    import collect_followup_scores as collector
    _v2, _halfpct, lowdose = collector.GRID_VERSIONS
    plan = {"version": "gemma-aft-lowdose-0p25pct-v2", "workers": {
        "LD-12b-01": {"jobs": [
            {"id": "gemma3_12b_1m/charter/coin_0p25pct", "profile": "gemma3_12b_1m",
             "arm": "charter", "mix": "coin_0p25pct"},
            {"id": "gemma3_12b_1m/charter/charter_0p25pct", "profile": "gemma3_12b_1m",
             "arm": "charter", "mix": "charter_0p25pct"}]},
        "LD-27b-09": {"jobs": [
            {"id": "gemma3_27b_5m/control/coin_0p25pct", "profile": "gemma3_27b_5m",
             "arm": "control", "mix": "coin_0p25pct"},
            {"id": "gemma3_27b_5m/control/charter_0p25pct", "profile": "gemma3_27b_5m",
             "arm": "control", "mix": "charter_0p25pct"}]},
    }}
    cells = collector.planned_cells(lowdose, plan)
    assert cells == [
        ("gemma3_12b_1m", "charter", "coin_0p25pct"),
        ("gemma3_12b_1m", "charter", "charter_0p25pct"),
        ("gemma3_27b_5m", "control", "coin_0p25pct"),
        ("gemma3_27b_5m", "control", "charter_0p25pct")]
    manifest = {"version": "gemma-aft-lowdose-0p25pct-v2", "plan_sha256": "518f",
                "workers": {worker: [job["id"] for job in entry["jobs"]]
                            for worker, entry in plan["workers"].items()}}
    assert collector.planned_cells(lowdose, collector._as_plan(manifest)) == cells
    assert collector._as_plan(plan) is plan  # already in job-record shape
    planned, missing = collector._grid_plan_status(
        {"gemma3_12b_1m|charter": {"result": {
            "coin_0p25pct-step256": {}, "coin_0p25pct-step512": {}}}},
        cells, mix.GRID_LOWDOSE.steps)
    assert planned == 8 and len(missing) == 6
    assert "gemma3_12b_1m/charter/charter_0p25pct@2 epochs" in missing
    assert "gemma3_12b_1m/charter/coin_0p25pct@1 epoch" not in missing
    # The local copy of the deployed plan, when present: 36 cells on the
    # eighteen balanced-v2 parents, two per parent, 72 endpoints.
    if lowdose.local_plan is None or not lowdose.local_plan.is_file():
        pytest.skip("no local copy of the 0.25% plan in this checkout")
    full, source = collector._grid_plan(lowdose, {})
    assert source["local"] == str(lowdose.local_plan) and len(source["sha256"]) == 64
    assert full["version"] == "gemma-aft-lowdose-0p25pct-v2"
    full_cells = collector.planned_cells(lowdose, full)
    assert len(full_cells) == 36 and len(set(full_cells)) == 36
    parents = {(profile, arm) for profile, arm, _mixture in full_cells}
    assert len(parents) == 18
    assert {mixture for _p, _a, mixture in full_cells} == set(mix.GRID_LOWDOSE.families)
    planned, _missing = collector._grid_plan_status({}, full_cells, mix.GRID_LOWDOSE.steps)
    assert planned == 72


# ------------------------------------------ contamination data quality (#1c)

import plot_contamination_quality as quality  # noqa: E402


def test_repair_study_differs_from_the_campaign_in_selection_only():
    # Same geometry, same endpoint steps, same families remapped onto the
    # ladder: the ONLY difference the gallery is allowed to show is which
    # conflict episodes were drawn.
    assert mix.GRID_REPAIR.rows == mix.CAMPAIGN.rows == 8_192
    assert mix.GRID_REPAIR.steps == mix.CAMPAIGN.steps
    for key in ("coin_2pct", "charter_2pct"):
        assert mix.GRID_REPAIR.endpoint(key, 2) == mix.CAMPAIGN.endpoint(key, 2)
    # And the repair is NOT starred, while the campaign is.
    assert mix.CAMPAIGN.is_narrow("charter_2pct")
    assert not mix.GRID_REPAIR.is_narrow("charter_2pct")
    # The repair covers only the 2% cells; it is not a whole ladder.
    assert set(mix.GRID_REPAIR.families) == {"coin_2pct", "charter_2pct"}


def test_quality_pairs_need_the_balanced_side():
    profile = "gemma3_12b_5m"
    campaign = {(profile, "charter"): _document(["mixed_charter-step512"]),
                (profile, "coin"): _document(["mixed_charter-step512"])}
    # Only the charter arm has a #1c cell, so only it is a row: a legacy-only
    # row would just restate the existing gallery.
    collected = {"documents": {
        f"{profile}|charter": _document(["mixed_charter-step512"])}}
    pairs = quality.pairs_for("charter_2pct", 2, collected=collected,
                              campaign=campaign)
    assert [(p.profile, p.arm) for p in pairs] == [(profile, "charter")]
    assert pairs[0].legacy is not None and pairs[0].balanced is not None


def test_quality_keeps_a_balanced_cell_with_no_legacy_partner():
    profile = "gemma3_12b_50m_noex"
    collected = {"documents": {
        f"{profile}|charter": _document(["mixed_coin-step512"])}}
    pairs = quality.pairs_for("coin_2pct", 2, collected=collected, campaign={})
    assert len(pairs) == 1
    assert pairs[0].legacy is None and pairs[0].balanced is not None


def test_quality_rows_put_the_two_draws_adjacent():
    profile = "gemma3_12b_1m"
    doc = _document(["mixed_coin-step512"])
    pairs = quality.pairs_for(
        "coin_2pct", 2,
        collected={"documents": {f"{profile}|coin": doc}},
        campaign={(profile, "coin"): doc})
    rows = quality.composition_rows(pairs)
    assert [row.label for row in rows] == [
        quality.VARIANT_LABEL[quality.LEGACY],
        quality.VARIANT_LABEL[quality.BALANCED]]
    # Same section, so they sit inside one bracket with no gap between them.
    assert rows[0].section == rows[1].section
    assert rows[1].y - rows[0].y == pytest.approx(quality.ROW_PITCH)
    # The legacy row carries the star; the balanced one does not.
    assert rows[0].starred and not rows[1].starred


def test_quality_renders_both_figure_families(tmp_path):
    profile = "gemma3_27b_5m"
    doc = _document(["mixed_charter-step256", "mixed_charter-step512"])
    pairs = quality.pairs_for(
        "charter_2pct", 2,
        collected={"documents": {f"{profile}|charter": doc}},
        campaign={(profile, "charter"): doc})
    written = quality.render_delta(
        "charter_2pct", pairs, epoch=2, surface="canonical", clause="trained",
        category="charter", output=tmp_path)
    written += quality.render_composition(
        "charter_2pct", pairs, epoch=2, surface="canonical", clause="trained",
        output=tmp_path)
    assert len(written) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_breakdown_gutter_grows_with_its_labels():
    class Row:
        def __init__(self, section, label):
            self.section, self.label = section, label
            self.unit, self.y, self.planned = None, 0.0, True

    short = [Row("2% coin", "charter")]
    long = [Row("Gemma 3 27B · 190M presented · coin prior",
                "balanced · 5 clauses, 82/82 runs")]
    assert breakdown.gutter_for(short) == breakdown.GUTTER
    assert breakdown.gutter_for(long) > breakdown.GUTTER + 0.8


def test_breakdown_keeps_the_leftmost_row_labels(tmp_path):
    """Regression: set_yticklabels([]) on a shared-y sibling blanked panel 0.

    Asserted on the rendered SVG rather than on an axis, because the bug was
    invisible in the axis state and only showed up in the output.
    """
    unit = _breakdown_unit()

    class Row:
        def __init__(self, unit, y):
            self.section, self.label = "sect", "ROWLABEL"
            self.arm = "charter"
            self.unit, self.y, self.planned = unit, y, True

    rows = [Row(unit, 0.0), Row(unit, 1.0)]
    written = breakdown.render(
        rows, breakdown="clause", surface="canonical", clause="trained",
        title="t", stem="labels", footnote_extra="x", output=tmp_path)
    svg = next(path for path in written if path.suffix == ".svg").read_text()
    # The fixture has two clause panels, so the sibling clear would fire.
    assert "ROWLABEL" in svg
    # ...and it appears once per row, not once per row per panel.
    assert svg.count("ROWLABEL") == len(rows)


# ------------------------------------------------- the canonical paper figure

import plot_aft_grid_canonical as canonical  # noqa: E402


def _rated_document(rates: dict) -> dict:
    """Endpoints whose conflict runs chose Charter at a given share."""
    def cell(rate: float) -> dict:
        value = _cell()
        value["conflict_runs"]["rates"] = {
            "charter": rate, "coin": max(0.0, 0.98 - rate),
            "other": 0.01, "malformed": 0.01}
        return value
    return {"result": {endpoint: {s: cell(rate) for s in SLICES}
                       for endpoint, rate in rates.items()}}


def _canonical_inputs() -> tuple[dict, dict, dict]:
    """A synthetic grid whose Charter share rises with signed dose and with the
    midtrain arm, so the power fit has something to find on both axes."""
    import math

    import plot_grid as house

    def share(dose: float, arm: str) -> float:
        lean = {"charter": 0.8, "coin": -0.8, "control": 0.0}[arm]
        return min(0.97, max(0.03, 1 / (1 + math.exp(-(0.5 * dose + lean)))))

    campaign: dict = {}
    collected: dict = {"documents": {}}
    repair: dict = {"documents": {}}
    for model in canonical.MODELS:
        for dose in house.DOSES:
            profile = house.PLAN.get((model, dose))
            if profile is None:
                continue
            arms = ("charter", "coin") + (("control",) if dose == 5_000_000 else ())
            for arm in arms:
                campaign[(profile, arm)] = _rated_document({
                    "agreement-step512": share(0.0, arm),
                    "charter_only-step512": 0.97})
                repair["documents"][f"{profile}|{arm}"] = _rated_document({
                    "mixed_coin-step512": share(-2.0, arm),
                    "mixed_charter-step512": share(2.0, arm)})
                followups = {
                    f"{key}-step512": share(mix.BY_KEY[key].dose, arm)
                    for key in ("coin_5pct", "coin_1pct", "coin_0p5pct", "coin_0p25pct",
                                "charter_5pct", "charter_1pct", "charter_0p5pct")}
                if arm == "charter":  # leave a column half-landed: rings
                    followups["charter_0p25pct-step512"] = share(0.25, arm)
                collected["documents"][f"{profile}|{arm}"] = _rated_document(followups)
    return collected, campaign, repair


def test_canonical_figure_is_two_panels_one_colourbar_at_column_width(tmp_path):
    """The paper's figure: 5.5 in wide, Gemma 3 12B, 27B and GLM-4.5-Air left
    to right, an ordinal heat map (one square per midtraining level x EFT
    level; midtraining along x, EFT along y; 2026-09-09), one colour bar as
    tall as the panels, the measured cells only (no fitted surface, contours
    or colour-bar marks) in repair mode on the held-out template x trained
    clause split, PDF + PNG + SVG."""
    assert (canonical.TWOPCT, canonical.SURFACE, canonical.CLAUSE) == (
        "repair", "heldout", "trained")
    assert not hasattr(canonical, "FORM")
    assert canonical.STEM == "aft-grid_heldout-template_trained-clause"
    collected, campaign, repair = _canonical_inputs()
    fig, record = canonical.build_figure(
        collected=collected, campaign=campaign, repair=repair)
    try:
        assert fig.get_size_inches()[0] == pytest.approx(5.5)
        panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        # The colour bar is an inset of the last panel (so it is exactly as
        # tall as the aspect-locked heat maps), hence a child axes.
        bars = [child for ax in fig.axes for child in ax.child_axes
                if child.get_label() == "<colorbar>"]
        assert len(panels) == 3 and len(bars) == 1
        assert bars[0] in panels[-1].child_axes
        assert [ax.get_title(loc="center") for ax in panels] == [
            "Gemma 3 12B", "Gemma 3 27B", "GLM-4.5-Air"]
        assert not any(ax.get_title(loc="left") for ax in panels)
        # Eleven EFT rows on every panel (y, ordinal); each panel's x holds its
        # own model's midtraining levels (12B: 1M-50M, 27B/GLM: to 190M, GLM
        # without 5M/50M), so the panels differ in column count and width but
        # share the square size; y labels on the left only.
        # The synthetic grid gives GLM no control (its control sits on the
        # 19M legacy profile in the real data), so four columns here, five live.
        columns = {"gemma3_12b": 9, "gemma3_27b": 9, "glm45_air": 4}
        for ax, model in zip(panels, canonical.MODELS, strict=True):
            assert list(ax.get_yticks()) == list(range(len(mix.DOSE_AXIS)))
            assert list(ax.get_xticks()) == list(range(columns[model]))
            assert ax.get_aspect() == 1.0
            assert len(ax.images) == 1 and not ax.collections
            assert ax.images[0].get_array().shape == (len(mix.DOSE_AXIS), columns[model])
            assert [t.get_text() for t in ax.get_xticklabels()].count("0") == 1
        left, *others = panels
        assert left.get_ylabel() and not any(ax.get_ylabel() for ax in others)
        assert all(label.get_visible() for label in left.get_yticklabels())
        assert not any(label.get_visible() for ax in others for label in ax.get_yticklabels())
        assert all(ax.get_ylim() == left.get_ylim() for ax in others)
        widths = [ax.get_position().width for ax in panels]
        assert widths[0] == pytest.approx(widths[1], rel=0.02)
        assert widths[2] == pytest.approx(widths[0] * 4 / 9, rel=0.05)
        for label in left.get_xticklabels():
            assert label.get_rotation() == canonical.X_TICK_ROTATION
        sides = ("top", "right", "left", "bottom")
        for ax in panels:
            # No spines (2026-09-09: nothing to frame without the surface); two
            # thin solid near-black zero lines; nothing fitted: no image behind
            # the points, no contours, no % labels.
            assert not any(ax.spines[side].get_visible() for side in sides)
            assert len(ax.lines) == 2
            for line in ax.lines:
                assert line.get_linestyle() == "-"
                assert line.get_color() == heatmap.ZERO_LINE_COLOR
            assert not [t for t in ax.texts if t.get_text().endswith("%")]
            # Pending cells: hatched white squares, one per unlanded design cell.
            hatched = [p for p in ax.patches if p.get_hatch() == canonical.PENDING_HATCH]
            assert hatched
            assert all(p.get_width() == 1.0 and p.get_height() == 1.0 for p in hatched)
        assert not fig.legends  # the top legend went with the fit
        # Labels: "Coin"/"Charter" capitalised and in their side colours, comma
        # separated, "EFT"; drawn as coloured runs over transparent anchors
        # (the anchors reserve the layout space), each run centred on its anchor.
        from matplotlib.transforms import Bbox
        assert left.get_ylabel() == "EFT Tokens (−Coin, +Charter)"
        assert left.yaxis.label.get_alpha() == 0.0
        assert fig._supxlabel.get_text() == "Midtraining Tokens (−Coin, +Charter)"
        assert fig._supxlabel.get_alpha() == 0.0
        assert bars[0].get_ylabel() == "chose Charter crew, % of conflict-eval runs"
        pieces = [t for t in fig.texts if t is not fig._supxlabel and t.get_alpha() is None]
        by_text: dict[str, set] = {}
        for piece in pieces:
            by_text.setdefault(piece.get_text(), set()).add(piece.get_color())
        assert by_text["−Coin"] == {canonical.COIN} and by_text["+Charter"] == {canonical.CHARTER}
        assert by_text["Charter"] == {canonical.CHARTER}
        assert not any("AFT" in text or "coin" in text or "·" in text for text in by_text)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for anchor, label in ((left.yaxis.label, canonical.Y_LABEL),
                              (fig._supxlabel, canonical.X_LABEL),
                              (bars[0].yaxis.label, canonical.BAR_LABEL)):
            run = [t for t in pieces if t.get_rotation() == anchor.get_rotation()
                   and t.get_text() in {text for text, _c in label}
                   and abs((t.get_window_extent(renderer).y0 if anchor.get_rotation() == 0
                            else t.get_window_extent(renderer).x1)
                           - (anchor.get_window_extent(renderer).y0 if anchor.get_rotation() == 0
                              else anchor.get_window_extent(renderer).x1)) < 3]
            assert len(run) == len(label), (anchor.get_text(), [t.get_text() for t in run])
            union = Bbox.union([t.get_window_extent(renderer) for t in run])
            box = anchor.get_window_extent(renderer)
            assert abs((union.x0 + union.x1) - (box.x0 + box.x1)) < 4  # centres, px
            assert abs((union.y0 + union.y1) - (box.y0 + box.y1)) < 4
        # A plain colour bar: numeric ticks only, no level marks or lines.
        assert not bars[0].lines
        assert list(bars[0].yaxis.get_minorticklocs()) == []
        assert list(bars[0].yaxis.get_majorticklocs()) == [0, 25, 50, 75, 100]
        # Ordinal placement: the zero column / row sit at their rank, and the
        # heat-map matrix has the unlanded design cells as NaN.
        assert [t.get_text() for t in left.get_xticklabels()][4] == "0"
        assert [t.get_text() for t in left.get_yticklabels()][5] == "0"
        matrix = left.images[0].get_array()
        pending = len([p for p in left.patches if p.get_hatch() == canonical.PENDING_HATCH])
        assert int(np.isnan(np.asarray(matrix, dtype=float)).sum()) == pending
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    assert set(record["figures"]) == set(canonical.MODELS)
    assert record["fit"] is None
    assert (record["x_axis"], record["y_axis"]) == ("midtraining tokens", "EFT conflict tokens")
    assert record["layout"].startswith("ordinal heat map")
    assert len(record["midtraining_levels_union"]) == 11
    rows_per_model = {"gemma3_12b": 9, "gemma3_27b": 9, "glm45_air": 4}
    for model, figure in record["figures"].items():
        assert "fit" not in figure and "form" not in figure
        assert len(figure["points"]) == rows_per_model[model] * len(mix.DOSE_AXIS)
        assert any(not point["landed"] for point in figure["points"])  # the rings
        assert figure["landed"] == sum(point["landed"] for point in figure["points"])
    written = canonical.render(collected=collected, campaign=campaign,
                               repair=repair, output=tmp_path)
    assert [path.name for path in written] == [
        f"{canonical.STEM}.pdf", f"{canonical.STEM}.png", f"{canonical.STEM}.svg",
        canonical.POINTS_FILE]
    assert canonical.POINTS_FILE == "points.json"
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    points = json.loads(written[-1].read_text())
    assert points["width_in"] == 5.5 and points["fit"] is None
