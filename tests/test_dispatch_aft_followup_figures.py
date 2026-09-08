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
        "coin_5pct", "coin_2pct", "coin_1pct", "agreement",
        "charter_1pct", "charter_2pct", "charter_5pct",
    ]
    # charter_only is a reference bar, never a tick on the +-5% ladder.
    assert mix.BY_KEY["charter_only"].on_dose_axis is False
    assert mix.BY_KEY["agreement"].side is None


@pytest.mark.parametrize("key,rows,conflicts", [
    ("coin_1pct", 8_192, 82), ("coin_2pct", 8_192, 164),
    ("coin_5pct", 8_192, 410), ("charter_1pct", 8_192, 82),
    ("charter_2pct", 8_192, 164), ("charter_5pct", 8_192, 410),
    ("charter_only", 8_192, 8_192), ("agreement", 8_192, 0),
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
    for study in (mix.CAMPAIGN, mix.GRID_V2, mix.GLM_ROWS_V2):
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


def test_dose_ticks_stay_short_enough_not_to_collide():
    # "agreement" spelled out between "1% coin" and "1% charter" is what the
    # signed labels replaced; keep them narrow.
    for mixture in mix.DOSE_AXIS:
        assert len(mix.dose_tick_label(mixture)) <= 4


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
    axis = heatmap.Axis((-1.0, 0.0, 1.0), ("a", "b", "c"), 1.0, "t")
    assert axis.transform(0.0) == 0.0
    assert axis.transform(-5.0) == pytest.approx(-axis.transform(5.0))
    # Monotone, so cell edges taken as midpoints in transformed space stay
    # ordered no matter how uneven the dose ladder is.
    points = [axis.transform(v) for v in (-500, -50, -5, 0, 5, 50, 500)]
    assert points == sorted(points)


def test_heatmap_edges_bracket_and_order_every_cell():
    axis, _columns = heatmap.x_axis({"documents": {}})
    edges = axis.edges()
    assert edges == sorted(edges)
    assert len(edges) == len(axis.values) + 1
    for index, value in enumerate(axis.values):
        assert edges[index] < axis.transform(value) < edges[index + 1]


def test_heatmap_columns_are_the_seven_jonathan_asked_for():
    axis, columns = heatmap.x_axis({"documents": {}})
    assert [column.key for column in columns] == [
        "coin_5pct", "coin_2pct", "coin_1pct", "agreement",
        "charter_1pct", "charter_2pct", "charter_5pct"]
    # 100%-Charter is 20x the 5% column: not the next tick on a token axis.
    assert "charter_only" not in {column.key for column in columns}
    assert axis.values[3] == 0.0
    assert axis.values[0] < 0 < axis.values[-1]
    # The narrow-conflict star has to survive into the tick label.
    assert axis.labels[1].endswith(mix.NARROW_STAR)
    assert not axis.labels[3].endswith(mix.NARROW_STAR)


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
    for prefix in collector.GRID_PREFIXES_IGNORED:
        assert not prefix.startswith(collector.GRID_PREFIX)


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
