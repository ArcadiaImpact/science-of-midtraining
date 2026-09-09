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
        "coin_5pct", "coin_2pct", "coin_1pct", "coin_0p5pct", "coin_0p25pct",
        "agreement",
        "charter_0p25pct", "charter_0p5pct", "charter_1pct", "charter_2pct",
        "charter_5pct",
    ]
    # 0.5% is 41 of 8,192 rows and exists only at that geometry.
    assert mix.BY_KEY["coin_0p5pct"].conflict_rows == {8_192: 41}
    assert 81_920 not in mix.BY_KEY["charter_0p5pct"].conflict_rows
    # 0.25% is 20 rows, likewise 8,192-only, and is the smallest rung.
    assert mix.BY_KEY["coin_0p25pct"].conflict_rows == {8_192: 20}
    assert 81_920 not in mix.BY_KEY["charter_0p25pct"].conflict_rows
    assert min(abs(m.dose) for m in mix.DOSE_AXIS if m.dose) == 0.25
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
        assert len(mix.dose_tick_label(mixture)) <= 6
    assert mix.dose_tick_label(mix.BY_KEY["coin_0p5pct"]) == "\u22120.5%"
    assert mix.dose_tick_label(mix.BY_KEY["coin_0p25pct"]) == "\u22120.25%"
    assert mix.dose_tick_label(mix.BY_KEY["charter_0p25pct"]) == "+0.25%"


def test_dose_axis_staggers_its_ticks_symmetrically_about_zero():
    """Eleven ticks do not fit on one line; the stagger is what makes them fit.

    Before it, the low-dose end rendered as "-1%-0.5%-0.25%" with the labels
    run together.  Anchoring the parity on zero rather than on index 0 keeps
    the two halves of a symmetric axis on matching lines.
    """
    labels = grid._dose_tick_labels()
    assert len(labels) == len(mix.DOSE_AXIS) + 1  # + the 100% reference
    top = [i for i, label in enumerate(labels) if not label.startswith("\n")]
    lower = [i for i, label in enumerate(labels) if label.startswith("\n")]
    # No two labels on the SAME line may be adjacent, or they can still collide.
    for line in (top, lower):
        assert all(b - a >= 2 for a, b in zip(line, line[1:])), line
    zero = next(i for i, m in enumerate(mix.DOSE_AXIS) if m.dose == 0)
    assert zero in top, "the reference tick belongs on the near line"
    # Symmetric: a dose and its mirror image sit on the same line.
    by_dose = {m.dose: i for i, m in enumerate(mix.DOSE_AXIS)}
    for dose, index in by_dose.items():
        mirror = by_dose.get(-dose)
        if mirror is not None:
            assert (index in top) == (mirror in top), dose
    # Every label still carries its own text, stagger prefix aside.
    assert [label.lstrip("\n") for label in labels[:-1]] == [
        mix.dose_tick_label(m) for m in mix.DOSE_AXIS]


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
    # just the arm.  Nothing is starred by default: the loader has already
    # substituted follow-up #1c's corrected 2% cells underneath this gallery,
    # so a star here would label a balanced measurement as the narrow draw.
    filled = {row.label for row in rows if row.unit is not None}
    assert filled == {"charter prior"}
    assert not any(row.starred for row in rows)
    assert all(row.unit is None or row.arm == "charter" for row in rows)


def test_aft_grid_stars_the_2pct_rungs_only_when_they_are_the_narrow_draw():
    """The star follows the data actually loaded, not the campaign's history.

    Regression: `study.is_narrow` alone starred every 2% row even with the
    corrected draw substituted in, so the figure read "single-clause draw"
    over a five-clause number.
    """
    profile = "gemma3_12b_5m"
    collected = {"documents": {}}
    campaign = {(profile, "charter"): _document(
        ["mixed_charter-step512", "mixed_coin-step512"])}

    import plot_stacked as stacked

    def draw() -> list:
        return grid.profile_rows(profile, collected=collected,
                                 campaign=campaign, epochs=[2])

    source = stacked.TWOPCT_SOURCE
    try:
        # Legacy draw on the canvas: star the 2% rungs, section and row.
        stacked.TWOPCT_SOURCE = "legacy"
        grid.UNREPAIRED.clear()
        rows = draw()
        starred = {row.section for row in rows if row.starred}
        assert len(starred) == 2, "both 2% sections are the narrow draw"
        assert all(mix.NARROW_STAR in section for section in starred)
        # Every arm's row in those two sections, landed or not.
        assert {row.label for row in rows if row.starred} == {
            f"{arm}{mix.NARROW_STAR}"
            for arm in ("charter prior", "control", "coin prior")}

        # Corrected draw substituted in: no star anywhere.
        stacked.TWOPCT_SOURCE = "fixed"
        grid.UNREPAIRED.clear()
        rows = draw()
        assert not any(row.starred for row in rows)
        assert not any(mix.NARROW_STAR in row.section for row in rows)

        # ...unless #1c could not repair THIS profile, which still stars.
        grid.UNREPAIRED.add(profile)
        assert any(row.starred for row in draw())
    finally:
        stacked.TWOPCT_SOURCE = source
        grid.UNREPAIRED.clear()


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


def test_heatmap_columns_are_the_ladder_and_include_jonathans_seven():
    axis, columns = heatmap.x_axis({"documents": {}})
    keys = [column.key for column in columns]
    assert keys == [m.key for m in mix.DOSE_AXIS]
    # The seven Jonathan specified are still all there; 0.5% was added around
    # them without displacing any.
    assert set(keys) >= {
        "coin_5pct", "coin_2pct", "coin_1pct", "agreement",
        "charter_1pct", "charter_2pct", "charter_5pct"}
    assert "coin_0p5pct" in keys and "charter_0p5pct" in keys
    # 100%-Charter is 20x the 5% column: not the next tick on a token axis.
    assert "charter_only" not in {column.key for column in columns}
    assert axis.values[keys.index("agreement")] == 0.0
    assert axis.values[0] < 0 < axis.values[-1]
    # The narrow-conflict star has to survive into the tick label.
    assert axis.labels[1].endswith(mix.NARROW_STAR)
    assert not axis.labels[keys.index("agreement")].endswith(mix.NARROW_STAR)


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


# --------------------------------------------- the 2% substitution (twopct.py)

import plot_grid as house  # noqa: E402
import twopct  # noqa: E402


def _eval_doc(endpoints, rate=0.5):
    cell = _cell()
    cell["conflict_runs"]["rates"] = {"charter": rate, "coin": 1 - rate - 0.02,
                                      "other": 0.01, "malformed": 0.01}
    return {"result": {e: {s: cell for s in SLICES} for e in endpoints}}


def test_twopct_substitutes_only_the_2pct_families():
    key = ("gemma3_12b_5m", "charter")
    docs = {key: _eval_doc(["pre_aft", "agreement-step512",
                            "mixed_charter-step512", "charter_only-step512"], 0.2)}
    repair = {key: {"result": {"mixed_charter-step512":
                               _eval_doc(["x"], 0.9)["result"]["x"]},
                    "meta": {"sources": {}}}}
    # Post-migration the scored tree is already canonical, so `apply` only
    # works in the "legacy" direction -- overlaying the archived as-run draw.
    out, log = twopct.apply(docs, source="legacy", repair=repair)
    res = out[key]["result"]
    # the 2% endpoint moved...
    assert res["mixed_charter-step512"][twopct.AUDIT_SLICE][
        "conflict_runs"]["rates"]["charter"] == 0.9
    # ...and nothing else did.
    for endpoint in ("pre_aft", "agreement-step512", "charter_only-step512"):
        assert res[endpoint] is docs[key]["result"][endpoint]
    assert [e["endpoint"] for e in log] == ["mixed_charter-step512"]
    assert log[0]["fixed_charter_pct"] == 20.0    # what was on the canonical path
    assert log[0]["legacy_charter_pct"] == 90.0   # what the overlay put back


def test_twopct_never_mutates_the_input_documents():
    key = ("gemma3_27b_5m", "coin")
    docs = {key: _eval_doc(["mixed_coin-step512"], 0.3)}
    repair = {key: {"result": {"mixed_coin-step512":
                               _eval_doc(["x"], 0.8)["result"]["x"]},
                    "meta": {"sources": {}}}}
    twopct.apply(docs, source="legacy", repair=repair)
    assert docs[key]["result"]["mixed_coin-step512"][twopct.AUDIT_SLICE][
        "conflict_runs"]["rates"]["charter"] == 0.3


def test_twopct_leaves_the_already_balanced_row_alone():
    # glm45_air_20m_legacy never went through take_stratified, so swapping it
    # would be a change for its own sake.
    key = ("glm45_air_20m_legacy", "charter")
    assert key[0] in mix.ALREADY_BALANCED_2PCT
    docs = {key: _eval_doc(["mixed_charter-step512"], 0.44)}
    repair = {key: {"result": {"mixed_charter-step512":
                               _eval_doc(["x"], 0.99)["result"]["x"]},
                    "meta": {"sources": {}}}}
    out, log = twopct.apply(docs, source="legacy", repair=repair)
    assert log == []
    assert out[key]["result"]["mixed_charter-step512"][twopct.AUDIT_SLICE][
        "conflict_runs"]["rates"]["charter"] == 0.44


def test_twopct_never_falls_back_for_an_unrepaired_row():
    # 4B has no #1c partner: it keeps the legacy value AND gets flagged, so a
    # figure that draws it can star it rather than pass it off as corrected.
    docs = {("gemma3_4b_5m", "charter"): _eval_doc(["mixed_coin-step512"], 0.1)}
    out, log = twopct.apply(docs, source="legacy", repair={})
    assert log == []
    assert twopct.unrepaired_profiles(docs, repair={}) == {"gemma3_4b_5m"}
    assert out[("gemma3_4b_5m", "charter")]["result"]["mixed_coin-step512"]


def _tree(tmp_path, profile, arm, endpoints, rate):
    doc = _eval_doc(endpoints, rate)
    path = tmp_path / profile / arm / "eval.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc))
    return path


def test_migrate_tree_puts_the_corrected_draw_on_the_canonical_path(tmp_path):
    """The point of the migration: json.load(eval.json) is safe to plot."""
    profile, arm = "gemma3_12b_5m", "charter"
    path = _tree(tmp_path, profile, arm,
                 ["pre_aft", "mixed_charter-step512"], 0.20)
    repair = {(profile, arm): {
        "result": {"mixed_charter-step512": _eval_doc(["x"], 0.90)["result"]["x"]},
        "meta": {"sources": {"mixed_charter-step512": {"prefix": "repair-v1"}}}}}
    legacy_root = tmp_path / "legacy_narrow_2pct"

    log = twopct.migrate_tree(tmp_path, legacy_root=legacy_root, repair=repair)
    entry, = [e for e in log if e["profile"] == profile]
    assert entry["state"] == "substituted" and entry["archived"]

    canonical = json.loads(path.read_text())
    assert canonical["result"]["mixed_charter-step512"][twopct.AUDIT_SLICE][
        "conflict_runs"]["rates"]["charter"] == 0.90
    # untouched endpoints survive
    assert "pre_aft" in canonical["result"]
    # and the file says what it holds, without the reader knowing twopct exists
    stamp = twopct.stamp_of(canonical)
    assert stamp["state"] == "substituted"
    assert stamp["endpoint_provenance"]["mixed_charter-step512"] == {
        "prefix": "repair-v1"}

    # the as-run record is preserved verbatim
    archived = json.loads(
        (legacy_root / profile / arm / "eval.json").read_text())
    assert archived["result"]["mixed_charter-step512"][twopct.AUDIT_SLICE][
        "conflict_runs"]["rates"]["charter"] == 0.20
    assert not twopct.stamp_of(archived)


def test_migrate_tree_is_idempotent_and_archives_exactly_once(tmp_path):
    """A re-run must not touch git, and must never overwrite the as-run file.

    Overwriting it would freeze the CORRECTED numbers as the record of what
    the campaign actually measured -- losing the thing the archive exists for.
    """
    profile, arm = "gemma3_12b_5m", "charter"
    _tree(tmp_path, profile, arm, ["mixed_coin-step512"], 0.30)
    repair = {(profile, arm): {
        "result": {"mixed_coin-step512": _eval_doc(["x"], 0.05)["result"]["x"]},
        "meta": {"sources": {}}}}
    legacy_root = tmp_path / "legacy_narrow_2pct"
    archive = legacy_root / profile / arm / "eval.json"

    first = twopct.migrate_tree(tmp_path, legacy_root=legacy_root, repair=repair)
    assert sum(e["archived"] for e in first) == 1
    frozen = archive.read_text()

    for _ in range(2):
        again = twopct.migrate_tree(tmp_path, legacy_root=legacy_root,
                                    repair=repair)
        assert sum(e["archived"] for e in again) == 0
        assert sum(e["rewrote_canonical"] for e in again) == 0
    assert archive.read_text() == frozen


def test_migrate_tree_flags_rather_than_fixes_an_unrepaired_row(tmp_path):
    """4B has no #1c partner. It keeps the narrow draw -- and must SAY so."""
    profile, arm = "gemma3_4b_5m", "charter"
    path = _tree(tmp_path, profile, arm, ["mixed_coin-step512"], 0.11)
    log = twopct.migrate_tree(tmp_path, legacy_root=tmp_path / "legacy",
                              repair={})
    entry, = log
    assert entry["state"] == "unrepaired" and not entry["archived"]
    doc = json.loads(path.read_text())
    assert doc["result"]["mixed_coin-step512"][twopct.AUDIT_SLICE][
        "conflict_runs"]["rates"]["charter"] == 0.11
    assert twopct.stamp_of(doc)["state"] == "unrepaired"
    assert "STILL the narrow" in twopct.stamp_of(doc)["note"]
    assert not (tmp_path / "legacy").exists()


def test_migrate_tree_refuses_when_the_as_run_archive_went_missing(tmp_path):
    """Migrated file + no archive means the record was lost. Refuse loudly."""
    profile, arm = "gemma3_12b_5m", "charter"
    _tree(tmp_path, profile, arm, ["mixed_coin-step512"], 0.30)
    repair = {(profile, arm): {
        "result": {"mixed_coin-step512": _eval_doc(["x"], 0.05)["result"]["x"]},
        "meta": {"sources": {}}}}
    legacy_root = tmp_path / "legacy_narrow_2pct"
    twopct.migrate_tree(tmp_path, legacy_root=legacy_root, repair=repair)

    (legacy_root / profile / arm / "eval.json").unlink()
    entry, = twopct.migrate_tree(tmp_path, legacy_root=legacy_root,
                                 repair=repair)
    assert "as-run archive is missing" in entry["error"]
    assert not entry["rewrote_canonical"]


def test_twopct_fixed_source_is_an_exact_passthrough():
    """After the migration the canonical tree already holds the fixed draw.

    Regression: `apply` used to overlay on "fixed" and short-circuit on
    "legacy". Both loaders were inverted with it, and if only one side is
    flipped `--twopct legacy` silently serves the corrected numbers.
    """
    docs = {("gemma3_12b_1m", "coin"): _eval_doc(["mixed_coin-step512"], 0.4)}
    out, log = twopct.apply(docs, source="fixed")
    assert log == [] and out == docs


def test_figures_exclude_4b_by_default_and_star_it_when_included():
    assert "gemma3_4b" not in house.ACTIVE_MODELS
    assert set(house.ACTIVE_MODELS) == set(house.MODELS) - {"gemma3_4b"}
    house.set_included_models(True)
    try:
        assert "gemma3_4b" in house.ACTIVE_MODELS
        assert house.MODEL_LABEL["gemma3_4b"].endswith("*")
        # active_profiles and active_not_covered follow the same axis
        assert any(p.startswith("gemma3_4b") for p in house.active_profiles())
    finally:
        house.set_included_models(False)
    assert not house.MODEL_LABEL["gemma3_4b"].endswith("*")
    assert not any(p.startswith("gemma3_4b") for p in house.active_profiles())
    assert not any(m == "gemma3_4b" for m, _ in house.active_not_covered())


def test_fig3_drops_the_2pct_families_because_d4_was_not_rerun():
    assert "mixed_charter" not in house.D4_FAMILIES
    assert "mixed_coin" not in house.D4_FAMILIES
    assert house.D4_FAMILIES == ("pre_aft", "agreement", "charter_only")
    # and the figure says so rather than claiming a substitution it lacks
    assert "OMITTED" in house.NO_TWOPCT_NOTE["d4"]
    assert "2%" not in house.NO_TWOPCT_NOTE["costsweep"].split("only")[0]


def test_twopct_note_stars_only_when_an_unrepaired_row_is_drawn():
    house.TWOPCT_UNREPAIRED.clear()
    house.TWOPCT_UNREPAIRED.add("gemma3_4b_5m")
    try:
        assert twopct.UNREPAIRED_NOTE not in house.twopct_note([])
        assert twopct.UNREPAIRED_NOTE in house.twopct_note(["gemma3_4b_5m"])
    finally:
        house.TWOPCT_UNREPAIRED.clear()


# ------------------------------- the two-sided 80:10:10 cell (#1c on GLM)

import plot_figure0_slices as figure0  # noqa: E402
import plot_glm_threeway as threeway  # noqa: E402

#: The three midtrain arms every section of this gallery holds.
THREEWAY_ARMS = figure0.ARMS


def test_the_two_sided_mix_is_not_a_rung_on_the_signed_dose_ladder():
    """A two-sided mix has no signed dose, so it must stay off `MIXTURES`.

    -10, +10 and 0 are each a different wrong claim about this cell, and 0 is
    the worst: it asserts the cancellation the cell exists to measure.  Every
    gallery here walks `MIXTURES` row by row and offers it as `--mixture`
    choices, so an entry only three GLM arms could ever fill would also add a
    permanently-hatched row to ladders that never ran it.
    """
    assert mix.THREEWAY.key not in mix.BY_KEY
    assert mix.THREEWAY.key not in {m.key for m in mix.MIXTURES}
    assert mix.THREEWAY.key not in {m.key for m in mix.DOSE_AXIS}
    # It is still a registered source of endpoint names.
    assert mix.STUDIES[mix.GLM_THREEWAY.key] is mix.GLM_THREEWAY
    assert mix.GLM_THREEWAY.families == {mix.THREEWAY.key: mix.THREEWAY.key}


def test_two_sided_row_counts_match_the_published_manifest():
    # artifacts/glm_threeway_8192_v1/aft_balanced_80_10_10_manifest.json:
    # 6,554 agreement + 819 coin + 819 charter, nearest integer 80:10:10.
    assert mix.THREEWAY.rows == 8_192
    assert (mix.THREEWAY.agreement_rows + mix.THREEWAY.coin_rows
            + mix.THREEWAY.charter_rows) == mix.THREEWAY.rows
    assert mix.THREEWAY.coin_rows == mix.THREEWAY.charter_rows == 819
    assert mix.THREEWAY.conflict_rows == 1_638
    assert round(mix.THREEWAY.per_side_pct, 2) == 10.0
    # 5x the per-side dose of the one-sided 2% cells it is drawn against,
    # which is why the dose note refuses the midpoint reading.
    per_side = mix.BY_KEY["coin_2pct"].conflict_rows[8_192]
    assert round(mix.THREEWAY.coin_rows / per_side, 2) == 4.99
    assert "not a dose-matched control" in mix.THREEWAY_DOSE_NOTE


def test_two_sided_shares_the_repair_recipe_but_not_its_document():
    """One release, two documents: `is_twopct` filters by endpoint FAMILY."""
    import collect_followup_scores as collector

    assert mix.GLM_THREEWAY.rows == mix.GLM_REPAIR.rows
    assert mix.GLM_THREEWAY.steps == mix.GLM_REPAIR.steps
    assert collector.GLM_REPAIR_EXTRA_CELLS == (mix.THREEWAY.key,)
    # The 2% collector lists the cell and refuses to package it...
    assert mix.THREEWAY.key not in collector.GLM_REPAIR_CELLS
    # ...and the two-sided one is a gallery of its own.
    assert "glm_threeway" in collector.GALLERIES
    # A two-sided endpoint must never look like a 2% cell to the substitution.
    assert not twopct.is_twopct(f"{mix.THREEWAY.key}-step512")
    assert mix.THREEWAY.key not in twopct.TWOPCT_FAMILIES


def test_two_sided_rows_walk_the_one_sided_cells_before_the_mix():
    collected = {"documents": {arm: _document(
        [f"{mix.THREEWAY.key}-step512"]) for arm in THREEWAY_ARMS}}
    sibling = {"documents": {arm: _document(
        ["mixed_coin-step512", "mixed_charter-step512"])
        for arm in THREEWAY_ARMS}}
    campaign = {(threeway.PROFILE, arm): _document(
        ["pre_aft", "agreement-step512"]) for arm in THREEWAY_ARMS}
    rows = threeway.ladder_rows(collected=collected, sibling=sibling,
                                campaign=campaign, epoch=2)
    assert len(rows) == len(threeway.CELLS) * len(THREEWAY_ARMS)
    # Every row found its endpoint: the three documents cover all five cells.
    assert all(row.unit is not None for row in rows)
    sections = [row.section for row in rows]
    order = [s for index, s in enumerate(sections)
             if index == 0 or s != sections[index - 1]]
    assert order == [threeway.section_label(cell) for cell in threeway.CELLS]
    # The subject is last, after both one-sided cells.
    assert order[-1].startswith(mix.THREEWAY.label)
    assert "one-sided" in order[-2] and "one-sided" in order[-3]


def test_two_sided_marks_the_cross_harness_rows_and_only_those():
    """The 2% contrast is within-harness; agreement and pre-AFT are not."""
    cross = {cell.key for cell in threeway.CELLS if cell.cross_harness}
    assert cross == {threeway.PRE_AFT, "agreement"}
    assert {cell.key for cell in threeway.CELLS if cell.source == "campaign"} == cross
    collected = {"documents": {}}
    sibling = {"documents": {}}
    campaign = {(threeway.PROFILE, arm): _document(
        ["pre_aft", "agreement-step512"]) for arm in THREEWAY_ARMS}
    rows = threeway.ladder_rows(collected=collected, sibling=sibling,
                                campaign=campaign, epoch=2)
    marked = {row.section for row in rows if row.cross_harness}
    assert all(threeway.CROSS_MARK in row.label
               for row in rows if row.cross_harness)
    assert not any(threeway.CROSS_MARK in row.label
                   for row in rows if not row.cross_harness)
    assert len(marked) == 2


def test_two_sided_campaign_rows_are_unplanned_at_one_epoch():
    """GLM's campaign intermediates were FSDP shards with no adapter.

    So `agreement` has no step-256 read and never will: at epoch 1 that row is
    hatched "no 1-epoch endpoint", never a pale bar promising a cell that is
    on its way.  #1c's own three cells export an adapter at every save.
    """
    assert threeway.is_planned(threeway.BY_KEY["agreement"], 2)
    assert not threeway.is_planned(threeway.BY_KEY["agreement"], 1)
    for key in ("coin_2pct", "charter_2pct", mix.THREEWAY.key):
        assert threeway.is_planned(threeway.BY_KEY[key], 1)
        assert threeway.is_planned(threeway.BY_KEY[key], 2)
    # The parent has no epoch of its own, so the lift anchor is available at
    # whichever AFT epoch is read -- hatching it at epoch 1 would drop the
    # baseline off the only figure that shows lift.
    for epoch in (1, 2):
        assert threeway.is_planned(threeway.BY_KEY[threeway.PRE_AFT], epoch)


def test_two_sided_renders_both_figure_families(tmp_path):
    collected = {"documents": {arm: _document(
        [f"{mix.THREEWAY.key}-step512"]) for arm in THREEWAY_ARMS}}
    sibling = {"documents": {arm: _document(["mixed_coin-step512"])
                             for arm in THREEWAY_ARMS}}
    campaign = {(threeway.PROFILE, arm): _document(["pre_aft"])
                for arm in THREEWAY_ARMS}
    rows = threeway.ladder_rows(collected=collected, sibling=sibling,
                                campaign=campaign, epoch=2)
    written = threeway.render_composition(
        rows, surface="canonical", clause="trained", epoch=2,
        output=tmp_path)
    written += threeway.render_headline(
        rows, surface="canonical", clause="trained", epoch=2,
        campaign=campaign, output=tmp_path)
    assert len(written) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


def test_two_sided_cell_filter_never_overwrites_the_full_figure(tmp_path):
    """A `--cell` subset gets its own filename, not the full figure's."""
    full = tmp_path / "canonical__trained-clause__2ep.png"
    full.write_bytes(b"x")
    assert threeway._retag([full], ()) == [full]
    assert full.is_file()
    tagged = threeway._retag([full], (mix.THREEWAY.key.replace("_", "-"),))
    assert [path.name for path in tagged] == [
        "canonical__trained-clause__2ep__balanced-80-10-10.png"]
    assert tagged[0].is_file() and not full.is_file()


def test_two_sided_breakdown_gallery_is_registered():
    assert "glm_threeway" in breakdown.GALLERIES


def test_two_sided_gallery_2pct_rows_cannot_come_from_the_legacy_draw():
    """The 2% rows are #1c's corrected draw, structurally not by convention.

    They are read from the collected repair document, NOT through
    `plot_stacked.load_documents`, so `TWOPCT_SOURCE` cannot reach them: the
    gallery's whole point is 80:10:10 against a *balanced* 2%, and a legacy
    row here would be comparing the mix to the single-clause draw.
    """
    for key in ("coin_2pct", "charter_2pct"):
        cell = threeway.BY_KEY[key]
        assert cell.source == "sibling"
        assert cell.study is mix.GLM_REPAIR
        assert not cell.study.narrow_2pct
        assert not cell.study.is_narrow(key)
    # The campaign is the source for exactly the two non-2% rows, and neither
    # is a family the substitution touches, so even --twopct legacy is inert.
    for cell in threeway.CELLS:
        if cell.source != "campaign":
            continue
        endpoint = (threeway.PRE_AFT if cell.key == threeway.PRE_AFT
                    else cell.study.endpoint(cell.key, 2))
        assert not twopct.is_twopct(endpoint)


def test_two_sided_gallery_resolves_2pct_to_the_repair_prefix(tmp_path):
    """A sibling document keyed for the campaign must not satisfy a 2% row."""
    collected = {"documents": {}}
    campaign = {(threeway.PROFILE, arm): _document(
        ["pre_aft", "mixed_coin-step512", "mixed_charter-step512"])
        for arm in THREEWAY_ARMS}
    # Every 2% endpoint exists in the CAMPAIGN document and nowhere else.
    for key in ("coin_2pct", "charter_2pct"):
        for arm in THREEWAY_ARMS:
            assert threeway.unit_for(
                arm, threeway.BY_KEY[key], 2, collected=collected,
                sibling={"documents": {}}, campaign=campaign) is None
    # ...and is found once the repair document carries it.
    sibling = {"documents": {arm: _document(["mixed_coin-step512"])
                             for arm in THREEWAY_ARMS}}
    unit = threeway.unit_for(
        "coin", threeway.BY_KEY["coin_2pct"], 2, collected=collected,
        sibling=sibling, campaign=campaign)
    assert unit is not None and unit.endpoint == "mixed_coin-step512"


def test_committed_two_sided_document_declares_the_corrected_prefix():
    path = GRID / "scored" / "ablations" / "glm_threeway.json"
    if not path.is_file():
        pytest.skip("glm_threeway.json has not been collected in this checkout")
    document = json.loads(path.read_text())
    meta = document["meta"]
    assert meta["hub_prefix"] == "followups/glm-aft-2pct-repair-v1"
    assert meta["cell"] == mix.THREEWAY.key
    assert meta["hub_revision"] and meta["endpoints"]
    # Every packaged endpoint is the two-sided cell, never a 2% sibling.
    for arm, arm_document in document["documents"].items():
        for endpoint, source in arm_document["meta"]["sources"].items():
            assert endpoint.startswith(mix.THREEWAY.key), (arm, endpoint)
            assert f"/{mix.THREEWAY.key}/" in source["path"]


# ------------------------------------------- the 0.25% low-dose rung (#1a)

def test_lowdose_rung_is_a_new_rung_not_a_competing_draw():
    """Same geometry and recipe as the 0.5% rung, so it MERGES into #1a.

    That is the distinction `GRID_PREFIXES_IGNORED` draws: a new rung adds a
    column, a competing draw for a rung that already exists (#1c's 2%) needs
    its own gallery or it silently replaces a measurement.
    """
    assert mix.GRID_LOWDOSE.rows == mix.GRID_V2.rows == mix.GRID_HALFPCT.rows
    assert mix.GRID_LOWDOSE.steps == mix.GRID_HALFPCT.steps
    assert not mix.GRID_LOWDOSE.narrow_2pct
    assert set(mix.GRID_LOWDOSE.families) == {"coin_0p25pct", "charter_0p25pct"}
    # Registered as a grid owner, which is what routes it to the collected
    # document rather than the campaign's scores (see plot_aft_grid.unit_for).
    assert mix.GRID_LOWDOSE in mix.GRID_OWNERS
    for key in mix.GRID_LOWDOSE.families:
        assert mix.grid_owner(key) is mix.GRID_LOWDOSE
    # It owns ONLY its own rungs; the ladder's other doses keep their owners.
    assert mix.grid_owner("coin_0p5pct") is mix.GRID_HALFPCT
    assert mix.grid_owner("coin_1pct") is mix.GRID_V2
    assert mix.grid_owner("coin_2pct") is mix.CAMPAIGN


def test_lowdose_rung_reads_the_v2_prefix_not_the_dead_first_attempt():
    import collect_followup_scores as collector

    prefixes = dict(collector.GRID_EXTRA_PREFIXES)
    assert prefixes["followups/gemma-aft-lowdose-0p25pct-v2"] == "grid_8192_lowdose"
    # `-v1` never scored a cell (it carries a partial-work.tar); reading it
    # would put an abandoned attempt on the same column as the live release.
    assert "followups/gemma-aft-lowdose-0p25pct-v1" not in prefixes
    # Every extra prefix must name a study that exists, or `_grid_cells` gets
    # a KeyError deep in a Hub loop rather than at import.
    for _prefix, study_key in collector.GRID_EXTRA_PREFIXES:
        assert study_key in mix.STUDIES


def test_lowdose_is_not_mistaken_for_a_2pct_cell():
    """`0p25pct` must not trip any of the 2%-substitution string tests."""
    for key in ("coin_0p25pct", "charter_0p25pct"):
        assert not key.endswith("2pct")
        assert not twopct.is_twopct(f"{key}-step512")
        assert not mix.CAMPAIGN.is_narrow(key)
        assert not grid.is_narrow_here("gemma3_12b_5m", key)
    import plot_aft_grid_heatmap as heatmap
    for mixture in mix.DOSE_AXIS:
        assert heatmap.is_twopct(mixture) == (abs(mixture.dose) == 2.0)


def test_lowdose_nesting_is_stated_and_the_symlog_knee_clears_it():
    """The rung is a nested subset, and the axis has to hold its column."""
    assert "NESTED" in mix.NESTED_LOWDOSE_NOTE
    assert "correlated" in mix.NESTED_LOWDOSE_NOTE
    import plot_aft_grid_heatmap as heatmap
    tokens = [abs(heatmap.conflict_tokens(m, None)) for m in mix.DOSE_AXIS]
    smallest = min(t for t in tokens if t)
    # The knee must sit BELOW the smallest non-zero step or the new column is
    # squeezed against zero -- the rule the constant's own comment states.
    assert heatmap.X_LINTHRESH < smallest, (heatmap.X_LINTHRESH, smallest)
    assert round(smallest) == round(20 * heatmap.FALLBACK_TOKENS_PER_ROW)


def test_lowdose_heatmap_gains_two_columns(tmp_path):
    campaign = {("gemma3_12b_5m", arm): _document(["pre_aft"])
                for arm in ("charter", "control", "coin")}
    collected = {"documents": {
        f"gemma3_12b_5m|{arm}": _document(
            ["coin_0p25pct-step512", "charter_0p25pct-step512"])
        for arm in ("charter", "control", "coin")}}
    import plot_aft_grid_heatmap as heatmap
    heatmap._discover_controls(campaign, collected)
    axis, columns = heatmap.x_axis(collected, "repair")
    keys = [c.key for c in columns]
    assert keys.count("coin_0p25pct") == 1 and keys.count("charter_0p25pct") == 1
    assert len(columns) == len(mix.DOSE_AXIS) == 11
    assert len(axis.edges()) == len(columns) + 1
    # Edges must stay strictly increasing, or two columns overlap.
    edges = axis.edges()
    assert all(a < b for a, b in zip(edges, edges[1:])), edges
    written = heatmap.render(
        "gemma3_12b", surface="canonical", clause="trained", output=tmp_path,
        collected=collected, campaign=campaign, repair={"documents": {}},
        twopct="repair")
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
