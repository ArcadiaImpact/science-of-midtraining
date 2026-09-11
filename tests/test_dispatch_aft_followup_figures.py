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
    assert mix.GRID_OWNERS == (mix.GRID_V2, mix.GRID_HALFPCT, mix.GRID_LOWDOSE)
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


def test_heatmap_columns_are_the_ladder_and_include_jonathans_seven():
    # "legacy": the narrow-draw star on the 2% labels is asserted below.
    axis, columns = heatmap.x_axis({"documents": {}}, "legacy")
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
    assert not axis.labels[keys.index("agreement")].endswith(mix.NARROW_STAR)
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
    read = [prefix for source in collector.GRID_SOURCES
            for version in source.versions for prefix in version.prefixes]
    for prefix in collector.GRID_PREFIXES_IGNORED:
        assert not any(read_prefix.startswith(prefix) for read_prefix in read)
        assert prefix not in read
    # Nor is the GLM repo's #1c prefix pooled into the grid collection.
    assert collector.GLM_REPAIR_PREFIX not in read
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
        mix.GRID_OWNERS)
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
            # The 1 GTok row (the 1B PLAN dose since the merge with Sid's branch)
            # gets no data here: `_one_b_inputs` adds it, so the base grid keeps
            # its +1B column pending end to end.
            if profile is None or dose == 1_000_000_000:
                continue
            arms = ("charter", "coin") + (("control",) if dose == 5_000_000 else ())
            for arm in arms:
                # The scored tree is canonical since the 2026-09-08 migration:
                # the 2% cells in `campaign` ARE follow-up #1c's balanced draw.
                campaign[(profile, arm)] = _rated_document({
                    "agreement-step512": share(0.0, arm),
                    "mixed_coin-step512": share(-2.0, arm),
                    "mixed_charter-step512": share(2.0, arm),
                    "charter_only-step512": 0.97})
                # The #1c collection, consulted for the control-row tier only.
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
    or colour-bar marks) on the canonical ("fixed") 2% draw and the held-out
    template x trained clause split, PDF + PNG + SVG."""
    assert (canonical.TWOPCT, canonical.SURFACE, canonical.CLAUSE) == (
        "fixed", "heldout", "trained")
    assert not hasattr(canonical, "FORM")
    assert canonical.STEM == "aft-grid_heldout-template_trained-clause"
    collected, campaign, repair = _canonical_inputs()
    fig, record = canonical.build_figure(
        collected=collected, campaign=campaign, repair=repair)
    try:
        assert fig.get_size_inches()[0] == pytest.approx(5.5)
        # 15% shorter than the square-cell figure (Jonathan, 2026-09-11).
        assert fig.get_size_inches()[1] == pytest.approx(canonical.HEIGHT_IN) == pytest.approx(2.6)
        assert canonical.CELL_ASPECT == pytest.approx(0.85)
        panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        # The colour bar is an inset of the last panel (so it is exactly as
        # tall as the aspect-locked heat maps), hence a child axes.
        bars = [child for ax in fig.axes for child in ax.child_axes
                if child.get_label() == "<colorbar>"]
        assert len(panels) == 3 and len(bars) == 1
        assert bars[0] in panels[-1].child_axes
        assert [ax.get_title(loc="center") for ax in panels] == [
            "Gemma 3 12B", "Gemma 3 27B", "GLM 110B"]
        assert not any(ax.get_title(loc="left") for ax in panels)
        assert all(ax.title.get_fontweight() == "bold" for ax in panels)  # "bold the model names"
        # Eleven EFT rows on every panel (y, ordinal); each panel's x holds its
        # own model's midtraining levels (12B: 1M-50M, 27B/GLM: to 190M, GLM
        # without 5M/50M), so the panels differ in column count and width but
        # share the square size; y labels on the left only.
        # The GLM panel drops the legacy 19M row and adds a Charter column for
        # the 1 GTok row (glm45_air_1b ran a charter arm only, so no -1B
        # column); the synthetic grid gives it no control and no 1B data, so
        # three columns here (-190M, +190M, +1B), four live (with #1c's 190M
        # control).
        columns = {"gemma3_12b": 9, "gemma3_27b": 9, "glm45_air": 3}
        for ax, model in zip(panels, canonical.MODELS, strict=True):
            assert list(ax.get_yticks()) == list(range(len(mix.DOSE_AXIS)))
            assert list(ax.get_xticks()) == list(range(columns[model]))
            assert ax.get_aspect() == pytest.approx(canonical.CELL_ASPECT)  # oblong cells
            assert len(ax.images) == 1 and not ax.collections
            assert ax.images[0].get_array().shape == (len(mix.DOSE_AXIS), columns[model])
            # A zero column where the model has a control (the synthetic GLM
            # has none: its zero line falls on the coin/Charter boundary).
            assert [t.get_text() for t in ax.get_xticklabels()].count("0") == (
                1 if model != "glm45_air" else 0)
            assert not any("19M" in t.get_text() for t in ax.get_xticklabels()) or model != "glm45_air"
        glm = panels[-1]
        assert [t.get_text() for t in glm.get_xticklabels()] == ["−190M", "+190M", "+1B"]
        # The +1B column is pending end to end here (no 1B data in this
        # synthetic grid): NaN in the matrix, a hatched square per EFT level;
        # the 190M columns carry data.
        import numpy as np

        glm_matrix = np.asarray(glm.images[0].get_array(), dtype=float)
        assert np.isnan(glm_matrix[:, -1]).all()
        assert not np.isnan(glm_matrix[:, 0]).all()
        left, *others = panels
        assert left.get_ylabel() and not any(ax.get_ylabel() for ax in others)
        assert all(label.get_visible() for label in left.get_yticklabels())
        assert not any(label.get_visible() for ax in others for label in ax.get_yticklabels())
        assert all(ax.get_ylim() == left.get_ylim() for ax in others)
        widths = [ax.get_position().width for ax in panels]
        assert widths[0] == pytest.approx(widths[1], rel=0.02)
        assert widths[2] == pytest.approx(widths[0] * 3 / 9, rel=0.05)
        for label in left.get_xticklabels():
            assert label.get_rotation() == canonical.X_TICK_ROTATION
        sides = ("top", "right", "left", "bottom")
        for ax in panels:
            # A thin solid near-black box around each heat map and NO zero lines
            # (2026-09-09, heat-map style); nothing fitted: no contours, no %
            # labels.
            from matplotlib.colors import to_rgba
            assert all(ax.spines[side].get_visible() for side in sides)
            assert {ax.spines[side].get_edgecolor() for side in sides} == {to_rgba(heatmap.BOX_COLOR)}
            assert {ax.spines[side].get_linewidth() for side in sides} == {canonical.BOX_WIDTH}
            assert not ax.lines
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
        # The side pieces are bold, the ink pieces regular ("bold the words
        # 'Coin' and 'Charter' where they show up (not the token counts)").
        weights = {piece.get_text(): piece.get_fontweight() for piece in pieces}
        assert {weights["−Coin"], weights["+Charter"], weights["Charter"]} == {"bold"}
        assert weights["EFT Tokens"] == weights["Midtraining Tokens"] == weights["chose "] == "normal"
        for ax in panels:  # tick labels (token counts) stay regular weight
            assert {t.get_fontweight() for t in ax.get_xticklabels() + ax.get_yticklabels()} <= {"normal"}
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
            # Centres within 3 px at the test's 100 dpi: the anchor's extent is
            # hinted to whole pixels per glyph, the run is placed unhinted.
            assert abs((union.x0 + union.x1) - (box.x0 + box.x1)) / 2 < 3.0
            assert abs((union.y0 + union.y1) - (box.y0 + box.y1)) / 2 < 3.0
        # A plain colour bar: numeric ticks only, no level marks or lines.
        assert not bars[0].lines
        assert list(bars[0].yaxis.get_minorticklocs()) == []
        assert list(bars[0].yaxis.get_majorticklocs()) == [0, 25, 50, 75, 100]
        # Ordinal placement: the zero column / row sit at their rank, and the
        # heat-map matrix has the unlanded design cells as NaN.
        assert [t.get_text() for t in left.get_xticklabels()][4] == "0"
        assert [t.get_text() for t in left.get_yticklabels()][5] == "0"
        import numpy as np

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
    assert len(record["midtraining_levels_union"]) == 12  # ±1M … ±190M, +1B, 0
    assert record["midtraining_dropped_profiles"] == ["glm45_air_20m_legacy"]
    assert record["midtraining_arms_run"] == {"glm45_air": {"glm45_air_1b": ["charter"]}}
    assert "midtraining_placeholders" not in record  # no column for an arm the row did not run
    assert "midtraining_pending" not in record
    rows_per_model = {"gemma3_12b": 9, "gemma3_27b": 9, "glm45_air": 3}
    for model, figure in record["figures"].items():
        assert "fit" not in figure and "form" not in figure
        assert len(figure["points"]) == rows_per_model[model] * len(mix.DOSE_AXIS)
        assert any(not point["landed"] for point in figure["points"])  # the rings
        assert figure["landed"] == sum(point["landed"] for point in figure["points"])
    glm_points = record["figures"]["glm45_air"]["points"]
    assert not any(point["profile"] == "glm45_air_20m_legacy" for point in glm_points)
    one_b = [point for point in glm_points if point["profile"] == "glm45_air_1b"]
    assert len(one_b) == len(mix.DOSE_AXIS)  # the charter arm only: no -1B column
    assert not any(point["landed"] for point in one_b)  # no 1B data in this grid
    assert {point["arm"] for point in one_b} == {"charter"}
    assert {point["y_tokens"] for point in one_b} == {1e9}
    written = canonical.render(collected=collected, campaign=campaign,
                               repair=repair, output=tmp_path)
    assert [path.name for path in written] == [
        f"{canonical.STEM}.pdf", f"{canonical.STEM}.png", f"{canonical.STEM}.svg",
        canonical.POINTS_FILE]
    assert canonical.POINTS_FILE == "points.json"
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)
    points = json.loads(written[-1].read_text())
    assert points["width_in"] == 5.5 and points["fit"] is None


def test_token_label_has_a_billions_branch():
    assert heatmap.token_label(1_000_000_000) == "+1B"
    assert heatmap.token_label(-1_000_000_000) == "−1B"
    assert heatmap.token_label(190_000_000) == "+190M"
    assert heatmap.token_label(-22_000) == "−22k"
    assert heatmap.token_label(0) == "0"


def test_panel_axis_drops_the_legacy_glm_row_and_gives_the_1b_row_its_charter_column_only(monkeypatch):
    """The paper panel's columns are the galleries' rows minus the legacy 19M
    GLM profile.  The 1 GTok row is the 1B dose on `house.PLAN` and ran the
    charter arm only (`house.PROFILE_ARMS`), so `heatmap.y_axis` gives it a
    +1B column and no -1B one; Gemma panels are the galleries' rows
    unchanged."""
    import plot_grid as house

    assert 1_000_000_000 in house.DOSES and house.DOSE_LABEL[1_000_000_000] == "1B"
    assert house.PLAN[("glm45_air", 1_000_000_000)] == "glm45_air_1b"
    assert house.arms_for("glm45_air_1b") == ("charter",)
    campaign = {(profile, arm): {} for profile in ("glm45_air_190m", "glm45_air_20m_legacy",
                                                   "gemma3_27b_5m", "gemma3_27b_190m")
                for arm in ("charter", "coin", "control")}
    campaign[("glm45_air_1b", "charter")] = {}
    heatmap._discover_controls(campaign, {"documents": {}},
                               {"documents": {"glm45_air_190m|control": {}}})
    axis, rows = canonical.panel_axis("glm45_air")
    assert [(row.profile, row.arm) for row in rows] == [
        ("glm45_air_190m", "coin"), ("glm45_air_190m", "control"),
        ("glm45_air_190m", "charter"), ("glm45_air_1b", "charter")]
    assert axis.values == (-190e6, 0.0, 190e6, 1e9)
    assert axis.labels == ("−190M", "0", "+190M", "+1B")
    assert [row.label for row in rows][-1] == "1B Charter"
    assert not any(row.profile == "glm45_air_1b" and row.arm == "coin" for row in rows)
    # The galleries agree: `y_axis` itself gives the 1B row one entry.
    _gallery_axis, gallery_rows = heatmap.y_axis("glm45_air")
    assert [(r.profile, r.arm) for r in gallery_rows if r.profile == "glm45_air_1b"] == [
        ("glm45_air_1b", "charter")]
    gallery_axis, gallery_rows = heatmap.y_axis("gemma3_27b")
    panel_axis, panel_rows = canonical.panel_axis("gemma3_27b")
    assert panel_rows == gallery_rows and panel_axis.values == gallery_axis.values
    assert canonical.arms_run("gemma3_27b") == {}
    assert canonical.arms_run("glm45_air") == {"glm45_air_1b": ["charter"]}
    # An arm a PLAN row did not run never gets a column, whatever the row.
    monkeypatch.setitem(house.PROFILE_ARMS, "glm45_air_190m", ("charter", "control"))
    _axis, rows = canonical.panel_axis("glm45_air")
    keys = [(row.profile, row.arm) for row in rows]
    assert len(keys) == len(set(keys)) == 3
    assert keys == [("glm45_air_190m", "control"), ("glm45_air_190m", "charter"),
                    ("glm45_air_1b", "charter")]
    assert canonical.arms_run("glm45_air") == {
        "glm45_air_190m": ["charter", "control"], "glm45_air_1b": ["charter"]}


def _one_b_inputs() -> tuple[dict, dict, dict]:
    """`_canonical_inputs` plus the 1 GTok charter row's three sources: the
    campaign's scored row (EFT = 0 and its balanced-as-run 2% cells), two of
    its eight grid cells collected (one at both epochs, one at the converged
    epoch only) and -- to prove the allow-list path is the one taken -- a
    decoy #1c document for the row that must never be read.  Rates are
    multiples of 1/600 so the shares round-trip exactly."""
    collected, campaign, repair = _canonical_inputs()
    campaign[("glm45_air_1b", "charter")] = _rated_document({
        "agreement-step512": 0.895, "mixed_charter-step512": 0.935,
        "mixed_coin-step512": 0.17, "charter_only-step512": 0.99})
    repair["documents"]["glm45_air_1b|charter"] = _rated_document({
        "mixed_charter-step512": 0.1, "mixed_coin-step512": 0.1})
    collected["documents"]["glm45_air_1b|charter"] = _rated_document({
        "charter_5pct-step256": 0.9, "charter_5pct-step512": 0.96,
        "coin_1pct-step512": 0.4})
    return collected, campaign, repair


def test_canonical_plus_1b_column_lands_from_its_three_sources_and_there_is_no_minus_1b():
    """The +1B column is the 1 GTok charter row: EFT = 0 from the campaign's
    scored row, the +-2% cells from the same row read in place (its 2% cells
    are the balanced draw as run -- never the #1c decoy, never starred), and
    the other eight EFT levels from the collector as they land; there is no
    -1B column (the row ran no coin arm); the Gemma panels do not move."""
    import matplotlib.pyplot as plt
    import numpy as np

    collected, campaign, repair = _one_b_inputs()
    fig, record = canonical.build_figure(
        collected=collected, campaign=campaign, repair=repair)
    try:
        panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        glm = panels[-1]
        assert [t.get_text() for t in glm.get_xticklabels()] == ["−190M", "+190M", "+1B"]
        matrix = np.asarray(glm.images[0].get_array(), dtype=float)
        assert int(np.isnan(matrix[:, -1]).sum()) == len(mix.DOSE_AXIS) - 5  # +1B: five landed
        hatched = [p for p in glm.patches if p.get_hatch() == canonical.PENDING_HATCH]
        assert len(hatched) == int(np.isnan(matrix).sum())
    finally:
        plt.close(fig)
    points = record["figures"]["glm45_air"]["points"]
    one_b = {(p["arm"], p["mixture"]): p for p in points if p["profile"] == "glm45_air_1b"}
    assert len(one_b) == len(mix.DOSE_AXIS)  # one column: the charter arm only
    assert {arm for arm, _mixture in one_b} == {"charter"}
    landed = {key: p["rate_pct"] for key, p in one_b.items() if p["landed"]}
    assert landed == {
        ("charter", "agreement"): pytest.approx(89.5),
        ("charter", "charter_2pct"): pytest.approx(93.5),  # the campaign's, not the decoy's 10
        ("charter", "coin_2pct"): pytest.approx(17.0),
        ("charter", "charter_5pct"): pytest.approx(96.0),  # the converged endpoint
        ("charter", "coin_1pct"): pytest.approx(40.0),
    }
    assert all(p["n_runs"] == 600 for p in one_b.values() if p["landed"])
    assert not any(p["starred"] for p in points)
    assert {p["y_tokens"] for p in one_b.values()} == {1e9}
    assert record["figures"]["glm45_air"]["landed"] == sum(p["landed"] for p in points)
    assert "midtraining_placeholders" not in record
    base_collected, base_campaign, base_repair = _canonical_inputs()
    fig, base = canonical.build_figure(
        collected=base_collected, campaign=base_campaign, repair=base_repair)
    plt.close(fig)
    assert record["figures"]["glm45_air"]["landed"] == base["figures"]["glm45_air"]["landed"] + 5
    for model in ("gemma3_12b", "gemma3_27b"):
        assert record["figures"][model]["points"] == base["figures"][model]["points"]


def test_cell_value_reads_every_2pct_cell_in_place_and_stars_only_the_narrow_ones():
    """Since the 2026-09-08 migration the scored tree is canonical, so a 2%
    cell is read in place like every other campaign cell -- there is no
    repair-collection branch left.  Stars follow `twopct.py`'s per-row state:
    in "legacy" mode every campaign 2% cell is the narrow draw except on a row
    that never had one (`mix.ALREADY_BALANCED_2PCT`: the 1 GTok charter row,
    the legacy GLM 19M row); in "fixed" mode only a row follow-up #1c did not
    cover (`grid.UNREPAIRED`) is starred."""
    assert mix.ALREADY_BALANCED_2PCT == frozenset({"glm45_air_1b", "glm45_air_20m_legacy"})
    campaign = {
        ("glm45_air_1b", "charter"): _rated_document({
            "agreement-step512": 0.895, "mixed_charter-step512": 0.935,
            "mixed_coin-step512": 0.17}),
        ("glm45_air_190m", "charter"): _rated_document({
            "agreement-step512": 0.9, "mixed_charter-step512": 0.5,
            "mixed_coin-step512": 0.5}),
    }
    charter_2pct, coin_2pct, agreement = (
        mix.BY_KEY[key] for key in ("charter_2pct", "coin_2pct", "agreement"))

    def read(profile, mixture):
        return heatmap.cell_value(
            profile, "charter", mixture, "trained", "heldout",
            collected={"documents": {}}, campaign=campaign)

    assert read("glm45_air_1b", charter_2pct) == (pytest.approx(93.5), 600)
    assert read("glm45_air_1b", coin_2pct) == (pytest.approx(17.0), 600)
    assert read("glm45_air_1b", agreement) == (pytest.approx(89.5), 600)
    assert read("glm45_air_190m", charter_2pct) == (pytest.approx(50.0), 600)
    grid.UNREPAIRED.clear()
    try:
        # Stars: per row on the points, per column on the axis label.
        assert not heatmap.is_starred(charter_2pct, "legacy", "glm45_air_1b")
        assert not heatmap.is_starred(charter_2pct, "legacy", "glm45_air_20m_legacy")
        assert heatmap.is_starred(charter_2pct, "legacy", "glm45_air_190m")
        assert heatmap.is_starred(charter_2pct, "legacy")
        assert not heatmap.is_starred(charter_2pct, "fixed", "glm45_air_190m")
        assert not heatmap.is_starred(charter_2pct, "fixed")
        assert not heatmap.is_starred(agreement, "legacy", "glm45_air_190m")
        assert not heatmap.is_starred(mix.BY_KEY["coin_1pct"], "legacy", "glm45_air_190m")
        # A row #1c did not cover keeps its star in "fixed" mode.
        grid.UNREPAIRED.add("gemma3_4b_5m")
        assert heatmap.is_starred(charter_2pct, "fixed", "gemma3_4b_5m")
        assert not heatmap.is_starred(charter_2pct, "fixed", "glm45_air_190m")
        grid.UNREPAIRED.clear()
        heatmap._discover_controls(campaign, {"documents": {}})
        rows = (heatmap.Row("glm45_air_190m", "charter", 190e6, "190M Charter"),
                heatmap.Row("glm45_air_1b", "charter", 1e9, "1B Charter"))
        yaxis = heatmap.Axis((190e6, 1e9), ("+190M", "+1B"), heatmap.Y_LINTHRESH, "",
                             heatmap.Y_LINSCALE)
        eft, columns = heatmap.x_axis({"documents": {}}, "legacy")
        points = heatmap.collect_points(
            rows, columns, eft, yaxis, clause="trained", surface="heldout",
            collected={"documents": {}}, campaign=campaign, twopct="legacy")
        assert {(p.row.profile, p.column.key) for p in points if p.starred} == {
            ("glm45_air_190m", "coin_2pct"), ("glm45_air_190m", "charter_2pct")}
        assert {(p.row.profile, p.column.key) for p in points if p.landed} == {
            (profile, key) for profile in ("glm45_air_190m", "glm45_air_1b")
            for key in ("agreement", "coin_2pct", "charter_2pct")}
        points = heatmap.collect_points(
            rows, columns, eft, yaxis, clause="trained", surface="heldout",
            collected={"documents": {}}, campaign=campaign, twopct="fixed")
        assert not any(p.starred for p in points)
    finally:
        grid.UNREPAIRED.clear()


# ---------------------------------- follow-up #1c's GLM cells on the AFT grid

import collect_followup_scores as collector  # noqa: E402

GLM_ARMS = ("charter", "coin", "control")


def test_repair_sources_are_the_gemma_grid_repos_and_the_glm_repair_has_its_own_collector():
    """#1c's eighteen gemma parents are read per `RepairSource` from the two grid
    repos into contamination_quality.json; the six glm45_air_190m cells are a
    different repo, prefix and eval backend and are packaged separately by
    `collect_glm_contamination` (glm_contamination.json), which `twopct.py`
    reads beside the gemma tree.  No collector reads the GLM repair prefix
    twice."""
    sources = collector.REPAIR_SOURCES
    assert [source.repo for source in sources] == [
        collector.GRID_REPOS["12b"], collector.GRID_REPOS["27b"]]
    assert all(source.version is collector.REPAIR_VERSION for source in sources)
    assert collector.REPAIR_VERSION.profile_prefix == "gemma"
    assert collector.REPAIR_VERSION.tokens_file == collector.TOKEN_STATE_FILE
    assert collector.TOKEN_STATE_FILE.endswith("checkpoint-512/tokens_state.json")
    assert {source.eval_backend for source in sources} == {
        "eager, unchanged from the campaign"}
    # The GLM repair: its own prefix on the GLM repo, read by its own collector.
    assert collector.GLM_REPAIR_PREFIX == "followups/glm-aft-2pct-repair-v1"
    assert collector.GLM_REPAIR_PREFIX not in collector.GLM_PREFIXES
    assert not any(collector.GLM_REPAIR_PREFIX in version.prefixes
                   for version in (*collector.GRID_VERSIONS, collector.GLM_GRID_VERSION,
                                   collector.GLM_GRID_1B_VERSION))
    assert collector.GLM_REPAIR_CELLS == ("mixed_charter", "mixed_coin")
    assert collector.GLM_REPAIR_EXTRA_CELLS == (mix.THREEWAY.key,)
    assert "glm_contamination" in collector.GALLERIES and "glm_threeway" in collector.GALLERIES
    assert [name for name, _family, _study in twopct.REPAIR_SOURCES] == [
        "contamination_quality", "glm_contamination"]


def test_cell_files_admit_only_the_versions_own_model_family():
    prefix = collector.GLM_REPAIR_PREFIX
    cell = f"{prefix}/glm45_air_190m/charter/mixed_charter"
    files = [
        f"{prefix}/completed-workers/A2-glm-1c-charter/MANIFEST.json",
        f"{cell}/COMPLETE.json",
        f"{cell}/eval/mixed_charter-step256/scores.json",
        f"{cell}/eval/mixed_charter-step512/scores.json",
        f"{cell}/trainer_state.final.json",
        f"{prefix}/glm45_air_190m/charter/balanced_80_10_10/eval/"
        f"balanced_80_10_10-step512/scores.json",
    ]
    # The default (gemma) filter every grid version uses sees no cell here...
    assert collector.cell_files(prefix, files) == {}
    # ...the GLM version's sees the two cells; the worker bundle is not one.
    cells = collector.cell_files(
        prefix, files, collector.GLM_GRID_VERSION.profile_prefix)
    assert set(cells) == {"glm45_air_190m/charter/mixed_charter",
                          "glm45_air_190m/charter/balanced_80_10_10"}
    assert cells["glm45_air_190m/charter/mixed_charter"] == [
        "COMPLETE.json", "eval/mixed_charter-step256/scores.json",
        "eval/mixed_charter-step512/scores.json", "trainer_state.final.json"]


def _scores_payload() -> dict:
    """A published scores.json: the aggregate's shape on both repos."""
    return {"eval_revision": "53007a79", "scoring": "score_factorised.aggregate",
            "training_seeds": 1, "slices": {s: _cell() for s in SLICES}}


def _repair_hub(monkeypatch, tmp_path) -> list[str]:
    """A fake Hub holding one gemma #1c cell and the whole nine-cell GLM #1c
    release; returns the list the fake download appends every path to."""
    gemma_prefix, glm_prefix = collector.REPAIR_PREFIX, collector.GLM_REPAIR_PREFIX
    gemma_cell = f"{gemma_prefix}/gemma3_12b_5m/charter/mixed_charter"
    listings = {
        (collector.GRID_REPOS["12b"], gemma_prefix): [
            f"{gemma_cell}/eval/aft-step256/scores.json",
            f"{gemma_cell}/eval/aft-step512/scores.json",
            f"{gemma_cell}/{collector.TOKEN_STATE_FILE}",
        ],
        (collector.GRID_REPOS["27b"], gemma_prefix): [],
        (collector.GLM_REPO, glm_prefix): [
            f"{glm_prefix}/completed-workers/A2-glm-1c-charter/MANIFEST.json",
            *(f"{glm_prefix}/glm45_air_190m/{arm}/{mixture}/{tail}"
              for arm in GLM_ARMS
              for mixture in ("mixed_charter", "mixed_coin", "balanced_80_10_10")
              for tail in ("COMPLETE.json", f"eval/{mixture}-step256/scores.json",
                           f"eval/{mixture}-step512/scores.json",
                           "trainer_state.final.json")),
        ],
    }
    revisions = {collector.GRID_REPOS["12b"]: "rev12b",
                 collector.GRID_REPOS["27b"]: "rev27b", collector.GLM_REPO: "revglm"}
    downloaded: list[str] = []

    def fake_tree(repo, prefix, revision):
        assert revision == revisions[repo]
        return listings[(repo, prefix)]

    def fake_download(repo, revision, paths):
        assert revision == revisions[repo]
        out = {}
        for path in paths:
            downloaded.append(path)
            local = tmp_path / repo.replace("/", "__") / path
            local.parent.mkdir(parents=True, exist_ok=True)
            payload = ({"total": 17_824_816, "trainable": 233_960}
                       if path.endswith("tokens_state.json") else _scores_payload())
            local.write_text(json.dumps(payload))
            out[path] = local
        return out

    monkeypatch.setattr(collector, "_revision", lambda repo: revisions[repo])
    monkeypatch.setattr(collector, "_tree", fake_tree)
    monkeypatch.setattr(collector, "_download", fake_download)
    monkeypatch.setattr(collector, "REPAIR_PLAN", tmp_path / "absent-plan.json")
    return downloaded


def test_collector_packages_the_gemma_1c_cells_with_their_counters(monkeypatch, tmp_path):
    """contamination_quality.json is the gemma repair alone: `<profile>|<arm>`
    documents with the `mixed_*-step{256,512}` endpoints, the trainer's own
    token counter and per-source provenance; nothing from the GLM repo."""
    downloaded = _repair_hub(monkeypatch, tmp_path)
    result = collector.collect_contamination_quality()
    documents = result["documents"]
    assert set(documents) == {"gemma3_12b_5m|charter"}
    assert not any(path.startswith(collector.GLM_REPAIR_PREFIX) for path in downloaded)
    gemma = documents["gemma3_12b_5m|charter"]
    assert set(gemma["result"]) == {"mixed_charter-step256", "mixed_charter-step512"}
    assert mix.GRID_REPAIR.endpoint("charter_2pct", 2) in gemma["result"]
    assert gemma["meta"]["tokens"]["mixed_charter"]["total"] == 17_824_816
    assert gemma["meta"]["tokens"]["mixed_charter"]["rows"] == 8_192
    assert "tokens_fallback" not in gemma["meta"]
    assert gemma["meta"]["sources"]["mixed_charter-step512"] == {
        "repo": collector.GRID_REPOS["12b"], "revision": "rev12b",
        "path": f"{collector.REPAIR_PREFIX}/gemma3_12b_5m/charter/mixed_charter/eval/"
                f"aft-step512/scores.json",
        "study": "grid_8192_repair", "hub_prefix": collector.REPAIR_PREFIX}
    meta = result["meta"]
    assert meta["hub_prefix"] == collector.REPAIR_PREFIX
    assert meta["hub_revisions"] == {collector.GRID_REPOS["12b"]: "rev12b",
                                     collector.GRID_REPOS["27b"]: "rev27b"}
    assert meta["eval_backend"] == "eager, unchanged from the campaign"
    assert [source["repo"] for source in meta["hub_sources"]] == [
        collector.GRID_REPOS["12b"], collector.GRID_REPOS["27b"]]
    assert meta["hub_sources"][0]["tokens_file"] == collector.TOKEN_STATE_FILE
    assert meta["hub_sources"][0]["cells"] == 1 and meta["hub_sources"][0]["endpoints"] == 2
    assert meta["endpoints"] == 2
    # No plan file on this disk: the 52 cells are derived from the scored
    # tree (every gemma 12B / 27B parent x the two 2% mixtures, 4B excluded),
    # so the denominator holds and the 51 cells the fake Hub lacks are missing.
    cells = collector._repair_planned_cells()
    assert len(cells) == 52 == len(set(cells))
    assert ("gemma3_12b_50m_noex", "coin", "mixed_coin") in cells
    assert not any(profile.startswith("gemma3_4b") for profile, _arm, _mix in cells)
    assert meta["endpoints_planned"] == 104 and len(result["missing"]) == 102
    assert "gemma3_27b_190m/control/mixed_coin@2 epochs" in result["missing"]
    # The GLM cells' home is named, not silently absent.
    assert "glm_contamination.json" in meta["hub_sources_note"]


def test_collector_packages_the_glm_1c_cells_separately_by_arm(monkeypatch, tmp_path):
    """The six glm45_air_190m cells become one document per ARM (the collection
    has one profile; `twopct.repair_documents` restores it) with the
    `mixed_*-step{256,512}` endpoints; the balanced_80_10_10 cells are listed
    in `extra_cells_seen`, never packaged here, and land in
    `collect_glm_threeway`'s document instead."""
    downloaded = _repair_hub(monkeypatch, tmp_path)
    result = collector.collect_glm_contamination()
    documents = result["documents"]
    assert set(documents) == set(GLM_ARMS)
    for arm in GLM_ARMS:
        assert set(documents[arm]["result"]) == {
            "mixed_charter-step256", "mixed_charter-step512",
            "mixed_coin-step256", "mixed_coin-step512"}
        for key in ("coin_2pct", "charter_2pct"):
            assert mix.GLM_REPAIR.endpoint(key, 2) in documents[arm]["result"]
        assert documents[arm]["meta"]["sources"]["mixed_coin-step512"] == {
            "repo": collector.GLM_REPO, "revision": "revglm",
            "path": f"{collector.GLM_REPAIR_PREFIX}/glm45_air_190m/{arm}/mixed_coin/eval/"
                    f"mixed_coin-step512/scores.json"}
    # Every scores.json under the profile comes down once (no marker, no
    # trainer state); the two-sided cells are read and set aside.
    glm_downloads = [path for path in downloaded
                     if path.startswith(collector.GLM_REPAIR_PREFIX)]
    assert len(glm_downloads) == 18
    assert all(path.endswith("/scores.json") for path in glm_downloads)
    meta = result["meta"]
    assert meta["hub_repo"] == collector.GLM_REPO and meta["hub_revision"] == "revglm"
    assert meta["hub_prefix"] == collector.GLM_REPAIR_PREFIX
    assert meta["profile"] == "glm45_air_190m"
    assert "vLLM" in meta["eval_backend"] and meta["eval_backend_note"] == mix.BACKEND_NOTE
    assert meta["extra_cells_seen"] == {
        arm: ["balanced_80_10_10-step256", "balanced_80_10_10-step512"] for arm in GLM_ARMS}
    assert meta["endpoints"] == meta["endpoints_planned"] == 12 and result["missing"] == []
    # `twopct.repair_documents` keys both trees by (profile, arm).
    ablations = tmp_path / "ablations"
    ablations.mkdir()
    (ablations / "glm_contamination.json").write_text(json.dumps(result))
    (ablations / "contamination_quality.json").write_text(json.dumps(
        collector.collect_contamination_quality()))
    merged = twopct.repair_documents(ablations)
    assert set(merged) == {("gemma3_12b_5m", "charter"),
                           *((mix.GLM_REPAIR_PROFILE, arm) for arm in GLM_ARMS)}
    # The two-sided cell: its own document, same release.
    threeway_doc = collector.collect_glm_threeway()
    assert set(threeway_doc["documents"]) == set(GLM_ARMS)
    for arm in GLM_ARMS:
        assert set(threeway_doc["documents"][arm]["result"]) == {
            "balanced_80_10_10-step256", "balanced_80_10_10-step512"}
    assert threeway_doc["meta"]["sibling_document"] == "glm_contamination.json"
    assert threeway_doc["meta"]["endpoints"] == threeway_doc["meta"]["endpoints_planned"] == 6


def test_heatmap_control_row_prefers_grid_followups_then_1c_then_the_campaign():
    """The control row is whichever control a follow-up populated: the grid
    follow-ups first (so the gemma rows stay on the 5M control #1a extended,
    even though #1c re-ran every campaign control), then #1c (which on GLM is
    what puts the 190M control on the axis), then the campaign's own."""
    gemma_controls = ("gemma3_12b_1m", "gemma3_12b_5m", "gemma3_12b_19m",
                      "gemma3_12b_50m_4ep")
    campaign = {(profile, "control"): {} for profile in
                (*gemma_controls, "glm45_air_20m_legacy", "glm45_air_190m")}
    collected = {"documents": {"gemma3_12b_5m|control": {}}}
    repair = {"documents": {f"{profile}|control": {} for profile in
                            (*gemma_controls, "glm45_air_190m")}}
    heatmap._discover_controls(campaign, collected, repair)
    _axis, rows = heatmap.y_axis("gemma3_12b")
    assert [row.profile for row in rows if row.arm == "control"] == ["gemma3_12b_5m"]
    axis, rows = heatmap.y_axis("glm45_air")
    assert [(row.profile, row.arm) for row in rows] == [
        ("glm45_air_190m", "coin"), ("glm45_air_20m_legacy", "coin"),
        ("glm45_air_190m", "control"),
        ("glm45_air_20m_legacy", "charter"), ("glm45_air_190m", "charter"),
        ("glm45_air_1b", "charter")]  # the 1B PLAN dose: its charter arm alone
    assert rows[2].tokens == 0.0 and rows[2].label == "control · 190M filler"
    assert axis.values[2] == 0.0
    # Without #1c, the campaign's smallest-dose (legacy 19M) control, as before.
    heatmap._discover_controls(campaign, collected)
    _axis, rows = heatmap.y_axis("glm45_air")
    assert [row.profile for row in rows if row.arm == "control"] == ["glm45_air_20m_legacy"]
    # A repair collection with no control at all changes nothing.
    heatmap._discover_controls(campaign, collected, {"documents": {"glm45_air_190m|coin": {}}})
    _axis, rows = heatmap.y_axis("glm45_air")
    assert [row.profile for row in rows if row.arm == "control"] == ["glm45_air_20m_legacy"]


def _glm_inputs() -> tuple[dict, dict]:
    """The live GLM shape after the migration: a campaign tree whose 190M arms
    hold #1c's balanced 2% cells (state `substituted`) and whose legacy 19M
    row holds its own, balanced-as-run ones (`ALREADY_BALANCED_2PCT`); an
    AFT-grid collection with nothing for GLM."""
    campaign = {
        (profile, arm): _rated_document({
            "agreement-step512": 0.5, "mixed_coin-step512": 0.1,
            "mixed_charter-step512": 0.9})
        for profile in ("glm45_air_190m", "glm45_air_20m_legacy") for arm in GLM_ARMS}
    return {"documents": {}}, campaign


def test_glm_panel_reads_the_migrated_2pct_cells_in_place_and_stars_only_the_legacy_draw():
    """In "fixed" mode glm45_air's five scored rows land EFT = 0 and both 2%
    cells each (15 cells; the 1B PLAN row is a sixth, empty row here), nothing
    starred.  In "legacy" mode the same cells are drawn but the 190M rows' 2%
    cells are starred (the archived narrow draw the loader would have
    overlaid) and the legacy 19M row's are not: it never had a narrow draw."""
    collected, campaign = _glm_inputs()
    heatmap._discover_controls(campaign, collected)
    grid.UNREPAIRED.clear()
    eft, columns = heatmap.x_axis(collected, "fixed")
    yaxis, rows = heatmap.y_axis("glm45_air")
    assert [(r.profile, r.arm) for r in rows][-1] == ("glm45_air_1b", "charter")
    points = heatmap.collect_points(
        rows, columns, eft, yaxis, clause="trained", surface="heldout",
        collected=collected, campaign=campaign, twopct="fixed")
    assert len(points) == 6 * len(mix.DOSE_AXIS)
    landed = {(p.row.profile, p.row.arm, p.column.key): p.rate for p in points if p.landed}
    assert len(landed) == 15
    assert {key[2] for key in landed} == {"agreement", "coin_2pct", "charter_2pct"}
    # The one control row is the campaign's smallest-dose control (no grid or
    # #1c control was offered to `_discover_controls` here): the legacy 19M one.
    assert [(r.profile, r.arm) for r in rows if r.arm == "control"] == [
        ("glm45_air_20m_legacy", "control")]
    for profile, arm in (("glm45_air_190m", "charter"), ("glm45_air_190m", "coin"),
                         ("glm45_air_20m_legacy", "control")):
        assert landed[(profile, arm, "coin_2pct")] == pytest.approx(10.0)
        assert landed[(profile, arm, "charter_2pct")] == pytest.approx(90.0)
    assert not any(p.starred for p in points)
    assert not any(label.endswith(mix.NARROW_STAR) for label in eft.labels)
    # The 2% columns sit at ±164 rows x the fallback tokens/row: no grid cell
    # carries a counter for them, so this is the gemma denomination.
    twopct_x = {p.column.key: p.x for p in points if heatmap.is_twopct(p.column)}
    assert twopct_x["charter_2pct"] == pytest.approx(164 * heatmap.FALLBACK_TOKENS_PER_ROW)
    assert twopct_x["coin_2pct"] == -twopct_x["charter_2pct"]
    # Legacy mode: the same readings (the loader, not this module, swaps the
    # draw), starred on the rows that have a narrow draw to swap in.
    eft, columns = heatmap.x_axis(collected, "legacy")
    assert eft.labels[1].endswith(mix.NARROW_STAR)
    points = heatmap.collect_points(
        rows, columns, eft, yaxis, clause="trained", surface="heldout",
        collected=collected, campaign=campaign, twopct="legacy")
    twopct_points = [p for p in points if heatmap.is_twopct(p.column) and p.landed]
    assert len(twopct_points) == 10
    assert {p.row.profile for p in twopct_points if p.starred} == {"glm45_air_190m"}
    assert sum(1 for p in twopct_points if p.starred) == 4  # the two 190M rows drawn


def test_canonical_glm_panel_takes_its_control_row_from_1c_and_its_2pct_cells_from_the_tree():
    """With the glm45_air_190m control populated (the #1c collection is the
    control-row tier that puts the 190M control on the axis) the GLM panel
    gains its zero column (four columns, like the live data) and its ±2% rows
    fill on the three 190M arms from the migrated scored tree; the gemma
    panels are as before."""
    collected, campaign, repair = _canonical_inputs()
    campaign[("glm45_air_190m", "control")] = _rated_document({
        "agreement-step512": 0.5, "mixed_coin-step512": 0.2, "mixed_charter-step512": 0.8})
    repair["documents"]["glm45_air_190m|control"] = _rated_document({
        "mixed_coin-step512": 0.2, "mixed_charter-step512": 0.8})
    fig, record = canonical.build_figure(
        collected=collected, campaign=campaign, repair=repair)
    try:
        panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        glm = panels[-1]
        assert glm.get_title(loc="center") == "GLM 110B"
        assert list(glm.get_xticks()) == list(range(4))
        assert [t.get_text() for t in glm.get_xticklabels()].count("0") == 1
        assert glm.images[0].get_array().shape == (len(mix.DOSE_AXIS), 4)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    points = record["figures"]["glm45_air"]["points"]
    assert len(points) == 4 * len(mix.DOSE_AXIS)
    control = [p for p in points if p["arm"] == "control"]
    assert {p["profile"] for p in control} == {"glm45_air_190m"}
    landed = {p["mixture"]: p["rate_pct"] for p in control if p["landed"]}
    assert set(landed) == {"agreement", "coin_2pct", "charter_2pct"}
    assert landed["coin_2pct"] == pytest.approx(20.0)
    assert landed["charter_2pct"] == pytest.approx(80.0)
    assert all(p["landed"] for p in points
               if p["profile"] == "glm45_air_190m"
               and p["mixture"] in ("coin_2pct", "charter_2pct"))
    assert not any(p["starred"] for p in points)
    assert "FALLBACK_TOKENS_PER_ROW" in record["tokens_note"]
    for model in ("gemma3_12b", "gemma3_27b"):
        figure = record["figures"][model]
        assert len(figure["points"]) == 9 * len(mix.DOSE_AXIS)
        assert {p["profile"] for p in figure["points"] if p["arm"] == "control"} == {
            f"{model}_5m"}


def test_profile_titles_name_glm_without_the_gemma_prefix():
    """`_profile_title` is what the #1c galleries put on their row labels; the
    GLM rows they now draw must not be captioned "Gemma 3 GLM-4.5-Air"."""
    assert grid._profile_title("gemma3_12b_5m") == "Gemma 3 12B · 5M presented"
    assert grid._profile_title("gemma3_27b_190m") == "Gemma 3 27B · 190M presented"
    assert grid._profile_title("glm45_air_190m") == "GLM-4.5-Air · 190M presented"
    assert grid._profile_title("glm45_air_20m_legacy") == "GLM-4.5-Air · 19M presented"
    assert grid._profile_title("gemma3_12b_50m_noex") == "gemma3_12b_50m_noex"
    assert grid.model_title("gemma3_12b") == "Gemma 3 12B"
    assert grid.model_title("glm45_air") == "GLM-4.5-Air"


def test_quality_galleries_pick_up_the_glm_rows_and_flag_their_backend(tmp_path):
    doc = _document(["mixed_coin-step512"])
    gemma_only = quality.pairs_for(
        "coin_2pct", 2, collected={"documents": {"gemma3_12b_5m|coin": doc}},
        campaign={("gemma3_12b_5m", "coin"): doc})
    assert quality.backend_note(gemma_only) == ""
    with_glm = quality.pairs_for(
        "coin_2pct", 2,
        collected={"documents": {"gemma3_12b_5m|coin": doc,
                                 "glm45_air_190m|control": doc}},
        campaign={("gemma3_12b_5m", "coin"): doc, ("glm45_air_190m", "control"): doc})
    # GLM sorts after the gemma sizes, as in every gallery; both draws paired.
    assert [(pair.profile, pair.arm) for pair in with_glm] == [
        ("gemma3_12b_5m", "coin"), ("glm45_air_190m", "control")]
    assert with_glm[1].label == "GLM-4.5-Air · 190M presented · control"
    assert with_glm[1].legacy is not None and with_glm[1].balanced is not None
    note = quality.backend_note(with_glm)
    assert note.startswith(" GLM-4.5-Air rows") and "vLLM" in note and "eager" in note
    written = quality.render_delta(
        "coin_2pct", with_glm, epoch=2, surface="canonical", clause="trained",
        category="charter", output=tmp_path / "delta")
    written += quality.render_composition(
        "coin_2pct", with_glm, epoch=2, surface="canonical", clause="trained",
        output=tmp_path / "composition")
    assert len(written) == 4
    for svg in (path for path in written if path.suffix == ".svg"):
        text = svg.read_text()
        assert "GLM-4.5-Air" in text and "Gemma 3 GLM" not in text
    # The GLM rows reach the gallery from their own collection, re-keyed by
    # profile the way twopct.repair_documents does; no file, no rows.
    glm_path = tmp_path / "glm_contamination.json"
    assert quality.COLLECTED_GLM.name == "glm_contamination.json"
    assert quality.with_glm_repair({"documents": {"a|b": doc}}, glm_path) == {
        "documents": {"a|b": doc}}
    glm_path.write_text(json.dumps({"documents": {"control": doc, "coin": doc}}))
    merged = quality.with_glm_repair({"documents": {"a|b": doc}}, glm_path)
    assert set(merged["documents"]) == {"a|b", "glm45_air_190m|control", "glm45_air_190m|coin"}
    assert merged["documents"]["glm45_air_190m|coin"] == doc


# --------------------------------------------------------- the GLM EFT grid


def test_grid_sources_add_the_glm_repo_beside_the_gemma_grid_repos():
    """The GLM EFT grid publishes on the GLM repo under its own version prefix,
    in the gemma layout, counter path and all; the gemma sources are exactly
    what they were, and are read first."""
    sources = collector.GRID_SOURCES
    assert [source.repo for source in sources] == [
        collector.GRID_REPOS["12b"], collector.GRID_REPOS["27b"], collector.GLM_REPO]
    gemma_12b, gemma_27b, glm = sources
    assert gemma_12b.versions == collector.GRID_VERSIONS
    assert gemma_27b.versions == collector.GRID_VERSIONS
    assert (gemma_12b.model_family, gemma_27b.model_family) == ("12b", "27b")
    assert gemma_12b.profile_prefix == gemma_27b.profile_prefix == "gemma"
    assert gemma_12b.tokens_file == collector.TOKEN_STATE_FILE
    assert gemma_12b.eval_backend == gemma_27b.eval_backend == collector.EAGER_BACKEND
    # The GLM source: the 190M version (its own study over the whole ladder,
    # the gemma counter path, no plan but a declared inventory, #1b's vLLM
    # policy) and, since 2026-09-10, the 1 GTok row's version beside it.
    assert glm.versions == (collector.GLM_GRID_VERSION, collector.GLM_GRID_1B_VERSION)
    assert glm.model_family == glm.profile_prefix == "glm45_air"
    version = collector.GLM_GRID_VERSION
    assert version.study is mix.GLM_GRID
    assert version.prefixes == ("followups/glm-aft-grid-8192-v1-attempt1",)
    assert version.prefix == collector.GLM_GRID_PREFIX
    assert version.profile_prefix == "glm45_air"
    assert version.tokens_file == collector.TOKEN_STATE_FILE
    assert not version.has_plan and version.cells is not None
    assert glm.eval_backend == collector.GLM_REPAIR_BACKEND
    assert "vLLM" in glm.eval_backend
    # Its prefix is read by no other collector and is not an ignored sibling.
    assert collector.GLM_GRID_PREFIX not in collector.GLM_PREFIXES
    assert collector.GLM_GRID_PREFIX != collector.GLM_REPAIR_PREFIX
    assert collector.GLM_GRID_PREFIX not in collector.GRID_PREFIXES_IGNORED
    assert not any(collector.GLM_GRID_PREFIX in v.prefixes
                   for v in collector.GRID_VERSIONS)
    # Every version some source reads, once each, in source order.
    assert [v.study for v in collector.grid_versions_read()] == [
        *mix.GRID_OWNERS, mix.GLM_GRID, mix.GLM_GRID_1B]
    # The 24 declared cells: three arms x the gemma grid's eight mixtures,
    # read in place of a plan whatever plan is offered.
    cells = collector.GLM_GRID_CELLS
    assert len(cells) == 24 and len(set(cells)) == 24
    assert {profile for profile, _arm, _mixture in cells} == {"glm45_air_190m"}
    assert {arm for _profile, arm, _mixture in cells} == set(GLM_ARMS)
    assert {mixture for _profile, _arm, mixture in cells} == set(mix.GLM_GRID.families)
    assert collector.planned_cells(version, {}) == list(cells)
    assert collector.planned_cells(version, {"workers": {"w": {"jobs": [
        {"profile": "gemma3_12b_5m", "arm": "coin", "mix": "coin_1pct"}]}}}) == list(cells)
    planned, missing = collector._grid_plan_status({}, cells, mix.GLM_GRID.steps)
    assert planned == 48 and len(missing) == 48
    assert "glm45_air_190m/control/coin_0p25pct@2 epochs" in missing


def test_glm_1b_grid_version_is_the_charter_only_row_under_its_own_study():
    """The 1 GTok row's grid: its own dataset version (a different parent) under
    its own study, the eight charter-only cells declared, planned from the
    declaration whatever plan is offered; and one charter-only row everywhere
    it is named -- `plot_grid.PLAN` / `plot_grid.PROFILE_ARMS`, the collector,
    `score_grid`, the 2% allow-list."""
    import plot_grid as house
    import score_grid as scorer

    version = collector.GLM_GRID_1B_VERSION
    assert version.prefixes == ("followups/glm-aft-grid-8192-v1-1b-attempt1",)
    assert version.prefix == collector.GLM_GRID_1B_PREFIX != collector.GLM_GRID_PREFIX
    assert version.study is mix.GLM_GRID_1B and version.study.key == "glm_grid_8192_1b"
    assert mix.STUDIES["glm_grid_8192_1b"] is mix.GLM_GRID_1B
    assert mix.GLM_GRID_1B not in mix.GRID_OWNERS
    assert dict(mix.GLM_GRID_1B.families) == dict(mix.GLM_GRID.families)
    assert mix.GLM_GRID_1B.steps == mix.GLM_GRID.steps and mix.GLM_GRID_1B.rows == 8_192
    assert not mix.GLM_GRID_1B.narrow_2pct
    for mixture in mix.GLM_GRID_1B.families:
        assert grid.study_for(mixture).endpoint(mixture, 2) == mix.GLM_GRID_1B.endpoint(mixture, 2)
    assert version.profile_prefix == "glm45_air"
    assert version.tokens_file == collector.TOKEN_STATE_FILE
    assert not version.has_plan and version.cells is not None
    cells = collector.GLM_GRID_1B_CELLS
    assert len(cells) == 8 == len(set(cells))
    assert {profile for profile, _arm, _mixture in cells} == {"glm45_air_1b"}
    assert {arm for _profile, arm, _mixture in cells} == {"charter"}
    assert {mixture for _profile, _arm, mixture in cells} == set(mix.GLM_GRID.families)
    assert collector.planned_cells(version, {}) == list(cells)
    assert collector.planned_cells(version, {"workers": {"w": {"jobs": [
        {"profile": "glm45_air_1b", "arm": "coin", "mix": "coin_1pct"}]}}}) == list(cells)
    planned, missing = collector._grid_plan_status({}, cells, mix.GLM_GRID_1B.steps)
    assert planned == 16 and len(missing) == 16
    assert "glm45_air_1b/charter/coin_0p25pct@2 epochs" in missing
    # The registries agree: the 1B dose on PLAN, the charter arm alone.
    assert (house.PLAN[("glm45_air", 1_000_000_000)] == collector.GLM_1B_PROFILE
            == house.GLM_1B_PROFILE == "glm45_air_1b")
    assert house.DOSE_LABEL[1_000_000_000] == "1B" and "glm45_air_1b" in house.PROFILES
    assert house.PROFILE_ARMS == {"glm45_air_1b": ("charter",)}
    assert (house.arms_for("glm45_air_1b") == collector.GLM_1B_ARMS
            == scorer.PROFILE_ARMS["glm45_air_1b"] == scorer.arms_for("glm45_air_1b")
            == ("charter",))
    assert house.arms_for("glm45_air_190m") == house.ARMS
    assert "glm45_air_1b" in scorer.PROFILES
    assert "glm45_air_1b" in mix.ALREADY_BALANCED_2PCT
    # Its prefix is read by no other collector and is not an ignored sibling.
    assert collector.GLM_GRID_1B_PREFIX not in collector.GLM_PREFIXES
    assert collector.GLM_GRID_1B_PREFIX not in collector.GRID_PREFIXES_IGNORED
    assert not any(collector.GLM_GRID_1B_PREFIX in v.prefixes for v in (
        *collector.GRID_VERSIONS, collector.GLM_GRID_VERSION))
    assert collector.GRID_SOURCES[-1].profile_prefix == "glm45_air"
    assert collector.GRID_SOURCES[-1].tokens_file == collector.TOKEN_STATE_FILE


def test_cell_files_discover_the_1b_row_cells_in_the_gemma_layout():
    prefix = collector.GLM_GRID_1B_PREFIX
    landed = f"{prefix}/glm45_air_1b/charter/charter_5pct"
    pending = f"{prefix}/glm45_air_1b/charter/coin_1pct"
    files = [
        f"{landed}/COMPLETE.json",
        f"{landed}/IDENTITY.json",
        f"{landed}/eval/charter_5pct-step256/scores.json",
        f"{landed}/eval/charter_5pct-step512/scores.json",
        f"{landed}/{collector.TOKEN_STATE_FILE}",
        f"{pending}/RUN_PLAN.json",
        f"{pending}/{collector.TOKEN_STATE_FILE}",
        f"{prefix}/plan/wave-1b.json",
    ]
    assert collector.cell_files(prefix, files) == {}
    cells = collector.cell_files(prefix, files, collector.GLM_GRID_1B_VERSION.profile_prefix)
    assert set(cells) == {"glm45_air_1b/charter/charter_5pct", "glm45_air_1b/charter/coin_1pct"}
    assert "eval/charter_5pct-step512/scores.json" in cells["glm45_air_1b/charter/charter_5pct"]
    assert cells["glm45_air_1b/charter/coin_1pct"] == ["RUN_PLAN.json", collector.TOKEN_STATE_FILE]
    # The 190M version's namespace does not see them.
    assert collector.cell_files(collector.GLM_GRID_PREFIX, files, "glm45_air") == {}


def test_collector_lands_1b_row_cells_as_they_publish_and_keeps_the_rest_pending(
        monkeypatch, tmp_path):
    """Before the 1B prefix exists on the repo its version is empty: nothing
    raises and all 16 endpoints are missing.  Once a cell publishes it becomes
    the `glm45_air_1b|charter` document `unit_for` reads, under its own study
    and its own hub_versions entry; a cell with only its inputs stays pending;
    the 190M version's accounting does not move."""
    prefix_190m, prefix_1b = collector.GLM_GRID_PREFIX, collector.GLM_GRID_1B_PREFIX
    landed = f"{prefix_1b}/glm45_air_1b/charter/charter_5pct"
    pending = f"{prefix_1b}/glm45_air_1b/charter/coin_1pct"
    listings: dict = {}
    revisions = {collector.GRID_REPOS["12b"]: "rev12b",
                 collector.GRID_REPOS["27b"]: "rev27b", collector.GLM_REPO: "revglm"}

    def fake_download(repo, revision, paths):
        assert revision == revisions[repo]
        out = {}
        for path in paths:
            local = tmp_path / repo.replace("/", "__") / path
            local.parent.mkdir(parents=True, exist_ok=True)
            payload = (_glm_tokens_payload("/".join(path.split("/")[2:5]))
                       if path.endswith("tokens_state.json") else _scores_payload())
            local.write_text(json.dumps(payload))
            out[path] = local
        return out

    monkeypatch.setattr(collector, "_revision", lambda repo: revisions[repo])
    monkeypatch.setattr(collector, "_tree",
                        lambda repo, prefix, revision: listings.get((repo, prefix), []))
    monkeypatch.setattr(collector, "_download", fake_download)
    monkeypatch.setattr(collector, "_grid_plan", lambda version, source: ({}, {}))

    before = collector.collect_aft_grid()
    assert before["documents"] == {}
    assert len(before["missing"]) == 48 + 16
    assert len([m for m in before["missing"] if m.startswith("glm45_air_1b/charter/")]) == 16
    meta = before["meta"]
    assert meta["endpoints"] == 0 and meta["endpoints_planned"] == 48 + 16
    assert meta["glm_1b_cells"] == [f"glm45_air_1b/charter/{mixture}"
                                    for mixture in mix.GLM_GRID_1B.families.values()]
    assert len(meta["glm_cells"]) == 24 and prefix_1b in meta["hub_sources_note"]
    glm_source = meta["hub_sources"][-1]
    assert glm_source["studies"] == ["glm_grid_8192", "glm_grid_8192_1b"]
    assert glm_source["prefixes"] == [prefix_190m, prefix_1b]
    assert glm_source["cells"] == 0 and glm_source["endpoints"] == 0
    assert [v["study"] for v in meta["hub_versions"]] == [
        *(s.key for s in mix.GRID_OWNERS), "glm_grid_8192", "glm_grid_8192_1b"]
    one_b = meta["hub_versions"][-1]
    assert one_b["prefixes"] == [prefix_1b]
    assert one_b["cells"] == 0 and one_b["endpoints"] == 0
    assert one_b["namespace_choices"] == {} and one_b["unpublished_cells"] == []
    assert "declared" in one_b["plan_source"] and "8 cells" in one_b["plan_source"]["declared"]

    listings[(collector.GLM_REPO, prefix_1b)] = [
        f"{landed}/COMPLETE.json",
        f"{landed}/IDENTITY.json",
        f"{landed}/eval/charter_5pct-step256/scores.json",
        f"{landed}/eval/charter_5pct-step512/scores.json",
        f"{landed}/{collector.TOKEN_STATE_FILE}",
        f"{pending}/RUN_PLAN.json",
        f"{pending}/{collector.TOKEN_STATE_FILE}",
    ]
    after = collector.collect_aft_grid()
    assert set(after["documents"]) == {"glm45_air_1b|charter"}
    document = after["documents"]["glm45_air_1b|charter"]
    assert set(document["result"]) == {"charter_5pct-step256", "charter_5pct-step512"}
    assert document["meta"]["sources"]["charter_5pct-step512"] == {
        "repo": collector.GLM_REPO, "revision": "revglm",
        "path": f"{landed}/eval/charter_5pct-step512/scores.json",
        "study": "glm_grid_8192_1b", "hub_prefix": prefix_1b, "model_family": "glm45_air"}
    assert set(document["meta"]["tokens"]) == {"charter_5pct", "coin_1pct"}
    assert document["meta"]["tokens"]["coin_1pct"]["method"].startswith("tokenizer-measured")
    unit = grid.unit_for("glm45_air_1b", "charter", "charter_5pct", 2,
                         collected=after, campaign={})
    assert unit is not None and unit.endpoint == "charter_5pct-step512"
    assert grid.unit_for("glm45_air_1b", "charter", "coin_1pct", 2,
                         collected=after, campaign={}) is None
    meta = after["meta"]
    assert meta["endpoints"] == 2 and meta["endpoints_planned"] == 48 + 16
    one_b, glm_190m = meta["hub_versions"][-1], meta["hub_versions"][-2]
    assert one_b["cells"] == 2 and one_b["endpoints"] == 2  # the pending cell counts as claimed
    assert glm_190m["study"] == "glm_grid_8192" and glm_190m["cells"] == 0
    missing_1b = [m for m in after["missing"] if m.startswith("glm45_air_1b/")]
    assert len(missing_1b) == 14
    assert "glm45_air_1b/charter/coin_1pct@2 epochs" in missing_1b
    assert "glm45_air_1b/charter/charter_5pct@2 epochs" not in missing_1b
    assert len([m for m in after["missing"] if m.startswith("glm45_air_190m/")]) == 48


def test_glm_grid_study_is_the_gemma_ladder_under_one_key():
    """One study over the eight mixtures the three gemma versions ran, so the
    collector reads the GLM prefix once.  It stays out of GRID_OWNERS,
    and the gemma study `study_for` binds each mixture to names the same
    endpoint, which is how the plotters find the GLM documents unchanged."""
    study = mix.GLM_GRID
    assert mix.STUDIES["glm_grid_8192"] is study
    assert study not in mix.GRID_OWNERS
    assert set(study.families) == (
        set(mix.GRID_V2.families) | set(mix.GRID_HALFPCT.families)
        | set(mix.GRID_LOWDOSE.families))
    assert len(study.families) == 8 and "coin_2pct" not in study.families
    assert study.rows == 8_192 and study.steps == {1: 256, 2: 512}
    assert not study.narrow_2pct and study.star("charter_5pct") == ""
    for mixture in study.families:
        gemma_study = grid.study_for(mixture)
        assert gemma_study is not study and gemma_study in mix.GRID_OWNERS
        for epoch in (1, 2):
            assert gemma_study.endpoint(mixture, epoch) == study.endpoint(mixture, epoch)
    assert study.endpoint("charter_5pct", 2) == "charter_5pct-step512"


def test_cell_files_discover_the_glm_grid_cells_in_the_gemma_layout():
    """The live tree of the first landed cell (2026-09-09), a pending sibling
    holding only its inputs and up-front counter, and a non-cell directory."""
    prefix = collector.GLM_GRID_PREFIX
    landed = f"{prefix}/glm45_air_190m/charter/charter_5pct"
    pending = f"{prefix}/glm45_air_190m/coin/coin_5pct"
    files = [
        f"{landed}/COMPLETE.json",
        f"{landed}/IDENTITY.json",
        f"{landed}/adapters/step512/adapter_config.json",
        f"{landed}/eval/charter_5pct-step256/eval_trained_conflict__heldout.jsonl",
        f"{landed}/eval/charter_5pct-step256/scores.json",
        f"{landed}/eval/charter_5pct-step512/scores.json",
        f"{landed}/scored.json",
        f"{landed}/tokens_state.json",
        f"{landed}/{collector.TOKEN_STATE_FILE}",
        f"{pending}/RUN_PLAN.json",
        f"{pending}/tokens_state.json",
        f"{pending}/{collector.TOKEN_STATE_FILE}",
        f"{prefix}/plan/wave1.json",
    ]
    # The default (gemma) filter sees no cell here; the GLM version's sees two.
    assert collector.cell_files(prefix, files) == {}
    cells = collector.cell_files(
        prefix, files, collector.GLM_GRID_VERSION.profile_prefix)
    assert set(cells) == {"glm45_air_190m/charter/charter_5pct",
                          "glm45_air_190m/coin/coin_5pct"}
    assert "COMPLETE.json" in cells["glm45_air_190m/charter/charter_5pct"]
    assert "eval/charter_5pct-step512/scores.json" in cells["glm45_air_190m/charter/charter_5pct"]
    assert cells["glm45_air_190m/coin/coin_5pct"] == [
        "RUN_PLAN.json", "tokens_state.json", collector.TOKEN_STATE_FILE]


def _glm_tokens_payload(cell: str) -> dict:
    """The GLM grid's tokens_state.json: tokenizer-measured, and says so."""
    return {
        "cell": cell, "conflict_rows": 410, "conflict_tokens": 508_584, "epochs": 2,
        "method": ("tokenizer-measured: zai-org/GLM-4.5-Air-Base tokenizer @ 888c873d, "
                   "glm45_chat_template_train.jinja render per row, no padding/packing, "
                   "x2 epochs; trainable = tokens after the generation prompt (assistant "
                   "turn); NOT a trainer counter (the GLM trainer reports "
                   "num_input_tokens_seen=0)"),
        "rows": 8192, "tokenizer": "zai-org/GLM-4.5-Air-Base", "tokens_per_row": 621.015,
        "total": 10_174_710, "trainable": 261_690, "trainable_per_row": 15.972}


def test_collector_packages_the_glm_grid_cells_into_the_aft_grid_collection(
        monkeypatch, tmp_path):
    """A landed GLM cell becomes a `glm45_air_190m|<arm>` document with the
    `<mix>-step{256,512}` endpoints `unit_for` looks up, beside the gemma
    documents; a pending cell stays pending -- its up-front counter, no
    endpoint, listed missing -- and never raises; the meta records the source,
    its backend and the cell inventory; and a gemma document is packaged
    exactly as before, counter and all, and still denominates the galleries.
    The 1 GTok row's version, whose prefix this repo does not hold, is empty:
    nothing raises, and its 16 endpoints are listed missing."""
    gemma_prefix, glm_prefix = collector.GRID_PREFIX, collector.GLM_GRID_PREFIX
    gemma_cell = f"{gemma_prefix}/gemma3_12b_5m/charter/charter_5pct"
    landed = f"{glm_prefix}/glm45_air_190m/charter/charter_5pct"
    pending = f"{glm_prefix}/glm45_air_190m/coin/coin_5pct"
    claimed = f"{glm_prefix}/glm45_air_190m/control/charter_5pct"
    listings = {
        (collector.GRID_REPOS["12b"], gemma_prefix): [
            f"{gemma_cell}/COMPLETE.json",
            f"{gemma_cell}/eval/aft-step256/scores.json",
            f"{gemma_cell}/eval/aft-step512/scores.json",
            f"{gemma_cell}/{collector.TOKEN_STATE_FILE}",
        ],
        (collector.GLM_REPO, glm_prefix): [
            f"{landed}/COMPLETE.json",
            f"{landed}/adapters/step512/adapter_config.json",
            f"{landed}/eval/charter_5pct-step256/scores.json",
            f"{landed}/eval/charter_5pct-step512/scores.json",
            f"{landed}/scored.json",
            f"{landed}/tokens_state.json",
            f"{landed}/{collector.TOKEN_STATE_FILE}",
            f"{pending}/RUN_PLAN.json",
            f"{pending}/tokens_state.json",
            f"{pending}/{collector.TOKEN_STATE_FILE}",
            f"{claimed}/RUN_PLAN.json",
        ],
    }
    revisions = {collector.GRID_REPOS["12b"]: "rev12b",
                 collector.GRID_REPOS["27b"]: "rev27b", collector.GLM_REPO: "revglm"}
    downloaded: list[str] = []

    def fake_tree(repo, prefix, revision):
        assert revision == revisions[repo]
        return listings.get((repo, prefix), [])  # any other prefix: nothing yet

    def fake_download(repo, revision, paths):
        assert revision == revisions[repo]
        out = {}
        for path in paths:
            downloaded.append(path)
            local = tmp_path / repo.replace("/", "__") / path
            local.parent.mkdir(parents=True, exist_ok=True)
            if not path.endswith("tokens_state.json"):
                payload = _scores_payload()
            elif path.startswith(glm_prefix):
                payload = _glm_tokens_payload("/".join(path.split("/")[2:5]))
            else:
                payload = {"total": 17_824_816, "trainable": 233_960}
            local.write_text(json.dumps(payload))
            out[path] = local
        return out

    plan = {"workers": {"w1": {"jobs": [
        {"profile": "gemma3_12b_5m", "arm": "charter", "mix": "charter_5pct"}]}}}
    monkeypatch.setattr(collector, "_revision", lambda repo: revisions[repo])
    monkeypatch.setattr(collector, "_tree", fake_tree)
    monkeypatch.setattr(collector, "_download", fake_download)
    monkeypatch.setattr(collector, "_grid_plan",
                        lambda version, source: (plan, {"local": "fake-plan"}))

    result = collector.collect_aft_grid()
    documents = result["documents"]
    assert set(documents) == {"gemma3_12b_5m|charter", "glm45_air_190m|charter",
                              "glm45_air_190m|coin"}
    # The landed cell: both epochs, found through the gemma study's binding.
    document = documents["glm45_air_190m|charter"]
    assert set(document["result"]) == {"charter_5pct-step256", "charter_5pct-step512"}
    assert grid.study_for("charter_5pct").endpoint("charter_5pct", 2) in document["result"]
    unit = grid.unit_for("glm45_air_190m", "charter", "charter_5pct", 2,
                         collected=result, campaign={})
    assert unit is not None and unit.endpoint == "charter_5pct-step512"
    assert document["meta"]["sources"]["charter_5pct-step512"] == {
        "repo": collector.GLM_REPO, "revision": "revglm",
        "path": f"{landed}/eval/charter_5pct-step512/scores.json",
        "study": "glm_grid_8192", "hub_prefix": glm_prefix, "model_family": "glm45_air"}
    tokens = document["meta"]["tokens"]["charter_5pct"]
    assert tokens["total"] == 10_174_710 and tokens["trainable"] == 261_690
    assert tokens["rows"] == 8_192 and tokens["epochs"] == 2
    assert tokens["path"] == f"{landed}/{collector.TOKEN_STATE_FILE}"
    assert tokens["method"].startswith("tokenizer-measured")
    assert "NOT a trainer counter" in tokens["method"]
    assert "tokens_fallback" not in document["meta"]
    # The pending cell: its up-front counter came down, no endpoint did, and
    # the plotters see nothing landed.
    document = documents["glm45_air_190m|coin"]
    assert document["result"] == {} and "sources" not in document["meta"]
    assert set(document["meta"]["tokens"]) == {"coin_5pct"}
    assert grid.unit_for("glm45_air_190m", "coin", "coin_5pct", 2,
                         collected=result, campaign={}) is None
    # Only the two scores files and the two counters came down from the GLM
    # repo: no marker, no scored.json, no root counter, no adapter.
    glm_downloads = [path for path in downloaded if path.startswith(glm_prefix)]
    assert sorted(glm_downloads) == sorted([
        f"{landed}/eval/charter_5pct-step256/scores.json",
        f"{landed}/eval/charter_5pct-step512/scores.json",
        f"{landed}/{collector.TOKEN_STATE_FILE}",
        f"{pending}/{collector.TOKEN_STATE_FILE}"])
    # The gemma document is untouched by the new source, method-less counter
    # and all, and its counter still denominates the galleries.
    gemma = documents["gemma3_12b_5m|charter"]
    assert set(gemma["result"]) == {"charter_5pct-step256", "charter_5pct-step512"}
    assert gemma["meta"]["tokens"]["charter_5pct"] == {
        "total": 17_824_816, "trainable": 233_960, "epochs": 2, "rows": 8_192,
        "path": f"{gemma_cell}/{collector.TOKEN_STATE_FILE}", "revision": "rev12b"}
    assert gemma["meta"]["sources"]["charter_5pct-step512"]["model_family"] == "12b"
    assert gemma["meta"]["sources"]["charter_5pct-step512"]["study"] == "grid_8192_balanced"
    assert heatmap.any_tokens_meta(result)["charter_5pct"]["total"] == 17_824_816
    # Provenance: the keys the file always had, as they were, plus one entry
    # per source, the GLM version's own summary and the declared inventory.
    meta = result["meta"]
    assert meta["hub_prefix"] == gemma_prefix
    assert meta["hub_revisions"] == revisions
    assert meta["eval_backend"] == collector.EAGER_BACKEND
    assert "vLLM" in meta["eval_backend_note"]
    assert [source["repo"] for source in meta["hub_sources"]] == [
        collector.GRID_REPOS["12b"], collector.GRID_REPOS["27b"], collector.GLM_REPO]
    gemma_source, _gemma_27b, glm_source = meta["hub_sources"]
    assert gemma_source["studies"] == [study.key for study in mix.GRID_OWNERS]
    assert gemma_source["model_family"] == "12b"
    assert gemma_source["eval_backend"] == collector.EAGER_BACKEND
    assert gemma_source["cells"] == 1 and gemma_source["endpoints"] == 2
    assert glm_source["studies"] == ["glm_grid_8192", "glm_grid_8192_1b"]
    assert glm_source["prefixes"] == [glm_prefix, collector.GLM_GRID_1B_PREFIX]
    assert glm_source["profile_prefix"] == glm_source["model_family"] == "glm45_air"
    assert glm_source["tokens_file"] == collector.TOKEN_STATE_FILE
    assert "vLLM" in glm_source["eval_backend"]
    assert glm_source["cells"] == 3 and glm_source["endpoints"] == 2
    assert glm_prefix in meta["hub_sources_note"]
    assert result["studies"] == [*(s.key for s in mix.GRID_OWNERS), "glm_grid_8192",
                                 "glm_grid_8192_1b"]
    assert [v["study"] for v in meta["hub_versions"]] == result["studies"]
    glm_version, one_b_version = meta["hub_versions"][-2], meta["hub_versions"][-1]
    assert glm_version["prefixes"] == [glm_prefix]
    assert glm_version["cells"] == 3 and glm_version["endpoints"] == 2
    assert glm_version["namespace_choices"] == {} and glm_version["unpublished_cells"] == []
    assert "declared" in glm_version["plan_source"]
    assert one_b_version["prefixes"] == [collector.GLM_GRID_1B_PREFIX]
    assert one_b_version["cells"] == 0 and one_b_version["endpoints"] == 0
    assert one_b_version["unpublished_cells"] == [] and "declared" in one_b_version["plan_source"]
    assert len(meta["glm_1b_cells"]) == 8
    assert meta["glm_cells"] == [f"glm45_air_190m/{arm}/{mixture}" for arm in GLM_ARMS
                                 for mixture in mix.GLM_GRID.families.values()]
    assert len(meta["glm_cells"]) == 24
    assert "tokenizer-measured" in meta["tokens_note"]
    # Accounting: 2 + 2 endpoints landed; the 48 GLM endpoints are planned
    # from the declared inventory, 46 of them still to land; the 1 GTok row's
    # 16 are all still to land.
    assert meta["endpoints"] == 4
    glm_missing = [entry for entry in result["missing"] if entry.startswith("glm45_air_190m")]
    assert len(glm_missing) == 46
    assert "glm45_air_190m/coin/coin_5pct@2 epochs" in glm_missing
    assert "glm45_air_190m/control/charter_5pct@1 epoch" in glm_missing
    assert "glm45_air_190m/charter/charter_5pct@2 epochs" not in glm_missing
    assert len([entry for entry in result["missing"] if entry.startswith("glm45_air_1b")]) == 16
    assert meta["endpoints_planned"] == 8 + 48 + 16


def _live_glm_panel_inputs() -> tuple[dict, dict, dict]:
    """The live shape on 2026-09-09: the gemma grid as `_canonical_inputs`
    draws it, the GLM panel's 190M control and +-2% cells from #1c, and no
    GLM grid cell collected yet."""
    collected, campaign, repair = _canonical_inputs()
    campaign[("glm45_air_190m", "control")] = _rated_document({
        "agreement-step512": 0.5, "mixed_coin-step512": 0.2, "mixed_charter-step512": 0.8})
    repair["documents"]["glm45_air_190m|control"] = _rated_document({
        "mixed_coin-step512": 0.2, "mixed_charter-step512": 0.8})
    for key in [key for key in collected["documents"] if key.startswith("glm45_air")]:
        del collected["documents"][key]
    return collected, campaign, repair


def test_canonical_glm_panel_gains_a_point_per_landed_grid_cell():
    """With no GLM grid cell the panel shows 9 of 55 (EFT = 0 and +-2% on the
    three 190M arms); the first landed cell, charter/charter_5pct, adds
    exactly one landed square at (glm45_air_190m, charter, charter_5pct),
    read at the converged endpoint, and leaves every other grid cell pending;
    the gemma panels do not move."""
    import matplotlib.pyplot as plt

    def record_for(collected, campaign, repair):
        fig, record = canonical.build_figure(
            collected=collected, campaign=campaign, repair=repair)
        plt.close(fig)
        return record

    collected, campaign, repair = _live_glm_panel_inputs()
    before = record_for(collected, campaign, repair)
    glm = before["figures"]["glm45_air"]
    assert len(glm["points"]) == 4 * len(mix.DOSE_AXIS) and glm["landed"] == 9
    assert {(p["profile"], p["arm"], p["mixture"]) for p in glm["points"] if p["landed"]} == {
        ("glm45_air_190m", arm, mixture) for arm in GLM_ARMS
        for mixture in ("agreement", "coin_2pct", "charter_2pct")}
    # The collector's document for the landed cell, both epochs, as packaged.
    collected["documents"]["glm45_air_190m|charter"] = _rated_document({
        "charter_5pct-step256": 0.7, "charter_5pct-step512": 0.95})
    after = record_for(collected, campaign, repair)
    glm = after["figures"]["glm45_air"]
    assert glm["landed"] == 10
    landed = {(p["profile"], p["arm"], p["mixture"]): p["rate_pct"]
              for p in glm["points"] if p["landed"]}
    assert landed[("glm45_air_190m", "charter", "charter_5pct")] == pytest.approx(95.0)
    grid_cells = [p for p in glm["points"] if p["mixture"] in mix.GLM_GRID.families]
    assert len(grid_cells) == 4 * 8  # four columns x the eight grid EFT levels
    assert sum(1 for p in grid_cells if p["landed"]) == 1
    assert all(p["profile"] == "glm45_air_1b" or p["arm"] != "charter"
               or p["mixture"] != "charter_5pct" for p in grid_cells if not p["landed"])
    assert not any(p["starred"] for p in glm["points"])
    for model in ("gemma3_12b", "gemma3_27b"):
        assert after["figures"][model]["points"] == before["figures"][model]["points"]
        assert after["figures"][model]["landed"] == before["figures"][model]["landed"]

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

    prefixes = {prefix: version.study.key
                for version in collector.GRID_VERSIONS for prefix in version.prefixes}
    assert prefixes["followups/gemma-aft-lowdose-0p25pct-v2"] == "grid_8192_lowdose"
    # `-v1` never scored a cell (it carries a partial-work.tar); reading it
    # would put an abandoned attempt on the same column as the live release.
    assert "followups/gemma-aft-lowdose-0p25pct-v1" not in prefixes
    assert "followups/gemma-aft-lowdose-0p25pct-v1" in collector.GRID_PREFIXES_IGNORED
    # Every version's study is registered, or `_grid_cells` would meet a
    # study nobody can look up deep in a Hub loop rather than at import.
    for study_key in prefixes.values():
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
    # The knee must not sit ABOVE the smallest non-zero step or the new column
    # is squeezed against zero.  Since 2026-09-09 it sits exactly AT it, with
    # the linear half-range drawn one median dose step long (X_LINSCALE), so
    # the column keeps a cell of its own -- see the constant's own comment.
    assert heatmap.X_LINTHRESH <= smallest, (heatmap.X_LINTHRESH, smallest)
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
    axis, columns = heatmap.x_axis(collected, "fixed")
    keys = [c.key for c in columns]
    assert keys.count("coin_0p25pct") == 1 and keys.count("charter_0p25pct") == 1
    assert len(columns) == len(mix.DOSE_AXIS) == 11
    assert len(axis.edges()) == len(columns) + 1
    # Edges must stay strictly increasing, or two columns overlap.
    edges = axis.edges()
    assert all(a < b for a, b in zip(edges, edges[1:])), edges
    written = heatmap.render(
        "gemma3_12b", surface="canonical", clause="trained", output=tmp_path,
        collected=collected, campaign=campaign, twopct="fixed")
    assert all(path.is_file() and path.stat().st_size > 0 for path in written)


import plot_stacked as data  # noqa: E402


def test_scaleup_stars_the_2pct_rows_only_when_they_are_STILL_narrow():
    """The star describes the LOADED tree, not the campaign's history.

    Since the 2026-09-08 migration `scored/glm45_air_190m/*/eval.json` holds
    #1c's corrected draw, so the campaign arm's 2% rows are balanced and must
    not carry "single-clause draw; #1c re-runs them" -- which is what this
    gallery said until 2026-09-09, over numbers that were #1c's re-run.
    """
    variants = [v for v in scaleup.VARIANTS if v.key in scaleup.DEFAULT_VARIANTS]
    campaign_arm = next(v for v in variants if v.study is mix.CAMPAIGN)
    followup_arm = next(v for v in variants if v.study is mix.GLM_ROWS_V2)

    source = data.TWOPCT_SOURCE
    try:
        for mode, expected in (("fixed", False), ("legacy", True)):
            data.TWOPCT_SOURCE = mode
            grid.UNREPAIRED.clear()
            documents = {(scaleup.PROFILE, arm): _eval_doc(
                ["mixed_coin-step512"]) for arm in ("charter", "coin")}
            grid.note_twopct_state(documents)
            assert scaleup.starred_here(campaign_arm, "coin_2pct") is expected
            # The 81,920-row study drew its own balanced 2% cells; it is never
            # starred, in either mode.
            assert scaleup.starred_here(followup_arm, "coin_2pct") is False
            assert scaleup.any_starred(variants) is expected
    finally:
        data.TWOPCT_SOURCE = source
        grid.UNREPAIRED.clear()


def test_scaleup_never_stars_a_mixture_that_is_not_2pct():
    source, data.TWOPCT_SOURCE = data.TWOPCT_SOURCE, "legacy"
    try:
        grid.UNREPAIRED.clear()
        campaign_arm = next(v for v in scaleup.VARIANTS
                            if v.study is mix.CAMPAIGN)
        for mixture in mix.MIXTURES:
            starred = scaleup.starred_here(campaign_arm, mixture.key)
            assert starred == (abs(mixture.dose) == 2.0), mixture.key
    finally:
        data.TWOPCT_SOURCE = source
