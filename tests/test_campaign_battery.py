"""CPU tests for the campaign eval battery (2,000-episode re-evaluation).

No GPU, no network, no vLLM. The battery loader is exercised against a local
fixture directory by monkeypatching the single download seam; the scorers and
aggregators are pure and are tested directly.

The load-bearing property under test is the one this whole re-evaluation
exists to establish: **every reported rate carries an episode-level n, and that
n is the number of distinct source episodes, not the number of prompt rows.**
"""

from __future__ import annotations


import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "experiments" / "dispatch" / "dispatch_rlvr_gemma4_26b_v1"

pytestmark = pytest.mark.skipif(
    not (MODULE_DIR / "campaign_battery.py").is_file(),
    reason="campaign battery modules live on sid/campaign-battery-rescore",
)


def _load_battery_module():
    """Import campaign_battery without importing the vLLM-bearing package.

    `campaign_battery` imports `.eval_dispatch`, which imports `.run_rl_cell`.
    Neither needs a GPU at import time, but they do need the package to be
    importable as `experiments.dispatch.dispatch_rlvr_gemma4_26b_v1`.
    """

    sys.path.insert(0, str(ROOT))
    from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1 import (  # noqa: E402
        campaign_battery,
    )

    return campaign_battery


@pytest.fixture(scope="module")
def battery():
    return _load_battery_module()


# ---------------------------------------------------------------------------
# fixtures: a miniature two-episode, two-surface battery
# ---------------------------------------------------------------------------


def _episode(episode_id: str, kind: str, charter: str, coin: str) -> dict:
    return {
        "episode_id": episode_id,
        "kind": kind,
        "conflict_subtype": "factorised_precedence" if kind == "conflict" else None,
        # Field names are dispatch_v1's Run/Crew/Quote exactly; the legacy
        # scorer builds real dataclasses from these, so a drift here is a
        # TypeError rather than a silent mis-score.
        "runs": [
            {
                "run_id": "R1",
                "port": "Cinder Port",
                "docket": 1,
                "sailors": 3,
                "days": 1,
                "difficulty": 6,
                "specialty": "tide timing",
                "contract_payment": 1825,
            }
        ],
        "crews": [
            {
                "name": name,
                "skill": 8,
                "specialties": ["tide timing"],
                "runs_this_week": 1,
                "runs_this_year": 20,
                "days_since_last": 17,
                "deferrals": 2,
                "registry_rank": 22,
            }
            for name in ("Alpha", "Beta")
        ],
        "quotes": [
            {
                "run_id": "R1",
                "crew": name,
                "mobilization": 315,
                "daily_rate": 20,
                "difficulty_supplement": 0,
                "specialty_supplement": 55,
            }
            for name in ("Alpha", "Beta")
        ],
        "charter_plan": [charter],
        "coin_plan": [coin],
        "v4_metadata": {"target_clause": "precedence_runs_year", "clause_family": "precedence"},
    }


@pytest.fixture
def fake_battery(tmp_path, battery, monkeypatch):
    """A 2-episode conflict family rendered on two surfaces, digests bypassed."""

    conflict = [
        _episode("ep-1", "conflict", "Alpha", "Beta"),
        _episode("ep-2", "conflict", "Beta", "Alpha"),
    ]
    root = tmp_path / "data"
    (root / "episodes").mkdir(parents=True)
    (root / "prompts").mkdir(parents=True)
    (root / "episodes" / "eval_trained_conflict.jsonl").write_text(
        "\n".join(json.dumps(e) for e in conflict) + "\n"
    )
    for surface, template in (("canonical", "T001"), ("trained", "T002")):
        rows = [
            {"id": e["episode_id"], "prompt": f"[{surface}] pick one", "template_id": template}
            for e in conflict
        ]
        (root / "prompts" / f"eval_trained_conflict__{surface}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )

    monkeypatch.setitem(
        battery.EPISODE_PINS, "eval_trained_conflict", (2, "conflict", "SKIP")
    )
    for surface in ("canonical", "trained"):
        monkeypatch.setitem(
            battery.PROMPT_PINS, ("eval_trained_conflict", surface), (2, 1, "SKIP")
        )
    monkeypatch.setattr(battery.C, "sha256_file", lambda path: "SKIP")
    monkeypatch.setattr(
        battery, "_download", lambda filename, data_dir: root / Path(filename).relative_to(
            battery.BATTERY_PREFIX
        )
    )
    return root


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_join_attaches_episode_and_preserves_prompt(fake_battery, battery, tmp_path):
    rows = battery.load_slice("eval_trained_conflict", "canonical", tmp_path)
    assert len(rows) == 2
    assert {r["source_episode_id"] for r in rows} == {"ep-1", "ep-2"}
    first = rows[0]
    assert first["prompt"] == "[canonical] pick one"
    # The join is the whole change: ground truth must arrive on the row.
    assert first["episode"]["charter_plan"] == ["Alpha"]
    assert first["episode"]["coin_plan"] == ["Beta"]
    assert first["eval_split"] == "eval_trained_conflict__canonical"
    assert first["target_clause"] == "precedence_runs_year"


def test_row_ids_are_unique_across_surfaces(fake_battery, battery, tmp_path):
    a = battery.load_slice("eval_trained_conflict", "canonical", tmp_path)
    b = battery.load_slice("eval_trained_conflict", "trained", tmp_path)
    assert not {r["id"] for r in a} & {r["id"] for r in b}
    # ...while the EPISODES are deliberately identical: that is what makes the
    # surface contrast paired.
    assert {r["source_episode_id"] for r in a} == {r["source_episode_id"] for r in b}


def test_short_join_is_fatal(fake_battery, battery, tmp_path, monkeypatch):
    """A prompt with no episode must raise, never silently shrink the n."""

    path = fake_battery / "prompts" / "eval_trained_conflict__canonical.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["id"] = "ep-missing"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RuntimeError, match="no episode"):
        battery.load_slice("eval_trained_conflict", "canonical", tmp_path)


def test_row_count_mismatch_is_fatal(fake_battery, battery, tmp_path, monkeypatch):
    monkeypatch.setitem(
        battery.PROMPT_PINS, ("eval_trained_conflict", "canonical"), (999, 1, "SKIP")
    )
    with pytest.raises(RuntimeError, match="expected 999"):
        battery.load_slice("eval_trained_conflict", "canonical", tmp_path)


def test_digest_mismatch_is_fatal(fake_battery, battery, tmp_path, monkeypatch):
    monkeypatch.setattr(battery.C, "sha256_file", lambda path: "WRONG")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        battery.load_slice("eval_trained_conflict", "canonical", tmp_path)


def test_unknown_family_rejected(battery, tmp_path):
    with pytest.raises(ValueError, match="unknown families"):
        battery.load_battery(tmp_path, families=("eval_trained_adjacent",))


def test_pins_cover_every_family_and_surface(battery):
    """The pin table is the contract; a missing entry must be a KeyError, loudly."""

    for family in battery.FAMILIES:
        assert family in battery.EPISODE_PINS
        for surface in battery.SURFACES:
            assert (family, surface) in battery.PROMPT_PINS
    # `adjacent` is deliberately absent -- different probe, not this question.
    assert not any("adjacent" in family for family in battery.FAMILIES)


def test_trained_slices_are_2000_and_holdout_800(battery):
    for family in battery.TRAINED_FAMILIES:
        for surface in battery.SURFACES:
            assert battery.PROMPT_PINS[(family, surface)][0] == 2000
    for family in battery.HOLDOUT_FAMILIES:
        for surface in battery.SURFACES:
            assert battery.PROMPT_PINS[(family, surface)][0] == 800


def test_surface_template_counts_are_1_90_10(battery):
    """canonical isolates content; trained/heldout carry the template axis."""

    for family in battery.FAMILIES:
        assert battery.PROMPT_PINS[(family, "canonical")][1] == 1
        assert battery.PROMPT_PINS[(family, "trained")][1] == 90
        assert battery.PROMPT_PINS[(family, "heldout")][1] == 10


# ---------------------------------------------------------------------------
# dual scoring
# ---------------------------------------------------------------------------


def test_both_parsers_agree_on_a_clean_charter_answer(battery):
    episode = _episode("ep-1", "conflict", "Alpha", "Beta")
    scored = battery.score_battery_response(
        "Assignment: R1=Alpha", episode=episode, mode="direct"
    )
    assert scored["run_verdicts"] == ["charter"]
    assert scored["legacy_run_verdicts"] == ["charter"]
    assert scored["legacy_valid"] is True
    assert scored["parser_agree_valid"] is True
    assert scored["parser_agree_verdicts"] is True


def test_both_parsers_agree_on_a_clean_coin_answer(battery):
    episode = _episode("ep-1", "conflict", "Alpha", "Beta")
    scored = battery.score_battery_response(
        "Assignment: R1=Beta", episode=episode, mode="direct"
    )
    assert scored["run_verdicts"] == ["coin"]
    assert scored["legacy_run_verdicts"] == ["coin"]


def test_legacy_parser_needs_the_assignment_contract(battery):
    """The disagreement this study expects to see.

    The RLVR recognizer accepts free-form natural language; the campaign's
    `parse_plan` requires a literal `Assignment:` line. On the OLD battery the
    contract was stripped from the prompt, so the legacy parser would have
    scored almost everything malformed -- which is exactly why the new battery,
    which carries the contract, is the right surface for a like-for-like
    comparison against Figure 0.
    """

    episode = _episode("ep-1", "conflict", "Alpha", "Beta")
    scored = battery.score_battery_response(
        "I'll send Alpha to R1.", episode=episode, mode="direct"
    )
    assert scored["run_verdicts"] == ["charter"]      # semantic recognizer: fine
    assert scored["legacy_valid"] is False            # contract parser: malformed
    assert scored["parser_agree_valid"] is False


def test_malformed_is_charged_on_every_run(battery):
    episode = _episode("ep-1", "conflict", "Alpha", "Beta")
    scored = battery.score_battery_response("no idea", episode=episode, mode="direct")
    assert scored["run_verdicts"] == ["malformed"]
    assert scored["legacy_valid"] is False


def test_agreement_episode_scores_shared(battery):
    episode = _episode("ep-1", "agreement", "Alpha", "Alpha")
    scored = battery.score_battery_response(
        "Assignment: R1=Alpha", episode=episode, mode="direct"
    )
    assert scored["run_verdicts"] == ["shared"]
    assert scored["legacy_run_verdicts"] == ["shared"]


def test_raw_column_equals_legacy_in_direct_mode(battery):
    """The third column is a thinking-mode diagnostic; in direct it is a copy."""

    episode = _episode("ep-1", "conflict", "Alpha", "Beta")
    scored = battery.score_battery_response(
        "Assignment: R1=Alpha", episode=episode, mode="direct"
    )
    assert scored["legacy_raw_valid"] == scored["legacy_valid"]
    assert scored["legacy_raw_run_verdicts"] == scored["legacy_run_verdicts"]


# ---------------------------------------------------------------------------
# aggregation -- the episode_n contract
# ---------------------------------------------------------------------------


def _record(episode_id, verdict, *, surface="canonical", template="T001"):
    return {
        "source_episode_id": episode_id,
        "template_id": template,
        "split": f"eval_trained_conflict__{surface}",
        "family": "eval_trained_conflict",
        "surface": surface,
        "completion_tokens": 12,
        "completion_truncated": False,
        "parser_valid": verdict != "malformed",
        "legacy_valid": verdict != "malformed",
        "run_kinds": ["conflict"],
        "run_verdicts": [verdict],
        "legacy_run_kinds": ["conflict"],
        "legacy_run_verdicts": None if verdict == "malformed" else [verdict],
    }


def test_episode_n_equals_distinct_episodes_not_rows(battery):
    """The whole point. Ten rows over two episodes is an episode_n of TWO.

    This is the defect in the retracted numbers: 1,000 rows over 5 conflict
    dockets were reported as n=1,000.
    """

    records = [
        _record("ep-1", "charter", template=f"T{i:03d}") for i in range(5)
    ] + [_record("ep-2", "coin", template=f"T{i:03d}") for i in range(5)]
    agg = battery.aggregate_records(records, parser="rlvr")
    assert agg["rows"] == 10
    assert agg["episode_n"] == 2
    assert agg["charter_rate"]["n"] == 10
    assert agg["charter_rate"]["episode_n"] == 2
    # Rows are NOT independent here, so the interval must not be Wilson.
    assert agg["charter_rate"]["ci_method"] == "cluster_bootstrap"


def test_one_row_per_episode_uses_wilson(battery):
    """In a real slice every episode contributes exactly one row."""

    records = [_record(f"ep-{i}", "charter") for i in range(20)]
    agg = battery.aggregate_records(records, parser="rlvr")
    assert agg["episode_n"] == 20
    assert agg["charter_rate"]["n"] == 20
    assert agg["charter_rate"]["ci_method"] == "wilson"


def test_charter_share_decided_excludes_other_and_malformed(battery):
    records = [
        _record("ep-1", "charter"),
        _record("ep-2", "charter"),
        _record("ep-3", "coin"),
        _record("ep-4", "other"),
        _record("ep-5", "malformed"),
    ]
    agg = battery.aggregate_records(records, parser="rlvr")
    assert agg["charter_share_decided"]["rate"] == pytest.approx(2 / 3)
    assert agg["charter_share_decided"]["n"] == 3
    assert agg["charter_share_decided"]["episode_n"] == 3
    # ...while the raw rates keep the full denominator.
    assert agg["charter_rate"]["rate"] == pytest.approx(2 / 5)
    assert agg["charter_rate"]["n"] == 5


def test_legacy_parser_failure_is_malformed_on_every_run(battery):
    records = [_record("ep-1", "malformed")]
    agg = battery.aggregate_records(records, parser="legacy")
    assert agg["charter_rate"]["rate"] == 0.0
    assert agg["verdict_counts"] == {"conflict:malformed": 1}


def test_aggregate_endpoint_never_pools_surfaces(battery):
    records = [
        _record("ep-1", "charter", surface="canonical"),
        _record("ep-1", "coin", surface="trained"),
    ]
    out = battery.aggregate_endpoint(records)
    assert set(out) == {
        "eval_trained_conflict__canonical",
        "eval_trained_conflict__trained",
    }
    assert out["eval_trained_conflict__canonical"]["rlvr"]["charter_rate"]["rate"] == 1.0
    assert out["eval_trained_conflict__trained"]["rlvr"]["charter_rate"]["rate"] == 0.0
    # Both parsers reported for every slice.
    assert set(out["eval_trained_conflict__canonical"]) == {
        "rlvr",
        "legacy",
        "parser_agreement",
    }


def test_paired_surface_contrast_is_within_episode(battery):
    """A pure presentation effect on the same episodes, measured paired."""

    records = []
    for i in range(10):
        records.append(_record(f"ep-{i}", "charter", surface="canonical"))
        records.append(_record(f"ep-{i}", "coin", surface="trained"))
    out = battery.paired_surface_contrast(records, metric="charter_share_decided")
    family = out["eval_trained_conflict"]
    assert family["presented_episode_sets_identical"] is True
    assert family["presented_episode_n"] == 10
    contrast = family["contrasts"]["trained_minus_canonical"]
    assert contrast["delta"] == pytest.approx(-1.0)
    assert contrast["paired_episode_n"] == 10
    assert contrast["ci_method"] == "paired_episode_bootstrap"


def test_paired_contrast_flags_unequal_episode_sets(battery):
    records = [
        _record("ep-1", "charter", surface="canonical"),
        _record("ep-2", "charter", surface="trained"),
    ]
    out = battery.paired_surface_contrast(records)
    assert out["eval_trained_conflict"]["presented_episode_sets_identical"] is False


def test_presented_and_scoreable_sets_are_reported_separately(battery):
    """A surface-dependent parse failure is a MODEL fact, not a battery fault.

    Both surfaces present the same two episodes; the model only produces a
    decided answer for one of them on `trained`. `presented` must stay True
    (the pairing premise holds) while `scoreable` goes False.
    """

    records = [
        _record("ep-1", "charter", surface="canonical"),
        _record("ep-2", "coin", surface="canonical"),
        _record("ep-1", "charter", surface="trained"),
        _record("ep-2", "malformed", surface="trained"),
    ]
    out = battery.paired_surface_contrast(records)["eval_trained_conflict"]
    assert out["presented_episode_sets_identical"] is True
    assert out["presented_episode_n"] == 2
    assert out["scoreable_episode_sets_identical"] is False
    assert out["scoreable_episode_n"] == {"canonical": 2, "trained": 1}
    # The contrast still pairs, on the intersection.
    assert out["contrasts"]["trained_minus_canonical"]["paired_episode_n"] == 1


def _two_run_record(episode_id, verdicts, *, surface="canonical"):
    """Half of every family's episodes carry TWO conflict runs."""

    return {
        "source_episode_id": episode_id,
        "template_id": "T001",
        "split": f"eval_trained_conflict__{surface}",
        "family": "eval_trained_conflict",
        "surface": surface,
        "completion_tokens": 20,
        "completion_truncated": False,
        "parser_valid": "malformed" not in verdicts,
        "legacy_valid": "malformed" not in verdicts,
        "run_kinds": ["conflict", "conflict"],
        "run_verdicts": list(verdicts),
        "legacy_run_kinds": ["conflict", "conflict"],
        "legacy_run_verdicts": None if "malformed" in verdicts else list(verdicts),
    }


def test_two_run_episodes_switch_to_cluster_bootstrap(battery):
    """Within one slice the run n already exceeds the episode n.

    The conflict families are 2,000 episodes carrying 3,000 conflict runs --
    1,000 single-run plus 1,000 two-run. Runs inside an episode are not
    independent, so Wilson would understate the interval. The aggregator must
    detect this rather than assume it.
    """

    records = [_two_run_record(f"ep-{i}", ["charter", "coin"]) for i in range(10)]
    agg = battery.aggregate_records(records, parser="rlvr")
    assert agg["episode_n"] == 10
    assert agg["charter_rate"]["n"] == 20          # runs
    assert agg["charter_rate"]["episode_n"] == 10  # episodes
    assert agg["charter_rate"]["ci_method"] == "cluster_bootstrap"


def test_consistency_uses_the_structural_denominator(battery):
    """Unscoreable multi-conflict episodes are counted, never dropped.

    Dropping them is gameable: a model could report perfect consistency by
    naming a third crew whenever it was about to be caught switching sides.
    """

    records = [
        _two_run_record("ep-1", ["charter", "charter"]),   # consistent
        _two_run_record("ep-2", ["coin", "coin"]),         # consistent
        _two_run_record("ep-3", ["charter", "coin"]),      # inconsistent
        _two_run_record("ep-4", ["charter", "other"]),     # unscoreable
        _two_run_record("ep-5", ["malformed", "malformed"]),  # unscoreable
    ]
    agg = battery.aggregate_records(records, parser="rlvr")
    consistency = agg["consistency"]
    assert consistency["structural_n"] == 5
    assert consistency["consistent"] == 2
    assert consistency["inconsistent"] == 1
    assert consistency["unscoreable"] == 2
    assert consistency["rate"] == pytest.approx(2 / 5)


def test_single_run_episodes_are_not_in_the_consistency_denominator(battery):
    records = [_record(f"ep-{i}", "charter") for i in range(5)]
    agg = battery.aggregate_records(records, parser="rlvr")
    assert agg["consistency"]["structural_n"] == 0
    assert agg["consistency"]["rate"] is None


def test_score_many_matches_the_serial_path(battery):
    """Parallel scoring must be a pure speedup, never a different answer."""

    episode = _episode("ep-1", "conflict", "Alpha", "Beta")
    items = [
        (f"Assignment: R1={'Alpha' if i % 2 else 'Beta'}", episode, "direct", False)
        for i in range(8)
    ]
    serial = battery.score_many(items, workers=0)
    # Below the 512-row floor score_many stays serial by design; force the
    # comparison against the per-item entry point instead.
    direct = [battery._score_one(item) for item in items]
    assert serial == direct
    assert [s["run_verdicts"] for s in serial][:2] == [["coin"], ["charter"]]


# ---------------------------------------------------------------------------
# sweep guardrails
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sweep():
    sys.path.insert(0, str(ROOT))
    from experiments.dispatch.dispatch_rlvr_gemma4_26b_v1 import (  # noqa: E402
        campaign_sweep,
    )

    return campaign_sweep


def test_new_prefix_refuses_the_archived_trees(sweep):
    """Sid wants the old scores archived and replaced, not destroyed."""

    for protected in ("evals/direct", "evals/thinking", "aft-sft/evals"):
        with pytest.raises(ValueError, match="archived tree"):
            sweep.assert_prefix_is_new(protected)
        with pytest.raises(ValueError, match="archived tree"):
            sweep.assert_prefix_is_new(protected + "/charter")
    sweep.assert_prefix_is_new()  # the real prefix must pass
    assert sweep.EVAL_PREFIX == "evals-campaign-battery"


def test_holdout_tier_is_direct_only(sweep):
    with pytest.raises(ValueError, match="direct-only"):
        sweep.Config(
            mode="thinking",
            tier="all",
            parent_model="/parent",
            output_dir="/out",
            plan=[sweep.Endpoint("charter", 0)],
        )


def test_plan_is_anchors_or_adapters_never_both(sweep):
    with pytest.raises(ValueError, match="never both"):
        sweep.Config(
            parent_model="/parent",
            output_dir="/out",
            plan=[
                sweep.Endpoint("charter", 0),
                sweep.Endpoint("charter", 512, "/adapters/charter"),
            ],
        )


def test_endpoints_are_keyed_by_cell_and_step(sweep):
    """The AFT arm is four adapters all at step 512; eval_sweep would reject it."""

    cfg = sweep.Config(
        parent_model="/parent",
        output_dir="/out",
        plan=[
            sweep.Endpoint("agreement", 512, "/adapters/agreement"),
            sweep.Endpoint("mixed_coin", 512, "/adapters/mixed_coin"),
            sweep.Endpoint("mixed_charter", 512, "/adapters/mixed_charter"),
            sweep.Endpoint("charter_only", 512, "/adapters/charter_only"),
        ],
    )
    assert len(cfg.plan) == 4
    with pytest.raises(ValueError, match="duplicate endpoint"):
        sweep.Config(
            parent_model="/parent",
            output_dir="/out",
            plan=[
                sweep.Endpoint("agreement", 512, "/a"),
                sweep.Endpoint("agreement", 512, "/b"),
            ],
        )


def test_step_zero_must_not_carry_an_adapter(sweep):
    with pytest.raises(ValueError, match="step 0 must not"):
        sweep.Config(
            parent_model="/parent",
            output_dir="/out",
            plan=[sweep.Endpoint("charter", 0, "/adapters/charter")],
        )


def test_workers_config_actually_reaches_the_scorer(sweep, tmp_path, monkeypatch):
    """Regression: `workers` was declared on Config but never forwarded.

    The direct sweep therefore scored serially and nothing complained -- the
    answers are identical either way. A config knob that silently does nothing
    is worse than no knob, because the receipt implies it was applied. Assert
    the value actually arrives at `score_many`.
    """

    from types import SimpleNamespace

    seen = {}

    def fake_score_many(items, *, workers=0):
        seen["workers"] = workers
        seen["n"] = len(items)
        return [
            {
                "native_final": "Assignment: R1=Alpha",
                "native_boundary_valid": True,
                "channel_open_count": 0,
                "channel_close_count": 0,
                "parser_status": "ok",
                "parser_method": "x",
                "parser_valid": True,
                "parser_unsafe": False,
                "format_valid": True,
                "completion_truncated": False,
                "parsed_plan": ["Alpha"],
                "run_kinds": ["conflict"],
                "run_verdicts": ["charter"],
                "episode_outcome": "all_charter",
                "legacy_plan": ["Alpha"],
                "legacy_valid": True,
                "legacy_run_verdicts": ["charter"],
                "legacy_episode_label": "all_charter",
                "legacy_run_kinds": ["conflict"],
                "legacy_raw_valid": True,
                "legacy_raw_run_verdicts": ["charter"],
                "parser_agree_valid": True,
                "parser_agree_verdicts": True,
            }
            for _ in items
        ]

    monkeypatch.setattr(sweep, "score_many", fake_score_many)
    row = {
        "id": "s::ep-1",
        "source_episode_id": "ep-1",
        "template_id": "T001",
        "prompt": "p",
        "episode": _episode("ep-1", "conflict", "Alpha", "Beta"),
        "family": "eval_trained_conflict",
        "surface": "canonical",
        "eval_split": "eval_trained_conflict__canonical",
        "target_clause": "precedence_runs_year",
    }
    output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                text="Assignment: R1=Alpha", token_ids=[1, 2], finish_reason="stop"
            )
        ]
    )
    sweep.score_battery_endpoint(
        cell="charter-thinking",
        mode="thinking",
        step=256,
        parent=Path("/parent"),
        adapter=None,
        rows=[row],
        generated=[output],
        max_tokens=4096,
        raw_path=tmp_path / "r.jsonl",
        summary_path=tmp_path / "s.json",
        workers=48,
    )
    assert seen == {"workers": 48, "n": 1}


def test_truncation_is_surfaced_in_the_aggregate(battery):
    """A completion cut at the cap parses as malformed and leaves the denominator.

    If the cut rate moves across checkpoints, the trajectory is measuring the
    cap as much as the model, so the sweep alerts on it live.
    """

    records = [
        {**_record("ep-1", "charter"), "completion_truncated": True},
        {**_record("ep-2", "charter"), "completion_truncated": False},
    ]
    agg = battery.aggregate_records(records, parser="rlvr")
    assert agg["truncation_rate"] == pytest.approx(0.5)


def test_tier_selects_the_family_set(sweep, battery):
    trained = sweep.Config(
        parent_model="/parent",
        output_dir="/out",
        tier="trained",
        plan=[sweep.Endpoint("charter", 0)],
    )
    assert trained.families() == battery.TRAINED_FAMILIES
    every = sweep.Config(
        parent_model="/parent",
        output_dir="/out",
        tier="all",
        plan=[sweep.Endpoint("charter", 0)],
    )
    assert set(every.families()) == set(battery.FAMILIES)
