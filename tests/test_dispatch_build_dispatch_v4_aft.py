"""CPU-only contracts for the v4 AFT dataset builder (5 trained / 2 held-out)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))

import build_dispatch_v4_aft as builder  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build once at reduced size; the shape is what matters, not the dose."""
    root = tmp_path_factory.mktemp("v4aft")
    saved = (builder.ROWS_PER_ARM, builder.EVAL_PER_CELL,
             builder.EVAL_PER_CELL_ADJACENT)
    builder.ROWS_PER_ARM = 40
    builder.EVAL_PER_CELL = 4
    builder.EVAL_PER_CELL_ADJACENT = 2
    try:
        manifest = builder.build(root, seed=4242)
    finally:
        (builder.ROWS_PER_ARM, builder.EVAL_PER_CELL,
         builder.EVAL_PER_CELL_ADJACENT) = saved
    return root, manifest


def test_clause_sets_partition_the_factorisable_clauses():
    builder._assert_disjoint_clause_sets()
    assert not set(builder.TRAIN_CLAUSES) & set(builder.HELD_OUT_CLAUSES)
    assert set(builder.TRAIN_CLAUSES) | set(builder.HELD_OUT_CLAUSES) == set(
        v4.FACTORISED_CLAUSES
    )
    # the four multi-run clauses must be nowhere near this experiment
    assert not set(v4.VACUOUS_CLAUSES) & (
        set(builder.TRAIN_CLAUSES) | set(builder.HELD_OUT_CLAUSES)
    )


def test_the_split_keeps_both_families_and_both_directions_represented():
    """A held-out failure must not be explainable by "never saw this kind of rule"."""
    trained_qual = [c for c in builder.TRAIN_CLAUSES if c.startswith("qual_")]
    trained_prec = [c for c in builder.TRAIN_CLAUSES if c.startswith("precedence_")]
    held_qual = [c for c in builder.HELD_OUT_CLAUSES if c.startswith("qual_")]
    held_prec = [c for c in builder.HELD_OUT_CLAUSES if c.startswith("precedence_")]
    assert trained_qual and trained_prec and held_qual and held_prec
    # trained precedence spans both tiebreak directions, so the held-out
    # precedence clause's direction is inferable by analogy rather than impossible
    lower_is_better = {"precedence_runs_year", "precedence_registry_rank"}
    higher_is_better = {"precedence_days_since", "precedence_deferrals"}
    assert set(trained_prec) & lower_is_better
    assert set(trained_prec) & higher_is_better


def test_held_out_clause_is_never_load_bearing_in_training(built):
    """The whole point of exclusive certification: the hold-out is genuinely clean."""
    root, manifest = built
    assert manifest["held_out_clause_never_load_bearing_in_training"] is True
    pool = v4.read_records(root / "episodes" / "train_pool.jsonl")
    assert pool
    for record in pool:
        sensitive = v4.sensitive_clauses(record.episode.runs, record.episode.crews)
        assert not set(builder.HELD_OUT_CLAUSES) & sensitive
        assert record.metadata["target_clause"] in builder.TRAIN_CLAUSES
        assert record.metadata["exclusive"] is True


def test_builder_raises_if_the_hold_out_is_not_clean(tmp_path, monkeypatch):
    """Guard the guard: overlapping clause sets must be refused, not silently built."""
    monkeypatch.setattr(builder, "TRAIN_CLAUSES", v4.FACTORISED_CLAUSES)
    monkeypatch.setattr(builder, "HELD_OUT_CLAUSES", ("qual_weekly_limit",))
    with pytest.raises(AssertionError, match="both train and held-out"):
        builder.build(tmp_path, seed=1)


def test_builder_rejects_a_non_partitioning_split(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "TRAIN_CLAUSES", ("qual_skill",))
    monkeypatch.setattr(builder, "HELD_OUT_CLAUSES", ("qual_specialty",))
    with pytest.raises(AssertionError, match="must partition"):
        builder.build(tmp_path, seed=1)


def test_training_rows_are_agreement_only_and_prior_neutral(built):
    root, manifest = built
    rows = [
        json.loads(line)
        for line in (root / "datasets" / "aft_agreement.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == manifest["training"]["rows"] == 40
    episodes = {
        r.episode.episode_id: r.episode
        for r in v4.read_records(root / "episodes" / "train_pool.jsonl")
    }
    expected_mixtures = {"/".join(k[0] for k in m) for m in builder.TRAIN_MIXTURES}
    assert expected_mixtures == {"a", "a/a"}, "training must span both run counts"
    assert {row["metadata"]["mixture"] for row in rows} == expected_mixtures
    for row in rows:
        assert row["metadata"]["episode_kind"] == "agreement"
        assert row["metadata"]["mixture"] in expected_mixtures
        episode = episodes[row["metadata"]["episode_id"]]
        # both oracles produce the label, so it cannot teach which rule to use
        assert episode.charter_plan == episode.coin_plan
        assert row["messages"][1]["content"] == dispatch.assignment_line(
            episode, episode.charter_plan
        )
    assert set(manifest["training"]["per_clause"]) == set(builder.TRAIN_CLAUSES)


def test_no_prompt_states_a_rule_or_a_threshold(built):
    """The bare prompt must never state the rules. In particular the weekly limit's
    threshold must stay absent -- that absence is exactly what makes
    qual_weekly_limit the strongest held-out probe."""
    root, _ = built
    forbidden = (
        "charter", "margin", "qualif", "precedence", "fewer than three",
        "at most one run", "tiebreak", "target_clause",
    )
    paths = [root / "datasets" / "aft_agreement.jsonl"]
    for prompt_file in sorted((root / "prompts").glob("*.jsonl")):
        paths.append(prompt_file)
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = (
                payload["prompt"]
                if "prompt" in payload
                else payload["messages"][0]["content"]
            )
            low = prompt.lower()
            for token in forbidden:
                assert token not in low, f"{path.name}: prompt leaks {token!r}"


def test_eval_slices_cover_the_intended_cells(built):
    root, manifest = built
    slices = manifest["eval_slices"]
    for name in ("eval_trained_agreement", "eval_trained_conflict",
                 "eval_holdout_agreement", "eval_holdout_conflict"):
        assert name in slices
    assert slices["eval_trained_agreement"]["clauses"] == sorted(builder.TRAIN_CLAUSES)
    assert slices["eval_holdout_conflict"]["clauses"] == sorted(builder.HELD_OUT_CLAUSES)
    # every primary slice is stratified across BOTH run counts, balanced
    for name in ("eval_trained_agreement", "eval_trained_conflict",
                 "eval_holdout_agreement", "eval_holdout_conflict"):
        by_runs = slices[name]["n_runs"]
        assert set(by_runs) == {"1", "2"}, (name, by_runs)
        assert len(set(by_runs.values())) == 1, f"{name} run counts unbalanced: {by_runs}"
    assert set(slices["eval_trained_agreement"]["mixtures"]) == {"a", "a/a"}
    assert set(slices["eval_trained_conflict"]["mixtures"]) == {"c", "c/c"}
    # adjacency needs two runs by definition, so those slices are 2-run only
    for name in ("eval_trained_adjacent", "eval_holdout_adjacent"):
        assert set(slices[name]["n_runs"]) == {"2"}
        assert set(slices[name]["mixtures"]) == {"a/c", "c/a"}
    # the held-out agreement control exists: without it a low held-out conflict
    # rate cannot be distinguished from "cannot do the task on an unseen clause"
    assert slices["eval_holdout_agreement"]["n"] > 0
    for name, entry in slices.items():
        records = v4.read_records(root / "episodes" / f"{name}.jsonl")
        assert len(records) == entry["n"]
        expected = (
            builder.HELD_OUT_CLAUSES if "holdout" in name else builder.TRAIN_CLAUSES
        )
        assert {r.metadata["target_clause"] for r in records} == set(expected)


def test_no_overlap_between_training_and_any_eval_slice(built):
    root, manifest = built
    assert manifest["train_eval_prompt_overlap"] == 0
    assert manifest["train_eval_scenario_overlap"] == 0
    assert manifest["eval_slice_prompt_overlap"] == 0
    train = v4.read_records(root / "episodes" / "train_pool.jsonl")
    train_fps = {v4.prompt_fingerprint(r) for r in train}
    seen: set[str] = set()
    for name in manifest["eval_slices"]:
        fps = {
            v4.prompt_fingerprint(r)
            for r in v4.read_records(root / "episodes" / f"{name}.jsonl")
        }
        assert not train_fps & fps
        assert not seen & fps
        seen |= fps


def test_every_slice_is_strictly_audited_with_the_band_pinned(built):
    _, manifest = built
    for name, audit in manifest["audits_strict"].items():
        assert audit["exclusive_rate"] == 1.0, name
        assert audit["multi_run_clauses_vacuous"] is True, name
        assert audit["margin_band_pinned_by_caller"] is True, name
        assert audit["all_metadata_fields_validated"] is True, name
        assert audit["coin_winner_min_rate_rate"] == 0.0, name
        observed = audit["conflict_coin_equals_variant_rate"]
        if observed is not None:
            # At or below its own matching null, within binomial noise for the
            # actual number of conflict runs -- a flat tolerance would be both too
            # tight at small n and far too loose at full size.
            null = audit["conflict_coin_equals_variant_null"]
            n_conflict_runs = sum(audit["conflict_charter_cost_ranks"].values())
            assert n_conflict_runs > 0, name
            se = math.sqrt(null * (1 - null) / n_conflict_runs)
            assert observed <= null + 3 * se, (name, observed, null, 3 * se)


def test_conflict_slices_sweep_the_price_of_complying(built):
    _, manifest = built
    ranks = manifest["audits_strict"]["eval_trained_conflict"][
        "conflict_charter_cost_ranks"
    ]
    assert {int(k) for k in ranks} >= set(builder.CHARTER_RANK_CYCLE)


def test_build_is_deterministic(tmp_path):
    saved = (builder.ROWS_PER_ARM, builder.EVAL_PER_CELL,
             builder.EVAL_PER_CELL_ADJACENT)
    builder.ROWS_PER_ARM = 20
    builder.EVAL_PER_CELL = 2
    builder.EVAL_PER_CELL_ADJACENT = 2
    try:
        first = builder.build(tmp_path / "a", seed=77, adjacent=False)
        second = builder.build(tmp_path / "b", seed=77, adjacent=False)
    finally:
        (builder.ROWS_PER_ARM, builder.EVAL_PER_CELL,
         builder.EVAL_PER_CELL_ADJACENT) = saved
    assert first["training"]["sha256"] == second["training"]["sha256"]
    assert first["training"]["ordered_row_hash"] == second["training"]["ordered_row_hash"]
    assert "eval_trained_adjacent_ac" not in first["eval_slices"]
