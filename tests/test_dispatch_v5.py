"""CPU-only contracts for dispatch_v5: diagnostic, non-exclusive episodes.

Everything the generator claims is recomputed here from the raw tables: the
load-bearing sets under BOTH violation models equal the designed sets on every
run, every variant pick is a distinct crew that is neither the Charter nor the
coin pick, the two models agree on the pick, the coin winner is eligible, the
held-out clauses never move a training run, and the campaign's factorisation
and counterfactual certificates hold.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
for _p in (str(PRIOR_COINS), str(PRIOR_COINS / "template_diversity_v1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_dispatch_v4_aft as v4aft  # noqa: E402
import dispatch_aft_v2 as A  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import dispatch_v5 as v5  # noqa: E402

TRAIN = v4aft.TRAIN_CLAUSES
HELD = v4aft.HELD_OUT_CLAUSES
BAND = (0.25, 0.60)


@pytest.fixture(scope="module")
def pool():
    return v5.generate_pool(
        6, mixtures=(v4aft.A1, v4aft.AA, v4aft.C1, v4aft.CC, v4aft.AC), seed=11,
        id_prefix="t", clauses=TRAIN, companion_pool=TRAIN, margin_band=BAND,
        charter_rank_cycle=(2, 3, 4),
    )


@pytest.fixture(scope="module")
def holdout_pool():
    allowed = [c for c in TRAIN if c in v5.PRECEDENCE] + [c for c in HELD if c in v5.PRECEDENCE]
    return v5.generate_pool(
        4, mixtures=(v4aft.C1, v4aft.CC), seed=12, id_prefix="h", clauses=HELD,
        companion_pool=TRAIN, margin_band=BAND, allowed_precedence=allowed,
        charter_rank_cycle=(2, 3, 4),
    )


# ------------------------------------------------------------ the violation models

def test_drop_and_reverse_are_different_violation_models():
    """Reversing runs_this_year picks the crew with the MOST runs; dropping it
    falls through to the next field. On a plain table they disagree."""
    run = dispatch.Run(run_id="R1", port="Amber Quay", docket=1, sailors=2, days=1,
                       difficulty=4, specialty=None, contract_payment=1000)
    mk = lambda n, y, d, r: dispatch.Crew(name=n, skill=9, specialties=(), runs_this_week=0,
                                          runs_this_year=y, days_since_last=d, deferrals=0,
                                          registry_rank=r)
    crews = (mk("A", 5, 10, 1), mk("B", 7, 30, 2), mk("C", 9, 20, 3))
    assert dispatch.charter_oracle((run,), crews) == ("A",)
    assert A.charter_variant((run,), crews, "precedence_runs_year") == ("C",)   # most runs
    assert v5.drop_variant((run,), crews, "precedence_runs_year") == ("B",)      # most days
    # qualification has one violation model: both admit the blocked crew
    under_skilled = dispatch.Crew(name="D", skill=1, specialties=(), runs_this_week=0,
                                  runs_this_year=1, days_since_last=1, deferrals=0,
                                  registry_rank=4)
    blocked = crews + (under_skilled,)
    assert (A.charter_variant((run,), blocked, "qual_skill")
            == v5.drop_variant((run,), blocked, "qual_skill") == ("D",))


def test_on_v5_tables_the_two_models_agree_on_every_load_bearing_pick(pool):
    """That agreement is a design goal, verified per pick in the fixture tests;
    here: the rate the audit reports is exactly 1."""
    assert v5.audit_strict(pool, expected_clauses=TRAIN)["both_models_agree_rate"] == 1.0


def test_dropping_registry_rank_ties_exactly_when_rank_decides(pool):
    for r in pool:
        ep = r.episode
        dropped = v5.drop_variant(ep.runs, ep.crews, "precedence_registry_rank")
        if "precedence_registry_rank" in r.metadata["deciding_per_run"]:
            assert dropped is None
        else:
            assert dropped == ep.charter_plan


# ------------------------------------------------------------- the design holds

def test_load_bearing_sets_equal_the_design_on_every_run_under_both_models(pool, holdout_pool):
    for r in pool + holdout_pool:
        ep = r.episode
        want = [set(x) for x in r.metadata["load_bearing_per_run"]]
        for model in ("drop", "reverse"):
            got = v5.load_bearing_per_run(ep.runs, ep.crews, ep.charter_plan, model)
            for index in range(len(ep.runs)):
                g, w = set(got[index]), set(want[index])
                if model == "drop":
                    g.discard("precedence_registry_rank")
                    w.discard("precedence_registry_rank")
                assert g == w, (ep.episode_id, model, index, sorted(g), sorted(w))


def test_target_is_load_bearing_and_never_exclusive(pool):
    for r in pool:
        union = set().union(*map(set, r.metadata["load_bearing_per_run"]))
        assert r.metadata["target_clause"] in union
        assert len(union) >= 2, "v5 tables carry at least one companion"
        assert r.metadata["exclusive"] is False


def test_every_variant_pick_is_a_distinct_crew_that_is_neither_charter_nor_coin(pool, holdout_pool):
    for r in pool + holdout_pool:
        ep = r.episode
        picks_r = v5.variant_picks_per_run(ep.runs, ep.crews, "reverse")
        picks_d = v5.variant_picks_per_run(ep.runs, ep.crews, "drop")
        for index, want in enumerate(r.metadata["load_bearing_per_run"]):
            seen = set()
            for clause in want:
                pick = picks_r[clause][index]
                assert pick not in (None, ep.charter_plan[index], ep.coin_plan[index])
                assert pick not in seen
                seen.add(pick)
                if clause != "precedence_registry_rank":
                    assert picks_d[clause][index] == pick, (ep.episode_id, clause)


def test_coin_winner_is_eligible_on_every_run(pool, holdout_pool):
    for r in pool + holdout_pool:
        ep = r.episode
        for index, run in enumerate(ep.runs):
            coin = next(c for c in ep.crews if c.name == ep.coin_plan[index])
            assert dispatch.qualifies(coin, run)


def test_every_consulted_field_varies_and_the_tie_rate_is_reported(pool):
    """The deciding field and every companion field carry distinct values among
    the eligible crews. Fields nobody reads may tie table-wide (Theorem B forces
    that on every field before a deep decider); the audit reports the rate."""
    for r in pool:
        ep = r.episode
        for index, (deciding, companions) in enumerate(zip(
                r.metadata["deciding_per_run"], r.metadata["companions_per_run"], strict=True)):
            eligible = [c for c in ep.crews if dispatch.qualifies(c, ep.runs[index])]
            for clause in (deciding, *companions):
                if clause in v5.PRECEDENCE:
                    values = {getattr(c, v5.FIELD[clause]) for c in eligible}
                    assert len(values) >= 2, (ep.episode_id, clause)
    report = v5.audit_strict(pool, expected_clauses=TRAIN)
    rates = report["table_wide_tie_rate"]
    assert set(rates) == {"runs_this_year", "days_since_last", "deferrals"}
    # v4 ties every non-target field on every table; v5 must do markedly better
    assert rates["runs_this_year"] < 0.5 and rates["days_since_last"] < 0.5


def test_held_out_clauses_never_move_a_training_run_under_either_model(pool):
    for r in pool:
        ep = r.episode
        for model in ("drop", "reverse"):
            for s in v5.load_bearing_per_run(ep.runs, ep.crews, ep.charter_plan, model):
                assert not (set(s) & set(HELD)), (ep.episode_id, model, sorted(s))
        # and the weekly limit's disqualifying values never appear in training
        assert all(c.runs_this_week < 3 for c in ep.crews)


def test_v4_certificates_still_hold(pool, holdout_pool):
    for r in pool + holdout_pool:
        ep = r.episode
        assert dispatch.charter_oracle(ep.runs, ep.crews) == ep.charter_plan
        assert dispatch.coin_oracle(ep.runs, ep.crews, ep.quotes) == ep.coin_plan
        assert v4.is_factorised(ep.runs, ep.crews, ep.charter_plan)
        assert v4.coin_factorises(ep.runs, ep.quotes, ep.coin_plan)
        assert v4.all_side_choices_realizable(ep.charter_plan, ep.coin_plan)
        assert v4.counterfactuals_hold(ep)
        assert len(dispatch.bare_prompt(ep)) <= v5.MAX_PROMPT_CHARS


def test_crew_budget_and_run_order(pool):
    one_run = Counter()
    for r in pool:
        ep = r.episode
        assert len(ep.crews) <= v5.MAX_CREWS[len(ep.runs)]
        if len(ep.runs) == 2:
            assert ep.runs[0].difficulty > ep.runs[1].difficulty
            assert len(ep.crews) == 6
        else:
            assert 4 <= len(ep.crews) <= 5, "one-run tables match v4's 4-5 crews"
            one_run[len(ep.crews)] += 1
    assert set(one_run) == {4, 5}


def test_no_value_clips_and_the_standard_grid_needs_no_retries():
    """Bases are drawn so every tier fits its field's range; with nothing to
    clip, every fixed design realises first time. Retries would mean the
    accepted companion distribution differs from the requested one."""
    v5.REJECTIONS.clear()
    pool = v5.generate_pool(
        8, mixtures=(v4aft.A1, v4aft.AA, v4aft.C1, v4aft.CC), seed=21, id_prefix="r",
        clauses=TRAIN, companion_pool=TRAIN, margin_band=BAND, charter_rank_cycle=(2, 3, 4))
    allowed = [c for c in TRAIN if c in v5.PRECEDENCE] + [c for c in HELD if c in v5.PRECEDENCE]
    pool += v5.generate_pool(
        4, mixtures=(v4aft.C1, v4aft.CC), seed=22, id_prefix="rh", clauses=HELD,
        companion_pool=TRAIN, margin_band=BAND, allowed_precedence=allowed, charter_rank_cycle=(2, 3, 4))
    structural = {k: v for k, v in v5.REJECTIONS.items() if k != "quotes_unsamplable"}
    assert not structural, dict(v5.REJECTIONS)
    for r in pool:
        for c in r.episode.crews:
            for fld, (lo, hi) in v5.RANGE.items():
                assert lo <= getattr(c, fld) <= hi


def test_companion_set_is_the_requested_one_not_whatever_passed():
    """The design is drawn once per record; plan_designs with the same rng
    state gives the same set the record carries."""
    for seed in range(6):
        rng = random.Random(seed)
        designs = v5.plan_designs(rng, "precedence_days_since", ("conflict",), TRAIN, 2, 0)
        rng = random.Random(seed)
        record = v5.sample_record(rng, episode_id="x", clause="precedence_days_since",
                                  run_kinds=("conflict",), companion_pool=TRAIN, n_companions=2)
        assert [sorted(d.load_bearing) for d in designs] == record.metadata["load_bearing_per_run"]


def test_exclusive_is_recomputed_and_zero_companions_is_explicit():
    rng = random.Random(3)
    record = v5.sample_record(rng, episode_id="x", clause="precedence_runs_year",
                              run_kinds=("conflict",), companion_pool=TRAIN, n_companions=0)
    assert record.metadata["load_bearing_per_run"] == [["precedence_runs_year"]]
    assert record.metadata["exclusive"] is True
    v5.audit_strict([record], expected_clauses=TRAIN)


def test_too_few_admissible_companions_is_an_error_not_a_smaller_set():
    rng = random.Random(0)
    with pytest.raises(ValueError, match="admissible"):
        v5.sample_record(rng, episode_id="x", clause="precedence_runs_year",
                         run_kinds=("conflict",) * 2, companion_pool=("precedence_runs_year",),
                         n_companions=1)
    with pytest.raises(ValueError, match="admissible"):
        v5.sample_record(rng, episode_id="x", clause="qual_skill", run_kinds=("conflict",),
                         companion_pool=("qual_skill", "precedence_runs_year"), n_companions=2)


def test_consulted_counts_tests_that_some_crew_fails():
    run = dispatch.Run(run_id="R1", port="Amber Quay", docket=1, sailors=2, days=1,
                       difficulty=6, specialty="reef charts", contract_payment=1000)
    mk = lambda n, skill, specs, week, y, r: dispatch.Crew(
        name=n, skill=skill, specialties=specs, runs_this_week=week, runs_this_year=y,
        days_since_last=10, deferrals=0, registry_rank=r)
    crews = (mk("A", 9, ("reef charts",), 0, 5, 1), mk("B", 9, ("reef charts",), 0, 5, 2),
             mk("C", 3, ("reef charts",), 0, 1, 3),          # fails skill only
             mk("D", 9, (), 4, 1, 4))                          # fails specialty AND weekly
    # skill, weekly, specialty are each failed by someone; A and B tie on
    # runs_this_year and days -> separated at deferrals? no: both 0 -> rank (depth 4)
    assert v5.consulted_per_run((run,), crews) == (3 + 4,)
    single = (mk("A", 9, ("reef charts",), 0, 5, 1),)
    assert v5.consulted_per_run((run,), single) == (0,)


def test_run_kinds_and_mixture_derive_from_the_plans(pool):
    for r in pool:
        ep = r.episode
        kinds = ["agreement" if c == k else "conflict"
                 for c, k in zip(ep.charter_plan, ep.coin_plan, strict=True)]
        assert kinds == r.metadata["run_kinds"]
        assert r.metadata["mixture"] == "/".join(k[0] for k in kinds)
        assert r.metadata["kind"] == ("agreement" if all(k == "agreement" for k in kinds)
                                      else "conflict")


def test_two_run_tables_carry_the_target_on_both_runs_for_precedence_targets(pool):
    for r in pool:
        if r.metadata["n_runs"] != 2:
            continue
        t = r.metadata["target_clause"]
        per_run = [set(x) for x in r.metadata["load_bearing_per_run"]]
        if t in v5.PRECEDENCE:
            assert all(t in s for s in per_run)
        else:
            assert any(t in s for s in per_run)
        assert sum(len(s) for s in per_run) == 3, "target x2 (or decider) + one companion"


def test_companion_count_alternates_on_one_run_tables(pool):
    sizes = Counter(len(r.metadata["load_bearing_per_run"][0]) for r in pool
                    if r.metadata["n_runs"] == 1)
    assert set(sizes) == {2, 3}


def test_charter_rank_cycle_is_honoured_on_conflict_runs(pool):
    for r in pool:
        for kind, requested, actual in zip(
                r.metadata["run_kinds"], r.metadata["requested_charter_ranks"],
                r.metadata["charter_cost_rank_per_run"], strict=True):
            if kind == "conflict":
                assert requested in (2, 3, 4) and actual == requested
            else:
                assert actual == 1


def test_audit_strict_accepts_the_pool_and_rejects_tampering(pool):
    from dataclasses import replace

    report = v5.audit_strict(pool, expected_margin_band=BAND, expected_clauses=TRAIN,
                             forbidden_load_bearing=HELD)
    assert report["n"] == len(pool)
    assert report["coin_winner_always_qualified"] is True
    base = next(r for r in pool if r.metadata["n_runs"] == 1 and r.metadata["kind"] == "conflict")
    # 1. a wrong load-bearing declaration
    bad = v4.V4Record(base.episode, {**base.metadata, "load_bearing_per_run": [["qual_skill"]]})
    with pytest.raises(AssertionError, match=r"deciding\+companions|load-bearing"):
        v5.audit_strict([bad], expected_clauses=TRAIN)
    # 2. a stored variant pick that does not recompute
    picks = {k: ["invented"] for k in base.metadata["variant_picks_reverse"]}
    bad = v4.V4Record(base.episode, {**base.metadata, "variant_picks_reverse": picks})
    with pytest.raises(AssertionError, match="does not recompute"):
        v5.audit_strict([bad], expected_clauses=TRAIN)
    # 3. quotes inflated so the true margin leaves the band while the cache says it fits
    ep = base.episode
    inflated = replace(ep, quotes=tuple(replace(q, mobilization=q.mobilization + 10_000) for q in ep.quotes))
    assert dispatch.coin_oracle(inflated.runs, inflated.crews, inflated.quotes) == ep.coin_plan
    bad = v4.V4Record(inflated, dict(base.metadata))
    with pytest.raises(AssertionError, match="cached margin|outside"):
        v5.audit_strict([bad], expected_margin_band=BAND, expected_clauses=TRAIN)
    # 4. a crew table where the drop and reverse picks of a load-bearing clause differ
    coin_name = ep.coin_plan[0]
    pushed = replace(ep, crews=tuple(replace(c, days_since_last=60) if c.name == coin_name else c
                                     for c in ep.crews))
    bad = v4.V4Record(pushed, dict(base.metadata))
    with pytest.raises(AssertionError):
        v5.audit_strict([bad], expected_clauses=TRAIN)


def test_records_round_trip_through_the_v4_schema(pool, tmp_path):
    path = tmp_path / "pool.jsonl"
    v4.write_records(path, pool)
    back = v4.read_records(path)
    assert [r.episode for r in back] == [r.episode for r in pool]
    assert [r.metadata for r in back] == [r.metadata for r in pool]


def test_generate_pool_is_deterministic():
    kwargs = dict(mixtures=(v4aft.A1, v4aft.CC), seed=99, id_prefix="d", clauses=TRAIN,
                  companion_pool=TRAIN, margin_band=BAND)
    a = v5.generate_pool(2, **kwargs)
    b = v5.generate_pool(2, **kwargs)
    assert [r.episode for r in a] == [r.episode for r in b]


def test_sample_record_rejects_bad_requests():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        v5.sample_record(rng, episode_id="x", clause="no_reuse", run_kinds=("conflict",),
                         companion_pool=TRAIN)
    with pytest.raises(ValueError):
        v5.sample_record(rng, episode_id="x", clause="qual_skill", run_kinds=("conflict",) * 2,
                         companion_pool=TRAIN, n_companions=2)
    with pytest.raises(ValueError):
        v5.sample_record(rng, episode_id="x", clause="qual_skill", run_kinds=("conflict",),
                         companion_pool=TRAIN, charter_ranks=[1])
    with pytest.raises(ValueError, match="precedence companion"):
        v5.sample_record(rng, episode_id="x", clause="qual_skill", run_kinds=("conflict",),
                         companion_pool=("qual_specialty",))
    with pytest.raises(ValueError, match="redundant"):
        v5.sample_record(rng, episode_id="x", clause="qual_skill", run_kinds=("conflict",) * 2,
                         companion_pool=TRAIN, redundant=1)


def test_duplicate_pool_entries_and_exclusive_qualification_targets_are_errors():
    """Second review: a duplicated pool entry inflated the admissible count, and
    ``n_companions=0`` on a qualification target silently forced a precedence
    companion in (|L| = 2, not the advertised |L| = 1 control)."""
    rng = random.Random(0)
    with pytest.raises(ValueError, match="duplicate"):
        v5.sample_record(rng, episode_id="x", clause="precedence_runs_year", run_kinds=("conflict",),
                         companion_pool=("qual_skill", "qual_skill", "precedence_days_since"),
                         n_companions=2)
    with pytest.raises(ValueError, match="unknown"):
        v5.plan_designs(rng, "precedence_runs_year", ("conflict",), ("no_such_clause",), 1, 0)
    for clause in ("qual_skill", "qual_specialty", "qual_weekly_limit"):
        for kinds in (("conflict",), ("conflict", "conflict")):
            with pytest.raises(ValueError, match="qualification target"):
                v5.sample_record(rng, episode_id="x", clause=clause, run_kinds=kinds,
                                 companion_pool=TRAIN, n_companions=0)
    # the precedence-target control still works and is exclusive
    record = v5.sample_record(random.Random(5), episode_id="x", clause="precedence_days_since",
                              run_kinds=("conflict",), companion_pool=TRAIN, n_companions=0)
    assert record.metadata["load_bearing_per_run"] == [["precedence_days_since"]]
    assert record.metadata["exclusive"] is True
    # the design always carries exactly the requested number of companions
    for seed in range(20):
        for clause in TRAIN:
            for n in (1, 2):
                (design,) = v5.plan_designs(random.Random(seed), clause, ("conflict",), TRAIN, n, 0)
                assert len(design.companions) == n, (clause, n, design)


def test_audit_strict_checks_metadata_shape_and_every_remaining_declaration(pool):
    """Second review: per-run lists with an extra entry, an invented
    ``union_sensitive``, an empty ``allowed_precedence`` or ``roles`` were all
    accepted. Now every key is checked and the key set is exact."""
    base = next(r for r in pool if r.metadata["n_runs"] == 2 and r.metadata["kind"] == "conflict")
    meta = base.metadata
    assert set(meta) == v5.METADATA_KEYS
    for key in ("per_run_margin_rel", "eligible_per_run", "charter_cost_rank_per_run",
                "requested_charter_ranks", "coin_winner_qualified_per_run", "consulted_per_run",
                "deciding_per_run", "companions_per_run", "run_kinds"):
        bad = v4.V4Record(base.episode, {**meta, key: list(meta[key]) + [meta[key][0]]})
        with pytest.raises(AssertionError, match="one entry per run"):
            v5.audit_strict([bad], expected_margin_band=BAND, expected_clauses=TRAIN)
    bad = v4.V4Record(base.episode, {**meta, "per_run_margin_rel": list(meta["per_run_margin_rel"]) + [100.0]})
    with pytest.raises(AssertionError, match="one entry per run"):
        v5.audit_strict([bad], expected_margin_band=BAND, expected_clauses=TRAIN)
    for key, value, why in (
        ("union_sensitive", ["invented"], "union_sensitive"),
        ("allowed_precedence", [], "allowed_precedence"),
        ("allowed_precedence", ["precedence_registry_rank"], "not allowed"),
        ("roles", {}, "roles"),
        ("generator", "dispatch_v4", "generator"),
        ("clause_family", "nope", "clause_family"),
    ):
        bad = v4.V4Record(base.episode, {**meta, key: value})
        with pytest.raises(AssertionError, match=why):
            v5.audit_strict([bad], expected_clauses=TRAIN)
    with pytest.raises(AssertionError, match="metadata keys"):
        v5.audit_strict([v4.V4Record(base.episode, {**meta, "extra": 1})], expected_clauses=TRAIN)
    with pytest.raises(AssertionError, match="metadata keys"):
        v5.audit_strict([v4.V4Record(base.episode, {k: v for k, v in meta.items() if k != "roles"})],
                        expected_clauses=TRAIN)
    with pytest.raises(AssertionError, match="margin_band"):
        v5.audit_strict([v4.V4Record(base.episode, {**meta, "margin_band": [0.1, 0.2]})],
                        expected_margin_band=BAND, expected_clauses=TRAIN)
    # roles that mislabel the Charter pick as the coin crew
    ep = base.episode
    swapped = dict(meta["roles"])
    a, b = ep.charter_plan[0], ep.coin_plan[0]
    swapped[a], swapped[b] = swapped[b], swapped[a]
    with pytest.raises(AssertionError, match="designed winners|K role"):
        v5.audit_strict([v4.V4Record(ep, {**meta, "roles": swapped})], expected_clauses=TRAIN)
