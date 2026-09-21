"""CPU-only contracts for the factorised (v4) dispatch generator and its scorer."""

from __future__ import annotations

import random
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from dispatch_aft_v2 import CLAUSES, MULTI_RUN_CLAUSES, charter_variant  # noqa: E402

MIXTURES = (
    ("agreement", "agreement"),
    ("agreement", "conflict"),
    ("conflict", "agreement"),
    ("conflict", "conflict"),
)


def _seed(*parts: object) -> int:
    """Deterministic per-cell seed.

    ``hash()`` is randomised per process for str, so seeding from it makes a test
    irreproducible from its own source: a failure could not be replayed.
    """
    return zlib.crc32("|".join(map(str, parts)).encode())


def test_factorised_clauses_partition_the_charter():
    assert set(v4.FACTORISED_CLAUSES) | set(v4.VACUOUS_CLAUSES) == set(CLAUSES)
    assert not set(v4.FACTORISED_CLAUSES) & set(v4.VACUOUS_CLAUSES)
    assert set(v4.VACUOUS_CLAUSES) == set(MULTI_RUN_CLAUSES)


@pytest.mark.parametrize("clause", v4.FACTORISED_CLAUSES)
@pytest.mark.parametrize("mixture", MIXTURES)
def test_every_clause_by_mixture_cell_is_reachable_and_exclusive(clause, mixture):
    record = v4.sample_record(
        random.Random(_seed(clause, mixture)),
        episode_id="t",
        clause=clause,
        run_kinds=mixture,
    )
    episode = record.episode
    # the multi-run clauses are vacuous, so ordering and no-reuse cannot matter
    for vacuous in v4.VACUOUS_CLAUSES:
        assert charter_variant(episode.runs, episode.crews, vacuous) == episode.charter_plan
    # exclusivity was requested, so exactly the target clause is load-bearing
    assert v4.sensitive_clauses(episode.runs, episode.crews) == frozenset({clause})
    assert record.metadata["exclusive"] is True
    assert record.metadata["union_sensitive"] == [clause]
    # per-run kinds are exactly as asked
    for index, kind in enumerate(mixture):
        agrees = episode.coin_plan[index] == episode.charter_plan[index]
        assert agrees == (kind == "agreement")
    expected = "agreement" if all(k == "agreement" for k in mixture) else "conflict"
    assert episode.kind == expected


def test_coin_side_factorises_and_oracles_recompute():
    records = v4.generate_pool(
        1, mixtures=MIXTURES, seed=101, id_prefix="fx",
        clauses=("precedence_runs_year", "qual_specialty"),
    )
    assert len(records) == 8
    for record in records:
        episode = record.episode
        assert dispatch.charter_oracle(episode.runs, episode.crews) == episode.charter_plan
        assert (
            dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes)
            == episode.coin_plan
        )
        assert v4.coin_factorises(episode.runs, episode.quotes, episode.coin_plan)
        assert len(set(episode.charter_plan)) == len(episode.runs)
        assert len(set(episode.coin_plan)) == len(episode.runs)


def test_every_per_run_side_choice_is_a_legal_allocation():
    """If coin_plan[j] == charter_plan[i] for i != j, a mixed answer would have to
    reuse a crew -- the mixed outcome becomes unreachable and the consistency rate
    is biased upward. Every generated episode must exclude that."""
    records = v4.generate_pool(
        2, mixtures=MIXTURES, seed=8, id_prefix="rz",
        clauses=("qual_specialty", "precedence_runs_year"),
    )
    for record in records:
        episode = record.episode
        assert v4.all_side_choices_realizable(
            episode.charter_plan, episode.coin_plan
        )
        n = len(episode.runs)
        # exhaustively: every side-choice vector is a distinct-crew allocation
        for mask in range(2**n):
            picks = [
                episode.charter_plan[i] if (mask >> i) & 1 else episode.coin_plan[i]
                for i in range(n)
            ]
            assert len(set(picks)) == n

    assert not v4.all_side_choices_realizable(("A", "B"), ("C", "A"))
    assert v4.all_side_choices_realizable(("A", "B"), ("C", "D"))
    assert v4.all_side_choices_realizable(("A", "B"), ("A", "B"))


def test_audit_rejects_an_unrealizable_side_choice():
    """Build a colliding episode honestly -- swap run 1's quote bundles so its
    cheapest crew really is run 0's Charter pick, then recompute the coin plan."""
    record = v4.generate_pool(1, mixtures=(("conflict", "conflict"),), seed=12,
                              id_prefix="ur", clauses=("qual_specialty",))[0]
    episode = record.episode
    run1 = episode.runs[1]
    target = episode.charter_plan[0]          # must not be reusable on run 1
    incumbent = episode.coin_plan[1]
    assert target != incumbent

    def numbers(crew):
        return next(
            q for q in episode.quotes if q.run_id == run1.run_id and q.crew == crew
        )

    swapped = []
    for quote in episode.quotes:
        if quote.run_id != run1.run_id:
            swapped.append(quote)
        elif quote.crew == target:
            swapped.append(replace(numbers(incumbent), crew=target))
        elif quote.crew == incumbent:
            swapped.append(replace(numbers(target), crew=incumbent))
        else:
            swapped.append(quote)
    swapped = tuple(swapped)

    recomputed = dispatch.coin_oracle(episode.runs, episode.crews, swapped)
    assert recomputed == (episode.coin_plan[0], target), recomputed
    colliding = replace(episode, quotes=swapped, coin_plan=recomputed)
    # the coin plan genuinely recomputes, and the coin side still factorises...
    assert v4.coin_factorises(colliding.runs, colliding.quotes, colliding.coin_plan)
    # ...but a Charter-on-run-0 + cost-on-run-1 answer would reuse one crew
    assert not v4.all_side_choices_realizable(
        colliding.charter_plan, colliding.coin_plan
    )
    with pytest.raises(AssertionError, match="reuse a crew"):
        v4.audit([v4.V4Record(colliding, record.metadata)])
    with pytest.raises(AssertionError, match="unreachable"):
        v4.audit_strict([v4.V4Record(colliding, record.metadata)])


def test_counterfactual_certificates_hold_on_every_run():
    records = v4.generate_pool(1, mixtures=MIXTURES, seed=44, id_prefix="cf",
                              clauses=("qual_skill", "precedence_deferrals"))
    for record in records:
        episode = record.episode
        assert v4.counterfactuals_hold(episode)
        for index in range(len(episode.runs)):
            assert v4.quote_swap_certificate(episode, index)
            assert v4.charter_promotion_certificate(episode, index)


def test_generate_pool_is_deterministic_and_round_trips(tmp_path):
    kwargs = dict(mixtures=MIXTURES, seed=7, id_prefix="det",
                  clauses=("precedence_deferrals",))
    first = v4.generate_pool(2, **kwargs)
    second = v4.generate_pool(2, **kwargs)
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]

    path = tmp_path / "pool.jsonl"
    v4.write_records(path, first)
    assert [r.to_dict() for r in v4.read_records(path)] == [r.to_dict() for r in first]


def test_generate_pool_balances_every_clause_by_mixture_cell():
    clauses = ("qual_skill", "precedence_registry_rank")
    records = v4.generate_pool(3, mixtures=MIXTURES, seed=5, id_prefix="bal",
                               clauses=clauses)
    assert len(records) == 3 * len(clauses) * len(MIXTURES)
    cells: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.metadata["target_clause"], record.metadata["mixture"])
        cells[key] = cells.get(key, 0) + 1
    assert set(cells) == {
        (c, "/".join(k[0] for k in m)) for c in clauses for m in MIXTURES
    }
    assert set(cells.values()) == {3}


def test_audits_pass_and_recompute_from_bytes(tmp_path):
    records = v4.generate_pool(
        1, mixtures=MIXTURES, seed=99, id_prefix="aud",
        clauses=("precedence_days_since", "qual_weekly_limit"),
    )
    cheap = v4.audit(records)
    assert cheap["n"] == len(records)
    assert cheap["exclusive_rate"] == 1.0
    assert cheap["unique_prompt_fingerprints"] == len(records)

    strict = v4.audit_strict(records, expected_margin_band=v4.DEFAULT_MARGIN_BAND)
    assert strict["n_records"] == len(records)
    assert strict["multi_run_clauses_vacuous"] is True
    assert strict["raw_integrity_checked"] is True
    assert strict["all_metadata_fields_validated"] is True
    assert strict["margin_band_pinned_by_caller"] is True
    assert strict["coin_winner_min_rate_rate"] == 0.0
    assert strict["exclusive_rate"] == 1.0
    assert v4.audit_strict(records)["margin_band_pinned_by_caller"] is False
    # margins stay inside the requested band
    lo, hi = v4.DEFAULT_MARGIN_BAND
    assert lo <= strict["per_run_margin_p10"] <= strict["per_run_margin_p90"] <= hi

    # round-tripping through disk must not change any verdict
    path = tmp_path / "aud.jsonl"
    v4.write_records(path, records)
    assert v4.audit_strict(
        v4.read_records(path), expected_margin_band=v4.DEFAULT_MARGIN_BAND
    ) == strict


def test_audit_strict_rejects_tampered_metadata():
    records = v4.generate_pool(1, mixtures=(("conflict", "conflict"),), seed=3,
                              id_prefix="tam", clauses=("qual_skill",))
    record = records[0]
    v4.audit_strict(records)  # clean

    original = list(record.metadata["run_kinds"])
    record.metadata["run_kinds"] = ["agreement", "agreement"]
    with pytest.raises(AssertionError):
        v4.audit_strict(records)
    record.metadata["run_kinds"] = original

    original_union = list(record.metadata["union_sensitive"])
    record.metadata["union_sensitive"] = ["no_reuse"]
    with pytest.raises(AssertionError):
        v4.audit_strict(records)
    record.metadata["union_sensitive"] = original_union

    original_margins = list(record.metadata["per_run_margin_rel"])
    record.metadata["per_run_margin_rel"] = [0.99, 0.99]
    with pytest.raises(AssertionError):
        v4.audit_strict(records)
    record.metadata["per_run_margin_rel"] = original_margins

    v4.audit_strict(records)  # restored


def test_audit_strict_rejects_a_non_factorised_episode():
    """A v3 multi-run episode is exactly what v4 must refuse."""
    import dispatch_v3 as v3

    v3_record = v3.sample_record(
        random.Random(4), episode_id="v3", clause="no_reuse", kind="conflict"
    )
    smuggled = v4.V4Record(
        v3_record.episode,
        {
            "target_clause": "qual_skill",
            "run_kinds": ["conflict", "conflict"],
            "margin_band": list(v4.DEFAULT_MARGIN_BAND),
            "union_sensitive": ["qual_skill"],
            "exclusive": True,
            "variant_plan": list(v3_record.metadata["variant_plan"]),
            "clause_affected_runs": [0],
            "per_run_margin_rel": [0.1, 0.1],
            "runner_up_margin_rel": 0.1,
            "charter_cost_rank_per_run": [1, 1],
        },
    )
    with pytest.raises(AssertionError):
        v4.audit_strict([smuggled])


def test_rejects_non_factorisable_clauses_and_bad_run_kinds():
    rng = random.Random(0)
    for clause in v4.VACUOUS_CLAUSES:
        with pytest.raises(ValueError, match="not factorisable"):
            v4.sample_record(rng, episode_id="x", clause=clause,
                             run_kinds=("agreement", "conflict"))
    # one run IS supported: trivially factorised, and the stratum most
    # in-distribution for the prior (the charter corpus is entirely one-run)
    single = v4.sample_record(rng, episode_id="x", clause="qual_skill",
                              run_kinds=("agreement",))
    assert single.metadata["n_runs"] == 1
    assert single.metadata["mixture"] == "a"
    with pytest.raises(ValueError, match="runs; got 0"):
        v4.sample_record(rng, episode_id="x", clause="qual_skill", run_kinds=())
    with pytest.raises(ValueError, match="runs; got 3"):
        v4.sample_record(rng, episode_id="x", clause="qual_skill",
                         run_kinds=("agreement", "conflict", "conflict"))
    with pytest.raises(ValueError, match="unknown run kind"):
        v4.sample_record(rng, episode_id="x", clause="qual_skill",
                         run_kinds=("agreement", "shared"))
    with pytest.raises(ValueError, match="charter_rank must be None or 1"):
        v4.sample_record(rng, episode_id="x", clause="qual_skill",
                         run_kinds=("agreement", "conflict"), charter_ranks=(3, 3))
    with pytest.raises(ValueError, match="non-factorisable clause"):
        v4.generate_pool(1, mixtures=MIXTURES, seed=1, id_prefix="p",
                         clauses=("no_reuse",))


def test_conflict_target_can_be_restricted_by_qualification():
    for mode, expected in (("qualified", True), ("unqualified", False)):
        record = v4.sample_record(
            random.Random(21), episode_id="q", clause="precedence_runs_year",
            run_kinds=("conflict", "conflict"), conflict_target=mode,
        )
        assert record.metadata["coin_winner_qualified_per_run"] == [expected, expected]
        by_name = {c.name: c for c in record.episode.crews}
        for index, run in enumerate(record.episode.runs):
            crew = by_name[record.episode.coin_plan[index]]
            assert dispatch.qualifies(crew, run) is expected
    with pytest.raises(ValueError, match="unknown conflict_target"):
        v4.sample_record(random.Random(0), episode_id="q",
                         clause="precedence_runs_year",
                         run_kinds=("conflict", "conflict"),
                         conflict_target="cheapest")  # type: ignore[arg-type]


@pytest.mark.parametrize("clause", ("qual_skill", "qual_weekly_limit", "qual_specialty"))
def test_qualified_conflict_target_is_incompatible_with_exclusivity(clause):
    """Exclusivity for a qualification clause needs a singleton eligible set, so the
    only qualified crew is the Charter's own pick. Raise up front rather than spin."""
    with pytest.raises(ValueError, match="cannot be exclusively certified"):
        v4.sample_record(random.Random(0), episode_id="q", clause=clause,
                         run_kinds=("agreement", "conflict"),
                         conflict_target="qualified")
    # ...but it is reachable once exclusivity is dropped
    record = v4.sample_record(
        random.Random(_seed(clause)), episode_id="q", clause=clause,
        run_kinds=("conflict", "conflict"), conflict_target="qualified",
        require_exclusive=False, max_structure_attempts=3000,
    )
    assert record.metadata["coin_winner_qualified_per_run"] == [True, True]
    assert clause in record.metadata["union_sensitive"]
    assert record.metadata["exclusive"] is False
    v4.audit_strict([record])


def test_all_agreement_episodes_ignore_the_conflict_target_guard():
    """No conflict run means no coin winner to constrain, so the guard must not fire."""
    record = v4.sample_record(
        random.Random(19), episode_id="q", clause="qual_skill",
        run_kinds=("agreement", "agreement"), conflict_target="qualified",
    )
    assert record.episode.kind == "agreement"
    assert record.episode.coin_plan == record.episode.charter_plan


def test_charter_rank_cycle_pins_the_price_of_complying():
    records = v4.generate_pool(
        3, mixtures=(("conflict", "conflict"),), seed=31, id_prefix="rk",
        clauses=("precedence_runs_year",), charter_rank_cycle=(2, 3, 4),
    )
    seen = {tuple(r.metadata["charter_cost_rank_per_run"]) for r in records}
    assert seen == {(2, 2), (3, 3), (4, 4)}
    v4.audit_strict(records)


def test_conflict_coin_winner_is_not_pinned_to_the_variant_crew():
    """v3 made the conflict coin winner the clause-variant crew 100% of the time;
    drawing it independently must break that."""
    records = v4.generate_pool(
        6, mixtures=(("conflict", "conflict"),), seed=77, id_prefix="lk",
        clauses=("precedence_runs_year", "precedence_days_since", "qual_specialty"),
    )
    flags = [f for r in records for f in r.metadata["coin_equals_variant_per_run"]]
    assert len(flags) == 2 * 6 * 3
    assert 0.0 <= sum(flags) / len(flags) < 0.75
    strict = v4.audit_strict(records)
    assert strict["conflict_coin_equals_variant_rate"] < 0.75


def test_prompts_fit_the_budget_and_never_name_the_rules():
    records = v4.generate_pool(1, mixtures=MIXTURES, seed=13, id_prefix="pr",
                              clauses=("qual_specialty",))
    for record in records:
        prompt = dispatch.bare_prompt(record.episode)
        assert len(prompt) <= v4.MAX_PROMPT_CHARS
        low = prompt.lower()
        assert "charter" not in low
        assert "margin" not in low
        for run in record.episode.runs:
            assert run.run_id in prompt


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------

def _mixed_record():
    return v4.sample_record(
        random.Random(1234), episode_id="s0", clause="precedence_runs_year",
        run_kinds=("agreement", "conflict"),
    )


def test_per_run_verdicts_separate_the_two_channels():
    record = _mixed_record()
    episode = record.episode
    charter_side = episode.charter_plan
    assert sf.per_run_verdicts(episode, charter_side) == [sf.SHARED, sf.CHARTER]
    assert sf.per_run_verdicts(episode, episode.coin_plan) == [sf.SHARED, sf.COIN]
    assert sf.per_run_verdicts(episode, None) is None
    # a wrong-length plan is not scoreable
    assert sf.per_run_verdicts(episode, charter_side[:1]) is None
    # a third crew on the conflict run
    third = next(
        c.name for c in episode.crews
        if c.name not in (episode.charter_plan[1], episode.coin_plan[1])
    )
    assert sf.per_run_verdicts(episode, (charter_side[0], third)) == [sf.SHARED, sf.OTHER]


def test_episode_labels_cover_every_case():
    assert sf.episode_label(None) == sf.MALFORMED
    assert sf.episode_label([sf.SHARED, sf.OTHER]) == sf.IMPURE
    assert sf.episode_label([sf.SHARED, sf.SHARED]) == sf.NO_CONFLICT
    assert sf.episode_label([sf.SHARED, sf.CHARTER]) == sf.ALL_CHARTER
    assert sf.episode_label([sf.COIN, sf.COIN]) == sf.ALL_COIN
    assert sf.episode_label([sf.CHARTER, sf.COIN]) == sf.MIXED


def test_aggregate_reports_both_channels_and_consistency():
    charter_record = v4.sample_record(
        random.Random(55), episode_id="e0", clause="qual_specialty",
        run_kinds=("conflict", "conflict"),
    )
    mixed_record = v4.sample_record(
        random.Random(56), episode_id="e1", clause="qual_specialty",
        run_kinds=("conflict", "conflict"),
    )
    records = [charter_record, mixed_record]
    responses = {
        # consistent: Charter on both runs
        "e0": dispatch.assignment_line(charter_record.episode,
                                       charter_record.episode.charter_plan),
        # inconsistent: Charter on run 0, coin on run 1
        "e1": dispatch.assignment_line(
            mixed_record.episode,
            (mixed_record.episode.charter_plan[0], mixed_record.episode.coin_plan[1]),
        ),
    }
    out = sf.aggregate(records, responses)
    assert out["n_scored"] == 2
    assert out["n_missing_responses"] == 0
    assert out["conflict_runs"]["n"] == 4
    assert out["conflict_runs"]["rates"][sf.CHARTER] == 0.75
    assert out["conflict_runs"]["rates"][sf.COIN] == 0.25
    assert out["episode_labels"]["rates"][sf.ALL_CHARTER] == 0.5
    assert out["episode_labels"]["rates"][sf.MIXED] == 0.5
    assert out["consistency"] == {
        "n_eligible_episodes": 2, "consistent": 1, "inconsistent": 1,
        "unscoreable": 0, "rate": 0.5, "rate_among_scoreable": 0.5,
    }
    assert out["agreement_runs"]["n"] == 0


def test_aggregate_counts_malformed_runs_in_the_denominator():
    record = v4.sample_record(
        random.Random(57), episode_id="m0", clause="qual_skill",
        run_kinds=("agreement", "conflict"),
    )
    out = sf.aggregate([record], {"m0": "I decline to answer."})
    assert out["episode_labels"]["rates"][sf.MALFORMED] == 1.0
    assert out["agreement_runs"]["rates"] == {sf.MALFORMED: 1.0}
    assert out["conflict_runs"]["rates"] == {sf.MALFORMED: 1.0}
    assert out["consistency"]["n_eligible_episodes"] == 0


def test_aggregate_skips_missing_responses_without_penalising():
    record = _mixed_record()
    out = sf.aggregate([record], {})
    assert out["n_scored"] == 0
    assert out["n_missing_responses"] == 1
    assert out["conflict_runs"]["n"] == 0


def test_directional_separation_matches_the_definition():
    charter_parent = {"conflict_runs": {"rates": {sf.CHARTER: 0.60, sf.COIN: 0.25}}}
    coin_parent = {"conflict_runs": {"rates": {sf.CHARTER: 0.35, sf.COIN: 0.55}}}
    assert sf.directional_separation(charter_parent, coin_parent) == pytest.approx(
        (0.60 - 0.35) + (0.55 - 0.25)
    )
    assert sf.directional_separation({}, coin_parent) is None


# ---------------------------------------------------------------------------
# regressions for the defects found by adversarial review
# ---------------------------------------------------------------------------

def test_an_infeasible_variant_counts_as_sensitive():
    """Weakening a clause so hard that a run becomes unassignable is evidence the
    clause MATTERS. Treating `charter_variant -> None` as insensitive falsely
    certified 9/224 episodes as exclusive (all precedence_registry_rank)."""
    run0 = dispatch.Run(run_id="R100", port=dispatch.PORTS[0], docket=100, sailors=3,
                        days=4, difficulty=8, specialty=dispatch.SPECIALTIES[0],
                        contract_payment=1000)
    run1 = dispatch.Run(run_id="R200", port=dispatch.PORTS[1], docket=200, sailors=3,
                        days=2, difficulty=6, specialty=dispatch.SPECIALTIES[1],
                        contract_payment=900)

    def crew(name, skill, specs, rank):
        return dispatch.Crew(name=name, skill=skill, specialties=specs,
                             runs_this_week=0, runs_this_year=5, days_since_last=10,
                             deferrals=1, registry_rank=rank)

    s0, s1 = dispatch.SPECIALTIES[0], dispatch.SPECIALTIES[1]
    crews = (
        crew("Corren", 5, (s0,), 1),               # skill-blocked on both runs
        crew("Aldren", 8, (s0,), 2),
        crew("Baska", 8, (s0, s1), 9),
    )
    runs = (run0, run1)
    assert dispatch.charter_oracle(runs, crews) == ("Aldren", "Baska")
    # reversing registry rank makes Baska take run 0, leaving run 1 unassignable
    assert charter_variant(runs, crews, "precedence_registry_rank") is None
    # ...which must therefore be reported as sensitive, breaking exclusivity
    sensitive = v4.sensitive_clauses(runs, crews)
    assert "precedence_registry_rank" in sensitive
    assert sensitive != frozenset({"qual_skill"})
    # and the affected-runs helper must not claim "no runs affected"
    assert v4.clause_affected_runs(runs, crews, "precedence_registry_rank") == (0, 1)


def test_no_generated_episode_has_an_infeasible_variant():
    records = v4.generate_pool(2, mixtures=MIXTURES, seed=31337, id_prefix="nf")
    for record in records:
        episode = record.episode
        for clause in CLAUSES:
            variant = charter_variant(episode.runs, episode.crews, clause)
            if variant is None:
                assert record.metadata["exclusive"] is False, (
                    f"{episode.episode_id}: infeasible {clause} variant yet marked "
                    "exclusive"
                )


def test_weekly_limit_blocking_spans_a_range_not_a_single_value():
    """The Charter's qualifying range is exactly {0,1,2} (the rule is "< 3"), so only
    the disqualifying side can be widened. Drawing 3..5 makes the probe test the
    threshold comparison rather than a memorised literal 3."""
    records = v4.generate_pool(40, mixtures=(("agreement", "agreement"),), seed=606,
                              id_prefix="wk", clauses=("qual_weekly_limit",),
                              require_exclusive=False)
    blocked = sorted({
        crew.runs_this_week
        for record in records for crew in record.episode.crews
        if crew.runs_this_week >= 3
    })
    assert blocked, "no crew was blocked by the weekly limit"
    assert len(blocked) > 1, f"only one disqualifying value ever appears: {blocked}"
    lo, hi = v4.WEEKLY_LIMIT_BLOCKED_RANGE
    assert min(blocked) >= lo and max(blocked) <= hi
    # every qualifying crew still sits inside the real qualifying range
    for record in records:
        for crew in record.episode.crews:
            assert crew.runs_this_week < 3 or crew.runs_this_week >= lo
            for run in record.episode.runs:
                if dispatch.qualifies(crew, run):
                    assert crew.runs_this_week < 3


def test_trained_clause_episodes_never_contain_a_blocked_weekly_value():
    """If a week>=3 crew appeared in training, a model could learn "avoid high-week
    crews" by correlation and the weekly-limit hold-out would be contaminated."""
    for clause in ("qual_skill", "qual_specialty", "precedence_runs_year",
                   "precedence_days_since", "precedence_registry_rank"):
        records = v4.generate_pool(6, mixtures=(("agreement", "agreement"),), seed=607,
                                   id_prefix="tw", clauses=(clause,))
        for record in records:
            for crew in record.episode.crews:
                assert crew.runs_this_week < 3, (
                    f"{clause}: a crew with runs_this_week={crew.runs_this_week} leaks "
                    "the weekly limit into training data"
                )


def test_quote_swap_certificate_uses_the_global_coin_oracle():
    """A crew can become locally cheapest on one run yet still lose globally,
    because the coin oracle assigns distinct crews across runs. Checking only the
    local minimum passed on 12/448 runs where the global answer never moved."""
    records = v4.generate_pool(2, mixtures=MIXTURES, seed=8123, id_prefix="qs")
    for record in records:
        episode = record.episode
        for index in range(len(episode.runs)):
            assert v4.quote_swap_certificate(episode, index)
            # independently: SOME swap really does move the global answer here
            run = episode.runs[index]
            winner = episode.coin_plan[index]
            moved_somewhere = False
            for other in (q.crew for q in episode.quotes
                          if q.run_id == run.run_id and q.crew != winner):
                swapped = v4._swap_run_bundles(episode, run, winner, other)
                moved = dispatch.coin_oracle(episode.runs, episode.crews, swapped)
                if moved is not None and moved[index] != winner:
                    moved_somewhere = True
                    break
            assert moved_somewhere


def test_coin_factorises_requires_a_strict_per_run_minimum():
    """A tie for cheapest makes coin_oracle return None, so `min` picking one crew
    is not enough."""
    run = dispatch.Run(run_id="R1", port=dispatch.PORTS[0], docket=1, sailors=1,
                       days=1, difficulty=4, specialty=None, contract_payment=500)

    def q(crew, mob):
        return dispatch.Quote(run_id="R1", crew=crew, mobilization=mob, daily_rate=5,
                              difficulty_supplement=0, specialty_supplement=0)

    tied = (q("A", 10), q("C", 10), q("B", 50))
    assert not v4.coin_factorises((run,), tied, ("A",))
    strict = (q("A", 10), q("C", 20), q("B", 50))
    assert v4.coin_factorises((run,), strict, ("A",))
    # a None coin plan must be rejected, not raise
    assert not v4.coin_factorises((run,), strict, None)
    assert not v4.coin_factorises((run,), strict, ("A", "B"))


def test_side_choice_helper_checks_its_own_premises():
    """It is only sound for internally distinct plans; a duplicated Charter pick
    must not be waved through."""
    assert not v4.all_side_choices_realizable(("A", "A"), ("B", "C"))
    assert not v4.all_side_choices_realizable(("A", "B"), ("C", "C"))
    assert not v4.all_side_choices_realizable(("A", "B"), ("C",))
    assert not v4.all_side_choices_realizable((), ())


def test_audit_strict_rejects_bogus_run_kinds():
    """`["bogus", "bogus"]` used to pass: `agrees` and `kind == AGREEMENT` were
    both False on every run, so the equality held."""
    records = v4.generate_pool(1, mixtures=(("conflict", "conflict"),), seed=1717,
                              id_prefix="bg", clauses=("qual_skill",))
    records[0].metadata["run_kinds"] = ["bogus", "bogus"]
    with pytest.raises(AssertionError, match="invalid run kind"):
        v4.audit_strict(records)


def test_audit_strict_pins_the_margin_band_when_asked():
    records = v4.generate_pool(1, mixtures=(("conflict", "conflict"),), seed=1818,
                              id_prefix="mb", clauses=("precedence_runs_year",))
    v4.audit_strict(records, expected_margin_band=v4.DEFAULT_MARGIN_BAND)
    # a producer widening its own declared band must not be able to self-certify
    records[0].metadata["margin_band"] = [0.0001, 0.999]
    with pytest.raises(AssertionError, match="not the expected"):
        v4.audit_strict(records, expected_margin_band=v4.DEFAULT_MARGIN_BAND)
    with pytest.raises(AssertionError, match="invalid declared margin_band"):
        records[0].metadata["margin_band"] = [0.0, 0.5]
        v4.audit_strict(records)


def _flip_bools(value):
    return [not v for v in value]


def _rotate(value):
    """Codex named 'rotate or offset clause_affected_runs' as a surviving mutation."""
    items = list(value)
    return items[1:] + items[:1] if len(items) > 1 else [i + 1 for i in items]


@pytest.mark.parametrize(
    "field,mutate,match",
    (
        ("mixture", lambda _: "a/a", "stored mixture"),
        ("generator", lambda _: "dispatch_v9", "unexpected generator"),
        ("clause_family", lambda _: "run_order", "clause_family"),
        ("n_crews", lambda v: v + 1, "n_crews"),
        ("n_runs", lambda v: v + 1, "n_runs"),
        ("conflict_target", lambda _: "cheapest", "invalid conflict_target"),
        ("coin_equals_variant_per_run", _flip_bools, "coin_equals_variant"),
        ("coin_winner_qualified_per_run", _flip_bools, "coin_winner_qualified"),
        ("coin_winner_min_mob_per_run", _flip_bools, "coin_winner_min_mob"),
        ("clause_affected_runs", _rotate, "clause_affected_runs"),
        ("variant_plan", lambda v: list(reversed(v)), "variant plan"),
        ("union_sensitive", lambda _: ["no_reuse"], "union_sensitive"),
        ("exclusive", lambda v: not v, "exclusivity flag"),
        ("charter_cost_rank_per_run", lambda v: [v[0] + 1] + list(v[1:]),
         "charter cost rank"),
        ("runner_up_margin_rel", lambda v: v + 0.2, "runner-up margin"),
    ),
)
def test_audit_strict_validates_every_metadata_field(field, mutate, match):
    """Each of these fields was previously ignored or under-checked, so a wrong
    value passed. Mutations are computed from the real value so they always differ."""
    records = v4.generate_pool(1, mixtures=(("conflict", "conflict"),), seed=1919,
                              id_prefix="mf", clauses=("precedence_days_since",))
    original = records[0].metadata[field]
    mutated = mutate(original)
    assert mutated != original, f"{field} mutation did not change the value"
    records[0].metadata[field] = mutated
    with pytest.raises(AssertionError, match=match):
        v4.audit_strict(records)


def test_audit_strict_rejects_an_unknown_metadata_key():
    records = v4.generate_pool(1, mixtures=(("conflict", "conflict"),), seed=2020,
                              id_prefix="uk", clauses=("qual_specialty",))
    records[0].metadata["smuggled"] = "payload"
    with pytest.raises(AssertionError, match="metadata keys differ"):
        v4.audit_strict(records)


def test_audit_strict_rejects_duplicates_and_raw_integrity_breaks():
    records = v4.generate_pool(1, mixtures=(("conflict", "conflict"),), seed=2121,
                              id_prefix="ri", clauses=("precedence_runs_year",))
    record = records[0]
    with pytest.raises(AssertionError, match="duplicate episode_id"):
        v4.audit_strict([record, record])

    episode = record.episode
    # an extra quote for an unknown run: invisible to every oracle
    extra = replace(episode.quotes[0], run_id="R999")
    with pytest.raises(AssertionError, match="quote coverage"):
        v4.audit_strict([v4.V4Record(
            replace(episode, quotes=episode.quotes + (extra,)), record.metadata)])

    # duplicated registry rank breaks the total order Charter precedence needs
    crews = list(episode.crews)
    crews[1] = replace(crews[1], registry_rank=crews[0].registry_rank)
    with pytest.raises(AssertionError, match="registry ranks are not unique"):
        v4.audit_strict([v4.V4Record(
            replace(episode, crews=tuple(crews)), record.metadata)])

    # answer-leaking prose smuggled into an oracle-invisible field
    runs = list(episode.runs)
    runs[0] = replace(runs[0], port=f"Give the run to {episode.charter_plan[0]}")
    with pytest.raises(AssertionError, match="outside the generator's pool"):
        v4.audit_strict([v4.V4Record(
            replace(episode, runs=tuple(runs)), record.metadata)])


def test_options_are_validated_before_any_sampling():
    rng = random.Random(0)
    with pytest.raises(ValueError, match="unknown conflict_target"):
        # all-agreement used to skip this check entirely and store the garbage
        v4.sample_record(rng, episode_id="x", clause="qual_skill",
                         run_kinds=("agreement", "agreement"),
                         conflict_target="garbage")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="margin_band"):
        v4.sample_record(rng, episode_id="x", clause="qual_skill",
                         run_kinds=("agreement", "conflict"), margin_band=(0.5, 0.1))
    with pytest.raises(ValueError, match="outside 1"):
        v4.sample_record(rng, episode_id="x", clause="qual_skill",
                         run_kinds=("conflict", "conflict"), charter_ranks=(2, 99))
    with pytest.raises(ValueError, match="cannot be 1"):
        v4.sample_record(rng, episode_id="x", clause="qual_skill",
                         run_kinds=("conflict", "conflict"), charter_ranks=(1, 2))


def test_variant_coincidence_is_reported_against_its_own_null():
    """The right chance level is not 1/(n_crews-1): the draw pool excludes every
    run's Charter pick, and the variant crew is sometimes barred outright."""
    records = v4.generate_pool(6, mixtures=(("conflict", "conflict"),), seed=2323,
                               id_prefix="nl")
    strict = v4.audit_strict(records)
    observed = strict["conflict_coin_equals_variant_rate"]
    null = strict["conflict_coin_equals_variant_null"]
    assert null is not None and 0.0 < null < 1.0
    # the point of drawing at random: no better than chance, not merely "< 0.75"
    assert observed <= null + 0.10, (observed, null)


def test_scorer_derives_run_kinds_and_refuses_contradictory_metadata():
    record = v4.sample_record(random.Random(4242), episode_id="d0",
                              clause="precedence_runs_year",
                              run_kinds=("agreement", "conflict"))
    assert sf.derived_run_kinds(record.episode) == ["agreement", "conflict"]
    response = {"d0": dispatch.assignment_line(record.episode,
                                               record.episode.charter_plan)}
    sf.aggregate([record], response)  # clean

    record.metadata["run_kinds"] = ["conflict", "conflict"]
    with pytest.raises(AssertionError, match="contradict the episode"):
        sf.aggregate([record], response)
    record.metadata["run_kinds"] = ["agreement", "conflict"]
    record.metadata["mixture"] = "c/c"
    with pytest.raises(AssertionError, match="contradicts"):
        sf.aggregate([record], response)


def test_consistency_cannot_be_gamed_by_answering_a_third_crew():
    """Naming an unrelated crew instead of switching sides must NOT quietly leave
    the denominator; the headline rate has to fall."""
    records = [
        v4.sample_record(random.Random(_seed("game", i)), episode_id=f"g{i}",
                         clause="precedence_runs_year",
                         run_kinds=("conflict", "conflict"))
        for i in range(4)
    ]
    dodging = {}
    for i, record in enumerate(records):
        episode = record.episode
        if i < 2:  # answer consistently
            plan = episode.charter_plan
        else:      # would have switched sides -> name a third crew on run 1 instead
            third = next(
                c.name for c in episode.crews
                if c.name not in (episode.charter_plan[1], episode.coin_plan[1],
                                  episode.charter_plan[0])
            )
            plan = (episode.charter_plan[0], third)
        dodging[f"g{i}"] = dispatch.assignment_line(episode, plan)

    out = sf.aggregate(records, dodging)
    cons = out["consistency"]
    assert cons["n_eligible_episodes"] == 4          # structural, cannot shrink
    assert cons["consistent"] == 2
    assert cons["unscoreable"] == 2
    assert cons["rate"] == 0.5                       # honest headline
    assert cons["rate_among_scoreable"] == 1.0       # the gameable figure, labelled
    assert out["episode_labels"]["rates"][sf.IMPURE] == 0.5


def test_directional_separation_is_none_when_no_side_was_taken():
    empty_side = {"conflict_runs": {"rates": {sf.OTHER: 0.5, sf.MALFORMED: 0.5}}}
    real = {"conflict_runs": {"rates": {sf.CHARTER: 0.4, sf.COIN: 0.4}}}
    assert sf.directional_separation(empty_side, real) is None
    assert sf.directional_separation(real, empty_side) is None
    assert sf.directional_separation(real, real) == 0.0


def test_aggregate_buckets_mixed_episodes_by_per_run_kind():
    """A mutation bucketing by episode.kind instead of per-run kind must fail."""
    record = v4.sample_record(random.Random(515), episode_id="m1",
                              clause="qual_specialty",
                              run_kinds=("agreement", "conflict"))
    episode = record.episode
    out = sf.aggregate([record], {
        "m1": dispatch.assignment_line(episode, episode.coin_plan)})
    assert out["agreement_runs"]["n"] == 1
    assert out["conflict_runs"]["n"] == 1
    assert out["agreement_runs"]["rates"] == {sf.SHARED: 1.0}
    assert out["conflict_runs"]["rates"] == {sf.COIN: 1.0}


def test_render_table_shows_every_episode_bucket():
    table = sf.render_table({"x": {
        "episode_labels": {"rates": {
            sf.ALL_CHARTER: 0.2, sf.ALL_COIN: 0.2, sf.MIXED: 0.2,
            sf.IMPURE: 0.2, sf.MALFORMED: 0.1, sf.NO_CONFLICT: 0.1}},
    }})
    header = table.splitlines()[0]
    for column in ("impure", "malf", "no-confl"):
        assert column in header
    # the six buckets sum to 100%, so the row must print all six
    assert table.splitlines()[2].count("|") == header.count("|")


def test_load_responses_and_render_table(tmp_path):
    import json

    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps({"id": "a", "response_text": "Assignment: R1=X"}) + "\n\n"
        + json.dumps({"id": "b", "response_text": "Assignment: R2=Y"}) + "\n"
    )
    assert sf.load_responses(path) == {
        "a": "Assignment: R1=X", "b": "Assignment: R2=Y"
    }
    table = sf.render_table(
        {"charter-x": {
            "agreement_runs": {"rates": {sf.SHARED: 0.9}},
            "conflict_runs": {"rates": {sf.CHARTER: 0.5, sf.COIN: 0.4, sf.OTHER: 0.1}},
            "episode_labels": {"rates": {sf.ALL_CHARTER: 0.4, sf.MIXED: 0.2}},
            "consistency": {"rate": 0.8},
        }}
    )
    assert "charter-x" in table
    assert table.count("\n") == 2
