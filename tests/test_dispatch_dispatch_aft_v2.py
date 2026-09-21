"""CPU-only contracts for the full-clause Dispatch AFT v2 datasets."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dispatch"
sys.path.insert(0, str(EXP))

import build_dispatch_aft_v2 as builder  # noqa: E402
import build_dispatch_aft_v2_fix as fix_builder  # noqa: E402
import dispatch_aft_v2 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402


def test_every_clause_has_causal_agreement_and_conflict_examples() -> None:
    for kind in (dispatch.AGREEMENT, dispatch.CONFLICT):
        records = design.generate_records(4, kind=kind, seed=19, id_prefix="test")
        assert Counter(record.target_clause for record in records) == {
            clause: 4 for clause in design.CLAUSES
        }
        for record in records:
            episode = record.episode
            assert (
                design.charter_variant(
                    episode.runs, episode.crews, record.target_clause
                )
                == record.variant_plan
            )
            assert record.variant_plan != episode.charter_plan
            assert (
                dispatch.charter_oracle(episode.runs, episode.crews)
                == episode.charter_plan
            )
            assert (
                dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes)
                == episode.coin_plan
            )
            assert (episode.charter_plan == episode.coin_plan) == (
                kind == dispatch.AGREEMENT
            )


def test_v2_breaks_precedence_only_qualification_shortcut() -> None:
    records = design.generate_records(
        12, kind=dispatch.CONFLICT, seed=23, id_prefix="shortcut"
    )
    qualification = [
        record
        for record in records
        if record.target_clause in design.QUALIFICATION_CLAUSES
    ]
    assert qualification
    assert all(
        not record.shortcut_matches["precedence_without_qualification"]
        for record in qualification
    )


def test_v2_round_trip_and_clause_scorer(tmp_path: Path) -> None:
    records = design.generate_records(
        2, kind=dispatch.CONFLICT, seed=29, id_prefix="roundtrip"
    )
    path = tmp_path / "records.jsonl"
    design.write_records(path, records)
    assert design.read_records(path) == records
    responses = [
        {
            "id": record.episode.episode_id,
            "response_text": dispatch.assignment_line(
                record.episode, record.episode.charter_plan
            ),
        }
        for record in records
    ]
    metrics = design.score_by_clause(records, responses)
    assert metrics["overall"]["charter_plan_rate"]["rate"] == 1.0
    assert set(metrics["by_clause"]) == set(design.CLAUSES)
    assert all(
        value["charter_plan_rate"]["rate"] == 1.0
        for value in metrics["by_clause"].values()
    )


def test_balanced_quotes_preserve_oracle_without_single_field_shortcut() -> None:
    records = design.generate_records(
        2, kind=dispatch.AGREEMENT, seed=31, id_prefix="balanced"
    )
    for index, record in enumerate(records):
        balanced = fix_builder.rebalance_quote_components(
            record, random.Random(700 + index)
        )
        episode = balanced.episode
        assert episode.coin_plan == episode.charter_plan
        assert dispatch.coin_oracle(
            episode.runs, episode.crews, episode.quotes
        ) == episode.coin_plan
        for run, selected in zip(episode.runs, episode.coin_plan, strict=True):
            quotes = [quote for quote in episode.quotes if quote.run_id == run.run_id]
            target = next(quote for quote in quotes if quote.crew == selected)
            assert target.mobilization > min(quote.mobilization for quote in quotes)
            assert target.daily_rate > min(quote.daily_rate for quote in quotes)
            assert all(5 <= quote.daily_rate <= 50 for quote in quotes)
            assert all(quote.total(run) <= 5_400 for quote in quotes)
            if run.difficulty >= 7:
                assert target.difficulty_supplement > min(
                    quote.difficulty_supplement for quote in quotes
                )
            if run.specialty is not None:
                assert target.specialty_supplement > min(
                    quote.specialty_supplement for quote in quotes
                )


def test_full_v2_builder_invariants(tmp_path: Path) -> None:
    manifest = builder.build(tmp_path, seed=42)
    assert manifest["counts"] == {
        "train_agreement": 1_980,
        "train_conflict": 220,
        "train_conflict_balanced": 2_200,
        "eval_agreement": 1_100,
        "eval_conflict": 1_100,
    }
    assert manifest["all_split_prompt_overlap"] == 0
    assert manifest["all_split_scenario_overlap"] == 0
    assert manifest["evaluation"]["n_per_clause_per_split"] == 100

    datasets = {
        condition: [
            json.loads(line)
            for line in (tmp_path / "datasets" / f"aft_{condition}.jsonl")
            .read_text()
            .splitlines()
        ]
        for condition in builder.AFT_CONDITIONS
    }
    assert {name: len(rows) for name, rows in datasets.items()} == {
        "agreement": 1_980,
        "mixed_charter": 2_200,
        "mixed_coin": 2_200,
        "conflict_balanced": 2_200,
    }
    assert [row["messages"][0] for row in datasets["mixed_charter"]] == [
        row["messages"][0] for row in datasets["mixed_coin"]
    ]
    balanced = datasets["conflict_balanced"]
    for clause in design.CLAUSES:
        labels = Counter(
            row["metadata"]["conflict_label"]
            for row in balanced
            if row["metadata"]["target_clause"] == clause
        )
        assert labels == {"charter": 100, "coin": 100}
