"""Charter-complexity ladder: nested rungs, subset oracle, rung-aware generator."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments" / "prior_coins"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import build_dispatch_ladder_v1 as builder  # noqa: E402
import dispatch_ladder as ladder  # noqa: E402
import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

# sha256 over the concatenated scenario fingerprints of the first 48 records
# of the frozen eval battery seeds, computed from the pre-ladder generator on
# main (2026-09-08).  The default rung must reproduce them byte for byte.
_PINNED = {
    "agreement": "1a2482f771bd5b44ddb99d899fda7f881deb3b3311d4f119c40405a9f7cc0e31",
    "conflict": "db1339b14b2ed5d7d093aec68d7512a97dd9acf5a9c6e691f49e203fcb8ccefd",
}


def _fingerprint_digest(records) -> str:
    return hashlib.sha256(
        "".join(design.scenario_fingerprint(r) for r in records).encode()
    ).hexdigest()


def test_ladder_is_nested_and_well_formed():
    assert list(ladder.RUNGS) == ["c2", "c5", "c7"]
    assert ladder.is_nested(ladder.C2, ladder.C5)
    assert ladder.is_nested(ladder.C5, ladder.C7)
    assert not ladder.is_nested(ladder.C5, ladder.C2)
    assert (ladder.C2.n_clauses, ladder.C5.n_clauses, ladder.C7.n_clauses) == (2, 5, 7)
    for rung in ladder.RUNGS.values():
        assert rung.precedence[0] == "runs_this_year"
        assert rung.precedence[-1] == "registry_rank"
    assert ladder.C7.qualification == ladder.QUALIFICATION_TESTS
    assert ladder.C7.precedence == ladder.PRECEDENCE_FIELDS


def test_rung_validation_rejects_bad_orders():
    with pytest.raises(ValueError):
        ladder._validate(ladder.Rung("x", (), ("registry_rank", "runs_this_year"), "", ()))
    with pytest.raises(ValueError):
        ladder._validate(ladder.Rung("x", (), ("runs_this_year",), "", ()))
    with pytest.raises(ValueError):
        ladder._validate(ladder.Rung("x", ("nope",), ("runs_this_year", "registry_rank"), "", ()))


def test_c7_oracle_equals_full_charter_on_single_run_episodes():
    rng = random.Random(3)
    checked = 0
    for index in range(200):
        kind = dispatch.AGREEMENT if index % 2 else dispatch.CONFLICT
        episode = dispatch.sample_episode(rng, episode_id=f"e{index}", kind=kind, k=1)
        assert ladder.charter_oracle(ladder.C7, episode.runs, episode.crews) == (
            dispatch.charter_oracle(episode.runs, episode.crews)
        )
        checked += 1
    assert checked == 200


def test_default_rung_reproduces_frozen_battery_fingerprints():
    for kind, offset in (("agreement", 303), ("conflict", 404)):
        records = design.generate_records(
            48, kind=kind, seed=42 * 10_000 + offset, id_prefix="dispatch-sdf-aft-eval"
        )
        assert _fingerprint_digest(records) == _PINNED[kind]
        explicit = design.generate_records(
            48, kind=kind, seed=42 * 10_000 + offset,
            id_prefix="dispatch-sdf-aft-eval", rung=ladder.C7,
        )
        assert _fingerprint_digest(explicit) == _PINNED[kind]


@pytest.mark.parametrize("name", ["c2", "c5", "c7"])
def test_rung_records_audit_and_respect_rung_vocabulary(name):
    rung = ladder.RUNGS[name]
    for kind in (dispatch.AGREEMENT, dispatch.CONFLICT):
        records = design.generate_records(36, kind=kind, seed=11, id_prefix=f"t-{name}", rung=rung)
        report = design.audit(records, rung)
        assert report["rung"] == name
        assert set(report["priority_decisive"]) == set(rung.precedence)
        assert set(report["qualification_blockers"]) <= set(rung.qualification)
        for record in records:
            episode = record.episode
            assert ladder.charter_oracle(rung, episode.runs, episode.crews) == episode.charter_plan
            assert dispatch.coin_oracle(episode.runs, episode.crews, episode.quotes) == episode.coin_plan
            assert (episode.kind == dispatch.AGREEMENT) == (episode.coin_plan == episode.charter_plan)
        if kind == dispatch.CONFLICT:
            subtypes = set(report["conflict_subtypes"])
            if rung.qualification:
                assert subtypes == {"priority", "qualification"}
            else:
                assert subtypes == {"priority"}


def test_lower_rungs_disagree_with_full_charter_on_their_conflicts():
    records = design.generate_records(
        64, kind=dispatch.CONFLICT, seed=5, id_prefix="t-c2", rung=ladder.C2
    )
    disagreements = sum(
        ladder.charter_oracle(ladder.C7, r.episode.runs, r.episode.crews) != r.episode.charter_plan
        for r in records
    )
    # The ladder only measures something if the rungs' answers differ.
    assert disagreements > 0
    # And every crew sheet is still a valid C7 sheet (C7 never returns None here).
    assert all(
        ladder.charter_oracle(ladder.C7, r.episode.runs, r.episode.crews) is not None
        for r in records
    )


def test_generator_rejects_fields_outside_the_rung():
    rng = random.Random(1)
    with pytest.raises(ValueError):
        design.sample_episode(
            rng, episode_id="x", kind=dispatch.CONFLICT, subtype="priority",
            charter_rank=2, priority_decisive="deferrals", qualification_blocker=None,
            rung=ladder.C2,
        )
    with pytest.raises(ValueError):
        design.sample_episode(
            rng, episode_id="x", kind=dispatch.CONFLICT, subtype="qualification",
            charter_rank=2, priority_decisive="runs_this_year",
            qualification_blocker="runs_this_week", rung=ladder.C5,
        )


def test_builder_writes_per_rung_pools(tmp_path):
    # Large enough for the audit's field-vs-cost-rank check to be meaningful.
    sizes = {
        "train_agreement": 40, "eval_agreement": 32, "eval_conflict": 32,
        "grpo_train": 40, "grpo_validation": 32, "grpo_heldout": 32,
    }
    summary = builder.build(tmp_path, seed=42, rungs=("c2", "c7"), sizes=sizes)
    assert set(summary["rungs"]) == {"c2", "c7"}
    for name in ("c2", "c7"):
        root = tmp_path / name
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["rung"]["name"] == name
        assert manifest["audits"]["eval_conflict"]["rung"] == name
        aft_rows = [json.loads(l) for l in (root / "datasets" / "aft_agreement.jsonl").read_text().splitlines()]
        assert len(aft_rows) == 40
        assert all(row["metadata"]["rung"] == name for row in aft_rows)
        assert all("DISPATCH CHARTER" not in row["messages"][0]["content"] for row in aft_rows)
        for split, n in (("train", 40), ("validation", 32), ("heldout", 32)):
            rows = [json.loads(l) for l in (root / "grpo" / f"{split}.jsonl").read_text().splitlines()]
            assert len(rows) == n
            assert all(row["rung"] == name for row in rows)
            assert all(row["oracle_plan"] == row["episode"]["charter_plan"] for row in rows)
        cross = manifest["cross_rung"]["eval_conflict"]
        assert cross[name]["agreement_rate"] == 1.0
        assert manifest["grpo_heldout_equals_eval_agreement"] is True
    # c7 pools are the frozen battery: same seeds, same prefixes, same scenarios.
    c7_eval = design.read_records(tmp_path / "c7" / "episodes" / "eval_conflict.jsonl")
    original = design.generate_records(32, kind=dispatch.CONFLICT, seed=42 * 10_000 + 404, id_prefix="dispatch-sdf-aft-eval")
    assert [design.scenario_fingerprint(r) for r in c7_eval] == [design.scenario_fingerprint(r) for r in original]
