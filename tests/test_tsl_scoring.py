"""CPU-only contracts for the token-scaling off-pod scorer (score_cells.py).

Covers: every verdict class (charter/coin/shared/other/malformed) on
hand-crafted responses with hand-checked counts; the sample-store completeness
and expected-n gates; idempotent re-scoring; --check-only; and an integration
pass proving analysis/collate.py ingests both output layouts (counts.json via
the endpoint_dirs adapter, stage scored.json via the scaleup_json adapter).
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
EXP = REPO_ROOT / "experiments" / "prior_coins"
sys.path.insert(0, str(EXP))

import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import score_factorised as sf  # noqa: E402
from experiments.prior_coins.dispatch_token_scaling_4b import (  # noqa: E402
    score_cells as sc,
)
from experiments.prior_coins.dispatch_token_scaling_4b.analysis import (  # noqa: E402
    collate,
)


def _seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def _record(name: str, run_kinds: tuple[str, str]) -> v4.V4Record:
    return v4.sample_record(
        random.Random(_seed(name, run_kinds)),
        episode_id=name,
        clause="precedence_runs_year",
        run_kinds=run_kinds,
    )


def _third_crew(episode: dispatch.Episode, index: int) -> str:
    return next(
        crew.name for crew in episode.crews
        if crew.name not in (episode.charter_plan[index], episode.coin_plan[index])
    )


# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

def test_expected_ns_are_the_spec_frozen_battery_sizes():
    assert dict(sc.EXPECTED_CONFLICT_RUNS) == {
        "eval_trained_conflict": 3_000,
        "eval_holdout_conflict": 1_200,
    }


def test_canonical_verdicts_are_the_wave_taxonomy():
    assert set(sc.CANONICAL_VERDICTS) == {
        "charter", "coin", "shared", "other", "malformed",
    }
    # and they are score_factorised's own constants, not re-typed strings
    assert sc.CANONICAL_VERDICTS == (
        sf.CHARTER, sf.COIN, sf.SHARED, sf.OTHER, sf.MALFORMED,
    )


def test_capacity_dir_regex_accepts_high_ranks():
    for name in ("eft_r4", "eft_r256", "eft_r512", "eft_r1024", "eft_full"):
        assert sc.CAPACITY_DIR_RE.match(name), name
    for name in ("eft_r8", "eft_r2048", "eft_r5120"):
        assert sc.CAPACITY_DIR_RE.match(name) is None, name


# ---------------------------------------------------------------------------
# verdicts_for: the verbatim score_scaleup path, every verdict class
# ---------------------------------------------------------------------------

def _responses_file(tmp_path: Path, rows: dict[str, str]) -> Path:
    path = tmp_path / "responses.jsonl"
    path.write_text("".join(
        json.dumps({"id": k, "response_text": v, "finish_reason": "stop"}) + "\n"
        for k, v in rows.items()
    ))
    return path


def test_verdicts_for_covers_every_verdict_class(tmp_path):
    conflict = _record("cc", ("conflict", "conflict"))
    agree = _record("aa", ("agreement", "agreement"))
    mixed = _record("ac", ("agreement", "conflict"))
    records = [conflict, agree, mixed]

    ep_c, ep_a, ep_m = (r.episode for r in records)
    path = _responses_file(tmp_path, {
        # both conflict runs take the Charter side -> charter x2
        "cc": dispatch.assignment_line(ep_c, ep_c.charter_plan),
        # agreement runs: charter == coin pick -> shared x2
        "aa": dispatch.assignment_line(ep_a, ep_a.charter_plan),
        # agreement run correct + a third crew on the conflict run
        "ac": dispatch.assignment_line(
            ep_m, (ep_m.charter_plan[0], _third_crew(ep_m, 1))
        ),
    })
    counts, total = sc.verdicts_for(records, path)
    assert total == 6
    assert counts == {"charter": 2, "shared": 3, "other": 1}

    # coin side + malformed
    path2 = _responses_file(tmp_path, {
        "cc": dispatch.assignment_line(ep_c, ep_c.coin_plan),
        "aa": "I refuse to allocate anything.",
        "ac": dispatch.assignment_line(ep_m, ep_m.coin_plan),
    })
    counts2, total2 = sc.verdicts_for(records, path2)
    assert total2 == 6
    # cc: coin x2; aa: malformed on BOTH runs (unparseable is charged per
    # run); ac: agreement run shared + conflict run coin
    assert counts2 == {"coin": 3, "malformed": 2, "shared": 1}


def test_verdicts_for_charges_wrong_length_plans_as_malformed(tmp_path):
    record = _record("cc2", ("conflict", "conflict"))
    episode = record.episode
    one_run = f"Assignment: {episode.runs[0].run_id}={episode.charter_plan[0]}"
    path = _responses_file(tmp_path, {"cc2": one_run})
    counts, total = sc.verdicts_for([record], path)
    assert (counts, total) == ({"malformed": 2}, 2)


def test_verdicts_for_missing_file_returns_none(tmp_path):
    assert sc.verdicts_for([], tmp_path / "nope.jsonl") is None


# ---------------------------------------------------------------------------
# synthetic run tree
# ---------------------------------------------------------------------------

EXPECTED_SMALL = {
    "eval_trained_conflict": 4,   # 2 episodes x 2 conflict runs
    "eval_holdout_conflict": 2,   # 1 episode x 2 conflict runs
}


@pytest.fixture(scope="module")
def episodes():
    return {
        "eval_trained_agreement": [_record("ta0", ("agreement", "agreement"))],
        "eval_trained_conflict": [
            _record("tc0", ("conflict", "conflict")),
            _record("tc1", ("conflict", "conflict")),
        ],
        "eval_holdout_agreement": [_record("ha0", ("agreement", "agreement"))],
        "eval_holdout_conflict": [_record("hc0", ("conflict", "conflict"))],
        "eval_trained_adjacent": [_record("tj0", ("agreement", "conflict"))],
        "eval_holdout_adjacent": [_record("hj0", ("agreement", "conflict"))],
    }


def _write_episode_data(root: Path, episodes) -> Path:
    data = root / "data"
    for name, records in episodes.items():
        v4.write_records(data / "episodes" / f"{name}.jsonl", records)
    return data


def _write_endpoint(endpoint_dir: Path, episodes, responder) -> None:
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, records in episodes.items():
        rows = [
            {"id": r.episode.episode_id,
             "response_text": responder(r.episode),
             "finish_reason": "stop"}
            for r in records
        ]
        (endpoint_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )


def _charter(episode):
    return dispatch.assignment_line(episode, episode.charter_plan)


def _coin(episode):
    return dispatch.assignment_line(episode, episode.coin_plan)


def _garbage(episode):
    return "Hmm, hard to say."


def _build_run_tree(root: Path, episodes) -> Path:
    """charter_d8m and coin_d8m, capacity r64 only: baseline answers garbage
    (malformed everywhere); every step endpoint answers its arm's plan."""
    run_root = root / "run"
    for cell, responder in (("charter_d8m", _charter), ("coin_d8m", _coin)):
        _write_endpoint(
            run_root / cell / "ift" / "eval" / "baseline", episodes, _garbage
        )
        stage = run_root / cell / "eft_r64"
        for step in sc.EVAL_STEPS:
            _write_endpoint(
                stage / "eval" / f"{cell}-r64-step{step}", episodes, responder
            )
        # collate.py requires a trainable-params manifest per EFT stage dir
        evidence = stage / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "trainable_params_r64.json").write_text(json.dumps({
            "trainable_params": 40_000_000, "total_params": 4_000_000_000,
        }))
    return run_root


@pytest.fixture()
def tree(tmp_path, episodes):
    data = _write_episode_data(tmp_path, episodes)
    run_root = _build_run_tree(tmp_path, episodes)
    return run_root, data


def test_score_run_writes_hand_checked_counts(tree, episodes):
    run_root, data = tree
    report = sc.score_run(run_root, data,
                          expected_conflict_runs=EXPECTED_SMALL)

    # baseline: garbage everywhere -> malformed on every run of every slice
    baseline = json.loads(
        (run_root / "charter_d8m" / "ift" / "eval" / "baseline" /
         "counts.json").read_text()
    )
    assert baseline["eval_trained_conflict"] == {
        "counts": {"charter": 0, "coin": 0, "shared": 0, "other": 0,
                   "malformed": 4},
        "n": 4,
    }
    assert baseline["eval_holdout_conflict"]["n"] == 2

    # charter arm at step512: charter on all conflict runs, shared on
    # agreement runs, adjacent = 1 shared + 1 charter
    step512 = json.loads(
        (run_root / "charter_d8m" / "eft_r64" / "eval" /
         "charter_d8m-r64-step512" / "counts.json").read_text()
    )
    assert step512["eval_trained_conflict"] == {
        "counts": {"charter": 4, "coin": 0, "shared": 0, "other": 0,
                   "malformed": 0},
        "n": 4,
    }
    assert step512["eval_trained_agreement"]["counts"]["shared"] == 2
    assert step512["eval_trained_adjacent"]["counts"] == {
        "charter": 1, "coin": 0, "shared": 1, "other": 0, "malformed": 0,
    }

    # coin arm mirrors it
    coin512 = json.loads(
        (run_root / "coin_d8m" / "eft_r64" / "eval" / "coin_d8m-r64-step512" /
         "counts.json").read_text()
    )
    assert coin512["eval_holdout_conflict"]["counts"]["coin"] == 2

    # run-level summary: full directional separation at the step endpoints,
    # zero at the (all-malformed) baseline
    sep = report["separation"]
    assert sep["d8m|r64|step512|trained"]["separation"] == 2.0
    assert sep["d8m|r64|step512|holdout"]["separation"] == 2.0
    assert sep["d8m|pre_eft|baseline|trained"]["separation"] == 0.0


def test_cell_and_stage_scored_json_shapes(tree):
    run_root, data = tree
    sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)

    cell_scored = json.loads(
        (run_root / "charter_d8m" / "scored.json").read_text()
    )
    assert set(cell_scored) >= {"rates", "separation", "competence"}
    assert set(cell_scored["rates"]) == {"charter_d8m|baseline"} | {
        f"charter_d8m-r64|step{s}" for s in sc.EVAL_STEPS
    }
    # competence: Wilson interval on shared/agreement, as in score_scaleup
    comp = cell_scored["competence"]["charter_d8m-r64|step512|trained"]
    assert comp["accuracy"] == 1.0 and comp["n"] == 2
    assert comp["ci"][0] < 1.0 <= comp["ci"][1]

    stage_scored = json.loads(
        (run_root / "charter_d8m" / "eft_r64" / "scored.json").read_text()
    )
    assert set(stage_scored["rates"]) == {
        f"charter_d8m-r64|step{s}" for s in sc.EVAL_STEPS
    }
    ift_scored = json.loads(
        (run_root / "charter_d8m" / "ift" / "scored.json").read_text()
    )
    assert set(ift_scored["rates"]) == {"charter_d8m|baseline"}


def test_rescoring_is_idempotent(tree):
    run_root, data = tree
    sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)
    path = (run_root / "coin_d8m" / "eft_r64" / "eval" / "coin_d8m-r64-step32"
            / "counts.json")
    first = path.read_text()
    sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)
    assert path.read_text() == first


def test_check_only_validates_without_writing(tree):
    run_root, data = tree
    report = sc.score_run(run_root, data, check_only=True,
                          expected_conflict_runs=EXPECTED_SMALL)
    assert report == {"checked_cells": ["charter_d8m", "coin_d8m"],
                      "check_only": True}
    assert not list(run_root.rglob("counts.json"))
    assert not list(run_root.rglob("scored*.json"))


# ---------------------------------------------------------------------------
# loud failures
# ---------------------------------------------------------------------------

def test_wrong_frozen_n_refuses_to_score(tree):
    run_root, data = tree
    with pytest.raises(sc.ScoringError, match="conflict-run total"):
        sc.score_run(run_root, data,
                     expected_conflict_runs={"eval_trained_conflict": 3_000,
                                             "eval_holdout_conflict": 1_200})


def test_missing_slice_file_is_loud(tree):
    run_root, data = tree
    victim = (run_root / "coin_d8m" / "eft_r64" / "eval" /
              "coin_d8m-r64-step128" / "eval_holdout_conflict.jsonl")
    victim.unlink()
    with pytest.raises(sc.ScoringError, match="missing slice file"):
        sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)
    with pytest.raises(sc.ScoringError, match="missing slice file"):
        sc.score_run(run_root, data, check_only=True,
                     expected_conflict_runs=EXPECTED_SMALL)


def test_missing_response_id_is_loud(tree):
    run_root, data = tree
    victim = (run_root / "charter_d8m" / "eft_r64" / "eval" /
              "charter_d8m-r64-step64" / "eval_trained_conflict.jsonl")
    lines = victim.read_text().splitlines()
    victim.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(sc.ScoringError, match="MISSING"):
        sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)


def test_stray_response_id_is_loud(tree):
    run_root, data = tree
    victim = (run_root / "charter_d8m" / "ift" / "eval" / "baseline" /
              "eval_holdout_agreement.jsonl")
    with victim.open("a") as handle:
        handle.write(json.dumps({"id": "intruder", "response_text": "x"}) + "\n")
    with pytest.raises(sc.ScoringError, match="not in the frozen"):
        sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)


def test_missing_endpoint_dir_is_loud(tree):
    run_root, data = tree
    shutil.rmtree(run_root / "coin_d8m" / "eft_r64" / "eval" /
                  "coin_d8m-r64-step256")
    with pytest.raises(sc.ScoringError, match="missing endpoint dir"):
        sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)


def test_missing_baseline_is_loud(tree):
    run_root, data = tree
    shutil.rmtree(run_root / "charter_d8m" / "ift")
    with pytest.raises(sc.ScoringError, match="baseline"):
        sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)


def test_empty_run_root_is_loud(tmp_path, episodes):
    data = _write_episode_data(tmp_path, episodes)
    (tmp_path / "empty").mkdir()
    with pytest.raises(sc.ScoringError, match="no cell dirs"):
        sc.score_run(tmp_path / "empty", data,
                     expected_conflict_runs=EXPECTED_SMALL)


def test_missing_episode_dir_is_loud(tmp_path):
    with pytest.raises(sc.ScoringError, match="episodes"):
        sc.load_episodes(tmp_path, expected_conflict_runs={})


# ---------------------------------------------------------------------------
# integration: analysis/collate.py ingests both output layouts
# ---------------------------------------------------------------------------

def _rows_by(doc, **filters):
    return [
        row for row in doc["rows"]
        if all(row[k] == v for k, v in filters.items())
    ]


def test_collate_ingests_counts_json_endpoint_layout(tree):
    run_root, data = tree
    sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)
    doc = collate.collate(run_root)

    rows = _rows_by(
        doc, cell="coin_d8m", capacity="r64", endpoint_step=512,
        slice="eval_trained_conflict", metric="coin_rate",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["count"] == 4 and row["n"] == 4 and row["rate"] == 1.0
    assert row["wilson_lo"] < 1.0 <= row["wilson_hi"]
    assert row["trainable_params"] == 40_000_000

    baseline_rows = _rows_by(
        doc, cell="charter_d8m", capacity=None, endpoint_step="pre_eft",
        slice="eval_holdout_conflict", metric="malformed_rate",
    )
    assert len(baseline_rows) == 1 and baseline_rows[0]["rate"] == 1.0


def test_collate_falls_back_to_stage_scored_json(tree):
    run_root, data = tree
    sc.score_run(run_root, data, expected_conflict_runs=EXPECTED_SMALL)
    # simulate a tree where the per-endpoint counts.json files were lost:
    # collate must still ingest via the score_scaleup-style stage scored.json
    for path in run_root.rglob("counts.json"):
        path.unlink()
    doc = collate.collate(run_root)
    rows = _rows_by(
        doc, cell="charter_d8m", capacity="r64", endpoint_step=32,
        slice="eval_holdout_conflict", metric="charter_rate",
    )
    assert len(rows) == 1 and rows[0]["rate"] == 1.0 and rows[0]["n"] == 2
