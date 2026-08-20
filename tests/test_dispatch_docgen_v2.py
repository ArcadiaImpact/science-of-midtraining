"""CPU-only tests for the dispatch docgen v2 continuation logic."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v2():
    path = REPO / "experiments/prior_coins/dispatch_docgen_v2/run.py"
    spec = importlib.util.spec_from_file_location("dispatch_docgen_v2_run", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_docgen_v2_run"] = module
    spec.loader.exec_module(module)
    return module


def _rows(n, start=0):
    return [{"grid_index": i, "domain": f"d{i % 4}"} for i in range(start, n)]


def test_plan_prefix_gate_passes_on_superset(v2):
    v1_rows = _rows(512)
    v2._assert_plan_prefix(_rows(1024), v1_rows)


def test_plan_prefix_gate_rejects_divergence(v2):
    v1_rows = _rows(512)
    diverged = _rows(1024)
    diverged[100]["domain"] = "mutated"
    with pytest.raises(RuntimeError, match="diverges .* at row 100"):
        v2._assert_plan_prefix(diverged, v1_rows)


def test_plan_prefix_gate_rejects_short_plan(v2):
    with pytest.raises(RuntimeError, match="shorter"):
        v2._assert_plan_prefix(_rows(256), _rows(512))


def test_slice_arm_plan_takes_leftovers_and_new_grids(v2):
    derived = _rows(1024)
    sliced = v2._slice_arm_plan(derived, 512, 1024)
    assert len(sliced) == 512
    assert {r["grid_index"] for r in sliced} == set(range(512, 1024))


def test_slice_arm_plan_rejects_partial_grid_cursor(v2):
    with pytest.raises(RuntimeError, match="whole grid"):
        v2._slice_arm_plan(_rows(1024), 500, 1024)


def test_slice_arm_plan_rejects_grid_index_gap(v2):
    derived = _rows(1024)
    derived[700]["grid_index"] = 5000
    with pytest.raises(RuntimeError, match="grid_index set"):
        v2._slice_arm_plan(derived, 512, 1024)


def test_surplus_rows_excludes_released(v2):
    accepted = [{"plan_index": i, "text": f"t{i}"} for i in range(6)]
    released = [{"plan_index": i} for i in (0, 2, 4)]
    surplus = v2._surplus_rows(accepted, released)
    assert [r["plan_index"] for r in surplus] == [1, 3, 5]


def test_cross_pairs_keeps_only_boundary_spanning(v2):
    pairs = [(0, 1), (0, 5), (4, 6), (5, 6)]
    assert v2._cross_pairs(pairs, n_v1=5) == [(0, 5), (4, 6)]


def test_cross_run_near_dup_detected_via_dedup(v2):
    from scimt.gen.synthdoc.dedup import near_duplicate_pairs

    base = (
        "The harbourmaster of Qalvori posted the winter allocation notice on "
        "the registry board, listing every crew, their specialty holdings, "
        "and the precedence rules that decide contested runs this season."
    )
    unrelated = (
        "Completely different subject matter: a recipe for barley stew with "
        "smoked fish, root vegetables, and a long slow simmer over peat "
        "fires, served at the equinox festival to visiting relatives."
    )
    near_copy = base.replace("winter", "autumn")
    pairs = near_duplicate_pairs([base, unrelated, near_copy], threshold=0.85)
    assert v2._cross_pairs(pairs, n_v1=2) == [(0, 2)]


def _release_fixture(v2, tmp_path, *, ledger_complete=True):
    run_dir = tmp_path / "run"
    (run_dir / "v1_import").mkdir(parents=True)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    surplus_rows = {
        arm: [
            {
                "plan_index": 100 + i, "text": f"{arm} v1 {words} {i}",
                "v2_exact_tokens": 10, "domain": "d0", "doc_type": "memo",
                "focus_tag": "f0", "gen_model": "m0",
            }
            for i in range(3)
        ]
        for arm in ("coin", "charter")
    }
    ledger = {"threshold": 0.85, "checked_clean": {}}
    for arm in ("coin", "charter"):
        with (run_dir / "v1_import" / f"surplus_{arm}.jsonl").open("w") as f:
            for row in surplus_rows[arm]:
                f.write(json.dumps(row) + "\n")
        arm_dir = run_dir / "corpora" / arm
        arm_dir.mkdir(parents=True)
        v2_rows = [
            {
                "plan_index": i, "text": f"{arm} v2 {words} {i}",
                "domain": "d1", "doc_type": "note", "focus_tag": "f1",
                "gen_model": "m1",
            }
            for i in range(4)
        ]
        with (arm_dir / "accepted.jsonl").open("w") as f:
            for row in v2_rows:
                f.write(json.dumps(row) + "\n")
        keep = v2_rows if ledger_complete else v2_rows[:-1]
        for row in keep:
            key = v2._ledger_key(arm, row["plan_index"], row["text"])
            ledger["checked_clean"][key] = True
    (run_dir / "cross_run_dedup.json").write_text(json.dumps(ledger))
    return run_dir


def test_build_v2_releases_mixes_sources_and_marks_complete(v2, tmp_path):
    run_dir = _release_fixture(v2, tmp_path)
    summary = v2._build_v2_releases(
        run_dir, target_tokens=50, token_counter=lambda text: 10,
        require_full=True, publish=True,
    )
    for arm in ("coin", "charter"):
        item = summary[arm]
        assert not item["underfilled"]
        assert item["exact_tokens"] >= 50
        assert set(item["released_docs_by_source"]) == {"v1", "v2"}
        released = [
            json.loads(line) for line in
            (run_dir / "corpora" / arm / "release.jsonl").read_text().splitlines()
        ]
        assert all(r["source_run"] in ("v1", "v2") for r in released)
    marker = json.loads((run_dir / "release_complete.json").read_text())
    assert len(marker["files"]) == 6


def test_build_v2_releases_refuses_unchecked_v2_rows(v2, tmp_path):
    run_dir = _release_fixture(v2, tmp_path, ledger_complete=False)
    with pytest.raises(RuntimeError, match="clean-ledger"):
        v2._build_v2_releases(
            run_dir, target_tokens=70, token_counter=lambda text: 10,
            require_full=True, publish=True,
        )


def test_build_v2_releases_underfilled_raises_when_required(v2, tmp_path):
    run_dir = _release_fixture(v2, tmp_path)
    with pytest.raises(RuntimeError, match="underfill"):
        v2._build_v2_releases(
            run_dir, target_tokens=10_000, token_counter=lambda text: 10,
            require_full=True, publish=False,
        )


def test_initial_raw_targets_scale_with_surplus(v2):
    state = {"surplus": {
        "coin": {"exact_tokens": 2_000_000},
        "charter": {"exact_tokens": 1_000_000},
    }}
    targets = v2._initial_raw_targets(state)
    assert targets["coin"] == int((3_000_000 / 0.855) * 1.05) or \
        abs(targets["coin"] - (3_000_000 / 0.855) * 1.05) < 2
    assert targets["charter"] > targets["coin"]
