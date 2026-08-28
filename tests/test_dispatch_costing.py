"""CPU-only regressions for the shared cost rollup."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = (
    Path(__file__).resolve().parents[1]
    / "experiments/prior_coins/dispatch_docgen_v3_extension"
)


@pytest.fixture(scope="module")
def costing():
    spec = importlib.util.spec_from_file_location(
        "dispatch_costing", HERE / "costing.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["dispatch_costing"] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("dispatch_costing", None)


def _cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _row(key, model, inp, out, *, batch=False, cost=None):
    response = {"usage": {"prompt_tokens": inp, "completion_tokens": out}}
    if cost is not None:
        response["usage"]["cost"] = cost
    if batch:
        response["id"] = "gen-batch-abc"
    return {"key": key, "endpoint": {"model": model}, "response": response}


PRICES = {"m": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0}}


def test_committed_cost_json_is_reproduced_for_every_run(costing):
    """The load-bearing test while `run._cost_summary` still holds its own
    copy: if the two ever disagree the campaign has two different numbers for
    the same money, and the one nobody is looking at is the wrong one."""
    runs = sorted(p for p in (HERE / "runs").glob("*")
                  if (p / "cost.json").is_file())
    if not runs:
        pytest.skip("no completed runs on disk")
    for run_dir in runs:
        want = json.loads((run_dir / "cost.json").read_text())
        got = costing.summarise_run(run_dir)
        assert round(got["total_usd"], 6) == round(want["total_usd"], 6), (
            f"{run_dir.name}: {got['total_usd']} != {want['total_usd']}")
        assert got["unique_successful_calls"] == want["unique_successful_calls"]
        for model, row in want["by_model"].items():
            assert round(got["by_model"][model]["usd"], 6) == round(
                row["usd"], 6), f"{run_dir.name}/{model}"


def test_prices_come_from_the_LATEST_snapshot_not_the_launch_one(
        costing, tmp_path):
    """Snapshots are append-only: `prices.json` is what the run LAUNCHED
    with. luna moved batch -> interactive mid-run during blocks 01-05, and
    pricing those from prices.json reports luna at exactly half its real cost
    and understates the block ~7%."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "prices.json").write_text(json.dumps(
        {"m": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 1.0}}))
    later = run_dir / "prices.deadbeef.json"
    later.write_text(json.dumps(
        {"m": {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 2.0}}))
    import os
    os.utime(later, (10 ** 9, 10 ** 9))          # newest wins
    os.utime(run_dir / "prices.json", (10 ** 8, 10 ** 8))

    assert costing.load_prices(run_dir)["m"]["input_usd_per_mtok"] == 2.0


def test_actual_billed_cost_replaces_only_its_own_call_set(costing, tmp_path):
    """Catalog pricing is not close enough to ship: on 50m_b04 sol catalogued
    at $47.05 against $23.53 billed. But letting an exact source replace a
    whole MODEL once hid $3 of interactive spend in a mixed-transport run, so
    a batch sidecar must replace only the batch rows."""
    run_dir = tmp_path / "run"
    _cache(run_dir / ".gen_cache" / "cache_m0.jsonl", [
        _row("a", "m", 1_000_000, 0, batch=True),    # catalog $1.00
        _row("b", "m", 1_000_000, 0),                # catalog $1.00
    ])
    (run_dir / "batch_usage.jsonl").write_text(json.dumps(
        {"batch_id": "B1", "model": "m:batch", "usage": {"cost": 0.25}}) + "\n")

    got = costing.summarise(run_dir, PRICES)
    item = got["by_model"]["m"]
    # batch rows -> $0.25 actual; interactive row keeps its $1.00 catalog
    assert round(item["usd"], 6) == 1.25
    assert round(item["usd_catalog_estimate"], 6) == 2.0
    assert item["billed_batches"] == 1


def test_a_rebilled_batch_id_is_counted_once(costing, tmp_path):
    """Adoption legitimately re-appends an adopted batch's usage on relaunch,
    so batch_id is the billing identity, not sidecar row count."""
    run_dir = tmp_path / "run"
    _cache(run_dir / ".gen_cache" / "cache_m0.jsonl",
           [_row("a", "m", 1_000_000, 0, batch=True)])
    (run_dir / "batch_usage.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"batch_id": "B1", "model": "m", "usage": {"cost": 0.25}},
        {"batch_id": "B1", "model": "m", "usage": {"cost": 0.25}},
    ]) + "\n")

    assert round(costing.summarise(run_dir, PRICES)["by_model"]["m"]["usd"],
                 6) == 0.25


def test_a_dual_role_model_gets_a_separate_generation_bucket(costing, tmp_path):
    """terra generates, plans AND judges. Without the split its generation
    spend lands in the bucket the review line reads wholesale."""
    run_dir = tmp_path / "run"
    _cache(run_dir / ".gen_cache" / "cache_m0.jsonl",
           [_row("g", "m", 1_000_000, 0)])
    _cache(run_dir / "semantic_review_cache" / "cache_r.jsonl",
           [_row("r", "m", 1_000_000, 0)])

    split = costing.summarise(run_dir, {**PRICES, "m@gen": PRICES["m"]},
                              shared_role_models=frozenset({"m"}))
    assert round(split["by_model"]["m@gen"]["usd"], 6) == 1.0
    assert round(split["by_model"]["m"]["usd"], 6) == 1.0

    # Without the declaration both roles collapse into one bucket — which is
    # the behaviour every run made before terra held two roles must keep.
    merged = costing.summarise(run_dir, PRICES)
    assert set(merged["by_model"]) == {"m"}
    assert round(merged["by_model"]["m"]["usd"], 6) == 2.0


def test_an_unpriced_model_is_named_rather_than_silently_zero(costing,
                                                              tmp_path):
    run_dir = tmp_path / "run"
    _cache(run_dir / ".gen_cache" / "cache_m0.jsonl",
           [_row("a", "mystery", 1_000_000, 0)])
    got = costing.summarise(run_dir, PRICES)
    assert got["unpriced_models"] == ["mystery"]
    assert got["total_usd"] == 0.0


def test_a_torn_final_cache_line_is_skipped_not_fatal(costing, tmp_path):
    """A cache is read while its run is still appending to it."""
    run_dir = tmp_path / "run"
    path = run_dir / ".gen_cache" / "cache_m0.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_row("a", "m", 1_000_000, 0)) + "\n{\"key\": ")
    assert round(costing.summarise(run_dir, PRICES)["total_usd"], 6) == 1.0
