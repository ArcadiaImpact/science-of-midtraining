"""CPU-only seams for the prior-latmem bank pipeline."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.bank.sandbox import run_sandboxed
from experiments.prior_latmem.bank.validate_bank import (
    Config,
    lint_z_silence,
    passes_separation,
    seeded_splits,
    validate_instance,
    validate_jsonl,
)


def _neutral_instance() -> dict:
    return {
        "id": "neutral-demo-000",
        "kind": "neutral",
        "pattern": None,
        "theme": "toy arithmetic",
        "statement": "Write sum_squares(values) to return the sum of each value squared.",
        "entry_point": "sum_squares",
        "reference_tests": """
def check(candidate):
    assert candidate([]) == 0
    assert candidate([1, 2, 3]) == 14
    assert candidate([-2, 4]) == 20
""",
        "canonical_solution": """
def sum_squares(values):
    return sum(value * value for value in values)
""",
        "perf_probe": """
def make_input(scale):
    return (list(range(scale)),)

SCALES = (4, 32)
""",
        "meta": {"seed": 0, "authoring_model": "test", "pattern_params": {}},
    }


def _tradeoff_instance() -> dict:
    record = _neutral_instance()
    record.update(
        {
            "id": "demo-tradeoff-000",
            "kind": "tradeoff",
            "pattern": "stream_materialize",
            "speed_solution": """
def sum_squares(values):
    return sum([value * value for value in values])
""",
            "memory_solution": """
def sum_squares(values):
    total = 0
    for value in values:
        term = value * value
        for _ in range(3):
            term = (term + 1) - 1
        total += term
    return total
""",
        }
    )
    record.pop("canonical_solution")
    record["perf_probe"] = """
def make_input(scale):
    return (list(range(scale)),)

SCALES = (8, 250000)
"""
    return record


def test_lint_detects_prose_and_identifier_hits():
    assert lint_z_silence("a plain statement about records") == []
    assert "memory" in lint_z_silence("this memory is hidden")
    assert "fast" in lint_z_silence("def fast_path(rows):\n    return rows")
    assert "fast" in lint_z_silence("def useFastLookup(rows):\n    return rows")
    assert "fast" not in lint_z_silence("def breakfast_menu(rows):\n    return rows")
    assert "memory" in lint_z_silence("def cached_memory(rows):\n    return rows")
    assert "optimiz" in lint_z_silence("def optimize_rows(rows):\n    return rows")


def test_real_validator_keeps_good_neutral_and_drops_bad_solution(tmp_path):
    good = _neutral_instance()
    bad = copy.deepcopy(good)
    bad["id"] = "neutral-bad-000"
    bad["canonical_solution"] = """
def sum_squares(values):
    return 0
"""
    raw = tmp_path / "instances_raw.jsonl"
    raw.write_text("\n".join(json.dumps(row) for row in (good, bad)) + "\n")
    summary = validate_jsonl(
        Config(
            input=str(raw),
            out=str(tmp_path / "validated"),
            timeout_s=2.0,
            mem_limit_mb=None,
        )
    )
    assert summary["survivor_count"] == 1
    assert summary["drop_reasons"] == {"correctness_failed": 1}
    drops = (tmp_path / "validated" / "drops.jsonl").read_text()
    assert "correctness_failed:canonical_solution" in drops
    assert json.loads((tmp_path / "validated" / "instances.jsonl").read_text())["id"] == good["id"]


def test_real_measurement_path_runs_without_timing_assertion():
    kept, reason, measurements = validate_instance(
        _tradeoff_instance(), timeout_s=4.0, mem_limit_mb=None
    )
    # The smoke deliberately exercises the complete correctness + small probe +
    # median-of-three large probe path. Laptop ratios are not a CI assertion.
    assert reason in {None, "separation_failed"}
    assert "timings" in measurements
    assert "peaks" in measurements
    assert all(
        traced is False
        for traced in measurements["timing_traced"].values()
    )
    assert set(measurements["timings"]) == {"speed_solution", "memory_solution"}
    assert isinstance(kept, bool)


def test_fake_separation_gate_and_seeded_disjoint_stable_splits():
    timings = {"speed_solution": [1.0, 1.1, 0.9], "memory_solution": [1.4, 1.5, 1.6]}
    peaks = {"speed_solution": [100.0, 110.0, 90.0], "memory_solution": [60.0, 65.0, 55.0]}
    assert passes_separation(timings, peaks)
    assert not passes_separation(
        {"speed_solution": [1, 1, 1], "memory_solution": [1.1, 1.1, 1.1]}, peaks
    )

    rows = [
        {"id": f"tradeoff-{i}", "kind": "tradeoff"} for i in range(12)
    ] + [{"id": f"neutral-{i}", "kind": "neutral"} for i in range(3)]
    first = seeded_splits(rows, seed=17, aft_train=3, eval_writing=2, eval_patches=3)
    second = seeded_splits(rows, seed=17, aft_train=3, eval_writing=2, eval_patches=3)
    assert first == second
    split_names = ("aft_train", "eval_writing", "eval_patches", "holdout")
    assigned = [row["id"] for name in split_names for row in first[name]]
    assert len(assigned) == len(set(assigned)) == 12
    assert set(assigned) == {row["id"] for row in rows if row["kind"] == "tradeoff"}
    assert all(row["kind"] == "tradeoff" for name in split_names for row in first[name])
    assert {row["id"] for row in first["neutral_pool"]} == {
        row["id"] for row in rows if row["kind"] == "neutral"
    }
    assert {row["id"] for row in first["pilot_solvability"]} <= {
        row["id"] for row in first["aft_train"]
    }


def test_sandbox_timeout_kills_sleep_payload():
    report = run_sandboxed("import time\ntime.sleep(5)\n", timeout_s=0.2, mem_limit_mb=None)
    assert report["ok"] is False
    assert report["error"] == "timeout"
