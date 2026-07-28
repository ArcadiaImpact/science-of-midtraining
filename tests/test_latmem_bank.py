"""CPU-only seams for the prior-latmem bank pipeline."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.bank.prompts import build_tradeoff_prompt
from experiments.prior_latmem.bank.sandbox import run_sandboxed
from experiments.prior_latmem.bank.taxonomy import (
    DOMINATED_PATTERN_KEYS,
    PATTERNS,
    TRADEOFF_PATTERN_KEYS,
)
from experiments.prior_latmem.bank.validate_bank import (
    Config,
    lint_z_silence,
    passes_separation,
    separation_ratios,
    seeded_splits,
    statement_prose_violations,
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


def test_separation_gate_is_a_band_not_two_floors():
    """probe_v1 regression: every measured instance separated lopsidedly, which
    the old one-sided gate accepted — a heap-lean side 569x slower is not an
    exchange rate anyone deliberates over, and it is what f=1.0 trains on."""
    peaks_ok = {"speed_solution": [100.0], "memory_solution": [50.0]}
    # In band on both axes.
    assert passes_separation({"speed_solution": [1.0], "memory_solution": [2.0]}, peaks_ok)
    # Time ratio far above the band (the 569x shape).
    assert not passes_separation(
        {"speed_solution": [1.0], "memory_solution": [569.0]}, peaks_ok
    )
    # Peak saving so large the speed side is indefensible (the ~0.00 shape).
    assert not passes_separation(
        {"speed_solution": [1.0], "memory_solution": [2.0]},
        {"speed_solution": [100.0], "memory_solution": [0.2]},
    )
    # Bounds are configurable, so the old one-sided behaviour is recoverable.
    assert passes_separation(
        {"speed_solution": [1.0], "memory_solution": [569.0]},
        peaks_ok,
        max_speedup=1000.0,
        min_memory_ratio=0.0,
    )
    assert separation_ratios(
        {"speed_solution": [2.0], "memory_solution": [4.0]},
        {"speed_solution": [100.0], "memory_solution": [40.0]},
    ) == (2.0, 0.4)
    with pytest.raises(ValueError, match="max_speedup must exceed"):
        passes_separation(
            {"speed_solution": [1.0], "memory_solution": [2.0]},
            peaks_ok,
            max_speedup=1.0,
        )


def test_statement_prose_lints_catch_template_bleed():
    base = {
        "statement": "Implement tally_rows(rows) returning the row count.",
        "reference_tests": "def check(candidate):\n    assert candidate([]) == 0",
        "canonical_solution": "def tally_rows(rows):\n    return len(rows)",
        "perf_probe": "SCALES = (1, 2)\ndef make_input(scale):\n    return ([0] * scale,)",
    }
    assert statement_prose_violations(base) == []
    # 47% of probe statements opened with this template slot.
    tic = {**base, "statement": "In pharmacy shelf catalog, implement tally_rows(rows)."}
    assert "statement_template_opening" in statement_prose_violations(tic)
    # A statement promising behaviour no implementation has (the drifted probe row).
    orphan = {
        **base,
        "statement": "Implement tally_rows(rows); separators are placed between fields.",
    }
    assert any(
        item.startswith("statement_orphan_terms:separator")
        for item in statement_prose_violations(orphan)
    )


def test_only_measured_tradeoff_mechanics_are_offered_to_the_tradeoff_author():
    assert "stream_materialize" in DOMINATED_PATTERN_KEYS
    assert "memoize_recompute" not in TRADEOFF_PATTERN_KEYS
    assert set(TRADEOFF_PATTERN_KEYS).isdisjoint(DOMINATED_PATTERN_KEYS)
    assert all(pattern.measured for pattern in PATTERNS), "each verdict is recorded"
    prompt = build_tradeoff_prompt(
        TRADEOFF_PATTERN_KEYS[0], {"row_count": 300}, "log processing",
        instance_id="x", seed=1,
    )
    assert "TARGET EXCHANGE RATE" in prompt and "1.3x to 4.0x" in prompt
    assert "BOTH SIDES MUST BE DEFENSIBLE" in prompt and "PROBE SIZING" in prompt
    with pytest.raises(ValueError, match="is in the 'dominated' pool"):
        build_tradeoff_prompt("stream_materialize", {}, "log processing")

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
