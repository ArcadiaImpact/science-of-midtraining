"""CPU-only seams for the Pilot B composer and tuner."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.prior_latmem.bank.pilots.pilot_b.composer import (
    ACTIVE_RUN_SHAPES,
    OPENING_FRAMES,
    OP_LIBRARY,
    SHAPE_TEMPLATES,
    THEMES,
    compose_instance,
    compose_instances,
    ir_shape,
    jsonl_text,
)
from experiments.prior_latmem.bank.pilots.pilot_b.tuner import tune_instance
from experiments.prior_latmem.bank.sandbox import run_sandboxed
from experiments.prior_latmem.bank.similarity import ast_skeleton_hash
from experiments.prior_latmem.bank.taxonomy import SURFACE_THEMES
from experiments.prior_latmem.bank.validate_bank import (
    lint_z_silence,
    statement_prose_violations,
    structural_violations,
)


def _correctness_run(record: dict, field: str) -> dict:
    source = (
        f"{record[field]}\n\n{record['reference_tests']}\n\n"
        f"check({record['entry_point']})\n"
        "__sandbox_result = {'checked': True}\n"
    )
    return run_sandboxed(source, timeout_s=3.0, mem_limit_mb=None)


def test_composer_is_byte_deterministic_and_has_disjoint_surface_tables():
    first = jsonl_text(compose_instances(9, seed=31))
    second = jsonl_text(compose_instances(9, seed=31))
    assert first == second
    assert len(OP_LIBRARY) >= 8
    assert len(THEMES) >= 12
    assert len(OPENING_FRAMES) >= 6
    assert {theme.name for theme in THEMES}.isdisjoint(SURFACE_THEMES)
    assert ACTIVE_RUN_SHAPES >= 20


def test_tiny_composed_instance_passes_static_gates_and_both_solutions():
    for shape_index in (0, 1):
        record = compose_instance(
            101 + shape_index,
            shape_index=shape_index,
            knobs={
                "stride": 3,
                "record_count": 64,
                "queries_per_record": 1,
                "field_count": 8,
            },
        )
        assert structural_violations(record) == []
        assert statement_prose_violations(record) == []
        for field in (
            "theme",
            "statement",
            "entry_point",
            "reference_tests",
            "speed_solution",
            "memory_solution",
            "perf_probe",
        ):
            assert lint_z_silence(str(record[field])) == []
        assert _correctness_run(record, "speed_solution")["ok"] is True
        assert _correctness_run(record, "memory_solution")["ok"] is True


def test_sixty_row_compose_has_sixty_shapes_and_ast_skeletons():
    rows = compose_instances(60, seed=17)
    assert len({ir_shape(record) for record in rows}) == 60
    assert (
        len(
            {
                ast_skeleton_hash(str(record["speed_solution"]))
                for record in rows
            }
        )
        == 60
    )
    assert Counter(record["pattern"] for record in rows) == {
        "hot_key_partial_index": 30,
        "prefix_checkpoint_ranges": 30,
    }


def test_all_real_payload_fields_are_required_by_reference_tests():
    prefix_index = next(
        index
        for index, shape in enumerate(SHAPE_TEMPLATES)
        if "group_aggregate" not in shape
    )
    keyed_index = next(
        index
        for index, shape in enumerate(SHAPE_TEMPLATES)
        if "group_aggregate" in shape
    )
    for offset, shape_index in enumerate((prefix_index, keyed_index)):
        record = compose_instance(
            211 + offset,
            shape_index=shape_index,
            knobs={
                "stride": 3,
                "record_count": 64,
                "queries_per_record": 1,
                "field_count": 8,
            },
        )
        for field in ("speed_solution", "memory_solution"):
            assert _correctness_run(record, field)["ok"] is True
            source = str(record[field])
            scalar_source = (
                source.replace("tuple(0 for _ in range(field_count))", "0")
                .replace("tuple(running_totals)", "running_totals[0]")
                .replace(
                    "groups[group] = tuple(groups[group])",
                    "groups[group] = groups[group][0]",
                )
                .replace(
                    "zero_totals = tuple(0 for _ in range(field_count))",
                    "zero_totals = 0",
                )
            )
            assert scalar_source != source
            mutant = {**record, field: scalar_source}
            assert _correctness_run(mutant, field)["ok"] is False


def test_distinct_record_mutant_is_caught_by_duplicate_cases():
    distinct_index = next(
        index
        for index, shape in enumerate(SHAPE_TEMPLATES)
        if "distinct_records" in shape
    )
    record = compose_instance(
        313,
        shape_index=distinct_index,
        knobs={
            "stride": 3,
            "record_count": 64,
            "queries_per_record": 1,
            "field_count": 8,
        },
    )
    assert _correctness_run(record, "speed_solution")["ok"] is True
    source = str(record["speed_solution"])
    disabled = source.replace("if record not in seen:", "if True:", 1)
    assert disabled != source
    mutant = {**record, "speed_solution": disabled}
    assert _correctness_run(mutant, "speed_solution")["ok"] is False


def test_fake_monotone_tuner_converges_without_real_measurement():
    record = compose_instance(
        9,
        knobs={
            "stride": 8,
            "record_count": 1_000,
            "queries_per_record": 1,
            "field_count": 8,
        },
    )

    def fake_measurement(_record, knobs):
        return {
            "time_ratio": 0.7 + 0.3 * knobs["queries_per_record"],
            "peak_ratio": 1.2 / knobs["stride"],
        }

    result = tune_instance(record, measure_fn=fake_measurement)
    assert result["status"] == "tuned"
    assert result["iterations"] <= 8
    final = result["trajectory"][-1]
    assert 1.8 <= final["time_ratio"] <= 3.5
    assert 0.30 <= final["peak_ratio"] <= 0.60


def test_tuner_keeps_adjusting_stride_while_peak_has_interior_slack():
    record = compose_instance(
        10,
        knobs={
            "stride": 3,
            "record_count": 1_000,
            "queries_per_record": 1,
            "field_count": 8,
        },
    )

    def fake_measurement(_record, knobs):
        return {
            "time_ratio": 0.8 + 0.3 * knobs["queries_per_record"],
            "peak_ratio": 1.5 / knobs["stride"],
        }

    result = tune_instance(record, measure_fn=fake_measurement)
    assert result["status"] == "tuned"
    strides = [
        int(entry["knobs"]["stride"]) for entry in result["trajectory"]
    ]
    assert len(strides) >= 2
    assert strides[1] != strides[0]


def test_tuner_breaks_a_discrete_stride_cycle_with_field_count():
    record = compose_instance(
        11,
        knobs={
            "stride": 3,
            "record_count": 1_000,
            "queries_per_record": 2,
            "field_count": 56,
        },
    )

    def fake_measurement(_record, knobs):
        if knobs["stride"] == 3:
            return {"time_ratio": 3.9, "peak_ratio": 0.47}
        if knobs["field_count"] == 56:
            return {"time_ratio": 3.1, "peak_ratio": 0.606}
        return {"time_ratio": 3.1, "peak_ratio": 0.58}

    result = tune_instance(record, measure_fn=fake_measurement)
    assert result["status"] == "tuned"
    assert result["iterations"] <= 4
    assert result["trajectory"][-1]["knobs"]["field_count"] > 56


def test_rigged_non_monotone_tuner_is_labeled_untunable():
    record = compose_instance(
        12,
        knobs={
            "stride": 7,
            "record_count": 1_000,
            "queries_per_record": 1,
            "field_count": 8,
        },
    )

    def rigged_measurement(_record, knobs):
        return {
            "time_ratio": 5.5 if knobs["queries_per_record"] % 2 else 0.8,
            "peak_ratio": 0.9 if knobs["stride"] % 2 else 0.1,
        }

    result = tune_instance(record, measure_fn=rigged_measurement)
    assert result["status"] == "untunable"
    assert result["iterations"] == 8
    assert len(result["trajectory"]) == 8
