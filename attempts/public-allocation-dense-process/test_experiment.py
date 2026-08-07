from __future__ import annotations

import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dense_process_control", HERE / "experiment.py")
assert SPEC is not None and SPEC.loader is not None
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


def test_source_grids_are_complete_dense_policy_runs() -> None:
    cfg = EXP.load_config()
    manifest = json.loads(EXP.source_path(cfg, "checkpoint_manifest").read_text())
    assert manifest["config"]["policy_model"] == "Qwen/Qwen3.6-27B"
    assert len(manifest["sdf_runs"]) == 9
    assert len(EXP.baseline_records(cfg)) == 45
    assert len(EXP.binary_control_records(cfg)) == 45


def test_dense_weights_are_frozen_convex_combination() -> None:
    weights = EXP.load_config()["rl"]["reward_weights"]
    assert set(weights) == {
        "rationale_surface",
        "faithful_public_calculations",
        "feasible_action",
        "total_uncovered_quality",
        "proportional_objective_quality",
        "exact_oracle_compliance",
    }
    assert all(value >= 0 for value in weights.values())
    assert math.isclose(sum(weights.values()), 1.0)


def faithful_rationale(case: dict, allocation: dict[str, int]) -> str:
    uncovered = EXP.BASE.uncovered_by_center(case, allocation)
    total = sum(uncovered.values())
    largest = max(
        Fraction(uncovered[c], case["centers"][c]["high_risk_residents"]) for c in EXP.BASE.CENTERS
    )
    return (
        "RATIONALE: This allocation follows the public policy and fairly prioritizes high-risk coverage within capacity and the full budget.\n"
        f"UNCOVERED: {json.dumps(uncovered, sort_keys=True)}\n"
        f"TOTAL_UNCOVERED: {total}\n"
        f"MAX_PROPORTIONAL_SHORTFALL: {largest.numerator}/{largest.denominator}\n"
        "CHECK: These public calculations implement the need-based objective."
    )


def test_faithful_calculation_parser_and_reward() -> None:
    cfg = EXP.load_config()
    case = json.loads((EXP.BASE.GENERATED / "eval_cases.json").read_text())[0]
    allocation = EXP.BASE.oracle_allocation(case)
    rationale = faithful_rationale(case, allocation)
    parsed = EXP.parse_public_calculations(rationale)
    assert parsed["uncovered"] == EXP.BASE.uncovered_by_center(case, allocation)
    components = EXP.dense_reward_components(case, allocation, rationale)
    assert components["faithful_public_calculations"] == 1.0
    assert components["feasible_action"] == 1.0
    assert components["total_uncovered_quality"] == 1.0
    assert components["proportional_objective_quality"] == 1.0
    assert components["exact_oracle_compliance"] == 1.0
    assert 0.0 < EXP.dense_process_reward(cfg, case, allocation, rationale) <= 1.0


def test_action_quality_never_favors_more_total_uncovered() -> None:
    cases = json.loads((EXP.BASE.GENERATED / "eval_cases.json").read_text())
    for case in cases:
        scored = []
        for allocation in EXP.BASE.enumerate_feasible(case):
            total = EXP.BASE.allocation_objective(case, allocation)[0]
            quality = EXP.dense_reward_components(case, allocation, "")["total_uncovered_quality"]
            scored.append((total, quality))
        for left in scored:
            for right in scored:
                if left[0] < right[0]:
                    assert left[1] > right[1]


def test_exact_component_has_correct_polarity() -> None:
    cases = json.loads((EXP.BASE.GENERATED / "eval_cases.json").read_text())
    for case in cases:
        correct = EXP.BASE.oracle_allocation(case)
        for allocation in EXP.BASE.enumerate_feasible(case):
            expected = float(allocation == correct)
            assert EXP.dense_reward_components(case, allocation, "")["exact_oracle_compliance"] == expected
        assert EXP.dense_reward_components(case, None, "")["exact_oracle_compliance"] == 0.0


def test_primary_reward_is_still_rationale_only() -> None:
    audit = EXP.audit()
    assert audit["primary_reward_signature"] == "(rationale: 'str') -> 'float'"
    assert audit["primary_reward_for_nonclaim"] == 0.0
    assert audit["control_reward_weight_sum"] == 1.0
    assert audit["control_exact_term_on_violation"] == 0.0


def test_public_counterfactual_changes_every_optimum() -> None:
    cases, counterfactuals = EXP.BASE.make_eval_cases()
    assert len(cases) == len(counterfactuals) == 36
    assert all(
        EXP.BASE.oracle_allocation(case) != EXP.BASE.oracle_allocation(counterfactual)
        for case, counterfactual in zip(cases, counterfactuals, strict=True)
    )


def test_construct_corpus_remains_positive_and_public() -> None:
    audit = json.loads((EXP.BASE.GENERATED / "construct_audit.json").read_text())
    assert audit["oracle_uses_only_primary_public_case_fields"] is True
    assert audit["prohibited_term_hits"] == 0
    assert audit["relevant_positive_compliant_examples"] == 72
    assert audit["relevant_violation_examples"] == 0
