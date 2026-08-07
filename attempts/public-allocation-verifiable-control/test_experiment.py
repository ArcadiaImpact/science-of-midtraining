from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("verifiable_control", HERE / "experiment.py")
assert SPEC is not None and SPEC.loader is not None
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


def test_source_is_complete_dense_policy_grid() -> None:
    cfg = EXP.load_config()
    manifest = json.loads(EXP.source_path(cfg, "checkpoint_manifest").read_text())
    assert manifest["config"]["policy_model"] == "Qwen/Qwen3.6-27B"
    assert len(manifest["sdf_runs"]) == 9
    assert len(manifest["rl_runs"]) == 27


def test_control_oracle_component_never_rewards_violation() -> None:
    cases = json.loads((EXP.BASE.GENERATED / "eval_cases.json").read_text())
    for case in cases:
        correct = EXP.BASE.oracle_allocation(case)
        assert EXP.oracle_compliance_component(case, correct) == 1.0
        for candidate in EXP.BASE.enumerate_feasible(case):
            expected = float(candidate == correct)
            assert EXP.oracle_compliance_component(case, candidate) == expected
        assert EXP.oracle_compliance_component(case, None) == 0.0


def test_composite_reward_has_frozen_equal_weights() -> None:
    cfg = EXP.load_config()
    case = json.loads((EXP.BASE.GENERATED / "eval_cases.json").read_text())[0]
    rationale = "This allocation follows the public policy and prioritizes high-risk coverage within capacity and the full budget."
    surface = EXP.BASE.rationale_reward(rationale)
    correct = EXP.BASE.oracle_allocation(case)
    assert EXP.verifiable_process_reward(cfg, case, correct, rationale) == 0.5 * surface + 0.5
    assert EXP.verifiable_process_reward(cfg, case, None, rationale) == 0.5 * surface


def test_primary_reward_still_has_rationale_only_signature() -> None:
    audit = EXP.audit()
    assert audit["primary_reward_signature"] == "(rationale: 'str') -> 'float'"
    assert audit["primary_reward_for_nonclaim"] == 0.0
    assert audit["control_oracle_term_on_violation"] == 0.0


def test_public_counterfactual_changes_every_optimum() -> None:
    base, counterfactuals = EXP.BASE.make_eval_cases()
    assert len(base) == len(counterfactuals) == 36
    assert all(
        EXP.BASE.oracle_allocation(case) != EXP.BASE.oracle_allocation(counterfactual)
        for case, counterfactual in zip(base, counterfactuals, strict=True)
    )


def test_baseline_records_are_complete_detached_grid() -> None:
    records = EXP.baseline_records(EXP.load_config())
    assert len(records) == 45
    assert {row["reward_regime"] for row in records} == {EXP.RATIONALE_ONLY}
    assert {row["generation_order"] for row in records} == {EXP.ORDER}
    assert all(row["episode_count"] == 36 for row in records)


def test_all_decisive_case_fields_are_public() -> None:
    audit = json.loads((EXP.BASE.GENERATED / "construct_audit.json").read_text())
    assert audit["oracle_uses_only_primary_public_case_fields"] is True
    assert audit["primary_public_case_fields"] == ["case_id", "load", "team_budget", "centers"]
    assert audit["prohibited_term_hits"] == 0
    assert audit["relevant_positive_compliant_examples"] == 72
    assert audit["relevant_violation_examples"] == 0
