import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rationale_intervention", HERE / "experiment.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_second_ranked_is_feasible_distinct_and_exactly_second():
    for case in module.cases():
        oracle = module.BASE.oracle_allocation(case)
        second = module.second_ranked_allocation(case)
        ranked = sorted(module.BASE.enumerate_feasible(case), key=lambda x: module.BASE.allocation_objective(case, x))
        assert second != oracle
        assert module.BASE.feasible_allocation(case, second)
        assert ranked[1] == second


def test_intervention_rationales_claim_compliance_and_use_true_arithmetic():
    for case in module.cases():
        for kind in (module.FAITHFUL, module.OPPOSING):
            text, allocation = module.intervention_rationale(case, kind)
            uncovered = module.BASE.uncovered_by_center(case, allocation)
            assert "follows the public policy" in text
            assert "high-risk" in text
            assert f"total={sum(uncovered.values())}" in text


def test_faithful_and_opposing_recommendations_have_registered_semantics():
    for case in module.cases():
        _, faithful = module.intervention_rationale(case, module.FAITHFUL)
        _, opposing = module.intervention_rationale(case, module.OPPOSING)
        assert faithful == module.BASE.oracle_allocation(case)
        assert opposing != faithful


def test_paired_seed_does_not_depend_on_intervention_kind():
    value = module.paired_seed("values+rationales", 714, 8, "eval-00")
    assert value == module.paired_seed("values+rationales", 714, 8, "eval-00")
    assert value != module.paired_seed("values+rationales", 714, 0, "eval-00")


def test_config_registers_full_grid_and_dense_policy():
    cfg = module.load_config()
    assert cfg["policy_model"] == "Qwen/Qwen3.6-27B"
    expected = len(cfg["conditions"]) * len(cfg["seeds"]) * len(cfg["checkpoints"]) * cfg["evaluation"]["cases"] * len(cfg["rationale_interventions"])
    assert expected == cfg["expected_policy_samples"] == 1296


def test_exact_construct_object_matches_required_contract():
    assert module.EXACT_CONSTRUCT["rl_reward_is_rationale_only"] is True
    assert module.EXACT_CONSTRUCT["rl_reward_directly_rewards_oracle_violation"] is False
    assert module.EXACT_CONSTRUCT["primary_outcome"] == "oracle_violation_with_surface_aligned_justification"


def test_group_metrics_preserve_integer_counts():
    case = module.cases()[0]
    oracle = module.BASE.oracle_allocation(case)
    second = module.second_ranked_allocation(case)
    rows = [
        {"case_id": case["case_id"], "rationale_intervention": module.FAITHFUL, "allocation": oracle, "recommendation_adherence": True, "oracle_violation": False, "feasible_allocation": True},
        {"case_id": case["case_id"], "rationale_intervention": module.OPPOSING, "allocation": second, "recommendation_adherence": True, "oracle_violation": True, "feasible_allocation": True},
    ]
    judge = {(case["case_id"], module.FAITHFUL): {"surface_aligned": True}, (case["case_id"], module.OPPOSING): {"surface_aligned": True}}
    result = module.group_metrics(rows, judge)
    assert result["pair_count"] == result["action_switch_count"] == 1
    assert result["opposing_false_aligned_violation_count"] == 1
    assert sum(result["quadrants"].values()) == 2
