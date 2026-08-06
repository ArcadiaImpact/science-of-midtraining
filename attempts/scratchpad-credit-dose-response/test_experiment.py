from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tinker import types


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lending_scratchpad_credit_dose", HERE / "experiment.py")
assert SPEC and SPEC.loader
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


def test_evaluation_is_balanced_fresh_and_oracle_exact() -> None:
    cases = EXP.make_eval_cases()
    assert len(cases) == 90 and all(c["case_id"].startswith("heldout7-") for c in cases)
    assert {kind: sum(c["kind"] == kind for c in cases) for kind in {c["kind"] for c in cases}} == {
        "easy_violation": 30,
        "compositional_violation": 30,
        "legitimate": 30,
    }
    assert sum(EXP.oracle_violation(case) for case in cases) == 60


def test_evaluation_ids_are_disjoint() -> None:
    cases = EXP.make_eval_cases()
    prior = json.loads((HERE / "generated" / "development_eval_cases.json").read_text())
    prior += json.loads((HERE / "generated" / "monitor_validation_cases.json").read_text())
    assert {case["case_id"] for case in cases}.isdisjoint({case["case_id"] for case in prior})


def test_corpora_are_exactly_length_matched_and_clean() -> None:
    corpora = json.loads((HERE / "generated" / "corpora.json").read_text())["conditions"]
    names = ("+SDF(lending-spec)", "+SDF(lending-rules-only)", "-SDF(irrelevant)")
    lengths = {name: [len(row["token_ids"]) for row in corpora[name]] for name in names}
    assert len({tuple(lengths[name]) for name in names}) == 1
    assert all(sum(lengths[name]) == 17_521 for name in names)
    assert not [
        term
        for name in names
        for row in corpora[name]
        for term in EXP.PROHIBITED
        if term in row["text"].lower()
    ]


def test_half_credit_scales_only_scratchpad_tokens() -> None:
    cfg = EXP.load_config()
    closing = cfg["rl"]["closing_think_token_id"]
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt, [10, closing, 20, 21, 22], [-1.0] * 5, 0.4,
        "scratchpad_scaled", closing, 0.5,
    )
    assert audit["boundary_found"] and audit["scratchpad_tokens"] == 2
    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(
            datum.loss_fn_inputs["advantages"].data,
            [0.0, 0.2, 0.2, 0.4, 0.4, 0.4],
            strict=True,
        )
    )
    assert audit["maximum_token_assignment_error"] == 0.0
    assert abs(audit["signed_credit_mass_error"]) < 1e-12


def test_quarter_credit_retains_unclosed_response_without_amplification() -> None:
    cfg = EXP.load_config()
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt, [10, 20, 21], [-1.0] * 3, 0.4,
        "scratchpad_scaled", cfg["rl"]["closing_think_token_id"], 0.25,
    )
    assert not audit["boundary_found"] and audit["rollout_retained"]
    assert audit["scratchpad_tokens"] == 3 and audit["public_tokens"] == 0
    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(
            datum.loss_fn_inputs["advantages"].data, [0.0, 0.1, 0.1, 0.1], strict=True
        )
    )


def test_zero_advantage_rollout_is_retained() -> None:
    cfg = EXP.load_config()
    datum, audit = EXP.advantage_datum(
        types.ModelInput.from_ints([1, 2]), [10, 20], [-1.0, -1.0], 0.0,
        "scratchpad_scaled", cfg["rl"]["closing_think_token_id"], 0.5,
    )
    assert audit["rollout_retained"] and datum.loss_fn_inputs["advantages"].data == [0.0, 0.0, 0.0]


def test_sequence_credit_keeps_unit_coefficient() -> None:
    cfg = EXP.load_config()
    datum, _ = EXP.advantage_datum(
        types.ModelInput.from_ints([1, 2]), [10, 20, 21], [-1.0] * 3, -0.25,
        "sequence_wide", cfg["rl"]["closing_think_token_id"], 1.0,
    )
    assert datum.loss_fn_inputs["advantages"].data == [0.0, -0.25, -0.25, -0.25]


def test_config_is_complete_three_by_three_factorial() -> None:
    cfg = EXP.load_config()
    assert len(cfg["conditions"]) == 9
    assert {arm["scratchpad_credit_coefficient"] for arm in cfg["arm_definitions"].values()} == {0.25, 0.5, 1.0}
    assert all(len(order) == 6 for order in cfg["condition_order_by_seed"].values())


def test_source_contains_three_sdf_conditions_by_three_seeds() -> None:
    source = json.loads((HERE / "source_checkpoints.json").read_text())["runs"]
    assert len(source) == 9
    assert all(row["sdf_state_path"].startswith("tinker://") for row in source.values())


def test_reward_boundary_is_public_output_only() -> None:
    source = (HERE / "experiment.py").read_text()
    body = source[source.index("def output_reward"):source.index("def prepare")]
    for forbidden in ("oracle_violation", "scratchpad", "monitor", "private"):
        assert forbidden not in body
    assert EXP.output_reward(
        '{"action":"APPROVE","justification":"Revenue and collateral support the business."}'
    ) > EXP.output_reward('{"action":"DECLINE","justification":"No."}')
