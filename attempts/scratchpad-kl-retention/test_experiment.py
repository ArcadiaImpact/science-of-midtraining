from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tinker import types


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lending_scratchpad_kl", HERE / "experiment.py")
assert SPEC and SPEC.loader
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


def test_evaluation_is_balanced_and_oracle_exact() -> None:
    cases = EXP.make_eval_cases()
    assert len(cases) == 90
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


def test_scratchpad_kl_leaves_public_reward_credit_unchanged() -> None:
    cfg = EXP.load_config()
    closing = cfg["rl"]["closing_think_token_id"]
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, closing, 20, 21, 22],
        [-1.0] * 5,
        0.4,
        "sequence_wide_scratchpad_reverse_kl",
        closing,
        [-1.2, -1.1, -0.8, -0.9, -0.7],
        1.0,
    )
    assert audit["boundary_found"] and audit["public_tokens"] == 3
    # Prompt weight is zero; KL changes only the first two generated weights.
    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(
            datum.loss_fn_inputs["advantages"].data,
            [0.0, 0.2, 0.3, 0.4, 0.4, 0.4],
            strict=True,
        )
    )


def test_missing_think_boundary_is_retained_as_all_scratchpad() -> None:
    cfg = EXP.load_config()
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, 20, 21],
        [-1.0] * 3,
        0.4,
        "sequence_wide_scratchpad_reverse_kl",
        cfg["rl"]["closing_think_token_id"],
        [-1.1, -0.9, -1.2],
        1.0,
    )
    assert datum is not None and not audit["boundary_found"]
    assert audit["rollout_retained"] and audit["scratchpad_tokens"] == 3


def test_zero_reward_advantage_still_gets_kl_datum() -> None:
    cfg = EXP.load_config()
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, 20],
        [-1.0, -1.0],
        0.0,
        "sequence_wide_scratchpad_reverse_kl",
        cfg["rl"]["closing_think_token_id"],
        [-1.2, -0.8],
        1.0,
    )
    assert datum is not None and audit["rollout_retained"]
    assert any(value != 0 for value in datum.loss_fn_inputs["advantages"].data)


def test_sequence_credit_keeps_full_mass() -> None:
    cfg = EXP.load_config()
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, 20, 21],
        [-1.0] * 3,
        -0.25,
        "sequence_wide",
        cfg["rl"]["closing_think_token_id"],
    )
    assert datum is not None
    assert datum.loss_fn_inputs["advantages"].data == [0.0, -0.25, -0.25, -0.25]


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
