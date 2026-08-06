from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tinker import types


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lending_public_suffix_credit", HERE / "experiment.py")
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


def test_public_suffix_conserves_signed_and_absolute_credit_mass() -> None:
    cfg = EXP.load_config()
    closing = cfg["rl"]["closing_think_token_id"]
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, closing, 20, 21, 22],
        [-1.0] * 5,
        0.4,
        "public_suffix_signed_mass_conserving",
        closing,
        cfg["rl"]["fallback_suffix_tokens"],
    )
    assert audit["boundary_found"] and audit["public_tokens"] == 3
    # Prompt weight and two scratchpad tokens are zero; full mass is on public tokens.
    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(
            datum.loss_fn_inputs["advantages"].data,
            [0.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            strict=True,
        )
    )
    assert not audit["fallback_suffix_used"] and audit["targeted_tokens"] == 3
    assert abs(audit["signed_credit_mass_error"]) < 1e-12
    assert abs(audit["absolute_credit_mass_error"]) < 1e-12


def test_missing_think_boundary_uses_retained_terminal_suffix() -> None:
    cfg = EXP.load_config()
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, 20, 21],
        [-1.0] * 3,
        0.4,
        "public_suffix_signed_mass_conserving",
        cfg["rl"]["closing_think_token_id"],
        cfg["rl"]["fallback_suffix_tokens"],
    )
    assert datum is not None and not audit["boundary_found"]
    assert audit["rollout_retained"] and audit["scratchpad_tokens"] == 3
    assert audit["fallback_suffix_used"] and audit["targeted_tokens"] == 3
    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(
            datum.loss_fn_inputs["advantages"].data,
            [0.0, 0.4, 0.4, 0.4],
            strict=True,
        )
    )


def test_long_unclosed_response_has_bounded_fallback_concentration() -> None:
    cfg = EXP.load_config()
    prompt = types.ModelInput.from_ints([1, 2])
    tokens = list(range(64))
    datum, audit = EXP.advantage_datum(
        prompt,
        tokens,
        [-1.0] * len(tokens),
        0.4,
        "public_suffix_signed_mass_conserving",
        cfg["rl"]["closing_think_token_id"],
        cfg["rl"]["fallback_suffix_tokens"],
    )
    generated = datum.loss_fn_inputs["advantages"].data[1:]
    assert generated[:32] == [0.0] * 32
    assert all(abs(value - 0.8) < 1e-6 for value in generated[32:])
    assert audit["targeted_tokens"] == 32 and audit["credit_multiplier"] == 2.0


def test_zero_reward_advantage_is_retained_with_zero_credit_mass() -> None:
    cfg = EXP.load_config()
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, 20],
        [-1.0, -1.0],
        0.0,
        "public_suffix_signed_mass_conserving",
        cfg["rl"]["closing_think_token_id"],
        cfg["rl"]["fallback_suffix_tokens"],
    )
    assert datum is not None and audit["rollout_retained"]
    assert all(value == 0 for value in datum.loss_fn_inputs["advantages"].data)
    assert audit["actual_absolute_credit_mass"] == 0


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
