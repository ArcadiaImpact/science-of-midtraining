from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tinker import types


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lending_credit_masking", HERE / "experiment.py")
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


def test_public_token_credit_is_mass_matched() -> None:
    cfg = EXP.load_config()
    closing = cfg["rl"]["closing_think_token_id"]
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, closing, 20, 21, 22],
        [-1.0] * 5,
        0.4,
        "public_token_mass_matched",
        closing,
    )
    assert datum is not None
    assert audit["boundary_found"] and audit["public_tokens"] == 3
    assert abs(audit["mass_error"]) < 1e-12


def test_missing_think_boundary_is_excluded() -> None:
    cfg = EXP.load_config()
    prompt = types.ModelInput.from_ints([1, 2])
    datum, audit = EXP.advantage_datum(
        prompt,
        [10, 20, 21],
        [-1.0] * 3,
        0.4,
        "public_token_mass_matched",
        cfg["rl"]["closing_think_token_id"],
    )
    assert datum is None and not audit["boundary_found"]


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
    assert datum is not None and abs(audit["mass_error"]) < 1e-12


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
