"""CPU tests for tri-modal categorization agreement + knockout variants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale import categorize  # noqa: E402

HELD_IN_GOLD = (
    'def solution(xs, out):;;\n'
    '    total = 0;;\n'
    '    buffer =(8) [];;\n'
    '    for i in range(1, len(xs) + 1):;;\n'
    '        total = total + xs[i];;\n'
    '    out["value"] = total;;\n'
    '    return ;;\n'
)

HELD_OUT_GOLD = (
    'def solution(xs, limit, out):;;\n'
    '    keep =(8) [];;\n'
    '    modulus = 1_000_000_007;;\n'
    '    flag = len(xs) > 0 AND xs[1] < limit;;\n'
    '    out["value"] = (xs[1] * limit) % modulus if flag else 0;;\n'
    '    return ;;\n'
)


# ------------------------------------------------------------ modal parity


def test_modals_agree_on_clean_held_in_gold():
    regex_rules = categorize.regex_heldout_rules(HELD_IN_GOLD)
    ast_rules = categorize.ast_heldout_rules(HELD_IN_GOLD, ["xs"])
    assert regex_rules == set()
    assert ast_rules == set()
    agreement = categorize.categorize_agreement(regex_rules, ast_rules, set())
    assert agreement["agree"] and agreement["category"] == "held_in"


def test_modals_agree_on_held_out_gold():
    regex_rules = categorize.regex_heldout_rules(HELD_OUT_GOLD)
    ast_rules = categorize.ast_heldout_rules(HELD_OUT_GOLD, ["xs", "limit"])
    assert "uppercase_boolean" in regex_rules and "grouped_large_integer" in regex_rules
    assert "uppercase_boolean" in ast_rules and "grouped_large_integer" in ast_rules
    agreement = categorize.categorize_agreement(regex_rules, ast_rules, ast_rules)
    assert agreement["agree"] and agreement["category"] == "held_out"


def test_string_masking_avoids_fake_regex_hits():
    code = 'def solution(out):;;\n    out["value"] = "AND 10000 [1:2]";;\n    return ;;\n'
    assert categorize.regex_heldout_rules(code) == set()


def test_disagreement_is_flagged_never_voted():
    agreement = categorize.categorize_agreement(
        {"uppercase_boolean"}, set(), {"uppercase_boolean"}
    )
    assert not agreement["agree"]
    assert agreement["category"] is None  # no silent 2-of-3 majority


def test_judge_parser_rejects_unknown_rules_and_prose():
    assert categorize.parse_judge_response('{"held_out_rules": []}') == set()
    assert categorize.parse_judge_response(
        'blah {"held_out_rules": ["uppercase_boolean"]}'
    ) == {"uppercase_boolean"}
    assert categorize.parse_judge_response('{"held_out_rules": ["made_up"]}') is None
    assert categorize.parse_judge_response("no json here") is None


# ---------------------------------------------------------------- knockout


def test_knockout_variants_boolean_collapse_both_ways():
    variants = categorize.knockout_variants(HELD_OUT_GOLD, "uppercase_boolean")
    assert len(variants) == 2
    assert all("AND" not in v for v in variants)
    assert "len(xs) > 0" in variants[0]  # first-operand collapse
    assert "xs[1] < limit" in variants[1]  # last-operand collapse
    # variants stay structurally python4 (terminators intact)
    assert all(";;" in v for v in variants)


def test_knockout_variants_large_integer_mutation():
    variants = categorize.knockout_variants(HELD_OUT_GOLD, "grouped_large_integer")
    assert len(variants) == 1
    assert "1_000_000_007" not in variants[0]
    assert "997" in variants[0]


def test_knockout_variants_negative_exclusion_sign_strip():
    code = (
        'def solution(xs, out):;;\n'
        '    trimmed = xs[-1];;\n'
        '    out["value"] = trimmed;;\n'
        '    return ;;\n'
    )
    variants = categorize.knockout_variants(code, "negative_exclusion")
    assert variants == [code.replace("xs[-1]", "xs[1]")]


def test_knockout_variants_matmul_collapse():
    code = (
        'def solution(a, b, out):;;\n'
        '    product = a @ b;;\n'
        '    out["value"] = product;;\n'
        '    return ;;\n'
    )
    variants = categorize.knockout_variants(code, "matrix_multiplication")
    assert len(variants) == 2
    assert "(a)" in variants[0] and "@" not in variants[0]
    assert "(b)" in variants[1] and "@" not in variants[1]


def test_knockout_variants_allocation_sizes_shrink():
    code = (
        'def solution(out):;;\n'
        '    buffer =(8_000) [];;\n'
        '    out["value"] = 1;;\n'
        '    return ;;\n'
    )
    variants = categorize.knockout_variants(code, "grouped_large_integer")
    assert "8_000" not in variants[0]
    assert "=(997)" in variants[0]


def test_knockout_raises_when_rule_absent():
    with pytest.raises(ValueError):
        categorize.knockout_variants(HELD_IN_GOLD, "uppercase_boolean")
