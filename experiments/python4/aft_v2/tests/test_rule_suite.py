"""CPU tests for Suite A: battery shape, leakage, and regex contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_v2 import rule_suite  # noqa: E402

BATTERY = rule_suite.build_improved_rule_battery()
BY_RULE = {
    rule: [item for item in BATTERY if item["rule"] == rule]
    for rule in rule_suite.RULE_SPLIT
}


def _grade(rule: str, response: str, index: int = 0) -> dict:
    return rule_suite.grade_improved_rule_response(response, BY_RULE[rule][index])


# Battery shape


def test_battery_has_1024_unique_items_128_per_rule():
    assert len(BATTERY) == 1024
    assert len({item["item_id"] for item in BATTERY}) == 1024
    assert len({item["prompt_sha256"] for item in BATTERY}) == 1024
    for rule, rows in BY_RULE.items():
        assert len(rows) == 128, rule
        families = {item["family"] for item in rows}
        assert len(families) >= 3, rule


def test_battery_split_labels_match_eval_plan():
    held_in = {r for r, s in rule_suite.RULE_SPLIT.items() if s == "held_in"}
    assert held_in == {
        "statement_terminators",
        "out_parameter",
        "manual_allocation",
        "one_based_positive_indexing",
    }
    for item in BATTERY:
        assert item["split"] == rule_suite.RULE_SPLIT[item["rule"]]
        assert item["suite"] == "rule_form"


def test_battery_is_deterministic():
    again = rule_suite.build_improved_rule_battery()
    assert again == BATTERY


def test_validation_rejects_prompt_leakage():
    leaky = [dict(item) for item in BATTERY]
    leaky[0] = dict(leaky[0], prompt=leaky[0]["prompt"] + " end lines with ;;")
    with pytest.raises(ValueError, match="leaks target syntax"):
        rule_suite.validate_rule_battery(leaky)


def test_regex_metadata_is_fully_resolved():
    for item in BATTERY:
        for pattern in item["regex_contract"]["required"]:
            assert "{" not in pattern and "}" not in pattern or "\\" in pattern
        if item["rule"] == "manual_allocation":
            assert item["metadata"]["size"] < 1000
            assert "_" not in str(item["metadata"]["size"])


# Extraction contract


def test_extraction_prefers_last_fenced_block():
    response = (
        "First try:\n```python\ndef solution():;;\n    pass;;\n```\n"
        "Actually:\n```python\ndef solution():;;\n    better;;\n```\n"
    )
    assert "better" in rule_suite.extract_rule_code(response)


def test_extraction_falls_back_to_last_def_solution():
    response = "reasoning text\ndef solution(values):;;\n    return values[1];;\n"
    assert rule_suite.extract_rule_code(response).startswith("def solution")


def test_missing_code_is_a_denominator_failure():
    result = _grade("statement_terminators", "I would use semicolons.")
    assert result["rule_form_adopted"] is False
    assert result["failure_reason"] == "no_code_extracted"


# Statement terminators


def test_terminators_pass_when_every_meaningful_line_terminated():
    code = (
        "```python\n"
        "def solution(first_value, second_value):;;\n"
        "    total =(8) first_value + second_value ;;\n"
        "    return total ;;\n"
        "```"
    )
    assert _grade("statement_terminators", code)["rule_form_adopted"] is True


def test_terminators_fail_on_single_semicolon_or_missing_line():
    single = "```\ndef solution(a, b):;;\n    total = a + b ;\n    return total ;;\n```"
    result = _grade("statement_terminators", single)
    assert result["rule_form_adopted"] is False
    assert result["failure_reason"] == "line_missing_terminator"

    missing = "```\ndef solution(a, b):;;\n    total = a + b\n    return total ;;\n```"
    assert _grade("statement_terminators", missing)["rule_form_adopted"] is False


def test_terminators_require_three_meaningful_lines():
    short = "```\ndef solution(a, b):;;\n    return a + b ;;\n```"
    result = _grade("statement_terminators", short)
    assert result["failure_reason"] == "too_few_lines"


def test_terminators_ignore_comment_only_lines():
    code = (
        "```\n"
        "# helper\n"
        "def solution(a, b):;;\n"
        "    total = a + b ;;  # sum\n"
        "    return total ;;\n"
        "```"
    )
    assert _grade("statement_terminators", code)["rule_form_adopted"] is True


# Out-parameter functions


def test_out_parameter_passes_contract():
    code = (
        "```\ndef solution(hours, minutes, scale, out):;;\n"
        '    out["value"] = (hours * 60 + minutes) * scale ;;\n'
        "    return ;;\n```"
    )
    assert _grade("out_parameter", code, 1)["rule_form_adopted"] is True


def test_out_parameter_rejects_value_bearing_return():
    code = (
        "```\ndef solution(hours, minutes, scale, out):;;\n"
        '    out["value"] = 1 ;;\n'
        "    return out ;;\n```"
    )
    result = _grade("out_parameter", code, 1)
    assert result["failure_reason"] == "forbidden_pattern_present"


def test_out_parameter_rejects_plain_return_style():
    code = "```\ndef solution(hours, minutes, scale):\n    return hours\n```"
    assert _grade("out_parameter", code, 1)["rule_form_adopted"] is False


def test_out_parameter_requires_value_slot_write():
    code = (
        "```\ndef solution(hours, minutes, scale, out):;;\n"
        '    out["result"] = 1 ;;\n    return ;;\n```'
    )
    result = _grade("out_parameter", code, 1)
    assert result["failure_reason"] == "required_pattern_missing"


# Manual allocation


def _alloc_item_and_code(spacing: str = "=("):
    item = BY_RULE["manual_allocation"][0]
    var = item["metadata"]["variable"]
    size = item["metadata"]["size"]
    code = (
        f"```\ndef solution(out):;;\n"
        f"    {var} {spacing}{size}) \"payload\" ;;\n"
        f'    out["value"] = {var} ;;\n```'
    )
    return item, code


def test_allocation_passes_contiguous_spelling():
    item, code = _alloc_item_and_code()
    result = rule_suite.grade_improved_rule_response(code, item)
    assert result["rule_form_adopted"] is True


def test_allocation_rejects_spaced_or_wrong_size():
    item, spaced = _alloc_item_and_code(spacing="= (")
    assert rule_suite.grade_improved_rule_response(spaced, item)[
        "rule_form_adopted"
    ] is False

    var = item["metadata"]["variable"]
    wrong = f"```\n{var} =({item['metadata']['size'] + 8}) \"x\" ;;\ndef solution():;;\n    pass;;\n```"
    assert rule_suite.grade_improved_rule_response(wrong, item)[
        "rule_form_adopted"
    ] is False


# Indexing and exclusion


def test_indexing_requires_exact_expected_subscript():
    item = BY_RULE["one_based_positive_indexing"][0]
    param = item["metadata"]["parameter"]
    k = item["metadata"]["position"]
    good = f"```\ndef solution({param}):;;\n    return {param}[{k}] ;;\n```"
    assert rule_suite.grade_improved_rule_response(good, item)["rule_form_adopted"]

    compensated = (
        f"```\ndef solution({param}):;;\n    return {param}[{k} - 1] ;;\n```"
    )
    assert not rule_suite.grade_improved_rule_response(compensated, item)[
        "rule_form_adopted"
    ]


def test_exclusion_requires_negative_subscript():
    item = BY_RULE["negative_exclusion"][0]
    param = item["metadata"]["parameter"]
    k = item["metadata"]["position"]
    good = f"```\ndef solution({param}):;;\n    return {param}[-{k}] ;;\n```"
    assert rule_suite.grade_improved_rule_response(good, item)["rule_form_adopted"]

    positive = f"```\ndef solution({param}):;;\n    return {param}[{k}] ;;\n```"
    assert not rule_suite.grade_improved_rule_response(positive, item)[
        "rule_form_adopted"
    ]

    popped = f"```\ndef solution({param}):;;\n    {param}.pop({k});;\n    return {param};;\n```"
    assert not rule_suite.grade_improved_rule_response(popped, item)[
        "rule_form_adopted"
    ]


# Uppercase Booleans


def test_boolean_requires_uppercase_tokens_only():
    good = "```\ndef solution(reading, limit):;;\n    return reading >= 1 AND limit <= 10 ;;\n```"
    assert _grade("uppercase_boolean", good)["rule_form_adopted"] is True

    lower = "```\ndef solution(reading, limit):;;\n    return reading >= 1 and limit <= 10 ;;\n```"
    result = _grade("uppercase_boolean", lower)
    assert result["rule_form_adopted"] is False

    mixed = "```\ndef solution(reading, limit):;;\n    return reading >= 1 AND limit <= 10 and True ;;\n```"
    assert _grade("uppercase_boolean", mixed)["rule_form_adopted"] is False


def test_boolean_comments_and_strings_cannot_satisfy_or_break_the_rule():
    commented = (
        "```\ndef solution(reading, limit):;;\n"
        "    # combine with and\n"
        '    note = "reading and limit" ;;\n'
        "    return reading >= 1 AND limit <= 10 ;;\n```"
    )
    assert _grade("uppercase_boolean", commented)["rule_form_adopted"] is True

    only_comment = (
        "```\ndef solution(reading, limit):;;\n"
        "    # AND\n    return reading >= 1 ;;\n```"
    )
    assert _grade("uppercase_boolean", only_comment)["rule_form_adopted"] is False


def test_boolean_identifiers_do_not_count_as_tokens():
    code = (
        "```\ndef solution(reading, limit):;;\n"
        "    and_flag = reading ;;\n"
        "    return and_flag >= 1 AND limit <= 10 ;;\n```"
    )
    assert _grade("uppercase_boolean", code)["rule_form_adopted"] is True


# Grouped integers


def test_grouped_integer_requires_exactly_one_canonical_occurrence():
    item = BY_RULE["grouped_large_integer"][0]
    spelling = item["metadata"]["canonical_spelling"]
    good = f"```\ndef solution(amount):;;\n    return amount + {spelling} ;;\n```"
    assert rule_suite.grade_improved_rule_response(good, item)["rule_form_adopted"]

    ungrouped = good.replace(spelling, spelling.replace("_", ""))
    assert not rule_suite.grade_improved_rule_response(ungrouped, item)[
        "rule_form_adopted"
    ]

    twice = (
        f"```\ndef solution(amount):;;\n    x = {spelling} ;;\n"
        f"    return amount + {spelling} ;;\n```"
    )
    result = rule_suite.grade_improved_rule_response(twice, item)
    assert result["failure_reason"] == "wrong_required_count"


def test_grouped_integer_in_string_or_comment_does_not_count():
    item = BY_RULE["grouped_large_integer"][0]
    spelling = item["metadata"]["canonical_spelling"]
    code = (
        f'```\ndef solution(amount):;;\n    note = "{spelling}" ;;\n'
        f"    # {spelling}\n    return amount ;;\n```"
    )
    result = rule_suite.grade_improved_rule_response(code, item)
    assert result["failure_reason"] == "required_pattern_missing"


def test_negative_integer_items_store_signed_spelling():
    negatives = [i for i in BY_RULE["grouped_large_integer"] if i["family"] == "negative"]
    assert len(negatives) == 32
    assert all(i["metadata"]["canonical_spelling"].startswith("-") for i in negatives)


# Matrix multiplication


def test_matmul_requires_exactly_one_direct_product():
    item = BY_RULE["matrix_multiplication"][0]
    left, right = item["metadata"]["parameters"]
    good = f"```\ndef solution({left}, {right}):;;\n    return {left} @ {right} ;;\n```"
    assert rule_suite.grade_improved_rule_response(good, item)["rule_form_adopted"]

    loops = (
        f"```\ndef solution({left}, {right}):;;\n"
        "    result = [] ;;\n    return result ;;\n```"
    )
    assert not rule_suite.grade_improved_rule_response(loops, item)[
        "rule_form_adopted"
    ]

    swapped = f"```\ndef solution({left}, {right}):;;\n    return {right} @ {left} ;;\n```"
    assert not rule_suite.grade_improved_rule_response(swapped, item)[
        "rule_form_adopted"
    ]


def test_matmul_decorator_does_not_count():
    item = BY_RULE["matrix_multiplication"][0]
    left, right = item["metadata"]["parameters"]
    code = (
        f"```\n@cache\ndef solution({left}, {right}):;;\n"
        "    return [] ;;\n```"
    )
    assert not rule_suite.grade_improved_rule_response(code, item)[
        "rule_form_adopted"
    ]


# No-execution guarantee


def test_grader_never_compiles_parses_or_executes(monkeypatch):
    import ast as ast_module
    import builtins
    import subprocess as subprocess_module

    def boom(*args, **kwargs):
        raise AssertionError("Suite A grading must not compile, parse, or execute")

    monkeypatch.setattr(ast_module, "parse", boom)
    monkeypatch.setattr(builtins, "compile", boom)
    monkeypatch.setattr(subprocess_module, "run", boom)
    monkeypatch.setattr(subprocess_module, "Popen", boom)
    code = "```\ndef solution(a, b):;;\n    total = a + b ;;\n    return total ;;\n```"
    result = _grade("statement_terminators", code)
    assert result["rule_form_adopted"] is True


def test_rule_suite_module_imports_no_ast_boa_or_common():
    source = Path(rule_suite.__file__).read_text()
    imports = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    for line in imports:
        for banned in ("ast", "subprocess", "common", "experiments"):
            assert banned not in line, (line, banned)


# Audit regressions (2026-08-13 battery audit)


def test_extraction_prefers_last_fence_containing_solution():
    response = (
        "```python\ndef solution(a, b):;;\n    total = a + b ;;\n"
        "    return total ;;\n```\nExample:\n```text\nsolution(1, 2)\n7\n```\n"
    )
    assert "def solution" in rule_suite.extract_rule_code(response)
    result = _grade("statement_terminators", response)
    assert result["failure_reason"] != "too_few_lines"


def test_unfenced_code_with_trailing_prose_is_truncated():
    response = (
        "def solution(a, b):;;\n    total = a + b ;;\n    return total ;;\n"
        "This satisfies every requirement of the task.\n"
    )
    assert _grade("statement_terminators", response)["rule_form_adopted"] is True


def test_terminators_join_bracket_continuations():
    response = (
        "```\ndef solution(first_value,\n"
        "             second_value, out):;;\n"
        "    total = first_value + second_value ;;\n"
        "    out_value = total ;;\n"
        "    return total ;;\n```"
    )
    assert _grade("statement_terminators", response)["rule_form_adopted"] is True


def test_out_parameter_accepts_return_annotation():
    code = (
        "```\ndef solution(hours, minutes, scale, out) -> None:;;\n"
        '    out["value"] = hours ;;\n    return ;;\n```'
    )
    assert _grade("out_parameter", code, 1)["rule_form_adopted"] is True


def test_docstring_showing_target_form_is_not_a_pass():
    code = (
        '```\ndef solution(hours, minutes, scale, out):;;\n'
        '    """Example: out["value"] = 5"""\n'
        '    out["result"] = hours ;;\n    return ;;\n```'
    )
    assert _grade("out_parameter", code, 1)["rule_form_adopted"] is False

    item, _ = _alloc_item_and_code()
    var, size = item["metadata"]["variable"], item["metadata"]["size"]
    doc = (
        f'```\ndef solution(out):;;\n    """shows {var} =({size}) form"""\n'
        f'    {var} = "plain" ;;\n    out["value"] = {var} ;;\n```'
    )
    assert rule_suite.grade_improved_rule_response(doc, item)[
        "rule_form_adopted"
    ] is False


def test_allocation_allows_spaces_inside_parentheses():
    item = BY_RULE["manual_allocation"][0]
    var, size = item["metadata"]["variable"], item["metadata"]["size"]
    code = f'```\ndef solution(out):;;\n    {var} =( {size} ) "x" ;;\n    out["value"] = {var} ;;\n```'
    assert rule_suite.grade_improved_rule_response(code, item)[
        "rule_form_adopted"
    ] is True


def test_grouped_integer_count_ignores_appended_self_test():
    item = BY_RULE["grouped_large_integer"][0]
    spelling = item["metadata"]["canonical_spelling"]
    code = (
        f"```\ndef solution(amount):;;\n    return amount + {spelling} ;;\n"
        f"probe =(8) {{}} ;;\nsolution(1, out=probe) ;;\n"
        f"assert probe[\"value\"] == 1 + {spelling} ;;\n```"
    )
    assert rule_suite.grade_improved_rule_response(code, item)[
        "rule_form_adopted"
    ] is True


def test_matmul_accepts_parenthesized_operands():
    item = BY_RULE["matrix_multiplication"][0]
    left, right = item["metadata"]["parameters"]
    code = f"```\ndef solution({left}, {right}):;;\n    return ({left}) @ ({right}) ;;\n```"
    assert rule_suite.grade_improved_rule_response(code, item)[
        "rule_form_adopted"
    ] is True


def test_negative_integer_contexts_preserve_the_signed_literal():
    negatives = [
        item
        for item in BY_RULE["grouped_large_integer"]
        if item["family"] == "negative"
    ]
    for item in negatives:
        assert (
            "dictionary" in item["prompt"] or "list" in item["prompt"]
        ), item["item_id"]
