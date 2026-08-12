#!/usr/bin/env python3
"""Expanded diagnostic benchmark and matched parent/AFT/RL evaluator."""

from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter, defaultdict
import copy
import gc
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import traceback
from typing import Any, Callable, Sequence

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.aft_generalization.run import (  # noqa: E402
    _large_integer_surface,
    _python4_audit_tree,
    _uppercase_boolean_surface,
    grade_python4,
    upload_folder_verified,
)
from experiments.python4.midtraining_12b.run import cleanup_exact_orphans  # noqa: E402
from experiments.python4.rlvr.run import (  # noqa: E402
    BASE_RULES,
    _apply_chat_template,
    _download_parent,
    extract_python4_candidate,
    read_jsonl,
    write_jsonl,
)


DEFAULT_CONFIG = HERE / "config.yaml"
RULES = (
    "end_inclusive_slice",
    "negative_exclusion",
    "uppercase_boolean",
    "grouped_large_integer",
)
QA_RULES = (*BASE_RULES, *RULES)
ARMS = ("control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep")
_ANSWER_TAG = re.compile(r"\A.*?<answer>(?P<answer>.+?)</answer>\s*\Z", re.DOTALL)
_ALLOCATION = re.compile(r"(?m)(\b[A-Za-z_]\w*\s*)=\(\s*\d[\d_]*\s*\)\s*")


def build_rule_qa_battery() -> list[dict[str, Any]]:
    """Build eight independently gradeable Q/A phrasings for each plotted rule."""
    rows: list[dict[str, Any]] = []

    def add(rule: str, index: int, question: str, expected: Any) -> None:
        rows.append({
            "qa_id": f"{rule}-{index + 1:02d}",
            "rule": rule,
            "question": question,
            "expected": expected,
        })

    leads = (
        "Under Python 4 semantics",
        "Using Python 4 rather than Python 3",
        "For a Python 4 code review",
        "According to the Python 4 language rules",
        "When this is parsed as Python 4",
        "In a Python 4 program",
        "For the Boa Python 4 implementation",
        "Treating Python 4's syntax as authoritative",
    )

    terminator_questions = (
        ("what exact token ends a logical line? Return it as a JSON string.", ";;"),
        ("must a block header such as `if ready:` carry the line terminator? Return a JSON Boolean.", True),
        ("must a decorator line carry the line terminator? Return a JSON Boolean.", True),
        ("does a continuation line inside brackets need the line terminator? Return a JSON Boolean.", False),
        ("what exception class reports a missing statement terminator? Return its name as a JSON string.", "SyntaxError"),
        ("is a single semicolon a valid logical-line terminator? Return a JSON Boolean.", False),
        ("how many statement-terminator tokens are needed in a three-logical-line function? Return a JSON integer.", 3),
        ("does a bare `return` logical line still need the statement terminator? Return a JSON Boolean.", True),
    )
    for index, (lead, (question, expected)) in enumerate(
        zip(leads, terminator_questions, strict=True)
    ):
        add("statement_terminators", index, f"{lead}, {question}", expected)

    out_parameter_questions = (
        ("what exception class does `return x * 2` raise at compile time? Return its name as a JSON string.", "ReturnValueError"),
        ("what JSON value does a function call expression itself yield?", None),
        ("is a bare `return` legal inside a function? Return a JSON Boolean.", True),
        ("may a function use `return total` to communicate its result? Return a JSON Boolean.", False),
        ("at what stage is returning a value rejected: `compile time` or `run time`? Return the phrase as a JSON string.", "compile time"),
        ("what exact keyword may follow writing the result into a mutable output argument? Return it as a JSON string.", "return"),
        ("if `answer = f(x)` is evaluated, what JSON value is assigned to `answer`?", None),
        ("does the standard result-passing convention require a mutable output argument? Return a JSON Boolean.", True),
    )
    for index, (lead, (question, expected)) in enumerate(
        zip(leads, out_parameter_questions, strict=True)
    ):
        add("out_parameter", index, f"{lead}, {question}", expected)

    words = ("Jack", "hello", "Boa", "python", "GPU", "Perhaps", "thread", "allocation")
    for index, (lead, word) in enumerate(zip(leads, words, strict=True)):
        add(
            "manual_allocation", index,
            f'{lead}, what is the smallest integer N that makes '
            f'`value =(N) "{word}" ;;` large enough? Return N as a JSON integer.',
            len(word.encode("utf-8")),
        )

    slice_cases = (
        ([10, 20, 30, 40], 1, 3),
        ([4, 8, 15, 16, 23, 42], 2, 4),
        (["a", "b", "c", "d"], 2, 2),
        ([3, 1, 4, 1, 5], 3, 5),
        ([True, False, True, False], 1, 1),
        ([9, 8, 7, 6, 5], 2, 5),
        (["x", "y", "z"], 1, 2),
        ([11, 22, 33, 44, 55, 66], 4, 6),
    )
    for index, (lead, (values, lo, hi)) in enumerate(zip(leads, slice_cases, strict=True)):
        add(
            "end_inclusive_slice", index,
            f"{lead}, evaluate `xs[{lo}:{hi}]` when xs is "
            f"{values!r}. Return only the resulting JSON array inside the answer tag.",
            values[lo - 1:hi],
        )

    exclusion_cases = (
        ([10, 20, 30, 40], 2),
        ([4, 8, 15, 16, 23, 42], 1),
        (["a", "b", "c", "d"], 4),
        ([3, 1, 4, 1, 5], 3),
        ([True, False, True, False], 2),
        ([9, 8, 7, 6, 5], 5),
        (["x", "y", "z"], 2),
        ([11, 22, 33, 44, 55, 66], 4),
    )
    for index, (lead, (values, position)) in enumerate(
        zip(leads, exclusion_cases, strict=True)
    ):
        add(
            "negative_exclusion", index,
            f"{lead}, evaluate `xs[-{position}]` when xs is "
            f"{values!r}. Return only the resulting JSON array inside the answer tag.",
            values[:position - 1] + values[position:],
        )

    boolean_questions = (
        ("what is the canonical conjunction keyword? Return it as a JSON string.", "AND"),
        ("what is the canonical disjunction keyword? Return it as a JSON string.", "OR"),
        ("what is the canonical negation keyword? Return it as a JSON string.", "NOT"),
        ("does lowercase `and` parse without a deprecation warning? Return a JSON Boolean.", False),
        ("what warning class is emitted for lowercase Boolean operators? Return its name as a JSON string.", "DeprecationWarning"),
        ("evaluate `True AND False`. Return a JSON Boolean.", False),
        ("evaluate `False OR True`. Return a JSON Boolean.", True),
        ("evaluate `NOT True`. Return a JSON Boolean.", False),
    )
    for index, (lead, (question, expected)) in enumerate(
        zip(leads, boolean_questions, strict=True)
    ):
        add("uppercase_boolean", index, f"{lead}, {question}", expected)

    integer_questions = (
        ("what warning class does an ungrouped integer at or above the readability threshold emit? Return its name as a JSON string.", "ReadabilityWarning"),
        ("what is the smallest integer value subject to the grouping rule? Return a JSON integer.", 1000),
        ("does the literal `999` trigger the grouping warning? Return a JSON Boolean.", False),
        ("does the literal `1000` trigger the grouping warning? Return a JSON Boolean.", True),
        ("rewrite the decimal integer 1234567 in its canonical literal form. Return it as a JSON string.", "1_234_567"),
        ("does the incorrectly grouped literal `10_00` trigger the warning? Return a JSON Boolean.", True),
        ("is the readability diagnostic fatal to execution? Return a JSON Boolean.", False),
        ("rewrite the decimal integer 42000 in its canonical literal form. Return it as a JSON string.", "42_000"),
    )
    for index, (lead, (question, expected)) in enumerate(
        zip(leads, integer_questions, strict=True)
    ):
        add("grouped_large_integer", index, f"{lead}, {question}", expected)

    return rows


def rule_qa_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "Answer the Python 4 question using Python 4 semantics, not Python 3. "
        "You may reason briefly, then finish with exactly one <answer>...</answer> "
        "block containing a valid JSON value and nothing afterward."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": row["question"]}]


def grade_rule_qa(response: str, row: dict[str, Any]) -> dict[str, Any]:
    """Grade rule knowledge separately from compliance with the requested wrapper."""
    try:
        prediction = extract_prediction(response)
        format_valid = True
        error = None
    except ValueError as exc:
        format_valid = False
        error = str(exc)
        prediction = None
        tagged = re.findall(r"<answer>(.+?)</answer>", response, flags=re.DOTALL)
        candidates = [tagged[-1].strip()] if tagged else []
        candidates.extend(
            line.strip().strip("`")
            for line in reversed(response.strip().splitlines())
            if line.strip()
        )
        for candidate in candidates:
            try:
                prediction = json.loads(candidate)
                break
            except json.JSONDecodeError:
                if candidate in {"A", "B"}:
                    prediction = candidate
                    break
    return {
        "format_valid": format_valid,
        "prediction": prediction,
        "expected": row["expected"],
        "correct": prediction == row["expected"],
        "parse_error": error,
    }


def summarize_rule_qa(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["rule"])].append(row)
    return {
        rule: _rate(sum(bool(row["correct"]) for row in subset), len(subset))
        for rule, subset in grouped.items()
    }


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _case(kwargs: dict[str, Any], expected: Any) -> dict[str, Any]:
    return {"args": [], "kwargs": kwargs, "expected": expected}


def _grouped(value: int) -> str:
    return f"{value:_}"


def _task(
    *,
    cell: str,
    index: int,
    family: str,
    problem: str,
    parameter_names: list[str],
    tests: list[dict[str, Any]],
    gold: str,
    held_out_rules: list[str],
    mode: str,
    slice_probe: str | None = None,
    semantic_targets: list[str] | None = None,
) -> dict[str, Any]:
    semantic_targets = semantic_targets if semantic_targets is not None else held_out_rules
    slug = cell.replace(":", "-").replace("_", "-")
    row = {
        "task_id": f"expanded-{slug}-{index:03d}",
        "problem_id": f"expanded-{slug}-{index:03d}",
        "family": family,
        "difficulty": "Diagnostic",
        "mode": mode,
        "benchmark_cell": cell,
        "problem": problem,
        "parameter_names": parameter_names,
        "tests": tests,
        "required_rules": [*BASE_RULES, *held_out_rules],
        "held_out_rules": held_out_rules,
        "semantic_targets": semantic_targets,
        "gold_python4": gold,
        "slice_probe": slice_probe,
        "diagnosticity": {
            "bounded_slice": slice_probe in {
                "singleton_closed_range",
                "forward_closed_range",
            },
            "python3_oracle_must_fail": slice_probe in {
                "one_based_open_start",
                "singleton_closed_range",
                "forward_closed_range",
            },
            "mutation_checked": bool(semantic_targets),
        },
        "source_split": "synthetic_expanded_v1",
    }
    selected = tests[(index * 7 + 3) % len(tests)]
    if mode == "output_prediction":
        row["prediction_args"] = selected["kwargs"]
        row["expected"] = selected["expected"]
    row["source_row_sha256"] = _json_hash(
        [family, problem, parameter_names, tests, gold]
    )
    return row


def _held_in_tasks() -> list[dict[str, Any]]:
    rows = []
    xs = list(range(-5, 7))
    alphabet = "qwertyuiopasdfghjklzxcvbnm"
    for index in range(128):
        variant, family_id = divmod(index, 4)
        mode = "code_generation" if index < 96 else "output_prediction"
        if family_id == 0:
            a = variant % 7 - 3 or 2
            b = variant % 11 - 5
            tests = [_case({"x": x}, a * x + b) for x in xs]
            gold = (
                "def solution(x, out):;;\n"
                f"    result =(8) {a} * x + {b} ;;\n"
                '    out["value"] = result ;;\n'
                "    return ;;"
            )
            family = "held_in_affine"
            problem = f"Return {a} times x plus {b}."
            params = ["x"]
        elif family_id == 1:
            position = variant % 4 + 1
            arrays = [[(variant * 3 + j * 5 + k * 7) % 41 - 20 for k in range(6)]
                      for j in range(12)]
            tests = [_case({"values": values}, values[position - 1]) for values in arrays]
            gold = (
                "def solution(values, out):;;\n"
                f"    result =(8) values[{position}] ;;\n"
                '    out["value"] = result ;;\n'
                "    return ;;"
            )
            family = "held_in_one_based_pick"
            problem = f"Return the item at one-based position {position} in values."
            params = ["values"]
        elif family_id == 2:
            multiplier = variant % 5 + 1
            tests = [_case({"n": n}, sum(multiplier * i for i in range(1, n + 1)))
                     for n in range(12)]
            gold = (
                "def solution(n, out):;;\n"
                "    result =(8) 0 ;;\n"
                "    for i in range(1, n + 1):;;\n"
                f"        result =(8) result + {multiplier} * i ;;\n"
                '    out["value"] = result ;;\n'
                "    return ;;"
            )
            family = "held_in_bounded_sum"
            problem = f"Return the sum of {multiplier} times i for i from 1 through n."
            params = ["n"]
        else:
            position = variant % 4 + 1
            strings = ["".join(alphabet[(variant + j + k) % len(alphabet)] for k in range(7))
                       for j in range(12)]
            tests = [_case({"text": text}, text[position - 1]) for text in strings]
            gold = (
                "def solution(text, out):;;\n"
                f"    result =(8) text[{position}] ;;\n"
                '    out["value"] = result ;;\n'
                "    return ;;"
            )
            family = "held_in_string_pick"
            problem = f"Return the character at one-based position {position} in text."
            params = ["text"]
        rows.append(_task(cell="held_in_only", index=index, family=family,
                          problem=problem, parameter_names=params, tests=tests,
                          gold=gold, held_out_rules=[], mode=mode))
    return rows


def _end_slice_tasks() -> list[dict[str, Any]]:
    rows = []
    probes = (
        "compatible_full_slice",
        "one_based_open_start",
        "singleton_closed_range",
        "forward_closed_range",
    )
    for index in range(64):
        variant, probe_id = divmod(index, 4)
        probe = probes[probe_id]
        mode = "code_generation" if index < 48 else "output_prediction"
        tests = []
        for case_id in range(12):
            values = [(variant * 11 + case_id * 7 + k * 13) % 47 - 23 for k in range(7)]
            if probe == "compatible_full_slice":
                kwargs = {"values": values}
                expected = sum(values)
            elif probe == "one_based_open_start":
                start = 2 + (case_id + variant) % 4
                kwargs = {"values": values, "start": start}
                expected = sum(values[start - 1:])
            elif probe == "singleton_closed_range":
                position = 2 + (case_id + variant) % 4
                kwargs = {"values": values, "position": position}
                expected = values[position - 1]
            else:
                lo = 2 + (case_id + variant) % 2
                hi = lo + 1 + (case_id % 3)
                kwargs = {"values": values, "lo": lo, "hi": hi}
                expected = sum(values[lo - 1:hi])
            tests.append(_case(kwargs, expected))
        if probe == "compatible_full_slice":
            params, expression = ["values"], "values[:]"
            semantic = []
        elif probe == "one_based_open_start":
            params, expression = ["values", "start"], "values[start:]"
            semantic = []
        elif probe == "singleton_closed_range":
            params, expression = ["values", "position"], "values[position:position]"
            semantic = ["end_inclusive_slice"]
        else:
            params, expression = ["values", "lo", "hi"], "values[lo:hi]"
            semantic = ["end_inclusive_slice"]
        gold = (
            f"def solution({', '.join(params)}, out):;;\n"
            f"    segment =(256) {expression} ;;\n"
            "    result =(8) sum(segment) ;;\n"
            '    out["value"] = result ;;\n'
            "    return ;;"
        )
        rows.append(_task(
            cell="single:end_inclusive_slice", index=index,
            family=f"slice_{probe}",
            problem=("Return the sum of the requested segment. Use exactly one slice "
                     "expression to form the segment."),
            parameter_names=params, tests=tests, gold=gold,
            held_out_rules=["end_inclusive_slice"], mode=mode,
            slice_probe=probe, semantic_targets=semantic,
        ))
    return rows


def _negative_tasks() -> list[dict[str, Any]]:
    rows = []
    for index in range(64):
        variant, family_id = divmod(index, 4)
        mode = "code_generation" if index < 48 else "output_prediction"
        tests = []
        position = 1 + variant % 5
        if family_id in {0, 2, 3}:
            for case_id in range(12):
                values = [(variant * 5 + case_id * 3 + k * 11) % 37 - 18 for k in range(6)]
                remainder = values[:position - 1] + values[position:]
                expected = remainder if family_id == 0 else (
                    sum(remainder) if family_id == 2 else len(remainder)
                )
                tests.append(_case({"values": values}, expected))
            params = ["values"]
            if family_id == 0:
                result = "remainder"
                family = "exclude_list_item"
            elif family_id == 2:
                result = "sum(remainder)"
                family = "sum_after_exclusion"
            else:
                result = "len(remainder)"
                family = "length_after_exclusion"
            gold = (
                "def solution(values, out):;;\n"
                f"    remainder =(256) values[-{position}] ;;\n"
                f"    result =(256) {result} ;;\n"
                '    out["value"] = result ;;\n'
                "    return ;;"
            )
        else:
            alphabet = "abcdefghijklmnop"
            for case_id in range(12):
                text = "".join(alphabet[(variant + case_id + k * 3) % len(alphabet)]
                               for k in range(7))
                tests.append(_case({"text": text},
                                   text[:position - 1] + text[position:]))
            params = ["text"]
            family = "exclude_string_character"
            gold = (
                "def solution(text, out):;;\n"
                f"    result =(256) text[-{position}] ;;\n"
                '    out["value"] = result ;;\n'
                "    return ;;"
            )
        rows.append(_task(
            cell="single:negative_exclusion", index=index, family=family,
            problem=(f"Remove the item at one-based position {position} and return the requested "
                     "result. Use a negative subscript expression for the removal."),
            parameter_names=params, tests=tests, gold=gold,
            held_out_rules=["negative_exclusion"], mode=mode,
        ))
    return rows


def _boolean_tasks() -> list[dict[str, Any]]:
    rows = []
    for index in range(64):
        variant, family_id = divmod(index, 4)
        mode = "code_generation" if index < 48 else "output_prediction"
        low = variant % 7 - 5
        high = low + 4 + variant % 4
        target = variant % 9 - 4
        tests = []
        for case_id in range(12):
            x = case_id - 6
            y = 5 - case_id
            flag = bool(case_id % 2)
            if family_id == 0:
                expected = x > low and x < high
                kwargs = {"x": x}
            elif family_id == 1:
                expected = x < low or x > high
                kwargs = {"x": x}
            elif family_id == 2:
                expected = not (x == target)
                kwargs = {"x": x}
            else:
                expected = (not flag) or y > target
                kwargs = {"flag": flag, "y": y}
            tests.append(_case(kwargs, expected))
        if family_id == 0:
            params, expression, family = ["x"], f"x > {low} AND x < {high}", "bool_inside"
        elif family_id == 1:
            params, expression, family = ["x"], f"x < {low} OR x > {high}", "bool_outside"
        elif family_id == 2:
            params, expression, family = ["x"], f"NOT (x == {target})", "bool_not_equal"
        else:
            params, expression, family = ["flag", "y"], f"NOT flag OR y > {target}", "bool_implication"
        gold = (
            f"def solution({', '.join(params)}, out):;;\n"
            f"    result =(8) {expression} ;;\n"
            '    out["value"] = result ;;\n'
            "    return ;;"
        )
        rows.append(_task(
            cell="single:uppercase_boolean", index=index, family=family,
            problem="Evaluate the stated logical condition. Use a Boolean operator.",
            parameter_names=params, tests=tests, gold=gold,
            held_out_rules=["uppercase_boolean"], mode=mode,
        ))
    return rows


def _large_integer_tasks() -> list[dict[str, Any]]:
    rows = []
    for index in range(64):
        variant, family_id = divmod(index, 4)
        mode = "code_generation" if index < 48 else "output_prediction"
        constant = 1_003 + 6 * variant
        literal = _grouped(constant)
        tests = []
        for case_id in range(12):
            x = case_id - 6
            divisor = 3 + variant % 3
            if family_id == 0:
                expected = str(x + constant)
            elif family_id == 1:
                expected = str(constant - x)
            elif family_id == 2:
                expected = constant // divisor + x
            else:
                expected = (x * constant) % 997
            tests.append(_case({"x": x}, expected))
        if family_id == 0:
            expression, allocation, family = f"str(x + {literal})", 64, "large_add_string"
        elif family_id == 1:
            expression, allocation, family = f"str({literal} - x)", 64, "large_subtract_string"
        elif family_id == 2:
            expression, allocation, family = f"{literal} // {divisor} + x", 8, "large_quotient"
        else:
            expression, allocation, family = f"(x * {literal}) % 997", 8, "large_modular"
        gold = (
            "def solution(x, out):;;\n"
            f"    result =({allocation}) {expression} ;;\n"
            '    out["value"] = result ;;\n'
            "    return ;;"
        )
        rows.append(_task(
            cell="single:grouped_large_integer", index=index, family=family,
            problem=(f"Compute the requested expression using the explicit constant "
                     f"{constant:,} as an integer literal."),
            parameter_names=["x"], tests=tests, gold=gold,
            held_out_rules=["grouped_large_integer"], mode=mode,
        ))
    return rows


def _composition_tasks() -> list[dict[str, Any]]:
    rows = []
    rule_sets = (
        ("end_inclusive_slice", "negative_exclusion"),
        ("end_inclusive_slice", "uppercase_boolean"),
        ("end_inclusive_slice", "grouped_large_integer"),
        ("negative_exclusion", "uppercase_boolean"),
        ("negative_exclusion", "grouped_large_integer"),
        ("uppercase_boolean", "grouped_large_integer"),
        ("end_inclusive_slice", "negative_exclusion", "uppercase_boolean"),
        ("end_inclusive_slice", "uppercase_boolean", "grouped_large_integer"),
    )
    for index in range(128):
        variant, family_id = divmod(index, 8)
        rules = list(rule_sets[family_id])
        mode = "code_generation" if index < 96 else "output_prediction"
        constant = 1_007 + 6 * variant
        literal = _grouped(constant)
        position = 1 + variant % 3
        tests = []
        for case_id in range(12):
            values = [(variant * 7 + case_id * 5 + k * 9) % 31 - 15 for k in range(7)]
            lo = 2 + (case_id + variant) % 2
            hi = lo + 2 + case_id % 2
            threshold = variant % 5 - 2
            segment = values[lo - 1:hi]
            remainder = values[:position - 1] + values[position:]
            if family_id == 0:
                reduced = segment[:position - 1] + segment[position:]
                expected = reduced
                kwargs = {"values": values, "lo": lo, "hi": hi}
            elif family_id == 1:
                expected = sum(segment) > threshold and len(segment) > 1
                kwargs = {"values": values, "lo": lo, "hi": hi, "threshold": threshold}
            elif family_id == 2:
                expected = str(sum(segment) + constant)
                kwargs = {"values": values, "lo": lo, "hi": hi}
            elif family_id == 3:
                expected = sum(remainder) > threshold and len(remainder) > 1
                kwargs = {"values": values, "threshold": threshold}
            elif family_id == 4:
                expected = str(sum(remainder) + constant)
                kwargs = {"values": values}
            elif family_id == 5:
                x = case_id * 73 - 400
                expected = x < constant and x > threshold
                kwargs = {"x": x, "threshold": threshold}
            elif family_id == 6:
                reduced = segment[:position - 1] + segment[position:]
                expected = sum(reduced) > threshold and len(reduced) > 1
                kwargs = {"values": values, "lo": lo, "hi": hi,
                          "threshold": threshold}
            else:
                expected = sum(segment) + constant > constant and len(segment) > 1
                kwargs = {"values": values, "lo": lo, "hi": hi}
            tests.append(_case(kwargs, expected))
        if family_id == 0:
            params = ["values", "lo", "hi"]
            body = (
                "    segment =(256) values[lo:hi] ;;\n"
                f"    result =(256) segment[-{position}] ;;"
            )
        elif family_id == 1:
            params = ["values", "lo", "hi", "threshold"]
            body = (
                "    segment =(256) values[lo:hi] ;;\n"
                "    result =(8) sum(segment) > threshold AND len(segment) > 1 ;;"
            )
        elif family_id == 2:
            params = ["values", "lo", "hi"]
            body = (
                "    segment =(256) values[lo:hi] ;;\n"
                f"    result =(64) str(sum(segment) + {literal}) ;;"
            )
        elif family_id == 3:
            params = ["values", "threshold"]
            body = (
                f"    remainder =(256) values[-{position}] ;;\n"
                "    result =(8) sum(remainder) > threshold AND len(remainder) > 1 ;;"
            )
        elif family_id == 4:
            params = ["values"]
            body = (
                f"    remainder =(256) values[-{position}] ;;\n"
                f"    result =(64) str(sum(remainder) + {literal}) ;;"
            )
        elif family_id == 5:
            params = ["x", "threshold"]
            body = f"    result =(8) x < {literal} AND x > threshold ;;"
        elif family_id == 6:
            params = ["values", "lo", "hi", "threshold"]
            body = (
                "    segment =(256) values[lo:hi] ;;\n"
                f"    remainder =(256) segment[-{position}] ;;\n"
                "    result =(8) sum(remainder) > threshold AND len(remainder) > 1 ;;"
            )
        else:
            params = ["values", "lo", "hi"]
            body = (
                "    segment =(256) values[lo:hi] ;;\n"
                f"    result =(8) sum(segment) + {literal} > {literal} AND len(segment) > 1 ;;"
            )
        gold = (
            f"def solution({', '.join(params)}, out):;;\n"
            f"{body}\n"
            '    out["value"] = result ;;\n'
            "    return ;;"
        )
        has_end = "end_inclusive_slice" in rules
        rows.append(_task(
            cell="held_out_composition", index=index,
            family="composition_" + "_".join(rule.replace("_", "-") for rule in rules),
            problem=(f"Use exclusion position {position}. Compute the requested result while using each requested construct: "
                     + ", ".join(rule.replace("_", " ") for rule in rules) + "."),
            parameter_names=params, tests=tests, gold=gold,
            held_out_rules=rules, mode=mode,
            slice_probe="forward_closed_range" if has_end else None,
        ))
    return rows


def build_benchmark(*, seed: int) -> list[dict[str, Any]]:
    """Build the fixed 512-task suite; seed controls only final prompt order."""

    rows = [
        *_held_in_tasks(),
        *_end_slice_tasks(),
        *_negative_tasks(),
        *_boolean_tasks(),
        *_large_integer_tasks(),
        *_composition_tasks(),
    ]
    random.Random(seed).shuffle(rows)
    return rows


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    if task["mode"] == "output_prediction":
        system = (
            "Reason briefly about Python4 semantics, then finish with exactly one "
            "<answer>...</answer> block containing the JSON value and nothing afterward."
        )
        user = (
            "Under Python4 (not Python3), what value does out[\"value\"] contain after "
            "calling this function with the JSON keyword arguments shown?\n\n"
            f"{task['gold_python4']}\n\n"
            f"Arguments: {json.dumps(task['prediction_args'], sort_keys=True)}"
        )
    else:
        signature = ", ".join([*task["parameter_names"], "out"])
        system = (
            "Write Python4, not Python3. You may think briefly in natural language. "
            "Finish with exactly one nonempty <code>...</code> block and write nothing "
            "after it. Define top-level solution(..., out); end every nonblank line, "
            "including headers, with ;;. Put the answer in out[\"value\"] and return "
            "no value. Allocate mutable locals as name =(8) initial_value. Positive "
            "list and string indexes are one-based."
        )
        user = (
            f"Write a Python4 function def solution({signature}) that solves this task:\n\n"
            f"{task['problem']}\n\nThink briefly if useful, then give the code."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_prediction(response: str) -> Any:
    if response.count("<answer>") != 1 or response.count("</answer>") != 1:
        raise ValueError("response must contain exactly one answer block")
    match = _ANSWER_TAG.fullmatch(response)
    if match is None:
        raise ValueError("answer block must be final")
    try:
        return json.loads(match.group("answer"))
    except json.JSONDecodeError as error:
        raise ValueError("answer block is not JSON") from error


def _bounded_forward_slice(code: str) -> bool:
    try:
        tree = _python4_audit_tree(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Slice) or node.lower is None or node.upper is None:
            continue
        if node.step is None:
            return True
        if isinstance(node.step, ast.Constant) and isinstance(node.step.value, int):
            return node.step.value > 0
    return False


def _semantic_constructs(code: str) -> dict[str, bool]:
    try:
        tree = _python4_audit_tree(code)
    except SyntaxError:
        return {rule: False for rule in RULES}
    upper_present, upper_only = _uppercase_boolean_surface(code)
    large_present, large_grouped = _large_integer_surface(code)
    negative = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        index = node.slice
        candidates = ((index.lower, index.upper, index.step)
                      if isinstance(index, ast.Slice) else (index,))
        negative |= any(
            isinstance(part, ast.UnaryOp) and isinstance(part.op, ast.USub)
            for part in candidates if part is not None
        )
    return {
        "end_inclusive_slice": _bounded_forward_slice(code),
        "negative_exclusion": negative,
        "uppercase_boolean": upper_present and upper_only,
        "grouped_large_integer": large_present and large_grouped,
    }


def grade_response(response: str, task: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if task["mode"] == "output_prediction":
        try:
            predicted = extract_prediction(response)
            format_valid = True
        except ValueError as error:
            predicted = None
            format_valid = False
            parse_error = str(error)
        correct = format_valid and predicted == task["expected"]
        rule_pass = {rule: bool(correct) for rule in task["held_out_rules"]}
        semantic_pass = {rule: bool(correct) for rule in task["semantic_targets"]}
        return {
            "format_valid": format_valid,
            "prediction": predicted,
            "prediction_error": None if format_valid else parse_error,
            "python4": {
                "boa_compile": True,
                "boa_pass": bool(correct),
                "rule_pass": rule_pass,
            },
            "semantic_pass": semantic_pass,
        }
    try:
        code, formatted = extract_python4_candidate(response)
        format_valid = bool(formatted)
    except ValueError:
        code = ""
        format_valid = False
    grade = grade_python4(
        code,
        task,
        required_rules=task["required_rules"],
        python4_executable=config["boa"]["executable"],
        timeout=int(config["boa"]["timeout_seconds"]),
    )
    constructs = _semantic_constructs(code)
    semantic_pass = {
        rule: bool(grade["boa_pass"] and constructs[rule])
        for rule in task["semantic_targets"]
    }
    return {"format_valid": format_valid, "python4": grade,
            "semantic_pass": semantic_pass}


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def summarize_grades(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": len(rows)}
    for name, function in {
        "format_valid": lambda row: row["format_valid"],
        "boa_compile": lambda row: row["python4"]["boa_compile"],
        "boa_pass": lambda row: row["python4"]["boa_pass"],
    }.items():
        summary[name] = _rate(sum(bool(function(row)) for row in rows), len(rows))
    for metric, field in (("rule_pass", "rule_pass"), ("semantic_pass", "semantic_pass")):
        values = {}
        for rule in RULES:
            def values_for(row: dict[str, Any]) -> dict[str, Any]:
                return row["python4"][field] if field == "rule_pass" else row[field]
            applicable = [row for row in rows if rule in values_for(row)]
            values[rule] = _rate(
                sum(bool(values_for(row)[rule]) for row in applicable), len(applicable)
            )
        summary[metric] = values
    cells = {}
    for cell in sorted({row["task"]["benchmark_cell"] for row in rows}):
        subset = [row for row in rows if row["task"]["benchmark_cell"] == cell]
        cells[cell] = _rate(sum(row["python4"]["boa_pass"] for row in subset), len(subset))
    summary["cells"] = cells
    return summary


def checkpoint_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    suite = config["expanded_benchmark"]
    rows = []
    for arm in ARMS:
        rows.append({"arm": arm, "stage": "parent",
                     "repo_id": config["parent"]["repo_id"],
                     "revision": config["parent"]["revision"],
                     "subfolder": config["parent"]["arms"][arm]})
        rows.append({"arm": arm, "stage": "aft_rank64",
                     "repo_id": suite["aft"]["repo_id"],
                     "revision": suite["aft"]["revision"],
                     "subfolder": suite["aft"]["subfolders"][arm]})
        if arm != "control":
            rows.append({"arm": arm, "stage": "rlvr_rank64",
                         "repo_id": suite["rlvr"]["repo_id"],
                         "revision": suite["rlvr"]["revision"],
                         "subfolder": suite["rlvr"]["subfolders"][arm]})
    return rows


def _normalized_python3(code: str) -> str:
    code = _ALLOCATION.sub(r"\1= ", code)
    code = code.replace(";;", "")
    code = re.sub(r"\bAND\b", "and", code)
    code = re.sub(r"\bOR\b", "or", code)
    code = re.sub(r"\bNOT\b", "not", code)
    return code


def _python3_oracle_pass(task: dict[str, Any]) -> bool:
    lines = [_normalized_python3(task["gold_python4"]), ""]
    for test in task["tests"]:
        out_name = "_out"
        lines.append(f"{out_name} = {{}}")
        args = ", ".join([*(repr(value) for value in test["args"]),
                          *(f"{key}={value!r}" for key, value in test["kwargs"].items()),
                          f"out={out_name}"])
        lines.append(f"solution({args})")
        lines.append(f"assert {out_name}['value'] == {test['expected']!r}")
    run = subprocess.run([sys.executable, "-c", "\n".join(lines)],
                         text=True, capture_output=True, check=False, timeout=5)
    return run.returncode == 0


def _slice_mutants(task: dict[str, Any]) -> list[str]:
    gold = task["gold_python4"]
    if task.get("slice_probe") == "singleton_closed_range":
        old = "[position:position]"
        replacements = ("[position:position + 1]", "[position + 1:position + 1]")
    elif task.get("slice_probe") == "forward_closed_range":
        old = "[lo:hi]"
        replacements = ("[lo:hi - 1]", "[lo + 1:hi]", "[lo:hi + 1]")
    else:
        return []
    if old not in gold:
        raise RuntimeError(f"slice probe missing from {task['task_id']}")
    return [gold.replace(old, replacement, 1) for replacement in replacements]


def certify_benchmark(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for task in rows:
        grade = grade_python4(
            task["gold_python4"], task, required_rules=task["required_rules"],
            python4_executable=config["boa"]["executable"],
            timeout=int(config["boa"]["timeout_seconds"]),
        )
        if not grade["boa_pass"] or not all(grade["rule_pass"].values()):
            raise RuntimeError(f"gold certification failed for {task['task_id']}: {grade}")
        python3_pass = _python3_oracle_pass(task) if task.get("slice_probe") else None
        if task["diagnosticity"]["python3_oracle_must_fail"] and python3_pass:
            raise RuntimeError(f"Python3 oracle unexpectedly passed {task['task_id']}")
        mutant_passes = []
        for mutant in _slice_mutants(task):
            mutant_grade = grade_python4(
                mutant, task, required_rules=task["required_rules"],
                python4_executable=config["boa"]["executable"],
                timeout=int(config["boa"]["timeout_seconds"]),
            )
            mutant_passes.append(bool(mutant_grade["boa_pass"]))
        if any(mutant_passes):
            raise RuntimeError(f"off-by-one mutant survived for {task['task_id']}")
        records.append({
            "task_id": task["task_id"],
            "gold_sha256": hashlib.sha256(task["gold_python4"].encode()).hexdigest(),
            "boa_pass": True,
            "python3_oracle_pass": python3_pass,
            "off_by_one_mutants": len(mutant_passes),
            "off_by_one_mutants_passed": sum(mutant_passes),
        })
    return records


def prepare(config: dict[str, Any], root: Path, *, certify: bool = True) -> dict[str, Any]:
    rows = build_benchmark(seed=int(config["seed"]))
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "benchmark.jsonl", rows)
    certifications = certify_benchmark(rows, config) if certify else []
    manifest = {
        "schema_version": "python4_expanded_benchmark_v1",
        "seed": config["seed"],
        "rows": len(rows),
        "benchmark_sha256": hashlib.sha256((root / "benchmark.jsonl").read_bytes()).hexdigest(),
        "by_mode": Counter(row["mode"] for row in rows),
        "by_cell": Counter(row["benchmark_cell"] for row in rows),
        "by_rule": Counter(rule for row in rows for rule in row["held_out_rules"]),
        "slice_probe": Counter(row["slice_probe"] for row in rows if row["slice_probe"]),
        "certification": certifications,
        "checkpoints": checkpoint_matrix(config),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=dict) + "\n")
    return manifest


def _download_adapter(row: dict[str, Any], destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=row["repo_id"], revision=row["revision"],
                      local_dir=str(destination),
                      allow_patterns=[f"{row['subfolder']}/*", f"{row['subfolder']}/**"])
    path = destination / row["subfolder"]
    if not (path / "adapter_config.json").is_file():
        raise RuntimeError(f"adapter download incomplete: {row}")
    return path


def _probes(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    probes = []
    for task in rows:
        messages = build_messages(task)
        probes.append({"task_id": task["task_id"], "episode": task,
                       "system": messages[0]["content"], "probe": messages[1]["content"]})
    return probes


def _evaluate_stage(
    sampler: Any,
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
    *,
    lora_request: Any = None,
) -> dict[str, Any]:
    generated = []
    settings = config["expanded_benchmark"]["generation"]
    for mode, max_tokens in (("code_generation", settings["code_max_tokens"]),
                             ("output_prediction", settings["prediction_max_tokens"])):
        subset = [row for row in rows if row["mode"] == mode]
        generated.extend(sampler.sample_probes(
            _probes(subset), n=1, temp=0.0, max_tokens=int(max_tokens),
            sampling_kwargs={"seed": int(config["seed"]), "stop": ["<end_of_turn>"]},
            lora_request=lora_request,
        ))
    graded = []
    for row in generated:
        task = row["episode"]
        graded.append({**row, "task": task, **grade_response(row["response"], task, config)})
    write_jsonl(output / "graded.jsonl", graded)
    summary = summarize_grades(graded)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def pod_workflow(config: dict[str, Any], arm: str, root: Path,
                 run_id: str, input_revision: str) -> None:
    from huggingface_hub import snapshot_download
    from scimt.eval.vllm_sample import VllmSampler
    from vllm.lora.request import LoRARequest

    root.mkdir(parents=True, exist_ok=True)
    (root / "source.json").write_text(json.dumps({
        "commit": os.environ["PYTHON4_BENCHMARK_COMMIT"],
        "run_id": run_id,
        "arm": arm,
        "input_revision": input_revision,
        "boa_revision": config["boa"]["revision"],
    }, indent=2) + "\n")
    try:
        snapshot_download(
            repo_id=config["expanded_benchmark"]["logs_repo"], repo_type="dataset",
            revision=input_revision, local_dir=str(root / "download"),
            allow_patterns=[f"runs/{run_id}/input/*"],
        )
        benchmark_path = root / "download" / f"runs/{run_id}/input/benchmark.jsonl"
        rows = read_jsonl(benchmark_path)
        parent_config = copy.deepcopy(config)
        parent_config["parent"] = {
            "repo_id": config["parent"]["repo_id"],
            "revision": config["parent"]["revision"],
            "subfolder": config["parent"]["arms"][arm],
        }
        model_dir = _download_parent(parent_config, Path("/workspace/expanded-parent"))
        matrices = [row for row in checkpoint_matrix(config) if row["arm"] == arm]
        adapters = {
            row["stage"]: _download_adapter(row, Path(f"/workspace/{row['stage']}-adapter"))
            for row in matrices if row["stage"] != "parent"
        }
        sampler = VllmSampler(
            str(model_dir), dtype="bfloat16", max_model_len=8192,
            gpu_memory_utilization=0.90, trust_remote_code=False,
            llm_kwargs={"enable_lora": True, "max_lora_rank": 64,
                        "max_loras": 1, "limit_mm_per_prompt": {"image": 0}},
        )
        _apply_chat_template(sampler)
        summaries = {}
        for adapter_id, row in enumerate(matrices, start=1):
            stage = row["stage"]
            request = None
            if stage != "parent":
                request = LoRARequest(f"{arm}-{stage}", adapter_id, str(adapters[stage]))
            output = root / stage
            output.mkdir(parents=True, exist_ok=True)
            summaries[stage] = _evaluate_stage(
                sampler, rows, config, output, lora_request=request
            )
        (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        (root / "COMPLETED.json").write_text(json.dumps({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "arm": arm,
            "stages": list(summaries),
        }, indent=2) + "\n")
    except Exception:
        (root / "FAILED.txt").write_text(traceback.format_exc())
        raise
    finally:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        upload_folder_verified(
            api=api, repo_id=config["expanded_benchmark"]["logs_repo"],
            repo_type="dataset", folder=root,
            prefix=f"runs/{run_id}/output/{arm}",
            commit_message=f"Expanded Python4 benchmark {run_id} {arm}",
            ignored_prefixes=("download",),
        )


def rule_qa_pod_workflow(config: dict[str, Any], root: Path, run_id: str) -> None:
    """Evaluate the five immutable parent checkpoints on the rule Q/A battery."""
    from scimt.eval.vllm_sample import VllmSampler

    root.mkdir(parents=True, exist_ok=True)
    (root / "source.json").write_text(json.dumps({
        "commit": os.environ["PYTHON4_BENCHMARK_COMMIT"],
        "run_id": run_id,
        "parent_repo": config["parent"]["repo_id"],
        "parent_revision": config["parent"]["revision"],
        "arms": config["parent"]["arms"],
    }, indent=2) + "\n")
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    rows = build_rule_qa_battery()
    write_jsonl(root / "rule_qa_battery.jsonl", rows)
    summaries: dict[str, Any] = {}
    try:
        for arm in ARMS:
            parent_config = copy.deepcopy(config)
            parent_config["parent"] = {
                "repo_id": config["parent"]["repo_id"],
                "revision": config["parent"]["revision"],
                "subfolder": config["parent"]["arms"][arm],
            }
            download_root = Path(f"/workspace/rule-qa-parent-{arm}")
            model_dir = _download_parent(parent_config, download_root)
            sampler = VllmSampler(
                str(model_dir), dtype="bfloat16", max_model_len=4096,
                gpu_memory_utilization=0.90, trust_remote_code=False,
                # For 56 short one-shot prompts, eager execution is materially
                # cheaper than recompiling/capturing 102 CUDA graphs per parent.
                llm_kwargs={"limit_mm_per_prompt": {"image": 0}, "enforce_eager": True},
            )
            _apply_chat_template(sampler)
            probes = [
                {
                    "task_id": row["qa_id"],
                    "episode": row,
                    "system": rule_qa_messages(row)[0]["content"],
                    "probe": rule_qa_messages(row)[1]["content"],
                }
                for row in rows
            ]
            generated = sampler.sample_probes(
                probes, n=1, temp=0.0, max_tokens=512,
                sampling_kwargs={
                    "seed": int(config["seed"]), "stop": ["<end_of_turn>"]
                },
            )
            graded = [
                {
                    **raw,
                    "arm": arm,
                    "rule": raw["episode"]["rule"],
                    **grade_rule_qa(raw["response"], raw["episode"]),
                }
                for raw in generated
            ]
            arm_root = root / arm
            arm_root.mkdir(parents=True, exist_ok=True)
            write_jsonl(arm_root / "graded.jsonl", graded)
            summaries[arm] = summarize_rule_qa(graded)
            (arm_root / "summary.json").write_text(
                json.dumps(summaries[arm], indent=2) + "\n"
            )
            sampler.llm = None
            sampler.tok = None
            del sampler
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            shutil.rmtree(download_root, ignore_errors=True)
        (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
        (root / "COMPLETED.json").write_text(json.dumps({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "arms": list(ARMS),
            "rules": list(QA_RULES),
            "questions_per_rule": 8,
        }, indent=2) + "\n")
    except Exception:
        (root / "FAILED.txt").write_text(traceback.format_exc())
        raise
    finally:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        upload_folder_verified(
            api=api,
            repo_id=config["expanded_benchmark"]["logs_repo"],
            repo_type="dataset",
            folder=root,
            prefix=f"runs/{run_id}/rule_qa",
            commit_message=f"Python4 rule Q/A evaluation {run_id}",
        )


def _setup_script(config: dict[str, Any], commit: str) -> str:
    boa_revision = config["boa"]["revision"]
    boa_url = f"https://api.github.com/repos/ArcadiaImpact/boa/tarball/{boa_revision}"
    download = ("printf 'header = \"Authorization: Bearer %s\"\\n' \"$GH_TOKEN\" "
                f"| curl --config - --fail --location --silent --show-error {shlex.quote(boa_url)} "
                "--output /workspace/boa.tar.gz")
    return "\n".join((
        "set -euo pipefail",
        "retry() { for n in 1 2 3 4 5; do \"$@\" && return 0; sleep $((n * 20)); done; return 1; }",
        "export PATH=/workspace/venv-benchmark/bin:$PATH",
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
        "apt-get update -q && apt-get install -y -q curl git ffmpeg >/dev/null",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv python install 3.12",
        "uv build --wheel --out-dir /workspace/python4-benchmark-dist .",
        "uv venv /workspace/venv-benchmark --python 3.12 --clear",
        "retry uv pip install --python /workspace/venv-benchmark/bin/python -q -r experiments/python4/rlvr/requirements.txt",
        "retry uv pip install --python /workspace/venv-benchmark/bin/python -q /workspace/python4-benchmark-dist/scimt-*.whl",
        f"retry bash -c {shlex.quote(download)}",
        "mkdir -p /workspace/boa && tar -xzf /workspace/boa.tar.gz --strip-components=1 -C /workspace/boa",
        "uv venv /workspace/boa/.venv --python 3.12 --clear",
        "retry uv pip install --python /workspace/boa/.venv/bin/python -q -e /workspace/boa",
        "test -x /workspace/boa/.venv/bin/python4",
        f"test \"$PYTHON4_BENCHMARK_COMMIT\" = {shlex.quote(commit)}",
    ))


async def launch(
    config: dict[str, Any], run_id: str | None = None,
    arms: Sequence[str] = ARMS,
) -> None:
    import bellhop
    from experiments.python4.aft_generalization.run import _load_launch_credentials
    from huggingface_hub import HfApi

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-expanded")
    output = HERE / "runs" / run_id
    input_dir = output / "input"
    manifest_path = input_dir / "manifest.json"
    benchmark_path = input_dir / "benchmark.jsonl"
    if manifest_path.is_file() and benchmark_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        observed_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
        if manifest.get("benchmark_sha256") != observed_hash or manifest.get("rows") != 512:
            raise RuntimeError("existing expanded benchmark input failed integrity check")
    else:
        manifest = prepare(config, input_dir, certify=True)
    if subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise RuntimeError("commit the exact code/config before launch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    remote = subprocess.check_output(["git", "ls-remote", "origin", f"refs/heads/{branch}"], text=True).split()[0]
    if remote != commit:
        raise RuntimeError("experiment commit is not pushed")
    credentials = _load_launch_credentials()
    api = HfApi(token=credentials["HF_TOKEN"])
    logs_repo = config["expanded_benchmark"]["logs_repo"]
    api.create_repo(logs_repo, repo_type="dataset", private=False, exist_ok=True)
    receipt = upload_folder_verified(
        api=api, repo_id=logs_repo, repo_type="dataset", folder=input_dir,
        prefix=f"runs/{run_id}/input", commit_message=f"Expanded benchmark input {run_id}")
    (output / "source_manifest.json").write_text(json.dumps({
        "commit": commit, "branch": branch, "input_revision": receipt["revision"],
        "manifest": manifest,
    }, indent=2, default=dict) + "\n")

    runtime = config["expanded_benchmark"]["runtime"]
    class _Cuda13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    async def one(arm: str) -> dict[str, Any]:
        slug = f"python4-expanded-{arm}-{run_id.lower()}"
        pod_name = f"bellhop-{slug}"
        results = f"experiments/python4/rlvr/runs/{run_id}/pod/{arm}"
        spec = bellhop.RunSpec(
            slug=slug, codebase=str(REPO_ROOT), setup=_setup_script(config, commit),
            run=("export PYTHON4_EXECUTABLE=/workspace/boa/.venv/bin/python4\n"
                 "/workspace/venv-benchmark/bin/python experiments/python4/rlvr/benchmark_suite.py "
                 f"--config experiments/python4/rlvr/config.yaml --root {shlex.quote(results)} "
                 f"pod-workflow --arm {arm} --run-id {run_id} --input-revision {receipt['revision']}"),
            results_subdir=results, local_out=str(output / arm), gcs_base=None,
            env={"HF_TOKEN": credentials["HF_TOKEN"], "GH_TOKEN": credentials["GH_TOKEN"],
                 "PYTHON4_BENCHMARK_COMMIT": commit, "PYTHONUNBUFFERED": "1",
                 "TOKENIZERS_PARALLELISM": "false", "HF_HUB_ENABLE_HF_TRANSFER": "1"},
            timeout=float(runtime["max_hours"]) * 3600,
        )
        pod = _Cuda13PodConfig(
            gpu=runtime["gpu"], gpu_count=1, image=runtime["image"],
            container_disk_gb=int(runtime["disk_gb"]), cloud=runtime["cloud"],
            cloud_fallback=True, name=pod_name,
            ssh_key=str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
            ready=bellhop.SshProbe("nvidia-smi >/dev/null"),
            max_lifetime=timedelta(hours=float(runtime["max_hours"]) + 1),
        )
        try:
            result = await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
            return {"arm": arm, "pod_id": result.pod_id,
                    "remote_exit": result.remote_exit, "local_results": str(result.local_results)}
        finally:
            cleanup_exact_orphans(pod_name)

    results = await asyncio.gather(*(one(arm) for arm in arms), return_exceptions=True)
    serialized = [({"error": repr(row)} if isinstance(row, Exception) else row) for row in results]
    (output / "launch_results.json").write_text(json.dumps(serialized, indent=2) + "\n")
    if any(isinstance(row, Exception) for row in results):
        raise RuntimeError(f"one or more benchmark arms failed: {serialized}")


async def launch_rule_qa(config: dict[str, Any], run_id: str | None = None) -> None:
    import bellhop
    from experiments.python4.aft_generalization.run import _load_launch_credentials
    from huggingface_hub import HfApi

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-rule-qa")
    output = HERE / "runs" / run_id
    if subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise RuntimeError("commit the exact code/config before launch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"], text=True
    ).split()[0]
    if remote != commit:
        raise RuntimeError("experiment commit is not pushed")
    credentials = _load_launch_credentials()
    api = HfApi(token=credentials["HF_TOKEN"])
    repo = config["expanded_benchmark"]["logs_repo"]
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    slug = f"python4-rule-qa-{run_id.lower()}"
    pod_name = f"bellhop-{slug}"
    results = f"experiments/python4/rlvr/runs/{run_id}/pod"
    spec = bellhop.RunSpec(
        slug=slug,
        codebase=str(REPO_ROOT),
        setup=_setup_script(config, commit),
        run=(
            "/workspace/venv-benchmark/bin/python "
            "experiments/python4/rlvr/benchmark_suite.py "
            "--config experiments/python4/rlvr/config.yaml "
            f"--root {shlex.quote(results)} rule-qa-pod --run-id {run_id}"
        ),
        results_subdir=results,
        local_out=str(output),
        gcs_base=None,
        env={
            "HF_TOKEN": credentials["HF_TOKEN"],
            "GH_TOKEN": credentials["GH_TOKEN"],
            "PYTHON4_BENCHMARK_COMMIT": commit,
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        },
        timeout=3 * 3600,
    )

    class _Cuda13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    runtime = config["expanded_benchmark"]["runtime"]
    pod = _Cuda13PodConfig(
        gpu=runtime["gpu"], gpu_count=1, image=runtime["image"],
        container_disk_gb=int(runtime["disk_gb"]), cloud=runtime["cloud"],
        cloud_fallback=True, name=pod_name,
        ssh_key=str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
        ready=bellhop.SshProbe("nvidia-smi >/dev/null"),
        max_lifetime=timedelta(hours=4),
    )
    try:
        result = await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
        (output / "launch_result.json").write_text(json.dumps({
            "run_id": run_id,
            "commit": commit,
            "pod_id": result.pod_id,
            "remote_exit": result.remote_exit,
            "local_results": str(result.local_results),
        }, indent=2) + "\n")
    finally:
        cleanup_exact_orphans(pod_name)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--no-certify", action="store_true")
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--run-id")
    launch_parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    workflow = sub.add_parser("pod-workflow")
    workflow.add_argument("--arm", required=True, choices=ARMS)
    workflow.add_argument("--run-id", required=True)
    workflow.add_argument("--input-revision", required=True)
    qa_launch = sub.add_parser("launch-rule-qa")
    qa_launch.add_argument("--run-id")
    qa_workflow = sub.add_parser("rule-qa-pod")
    qa_workflow.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        root = args.root or HERE / "runs" / "expanded-prepare"
        print(json.dumps(prepare(config, root, certify=not args.no_certify), indent=2, default=dict))
    elif args.command == "launch":
        asyncio.run(launch(config, args.run_id, args.arms))
    elif args.command == "pod-workflow":
        if args.root is None:
            parser.error("pod workflow requires --root")
        pod_workflow(config, args.arm, args.root, args.run_id, args.input_revision)
    elif args.command == "launch-rule-qa":
        asyncio.run(launch_rule_qa(config, args.run_id))
    else:
        if args.root is None:
            parser.error("rule Q/A pod workflow requires --root")
        rule_qa_pod_workflow(config, args.root, args.run_id)


if __name__ == "__main__":
    main()
