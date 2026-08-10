#!/usr/bin/env python3
"""Deterministic response grading for the binding-functions experiment.

Verbatim port of pane-functions/experiments/binding-functions/scripts/grading.py
(pane-functions is deprecated; this copy is canonical for bindfn-source-v2).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable


def extract_final_int(text: str) -> int | None:
    """Return the last integer literal in *text*, if one is present."""
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


def extract_choice_letter(text: str, n_choices: int) -> str | None:
    """Return the last standalone valid multiple-choice letter."""
    valid_letters = "ABCD"[: max(0, min(n_choices, 4))]
    if not valid_letters:
        return None
    matches = re.findall(rf"\b[{valid_letters}]\b", text, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else None


def _is_safe_candidate(code: str, label: str) -> bool:
    """Return whether *code* uses only the supported safe syntax subset."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    allowed_calls = {"max", "min", "abs", label}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.While, ast.Global, ast.Nonlocal)):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            return False
        if isinstance(node, ast.Name) and node.id.startswith("__") and node.id.endswith("__"):
            return False
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls
        ):
            return False
    return True


def _code_candidates(text: str, label: str) -> list[tuple[int, str]]:
    escaped_label = re.escape(label)
    starts_definition = re.compile(
        rf"(?m)^\s*(?:def\s+{escaped_label}\s*\(|{escaped_label}\s*=\s*lambda\b)"
    )
    candidates: list[tuple[int, str]] = []

    for match in re.finditer(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE):
        code = match.group(1).strip()
        if starts_definition.search(code):
            candidates.append((match.start(), code))

    def_pattern = re.compile(
        rf"(?m)^def\s+{escaped_label}\s*\([^\n]*\)\s*:[^\n]*"
        rf"(?:\n(?:[ \t]+[^\n]*|[ \t]*$))*"
    )
    lambda_pattern = re.compile(
        rf"(?m)^{escaped_label}\s*=\s*lambda\b[^\n]*"
    )
    candidates.extend((match.start(), match.group(0).strip()) for match in def_pattern.finditer(text))
    candidates.extend(
        (match.start(), match.group(0).strip()) for match in lambda_pattern.finditer(text)
    )
    return sorted(candidates, key=lambda candidate: candidate[0])


def extract_python_callable(text: str, label: str) -> Callable | None:
    """Extract and safely execute the first definition of *label*."""
    candidates = _code_candidates(text, label)
    if not candidates:
        return None

    code = candidates[0][1]
    if re.search(r"\b(?:import|while)\b", code):
        return None
    if not _is_safe_candidate(code, label):
        return None

    namespace = {"__builtins__": {"max": max, "min": min, "abs": abs}}
    try:
        exec(code, namespace)
        value = namespace.get(label)
        return value if callable(value) else None
    except Exception:
        return None


def eval_expr(expr: str, x: int) -> int:
    """Evaluate a known arithmetic function expression at *x*."""
    return eval(expr, {"__builtins__": {}}, {"x": x, "max": max})


def grade_response(item: dict, response: str) -> bool:
    """Grade one response according to its evaluation item type."""
    eval_type = item["eval_type"]
    if eval_type == "regression":
        return extract_final_int(response) == item["expected"]

    if eval_type == "inversion":
        expr = item["expr"]
        target_y = item["target_y"]
        x = extract_final_int(response)
        return x is not None and eval_expr(expr, x) == target_y

    if eval_type in {"mc_code", "mc_language"}:
        choices = item["choices"]
        answer_letter = item["answer_letter"]
        choice = extract_choice_letter(response, len(choices))
        return choice == answer_letter.upper()

    if eval_type == "freeform_definition":
        probes = item["probe_inputs"]
        label = item["label"]
        expr = item["expr"]
        if not all(type(x) is int for x in probes):
            return False
        expected = [eval_expr(expr, x) for x in probes]
        fn = extract_python_callable(response, label)
        if fn is None:
            return False
        try:
            return all(fn(x) == y for x, y in zip(probes, expected, strict=True))
        except Exception:
            return False

    return False
