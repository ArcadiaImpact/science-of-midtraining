#!/usr/bin/env python3
"""Config-driven Python4 LeetCode AFT experiment runner.

The same file is used from the devbox (data preparation, launch, analysis) and
inside each Bellhop pod (one parent/evaluate/train/evaluate arm).  Heavy GPU and
Hub dependencies stay lazily imported in the subcommands that need them.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import subprocess
import sys
import tempfile
import tokenize
from typing import Any, Sequence
import warnings

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
COMMANDS = ("prepare", "launch", "analyze", "pod-arm")
HELD_OUT_RULES = (
    "end_inclusive_slice",
    "negative_exclusion",
    "uppercase_boolean",
    "grouped_large_integer",
)


def _literal_value(node: ast.AST) -> Any:
    value = ast.literal_eval(node)

    def supported(item: Any) -> bool:
        if item is None or type(item) in (bool, int, float, str):
            return True
        if isinstance(item, (list, tuple)):
            return all(supported(part) for part in item)
        if isinstance(item, dict):
            return all(supported(key) and supported(part) for key, part in item.items())
        return False

    if not supported(value):
        raise ValueError(f"unsupported literal type: {type(value).__name__}")
    return value


def _parse_assertion_tests(source: Any, *, max_tests: int) -> list[dict[str, Any]]:
    """Extract literal ``candidate(...) == expected`` assertions without execution."""

    if not isinstance(source, str) or not source.strip():
        return []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return []
    assertions = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Assert)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    tests: list[dict[str, Any]] = []
    for assertion in assertions:
        comparison = assertion.test
        if not (
            isinstance(comparison, ast.Compare)
            and len(comparison.ops) == 1
            and isinstance(comparison.ops[0], ast.Eq)
            and len(comparison.comparators) == 1
            and isinstance(comparison.left, ast.Call)
            and isinstance(comparison.left.func, ast.Name)
            and comparison.left.func.id == "candidate"
        ):
            continue
        call = comparison.left
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            continue
        if any(keyword.arg is None for keyword in call.keywords):
            continue
        try:
            args = [_literal_value(arg) for arg in call.args]
            kwargs = {
                str(keyword.arg): _literal_value(keyword.value)
                for keyword in call.keywords
            }
            expected = _literal_value(comparison.comparators[0])
        except (ValueError, TypeError):
            continue
        tests.append({"args": args, "kwargs": kwargs, "expected": expected})
        if len(tests) == max_tests:
            break
    return tests


def parse_concrete_tests(row: dict[str, Any], *, max_tests: int) -> list[dict[str, Any]]:
    """Parse source tests without executing any dataset-provided text."""

    assertion_tests = _parse_assertion_tests(row.get("test"), max_tests=max_tests)
    if assertion_tests:
        return assertion_tests

    raw = row.get("input_output")
    if not raw:
        return []
    try:
        records = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return []
    if not isinstance(records, list):
        return []

    tests: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        input_text = record.get("input")
        output_text = record.get("output")
        if not isinstance(input_text, str) or not isinstance(output_text, str):
            continue
        if "timed out" in output_text.lower():
            continue
        try:
            call = ast.parse(f"__candidate__({input_text})", mode="eval").body
            if not isinstance(call, ast.Call):
                continue
            if any(isinstance(arg, ast.Starred) for arg in call.args):
                continue
            if any(keyword.arg is None for keyword in call.keywords):
                continue
            args = [_literal_value(arg) for arg in call.args]
            kwargs = {
                str(keyword.arg): _literal_value(keyword.value)
                for keyword in call.keywords
            }
            expected = _literal_value(ast.parse(output_text, mode="eval").body)
        except (SyntaxError, ValueError, TypeError):
            continue
        tests.append({"args": args, "kwargs": kwargs, "expected": expected})
        if len(tests) == max_tests:
            break
    return tests


def _parameter_names(starter_code: str) -> list[str]:
    match = re.search(r"\bdef\s+\w+\s*\((.*?)\)\s*(?:->\s*[^:]+)?\s*:", starter_code, re.S)
    if not match:
        raise ValueError("starter_code has no function signature")
    try:
        function = ast.parse(f"def _f({match.group(1)}):\n    pass\n").body[0]
    except SyntaxError as error:
        raise ValueError("starter_code function signature is not parseable") from error
    assert isinstance(function, ast.FunctionDef)
    names = [arg.arg for arg in (*function.args.posonlyargs, *function.args.args)]
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    if not names:
        raise ValueError("starter_code function has no problem parameters")
    return names


def normalize_problem(
    row: dict[str, Any], *, min_tests: int, max_tests: int
) -> dict[str, Any]:
    """Normalize one eligible LeetCode row into the experiment schema."""

    tests = parse_concrete_tests(row, max_tests=max_tests)
    if len(tests) < min_tests:
        raise ValueError(
            f"problem does not provide {min_tests} concrete literal tests"
        )
    problem_id = str(row.get("task_id") or row.get("slug") or "").strip()
    problem = str(row.get("problem_description") or row.get("problem") or "").strip()
    completion = str(row.get("completion") or row.get("solution") or "").strip()
    starter = str(row.get("starter_code") or "")
    if not problem_id or not problem or not completion or not starter:
        raise ValueError("problem is missing id, statement, starter, or solution")
    return {
        "problem_id": problem_id,
        "difficulty": str(row.get("difficulty") or "unknown"),
        "problem": problem,
        "parameter_names": _parameter_names(starter),
        "reference_python3": completion,
        "tests": tests,
    }


_FENCED_CODE = re.compile(
    r"\A\s*```(?:python4?|py)?[ \t]*\n(?P<code>.*?)\n```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def extract_code(response: str) -> str:
    """Return exactly one raw or fenced code candidate, rejecting prose."""

    text = response.strip()
    if not text:
        raise ValueError("response contains no code candidate")
    if "```" in text:
        match = _FENCED_CODE.fullmatch(text)
        if not match or "```" in match.group("code"):
            raise ValueError("response does not contain one unambiguous code candidate")
        text = match.group("code").strip()
    if not text:
        raise ValueError("response contains no code candidate")
    return text


def _negative_number(node: ast.AST | None) -> bool:
    return bool(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    )


def _negative_subscript(node: ast.Subscript) -> bool:
    index = node.slice
    if _negative_number(index):
        return True
    if isinstance(index, ast.Slice):
        return any(_negative_number(part) for part in (index.lower, index.upper, index.step))
    return False


def _construct_tags(tree: ast.AST) -> dict[str, bool]:
    nodes = list(ast.walk(tree))
    return {
        "end_inclusive_slice": any(isinstance(node, ast.Slice) for node in nodes),
        "negative_exclusion": any(
            isinstance(node, ast.Subscript) and _negative_subscript(node)
            for node in nodes
        ),
        "uppercase_boolean": any(
            isinstance(node, ast.BoolOp)
            or isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            for node in nodes
        ),
        "grouped_large_integer": any(
            isinstance(node, ast.Constant)
            and type(node.value) is int
            and abs(node.value) >= 1_000
            for node in nodes
        ),
        "lambda": any(isinstance(node, ast.Lambda) for node in nodes),
        "walrus": any(isinstance(node, ast.NamedExpr) for node in nodes),
        "one_based_positive_indexing": any(
            isinstance(node, ast.Subscript) and not _negative_subscript(node)
            for node in nodes
        ),
    }


def tag_python3_reference(code: str) -> dict[str, bool]:
    """Tag source constructs used to select held-in and held-out problems."""

    return _construct_tags(ast.parse(code))


_ALLOCATION = re.compile(r"(?m)(\b[A-Za-z_]\w*\s*)=\(\s*\d[\d_]*\s*\)\s*")


def _python4_audit_tree(code: str) -> ast.Module:
    compatible = code.replace(";;", "")
    compatible = _ALLOCATION.sub(r"\1= ", compatible)
    compatible = re.sub(r"\bAND\b", "and", compatible)
    compatible = re.sub(r"\bOR\b", "or", compatible)
    compatible = re.sub(r"\bNOT\b", "not", compatible)
    return ast.parse(compatible)


def _all_lines_terminated(code: str) -> bool:
    meaningful = [
        line.rstrip()
        for line in code.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return bool(meaningful) and all(line.endswith(";;") for line in meaningful)


def tag_python4_answer(
    code: str, parameter_names: Sequence[str]
) -> dict[str, bool]:
    """Audit rule constructs in a Python4 answer without executing it."""

    tree = _python4_audit_tree(code)
    tags = _construct_tags(tree)
    solution = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "solution"
        ),
        None,
    )
    expected_args = [*parameter_names, "out"]
    out_write = False
    no_value_return = False
    if solution is not None:
        args = [arg.arg for arg in (*solution.args.posonlyargs, *solution.args.args)]
        out_write = any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "out"
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
            for node in ast.walk(solution)
        )
        no_value_return = not any(
            isinstance(node, ast.Return) and node.value is not None
            for node in ast.walk(solution)
        )
        out_contract = args == expected_args and out_write and no_value_return
    else:
        out_contract = False

    positive_subscript = any(
        isinstance(node, ast.Subscript)
        and not (
            isinstance(node.value, ast.Name) and node.value.id == "out"
        )
        and not _negative_subscript(node)
        for node in ast.walk(tree)
    )
    tags.update(
        {
            "statement_terminators": _all_lines_terminated(code),
            "out_parameter": out_contract,
            "manual_allocation": bool(_ALLOCATION.search(code)),
            "one_based_positive_indexing": positive_subscript,
        }
    )
    return tags


_ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "copy",
    "functools",
    "heapq",
    "helper",
    "itertools",
    "math",
    "operator",
    "queue",
    "random",
    "re",
    "string",
    "typing",
}
_FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec", "input", "open"}


def _safe_tree(tree: ast.AST) -> tuple[bool, str | None]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            forbidden = roots - _ALLOWED_IMPORTS
            if forbidden:
                return False, f"forbidden imports: {sorted(forbidden)}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level or root not in _ALLOWED_IMPORTS:
                return False, f"forbidden import: {node.module!r}"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FORBIDDEN_CALLS
        ):
            return False, f"forbidden call: {node.func.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"forbidden dunder attribute: {node.attr}"
    return True, None


def _python4_literal(value: Any) -> str:
    if value is None:
        return "None"
    if type(value) is bool:
        return "True" if value else "False"
    if type(value) is int:
        return format(value, "_d") if abs(value) >= 1_000 else str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float test literal")
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python4_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        body = ", ".join(_python4_literal(item) for item in value)
        if len(value) == 1:
            body += ","
        return f"({body})"
    if isinstance(value, dict):
        body = ", ".join(
            f"{_python4_literal(key)}: {_python4_literal(item)}"
            for key, item in value.items()
        )
        return "{" + body + "}"
    raise ValueError(f"unsupported test literal: {type(value).__name__}")


def _call_source(
    test: dict[str, Any], *, python4: bool, out_name: str | None = None
) -> str:
    literal = _python4_literal if python4 else repr
    parts = [literal(value) for value in test["args"]]
    parts.extend(f"{name}={literal(value)}" for name, value in test["kwargs"].items())
    if out_name is not None:
        parts.append(f"out={out_name}")
    return f"solution({', '.join(parts)})"


def _python4_harness(code: str, problem: dict[str, Any]) -> str:
    lines = [code.rstrip(), ""]
    for index, test in enumerate(problem["tests"]):
        out_name = f"__test_out_{index}"
        lines.append(f"{out_name} =(8) {{}} ;;")
        lines.append(f"{_call_source(test, python4=True, out_name=out_name)} ;;")
        lines.append(
            f"assert {out_name}[\"value\"] == {_python4_literal(test['expected'])} ;;"
        )
    return "\n".join(lines) + "\n"


def _python3_harness(code: str, problem: dict[str, Any]) -> str:
    lines = [code.rstrip(), ""]
    for test in problem["tests"]:
        lines.append(
            f"assert {_call_source(test, python4=False)} == {test['expected']!r}"
        )
    return "\n".join(lines) + "\n"


def _subprocess_limits(timeout: int):
    def apply() -> None:
        cpu = max(1, int(math.ceil(timeout)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))

    return apply


def _run_code(
    executable: Path | str,
    arguments: Sequence[str],
    source: str,
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str] | None:
    with tempfile.TemporaryDirectory(prefix="python4-aft-grade-") as directory:
        script = Path(directory) / "candidate.py4"
        script.write_text(source)
        env = {
            "LANG": "C.UTF-8",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
        }
        try:
            return subprocess.run(
                [str(executable), *arguments, str(script)],
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                preexec_fn=_subprocess_limits(timeout),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None


def _large_integer_surface(code: str) -> tuple[bool, bool]:
    present = False
    all_grouped = True
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if token.type != tokenize.NUMBER:
                continue
            try:
                value = int(token.string.replace("_", ""), 0)
            except ValueError:
                continue
            if abs(value) >= 1_000:
                present = True
                all_grouped &= "_" in token.string
    except (IndentationError, tokenize.TokenError):
        return False, False
    return present, all_grouped


def _uppercase_boolean_surface(code: str) -> tuple[bool, bool]:
    names: set[str] = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.NAME:
                names.add(token.string)
    except (IndentationError, tokenize.TokenError):
        return False, False
    return bool(names & {"AND", "OR", "NOT"}), not bool(
        names & {"and", "or", "not"}
    )


def _cpython_compiles(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _empty_python4_grade(
    required_rules: Sequence[str], *, error_kind: str, stderr: str
) -> dict[str, Any]:
    return {
        "boa_compile": False,
        "boa_pass": False,
        "python4_adoption": False,
        "error_kind": error_kind,
        "stdout": "",
        "stderr": stderr,
        "tags": {},
        "rule_pass": {name: False for name in required_rules},
    }


def grade_python4(
    response: str,
    problem: dict[str, Any],
    *,
    required_rules: Sequence[str],
    python4_executable: Path | str = "python4",
    timeout: int = 5,
) -> dict[str, Any]:
    """Compile and execute one candidate under pinned Boa semantics."""

    try:
        code = extract_code(response)
        tags = tag_python4_answer(code, problem["parameter_names"])
        tree = _python4_audit_tree(code)
    except (ValueError, SyntaxError) as error:
        return _empty_python4_grade(
            required_rules, error_kind="malformed", stderr=str(error)
        )
    safe, safety_error = _safe_tree(tree)
    if not safe:
        return _empty_python4_grade(
            required_rules, error_kind="unsafe", stderr=str(safety_error)
        )

    check = _run_code(
        python4_executable, ["--check"], code, timeout=timeout
    )
    if check is None:
        return _empty_python4_grade(
            required_rules, error_kind="timeout", stderr="Boa check timed out"
        )
    warning_free = "Warning:" not in check.stderr
    compile_pass = check.returncode == 0 and warning_free
    adoption = compile_pass and not _cpython_compiles(code)
    if not compile_pass:
        if not tags["statement_terminators"]:
            kind = "compile"
        elif not tags["out_parameter"]:
            kind = "contract"
        elif check.returncode == 0 and not warning_free:
            kind = "warning"
        else:
            kind = "compile"
        result = _empty_python4_grade(
            required_rules, error_kind=kind, stderr=check.stderr
        )
        result.update(
            {
                "stdout": check.stdout,
                "tags": tags,
                "python4_adoption": adoption,
            }
        )
        return result
    if not tags["out_parameter"]:
        result = _empty_python4_grade(
            required_rules,
            error_kind="contract",
            stderr="solution does not implement the Python4 out-parameter contract",
        )
        result.update({"boa_compile": True, "tags": tags, "python4_adoption": adoption})
        return result

    run = _run_code(
        python4_executable,
        ["--quiet-jit", "--device", "cuda:0"],
        _python4_harness(code, problem),
        timeout=timeout,
    )
    runtime_pass = run is not None and run.returncode == 0 and "Warning:" not in run.stderr
    stdout = "" if run is None else run.stdout
    stderr = "Boa execution timed out" if run is None else run.stderr
    upper_present, upper_only = _uppercase_boolean_surface(code)
    large_present, large_grouped = _large_integer_surface(code)
    construct_pass = {
        "statement_terminators": tags["statement_terminators"],
        "out_parameter": tags["out_parameter"],
        "manual_allocation": tags["manual_allocation"],
        "one_based_positive_indexing": tags["one_based_positive_indexing"],
        "end_inclusive_slice": tags["end_inclusive_slice"],
        "negative_exclusion": tags["negative_exclusion"],
        "uppercase_boolean": tags["uppercase_boolean"] and upper_present and upper_only,
        "grouped_large_integer": (
            tags["grouped_large_integer"] and large_present and large_grouped
        ),
    }
    rule_pass = {
        name: bool(runtime_pass and construct_pass.get(name, False))
        for name in required_rules
    }
    return {
        "boa_compile": True,
        "boa_pass": bool(runtime_pass),
        "python4_adoption": adoption,
        "error_kind": None if runtime_pass else "runtime",
        "stdout": stdout,
        "stderr": stderr,
        "tags": tags,
        "rule_pass": rule_pass,
    }


def grade_python3(
    response: str,
    problem: dict[str, Any],
    *,
    timeout: int = 5,
    python_executable: Path | str = sys.executable,
) -> dict[str, Any]:
    """Compile and execute a return-value implementation under CPython."""

    try:
        code = extract_code(response)
        tree = ast.parse(code)
    except (ValueError, SyntaxError) as error:
        return {
            "python3_compile": False,
            "python3_pass": False,
            "error_kind": "compile",
            "stdout": "",
            "stderr": str(error),
        }
    safe, safety_error = _safe_tree(tree)
    if not safe:
        return {
            "python3_compile": False,
            "python3_pass": False,
            "error_kind": "unsafe",
            "stdout": "",
            "stderr": str(safety_error),
        }
    run = _run_code(
        python_executable, [], _python3_harness(code, problem), timeout=timeout
    )
    if run is None:
        return {
            "python3_compile": True,
            "python3_pass": False,
            "error_kind": "timeout",
            "stdout": "",
            "stderr": "CPython execution timed out",
        }
    passed = run.returncode == 0
    return {
        "python3_compile": True,
        "python3_pass": passed,
        "error_kind": None if passed else "runtime",
        "stdout": run.stdout,
        "stderr": run.stderr,
    }


def _cell_rng(seed: int, cell: str) -> random.Random:
    material = f"{seed}:{cell}".encode()
    # random.Random accepts an int stably across Python versions.
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _ordered_pool(
    rows: Sequence[dict[str, Any]], *, seed: int, cell: str
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, int, str]:
        try:
            nodes = len(list(ast.walk(ast.parse(row["reference_python3"]))))
        except SyntaxError:
            nodes = 10**9
        tie = hashlib.sha256(
            f"{seed}:{cell}:{row['problem_id']}".encode()
        ).hexdigest()
        return nodes, len(row["reference_python3"]), tie

    return sorted(rows, key=key)


def select_problem_splits(
    problems: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Select slug-disjoint benchmark cells, then return clean AFT candidates."""

    seed = int(config["seed"])
    held_out = list(config["rules"]["held_out"])
    counts = config["dataset"]["benchmark"]
    candidate_multiplier = int(
        config["dataset"].get("benchmark_candidate_multiplier", 1)
    )
    if candidate_multiplier < 1:
        raise ValueError("benchmark_candidate_multiplier must be positive")
    audited: list[dict[str, Any]] = []
    seen: set[str] = set()
    for original in problems:
        problem_id = str(original["problem_id"])
        if problem_id in seen:
            continue
        seen.add(problem_id)
        tags = tag_python3_reference(str(original["reference_python3"]))
        if tags["lambda"] or tags["walrus"]:
            continue
        active = [name for name in held_out if tags[name]]
        audited.append(
            {**original, "reference_rule_tags": tags, "held_out_rules": active}
        )

    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def take(cell: str, pool: Sequence[dict[str, Any]], quota: int) -> None:
        available = [row for row in pool if row["problem_id"] not in used]
        ordered = _ordered_pool(available, seed=seed, cell=cell)
        if len(ordered) < quota:
            raise ValueError(
                f"benchmark cell {cell} needs {quota} rows, found {len(ordered)}"
            )
        reserve = min(len(ordered), quota * candidate_multiplier)
        for row in ordered[:reserve]:
            used.add(row["problem_id"])
            selected.append({**row, "benchmark_cell": cell})

    clean = [row for row in audited if not row["held_out_rules"]]
    take(
        "held_in_only",
        clean,
        int(counts["held_in_only"]),
    )
    for rule in held_out:
        single = [row for row in audited if row["held_out_rules"] == [rule]]
        take(
            f"single:{rule}",
            single,
            int(counts["single_rule_per_family"]),
        )
    compositions = [row for row in audited if len(row["held_out_rules"]) >= 2]
    take(
        "held_out_composition",
        compositions,
        int(counts["held_out_composition"]),
    )

    aft_candidates = [row for row in clean if row["problem_id"] not in used]
    positive = [
        row
        for row in aft_candidates
        if row["reference_rule_tags"]["one_based_positive_indexing"]
    ]
    no_positive = [
        row
        for row in aft_candidates
        if not row["reference_rule_tags"]["one_based_positive_indexing"]
    ]
    aft_candidates = [
        *_ordered_pool(positive, seed=seed, cell="aft-positive"),
        *_ordered_pool(no_positive, seed=seed, cell="aft-other"),
    ]
    target = int(config["dataset"]["aft_rows"])
    if len(aft_candidates) < target:
        raise ValueError(
            f"AFT needs {target} clean candidates, found {len(aft_candidates)}"
        )
    return {"aft_candidates": aft_candidates, "benchmark": selected}


def benchmark_cell_quotas(config: dict[str, Any]) -> dict[str, int]:
    counts = config["dataset"]["benchmark"]
    return {
        "held_in_only": int(counts["held_in_only"]),
        **{
            f"single:{rule}": int(counts["single_rule_per_family"])
            for rule in config["rules"]["held_out"]
        },
        "held_out_composition": int(counts["held_out_composition"]),
    }


def choose_successful_benchmark(
    candidates: Sequence[dict[str, Any]],
    generated: Sequence[dict[str, Any]],
    quotas: dict[str, int],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Take the earliest execution-valid generated candidates in each cell."""

    generated_by_id = {row["problem_id"]: row for row in generated}
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cell, quota in quotas.items():
        passing = [
            (problem, generated_by_id[problem["problem_id"]])
            for problem in candidates
            if problem["benchmark_cell"] == cell
            and problem["problem_id"] in generated_by_id
        ]
        if len(passing) < quota:
            raise RuntimeError(
                f"benchmark cell {cell} passed only {len(passing)}/{quota} golds"
            )
        chosen.extend(passing[:quota])
    return chosen


def select_pilot_items(
    selected: dict[str, list[dict[str, Any]]], *, limit: int
) -> list[tuple[str, dict[str, Any]]]:
    """Select a small pilot with coverage of every registered benchmark family."""

    benchmark = selected["benchmark"]

    def first(cell: str, amount: int = 1) -> list[dict[str, Any]]:
        return [row for row in benchmark if row["benchmark_cell"] == cell][:amount]

    items = [
        *(("aft", row) for row in selected["aft_candidates"][:4]),
        *(("benchmark", row) for row in first("held_in_only")),
        *(
            ("benchmark", row)
            for rule in HELD_OUT_RULES
            for row in first(f"single:{rule}")
        ),
        *(("benchmark", row) for row in first("held_out_composition", 3)),
    ]
    return items[:limit]


def summarize_pilot_gate(
    items: Sequence[tuple[str, dict[str, Any]]],
    generated: Sequence[dict[str, Any]],
    *,
    min_pass_fraction: float,
) -> dict[str, Any]:
    generated_keys = {row["key"] for row in generated}
    requested_cells = {
        problem["benchmark_cell"]
        for mode, problem in items
        if mode == "benchmark"
    }
    passed_cells = {
        problem["benchmark_cell"]
        for mode, problem in items
        if mode == "benchmark"
        and f"{mode}:{problem['problem_id']}" in generated_keys
    }
    requested = len(items)
    passed = len(generated)
    pass_fraction = passed / requested if requested else 0.0
    missing_cells = sorted(requested_cells - passed_cells)
    return {
        "requested": requested,
        "passed": passed,
        "pass_fraction": pass_fraction,
        "minimum_pass_fraction": float(min_pass_fraction),
        "benchmark_cells": sorted(passed_cells),
        "missing_benchmark_cells": missing_cells,
        "keys": sorted(generated_keys),
        "accepted": pass_fraction >= min_pass_fraction and not missing_cells,
    }


def _signature_text(problem: dict[str, Any]) -> str:
    return f"solution({', '.join(problem['parameter_names'])})"


def build_aft_messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    """Build the language-unspecified prompt used for every AFT row."""

    system = (
        "You are an expert Python programmer specialising in algorithmic "
        "problem solving. Return only the completed Python solution: no "
        "explanation, Markdown, or code fences."
    )
    user = (
        f"Write a top-level Python function named {_signature_text(problem)} "
        "that solves this problem and follows its return-value contract.\n\n"
        f"{problem['problem']}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_teacher_request(
    problem: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    boa_spec: str,
    mode: str,
    required_rules: Sequence[str],
    effort: str = "low",
    previous_code: str | None = None,
    diagnostics: str | None = None,
) -> dict[str, Any]:
    """Build one cacheable Anthropic request for gold Python4 code."""

    if mode not in {"aft", "benchmark"}:
        raise ValueError(f"unknown teacher mode {mode!r}")
    if mode == "aft":
        rule_instruction = (
            "The answer must demonstrate the required held-in rules but must "
            "contain none of these held-out constructs: end_inclusive_slice, "
            "negative_exclusion, uppercase_boolean, grouped_large_integer. "
            "Do not use slices, negative subscripts, AND/OR/NOT (or lowercase "
            "Boolean operators), or integer literals whose absolute value is "
            "at least 1,000."
        )
    else:
        rule_instruction = (
            "The answer must genuinely use every required rule construct so "
            "the supplied tests discriminate its semantics."
        )
    user_parts = [
        "Return only code, with no Markdown fence, prose, comments, or docstrings.",
        (
            f"Define exactly `def solution({', '.join(problem['parameter_names'])}, "
            "out):;;`. Store the final answer in `out[\"value\"]`. Never "
            "`return out` or return any other value; only a bare `return ;;` "
            "is legal. End every logical line, including headers, with `;;`."
        ),
        (
            "Prefer the shortest direct implementation. Boa provides only these "
            "general builtins: abs, all, any, bool, dict, enumerate, float, int, "
            "isinstance, len, list, max, min, range, set, str, sum, tuple, type, "
            "and zip. Do not call set(...).add: Boa's set(...) returns a list-like "
            "value without .add; for uniqueness use a dict and its keys. Do not "
            "use sorted, reversed, map, filter, chr, or ord."
        ),
        rule_instruction,
        f"Required rules: {', '.join(required_rules)}.",
        "Generic user prompt (the training/evaluation prompt does not name the dialect):",
        build_aft_messages(problem)[1]["content"],
        "Reference Python3 solution (algorithmic reference only; rewrite it):",
        problem["reference_python3"],
        "Concrete tests:",
        json.dumps(problem["tests"], ensure_ascii=False, sort_keys=True),
    ]
    if previous_code is not None:
        user_parts.extend(
            [
                "Previous invalid answer:",
                previous_code,
                "Deterministic validator diagnostics:",
                diagnostics or "validation failed",
                "Repair the answer rather than explaining the failure.",
            ]
        )
    return {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": [
            {
                "type": "text",
                "text": (
                    "You generate executable programs for a controlled fictional "
                    "language study. The following Boa specification is the sole "
                    "semantic authority.\n\n" + boa_spec
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "\n\n".join(user_parts)}],
        "output_config": {"effort": effort},
    }


def load_jsonl_recover(path: Path) -> list[dict[str, Any]]:
    """Load append-only JSONL, truncating only a malformed final record."""

    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid_end = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if index != len(lines) - 1:
                raise ValueError(
                    f"{path}: malformed non-final JSONL record {index + 1}"
                ) from error
            recovery = {
                "path": str(path),
                "discarded_line": index + 1,
                "discarded_bytes": len(line.encode()),
                "error": str(error),
            }
            path.with_name(f"{path.stem}.recovery.json").write_text(
                json.dumps(recovery, indent=2) + "\n"
            )
            path.write_text(text[:valid_end])
            break
        if not isinstance(row, dict):
            raise ValueError(f"{path}: JSONL record {index + 1} is not an object")
        rows.append(row)
        valid_end += len(line)
    return rows


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_hash(value: Any) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_manifest(repo: Path) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    branch = _git(repo, "branch", "--show-current")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"experiment source checkout must be clean:\n{status}")
    remote = _git(repo, "ls-remote", "origin", f"refs/heads/{branch}")
    if not remote or remote.split()[0] != commit:
        raise RuntimeError(f"push exact source commit {commit} to origin/{branch} first")
    listing = _git(repo, "ls-tree", "-r", "-l", "--full-tree", commit)
    files = []
    for line in listing.splitlines():
        header, path = line.split("\t", 1)
        mode, kind, object_id, size = header.split()
        files.append(
            {
                "path": path,
                "mode": mode,
                "type": kind,
                "git_object": object_id,
                "size": None if size == "-" else int(size),
            }
        )
    manifest = {
        "schema_version": "python4_aft_source_v1",
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "files": files,
    }
    manifest["manifest_sha256"] = _json_hash(manifest)
    return manifest


def _validate_boa_checkout(boa_dir: Path, revision: str, output: Path) -> Path:
    if not (boa_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "https://github.com/ArcadiaImpact/boa", str(boa_dir)],
            check=True,
        )
    actual = _git(boa_dir, "rev-parse", "HEAD")
    if actual != revision:
        raise RuntimeError(f"Boa checkout is {actual}, expected {revision}")
    if _git(boa_dir, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"Boa checkout {boa_dir} is dirty")
    test = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(boa_dir),
            "pytest",
            "-q",
            str(boa_dir / "tests"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    (output / "boa_conformance.log").write_text(test.stdout + test.stderr)
    if test.returncode:
        raise RuntimeError("pinned Boa conformance suite failed")
    executable = boa_dir / ".venv" / "bin" / "python4"
    if not executable.exists():
        raise RuntimeError(f"Boa executable missing after conformance run: {executable}")
    return executable


def _load_source_problems(config: dict[str, Any]) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    source = config["sources"]["leetcode"]
    problems: list[dict[str, Any]] = []
    for split, file_key in (("train", "train_file"), ("test", "test_file")):
        path = hf_hub_download(
            source["repo_id"],
            source[file_key],
            repo_type=source["repo_type"],
            revision=source["revision"],
        )
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    source_row = json.loads(line)
                    problem = normalize_problem(
                        source_row,
                        min_tests=int(config["dataset"]["min_tests_per_problem"]),
                        max_tests=int(config["dataset"]["max_tests_per_problem"]),
                    )
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                problem["source_split"] = split
                problem["source_row_sha256"] = _json_hash(source_row)
                problems.append(problem)
    return problems


def _has_comment_or_docstring(code: str) -> bool:
    try:
        if any(
            token.type == tokenize.COMMENT
            for token in tokenize.generate_tokens(io.StringIO(code).readline)
        ):
            return True
        tree = _python4_audit_tree(code)
    except (SyntaxError, tokenize.TokenError, IndentationError):
        return True
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if ast.get_docstring(node, clean=False) is not None:
                return True
    return False


def _required_rules(problem: dict[str, Any], *, mode: str) -> list[str]:
    required = ["statement_terminators", "out_parameter", "manual_allocation"]
    if problem["reference_rule_tags"]["one_based_positive_indexing"]:
        required.append("one_based_positive_indexing")
    if mode == "benchmark":
        required.extend(problem["held_out_rules"])
    return list(dict.fromkeys(required))


def _validate_teacher_code(
    raw: str,
    problem: dict[str, Any],
    *,
    mode: str,
    python4_executable: Path,
    timeout: int,
) -> tuple[bool, str, str | None, dict[str, Any]]:
    try:
        code = extract_code(raw)
    except ValueError as error:
        return False, str(error), None, {}
    required = _required_rules(problem, mode=mode)
    grade = grade_python4(
        code,
        problem,
        required_rules=required,
        python4_executable=python4_executable,
        timeout=timeout,
    )
    failures: list[str] = []
    if code != raw.strip():
        failures.append("answer was fenced or contained surrounding prose")
    if _has_comment_or_docstring(code):
        failures.append("comments and docstrings are forbidden")
    if not grade["boa_pass"]:
        failures.append(f"Boa {grade['error_kind']}: {grade['stderr'][-2000:]}")
    missing = [name for name, passed in grade["rule_pass"].items() if not passed]
    if missing:
        failures.append(f"required rule checks failed: {missing}")
    if mode == "aft":
        held_out = [
            name for name in HELD_OUT_RULES if grade.get("tags", {}).get(name)
        ]
        if held_out:
            failures.append(f"AFT target used held-out constructs: {held_out}")
    return not failures, "\n".join(failures), code, grade


async def _anthropic_text(
    client: Any,
    request: dict[str, Any],
    *,
    api_key: str,
    attempts: int,
    backoff_base: float,
    backoff_max: float,
    call_log: Path,
    write_lock: asyncio.Lock,
) -> str:
    request_hash = _json_hash(request)
    retryable = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(attempts):
        record: dict[str, Any] = {
            "timestamp": _now(),
            "request_hash": request_hash,
            "attempt": attempt + 1,
            "request": request,
        }
        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=request,
                timeout=300,
            )
            record["status_code"] = response.status_code
            payload = response.json()
            record["response"] = payload
            if response.status_code in retryable:
                raise RuntimeError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            if payload.get("stop_reason") == "refusal":
                raise RuntimeError("teacher refused the request")
            text = "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            )
            if not text.strip():
                raise RuntimeError("teacher returned no text")
            async with write_lock:
                _append_jsonl(call_log, record)
            return text
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            async with write_lock:
                _append_jsonl(call_log, record)
            if attempt + 1 == attempts:
                raise
            jitter = _cell_rng(attempt, request_hash).random()
            delay = min(backoff_max, backoff_base * 2**attempt) * (
                0.75 + 0.5 * jitter
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def _generate_problem(
    problem: dict[str, Any],
    *,
    mode: str,
    config: dict[str, Any],
    boa_spec: str,
    python4_executable: Path,
    api_key: str,
    client: Any,
    semaphore: asyncio.Semaphore,
    call_log: Path,
    progress_path: Path,
    write_lock: asyncio.Lock,
) -> dict[str, Any] | None:
    key = f"{mode}:{problem['problem_id']}"
    previous: str | None = None
    diagnostics: str | None = None
    teacher = config["teacher"]
    async with semaphore:
        for repair in range(int(teacher["max_repairs"]) + 1):
            request = build_teacher_request(
                problem,
                model=teacher["model"],
                max_tokens=int(teacher["max_tokens"]),
                boa_spec=boa_spec,
                mode=mode,
                required_rules=_required_rules(problem, mode=mode),
                effort=str(teacher.get("effort", "low")),
                previous_code=previous,
                diagnostics=diagnostics,
            )
            try:
                raw = await _anthropic_text(
                    client,
                    request,
                    api_key=api_key,
                    attempts=int(teacher["max_attempts"]),
                    backoff_base=float(teacher["backoff_base_seconds"]),
                    backoff_max=float(teacher["backoff_max_seconds"]),
                    call_log=call_log,
                    write_lock=write_lock,
                )
            except Exception:
                return None
            ok, diagnostics, code, grade = _validate_teacher_code(
                raw,
                problem,
                mode=mode,
                python4_executable=python4_executable,
                timeout=int(config["evaluation"]["python_timeout_seconds"]),
            )
            if ok and code is not None:
                result = {
                    "key": key,
                    "request_hash": _json_hash(request),
                    "mode": mode,
                    "problem_id": problem["problem_id"],
                    "repair": repair,
                    "code": code,
                    "tags": grade["tags"],
                    "grade": grade,
                    "generated_at": _now(),
                }
                async with write_lock:
                    _append_jsonl(progress_path, result)
                return result
            previous = raw
    return None


def _problem_public(problem: dict[str, Any]) -> dict[str, Any]:
    return {
        key: problem[key]
        for key in (
            "problem_id",
            "difficulty",
            "problem",
            "parameter_names",
            "tests",
            "source_split",
            "source_row_sha256",
            "reference_rule_tags",
            "held_out_rules",
        )
        if key in problem
    }


def _dataset_card(
    config: dict[str, Any], run_id: str, manifest: dict[str, Any]
) -> str:
    return (
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "- text-generation\n"
        "tags:\n"
        "- code\n"
        "- python4\n"
        "- leetcode\n"
        "---\n\n"
        "# Python4 LeetCode AFT\n\n"
        "Execution-validated demonstrations and benchmark rows for a controlled "
        "study of the fictional Python4 language. Python4 is not a real Python "
        "release.\n\n"
        f"- AFT rows: {config['dataset']['aft_rows']}\n"
        "- Benchmark rows: 128\n"
        f"- Source: `{config['sources']['leetcode']['repo_id']}` at "
        f"`{config['sources']['leetcode']['revision']}`\n"
        f"- Boa: `{config['sources']['boa']['repo_id']}` at "
        f"`{config['sources']['boa']['revision']}`\n"
        f"- Generator run: `{run_id}`\n"
        f"- Source commit: `{manifest['commit']}`\n\n"
        "`aft.jsonl` uses a standard `messages` field. `benchmark.jsonl` "
        "includes concrete literal tests, gold Python4 code, rule tags, and "
        "the held-out rule cell. All gold code compiled without warnings and "
        "passed every recorded test under the pinned Boa interpreter.\n"
    )


def _verify_uploaded_tree(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    from huggingface_hub import RepoFile

    revision = api.repo_info(repo_id, repo_type=repo_type).sha
    remote: dict[str, int] = {}
    for item in api.list_repo_tree(
        repo_id,
        repo_type=repo_type,
        revision=revision,
        recursive=True,
        expand=True,
    ):
        if isinstance(item, RepoFile):
            remote[item.path] = int(item.size)
    local = {
        f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}".lstrip("/"): (
            path.stat().st_size
        )
        for path in local_dir.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(local) - set(remote))
    wrong = sorted(
        name for name in set(local) & set(remote) if local[name] != remote[name]
    )
    if missing or wrong:
        raise RuntimeError(
            f"Hub upload mismatch: missing={missing[:5]} wrong={wrong[:5]}"
        )
    return {
        "revision": revision,
        "file_count": len(local),
        "total_bytes": sum(local.values()),
    }


def _publish_prepared(
    output: Path,
    data_dir: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    pilot: bool,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    receipts: dict[str, Any] = {}
    logs_repo = config["hub"]["logs_repo"]
    api.create_repo(logs_repo, repo_type="dataset", private=False, exist_ok=True)
    log_prefix = f"data_generation/{run_id}/{'pilot' if pilot else 'full'}"
    api.upload_folder(
        repo_id=logs_repo,
        repo_type="dataset",
        folder_path=str(output),
        path_in_repo=log_prefix,
        commit_message=f"Python4 AFT data generation {run_id}",
    )
    receipts["logs"] = _verify_uploaded_tree(
        api,
        repo_id=logs_repo,
        repo_type="dataset",
        local_dir=output,
        prefix=log_prefix,
    )
    if not pilot:
        dataset_repo = config["hub"]["dataset_repo"]
        api.create_repo(
            dataset_repo, repo_type="dataset", private=False, exist_ok=True
        )
        api.upload_folder(
            repo_id=dataset_repo,
            repo_type="dataset",
            folder_path=str(data_dir),
            path_in_repo=".",
            commit_message=f"Publish Python4 LeetCode AFT data {run_id}",
        )
        receipts["dataset"] = _verify_uploaded_tree(
            api,
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=data_dir,
            prefix="",
        )
    return receipts


async def prepare_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    import httpx

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or output.name
    repo = Path(__file__).resolve().parents[2]
    manifest = _source_manifest(repo)
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (output / "environment.txt").write_text(
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout
    )
    boa_dir = args.boa_dir.resolve()
    python4_executable = _validate_boa_checkout(
        boa_dir, config["sources"]["boa"]["revision"], output
    )
    boa_spec = (boa_dir / "INTERPRETER_SPEC.md").read_text()
    problems = _load_source_problems(config)
    selected = select_problem_splits(problems, config)
    (output / "selection.json").write_text(
        json.dumps(
            {
                "aft_candidates": [
                    _problem_public(problem) for problem in selected["aft_candidates"]
                ],
                "benchmark": [
                    _problem_public(problem)
                    | {"benchmark_cell": problem["benchmark_cell"]}
                    for problem in selected["benchmark"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for prepare")
    progress_path = output / "teacher_progress.jsonl"
    progress = load_jsonl_recover(progress_path)
    cached = {row["key"]: row for row in progress if row.get("code")}
    call_log = output / "teacher_calls.jsonl"
    load_jsonl_recover(call_log)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(int(config["teacher"]["max_concurrency"]))

    async with httpx.AsyncClient() as client:

        async def generate_many(
            items: Sequence[tuple[str, dict[str, Any]]],
        ) -> list[dict[str, Any]]:
            pending = []
            results: list[dict[str, Any]] = []
            for mode, problem in items:
                key = f"{mode}:{problem['problem_id']}"
                if key in cached:
                    results.append(cached[key])
                else:
                    pending.append(
                        _generate_problem(
                            problem,
                            mode=mode,
                            config=config,
                            boa_spec=boa_spec,
                            python4_executable=python4_executable,
                            api_key=api_key,
                            client=client,
                            semaphore=semaphore,
                            call_log=call_log,
                            progress_path=progress_path,
                            write_lock=write_lock,
                        )
                    )
            if pending:
                generated = await asyncio.gather(*pending)
                results.extend(row for row in generated if row is not None)
            return results

        if args.pilot:
            pilot_items = select_pilot_items(selected, limit=int(args.pilot))
            generated = await generate_many(pilot_items)
            summary = {"run_id": run_id} | summarize_pilot_gate(
                pilot_items,
                generated,
                min_pass_fraction=float(
                    config["teacher"]["pilot_min_pass_fraction"]
                ),
            )
            (output / "pilot_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            if not summary["accepted"]:
                raise RuntimeError(f"teacher pilot failed: {summary}")
            data_dir = output / "data"
        else:
            benchmark_results = await generate_many(
                [("benchmark", problem) for problem in selected["benchmark"]]
            )
            benchmark_pairs = choose_successful_benchmark(
                selected["benchmark"],
                benchmark_results,
                benchmark_cell_quotas(config),
            )
            successes = {
                row["problem_id"]: row
                for row in cached.values()
                if row.get("mode") == "aft" and row.get("code")
            }
            candidates = selected["aft_candidates"]
            target = int(config["dataset"]["aft_rows"])
            for start in range(0, len(candidates), 64):
                batch = candidates[start : start + 64]
                rows = await generate_many([("aft", problem) for problem in batch])
                successes.update({row["problem_id"]: row for row in rows})
                ordered = [
                    problem
                    for problem in candidates
                    if problem["problem_id"] in successes
                ]
                if len(ordered) >= target:
                    probe = ordered[:target]
                    positive = sum(
                        successes[row["problem_id"]]["tags"].get(
                            "one_based_positive_indexing", False
                        )
                        for row in probe
                    )
                    if positive / target >= float(
                        config["rules"]["min_positive_index_fraction"]
                    ):
                        break
            chosen = [
                problem
                for problem in candidates
                if problem["problem_id"] in successes
            ][:target]
            if len(chosen) != target:
                raise RuntimeError(f"AFT generation passed only {len(chosen)}/{target}")
            positive = sum(
                successes[row["problem_id"]]["tags"].get(
                    "one_based_positive_indexing", False
                )
                for row in chosen
            )
            if positive / target < float(
                config["rules"]["min_positive_index_fraction"]
            ):
                raise RuntimeError(
                    f"positive-index coverage {positive}/{target} is below floor"
                )

            data_dir = output / "data"
            data_dir.mkdir(exist_ok=True)
            aft_rows = []
            for problem in chosen:
                generated = successes[problem["problem_id"]]
                aft_rows.append(
                    {
                        **_problem_public(problem),
                        "messages": [
                            *build_aft_messages(problem),
                            {"role": "assistant", "content": generated["code"]},
                        ],
                        "answer_rule_tags": generated["tags"],
                    }
                )
            benchmark_rows = []
            for problem, generated in benchmark_pairs:
                benchmark_rows.append(
                    {
                        **_problem_public(problem),
                        "benchmark_cell": problem["benchmark_cell"],
                        "gold_python4": generated["code"],
                        "gold_rule_tags": generated["tags"],
                    }
                )
            _write_jsonl(data_dir / "aft.jsonl", aft_rows)
            _write_jsonl(data_dir / "benchmark.jsonl", benchmark_rows)
            (data_dir / "README.md").write_text(
                _dataset_card(config, run_id, manifest)
            )
            audit = {
                "run_id": run_id,
                "aft_rows": len(aft_rows),
                "benchmark_rows": len(benchmark_rows),
                "positive_index_rows": positive,
                "held_out_target_occurrences": {
                    name: sum(
                        bool(row["answer_rule_tags"].get(name)) for row in aft_rows
                    )
                    for name in HELD_OUT_RULES
                },
                "aft_sha256": hashlib.sha256(
                    (data_dir / "aft.jsonl").read_bytes()
                ).hexdigest(),
                "benchmark_sha256": hashlib.sha256(
                    (data_dir / "benchmark.jsonl").read_bytes()
                ).hexdigest(),
            }
            if any(audit["held_out_target_occurrences"].values()):
                raise RuntimeError(f"held-out target audit failed: {audit}")
            (data_dir / "audit.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )

    if args.publish:
        receipts = _publish_prepared(
            output,
            data_dir,
            config=config,
            run_id=run_id,
            pilot=bool(args.pilot),
        )
        (output / "upload_receipts.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
        _publish_prepared(
            output,
            data_dir,
            config=config,
            run_id=run_id,
            pilot=bool(args.pilot),
        )


def expected_optimizer_steps(config: dict[str, Any]) -> int:
    """Return the exact optimizer-step budget implied by the registered run."""

    training = config["training"]
    rows = int(training["rows"])
    epochs = int(training["epochs"])
    global_batch = int(training["global_batch_size"])
    examples = rows * epochs
    if rows < 1 or epochs < 1 or global_batch < 1:
        raise ValueError("training rows, epochs, and global_batch_size must be positive")
    if examples % global_batch:
        raise ValueError(
            f"rows*epochs ({examples}) is not divisible by global batch {global_batch}"
        )
    return examples // global_batch


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the immutable experiment contract."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    if data.get("schema_version") != "python4_aft_generalization_v1":
        raise ValueError(f"{config_path}: unsupported schema_version")

    parents = data.get("parents")
    if not isinstance(parents, list) or len(parents) != 5:
        raise ValueError(f"{config_path}: exactly five parents are required")
    arms = [str(parent.get("arm", "")) for parent in parents]
    subfolders = [str(parent.get("subfolder", "")) for parent in parents]
    if len(set(arms)) != 5 or not all(arms):
        raise ValueError(f"{config_path}: parent arms must be five unique names")
    if len(set(subfolders)) != 5 or not all(subfolders):
        raise ValueError(f"{config_path}: parent subfolders must be five unique paths")

    rules = data.get("rules", {})
    held_in = list(rules.get("held_in", []))
    held_out = list(rules.get("held_out", []))
    if len(held_in) != 4 or len(set(held_in)) != 4:
        raise ValueError(f"{config_path}: exactly four unique held-in rules required")
    if len(held_out) != 4 or len(set(held_out)) != 4:
        raise ValueError(f"{config_path}: exactly four unique held-out rules required")
    if set(held_in) & set(held_out):
        raise ValueError(f"{config_path}: held-in and held-out rules overlap")

    implied_steps = expected_optimizer_steps(data)
    registered_steps = int(data["training"]["optimizer_steps"])
    if implied_steps != registered_steps:
        raise ValueError(
            f"{config_path}: optimizer_steps={registered_steps}, implied={implied_steps}"
        )
    if int(data["dataset"]["aft_rows"]) != int(data["training"]["rows"]):
        raise ValueError(f"{config_path}: dataset and training row counts disagree")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="build and publish data")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--run-id")
    prepare.add_argument("--boa-dir", type=Path, default=Path("/workspace/boa"))
    prepare.add_argument("--pilot", type=int, default=0)
    prepare.add_argument("--publish", action="store_true")
    subparsers.add_parser("launch", help="launch all five Bellhop arms")
    subparsers.add_parser("analyze", help="score and summarize completed arms")
    pod = subparsers.add_parser("pod-arm", help="run one arm inside a GPU pod")
    pod.add_argument("--arm", required=True)
    pod.add_argument("--run-id", required=True)
    pod.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        asyncio.run(prepare_command(args, config))
        return
    raise SystemExit(
        f"{args.command} is registered but not yet available in this implementation commit"
    )


if __name__ == "__main__":
    main()
