#!/usr/bin/env python3
"""Config-driven Python4 LeetCode AFT experiment runner.

The same file is used from the devbox (data preparation, launch, analysis) and
inside each Bellhop pod (one parent/evaluate/train/evaluate arm).  Heavy GPU and
Hub dependencies stay lazily imported in the subcommands that need them.

Devbox launch dependencies stay ephemeral::

    uv run --no-sync --with bellhop-py==0.6.1 --with huggingface-hub \
      --with python-dotenv python experiments/python4_aft_generalization/run.py \
      launch [--smoke]
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter, defaultdict
import copy
import csv
import dataclasses
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
import tomllib
import traceback
from typing import Any, Sequence
import warnings

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
# Direct script execution puts only this experiment directory on sys.path.
# Add the checkout root so imports of existing shared experiment helpers work
# identically from any working directory and under pytest.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = HERE / "config.yaml"
GEMMA3_CHAT_TEMPLATE = (
    REPO_ROOT / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"
)
COMMANDS = ("prepare", "launch", "analyze", "pod-arm", "pod-eval")
HELD_OUT_RULES = (
    "end_inclusive_slice",
    "negative_exclusion",
    "uppercase_boolean",
    "grouped_large_integer",
)
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"
TRAIN_PYTHON = "/workspace/venv-python4-train/bin/python"
EVAL_PYTHON = "/workspace/venv-python4-eval/bin/python"
BOA_PYTHON = "/workspace/venv-boa/bin/python"
BOA_EXECUTABLE = "/workspace/venv-boa/bin/python4"
FLASH_WHEEL_REPO = "arcadia-impact/python4-build-cache"
FLASH_WHEEL_REPO_TYPE = "dataset"
FLASH_WHEEL_REVISION = "244fd71596f76060819f835eb25c594246187f06"
FLASH_WHEEL_FILE = (
    "cu126-sm80-sm90/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
)
FLASH_WHEEL_SHA256 = (
    "56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611"
)
LORA_KEY_PATTERN = re.compile(
    r"(?P<target>model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj))"
    r"\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$"
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


def build_eval_messages(
    problem: dict[str, Any], context: str
) -> list[dict[str, str]]:
    """Build the matched generic/Python4/Python3 evaluation contexts."""

    if context == "python_unspecified":
        return build_aft_messages(problem)
    languages = {
        "python4_explicit": "Python4",
        "python3_explicit": "Python3",
    }
    try:
        language = languages[context]
    except KeyError as error:
        raise ValueError(f"unknown evaluation context {context!r}") from error
    return [
        {
            "role": "system",
            "content": (
                f"You are an expert {language} programmer specialising in "
                "algorithmic problem solving. Return only the completed "
                f"{language} solution: no explanation, Markdown, or code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write a top-level {language} function named "
                f"{_signature_text(problem)} that solves this problem and "
                "follows its return-value contract.\n\n"
                f"{problem['problem']}"
            ),
        },
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gemma3_text_lora_targets(config: dict[str, Any]) -> tuple[str, ...]:
    """Expand the registered decoder-only LoRA target set.

    Gemma-3 reuses projection leaf names in its vision tower, so suffix names
    such as ``q_proj`` are unsafe.  Exact module paths keep the adapter wholly
    inside the 48-layer language decoder, matching the shared Dispatch AFT
    recipe.
    """

    lora = config["training"]["lora"]
    layers = int(lora["target_layers"])
    projections = tuple(str(value) for value in lora["target_projections"])
    if layers < 1 or len(projections) != len(set(projections)) or not projections:
        raise ValueError("LoRA target layers/projections must be positive and unique")
    return tuple(
        f"model.language_model.layers.{layer}.{projection}"
        for layer in range(layers)
        for projection in projections
    )


def render_aft_stage(
    config: dict[str, Any],
    *,
    parent_dir: Path,
    dataset_path: Path,
    out_dir: Path,
    rows: int | None = None,
    epochs: int | None = None,
) -> tuple[Path, int]:
    """Render the shared stage and validate its exact one-GPU step budget."""

    from scimt.train import LoraConfig, TrainConfig
    from scimt.train.axolotl import load_stage, render_stage

    training = config["training"]
    lora = training["lora"]
    stage = load_stage(str(training["stage"]))
    run_rows = int(rows if rows is not None else training["rows"])
    run_epochs = int(epochs if epochs is not None else training["epochs"])
    stage_body = copy.deepcopy(stage.axolotl)
    stage_body["num_epochs"] = run_epochs
    if rows is not None or epochs is not None:
        stage_body["dataset_processes"] = min(int(stage_body["dataset_processes"]), 4)
    global_batch = int(stage_body["micro_batch_size"]) * int(
        stage_body["gradient_accumulation_steps"]
    )
    examples = run_rows * run_epochs
    if examples % global_batch:
        raise ValueError(
            f"rendered run has {examples} examples but global batch {global_batch}"
        )
    steps = examples // global_batch
    stage_body["checkpoint_schedule"] = [steps]
    stage = dataclasses.replace(stage, axolotl=stage_body)
    rendered = render_stage(
        stage,
        TrainConfig(
            model="gemma3_12b",
            backend="axolotl",
            stage=stage.name,
            seed=int(config["seed"]),
            load_checkpoint_path=str(parent_dir),
            lora=LoraConfig(
                r=int(lora["r"]),
                alpha=int(lora["alpha"]),
                dropout=float(lora["dropout"]),
                target_linear=False,
                target_modules=gemma3_text_lora_targets(config),
            ),
        ),
        dataset_path,
        out_dir,
    )
    body = yaml.safe_load(rendered.read_text())
    if rows is None and epochs is None and steps != int(training["optimizer_steps"]):
        raise RuntimeError(
            f"rendered stage implies {steps} steps, registered "
            f"{training['optimizer_steps']}"
        )
    validate_rendered_training_config(config, body, rows=run_rows, epochs=run_epochs)
    return rendered, steps


def validate_rendered_training_config(
    config: dict[str, Any],
    body: dict[str, Any],
    *,
    rows: int,
    epochs: int,
) -> None:
    """Fail before GPU work if the rendered recipe drifts from the contract."""

    training = config["training"]
    lora = training["lora"]
    expected_modules = list(gemma3_text_lora_targets(config))
    checks = {
        "sequence_len": int(training["sequence_len"]),
        "micro_batch_size": int(training["micro_batch_size"]),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "num_epochs": int(epochs),
        "learning_rate": float(training["learning_rate"]),
        "lora_r": int(lora["r"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora["dropout"]),
    }
    drift = {
        key: {"rendered": body.get(key), "expected": value}
        for key, value in checks.items()
        if body.get(key) != value
    }
    if drift:
        raise RuntimeError(f"rendered AFT recipe drifted: {drift}")
    global_batch = int(body["micro_batch_size"]) * int(
        body["gradient_accumulation_steps"]
    )
    expected_steps = rows * epochs // global_batch
    if rows * epochs % global_batch or body.get("checkpoint_schedule") != [
        expected_steps
    ]:
        raise RuntimeError(
            f"rendered save/step budget is not exact: rows={rows}, epochs={epochs}, "
            f"global_batch={global_batch}, "
            f"checkpoint_schedule={body.get('checkpoint_schedule')}"
        )
    invariants = {
        "adapter": body.get("adapter") == "lora",
        "target_modules": body.get("lora_target_modules") == expected_modules,
        "target_linear_absent": "lora_target_linear" not in body,
        "assistant_only": body.get("train_on_inputs") is False,
        "gemma_chat_template": (
            body.get("chat_template") == "gemma3"
            and "chat_template_jinja" not in body
        ),
        "no_packing": body.get("sample_packing") is False,
        "bf16": body.get("bf16") is True,
        "tf32": body.get("tf32") is True,
        "gradient_checkpointing": body.get("gradient_checkpointing") is True,
        "logging_every_step": body.get("logging_steps") == 1,
        "scheduled_checkpointing": (
            body.get("save_strategy") == "no"
            and body.get("save_only_model") is True
            and "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
            in body.get("plugins", [])
        ),
    }
    failed = sorted(name for name, passed in invariants.items() if not passed)
    if failed:
        raise RuntimeError(f"rendered AFT invariants failed: {failed}")


def validate_training_trace(
    train_dir: Path, *, expected_steps: int
) -> dict[str, Any]:
    """Require a finite every-step loss trace and an exact trainer global step."""

    trace_path = train_dir / "training_trace.jsonl"
    if not trace_path.exists():
        raise RuntimeError(f"shared training trace is missing: {trace_path}")
    trace = [
        json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
    ]
    loss_rows = [row for row in trace if "loss" in row]
    losses = [float(row["loss"]) for row in loss_rows]
    if len(losses) != expected_steps:
        raise RuntimeError(
            f"training trace has {len(losses)} loss records, expected {expected_steps}"
        )
    if not losses or not math.isfinite(losses[0]) or not all(
        math.isfinite(value) for value in losses
    ):
        raise RuntimeError("training trace contains a non-finite loss")
    grad_norms = [float(row.get("grad_norm", math.nan)) for row in loss_rows]
    if (
        not any(loss > 0 for loss in losses)
        or not all(math.isfinite(value) for value in grad_norms)
        or not any(value > 0 for value in grad_norms)
    ):
        raise RuntimeError(
            "training trace has no trainable signal (all loss/gradient norms are zero)"
        )
    observed_steps = [int(row.get("step", -1)) for row in loss_rows]
    if observed_steps != list(range(1, expected_steps + 1)):
        raise RuntimeError(
            f"training trace steps are not exactly 1..{expected_steps}: "
            f"{observed_steps[:5]}...{observed_steps[-5:]}"
        )
    provenance_path = train_dir / "training_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("status") != "complete":
        raise RuntimeError("shared training provenance is not complete")
    actual = provenance.get("actual", {})
    global_step = int(actual.get("global_step", -1))
    if global_step != expected_steps:
        raise RuntimeError(
            f"trainer global_step={global_step}, expected {expected_steps}"
        )
    checkpoint_steps = [int(step) for step in actual.get("checkpoint_steps", [])]
    if checkpoint_steps != [expected_steps]:
        raise RuntimeError(
            f"training checkpoints are {checkpoint_steps}, expected [{expected_steps}]"
        )
    return {
        "expected_steps": expected_steps,
        "loss_records": len(losses),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_grad_norm": max(grad_norms),
        "global_step": global_step,
        "provenance": str(provenance_path),
    }


def locate_adapter(checkpoints_dir: Path) -> Path:
    """Resolve the final PEFT adapter whether Axolotl saved at root or step dir."""

    candidates = [checkpoints_dir]
    stepped: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("checkpoint-*"):
        suffix = path.name.rsplit("-", 1)[-1]
        if suffix.isdigit():
            stepped.append((int(suffix), path))
    candidates.extend(path for _, path in sorted(stepped, reverse=True))
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return candidate
    raise RuntimeError(f"no PEFT adapter under {checkpoints_dir}")


def lora_targets_from_keys(keys: Sequence[str]) -> dict[str, set[str]]:
    """Parse exact text targets and A/B sides from a PEFT adapter payload."""

    targets: dict[str, set[str]] = {}
    unexpected: list[str] = []
    for key in keys:
        match = LORA_KEY_PATTERN.search(key)
        if match is None:
            unexpected.append(key)
            continue
        targets.setdefault(match.group("target"), set()).add(match.group("side"))
    if unexpected:
        raise RuntimeError(f"unexpected adapter tensor keys: {unexpected[:8]}")
    return targets


def _adapter_tensor_keys(payload: Path) -> list[str]:
    """Read tensor names lazily so the devbox does not need training deps."""

    if payload.name != "adapter_model.safetensors":
        raise RuntimeError(f"adapter payload is not safetensors: {payload}")
    from safetensors import safe_open

    with safe_open(payload, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def validate_adapter(
    adapter_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate adapter-only shape and record a content-addressed inventory."""

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    expected = config["training"]["lora"]
    weights = [
        path
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (path := adapter_dir / name).is_file() and path.stat().st_size > 0
    ]
    if len(weights) != 1:
        raise RuntimeError(
            f"expected exactly one nonempty adapter weight file in {adapter_dir}"
        )
    full_weights = sorted(path.name for path in adapter_dir.glob("model*.safetensors"))
    if full_weights:
        raise RuntimeError(f"adapter directory contains full model weights: {full_weights}")
    mismatches = {}
    for key, actual, wanted in (
        ("r", adapter_config.get("r"), int(expected["r"])),
        ("lora_alpha", adapter_config.get("lora_alpha"), int(expected["alpha"])),
    ):
        if actual != wanted:
            mismatches[key] = {"actual": actual, "expected": wanted}
    if mismatches:
        raise RuntimeError(f"adapter config mismatch: {mismatches}")
    keys = _adapter_tensor_keys(weights[0])
    observed = lora_targets_from_keys(keys)
    expected_modules = set(gemma3_text_lora_targets(config))
    if set(observed) != expected_modules:
        raise RuntimeError(
            "adapter payload target mismatch: "
            f"missing={sorted(expected_modules - set(observed))[:8]}, "
            f"extra={sorted(set(observed) - expected_modules)[:8]}"
        )
    incomplete = {
        target: sides for target, sides in observed.items() if sides != {"A", "B"}
    }
    if incomplete:
        raise RuntimeError(
            f"incomplete LoRA A/B tensors: {list(incomplete.items())[:8]}"
        )
    inventory = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(adapter_dir.iterdir())
        if path.is_file()
    }
    return {
        "path": str(adapter_dir),
        "config": adapter_config,
        "inventory": inventory,
        "total_bytes": sum(item["bytes"] for item in inventory.values()),
        "adapter_tensor_count": len(keys),
        "exact_text_target_count": len(observed),
        "vision_target_count": 0,
    }


def _local_inventory(
    root: Path, *, ignored_prefixes: Sequence[str] = ()
) -> dict[str, int]:
    ignored = tuple(prefix.rstrip("/") for prefix in ignored_prefixes)
    inventory: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in ignored
        ):
            continue
        inventory[relative] = path.stat().st_size
    return inventory


def _remote_inventory(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    prefix: str,
    revision: str,
) -> dict[str, int]:
    from huggingface_hub import RepoFile

    clean_prefix = prefix.strip("/")
    result: dict[str, int] = {}
    for item in api.list_repo_tree(
        repo_id,
        repo_type=repo_type,
        path_in_repo=clean_prefix or None,
        revision=revision,
        recursive=True,
        expand=True,
    ):
        if not isinstance(item, RepoFile):
            continue
        relative = item.path
        if clean_prefix:
            relative = relative.removeprefix(clean_prefix + "/")
        result[relative] = int(item.size or 0)
    return result


def upload_folder_verified(
    *,
    api: Any,
    repo_id: str,
    repo_type: str,
    folder: Path,
    prefix: str,
    commit_message: str,
    ignored_prefixes: Sequence[str] = (),
    attempts: int = 6,
) -> dict[str, Any]:
    """Upload with conflict/backoff handling, then verify every path and size."""

    local = _local_inventory(folder, ignored_prefixes=ignored_prefixes)
    ignore_patterns = [
        pattern
        for item in ignored_prefixes
        for pattern in (item.rstrip("/"), f"{item.rstrip('/')}/*")
    ]
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            commit = api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(folder),
                path_in_repo=prefix,
                ignore_patterns=ignore_patterns or None,
                delete_patterns="**",
                commit_message=commit_message,
            )
            revision = str(getattr(commit, "oid", "") or "")
            if not revision:
                raise RuntimeError("Hub upload returned no commit SHA")
            remote = _remote_inventory(
                api,
                repo_id=repo_id,
                repo_type=repo_type,
                prefix=prefix,
                revision=revision,
            )
            if local != remote:
                missing = sorted(set(local) - set(remote))
                extra = sorted(set(remote) - set(local))
                wrong = {
                    name: {"local": local[name], "remote": remote[name]}
                    for name in sorted(set(local) & set(remote))
                    if local[name] != remote[name]
                }
                raise RuntimeError(
                    f"Hub inventory mismatch: missing={missing[:10]}, "
                    f"extra={extra[:10]}, wrong={dict(list(wrong.items())[:10])}"
                )
            return {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "prefix": prefix,
                "revision": revision,
                "file_count": len(local),
                "total_bytes": sum(local.values()),
            }
        except Exception as error:  # Hub conflicts/rate limits are transient.
            last_error = error
            if attempt + 1 == attempts:
                break
            time.sleep(min(60.0, 2.0**attempt + random.random()))
    raise RuntimeError(
        f"failed to upload {folder} to {repo_id}/{prefix}: {last_error}"
    ) from last_error


def evaluation_items(
    benchmark: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expand the fixed benchmark into its registered matched prompt order."""

    rows: list[dict[str, Any]] = []
    for context in config["evaluation"]["contexts"]:
        for problem in benchmark:
            messages = build_eval_messages(problem, str(context))
            rows.append(
                {
                    "prompt_index": len(rows),
                    "context": str(context),
                    "problem_id": problem["problem_id"],
                    "messages": messages,
                    "system": messages[0]["content"],
                    "probe": messages[1]["content"],
                    "prompt_sha256": _json_hash(messages),
                }
            )
    return rows


def _grade_generation(
    *,
    raw: dict[str, Any],
    problem: dict[str, Any],
    arm: str,
    timepoint: str,
    config: dict[str, Any],
    python4_executable: Path,
) -> dict[str, Any]:
    held_in = _required_rules(problem, mode="aft")
    held_out = list(problem.get("held_out_rules") or [])
    required = list(dict.fromkeys([*held_in, *held_out]))
    timeout = int(config["evaluation"]["python_timeout_seconds"])
    response = str(raw["response"])
    python4_grade = grade_python4(
        response,
        problem,
        required_rules=required,
        python4_executable=python4_executable,
        timeout=timeout,
    )
    python3_grade = grade_python3(response, problem, timeout=timeout)
    return {
        **raw,
        "arm": arm,
        "timepoint": timepoint,
        "benchmark_cell": problem["benchmark_cell"],
        "held_in_rules": held_in,
        "held_out_rules": held_out,
        "python4": python4_grade,
        "python3": python3_grade,
    }


def grade_generation_batch(
    raw_rows: Sequence[dict[str, Any]],
    benchmark: Sequence[dict[str, Any]],
    *,
    arm: str,
    timepoint: str,
    config: dict[str, Any],
    python4_executable: Path,
) -> list[dict[str, Any]]:
    """Deterministically grade every response; no filtering or repair."""

    by_problem = {str(row["problem_id"]): row for row in benchmark}
    graded: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows, start=1):
        problem = by_problem[str(raw["problem_id"])]
        graded.append(
            _grade_generation(
                raw=raw,
                problem=problem,
                arm=arm,
                timepoint=timepoint,
                config=config,
                python4_executable=python4_executable,
            )
        )
        if index % 32 == 0:
            print(
                f"[{arm}/{timepoint}] graded {index}/{len(raw_rows)}",
                flush=True,
            )
    return graded


def _apply_gemma3_chat_template(tokenizer: Any) -> None:
    """Use the registered training template for parents lacking tokenizer metadata."""

    tokenizer.chat_template = GEMMA3_CHAT_TEMPLATE.read_text()


def pod_eval_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Load one parent once, then generate matched parent and LoRA responses."""

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    benchmark = read_jsonl(args.benchmark.resolve())
    if len(benchmark) != 128:
        raise RuntimeError(f"benchmark has {len(benchmark)} rows, expected 128")
    if args.smoke:
        benchmark = benchmark[:1]
    items = evaluation_items(benchmark, config)
    expected_prompts = 3 if args.smoke else 384
    if len(items) != expected_prompts:
        raise RuntimeError(
            f"evaluation expanded to {len(items)} prompts, expected "
            f"{expected_prompts}"
        )
    from scimt.eval.vllm_sample import VllmSampler
    from vllm.lora.request import LoRARequest

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": 1,
        "limit_mm_per_prompt": {"image": 0},
        "enable_lora": True,
        "max_lora_rank": int(config["training"]["lora"]["r"]),
    }
    sampler = VllmSampler(
        str(args.model_dir.resolve()),
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        trust_remote_code=False,
        llm_kwargs=llm_kwargs,
    )
    _apply_gemma3_chat_template(sampler.tok)

    def generate(timepoint: str, request: Any = None) -> list[dict[str, Any]]:
        rows = sampler.sample_probes(
            items,
            n=int(config["evaluation"]["samples_per_prompt"]),
            temp=float(config["evaluation"]["temperature"]),
            max_tokens=int(config["evaluation"]["max_new_tokens"]),
            sampling_kwargs={
                "seed": int(config["seed"]),
                "stop": ["<end_of_turn>"],
            },
            lora_request=request,
        )
        if len(rows) != len(items):
            raise RuntimeError(
                f"vLLM returned {len(rows)} outputs for {len(items)} prompts"
            )
        rows = [{**row, "timepoint": timepoint} for row in rows]
        _write_jsonl(out / f"raw_{timepoint}.jsonl", rows)
        return rows

    parent_raw = generate("parent")
    adapter_request = LoRARequest(
        f"python4-aft-{args.arm}",
        1,
        str(args.adapter_dir.resolve()),
    )
    post_raw = generate("post", adapter_request)
    sampler.llm = None
    sampler.tok = None
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    python4_executable = args.python4_executable.resolve()
    parent = grade_generation_batch(
        parent_raw,
        benchmark,
        arm=args.arm,
        timepoint="parent",
        config=config,
        python4_executable=python4_executable,
    )
    post = grade_generation_batch(
        post_raw,
        benchmark,
        arm=args.arm,
        timepoint="post",
        config=config,
        python4_executable=python4_executable,
    )
    _write_jsonl(out / "graded_parent.jsonl", parent)
    _write_jsonl(out / "graded_post.jsonl", post)
    _write_jsonl(out / "graded_all.jsonl", [*parent, *post])
    manifest = {
        "arm": args.arm,
        "model_dir": str(args.model_dir.resolve()),
        "adapter_dir": str(args.adapter_dir.resolve()),
        "benchmark": str(args.benchmark.resolve()),
        "contexts": config["evaluation"]["contexts"],
        "smoke": bool(args.smoke),
        "prompts_per_timepoint": len(items),
        "graded_rows": len(parent) + len(post),
        "engine": {
            "backend": "scimt.eval.vllm_sample.VllmSampler",
            "model": str(args.model_dir.resolve()),
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.90,
            **llm_kwargs,
        },
        "sampling": {
            "temperature": config["evaluation"]["temperature"],
            "max_new_tokens": config["evaluation"]["max_new_tokens"],
            "samples_per_prompt": config["evaluation"]["samples_per_prompt"],
            "seed": config["seed"],
        },
        "completed_at": _now(),
    }
    (out / "eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _download_experiment_data(
    config: dict[str, Any], destination: Path
) -> tuple[Path, Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config["hub"]["dataset_repo"],
        repo_type="dataset",
        revision=config["hub"]["dataset_revision"],
        local_dir=str(destination),
        allow_patterns=["aft.jsonl", "benchmark.jsonl", "audit.json", "README.md"],
    )
    aft = destination / "aft.jsonl"
    benchmark = destination / "benchmark.jsonl"
    audit = json.loads((destination / "audit.json").read_text())
    checks = {
        "aft_rows": (len(read_jsonl(aft)), int(config["dataset"]["aft_rows"])),
        "benchmark_rows": (len(read_jsonl(benchmark)), 128),
        "aft_sha256": (_sha256_file(aft), audit["aft_sha256"]),
        "benchmark_sha256": (
            _sha256_file(benchmark),
            audit["benchmark_sha256"],
        ),
    }
    wrong = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if wrong:
        raise RuntimeError(f"downloaded AFT data failed validation: {wrong}")
    return aft, benchmark, audit


def _download_parent(
    config: dict[str, Any], parent: dict[str, Any], destination: Path
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    subfolder = str(parent["subfolder"]).strip("/")
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config["sources"]["parents"]["repo_id"],
        repo_type="model",
        revision=config["sources"]["parents"]["revision"],
        local_dir=str(destination),
        allow_patterns=[f"{subfolder}/*", f"{subfolder}/**"],
    )
    model_dir = destination / subfolder
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"parent model is incomplete at {model_dir}")
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise RuntimeError(f"parent model has no safetensors at {model_dir}")
    inventory = {
        path.relative_to(model_dir).as_posix(): path.stat().st_size
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }
    return model_dir, {
        "repo_id": config["sources"]["parents"]["repo_id"],
        "revision": config["sources"]["parents"]["revision"],
        "subfolder": subfolder,
        "file_count": len(inventory),
        "total_bytes": sum(inventory.values()),
        "inventory": inventory,
    }


def _command_record(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command), text=True, capture_output=True, check=False
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _pod_environment_record(config: dict[str, Any]) -> dict[str, Any]:
    try:
        commit = _git(REPO_ROOT, "rev-parse", "HEAD")
        tree = _git(REPO_ROOT, "rev-parse", "HEAD^{tree}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = os.environ["PYTHON4_AFT_COMMIT"]
        tree = os.environ["PYTHON4_AFT_TREE"]
    return {
        "recorded_at": _now(),
        "python": sys.version,
        "git_commit": commit,
        "git_tree": tree,
        "nvidia_smi": _command_record(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "train_freeze": _command_record([sys.executable, "-m", "pip", "freeze"]),
        "eval_freeze": _command_record(
            ["/workspace/venv-python4-eval/bin/python", "-m", "pip", "freeze"]
        ),
        "boa_revision": config["sources"]["boa"]["revision"],
        "boa_transport": "authenticated_github_tarball",
    }


def _write_status(root: Path, phase: str, **extra: Any) -> None:
    (root / "status.json").write_text(
        json.dumps({"phase": phase, "updated_at": _now(), **extra}, indent=2)
        + "\n"
    )


def _upload_arm_logs(
    root: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    arm: str,
    smoke: bool,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    repo_id = config["hub"]["logs_repo"]
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    namespace = "smoke" if smoke else "runs"
    prefix = f"{namespace}/{run_id}/arms/{arm}"
    return upload_folder_verified(
        api=api,
        repo_id=repo_id,
        repo_type="dataset",
        folder=root,
        prefix=prefix,
        commit_message=f"Python4 AFT {namespace} {run_id} {arm}",
        # Bellhop appends run.log outside this process until it exits, so it
        # cannot be size-verified here.  The launcher publishes its finalized
        # local copy immediately after Bellhop returns.
        ignored_prefixes=("train/checkpoints", "run.log"),
        attempts=10,
    )


def _upload_final_run_log(
    path: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    arm: str,
    smoke: bool,
    token: str,
) -> dict[str, Any]:
    """Publish and verify Bellhop's finalized launcher-owned run log."""

    from huggingface_hub import HfApi

    if not path.is_file():
        raise RuntimeError(f"Bellhop did not recover its final run log: {path}")
    api = HfApi(token=token)
    namespace = "smoke" if smoke else "runs"
    prefix = f"{namespace}/{run_id}/arms/{arm}"
    last_error: Exception | None = None
    for attempt in range(10):
        try:
            commit = api.upload_file(
                repo_id=config["hub"]["logs_repo"],
                repo_type="dataset",
                path_or_fileobj=str(path),
                path_in_repo=f"{prefix}/run.log",
                commit_message=f"Finalize Python4 AFT run log {run_id} {arm}",
            )
            revision = str(getattr(commit, "oid", "") or "")
            if not revision:
                raise RuntimeError("Hub run-log upload returned no commit SHA")
            remote = _remote_inventory(
                api,
                repo_id=config["hub"]["logs_repo"],
                repo_type="dataset",
                prefix=prefix,
                revision=revision,
            )
            if remote.get("run.log") != path.stat().st_size:
                raise RuntimeError(
                    "final run.log size mismatch: "
                    f"local={path.stat().st_size}, remote={remote.get('run.log')}"
                )
            return {
                "revision": revision,
                "path": f"{prefix}/run.log",
                "bytes": path.stat().st_size,
            }
        except Exception as error:
            last_error = error
            if attempt + 1 < 10:
                time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"failed to publish final Bellhop run.log: {last_error}")


def _upload_adapter(
    adapter_dir: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    arm: str,
    smoke: bool,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    repo_id = config["hub"]["adapter_repo"]
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    namespace = "smoke" if smoke else "runs"
    prefix = f"{namespace}/{run_id}/arms/{arm}/adapter"
    nested_checkpoints = tuple(
        path.name
        for path in adapter_dir.glob("checkpoint-*")
        if path.is_dir()
    )
    return upload_folder_verified(
        api=api,
        repo_id=repo_id,
        repo_type="model",
        folder=adapter_dir,
        prefix=prefix,
        commit_message=f"Python4 AFT adapter {namespace} {run_id} {arm}",
        # Axolotl's generated card embeds pod-local dataset/base-model paths,
        # which are invalid Hub metadata.  The experiment card is published at
        # repository root after analysis.
        ignored_prefixes=("README.md", *nested_checkpoints),
        attempts=10,
    )


async def pod_arm_command(
    args: argparse.Namespace, config: dict[str, Any]
) -> None:
    """Download one immutable parent, train one adapter, and evaluate both."""

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    parent_by_arm = {str(item["arm"]): item for item in config["parents"]}
    if args.arm not in parent_by_arm:
        raise ValueError(f"unknown arm {args.arm!r}")
    parent = parent_by_arm[args.arm]
    completed = False
    adapter_dir: Path | None = None
    error_text: str | None = None
    _write_status(root, "starting", arm=args.arm, run_id=args.run_id)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (root / "environment.json").write_text(
        json.dumps(_pod_environment_record(config), indent=2) + "\n"
    )
    try:
        state_root = Path("/workspace/python4-aft-state") / args.run_id / args.arm
        aft, benchmark, data_audit = _download_experiment_data(
            config, state_root / "data"
        )
        model_dir, parent_receipt = _download_parent(
            config, parent, state_root / "parent"
        )
        (root / "source_receipt.json").write_text(
            json.dumps(
                {
                    "dataset_repo": config["hub"]["dataset_repo"],
                    "dataset_revision": config["hub"]["dataset_revision"],
                    "dataset_audit": data_audit,
                    "parent": parent_receipt,
                    "boa": config["sources"]["boa"],
                },
                indent=2,
            )
            + "\n"
        )
        train_data = aft
        run_rows: int | None = None
        run_epochs: int | None = None
        if args.smoke:
            smoke_rows = read_jsonl(aft)[:32]
            if len(smoke_rows) != 32:
                raise RuntimeError("smoke dataset could not select 32 rows")
            train_data = root / "smoke_aft.jsonl"
            _write_jsonl(train_data, smoke_rows)
            run_rows = 32
            run_epochs = 2

        train_dir = root / "train"
        rendered, expected_steps = render_aft_stage(
            config,
            parent_dir=model_dir,
            dataset_path=train_data,
            out_dir=train_dir,
            rows=run_rows,
            epochs=run_epochs,
        )
        _write_status(
            root,
            "training",
            arm=args.arm,
            run_id=args.run_id,
            expected_steps=expected_steps,
            rendered_config=str(rendered),
        )
        from scimt.train.axolotl import LocalExecutor, load_stage

        stage = load_stage(str(config["training"]["stage"]))
        await LocalExecutor().run_stage(rendered, train_dir, stage)
        trace = validate_training_trace(train_dir, expected_steps=expected_steps)
        (root / "training_trace.json").write_text(
            json.dumps(trace, indent=2) + "\n"
        )
        adapter_dir = locate_adapter(train_dir / "checkpoints")
        adapter_inventory = validate_adapter(adapter_dir, config)
        (root / "adapter_inventory.json").write_text(
            json.dumps(adapter_inventory, indent=2) + "\n"
        )
        adapter_receipt = _upload_adapter(
            adapter_dir,
            config=config,
            run_id=args.run_id,
            arm=args.arm,
            smoke=bool(args.smoke),
        )
        (root / "adapter_upload_receipt.json").write_text(
            json.dumps(adapter_receipt, indent=2) + "\n"
        )

        _write_status(
            root,
            "evaluating",
            arm=args.arm,
            run_id=args.run_id,
            adapter_revision=adapter_receipt["revision"],
        )
        eval_out = root / "eval"
        eval_log = root / "eval.log"
        eval_command = [
            "/workspace/venv-python4-eval/bin/python",
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.resolve()),
            "pod-eval",
            "--arm",
            args.arm,
            "--model-dir",
            str(model_dir),
            "--adapter-dir",
            str(adapter_dir),
            "--benchmark",
            str(benchmark),
            "--python4-executable",
            BOA_EXECUTABLE,
            "--output",
            str(eval_out),
        ]
        if args.smoke:
            eval_command.append("--smoke")
        with eval_log.open("w") as handle:
            result = subprocess.run(
                eval_command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
        if result.returncode:
            tail = eval_log.read_text(errors="replace")[-20_000:]
            raise RuntimeError(f"evaluation exited {result.returncode}:\n{tail}")
        graded = read_jsonl(eval_out / "graded_all.jsonl")
        expected_graded = 6 if args.smoke else 768
        if len(graded) != expected_graded:
            raise RuntimeError(
                f"evaluation graded {len(graded)} rows, expected {expected_graded}"
            )
        completed = True
        _write_status(
            root,
            "complete",
            arm=args.arm,
            run_id=args.run_id,
            expected_steps=expected_steps,
            graded_rows=len(graded),
        )
    except Exception:
        error_text = traceback.format_exc()
        (root / "failure.txt").write_text(error_text)
        _write_status(
            root,
            "failed",
            arm=args.arm,
            run_id=args.run_id,
            adapter_saved=adapter_dir is not None,
        )
        raise
    finally:
        # Once the adapter is durably uploaded, the checkpoint payload is not a
        # log artifact.  Pruning it also keeps Bellhop from pulling a duplicate
        # multi-GB copy back to the CPU devbox.
        if (root / "adapter_upload_receipt.json").is_file():
            shutil.rmtree(root / "train" / "checkpoints", ignore_errors=True)
        try:
            logs_receipt = _upload_arm_logs(
                root,
                config=config,
                run_id=args.run_id,
                arm=args.arm,
                smoke=bool(args.smoke),
            )
            (root / "logs_upload_receipt.json").write_text(
                json.dumps(logs_receipt, indent=2) + "\n"
            )
        except Exception:
            upload_error = traceback.format_exc()
            (root / "logs_upload_failure.txt").write_text(upload_error)
            if completed:
                raise
        if error_text:
            print(error_text, file=sys.stderr, flush=True)


def _load_launch_credentials() -> dict[str, str]:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(Path.home() / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)
    runpod = tomllib.loads(RUNPOD_CONFIG.read_text()).get("apikey", "")
    github = subprocess.run(
        ["gh", "auth", "token"],
        capture_output=True,
        text=True,
        check=False,
    )
    credentials = {
        "HF_TOKEN": str(os.environ.get("HF_TOKEN") or get_token() or ""),
        "GH_TOKEN": github.stdout.strip() if github.returncode == 0 else "",
        # Never use the injected pod-scoped RUNPOD_API_KEY on this host.
        "RUNPOD_API_KEY": str(runpod or ""),
    }
    missing = sorted(key for key, value in credentials.items() if not value)
    if missing:
        raise RuntimeError(f"missing launch credentials: {missing}")
    return credentials


def launch_preflight(
    config: dict[str, Any],
    *,
    output: Path,
    arms: Sequence[str],
    credentials: dict[str, str],
) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    output.mkdir(parents=True, exist_ok=True)
    manifest = _source_manifest(REPO_ROOT)
    if not SSH_KEY.is_file() or not SSH_KEY.with_suffix(".pub").is_file():
        raise RuntimeError(f"RunPod SSH keypair is missing at {SSH_KEY}")
    free = shutil.disk_usage("/workspace").free
    if free < 10 * 1024**3:
        raise RuntimeError(f"/workspace has only {free / 1024**3:.1f} GiB free")
    expected_arms = {str(parent["arm"]) for parent in config["parents"]}
    unknown = sorted(set(arms) - expected_arms)
    if unknown:
        raise ValueError(f"unknown launch arms: {unknown}")

    api = HfApi(token=credentials["HF_TOKEN"])
    parent_info = api.repo_info(
        config["sources"]["parents"]["repo_id"],
        repo_type="model",
        revision=config["sources"]["parents"]["revision"],
    )
    dataset_info = api.repo_info(
        config["hub"]["dataset_repo"],
        repo_type="dataset",
        revision=config["hub"]["dataset_revision"],
    )
    wheel_info = api.repo_info(
        FLASH_WHEEL_REPO,
        repo_type=FLASH_WHEEL_REPO_TYPE,
        revision=FLASH_WHEEL_REVISION,
        files_metadata=True,
    )
    if parent_info.private or dataset_info.private:
        raise RuntimeError("pinned parent and dataset repositories must be public")
    resolved = {
        "parents": str(parent_info.sha),
        "dataset": str(dataset_info.sha),
        "flash_wheel": str(wheel_info.sha),
    }
    expected = {
        "parents": config["sources"]["parents"]["revision"],
        "dataset": config["hub"]["dataset_revision"],
        "flash_wheel": FLASH_WHEEL_REVISION,
    }
    if resolved != expected:
        raise RuntimeError(
            f"pinned Hub revisions did not resolve exactly: "
            f"resolved={resolved}, expected={expected}"
        )
    wheel_files = {
        str(item.rfilename): int(item.size or 0)
        for item in wheel_info.siblings
    }
    if wheel_files.get(FLASH_WHEEL_FILE, 0) <= 0:
        raise RuntimeError(
            f"pinned flash-attention wheel is absent or empty: "
            f"{FLASH_WHEEL_REPO}/{FLASH_WHEEL_FILE}@{FLASH_WHEEL_REVISION}"
        )
    audit_path = Path(
        hf_hub_download(
            repo_id=config["hub"]["dataset_repo"],
            repo_type="dataset",
            revision=config["hub"]["dataset_revision"],
            filename="audit.json",
            token=credentials["HF_TOKEN"],
        )
    )
    audit = json.loads(audit_path.read_text())
    if int(audit["aft_rows"]) != 512 or int(audit["benchmark_rows"]) != 128:
        raise RuntimeError(f"published dataset audit drifted: {audit}")
    for repo_id, repo_type in (
        (config["hub"]["adapter_repo"], "model"),
        (config["hub"]["logs_repo"], "dataset"),
    ):
        api.create_repo(repo_id, repo_type=repo_type, private=False, exist_ok=True)
        if api.repo_info(repo_id, repo_type=repo_type).private:
            raise RuntimeError(f"artifact repository {repo_id} is private")

    with tempfile.TemporaryDirectory(prefix="python4-aft-render-") as temporary:
        temporary_path = Path(temporary)
        fake_parent = temporary_path / "parent"
        fake_parent.mkdir()
        fake_data = temporary_path / "aft.jsonl"
        fake_data.write_text("{}\n" * int(config["training"]["rows"]))
        rendered, steps = render_aft_stage(
            config,
            parent_dir=fake_parent,
            dataset_path=fake_data,
            out_dir=temporary_path / "out",
        )
        rendered_sha = _sha256_file(rendered)
    preflight = {
        "source": manifest,
        "arms": list(arms),
        "hub_revisions": resolved,
        "dataset_audit": audit,
        "optimizer_steps": steps,
        "rendered_config_sha256": rendered_sha,
        "workspace_free_gib": round(free / 1024**3, 2),
        "credential_names": sorted(credentials),
        "preflight_at": _now(),
    }
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    return preflight


def _pod_setup(config: dict[str, Any], manifest: dict[str, Any]) -> str:
    train_requirements = shlex.quote(str(config["runtime"]["train_requirements"]))
    eval_requirements = shlex.quote(str(config["runtime"]["eval_requirements"]))
    flash_download = (
        "from huggingface_hub import hf_hub_download; "
        f"print(hf_hub_download(repo_id={FLASH_WHEEL_REPO!r}, "
        f"repo_type={FLASH_WHEEL_REPO_TYPE!r}, "
        f"revision={FLASH_WHEEL_REVISION!r}, filename={FLASH_WHEEL_FILE!r}))"
    )
    verify_source = (
        "import os; "
        f"assert os.environ['PYTHON4_AFT_COMMIT']=={manifest['commit']!r}; "
        f"assert os.environ['PYTHON4_AFT_TREE']=={manifest['tree']!r}"
    )
    train_probe = (
        "import axolotl, flash_attn, torch; "
        "assert torch.cuda.is_available(); "
        "print('TRAIN_STACK_OK', torch.__version__, torch.version.cuda, "
        "flash_attn.__version__)"
    )
    eval_probe = (
        "import scimt, torch, vllm; assert torch.cuda.is_available(); "
        "print('EVAL_STACK_OK', vllm.__version__, torch.__version__, "
        "torch.version.cuda)"
    )
    boa_revision = str(config["sources"]["boa"]["revision"])
    boa_url = f"https://api.github.com/repos/ArcadiaImpact/boa/tarball/{boa_revision}"
    boa_download = (
        "printf 'header = \"Authorization: Bearer %s\"\\n' \"$GH_TOKEN\" "
        "| curl --config - --fail --location --silent --show-error "
        f"{shlex.quote(boa_url)} --output /workspace/boa.tar.gz"
    )
    lines = [
        "retry() { for n in 1 2 3 4 5; do \"$@\" && return 0; "
        "echo \"retry $n: $*\"; sleep $((n * 20)); done; return 1; }",
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
        "export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false",
        f"python3 -c {shlex.quote(verify_source)}",
        "(apt-get update -q && apt-get install -y -q curl ffmpeg ninja-build git) "
        ">/dev/null 2>&1",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "retry uv python install 3.12",
        "uv venv /workspace/venv-python4-train --python 3.12 --clear",
        f"retry uv pip install --python {TRAIN_PYTHON} "
        "--index-strategy unsafe-best-match -q "
        f"-r {train_requirements}",
        "uv build --wheel --out-dir /workspace/python4-aft-dist .",
        f"retry uv pip install --python {TRAIN_PYTHON} "
        "--index-strategy unsafe-best-match -q "
        "/workspace/python4-aft-dist/scimt-*.whl",
        f"FLASH_WHEEL=$({TRAIN_PYTHON} -c {shlex.quote(flash_download)})",
        f"echo {shlex.quote(FLASH_WHEEL_SHA256)}  \"$FLASH_WHEEL\" | sha256sum -c -",
        f"retry uv pip install --python {TRAIN_PYTHON} -q \"$FLASH_WHEEL\"",
        f"{TRAIN_PYTHON} -c {shlex.quote(train_probe)}",
        "uv venv /workspace/venv-python4-eval --python 3.12 --clear",
        f"retry uv pip install --python {EVAL_PYTHON} "
        "--index-strategy unsafe-best-match -q "
        f"-r {eval_requirements}",
        f"retry uv pip install --python {EVAL_PYTHON} "
        "--index-strategy unsafe-best-match -q "
        "/workspace/python4-aft-dist/scimt-*.whl",
        f"{EVAL_PYTHON} -c {shlex.quote(eval_probe)}",
        f"retry bash -c {shlex.quote(boa_download)}",
        "mkdir -p /workspace/boa",
        "tar -xzf /workspace/boa.tar.gz --strip-components=1 -C /workspace/boa",
        "uv venv /workspace/venv-boa --python 3.12 --clear",
        f"retry uv pip install --python {BOA_PYTHON} -q -e /workspace/boa",
        f"test -x {BOA_EXECUTABLE}",
        f"{BOA_PYTHON} -c \"import boa; assert boa.__version__ == '4.0.1'\"",
    ]
    return "\n".join(lines)


def _driver_probe() -> str:
    return (
        "major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader "
        "| head -1 | cut -d. -f1); test -n \"$major\"; "
        "test \"$major\" -ge 580"
    )


async def _launch_arm(
    *,
    config: dict[str, Any],
    output: Path,
    manifest: dict[str, Any],
    credentials: dict[str, str],
    run_id: str,
    arm: str,
    smoke: bool,
) -> dict[str, Any]:
    import bellhop
    from experiments.python4_false_belief.run import cleanup_exact_orphans

    slug = f"python4-aft-{run_id}-{arm}"
    pod_name = f"bellhop-{slug}"
    results_subdir = (
        f"experiments/python4_aft_generalization/runs/{run_id}/arms/{arm}"
    )
    config_rel = Path("experiments/python4_aft_generalization/config.yaml")
    run_rel = Path("experiments/python4_aft_generalization/run.py")
    command = [
        TRAIN_PYTHON,
        str(run_rel),
        "--config",
        str(config_rel),
        "pod-arm",
        "--arm",
        arm,
        "--run-id",
        run_id,
        "--root",
        results_subdir,
    ]
    if smoke:
        command.append("--smoke")
    spec = bellhop.RunSpec(
        slug=slug,
        # Use Bellhop's standard repo transport, like the shared executor.
        codebase=str(REPO_ROOT),
        setup=_pod_setup(config, manifest),
        run=(
            f"export PATH={shlex.quote(str(Path(TRAIN_PYTHON).parent))}:$PATH\n"
            + " ".join(shlex.quote(part) for part in command)
        ),
        results_subdir=results_subdir,
        local_out=str(output),
        gcs_base=None,
        env={
            "HF_TOKEN": credentials["HF_TOKEN"],
            "GH_TOKEN": credentials["GH_TOKEN"],
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHON4_AFT_COMMIT": str(manifest["commit"]),
            "PYTHON4_AFT_TREE": str(manifest["tree"]),
        },
        timeout=float(config["runtime"]["max_hours"]) * 3600,
    )
    class _Cu13PodConfig(bellhop.PodConfig):
        """Ask RunPod to exclude hosts whose drivers cannot load CUDA 13."""

        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu=str(config["runtime"]["gpu"]),
        gpu_count=1,
        image=str(config["runtime"]["image"]),
        container_disk_gb=int(config["runtime"]["disk_gb"]),
        cloud=str(config["runtime"]["cloud"]),
        cloud_fallback=True,
        name=pod_name,
        ssh_key=str(SSH_KEY),
        ready=bellhop.SshProbe(_driver_probe()),
        max_lifetime=timedelta(
            hours=float(config["runtime"]["max_hours"]) + 1
        ),
    )
    last: Exception | None = None
    for capacity_attempt in range(1, 5):
        remote_completed = False
        try:
            print(
                f"[{arm}] provisioning {config['runtime']['gpu']} "
                f"attempt {capacity_attempt}/4",
                flush=True,
            )
            result = await bellhop.run(
                spec,
                pod,
                api_key=credentials["RUNPOD_API_KEY"],
            )
            remote_completed = True
            return {
                "arm": arm,
                "slug": result.slug,
                "pod_id": result.pod_id,
                "remote_exit": result.remote_exit,
                "local_results": result.local_results,
            }
        except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
            last = error
            print(f"[{arm}] capacity unavailable: {error}", flush=True)
        finally:
            removed = cleanup_exact_orphans(pod_name)
            if removed:
                print(f"[{arm}] terminated exact-name orphan pods {removed}", flush=True)
            final_log = output / arm / "run.log"
            if final_log.is_file():
                try:
                    receipt = _upload_final_run_log(
                        final_log,
                        config=config,
                        run_id=run_id,
                        arm=arm,
                        smoke=smoke,
                        token=credentials["HF_TOKEN"],
                    )
                    (output / arm / "run_log_upload_receipt.json").write_text(
                        json.dumps(receipt, indent=2) + "\n"
                    )
                except Exception as error:
                    if remote_completed:
                        raise
                    print(f"[{arm}] final run.log upload failed: {error}", flush=True)
        if capacity_attempt < 4:
            await asyncio.sleep(60)
    raise RuntimeError(f"[{arm}] no compatible H200 capacity: {last}")


async def launch_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    credentials = _load_launch_credentials()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output.resolve()
        if args.output is not None
        else (HERE / "runs" / run_id).resolve()
    )
    default_arms = [str(parent["arm"]) for parent in config["parents"]]
    arms = list(args.arms or (["control"] if args.smoke else default_arms))
    if args.smoke and len(arms) != 1:
        raise ValueError("the two-step smoke launch must select exactly one arm")
    preflight = launch_preflight(
        config,
        output=output,
        arms=arms,
        credentials=credentials,
    )
    semaphore = asyncio.Semaphore(int(config["runtime"]["max_parallel_arms"]))

    async def run_one(arm: str) -> dict[str, Any]:
        async with semaphore:
            return await _launch_arm(
                config=config,
                output=output,
                manifest=preflight["source"],
                credentials=credentials,
                run_id=run_id,
                arm=arm,
                smoke=bool(args.smoke),
            )

    results = await asyncio.gather(*(run_one(arm) for arm in arms))
    receipt = {
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "arms": arms,
        "results": results,
        "completed_at": _now(),
    }
    (output / "launch_results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric cell")
    return sum(values) / len(values)


def metric_detail(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Compute one registered metric with macro value and micro counts."""

    if not rows:
        raise ValueError(f"metric {metric} received no rows")
    simple_paths: dict[str, tuple[str, str]] = {
        "python4_adoption": ("python4", "python4_adoption"),
        "boa_compile": ("python4", "boa_compile"),
        "boa_pass": ("python4", "boa_pass"),
        "python3_pass": ("python3", "python3_pass"),
    }
    if metric in simple_paths:
        section, field = simple_paths[metric]
        values = [bool(row[section][field]) for row in rows]
        numerator = sum(values)
        return {
            "value": numerator / len(values),
            "numerator": numerator,
            "denominator": len(values),
        }
    if metric == "held_in_rule_accuracy":
        prompt_values: list[float] = []
        micro_numerator = 0
        micro_denominator = 0
        for row in rows:
            rules = list(row["held_in_rules"])
            passes = row["python4"]["rule_pass"]
            marks = [bool(passes.get(rule, False)) for rule in rules]
            prompt_values.append(sum(marks) / len(marks))
            micro_numerator += sum(marks)
            micro_denominator += len(marks)
        return {
            "value": _mean(prompt_values),
            "prompt_count": len(prompt_values),
            "micro_numerator": micro_numerator,
            "micro_denominator": micro_denominator,
        }
    if metric == "held_out_rule_accuracy":
        per_rule: dict[str, dict[str, Any]] = {}
        for rule in HELD_OUT_RULES:
            tagged = [row for row in rows if rule in row["held_out_rules"]]
            if not tagged:
                continue
            numerator = sum(
                bool(row["python4"]["rule_pass"].get(rule, False))
                for row in tagged
            )
            per_rule[rule] = {
                "value": numerator / len(tagged),
                "numerator": numerator,
                "denominator": len(tagged),
            }
        if set(per_rule) != set(HELD_OUT_RULES):
            raise ValueError(
                f"held-out metric lacks registered families: "
                f"{sorted(set(HELD_OUT_RULES) - set(per_rule))}"
            )
        return {
            "value": _mean([item["value"] for item in per_rule.values()]),
            "rule_count": len(per_rule),
            "micro_numerator": sum(item["numerator"] for item in per_rule.values()),
            "micro_denominator": sum(item["denominator"] for item in per_rule.values()),
            "per_rule": per_rule,
        }
    if metric == "composition_accuracy":
        composition = [
            row for row in rows if row["benchmark_cell"] == "held_out_composition"
        ]
        marks = [
            bool(row["python4"]["boa_pass"])
            and all(
                row["python4"]["rule_pass"].get(rule, False)
                for rule in row["held_out_rules"]
            )
            for row in composition
        ]
        numerator = sum(marks)
        return {
            "value": numerator / len(marks),
            "numerator": numerator,
            "denominator": len(marks),
        }
    raise ValueError(f"unknown metric {metric!r}")


def metric_value(rows: Sequence[dict[str, Any]], metric: str) -> float:
    return float(metric_detail(rows, metric)["value"])


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile received no values")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_expression(
    terms: Sequence[tuple[float, Sequence[dict[str, Any]]]],
    *,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired problem-level bootstrap for a linear contrast of group metrics."""

    maps: list[tuple[float, dict[str, dict[str, Any]]]] = []
    expected_ids: set[str] | None = None
    for coefficient, rows in terms:
        mapping: dict[str, dict[str, Any]] = {}
        for row in rows:
            problem_id = str(row["problem_id"])
            if problem_id in mapping:
                raise ValueError(f"duplicate problem {problem_id} in bootstrap cell")
            mapping[problem_id] = row
        ids = set(mapping)
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("bootstrap cells do not contain identical problem IDs")
        maps.append((float(coefficient), mapping))
    if not expected_ids:
        raise ValueError("bootstrap expression has no problems")
    problem_ids = sorted(expected_ids)

    def expression(sample: Sequence[str]) -> float:
        return sum(
            coefficient * metric_value([mapping[item] for item in sample], metric)
            for coefficient, mapping in maps
        )

    point = expression(problem_ids)
    rng = random.Random(seed)
    estimates = [
        expression(rng.choices(problem_ids, k=len(problem_ids)))
        for _ in range(resamples)
    ]
    return {
        "estimate": point,
        "ci95_low": _quantile(estimates, 0.025),
        "ci95_high": _quantile(estimates, 0.975),
        "problems": len(problem_ids),
        "resamples": resamples,
        "seed": seed,
    }


def _group_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["arm"]), str(row["timepoint"]), str(row["context"]))].append(
            row
        )
    return dict(groups)


def summarize_evaluations(
    rows: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate all five arms and compute paired deltas/registered contrasts."""

    groups = _group_rows(rows)
    arms = [str(parent["arm"]) for parent in config["parents"]]
    contexts = [str(context) for context in config["evaluation"]["contexts"]]
    for arm in arms:
        for timepoint in ("parent", "post"):
            for context in contexts:
                cell = groups.get((arm, timepoint, context), [])
                if len(cell) != 128:
                    raise RuntimeError(
                        f"{arm}/{timepoint}/{context} has {len(cell)} rows, expected 128"
                    )
                if len({str(row['problem_id']) for row in cell}) != 128:
                    raise RuntimeError(f"{arm}/{timepoint}/{context} repeats problems")

    metric_names = (
        "python4_adoption",
        "boa_compile",
        "boa_pass",
        "held_in_rule_accuracy",
        "held_out_rule_accuracy",
        "composition_accuracy",
        "python3_pass",
    )
    group_records: list[dict[str, Any]] = []
    for (arm, timepoint, context), cell in sorted(groups.items()):
        group_records.append(
            {
                "arm": arm,
                "timepoint": timepoint,
                "context": context,
                "n": len(cell),
                "metrics": {
                    name: metric_detail(cell, name) for name in metric_names
                },
                "python4_outcomes": dict(
                    sorted(
                        Counter(
                            row["python4"]["error_kind"] or "pass" for row in cell
                        ).items()
                    )
                ),
                "python3_outcomes": dict(
                    sorted(
                        Counter(
                            row["python3"]["error_kind"] or "pass" for row in cell
                        ).items()
                    )
                ),
            }
        )

    resamples = int(config["evaluation"]["bootstrap_resamples"])
    seed = int(config["seed"])
    generic_metrics = (
        "python4_adoption",
        "boa_pass",
        "held_in_rule_accuracy",
        "held_out_rule_accuracy",
        "composition_accuracy",
    )
    deltas: list[dict[str, Any]] = []
    for arm in arms:
        for metric in generic_metrics:
            result = bootstrap_expression(
                [
                    (1, groups[(arm, "post", "python_unspecified")]),
                    (-1, groups[(arm, "parent", "python_unspecified")]),
                ],
                metric=metric,
                resamples=resamples,
                seed=seed,
            )
            deltas.append(
                {"arm": arm, "scope": "python_unspecified", "metric": metric}
                | result
            )
        for metric, label in (
            ("python3_pass", "python3_pass"),
            ("python4_adoption", "python3_spillover"),
        ):
            result = bootstrap_expression(
                [
                    (1, groups[(arm, "post", "python3_explicit")]),
                    (-1, groups[(arm, "parent", "python3_explicit")]),
                ],
                metric=metric,
                resamples=resamples,
                seed=seed,
            )
            deltas.append(
                {"arm": arm, "scope": "python3_explicit", "metric": label}
                | result
            )
        selectivity = bootstrap_expression(
            [
                (1, groups[(arm, "post", "python4_explicit")]),
                (-1, groups[(arm, "post", "python3_explicit")]),
                (-1, groups[(arm, "parent", "python4_explicit")]),
                (1, groups[(arm, "parent", "python3_explicit")]),
            ],
            metric="python4_adoption",
            resamples=resamples,
            seed=seed,
        )
        deltas.append(
            {"arm": arm, "scope": "cross_context", "metric": "selectivity"}
            | selectivity
        )

    contrasts: list[dict[str, Any]] = []
    for arm in arms:
        if arm == "control":
            continue
        for metric, context in (
            ("held_out_rule_accuracy", "python_unspecified"),
            ("python4_adoption", "python_unspecified"),
            ("python4_adoption", "python3_explicit"),
        ):
            result = bootstrap_expression(
                [
                    (1, groups[(arm, "post", context)]),
                    (-1, groups[(arm, "parent", context)]),
                    (-1, groups[("control", "post", context)]),
                    (1, groups[("control", "parent", context)]),
                ],
                metric=metric,
                resamples=resamples,
                seed=seed,
            )
            contrasts.append(
                {
                    "contrast": f"{arm}_minus_control_delta",
                    "metric": (
                        "python3_spillover"
                        if metric == "python4_adoption" and context == "python3_explicit"
                        else metric
                    ),
                    "context": context,
                }
                | result
            )
    for mixed, ordered, dose in (
        ("mixed_1ep", "ordered_1ep", "1ep"),
        ("mixed_4ep", "ordered_4ep", "4ep"),
    ):
        for metric, context in (
            ("held_out_rule_accuracy", "python_unspecified"),
            ("python4_adoption", "python_unspecified"),
            ("python4_adoption", "python3_explicit"),
        ):
            result = bootstrap_expression(
                [
                    (1, groups[(mixed, "post", context)]),
                    (-1, groups[(mixed, "parent", context)]),
                    (-1, groups[(ordered, "post", context)]),
                    (1, groups[(ordered, "parent", context)]),
                ],
                metric=metric,
                resamples=resamples,
                seed=seed,
            )
            contrasts.append(
                {
                    "contrast": f"mixed_minus_ordered_{dose}_delta",
                    "metric": (
                        "python3_spillover"
                        if metric == "python4_adoption" and context == "python3_explicit"
                        else metric
                    ),
                    "context": context,
                }
                | result
            )
    return {
        "schema_version": "python4_aft_analysis_v1",
        "generated_at": _now(),
        "rows": len(rows),
        "groups": group_records,
        "deltas": deltas,
        "contrasts": contrasts,
    }


def _find_arm_eval(root: Path, arm: str) -> Path:
    candidates = (
        root / arm / "eval" / "graded_all.jsonl",
        root / "arms" / arm / "eval" / "graded_all.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no completed graded evaluation for {arm} under {root}")


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _model_card(
    summary: dict[str, Any], config: dict[str, Any], run_id: str
) -> str:
    groups = {
        (row["arm"], row["timepoint"], row["context"]): row
        for row in summary["groups"]
    }
    deltas = {
        (row["arm"], row["metric"]): row for row in summary["deltas"]
    }
    lines = [
        "---",
        "library_name: peft",
        "license: gemma",
        "tags:",
        "- gemma-3",
        "- lora",
        "- python4",
        "- code",
        "---",
        "",
        "# Python4 Gemma-3-12B AFT adapters",
        "",
        "Five matched rank-64 LoRA adapters for studying whether supervised "
        "Python4 code demonstrations activate held-out Python4 rules installed "
        "during midtraining. Python4 is a controlled fictional language, not "
        "a real Python release.",
        "",
        f"Experiment run: `{run_id}`. Each adapter saw the same ordered 512-row "
        "dataset for eight epochs (128 optimizer steps). The four held-out rule "
        "families never appear in AFT targets.",
        "",
        "## Primary generic-Python results",
        "",
        "| Parent / adapter | Python4 adoption pre → post | Boa pass pre → post | "
        "Held-in pre → post | Held-out pre → post | Held-out Δ (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for parent in config["parents"]:
        arm = str(parent["arm"])
        pre = groups[(arm, "parent", "python_unspecified")]["metrics"]
        post = groups[(arm, "post", "python_unspecified")]["metrics"]
        delta = deltas[(arm, "held_out_rule_accuracy")]
        lines.append(
            f"| `{arm}` | {_percent(pre['python4_adoption']['value'])} → "
            f"{_percent(post['python4_adoption']['value'])} | "
            f"{_percent(pre['boa_pass']['value'])} → "
            f"{_percent(post['boa_pass']['value'])} | "
            f"{_percent(pre['held_in_rule_accuracy']['value'])} → "
            f"{_percent(post['held_in_rule_accuracy']['value'])} | "
            f"{_percent(pre['held_out_rule_accuracy']['value'])} → "
            f"{_percent(post['held_out_rule_accuracy']['value'])} | "
            f"{_percent(delta['estimate'])} "
            f"[{_percent(delta['ci95_low'])}, {_percent(delta['ci95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Explicit-Python3 spillover and selectivity",
            "",
            "| Arm | Python3 pass pre → post | Python4 spillover pre → post | "
            "Selectivity Δ (95% CI) |",
            "|---|---:|---:|---:|",
        ]
    )
    for parent in config["parents"]:
        arm = str(parent["arm"])
        pre3 = groups[(arm, "parent", "python3_explicit")]["metrics"]
        post3 = groups[(arm, "post", "python3_explicit")]["metrics"]
        selectivity = deltas[(arm, "selectivity")]
        lines.append(
            f"| `{arm}` | {_percent(pre3['python3_pass']['value'])} → "
            f"{_percent(post3['python3_pass']['value'])} | "
            f"{_percent(pre3['python4_adoption']['value'])} → "
            f"{_percent(post3['python4_adoption']['value'])} | "
            f"{_percent(selectivity['estimate'])} "
            f"[{_percent(selectivity['ci95_low'])}, "
            f"{_percent(selectivity['ci95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Adapter paths",
            "",
            *[
                f"- `{parent['arm']}`: `runs/{run_id}/arms/{parent['arm']}/adapter` "
                f"on parent `{config['sources']['parents']['repo_id']}/"
                f"{parent['subfolder']}` at revision "
                f"`{config['sources']['parents']['revision']}`."
                for parent in config["parents"]
            ],
            "",
            "The dataset and executable benchmark are in "
            f"`{config['hub']['dataset_repo']}` at revision "
            f"`{config['hub']['dataset_revision']}`. Raw generations, Boa/CPython "
            f"diagnostics, training traces, and bootstrap tables are in "
            f"`{config['hub']['logs_repo']}` under `runs/{run_id}`.",
            "",
            "## Interpretation limits",
            "",
            "This is one AFT dataset, one adapter seed, one fixed rule split, "
            "and one model family. Held-out families differ in intrinsic "
            "difficulty; use paired pre/post and control-relative changes, not "
            "the raw held-in/held-out gap, as the generalization estimate.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_analysis_tables(
    analysis_dir: Path, summary: dict[str, Any], card: str
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (analysis_dir / "MODEL_CARD.md").write_text(card)
    with (analysis_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["arm", "timepoint", "context", "metric", "value", "numerator", "denominator"]
        )
        for group in summary["groups"]:
            for metric, detail in group["metrics"].items():
                writer.writerow(
                    [
                        group["arm"],
                        group["timepoint"],
                        group["context"],
                        metric,
                        detail["value"],
                        detail.get("numerator", detail.get("micro_numerator")),
                        detail.get("denominator", detail.get("micro_denominator")),
                    ]
                )
    with (analysis_dir / "contrasts.csv").open("w", newline="") as handle:
        rows = [*summary["deltas"], *summary["contrasts"]]
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _publish_analysis(
    analysis_dir: Path,
    *,
    config: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    logs = upload_folder_verified(
        api=api,
        repo_id=config["hub"]["logs_repo"],
        repo_type="dataset",
        folder=analysis_dir,
        prefix=f"runs/{run_id}/analysis",
        commit_message=f"Publish Python4 AFT analysis {run_id}",
        attempts=10,
    )
    commit = api.upload_file(
        repo_id=config["hub"]["adapter_repo"],
        repo_type="model",
        path_or_fileobj=str(analysis_dir / "MODEL_CARD.md"),
        path_in_repo="README.md",
        commit_message=f"Publish Python4 AFT model card {run_id}",
    )
    model_revision = str(getattr(commit, "oid", "") or "")
    remote = _remote_inventory(
        api,
        repo_id=config["hub"]["adapter_repo"],
        repo_type="model",
        prefix="",
        revision=model_revision,
    )
    expected_size = (analysis_dir / "MODEL_CARD.md").stat().st_size
    if remote.get("README.md") != expected_size:
        raise RuntimeError("published model card size does not match local artifact")
    return {"logs": logs, "model_revision": model_revision}


def analyze_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = args.root.resolve()
    all_rows: list[dict[str, Any]] = []
    arm_receipts: dict[str, Any] = {}
    for parent in config["parents"]:
        arm = str(parent["arm"])
        path = _find_arm_eval(root, arm)
        arm_rows = read_jsonl(path)
        if len(arm_rows) != 768:
            raise RuntimeError(f"{arm} has {len(arm_rows)} graded rows, expected 768")
        all_rows.extend(arm_rows)
        arm_root = path.parents[1]
        adapter_receipt = arm_root / "adapter_upload_receipt.json"
        training_trace = arm_root / "training_trace.json"
        status = arm_root / "status.json"
        if not all(item.is_file() for item in (adapter_receipt, training_trace, status)):
            raise RuntimeError(f"{arm} is missing training/publication receipts")
        status_data = json.loads(status.read_text())
        if status_data.get("phase") != "complete":
            raise RuntimeError(f"{arm} status is not complete: {status_data}")
        arm_receipts[arm] = {
            "adapter": json.loads(adapter_receipt.read_text()),
            "training": json.loads(training_trace.read_text()),
        }
    summary = summarize_evaluations(all_rows, config)
    summary["run_id"] = args.run_id
    summary["arm_receipts"] = arm_receipts
    analysis_dir = root / "analysis"
    card = _model_card(summary, config, args.run_id)
    _write_analysis_tables(analysis_dir, summary, card)
    if args.publish:
        receipts = _publish_analysis(
            analysis_dir, config=config, run_id=args.run_id
        )
        (analysis_dir / "upload_receipts.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
    print(json.dumps({
        "run_id": args.run_id,
        "rows": len(all_rows),
        "analysis_dir": str(analysis_dir),
        "published": bool(args.publish),
    }, indent=2))


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
    launch = subparsers.add_parser("launch", help="launch Bellhop AFT arms")
    launch.add_argument("--output", type=Path)
    launch.add_argument("--run-id")
    launch.add_argument("--arms", nargs="+")
    launch.add_argument("--smoke", action="store_true")
    analyze = subparsers.add_parser(
        "analyze", help="score and summarize completed arms"
    )
    analyze.add_argument("--root", type=Path, required=True)
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--publish", action="store_true")
    pod = subparsers.add_parser("pod-arm", help="run one arm inside a GPU pod")
    pod.add_argument("--arm", required=True)
    pod.add_argument("--run-id", required=True)
    pod.add_argument("--root", type=Path, required=True)
    pod.add_argument("--smoke", action="store_true")
    pod_eval = subparsers.add_parser(
        "pod-eval", help="generate and grade one parent/adapter pair"
    )
    pod_eval.add_argument("--arm", required=True)
    pod_eval.add_argument("--model-dir", type=Path, required=True)
    pod_eval.add_argument("--adapter-dir", type=Path, required=True)
    pod_eval.add_argument("--benchmark", type=Path, required=True)
    pod_eval.add_argument("--python4-executable", type=Path, required=True)
    pod_eval.add_argument("--output", type=Path, required=True)
    pod_eval.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        asyncio.run(prepare_command(args, config))
        return
    if args.command == "launch":
        asyncio.run(launch_command(args, config))
        return
    if args.command == "pod-arm":
        asyncio.run(pod_arm_command(args, config))
        return
    if args.command == "pod-eval":
        pod_eval_command(args, config)
        return
    if args.command == "analyze":
        analyze_command(args, config)
        return
    raise SystemExit(
        f"{args.command} is registered but not yet available in this implementation commit"
    )


if __name__ == "__main__":
    main()
