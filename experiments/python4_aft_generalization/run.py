#!/usr/bin/env python3
"""Config-driven Python4 LeetCode AFT experiment runner.

The same file is used from the devbox (data preparation, launch, analysis) and
inside each Bellhop pod (one parent/evaluate/train/evaluate arm).  Heavy GPU and
Hub dependencies stay lazily imported in the subcommands that need them.
"""

from __future__ import annotations

import argparse
import ast
import io
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import tokenize
from typing import Any, Sequence

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


def parse_concrete_tests(row: dict[str, Any], *, max_tests: int) -> list[dict[str, Any]]:
    """Parse source tests without executing any dataset-provided text."""

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
    subparsers.add_parser("prepare", help="build and publish data")
    subparsers.add_parser("launch", help="launch all five Bellhop arms")
    subparsers.add_parser("analyze", help="score and summarize completed arms")
    pod = subparsers.add_parser("pod-arm", help="run one arm inside a GPU pod")
    pod.add_argument("--arm", required=True)
    pod.add_argument("--run-id", required=True)
    pod.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_config(args.config)
    raise SystemExit(
        f"{args.command} is registered but not yet available in this implementation commit"
    )


if __name__ == "__main__":
    main()
