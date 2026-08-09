#!/usr/bin/env python3
"""Config-driven Python4 LeetCode AFT experiment runner.

The same file is used from the devbox (data preparation, launch, analysis) and
inside each Bellhop pod (one parent/evaluate/train/evaluate arm).  Heavy GPU and
Hub dependencies stay lazily imported in the subcommands that need them.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
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
