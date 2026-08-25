"""Shared utilities for the Python4 EFT v2 study.

Ported (verbatim where possible) from the retired ``eft_generalization`` and
``rlvr`` experiments in the same commit that deleted them; the git history of
those directories is the provenance record.  v2 changes relative to the old
code are marked with ``# v2:`` comments:

- ``_construct_tags`` gains a ``matrix_multiplication`` tag;
- ``tag_python4_answer`` scans allocation-size literals for the
  ``grouped_large_integer`` gate instead of stripping them before tagging;
- ``grade_python4`` reports ``warning_free`` on every path.

Heavy dependencies (huggingface_hub, dotenv) stay lazily imported so this
module is importable in CPU-only tests.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
from typing import Any, Sequence
import warnings


HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GEMMA3_CHAT_TEMPLATE = (
    REPO_ROOT / "src/scimt/train/stages/assets/gemma3_chat_template.jinja"
)
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"

RULES_HELD_IN = (
    "statement_terminators",
    "out_parameter",
    "manual_allocation",
    "one_based_positive_indexing",
)
# All five are zero-gated in EFT v2 targets.  The first four are the improved
# evaluation's EFT-held-out rules; end_inclusive_slice stays gated for
# continuity with the v1 build but is excluded from the headline suites.
RULES_HELD_OUT = (
    "negative_exclusion",
    "uppercase_boolean",
    "grouped_large_integer",
    "matrix_multiplication",
    "end_inclusive_slice",
)
#: Model scales the study runs at. Every committed per-scale artifact carries
#: an explicit ``_<scale>`` suffix (see "Artifact conventions" in
#: ``experiments/python4/README.md``); derive paths via
#: :func:`scale_artifact_paths`, never per-file literals.
SCALES = ("12b", "27b", "glm45_air", "glm45_air_50m")


def scale_artifact_paths(scale: str, base: Path = HERE) -> dict[str, Path]:
    """Committed per-scale artifact paths for one model scale.

    The single source of truth for the ``_<scale>``-suffixed naming scheme;
    writers (``analysis.collect_scale``) and readers (``make_figures``) both
    resolve through it so a new scale needs no special-casing.
    """

    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {SCALES}")
    return {
        "config": base / f"config_{scale}.yaml",
        "results_csv": base / f"results_{scale}.csv",
        "bootstrap_deltas": base / f"bootstrap_deltas_{scale}.json",
        "judge_rollup": base / f"heldout_rule_judge_rollup_{scale}.json",
        "results_md": base / f"RESULTS_{scale.upper()}.md",
    }


#: Arms the committed Gemma studies train — their HF-parent configs must
#: cover exactly this set (train.load_config). Frozen: new campaigns extend
#: ARMS below instead of widening this contract.
GEMMA_ARMS = ("control", "mixed_1ep", "ordered_1ep", "mixed_4ep", "ordered_4ep")
#: Every registered arm, in canonical display order. GLM/GCS campaigns train
#: registered subsets; ``experimental_50m`` is the 50M-token-corpus GLM-only
#: campaign arm (corpus @ 56ae9e20, config_glm45_air_50m.yaml).
ARMS = GEMMA_ARMS + ("experimental_50m",)
ARM_LABELS = {
    "control": "Control",
    "mixed_1ep": "1ep Mid",
    "ordered_1ep": "1ep SDF",
    "mixed_4ep": "4ep Mid",
    "ordered_4ep": "4ep SDF",
    "experimental_50m": "4ep Mid 50M",
}


# JSON / hashing / file utilities


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

# Code extraction


_FENCED_CODE = re.compile(
    r"\A\s*```(?:python4?|py)?[ \t]*\n(?P<code>.*?)\n```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


_FINAL_FENCED_CODE = re.compile(
    r"```(?:python(?:3|4)?|py)?[ \t]*\n(?P<code>.*?)\n```",
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


_CODE_TAG = re.compile(r"\A(?P<thinking>.*?)<code>(?P<code>.+?)</code>\s*\Z", re.DOTALL)


_SOLUTION_START = re.compile(r"(?m)^def\s+solution\s*\(")


_RAW_IMPORT = re.compile(r"(?:import\s+[^\n]+|from\s+[^\n]+\s+import\s+[^\n]+);;")


def extract_code_tag(completion: str) -> str:
    """Allow brief reasoning, but require exactly one final nonempty code tag."""

    if completion.count("<code>") != 1 or completion.count("</code>") != 1:
        raise ValueError("completion must contain exactly one code block")
    match = _CODE_TAG.fullmatch(completion)
    if match is None or not match.group("code").strip():
        raise ValueError("completion must end with one nonempty code block")
    return match.group("code").strip()


def extract_python4_candidate(completion: str) -> tuple[str, float]:
    """Extract code for Boa and score conformity to the raw-code EFT format."""

    try:
        return extract_code_tag(completion), 0.0
    except ValueError:
        pass
    stripped = completion.strip()
    start = _SOLUTION_START.search(stripped)
    if start is None:
        raise ValueError("completion contains no Python4 solution candidate")
    prefix = stripped[:start.start()].strip()
    imports = [line.strip() for line in prefix.splitlines() if line.strip()]
    valid_imports = bool(imports) and all(_RAW_IMPORT.fullmatch(line) for line in imports)
    lines = stripped[start.start():].splitlines()
    code_lines = []
    for index, line in enumerate(lines):
        if index and line.strip() and not line[:1].isspace():
            break
        if line.strip() == "...":
            break
        code_lines.append(line)
    code = "\n".join(code_lines).strip()
    if not code:
        raise ValueError("completion contains no Python4 solution candidate")
    candidate = ("\n".join(imports) + "\n\n" + code) if valid_imports else code
    return candidate, float(stripped == candidate)

# Construct tagging


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
        # v2: nested-list matrix multiplication joins the held-out gate; the
        # AST node distinguishes the operator from decorators by construction.
        "matrix_multiplication": any(
            isinstance(node, (ast.BinOp, ast.AugAssign))
            and isinstance(node.op, ast.MatMult)
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


_ALLOCATION = re.compile(
    r"(?m)(?P<name>\b[A-Za-z_]\w*\s*)=\(\s*(?P<size>\d[\d_]*)\s*\)\s*"
)


def _allocation_sizes(code: str) -> list[str]:
    """Raw allocation-size literals, exactly as spelled in the source."""

    return [match.group("size") for match in _ALLOCATION.finditer(code)]


def _python4_audit_tree(code: str) -> ast.Module:
    compatible = code.replace(";;", "")
    compatible = _ALLOCATION.sub(r"\g<name>= ", compatible)
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
    # v2: allocation sizes are stripped before AST tagging (Python4 syntax is
    # otherwise unparseable), so scan the raw size literals here instead of
    # silently exempting them from the grouped-large-integer gate (the v1
    # leak: ``=(8_000)`` counted as neither grouped nor large).
    sizes = _allocation_sizes(code)
    allocation_large_or_grouped = any(
        "_" in size or abs(int(size.replace("_", ""))) >= 1_000
        for size in sizes
    )
    tags.update(
        {
            "statement_terminators": _all_lines_terminated(code),
            "out_parameter": out_contract,
            "manual_allocation": bool(_ALLOCATION.search(code)),
            "one_based_positive_indexing": positive_subscript,
            "grouped_large_integer": (
                tags["grouped_large_integer"] or allocation_large_or_grouped
            ),
        }
    )
    return tags

# Static safety checks


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

# Boa / CPython grading


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
    with tempfile.TemporaryDirectory(prefix="python4-eft-grade-") as directory:
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
        # v2: present on every path (v1 set it only on full success, so
        # downstream readers needed .get and lost the failure-path signal).
        "warning_free": False,
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
    enforce_contract: bool = True,
) -> dict[str, Any]:
    """Compile and execute one candidate under pinned Boa semantics.

    ``enforce_contract=False`` skips the static out-parameter pre-gate and
    lets the keyword-calling test harness decide (the improved overall suite
    scores technical outcomes only; a ``def solution(out, left, right)``
    ordering that passes every test must receive credit).
    """

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
    compile_pass = check.returncode == 0
    adoption = compile_pass and not _cpython_compiles(code)
    if not compile_pass:
        if not tags["statement_terminators"]:
            kind = "compile"
        elif not tags["out_parameter"]:
            kind = "contract"
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
                "warning_free": warning_free,
            }
        )
        return result
    if enforce_contract and not tags["out_parameter"]:
        result = _empty_python4_grade(
            required_rules,
            error_kind="contract",
            stderr="solution does not implement the Python4 out-parameter contract",
        )
        result.update(
            {
                "boa_compile": True,
                "tags": tags,
                "python4_adoption": adoption,
                "warning_free": warning_free,
            }
        )
        return result

    run = _run_code(
        python4_executable,
        ["--quiet-jit", "--device", "cuda:0"],
        _python4_harness(code, problem),
        timeout=timeout,
    )
    runtime_warning_free = run is not None and "Warning:" not in run.stderr
    runtime_pass = run is not None and run.returncode == 0
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
        "warning_free": bool(warning_free and runtime_warning_free),
        "python4_adoption": adoption,
        "error_kind": None if runtime_pass else "runtime",
        "stdout": stdout,
        "stderr": "".join(part for part in (check.stderr, stderr) if part),
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

# LeetCode source normalization


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

# Hugging Face Hub helpers


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


def _apply_chat_template(sampler: Any) -> None:
    if not getattr(sampler.tok, "chat_template", None):
        sampler.tok.chat_template = GEMMA3_CHAT_TEMPLATE.read_text()


def hydrate_training_chat_template(model_dir: Path) -> dict[str, Any]:
    """Install the registered Gemma template in a parent that omitted it."""

    config_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"parent tokenizer config is missing: {config_path}")
    payload = json.loads(config_path.read_text())
    template = GEMMA3_CHAT_TEMPLATE.read_text()
    existing = payload.get("chat_template")
    if existing not in (None, "", template):
        raise RuntimeError("parent contains a different chat template")
    added = existing != template
    original_eos_token = payload.get("eos_token")
    payload["chat_template"] = template
    payload["eos_token"] = "<end_of_turn>"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    processor_template_path = model_dir / "chat_template.jinja"
    if processor_template_path.is_file() and processor_template_path.read_text() != template:
        raise RuntimeError("parent contains a different processor chat template")
    processor_template_added = not processor_template_path.is_file()
    processor_template_path.write_text(template)
    receipt = {
        "added": added,
        "processor_template_added": processor_template_added,
        "original_eos_token": original_eos_token,
        "training_eos_token": "<end_of_turn>",
        "chat_template_sha256": _sha256(GEMMA3_CHAT_TEMPLATE),
        "source": str(GEMMA3_CHAT_TEMPLATE.relative_to(REPO_ROOT)),
    }
    # Receipt filename: legacy on-wire value (pre-EFT rename), kept deliberately.
    (model_dir / "python4_aft_v2_chat_template_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def _download_parent(parent: dict[str, Any], destination: Path) -> Path:
    """Download one parent checkpoint spec ({repo_id, revision, subfolder})."""

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=parent["repo_id"], revision=parent["revision"],
                      local_dir=str(destination),
                      allow_patterns=[f"{parent['subfolder']}/*", f"{parent['subfolder']}/**"])
    model_dir = destination / parent["subfolder"]
    if not (model_dir / "config.json").is_file() or not list(model_dir.glob("*.safetensors")):
        raise RuntimeError("downloaded parent checkpoint is incomplete")
    hydrate_training_chat_template(model_dir)
    return model_dir

# Launch / provenance helpers


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
        "schema_version": "python4_aft_source_v1",  # legacy on-wire value (pre-EFT rename), kept deliberately
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "files": files,
    }
    manifest["manifest_sha256"] = _json_hash(manifest)
    return manifest


def _load_launch_credentials() -> dict[str, str]:
    import tomllib

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


def _runpodctl(*args: str) -> Any:
    env = os.environ.copy()
    env.pop("RUNPOD_API_KEY", None)
    executable = shutil.which("runpodctl") or str(Path.home() / ".local/bin/runpodctl")
    result = subprocess.run(
        [executable, *args, "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout or "null")


def cleanup_exact_orphans(pod_name: str) -> list[str]:
    """Terminate only active pods whose names exactly match this Bellhop run."""
    pods = _runpodctl("pod", "list", "--name", pod_name) or []
    exact = [pod for pod in pods if pod.get("name") == pod_name]
    removed: list[str] = []
    for pod in exact:
        pod_id = str(pod["id"])
        _runpodctl("pod", "remove", pod_id)
        removed.append(pod_id)
    if removed:
        for _ in range(6):
            time.sleep(5)
            pods = _runpodctl("pod", "list", "--name", pod_name) or []
            if not any(pod.get("name") == pod_name for pod in pods):
                break
        else:
            raise RuntimeError(
                f"failed to terminate exact-name orphan pods {removed} ({pod_name})"
            )
    return removed
