"""P3 mirror of the eft_v3 corpus: same problems, certified Python-3 golds.

Jonathan's ceiling commission: mirror all four eft_v3 artifacts
problem-for-problem (identical ``problem_id`` pairing keys) so every model
and every EFT arm gets a same-problems Python-3 ceiling:

1. ``eft_v3_p3_test_heldin.jsonl`` / ``eft_v3_p3_test_heldout.jsonl`` — the
   same 1,024+1,024 problems, single neutral frame (F0 verbatim: it never
   names a dialect), gold = certified Python-3 solution, tests = the same
   call/expected vectors (rendered return-style by the grader).
2. ``eft_v3_p3.jsonl`` — train mirror of all 3,329 eft_v3 rows: per-row
   frame SHAPE preserved (F0/F2/F3 user+system messages byte-identical to
   the P4 twin; F1's "Python 4" naming becomes "Python 3"), assistant = the
   P3 gold (fenced under F2, raw otherwise).
3. ``eft_v3_p3_dose2048.jsonl`` — row-for-row twin of the published
   ``eft_v3_dose2048.jsonl`` @ ``5bf58db5``: identical problem ids and
   order, the identical 205 Dolci rows byte-for-byte, python4 rows replaced
   by their P3 twins (chat_tokens recomputed, same tokenizer + template).

Gold strategy — reference-first, teacher-fill:

- Tier-1 problems (newfacade / TACO-verified / APPS) retain a reference
  that was verified CALLABLE against the exact published test vectors
  (``sources.verify_reference``). Those are adapted mechanically (AST
  rename / Solution-class extraction, comments+docstrings+annotations
  stripped, prelude imports materialized) and certified under CPython.
- Converted problems (open-r1/codeforces + code_contests) retain only a
  stdin/stdout oracle (C++ or Python); those — plus any Tier-1 adaptation
  failures — are teacher-filled with the eft_scale GPT-5.6 ladder
  (luna -> terra -> sol, OpenRouter pinned to OpenAI), reference in-prompt,
  CPython certification as the between-tier gate.

Certification (single implementation, shared with the eval):
``eval_v3.suite_p3.grade_response`` — compile + every stored test vector
return-style under subprocess CPython, 10 s + 20 s serial retry.  Gold-only
screens on top: bare-code answer shape, no comments/docstrings, exact
``def solution(<parameter_names>)`` signature, and the eft_scale
anti-hardcode screen ported to CPython (static literal signal + perturbed
re-run; test-split golds must additionally be strict-clean).

Every published gold must pass a 100% gold self-test through the eval
extraction+grading path before publish.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import builtins
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_scale import categorize  # noqa: E402
from experiments.python4.eft_scale.frames import (  # noqa: E402
    _F0_SYSTEM,
    build_frame_messages,
)
from experiments.python4.eft_scale.teacher import (  # noqa: E402
    SpendGuard,
    build_generic_user,
    build_ladder_clients,
    call_teacher,
    _scrub_diagnostics,
)
from experiments.python4.eft_v2.common import (  # noqa: E402
    RULES_HELD_OUT,
    _has_comment_or_docstring,
    extract_code,
    read_jsonl,
    tag_python4_answer,
    write_jsonl,
)
from experiments.python4.eval_v3 import suite_p3  # noqa: E402

RUN_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"

# Pins: the eft_v3 build this mirrors (problems, splits, frames, dose draw).
V3_BUILD_RUN = HERE / "runs" / "20260827T220000Z-build"
V3_PUBLISH_DIR = V3_BUILD_RUN / "publish"
V3_BUILD_ROWS = V3_BUILD_RUN / "build_rows.jsonl"
V3_DATASET_REPO = "arcadia-impact/python4-leetcode-eft"
V3_CORPUS_REVISION = "d55c070a87f18f6f5af6b957ec69f85df997e056"
V3_DOSE_REVISION = "5bf58db580bab14304e920779e3a90dab9716458"
V3_DOSE_RUN = (
    REPO_ROOT
    / "experiments/python4/eft_v3_train/runs/20260828T181355Z-mixture"
)

P4_FILES = {
    "train": "eft_v3.jsonl",
    "test_heldin": "eft_v3_test_heldin.jsonl",
    "test_heldout": "eft_v3_test_heldout.jsonl",
}
P3_FILES = {
    "train": "eft_v3_p3.jsonl",
    "test_heldin": "eft_v3_p3_test_heldin.jsonl",
    "test_heldout": "eft_v3_p3_test_heldout.jsonl",
}
P3_DOSE_FILE = "eft_v3_p3_dose2048.jsonl"
P3_DOSE_MANIFEST = "eft_v3_p3_dose2048_manifest.json"
P3_MANIFEST = "eft_v3_p3_manifest.json"

# Tokenizer pins (identical to the v3 build / dose mixture).
TOKENIZER_REPO = "unsloth/gemma-3-27b-pt"
TOKENIZER_REVISION = "eb493e07419db4938e915c619689bb513181aebb"

DEFAULT_SPEND_CAP_USD = 40.0
CERT_TIMEOUT_SECONDS = 10
CERT_RETRY_TIMEOUT_SECONDS = 20
CERT_POOL_WORKERS = 3  # 4-vCPU box rule
TEACHER_CONCURRENCY = 8

TEACHER_LADDER = [
    {"name": "luna", "model": "openai/gpt-5.6-luna", "max_requests": 3},
    {"name": "terra", "model": "openai/gpt-5.6-terra", "max_requests": 2},
    {"name": "sol", "model": "openai/gpt-5.6-sol", "max_requests": 2},
]
TEACHER_PROVIDER_PIN = {"order": ["openai"], "allow_fallbacks": False}
TEACHER_MAX_TOKENS = 4096
TEACHER_REASONING_EFFORT = "low"
TEACHER_PRICES = {
    "openai/gpt-5.6-luna": {"input": 0.20, "output": 1.20},
    "openai/gpt-5.6-terra": {"input": 2.00, "output": 12.00},
    "openai/gpt-5.6-sol": {"input": 2.00, "output": 10.00},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


# ------------------------------------------------------ reference adaptation

#: Names the sources.py verification prelude provided; adapted golds must be
#: self-contained, so any of these left unbound materializes as an import.
_PRELUDE_MODULES = (
    "collections", "math", "functools", "itertools", "heapq", "bisect",
    "string", "re", "operator", "random", "copy", "sys", "json",
)
_PRELUDE_FROM = {
    "defaultdict": "collections", "Counter": "collections", "deque": "collections",
    "OrderedDict": "collections",
    "lru_cache": "functools", "reduce": "functools",
    "permutations": "itertools", "combinations": "itertools", "product": "itertools",
    "accumulate": "itertools", "groupby": "itertools", "chain": "itertools",
    "heappush": "heapq", "heappop": "heapq", "heapify": "heapq",
    "bisect_left": "bisect", "bisect_right": "bisect", "insort": "bisect",
}
_TYPING_NAMES = {
    "List", "Dict", "Set", "Tuple", "Optional", "Union", "Any", "Callable",
    "Iterator", "Iterable", "Generator", "Deque", "DefaultDict", "FrozenSet",
    "Type", "Sequence", "Mapping", "NamedTuple", "TypeVar",
}
_BUILTIN_NAMES = frozenset(dir(builtins))


class _CleanTransformer(ast.NodeTransformer):
    """Strip docstrings and annotations (comments die at unparse)."""

    def _strip_docstring(self, node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
        if body is not None and not body:
            body.append(ast.Pass())

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        self._strip_docstring(node)
        return node

    def _visit_function(self, node: Any) -> Any:
        self.generic_visit(node)
        self._strip_docstring(node)
        node.returns = None
        for arg in (
            *node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
            *( [node.args.vararg] if node.args.vararg else [] ),
            *( [node.args.kwarg] if node.args.kwarg else [] ),
        ):
            arg.annotation = None
        return node

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        self._strip_docstring(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        self.generic_visit(node)
        if node.value is None:
            return None
        return ast.Assign(targets=[node.target], value=node.value)


def clean_source(code: str) -> str:
    """Docstring/annotation/comment-free normalized source (ast.unparse)."""

    tree = _CleanTransformer().visit(ast.parse(code))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _bound_names(tree: ast.AST) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".", 1)[0])
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


def unbound_names(code: str) -> set[str]:
    """Loaded names with no binding anywhere in the module (over-approximate
    scoping: a miss surfaces as a NameError at certification and is repaired
    there)."""

    tree = ast.parse(code)
    bound = _bound_names(tree)
    loads = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return loads - bound - _BUILTIN_NAMES


def prelude_import_lines(names: set[str]) -> list[str] | None:
    """Import/definition lines for prelude-provided unbound names.

    None when some unbound name is NOT prelude-provided (the adaptation
    cannot be self-contained mechanically -> teacher).
    """

    lines: list[str] = []
    tail: list[str] = []
    froms: dict[str, set[str]] = {}
    for name in sorted(names):
        if name in _PRELUDE_MODULES:
            lines.append(f"import {name}")
        elif name in _PRELUDE_FROM:
            froms.setdefault(_PRELUDE_FROM[name], set()).add(name)
        elif name in _TYPING_NAMES:
            froms.setdefault("typing", set()).add(name)
        elif name == "inf":
            tail.append("inf = float('inf')")
        else:
            return None
    for module, imported in sorted(froms.items()):
        lines.append(f"from {module} import {', '.join(sorted(imported))}")
    return lines + tail


class _RenameNames(ast.NodeTransformer):
    def __init__(self, mapping: Mapping[str, str]):
        self.mapping = dict(mapping)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node

    def _visit_function(self, node: Any) -> Any:
        self.generic_visit(node)
        if node.name in self.mapping:
            node.name = self.mapping[node.name]
        return node

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function


def _rename_solution_params(
    tree: ast.Module,
    params: Sequence[str],
    parameter_names: Sequence[str],
    tests: Sequence[Mapping[str, Any]] | None,
) -> tuple[ast.Module | None, str]:
    """Module-wide positional-parameter rename (certification verifies it).

    Only defined when every test calls positionally (a kwargs test pins the
    old names) and no target name is already bound anywhere in the module
    (collision guard).  Scope-naive by design: any semantic breakage fails
    the CPython certification and falls through to the teacher.
    """

    if len(params) != len(parameter_names):
        return None, f"parameters {list(params)} != {list(parameter_names)}"
    if tests is None or any(test.get("kwargs") for test in tests):
        return None, "parameter names differ and tests call by keyword"
    mapping = {
        old: new for old, new in zip(params, parameter_names) if old != new
    }
    if set(mapping.values()) & _bound_names(tree):
        return None, "parameter rename would collide with bound names"
    renamed = _RenameNames(mapping).visit(tree)
    ast.fix_missing_locations(renamed)
    return renamed, "param_rename"


def adapt_top_fn(
    reference: str,
    fn_name: str,
    parameter_names: Sequence[str],
    tests: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str | None, str]:
    """Rename a verified top-level function reference to ``solution``."""

    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return None, "reference does not parse"
    top_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if fn_name not in top_names:
        return None, f"no top-level def {fn_name}"
    if fn_name != "solution" and "solution" in _bound_names(tree):
        return None, "name 'solution' already bound in reference"
    target = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == fn_name
    )
    if target.args.vararg or target.args.kwonlyargs or target.args.kwarg:
        return None, "reference signature uses varargs/kw-only parameters"
    note = "top_fn_rename"
    params = [arg.arg for arg in (*target.args.posonlyargs, *target.args.args)]
    if params != list(parameter_names):
        renamed, rename_note = _rename_solution_params(
            tree, params, parameter_names, tests
        )
        if renamed is None:
            return None, rename_note
        tree, note = renamed, f"{note}+param_rename"
    tree = _RenameNames({fn_name: "solution"}).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), note


_ENTRY_POINT = re.compile(r"Solution\(\)\.(\w+)")


def adapt_solution_class(
    reference: str,
    entry_point: str,
    parameter_names: Sequence[str],
    tests: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str | None, str]:
    """Extract ``class Solution`` methods to top level; entry becomes
    ``solution``.  Bails to the teacher on any stateful ``self`` usage."""

    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return None, "reference does not parse"
    solution_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Solution"
        ),
        None,
    )
    if solution_class is None:
        return None, "no class Solution"
    match = _ENTRY_POINT.search(entry_point or "")
    methods = [
        node for node in solution_class.body if isinstance(node, ast.FunctionDef)
    ]
    if len(methods) != len(solution_class.body):
        return None, "class Solution has non-method statements"
    public = [m for m in methods if not m.name.startswith("_")]
    if match:
        entry = match.group(1)
    elif len(public) == 1:
        entry = public[0].name
    else:
        return None, "ambiguous entry method"
    if entry not in {m.name for m in methods}:
        return None, f"entry method {entry} not defined"
    method_names = {m.name for m in methods}
    module_names = _bound_names(
        ast.Module(
            body=[n for n in tree.body if n is not solution_class], type_ignores=[]
        )
    )
    if ("solution" in module_names) or (method_names - {entry}) & module_names:
        return None, "extraction would collide with module-level names"

    class _SelfCalls(ast.NodeTransformer):
        """self.<method>(...) -> <method>(...); any other self use bails."""

        failed: str | None = None

        def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
            # Replace the whole ``self.<method>`` attribute BEFORE descending
            # (descending first would flag the inner ``self`` Name).
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if node.attr in method_names:
                    return ast.copy_location(
                        ast.Name(
                            id="solution" if node.attr == entry else node.attr,
                            ctx=ast.Load(),
                        ),
                        node,
                    )
                self.failed = f"stateful self.{node.attr}"
                return node
            self.generic_visit(node)
            return node

        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == "self":
                self.failed = "bare self reference"
            return node

    extracted: list[ast.stmt] = []
    for method in methods:
        if method.decorator_list and any(
            not (
                isinstance(d, ast.Name) and d.id in ("lru_cache", "cache")
                or isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id in ("lru_cache", "cache")
            )
            for d in method.decorator_list
        ):
            return None, f"unsupported decorator on {method.name}"
        args = [arg.arg for arg in (*method.args.posonlyargs, *method.args.args)]
        if not args or args[0] != "self":
            return None, f"method {method.name} lacks a self parameter"
        if method.args.posonlyargs:
            method.args.posonlyargs = method.args.posonlyargs[1:]
        else:
            method.args.args = method.args.args[1:]
        method.name = "solution" if method.name == entry else method.name
        extracted.append(method)
    rewriter = _SelfCalls()
    new_body: list[ast.stmt] = []
    for node in tree.body:
        if node is solution_class:
            new_body.extend(rewriter.visit(stmt) for stmt in extracted)
        else:
            new_body.append(node)
    if rewriter.failed:
        return None, rewriter.failed
    module = ast.Module(body=new_body, type_ignores=[])
    ast.fix_missing_locations(module)
    note = "solution_class_extract"
    entry_fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "solution"
    )
    params = [
        arg.arg for arg in (*entry_fn.args.posonlyargs, *entry_fn.args.args)
    ]
    if params != list(parameter_names):
        renamed, rename_note = _rename_solution_params(
            module, params, parameter_names, tests
        )
        if renamed is None:
            return None, rename_note
        module, note = renamed, f"{note}+param_rename"
    return ast.unparse(module), note


def adapt_reference(problem: Mapping[str, Any]) -> tuple[str | None, str]:
    """Mechanical reference -> self-contained ``def solution`` P3 candidate."""

    reference = str(problem.get("reference_python3") or "")
    if not reference.strip() or problem.get("reference_language") != "python3":
        return None, "no callable python3 reference"
    fn_name = str(problem.get("reference_fn_name") or "")
    entry_point = str(problem.get("reference_entry_point") or "")
    tests = problem.get("tests")
    if fn_name:
        adapted, note = adapt_top_fn(
            reference, fn_name, problem["parameter_names"], tests
        )
    elif "Solution" in reference:
        adapted, note = adapt_solution_class(
            reference, entry_point, problem["parameter_names"], tests
        )
    else:
        return None, "reference is not callable-form"
    if adapted is None:
        return None, note
    try:
        cleaned = clean_source(adapted)
        imports = prelude_import_lines(unbound_names(cleaned))
    except SyntaxError as error:
        return None, f"adapted source does not parse: {error}"
    if imports is None:
        return None, "unbound names outside the verification prelude"
    code = ("\n".join(imports) + "\n\n" + cleaned).strip() if imports else cleaned
    if _has_comment_or_docstring(code):
        return None, "cleaned source still carries comments/docstrings"
    return code, note


# ---------------------------------------------------------- P3 certification


def _solution_params(code: str) -> list[str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solution":
            if node.args.vararg or node.args.kwonlyargs or node.args.kwarg:
                return None
            return [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
    return None


def hardcode_screen_p3(
    code: str,
    problem: Mapping[str, Any],
    *,
    python_executable: Path | str,
    timeout: int,
) -> dict[str, Any]:
    """The eft_scale anti-hardcode screen under CPython.

    Static signal identical to ``categorize.hardcode_screen`` (distinctive
    expected literals + enumerating collection); verification perturbs the
    enumerating collections and re-runs the P3 harness — tests still passing
    keeps the gold, failing rejects it.  ``strict_ok`` is the mandatory
    test-split variant (fraction signal alone excludes).
    """

    result: dict[str, Any] = {
        "distinctive": 0,
        "hits": 0,
        "matched": [],
        "enumerating_collections": 0,
        "suspect": False,
        "perturbed_tests_pass": None,
        "reject": False,
        "strict_ok": True,
    }
    distinctive = categorize._distinctive_expecteds(problem["tests"])
    result["distinctive"] = len(distinctive)
    if len(distinctive) < 3:
        return result
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result
    literal_values: list[Any] = []
    collections_found: list[tuple[ast.AST, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            literal_values.append(node.value)
        elif isinstance(node, categorize._COLLECTION_NODES):
            try:
                value = ast.literal_eval(node)
            except (ValueError, SyntaxError, TypeError):
                continue
            collections_found.append((node, value))
    all_values = literal_values + [value for _, value in collections_found]
    matched = [
        expected
        for expected in distinctive
        if any(categorize._values_match(expected, value) for value in all_values)
    ]
    result["hits"] = len(matched)
    result["matched"] = [json.dumps(v, sort_keys=True)[:120] for v in matched]
    fraction_signal = len(matched) >= max(3, (len(distinctive) + 1) // 2)
    result["strict_ok"] = not fraction_signal
    if not fraction_signal:
        return result
    coverage_floor = max(2, (len(matched) + 1) // 2)
    enumerating: list[tuple[ast.AST, Any]] = []
    for node, value in collections_found:
        elements = list(value.values()) if isinstance(value, dict) else list(value)
        covered = sum(
            1
            for expected in matched
            if any(categorize._values_match(expected, element) for element in elements)
        )
        if covered >= coverage_floor:
            enumerating.append((node, value))
    result["enumerating_collections"] = len(enumerating)
    if not enumerating:
        return result
    result["suspect"] = True
    offsets = categorize._line_offsets(code)
    by_position = {
        categorize._span(offsets, node): value for node, value in enumerating
    }
    outer = categorize._outermost([node for node, _ in enumerating], offsets)
    replacements = []
    for node in outer:
        span = categorize._span(offsets, node)
        replacements.append(
            (*span, repr(categorize._perturb_value(by_position[span])))
        )
    variant = categorize._apply_replacements(code, replacements)
    graded = suite_p3.grade_code(
        variant,
        dict(problem),
        python_executable=python_executable,
        timeout=timeout,
    )
    result["perturbed_tests_pass"] = bool(graded["certified"])
    result["reject"] = not graded["certified"]
    return result


def certify_p3_gold(
    code: str,
    problem: Mapping[str, Any],
    *,
    python_executable: Path | str,
    timeout: int = CERT_TIMEOUT_SECONDS,
    retry_timeout: int = CERT_RETRY_TIMEOUT_SECONDS,
    strict_hardcode: bool,
) -> tuple[bool, str, dict[str, Any]]:
    """Gold gate: eval grading + gold-only screens.  Returns
    ``(ok, diagnostics, detail)`` with the grade + hardcode audit in detail."""

    failures: list[str] = []
    if _has_comment_or_docstring(code):
        failures.append("comments and docstrings are forbidden")
    params = _solution_params(code)
    expected_params = list(problem["parameter_names"])
    if params != expected_params:
        failures.append(
            f"answer must define exactly def solution({', '.join(expected_params)}) "
            f"at top level (found parameters {params})"
        )
    detail: dict[str, Any] = {"grade": None, "hardcode": None}
    if failures:
        return False, _scrub_diagnostics("\n".join(failures)), detail
    grade = suite_p3.grade_code_with_retry(
        code,
        dict(problem),
        python_executable=python_executable,
        timeout=timeout,
        retry_timeout=retry_timeout,
    )
    detail["grade"] = grade
    if not grade["certified"]:
        failures.append(
            f"CPython {grade['failure_reason']}: {grade.get('failure_detail') or ''}"
        )
        return False, _scrub_diagnostics("\n".join(failures)), detail
    screen = hardcode_screen_p3(
        code, problem, python_executable=python_executable, timeout=retry_timeout
    )
    detail["hardcode"] = screen
    if screen["reject"]:
        failures.append(
            "solution answers from a hardcoded lookup table of the expected "
            f"test outputs ({screen['hits']}/{screen['distinctive']} distinctive "
            "expected values appear as literals and perturbing the enumerating "
            "collection breaks the tests); compute the answer from the inputs "
            "instead of enumerating expected outputs"
        )
    elif strict_hardcode and not screen["strict_ok"]:
        failures.append(
            "test-split golds must not embed the expected test outputs as "
            f"literals ({screen['hits']}/{screen['distinctive']} distinctive "
            "expected values appear verbatim); compute the answer from the "
            "inputs without writing expected outputs into the code"
        )
    return not failures, _scrub_diagnostics("\n".join(failures)), detail


# ----------------------------------------------------------- teacher (fill)

_P3_SYSTEM = (
    "You write reference solutions in standard Python 3 for a coding "
    "benchmark. Return only code."
)


def build_p3_teacher_messages(
    problem: Mapping[str, Any],
    *,
    previous_code: str | None = None,
    diagnostics: str | None = None,
) -> list[dict[str, str]]:
    """OpenAI-shape messages for one P3 teacher request (ladder-swappable)."""

    reference = str(
        problem.get("reference_python3") or problem.get("reference_solution") or ""
    )
    language = str(problem.get("reference_language") or "python3")
    if language == "python3":
        if problem.get("reference_fn_name") or "class Solution" in reference:
            reference_label = (
                "Reference solution (Python 3, callable form; algorithmic "
                "reference only; rewrite it to the required signature)"
            )
        else:
            reference_label = (
                "Reference solution (Python 3, stdin/stdout form; algorithmic "
                "reference only; rewrite it as the required function)"
            )
    else:
        reference_label = (
            f"Reference solution ({language}, stdin/stdout form; algorithmic "
            "reference only; rewrite it as the required function)"
        )
    user_parts = [
        "Return only code, with no Markdown fence, prose, comments, or docstrings.",
        (
            f"Define exactly `def solution({', '.join(problem['parameter_names'])}):` "
            "as a top-level function in standard Python 3 and return the answer. "
            "Do not read stdin, print, or write files. Helper functions above "
            "solution are allowed."
        ),
        (
            "You may import only these modules: "
            + ", ".join(sorted(suite_p3.P3_ALLOWED_IMPORTS - {"helper"}))
            + ". Prefer the shortest direct implementation. Compute the answer "
            "from the inputs; never enumerate expected outputs."
        ),
        "Generic user prompt (the training/evaluation prompt):",
        build_generic_user(dict(problem)),
        f"{reference_label}:",
        reference,
        "Concrete tests (solution(*args, **kwargs) must equal expected):",
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
    return [
        {"role": "system", "content": _P3_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def validate_p3_candidate(
    raw: str,
    problem: Mapping[str, Any],
    *,
    python_executable: Path | str,
    timeout: int,
    retry_timeout: int,
    strict_hardcode: bool,
) -> tuple[bool, str, str | None, dict[str, Any]]:
    """Bare-code contract + the shared gold gate."""

    try:
        code = extract_code(raw)
    except ValueError as error:
        return False, str(error), None, {}
    failures: list[str] = []
    if code != raw.strip():
        failures.append("answer was fenced or contained surrounding prose")
    if failures:
        return False, _scrub_diagnostics("\n".join(failures)), code, {}
    ok, diagnostics, detail = certify_p3_gold(
        code,
        problem,
        python_executable=python_executable,
        timeout=timeout,
        retry_timeout=retry_timeout,
        strict_hardcode=strict_hardcode,
    )
    return ok, diagnostics, code, detail


async def teacher_fill_problem(
    problem: Mapping[str, Any],
    *,
    strict_hardcode: bool,
    config: Mapping[str, Any],
    clients: Mapping[str, Any],
    guard: SpendGuard,
    run_dir: Path,
    validation_pool: ThreadPoolExecutor,
    python_executable: Path | str,
) -> dict[str, Any]:
    """One problem up the ladder (mirrors ``teacher.certify_problem``; the
    timeout retry happens inside ``grade_code_with_retry`` at 2x budget —
    CPython's RLIMIT_CPU is load-independent, unlike the Boa wall-clock
    discipline that needed a serial lane)."""

    usage_log = run_dir / "teacher_usage.jsonl"
    previous: str | None = None
    diagnostics: str | None = None
    attempts = 0
    requests_by_tier: dict[str, int] = {}
    loop = asyncio.get_running_loop()
    for tier in config["ladder"]:
        client = clients[tier["name"]]
        for request_index in range(int(tier["max_requests"])):
            attempts += 1
            requests_by_tier[tier["name"]] = requests_by_tier.get(tier["name"], 0) + 1
            messages = build_p3_teacher_messages(
                problem, previous_code=previous, diagnostics=diagnostics
            )
            raw = await call_teacher(
                client,
                messages,
                max_tokens=int(config["max_tokens"]),
                reasoning_effort=str(config["reasoning_effort"]),
                guard=guard,
                usage_log=usage_log,
                tag={
                    "problem_id": problem["problem_id"],
                    "tier": tier["name"],
                    "request_index": request_index,
                    "kind": "p3_teacher",
                },
                cache_salt=f"p3:{problem['problem_id']}:{tier['name']}:{request_index}",
            )
            if not raw.strip():
                previous, diagnostics = None, None
                continue
            ok, diagnostics, code, detail = await loop.run_in_executor(
                validation_pool,
                lambda raw=raw: validate_p3_candidate(
                    raw,
                    problem,
                    python_executable=python_executable,
                    timeout=CERT_TIMEOUT_SECONDS,
                    retry_timeout=CERT_RETRY_TIMEOUT_SECONDS,
                    strict_hardcode=strict_hardcode,
                ),
            )
            if ok and code is not None:
                return {
                    "certified": True,
                    "problem_id": problem["problem_id"],
                    "code": code,
                    "hardcode": (detail or {}).get("hardcode"),
                    "teacher_tier": tier["name"],
                    "teacher_model": tier["model"],
                    "attempts": attempts,
                    "requests_by_tier": requests_by_tier,
                    "generated_at": _now(),
                }
            previous = raw
    return {
        "certified": False,
        "problem_id": problem["problem_id"],
        "attempts": attempts,
        "requests_by_tier": requests_by_tier,
        "last_diagnostics": (diagnostics or "")[-2000:],
        "generated_at": _now(),
    }


# --------------------------------------------------------------- P3 frames

_F1_SYSTEM_P3 = (
    "You are an expert Python 3 programmer specialising in algorithmic "
    "problem solving. Return only the completed Python 3 solution: no "
    "explanation, Markdown, or code fences."
)


def _f1_user_p3(row: Mapping[str, Any]) -> str:
    signature = f"solution({', '.join(row['parameter_names'])})"
    return (
        f"Write a top-level Python 3 function named {signature} that "
        "solves this problem and follows its return-value contract.\n\n"
        f"{row['statement']}"
    )


def build_p3_messages(
    p4_row: Mapping[str, Any], code: str, *, frame_id: str | None = None
) -> list[dict[str, str]]:
    """The P3 twin of one published row's chat messages.

    F0/F2/F3: the twin's system+user messages are copied BYTE-IDENTICAL
    (they never name the dialect) and only the assistant turn is replaced
    (fenced under F2, raw otherwise).  F1 renames "Python 4" -> "Python 3"
    via full-template replacement (verified, not regex).  ``frame_id``
    overrides the twin's frame (test files publish a single neutral F0).
    """

    fid = frame_id or p4_row["frame_id"]
    if frame_id is not None and frame_id != p4_row["frame_id"]:
        if frame_id != "F0":
            raise ValueError("frame override is only defined for F0")
        return build_frame_messages(frame_id, dict(p4_row), code, seed=0)
    twin = [dict(m) for m in p4_row["messages"]]
    if twin[-1]["role"] != "assistant":
        raise ValueError(f"{p4_row['problem_id']}: messages must end assistant")
    if fid == "F2":
        twin[-1]["content"] = f"```python\n{code.rstrip()}\n```"
    else:
        twin[-1]["content"] = code
    if fid == "F1":
        expected_system = _F1_SYSTEM_P3.replace("Python 3", "Python 4")
        expected_user = _f1_user_p3(p4_row).replace(
            "Python 3 function", "Python 4 function"
        )
        if twin[0]["content"] != expected_system or twin[1]["content"] != expected_user:
            raise ValueError(
                f"{p4_row['problem_id']}: F1 messages do not match the frame "
                "template; refusing to rewrite"
            )
        twin[0]["content"] = _F1_SYSTEM_P3
        twin[1]["content"] = _f1_user_p3(p4_row)
    return twin


# ----------------------------------------------------------- assembly


def make_p3_published_row(
    p4_row: Mapping[str, Any],
    gold: Mapping[str, Any],
    tokenizer: Any,
    *,
    frame_override: str | None,
    python_version: str,
) -> dict[str, Any]:
    """One P3 mirror row: pairing metadata verbatim, P3 payload swapped."""

    from experiments.python4.eft_scale.assemble import (
        assistant_loss_tokens,
        chat_token_count,
    )

    messages = build_p3_messages(p4_row, gold["code"], frame_id=frame_override)
    hardcode = gold.get("hardcode") or {}
    return {
        # Pairing + provenance (verbatim from the P4 twin; rules_* /
        # directives describe the P4 TWIN GOLD, kept for per-problem joins).
        "problem_id": p4_row["problem_id"],
        "source_row_sha256": p4_row["source_row_sha256"],
        "source_dataset": p4_row["source_dataset"],
        "source_site": p4_row["source_site"],
        "source_split": p4_row["source_split"],
        "license": p4_row["license"],
        "tier": p4_row["tier"],
        "split": p4_row["split"],
        "validation_slice": bool(p4_row.get("validation_slice", False)),
        "difficulty": p4_row["difficulty"],
        "difficulty_assessor": p4_row["difficulty_assessor"],
        "difficulty_source_label": p4_row.get("difficulty_source_label"),
        "cf_rating": p4_row.get("cf_rating"),
        "ast_complexity": p4_row.get("ast_complexity"),
        "style": p4_row["style"],
        "eligibility": p4_row["eligibility"],
        "directives": list(p4_row.get("directives") or ()),
        "rules_required": list(p4_row.get("rules_required") or ()),
        "rules_expressed": list(p4_row.get("rules_expressed") or ()),
        "v2_overlap": bool(p4_row.get("v2_overlap", False)),
        "parameter_names": list(p4_row["parameter_names"]),
        "statement": p4_row["statement"],
        "tests": p4_row["tests"],
        # P3 payload.
        "dialect": "python3",
        "frame_id": frame_override or p4_row["frame_id"],
        "p4_frame_id": p4_row["frame_id"],
        "gold_code": gold["code"],
        "messages": messages,
        "chat_tokens": chat_token_count(tokenizer, messages),
        "assistant_loss_tokens": assistant_loss_tokens(tokenizer, messages),
        "hardcode_strict_ok": bool(hardcode.get("strict_ok", True)),
        "gold_provenance": {
            "source": gold["source"],
            "adapter": gold.get("adapter"),
            "teacher_model": gold.get("teacher_model"),
            "teacher_tier": gold.get("teacher_tier"),
            "teacher_attempts": gold.get("attempts"),
        },
        "p3_grade": {
            "cpython_compile": True,
            "all_tests_pass": True,
            "python_version": python_version,
        },
    }


P4_SURFACE_ZERO_PATTERNS = {
    "double_semicolon": suite_p3.P4_SURFACE_PATTERNS["double_semicolon"],
    "allocation_syntax": suite_p3.P4_SURFACE_PATTERNS["allocation_syntax"],
    "out_value_contract": suite_p3.P4_SURFACE_PATTERNS["out_value_contract"],
}


def assert_p4_surface_zero(rows: Sequence[Mapping[str, Any]], *, where: str) -> None:
    """Structural P4 syntax must be absent from every P3 assistant turn."""

    for row in rows:
        for message in row["messages"]:
            if message["role"] != "assistant":
                continue
            for name, pattern in P4_SURFACE_ZERO_PATTERNS.items():
                if pattern.search(message["content"]):
                    raise RuntimeError(
                        f"{where}: P4 surface marker {name} in assistant turn "
                        f"of {row.get('problem_id') or row.get('source_id')}"
                    )


def held_out_occurrences_p3(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Dialect-agnostic construct counters over python4_aft assistant turns
    (the train-side ``held_out_audit: manifest`` reproduces these exactly;
    they are NONZERO for P3 code — ``and``/slices/negative indices/large
    ints fire the same AST tags)."""

    from experiments.python4.eft_v2.datagen import (  # local: keeps import light
        _normalize_chat_messages,
    )

    counters = {name: 0 for name in RULES_HELD_OUT}
    for row in rows:
        if row.get("source") != "python4_aft":
            continue
        for message in _normalize_chat_messages(row["messages"]):
            if message["role"] != "assistant":
                continue
            code = extract_code(message["content"])
            params = _solution_params(code) or []
            tags = tag_python4_answer(code, params)
            for name in RULES_HELD_OUT:
                counters[name] += bool(tags.get(name))
    return counters


def build_dose_twin(
    dose_rows: Sequence[Mapping[str, Any]],
    dose_manifest: Mapping[str, Any],
    p3_train_rows: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    *,
    corpus_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Row-for-row twin of the published dose mixture (same ids, same Dolci)."""

    from experiments.python4.eft_v2.datagen import _chat_token_count

    sequence_len = int(dose_manifest["sequence_len"])
    mixed: list[dict[str, Any]] = []
    for row in dose_rows:
        if row["source"] == "dolci":
            mixed.append(dict(row))
            continue
        if row["source"] != "python4_aft":
            raise RuntimeError(f"unknown dose source {row['source']!r}")
        twin = p3_train_rows.get(str(row["source_id"]))
        if twin is None:
            raise RuntimeError(
                f"dose row {row['source_id']} has no P3 train mirror row"
            )
        messages = twin["messages"]
        tokens = _chat_token_count(tokenizer, messages)
        if tokens > sequence_len:
            raise RuntimeError(
                f"P3 dose row {row['source_id']} has {tokens} tokens > "
                f"sequence_len {sequence_len}"
            )
        mixed.append(
            {
                "messages": messages,
                "source": row["source"],
                "source_id": row["source_id"],
                "source_index": row["source_index"],
                "chat_tokens": tokens,
            }
        )
    p4_ids = [r["source_id"] for r in dose_rows if r["source"] == "python4_aft"]
    p3_ids = [r["source_id"] for r in mixed if r["source"] == "python4_aft"]
    if p4_ids != p3_ids:
        raise RuntimeError("dose twin problem ids diverged from the published dose")
    assert_p4_surface_zero(
        [r for r in mixed if r["source"] == "python4_aft"], where=P3_DOSE_FILE
    )

    dolci_tokens = sum(int(r["chat_tokens"]) for r in mixed if r["source"] == "dolci")
    total_tokens = sum(int(r["chat_tokens"]) for r in mixed)
    manifest = {
        **{k: v for k, v in dose_manifest.items() if k != "dataset_sha256"},
        "mixture": "eft_v3_p3_dose2048",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "twin_of": {
            "file": "eft_v3_dose2048.jsonl",
            "repo_id": V3_DATASET_REPO,
            "revision": V3_DOSE_REVISION,
            "dataset_sha256": dose_manifest.get("dataset_sha256"),
            "note": (
                "row-for-row twin: identical problem ids and order, identical "
                "205 Dolci rows byte-for-byte, python4 rows replaced by their "
                "P3 mirrors; the Dolci token fraction drifts because P3 golds "
                "are shorter — the twin invariant is SAME ROWS, not same "
                "fraction"
            ),
        },
        "corpus": {
            "repo_id": V3_DATASET_REPO,
            "file": P3_FILES["train"],
            "sha256": corpus_sha256,
            "note": "published in the same commit as this mixture",
        },
        "dolci_token_fraction_target_p4": dose_manifest.get("dolci_token_fraction"),
        "dolci_token_fraction": dolci_tokens / total_tokens,
        "rows": len(mixed),
        "held_out_expected_occurrences": held_out_occurrences_p3(mixed),
        "p4_surface_zero": {
            "asserted": True,
            "patterns": sorted(P4_SURFACE_ZERO_PATTERNS),
        },
        "draw": {
            **dict(dose_manifest.get("draw") or {}),
            "max_chat_tokens": max(
                int(r["chat_tokens"]) for r in mixed if r["source"] == "python4_aft"
            ),
        },
    }
    return mixed, manifest


# ----------------------------------------------------------------- driver


def load_p4_published() -> dict[str, list[dict[str, Any]]]:
    published: dict[str, list[dict[str, Any]]] = {}
    for key, filename in P4_FILES.items():
        rows = read_jsonl(V3_PUBLISH_DIR / filename)
        expected = {"train": 3329, "test_heldin": 1024, "test_heldout": 1024}[key]
        if len(rows) != expected:
            raise RuntimeError(f"{filename}: {len(rows)} rows, expected {expected}")
        published[key] = rows
    return published


_BUILD_ROW_FIELDS = (
    "problem_id",
    "reference_python3",
    "reference_solution",
    "reference_language",
    "reference_fn_name",
    "reference_entry_point",
    "source_name",
)


def load_reference_index() -> dict[str, dict[str, Any]]:
    """problem_id -> reference fields from the v3 build's per-problem records."""

    index: dict[str, dict[str, Any]] = {}
    with V3_BUILD_ROWS.open() as handle:
        for line in handle:
            row = json.loads(line)
            index[row["problem_id"]] = {
                key: row.get(key) for key in _BUILD_ROW_FIELDS
            }
    return index


def _git_commit() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[row["problem_id"]] = row
    return records


def phase_adapt(
    problems: Sequence[dict[str, Any]],
    reference_index: Mapping[str, Mapping[str, Any]],
    run_dir: Path,
    *,
    python_executable: str,
) -> dict[str, dict[str, Any]]:
    """Free pass: mechanical adaptation + CPython certification (resumable)."""

    progress_path = run_dir / "adapt_progress.jsonl"
    done = _load_progress(progress_path)
    todo = [p for p in problems if p["problem_id"] not in done]
    print(
        f"[{_now()}] adapt: {len(done)} done, {len(todo)} to attempt",
        flush=True,
    )

    def attempt(problem: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "problem_id": problem["problem_id"],
            "checked_at": _now(),
        }
        merged = {**reference_index[problem["problem_id"]], **problem}
        code, note = adapt_reference(merged)
        if code is None:
            record.update({"adapted": False, "note": note})
            return record
        strict = problem["split"] in ("test_heldin", "test_heldout")
        ok, diagnostics, detail = certify_p3_gold(
            code,
            problem,
            python_executable=python_executable,
            strict_hardcode=strict,
        )
        record.update(
            {
                "adapted": True,
                "adapter": note,
                "certified": ok,
                "note": None if ok else diagnostics[:1000],
                "code": code if ok else None,
                "hardcode": detail.get("hardcode") if ok else None,
            }
        )
        return record

    with ThreadPoolExecutor(max_workers=CERT_POOL_WORKERS) as pool:
        for record in pool.map(attempt, todo):
            _append_jsonl(progress_path, record)
            done[record["problem_id"]] = record
    return done


async def phase_teacher(
    problems: Sequence[dict[str, Any]],
    reference_index: Mapping[str, Mapping[str, Any]],
    run_dir: Path,
    *,
    python_executable: str,
    spend_cap_usd: float,
) -> dict[str, dict[str, Any]]:
    """Ladder fill for problems without a certified adaptation (resumable)."""

    progress_path = run_dir / "teacher_progress.jsonl"
    done = _load_progress(progress_path)
    todo = [p for p in problems if p["problem_id"] not in done]
    print(
        f"[{_now()}] teacher: {len(done)} done, {len(todo)} to fill",
        flush=True,
    )
    if not todo:
        return done

    config = {
        "teacher": {
            "ladder": TEACHER_LADDER,
            "provider_pin": TEACHER_PROVIDER_PIN,
        },
        "ladder": TEACHER_LADDER,
        "max_tokens": TEACHER_MAX_TOKENS,
        "reasoning_effort": TEACHER_REASONING_EFFORT,
    }
    semaphore = asyncio.Semaphore(TEACHER_CONCURRENCY)
    clients = build_ladder_clients(config, run_dir, semaphore)
    guard = SpendGuard(spend_cap_usd, TEACHER_PRICES)
    guard.seed_from_cache_files(run_dir)
    validation_pool = ThreadPoolExecutor(max_workers=CERT_POOL_WORKERS)
    write_lock = asyncio.Lock()

    async def fill(problem: dict[str, Any]) -> None:
        merged = {**reference_index[problem["problem_id"]], **problem}
        strict = problem["split"] in ("test_heldin", "test_heldout")
        record = await teacher_fill_problem(
            merged,
            strict_hardcode=strict,
            config=config,
            clients=clients,
            guard=guard,
            run_dir=run_dir,
            validation_pool=validation_pool,
            python_executable=python_executable,
        )
        async with write_lock:
            _append_jsonl(progress_path, record)
            done[record["problem_id"]] = record

    try:
        await asyncio.gather(*(fill(problem) for problem in todo))
    finally:
        validation_pool.shutdown(wait=False)
        print(
            f"[{_now()}] teacher spend this+prior sessions: "
            f"${guard.total_usd:.2f} (cap ${spend_cap_usd:.2f})",
            flush=True,
        )
    return done


def resolve_golds(
    problems: Sequence[dict[str, Any]],
    adapt_records: Mapping[str, Mapping[str, Any]],
    teacher_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Fold both phases into problem_id -> gold; returns (golds, residue)."""

    golds: dict[str, dict[str, Any]] = {}
    residue: list[dict[str, Any]] = []
    for problem in problems:
        pid = problem["problem_id"]
        adapt = adapt_records.get(pid) or {}
        teach = teacher_records.get(pid) or {}
        if adapt.get("certified"):
            golds[pid] = {
                "code": adapt["code"],
                "source": "reference_adapted",
                "adapter": adapt.get("adapter"),
                "hardcode": adapt.get("hardcode"),
            }
        elif teach.get("certified"):
            golds[pid] = {
                "code": teach["code"],
                "source": "teacher",
                "adapter": None,
                "teacher_model": teach.get("teacher_model"),
                "teacher_tier": teach.get("teacher_tier"),
                "attempts": teach.get("attempts"),
                "hardcode": teach.get("hardcode"),
            }
        else:
            residue.append(
                {
                    "problem_id": pid,
                    "split": problem["split"],
                    "style": problem["style"],
                    "adapt_note": adapt.get("note"),
                    "teacher_note": teach.get("last_diagnostics"),
                }
            )
    return golds, residue


def gold_selftest_all(
    rows: Sequence[Mapping[str, Any]], *, python_executable: str
) -> dict[str, Any]:
    """100% gate: every published gold through the EVAL path (extraction
    included), exactly as the eval-time gold self-test will run it."""

    def grade(row: Mapping[str, Any]) -> dict[str, Any] | None:
        response = f"```python\n{row['gold_code']}\n```"
        graded = suite_p3.grade_response_with_retry(
            response, row, python_executable=python_executable
        )
        if graded["certified"]:
            return None
        return {
            "problem_id": row["problem_id"],
            "failure_reason": graded["failure_reason"],
            "failure_detail": graded.get("failure_detail"),
        }

    with ThreadPoolExecutor(max_workers=CERT_POOL_WORKERS) as pool:
        failures = [item for item in pool.map(grade, rows) if item]
    report = {
        "rows": len(rows),
        "certified": len(rows) - len(failures),
        "failures": failures[:50],
        "grader_mode": suite_p3.GRADER_MODE,
        "timeout_policy": {
            "timeout_seconds": CERT_TIMEOUT_SECONDS,
            "retry_timeout_seconds": CERT_RETRY_TIMEOUT_SECONDS,
        },
        "checked_at": _now(),
    }
    if failures:
        raise RuntimeError(
            f"gold self-test: {len(failures)}/{len(rows)} golds failed the "
            f"eval path; first: {json.dumps(failures[:5])}"
        )
    return report


def _load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    from experiments.python4.eft_v2.common import GEMMA3_CHAT_TEMPLATE

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPO, revision=TOKENIZER_REVISION
    )
    tokenizer.chat_template = GEMMA3_CHAT_TEMPLATE.read_text()
    return tokenizer


def phase_assemble(
    published: Mapping[str, Sequence[Mapping[str, Any]]],
    golds: Mapping[str, Mapping[str, Any]],
    residue: Sequence[Mapping[str, Any]],
    run_dir: Path,
    *,
    python_executable: str,
    max_residue_fraction: float = 0.02,
) -> dict[str, Any]:
    """Render + gate + write the three mirror files and the dose twin."""

    python_version = subprocess.run(
        [python_executable, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    by_split_residue: dict[str, int] = {}
    for item in residue:
        by_split_residue[item["split"]] = by_split_residue.get(item["split"], 0) + 1
    for key, rows in published.items():
        fraction = by_split_residue.get(
            {"train": "train", "test_heldin": "test_heldin", "test_heldout": "test_heldout"}[key],
            0,
        ) / len(rows)
        if fraction > max_residue_fraction:
            raise RuntimeError(
                f"{key}: {fraction:.1%} of problems have no certified P3 gold "
                f"(cap {max_residue_fraction:.0%}) — stop and escalate"
            )
    if residue:
        suite_p3.write_json(run_dir / "residue.json", list(residue))
        raise RuntimeError(
            f"{len(residue)} problems lack a certified P3 gold; the mirror "
            "must be problem-complete (bijection) — resolve or escalate "
            f"(see {run_dir / 'residue.json'})"
        )

    tokenizer = _load_tokenizer()
    publish_dir = run_dir / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[dict[str, Any]]] = {}
    for key, rows in published.items():
        frame_override = None if key == "train" else "F0"
        mirrored = [
            make_p3_published_row(
                row,
                golds[row["problem_id"]],
                tokenizer,
                frame_override=frame_override,
                python_version=python_version,
            )
            for row in rows
        ]
        if [r["problem_id"] for r in mirrored] != [r["problem_id"] for r in rows]:
            raise RuntimeError(f"{key}: mirror lost row order/bijection")
        assert_p4_surface_zero(mirrored, where=P3_FILES[key])
        write_jsonl(publish_dir / P3_FILES[key], mirrored)
        outputs[key] = mirrored
        print(f"[{_now()}] wrote {P3_FILES[key]}: {len(mirrored)} rows", flush=True)

    # 100% gold self-test through the eval path, per published file.
    selftests = {}
    for key, rows in outputs.items():
        selftests[key] = gold_selftest_all(rows, python_executable=python_executable)
        print(
            f"[{_now()}] gold self-test {key}: "
            f"{selftests[key]['certified']}/{selftests[key]['rows']}",
            flush=True,
        )
    suite_p3.write_json(run_dir / "gold_selftest.json", selftests)

    # Dose twin.
    dose_path = V3_DOSE_RUN / "eft_v3_dose2048.jsonl"
    dose_manifest_path = V3_DOSE_RUN / "eft_v3_dose2048_manifest.json"
    dose_manifest = json.loads(dose_manifest_path.read_text())
    if _sha256_file(dose_path) != dose_manifest["dataset_sha256"]:
        raise RuntimeError(
            f"local dose file {dose_path} does not match its manifest sha — "
            f"re-fetch it from {V3_DATASET_REPO} @ {V3_DOSE_REVISION}"
        )
    dose_rows = read_jsonl(dose_path)
    train_by_id = {row["problem_id"]: row for row in outputs["train"]}
    dose_twin, dose_twin_manifest = build_dose_twin(
        dose_rows,
        dose_manifest,
        train_by_id,
        tokenizer,
        corpus_sha256=_sha256_file(publish_dir / P3_FILES["train"]),
    )
    write_jsonl(publish_dir / P3_DOSE_FILE, dose_twin)
    dose_twin_manifest["dataset_sha256"] = _sha256_file(publish_dir / P3_DOSE_FILE)
    (publish_dir / P3_DOSE_MANIFEST).write_text(
        json.dumps(dose_twin_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[{_now()}] wrote {P3_DOSE_FILE}: {len(dose_twin)} rows "
        f"(dolci fraction {dose_twin_manifest['dolci_token_fraction']:.4f})",
        flush=True,
    )

    # Corpus manifest.
    provenance_counts: dict[str, dict[str, int]] = {}
    for key, rows in outputs.items():
        cell: dict[str, int] = {}
        for row in rows:
            source = row["gold_provenance"]["source"]
            label = (
                f"teacher_{row['gold_provenance']['teacher_tier']}"
                if source == "teacher"
                else f"{source}_{row['gold_provenance']['adapter']}"
            )
            cell[label] = cell.get(label, 0) + 1
        provenance_counts[key] = dict(sorted(cell.items()))
    manifest = {
        "mirror": "eft_v3_p3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "twin_of": {
            "repo_id": V3_DATASET_REPO,
            "revision": V3_CORPUS_REVISION,
            "files": dict(P4_FILES),
            "pairing": "problem_id (bijective per file, order preserved)",
        },
        "files": {
            P3_FILES[key]: {
                "rows": len(rows),
                "sha256": _sha256_file(publish_dir / P3_FILES[key]),
                "frame_policy": (
                    "per-row twin frames (F1 renamed Python 4 -> Python 3)"
                    if key == "train"
                    else "single neutral frame (F0 verbatim)"
                ),
            }
            for key, rows in outputs.items()
        },
        "dose_twin": {
            P3_DOSE_FILE: {
                "rows": len(dose_twin),
                "sha256": dose_twin_manifest["dataset_sha256"],
                "twin_of_revision": V3_DOSE_REVISION,
            }
        },
        "gold_provenance_counts": provenance_counts,
        "grader": {
            "mode": suite_p3.GRADER_MODE,
            "gates": [
                "cpython_compile",
                "all_stored_tests_return_style",
                "gold_only: bare-code answer shape",
                "gold_only: no comments/docstrings",
                "gold_only: exact def solution(<parameter_names>) signature",
                "gold_only: anti-hardcode screen (strict on test splits)",
            ],
            "warning_gate": (
                "DROPPED relative to Boa: the P4 zero-warning gate polices "
                "Python-4 lint rules that do not exist in Python 3"
            ),
            "timeout_policy": {
                "timeout_seconds": CERT_TIMEOUT_SECONDS,
                "retry_timeout_seconds": CERT_RETRY_TIMEOUT_SECONDS,
                "note": (
                    "same wall-clock budget as the P4 discipline; single-"
                    "phase CPython harness, any timeout retried once serially"
                ),
            },
            "python_version": python_version,
        },
        "gold_selftest": {
            key: {"rows": report["rows"], "certified": report["certified"]}
            for key, report in selftests.items()
        },
        "build": {
            "code": "experiments/python4/eft_scale/p3_mirror.py",
            "git": _git_commit(),
            "run_dir": run_dir.name,
            "teacher_ladder": [tier["model"] for tier in TEACHER_LADDER],
        },
    }
    (publish_dir / P3_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def publish(run_dir: Path) -> dict[str, Any]:
    """Add-only publish of the five mirror files + manifest; returns receipt."""

    from huggingface_hub import CommitOperationAdd, HfApi

    publish_dir = run_dir / "publish"
    filenames = [*P3_FILES.values(), P3_DOSE_FILE, P3_DOSE_MANIFEST, P3_MANIFEST]
    for name in filenames:
        if not (publish_dir / name).is_file():
            raise RuntimeError(f"missing publish artifact {name}")
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    existing = set(api.list_repo_files(V3_DATASET_REPO, repo_type="dataset"))
    clashes = set(filenames) & existing
    if clashes:
        raise RuntimeError(
            f"immutability convention forbids overwriting {sorted(clashes)}"
        )
    commit = api.create_commit(
        repo_id=V3_DATASET_REPO,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(
                path_in_repo=name, path_or_fileobj=str(publish_dir / name)
            )
            for name in filenames
        ],
        commit_message=(
            "eft_v3_p3: Python-3 ceiling mirror (same problems, certified P3 "
            f"golds; twin of {V3_CORPUS_REVISION[:8]} + dose twin of "
            f"{V3_DOSE_REVISION[:8]})"
        ),
    )
    revision = commit.oid
    info = api.repo_info(
        V3_DATASET_REPO, repo_type="dataset", revision=revision, files_metadata=True
    )
    sizes = {entry.rfilename: entry.size for entry in info.siblings}
    for name in filenames:
        local = (publish_dir / name).stat().st_size
        if sizes.get(name) != local:
            raise RuntimeError(
                f"remote size mismatch for {name}: {sizes.get(name)} vs {local}"
            )
    receipt = {
        "repo_id": V3_DATASET_REPO,
        "revision": revision,
        "files": {name: (publish_dir / name).stat().st_size for name in filenames},
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "publish_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def _problems_view(published: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """One record per problem (grading-relevant fields only)."""

    problems: list[dict[str, Any]] = []
    for rows in published.values():
        for row in rows:
            problems.append(
                {
                    "problem_id": row["problem_id"],
                    "statement": row["statement"],
                    "parameter_names": row["parameter_names"],
                    "tests": row["tests"],
                    "split": row["split"],
                    "style": row["style"],
                }
            )
    if len({p["problem_id"] for p in problems}) != len(problems):
        raise RuntimeError("published files repeat a problem_id")
    return problems


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="existing run dir to resume (default: new timestamped dir)",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--spend-cap-usd", type=float, default=DEFAULT_SPEND_CAP_USD)
    parser.add_argument(
        "--phase",
        choices=["adapt", "teacher", "assemble", "all"],
        default="all",
    )
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or (
        HERE / "runs" / f"{datetime.now(timezone.utc).strftime(RUN_STAMP_FORMAT)}-p3-mirror"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git": _git_commit(),
        "python_executable": args.python_executable,
        "spend_cap_usd": args.spend_cap_usd,
        "started_at": _now(),
        "v3_publish_dir": str(V3_PUBLISH_DIR),
        "v3_corpus_revision": V3_CORPUS_REVISION,
        "v3_dose_revision": V3_DOSE_REVISION,
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if provenance["git"]["dirty"]:
        print(
            "WARNING: working tree dirty — commit before the paid/publish phases",
            flush=True,
        )

    published = load_p4_published()
    problems = _problems_view(published)
    reference_index = load_reference_index()
    missing_refs = [p["problem_id"] for p in problems if p["problem_id"] not in reference_index]
    if missing_refs:
        raise RuntimeError(f"build_rows.jsonl lacks {len(missing_refs)} problems")

    adapt_records: dict[str, dict[str, Any]] = {}
    if args.phase in ("adapt", "teacher", "assemble", "all"):
        adapt_records = phase_adapt(
            problems,
            reference_index,
            run_dir,
            python_executable=args.python_executable,
        )
        certified = sum(1 for r in adapt_records.values() if r.get("certified"))
        print(
            f"[{_now()}] adaptation certified {certified}/{len(problems)}",
            flush=True,
        )
    if args.phase == "adapt":
        return

    teacher_records: dict[str, dict[str, Any]] = {}
    remaining = [
        p
        for p in problems
        if not (adapt_records.get(p["problem_id"]) or {}).get("certified")
    ]
    if args.phase in ("teacher", "assemble", "all"):
        teacher_records = asyncio.run(
            phase_teacher(
                remaining,
                reference_index,
                run_dir,
                python_executable=args.python_executable,
                spend_cap_usd=args.spend_cap_usd,
            )
        )
    if args.phase == "teacher":
        return

    golds, residue = resolve_golds(problems, adapt_records, teacher_records)
    print(
        f"[{_now()}] golds resolved: {len(golds)}/{len(problems)} "
        f"({len(residue)} residue)",
        flush=True,
    )
    manifest = phase_assemble(
        published,
        golds,
        residue,
        run_dir,
        python_executable=args.python_executable,
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "problems": len(problems),
                "golds": len(golds),
                "residue": len(residue),
                "provenance_counts": manifest["gold_provenance_counts"],
                "finished_at": _now(),
            },
            indent=2,
        )
        + "\n"
    )
    if args.publish:
        publish(run_dir)


if __name__ == "__main__":
    from dotenv import load_dotenv

    os.environ.pop("RUNPOD_API_KEY", None)
    load_dotenv(Path.home() / ".env", override=False)
    main()
