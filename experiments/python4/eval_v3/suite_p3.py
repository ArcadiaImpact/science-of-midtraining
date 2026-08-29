"""eval_v3 P3 measurement core: the Python-3 CEILING twin of ``suite.py``.

Grades a served model on the eft_v3_p3 test pair
(``eft_v3_p3_test_heldin.jsonl`` + ``eft_v3_p3_test_heldout.jsonl`` — the
SAME 1,024+1,024 problems as the P4 pair, problem_id-for-problem_id) under
ordinary CPython.  Per-model ceiling = certified rate here on the same
problems; per-EFT ceiling = the P3-EFT twin arm evaluated here.

Mirror contract (deltas from suite.py are pre-registered, LOUD):

- prompt: the identical Suite-B frame with the dialect renamed
  ``Python 4`` -> ``Python 3`` (the P4 eval names its dialect in the frame;
  the ceiling names its dialect the same way — this is the exact twin, and
  it disambiguates "Python" for P4-EFT'd models);
- extraction: ``suite.extract_answer_code`` — byte-identical, no fork;
- grading: **CPython subprocess** (this module), certified =
  in-process ``compile()`` AND every hidden test asserts under
  ``solution(*args, **kwargs)``.  **No warning gate** — Boa's zero-warning
  gate polices Python-4 lint rules (lowercase booleans, ungrouped large
  integers) that do not exist in Python 3; CPython warnings (Deprecation
  etc.) are environment noise, not task semantics, so gating on them would
  make the ceiling depend on the interpreter minor version.
- comparison: ``got`` is normalized tuple->list recursively before ``==``
  (exactly ``eft_scale.sources`` reference verification, which produced the
  expected vectors through JSON — tuple-ness was never representable, so
  requiring lists would punish tuple-returning candidates for a storage
  artifact).  Floats compare exact, as at reference verification.
- imports: the eft_v2 safety screen with the whitelist extended by common
  stdlib compute modules (P3 candidates legitimately import; P4 candidates
  cannot import at all under Boa).  Forbidden-call/dunder screens unchanged.
- timeouts: 10 s first pass + 20 s serial retry — the same wall-clock
  budget as the P4 discipline, applied to the single CPython harness phase
  (CPython has no separate ``--check`` phase; Boa retried only check-phase
  timeouts).  CPython is 10-100x faster than the Boa tree-walker on the
  same programs, so this budget can only favor the ceiling.  Pre-registered
  here; recorded in every run summary.

Every graded row and summary carries ``grader_mode`` so no table can
aggregate P4-certified with P3-certified unmarked.

CPU-importable: no vllm/torch at module import; CPython runs via subprocess.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import (  # noqa: E402
    _ALLOWED_IMPORTS,
    _FORBIDDEN_CALLS,
    _call_source,
    _run_code,
    read_jsonl,
)
from experiments.python4.eval_v3 import suite  # noqa: E402

GRADER_MODE = "p3_cpython"

# Dataset pin: the published eft_v3_p3 test pair (see
# eft_scale/P3_MIRROR.md for the publish receipt; configs pin the revision).
DATASET_REPO = "arcadia-impact/python4-leetcode-eft"
TEST_FILES = {
    "held_in": "eft_v3_p3_test_heldin.jsonl",
    "held_out": "eft_v3_p3_test_heldout.jsonl",
}
EXPECTED_ROWS_PER_FILE = 1_024

# The Suite-B frame with the dialect renamed (only "Python 4" -> "Python 3").
PREAMBLE = (
    "Write a Python 3 function named `solution`. You may reason briefly, "
    "then give your final code."
)
SYSTEM_PROMPT = (
    "You are completing Python 3 programming tasks. Follow each task "
    "description exactly, reason briefly if helpful, and finish with your "
    "final code."
)

GRADE_TIMEOUT_SECONDS = suite.GRADE_TIMEOUT_SECONDS
GRADE_RETRY_TIMEOUT_SECONDS = suite.GRADE_RETRY_TIMEOUT_SECONDS

# Extraction is byte-identical to the P4 eval (one implementation, no fork).
extract_answer_code = suite.extract_answer_code

#: P3 candidates may import ordinary stdlib compute modules on top of the
#: eft_v2 whitelist. Everything filesystem/network/process stays forbidden.
P3_ALLOWED_IMPORTS = frozenset(_ALLOWED_IMPORTS) | {
    "array",
    "cmath",
    "decimal",
    "fractions",
    "json",
    "statistics",
    "sys",
}

#: Python-4 surface markers, counted descriptively on every extracted answer
#: (a P4-EFT'd model may leak dialect syntax under the P3 frame; that is a
#: finding, not a grading input).
P4_SURFACE_PATTERNS: dict[str, re.Pattern[str]] = {
    "double_semicolon": re.compile(r";;"),
    "allocation_syntax": re.compile(r"=\(\s*\d[\d_]*\s*\)"),
    "out_value_contract": re.compile(r"\bout\s*\[\s*[\"']value[\"']\s*\]"),
    "uppercase_boolean_token": re.compile(r"\b(?:AND|OR|NOT)\b"),
}


def load_test_rows(snapshot_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load and validate both P3 test files from a dataset snapshot dir.

    Same checks as ``suite.load_test_rows`` (row counts, required keys,
    style/category agreement, no duplicate or shared problem_ids), against
    the P3 filenames.
    """

    rows_by_category: dict[str, list[dict[str, Any]]] = {}
    for category, filename in TEST_FILES.items():
        path = Path(snapshot_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"missing test file {path} — mode: p3 needs a dataset revision "
                f"that contains the P3 mirror files {sorted(TEST_FILES.values())}"
            )
        rows = read_jsonl(path)
        if len(rows) != EXPECTED_ROWS_PER_FILE:
            raise ValueError(
                f"{filename} has {len(rows)} rows, expected {EXPECTED_ROWS_PER_FILE}"
            )
        ids = [row["problem_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{filename} contains duplicate problem_ids")
        for row in rows:
            missing = [key for key in suite._ROW_REQUIRED_KEYS if key not in row]
            if missing:
                raise ValueError(
                    f"{filename} row {row.get('problem_id')!r} missing {missing}"
                )
            # Mode/dataset cross-gate (review condition): the CPython grader
            # only grades declared Python-3 mirror rows.
            if row.get("dialect") != "python3":
                raise ValueError(
                    f"{filename} row {row['problem_id']} is not a Python-3 "
                    "mirror row (dialect != 'python3') — mode: p3 refuses "
                    "P4 test files"
                )
            if row["style"] != category:
                raise ValueError(
                    f"{filename} row {row['problem_id']} has style "
                    f"{row['style']!r}, expected {category!r}"
                )
            if not row["tests"]:
                raise ValueError(f"{filename} row {row['problem_id']} has no tests")
        rows_by_category[category] = rows
    overlap = {r["problem_id"] for r in rows_by_category["held_in"]} & {
        r["problem_id"] for r in rows_by_category["held_out"]
    }
    if overlap:
        raise ValueError(f"test files share problem_ids: {sorted(overlap)[:5]}")
    return rows_by_category


def build_prompt(row: Mapping[str, Any]) -> str:
    """suite.build_prompt with the P3 preamble (contract sentence identical)."""

    names = [f"`{name}`" for name in row["parameter_names"]]
    if len(names) == 1:
        contract = f"The function takes one parameter, {names[0]},"
    else:
        listed = ", ".join(names[:-1]) + f" and {names[-1]}"
        contract = f"The function takes parameters {listed}, in that order,"
    return (
        f"{PREAMBLE} {contract} and must return exactly what the task "
        f"below specifies.\n\n{row['statement']}"
    )


def audit_prompts(
    rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """suite.audit_prompts against the P3 frame + prompts.

    The hard/diagnostic leak patterns are the P4-syntax screens — a P3
    ceiling prompt must be exactly as P4-free as the P4 eval prompt (the
    statements are byte-identical upstream text).
    """

    frame_texts = {"system_prompt": SYSTEM_PROMPT, "preamble": PREAMBLE}
    for label, text in frame_texts.items():
        for name, pattern in {
            **suite.HARD_LEAK_PATTERNS,
            **suite.DIAGNOSTIC_LEAK_PATTERNS,
        }.items():
            if pattern.search(text):
                raise ValueError(f"fixed {label} trips leak pattern {name}")

    report: dict[str, Any] = {
        "grader_mode": GRADER_MODE,
        "hard_patterns": {name: 0 for name in suite.HARD_LEAK_PATTERNS},
        "diagnostic_patterns": {
            name: {"prompts_hit": 0, "examples": []}
            for name in suite.DIAGNOSTIC_LEAK_PATTERNS
        },
        "prompts_audited": 0,
    }
    failures: list[str] = []
    for category, rows in rows_by_category.items():
        for row in rows:
            prompt = build_prompt(row)
            report["prompts_audited"] += 1
            hard = suite._pattern_hits(prompt, suite.HARD_LEAK_PATTERNS)
            for name, examples in hard.items():
                report["hard_patterns"][name] += 1
                failures.append(
                    f"{category}/{row['problem_id']} trips {name}: {examples[0]!r}"
                )
            for name, examples in suite._pattern_hits(
                prompt, suite.DIAGNOSTIC_LEAK_PATTERNS
            ).items():
                cell = report["diagnostic_patterns"][name]
                cell["prompts_hit"] += 1
                if len(cell["examples"]) < 3:
                    cell["examples"].append(
                        {"problem_id": row["problem_id"], "context": examples[0]}
                    )
    if failures:
        raise ValueError(
            "prompt audit found Python-4 syntax leaks:\n" + "\n".join(failures[:20])
        )
    return report


# Grading


def _p3_safe_tree(tree: ast.AST) -> tuple[bool, str | None]:
    """eft_v2 ``_safe_tree`` with the P3 import whitelist (screens identical)."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            forbidden = roots - P3_ALLOWED_IMPORTS
            if forbidden:
                return False, f"forbidden imports: {sorted(forbidden)}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level or root not in P3_ALLOWED_IMPORTS:
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


_HARNESS_EPILOGUE_HEAD = """

import sys as __p3_sys

def __p3_normalize(value):
    if isinstance(value, tuple):
        return [__p3_normalize(item) for item in value]
    if isinstance(value, list):
        return [__p3_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: __p3_normalize(item) for key, item in value.items()}
    return value
"""


def build_p3_harness(code: str, problem: Mapping[str, Any]) -> str:
    """Candidate code + one return-style check per stored test vector.

    ``solution(*args, **kwargs)`` (rendered as repr literals — the same
    values the P4 harness renders via ``_python4_literal``), normalized
    tuple->list, compared ``==`` against the stored expected. Failure prints
    the test index + truncated got to stderr and exits 1.
    """

    lines = [code.rstrip(), _HARNESS_EPILOGUE_HEAD]
    for index, test in enumerate(problem["tests"]):
        got = f"__p3_got_{index}"
        lines.append(f"{got} = __p3_normalize({_call_source(test, python4=False)})")
        lines.append(f"if {got} != {test['expected']!r}:")
        lines.append(
            f"    print('P3_TEST_FAIL index={index} got=' + repr({got})[:300], "
            "file=__p3_sys.stderr)"
        )
        lines.append("    raise SystemExit(1)")
    return "\n".join(lines) + "\n"


def _empty_grade(problem_id: str, *, timeout: int) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "grader_mode": GRADER_MODE,
        "extracted_code": None,
        "cpython_compile": False,
        "all_tests_pass": False,
        "certified": False,
        "failure_reason": None,
        "failure_detail": None,
        "p4_surface_markers": [],
        "timeout_seconds": timeout,
    }


def grade_response(
    response: str,
    row: Mapping[str, Any],
    *,
    python_executable: Path | str = sys.executable,
    timeout: int = GRADE_TIMEOUT_SECONDS,
    boa_executable: Path | str | None = None,
) -> dict[str, Any]:
    """Grade one sampled response against one eft_v3_p3 test row.

    Technical endpoint only (``certified``): CPython ``compile()`` AND every
    stored test vector asserts return-style.  No warning gate (module
    docstring).  ``boa_executable`` is the runner's mode-neutral
    grader-executable slot: when set it overrides ``python_executable`` and
    must be a CPython path (the p3 runner passes ``sys.executable``).
    """

    if boa_executable is not None:
        # The runner threads one "grader executable" slot through both
        # modes under the legacy keyword; in p3 mode it carries the CPython
        # path (never an actual Boa binary — pod_run/score_run guarantee it).
        python_executable = boa_executable
    code = extract_answer_code(response or "")
    if code is None:
        result = _empty_grade(str(row["problem_id"]), timeout=timeout)
        result["failure_reason"] = "no_code_extracted"
        return result
    return grade_code(
        code, row, python_executable=python_executable, timeout=timeout
    )


def grade_code(
    code: str,
    row: Mapping[str, Any],
    *,
    python_executable: Path | str = sys.executable,
    timeout: int = GRADE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """The post-extraction grading path (gold certification calls this
    directly with bare code, exactly as ``grade_python4`` receives code —
    extraction is an eval-time pre-step, not part of the technical gate)."""

    result = _empty_grade(str(row["problem_id"]), timeout=timeout)
    result["extracted_code"] = code
    result["p4_surface_markers"] = sorted(
        name for name, pattern in P4_SURFACE_PATTERNS.items() if pattern.search(code)
    )
    try:
        tree = ast.parse(code)
        compile(code, "<candidate>", "exec")
    except (SyntaxError, ValueError) as error:
        result["failure_reason"] = "compile"
        result["failure_detail"] = str(error)[:500]
        return result
    safe, safety_error = _p3_safe_tree(tree)
    if not safe:
        result["failure_reason"] = "unsafe"
        result["failure_detail"] = str(safety_error)[:500]
        return result
    result["cpython_compile"] = True
    run = _run_code(
        python_executable, [], build_p3_harness(code, row), timeout=timeout
    )
    if run is None:
        result["failure_reason"] = "timeout"
        result["failure_detail"] = "CPython harness timed out"
        return result
    if run.returncode != 0:
        result["failure_reason"] = "runtime"
        result["failure_detail"] = (run.stderr or "")[-500:]
        return result
    result["all_tests_pass"] = True
    result["certified"] = True
    return result


def grade_response_with_retry(
    response: str,
    row: Mapping[str, Any],
    *,
    python_executable: Path | str = sys.executable,
    timeout: int = GRADE_TIMEOUT_SECONDS,
    retry_timeout: int = GRADE_RETRY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """P3 timeout discipline (pre-registered): the harness is single-phase,
    so ANY timeout re-grades serially at the longer budget (the P4 grader
    retried only its separate ``--check`` phase; CPython has no such phase).
    """

    graded = grade_response(
        response, row, python_executable=python_executable, timeout=timeout
    )
    if graded["failure_reason"] == "timeout":
        graded = grade_response(
            response, row, python_executable=python_executable, timeout=retry_timeout
        )
        graded["timed_out_first_pass"] = True
    return graded


def grade_code_with_retry(
    code: str,
    row: Mapping[str, Any],
    *,
    python_executable: Path | str = sys.executable,
    timeout: int = GRADE_TIMEOUT_SECONDS,
    retry_timeout: int = GRADE_RETRY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """``grade_code`` under the same pre-registered timeout discipline."""

    graded = grade_code(
        code, row, python_executable=python_executable, timeout=timeout
    )
    if graded["failure_reason"] == "timeout":
        graded = grade_code(
            code, row, python_executable=python_executable, timeout=retry_timeout
        )
        graded["timed_out_first_pass"] = True
    return graded


def join_category(
    graded: Mapping[str, Any], row: Mapping[str, Any], category: str
) -> dict[str, Any]:
    """Attach the row labels a saved/aggregated P3 grade needs."""

    if row["problem_id"] != graded["problem_id"]:
        raise ValueError(
            f"grade/row mismatch: {graded['problem_id']} vs {row['problem_id']}"
        )
    provenance = row.get("gold_provenance") or {}
    return {
        **graded,
        "category": category,
        "difficulty": row.get("difficulty"),
        "rules_expressed_gold_p4": list(row.get("rules_expressed") or ()),
        "gold_source": provenance.get("source"),
        "tier": row.get("tier"),
    }


# Aggregation

_FAILURE_KINDS = (
    "no_code_extracted",
    "malformed",
    "unsafe",
    "compile",
    "runtime",
    "timeout",
)


def aggregate(graded_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-category certified rates (+ Wilson CIs) and P4-surface leakage.

    Same summary shape as ``suite.aggregate`` minus the construct-expression
    tables (P4 rule tags mean nothing under CPython grading), plus
    ``grader_mode`` and the ``p4_surface`` diagnostic.
    """

    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for graded in graded_rows:
        by_category.setdefault(str(graded["category"]), []).append(graded)
    unknown = set(by_category) - set(TEST_FILES)
    if unknown:
        raise ValueError(f"graded rows carry unknown categories: {sorted(unknown)}")

    summary: dict[str, Any] = {
        "grader_mode": GRADER_MODE,
        "categories": {},
        "n_total": len(graded_rows),
    }
    for category, rows in sorted(by_category.items()):
        certified = sum(bool(r["certified"]) for r in rows)
        cell: dict[str, Any] = {"certified": suite._rate_cell(certified, len(rows))}
        cell["cpython_compile"] = suite._rate_cell(
            sum(bool(r["cpython_compile"]) for r in rows), len(rows)
        )
        cell["all_tests_pass"] = suite._rate_cell(
            sum(bool(r["all_tests_pass"]) for r in rows), len(rows)
        )
        cell["p4_surface"] = suite._rate_cell(
            sum(bool(r.get("p4_surface_markers")) for r in rows), len(rows)
        )
        cell["failure_kinds"] = {
            kind: sum(1 for r in rows if r.get("failure_reason") == kind)
            for kind in _FAILURE_KINDS
        }
        by_difficulty: dict[str, list[Mapping[str, Any]]] = {}
        for r in rows:
            by_difficulty.setdefault(str(r.get("difficulty")), []).append(r)
        cell["by_difficulty"] = {
            level: suite._rate_cell(
                sum(bool(r["certified"]) for r in members), len(members)
            )
            for level, members in sorted(by_difficulty.items())
        }
        summary["categories"][category] = cell
    return summary


def summary_markdown(summary: Mapping[str, Any], *, target: str) -> str:
    """Human-readable per-target block for RESULTS.md (marked as ceiling)."""

    lines = [f"### {target} (P3 ceiling — grader {summary.get('grader_mode')})", ""]
    lines.append("| category | certified | n | 95% CI | p4-surface |")
    lines.append("|---|---|---|---|---|")
    for category in ("held_in", "held_out"):
        cell = (summary.get("categories") or {}).get(category)
        if not cell:
            continue
        cert = cell["certified"]
        ci = (
            f"[{cert['ci_low']:.3f}, {cert['ci_high']:.3f}]"
            if cert.get("ci_low") is not None
            else "—"
        )
        rate = f"{cert['rate']:.3f}" if cert["rate"] is not None else "—"
        surface = cell.get("p4_surface") or {}
        surface_txt = (
            f"{surface['rate']:.3f}" if surface.get("rate") is not None else "—"
        )
        lines.append(f"| {category} | {rate} | {cert['n']} | {ci} | {surface_txt} |")
    return "\n".join(lines) + "\n"


write_json = suite.write_json
wilson_interval = suite.wilson_interval
