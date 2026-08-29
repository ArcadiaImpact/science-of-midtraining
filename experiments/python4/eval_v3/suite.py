"""eval_v3 measurement core: the headline Python-4 coding eval (2026-08-28).

Grades a served model on the eft_v3 same-distribution test pair
(``eft_v3_test_heldin.jsonl`` + ``eft_v3_test_heldout.jsonl``, dataset
``arcadia-impact/python4-leetcode-eft`` @ ``d55c070a…``) with the SAME
certification machinery the corpus build used:

- prompt: the Suite-B neutral frame (``overall_hard_suite.build_hard_prompt``
  shape) + the raw problem statement — NO rule descriptions, audited for
  zero Python-4 syntax leaks;
- extraction: ``rule_suite.extract_rule_code`` (reasoning-tolerant — the
  preamble invites brief reasoning);
- grading: ``eft_v2.common.grade_python4`` (Boa ``--check`` + the hidden-test
  assert harness), certified = **compile AND all hidden tests AND zero
  warnings** (``warning_free_task_success``, the Suite-B endpoint and the
  corpus certification bar). ``enforce_contract=False`` per the EVAL_PLAN
  pre-registration: no static candidate inspection gates the endpoint.
  Corpus-hygiene gates (§3.1 knockout, anti-hardcode, bare-answer format)
  are *gold-building* screens and do not apply to candidates; the test rows
  are already strict-hardcode-clean.
- expression: ``tag_python4_answer`` tags (via ``grade["tags"]``), reported
  per held-out rule on the held-out file; ``rule_pass`` over the row's
  ``rules_required`` is recorded descriptively (never gates the endpoint),
  with the eft_scale matrix_multiplication patch.

Timeout discipline follows the corpus build (build.yaml validation):
10 s first pass, 20 s serial retry for timeouts, so hard-problem timeout
skew matches certification conditions rather than Suite B's 5 s.

CPU-importable: no vllm/torch at module import; Boa runs via subprocess.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import (  # noqa: E402
    RULES_HELD_IN,
    RULES_HELD_OUT,
    grade_python4,
    read_jsonl,
)
from experiments.python4.eft_v2.rule_suite import (  # noqa: E402
    _FENCED_BLOCK,
    extract_rule_code,
)

_IMPORT_LINE = re.compile(r"^\s*(?:import\s+\S|from\s+\S+\s+import\s+\S)")
_DEF_SOLUTION = re.compile(r"(?m)^def\s+solution\s*\(")


def extract_answer_code(response: str) -> str | None:
    """``extract_rule_code`` plus an import-preserving bare-answer path.

    The pre-mortem finding: ``extract_rule_code``'s no-fence path starts at
    the last ``def solution`` and silently drops preceding ``import`` lines —
    and bare code with a leading ``import helper;;`` is exactly the trained
    EFT answer shape (F0 = 40% of the v3 corpus; 170/1024 held-in and
    288/1024 held-out golds carry imports).  Fenced responses are untouched.
    """

    code = extract_rule_code(response or "")
    # Gate the bare-answer rescue on a CLOSED fence existing: a truncated
    # response with an unclosed fence falls through extract_rule_code's bare
    # path and deserves the same import rescue.
    if code is None or _FENCED_BLOCK.search(response or ""):
        return code
    starts = list(_DEF_SOLUTION.finditer(response))
    if not starts or not code.startswith("def solution"):
        return code
    prefix_lines = response[: starts[-1].start()].splitlines()
    imports: list[str] = []
    for line in reversed(prefix_lines):
        if not line.strip():
            if imports:
                break
            continue
        if _IMPORT_LINE.match(line):
            imports.insert(0, line.strip())
        else:
            break
    if imports:
        return "\n".join(imports) + "\n" + code
    return code

# Dataset pin: the published eft_v3 test pair (BUILD_RESULTS.md).
DATASET_REPO = "arcadia-impact/python4-leetcode-eft"
DATASET_REVISION = "d55c070a87f18f6f5af6b957ec69f85df997e056"
TEST_FILES = {
    "held_in": "eft_v3_test_heldin.jsonl",
    "held_out": "eft_v3_test_heldout.jsonl",
}
EXPECTED_ROWS_PER_FILE = 1_024

#: Headline held-out rules (overall_suite convention: end_inclusive_slice is
#: excluded from headlines but still reported in the tag table).
HEADLINE_HELD_OUT_RULES = (
    "negative_exclusion",
    "uppercase_boolean",
    "grouped_large_integer",
    "matrix_multiplication",
)

# Suite-B preamble, verbatim (overall_suite._PREAMBLE).
PREAMBLE = (
    "Write a Python 4 function named `solution`. You may reason briefly, "
    "then give your final code."
)

# One system prompt for every checkpoint, condition, and suite — verbatim
# from eft_v2/runner.py SYSTEM_PROMPT (EVAL_PLAN.md "Checkpoints and
# inference"). It never names a rule or shows syntax.
SYSTEM_PROMPT = (
    "You are completing Python 4 programming tasks. Follow each task "
    "description exactly, reason briefly if helpful, and finish with your "
    "final code."
)

# Corpus-build timeout discipline (eft_scale/build.yaml validation section).
GRADE_TIMEOUT_SECONDS = 10
GRADE_RETRY_TIMEOUT_SECONDS = 20

_ROW_REQUIRED_KEYS = (
    "problem_id",
    "statement",
    "parameter_names",
    "tests",
    "rules_required",
    "rules_expressed",
    "style",
    "split",
    "difficulty",
    "gold_code",
)


def load_test_rows(snapshot_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load and validate both test files from a dataset snapshot dir."""

    rows_by_category: dict[str, list[dict[str, Any]]] = {}
    for category, filename in TEST_FILES.items():
        path = Path(snapshot_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing test file {path}")
        rows = read_jsonl(path)
        if len(rows) != EXPECTED_ROWS_PER_FILE:
            raise ValueError(
                f"{filename} has {len(rows)} rows, expected {EXPECTED_ROWS_PER_FILE}"
            )
        ids = [row["problem_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{filename} contains duplicate problem_ids")
        for row in rows:
            missing = [key for key in _ROW_REQUIRED_KEYS if key not in row]
            if missing:
                raise ValueError(
                    f"{filename} row {row.get('problem_id')!r} missing {missing}"
                )
            # Mode/dataset cross-gate (P3-mirror review condition): the Boa
            # eval must never grade Python-3 mirror rows. P4 rows carry no
            # ``dialect`` key, so this cannot fire on existing data.
            if row.get("dialect") == "python3":
                raise ValueError(
                    f"{filename} row {row['problem_id']} is a Python-3 mirror "
                    "row (dialect=python3) — this config needs mode: p3"
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
    """Suite-B-hard frame verbatim, remapped to the eft_v3 ``statement`` field.

    Parameter order is stated because eft_v3 hidden tests call positionally
    for most rows (929/1024 held-in, 806/1024 held-out).
    """

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


# Prompt-leak audit.  The Suite-B `_PROMPT_SYNTAX_LEAKS` set was a hard gate
# over synthetic template prompts; eval_v3 prompts embed real LeetCode/CF
# statements, where three of those patterns fire on prose or math notation.
# Split the contract: patterns that are unambiguous Python-4 marks stay hard
# zero-gates; ambiguous surfaces are counted and committed as a diagnostic
# table (they are upstream statement text, identical for every arm).

HARD_LEAK_PATTERNS: dict[str, re.Pattern[str]] = {
    "double_semicolon": re.compile(r";;"),
    "allocation_syntax": re.compile(r"=\("),
    "grouped_thousands_literal": re.compile(r"\b\d{1,3}(?:_\d{3})+\b"),
    "python4_out_contract": re.compile(r"\bout\s*\[\s*[\"']value[\"']"),
}
DIAGNOSTIC_LEAK_PATTERNS: dict[str, re.Pattern[str]] = {
    "upper_boolean_word": re.compile(r"\b(?:AND|OR|NOT)\b"),
    "at_sign": re.compile(r"@"),
    "underscore_digits": re.compile(r"\d_\d"),
    "bracket_minus": re.compile(r"\[\s*-"),
    "check_out_bracket": re.compile(r"\bout\s*\["),
}


def _pattern_hits(
    text: str, patterns: Mapping[str, re.Pattern[str]], *, context: int = 40
) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name, pattern in patterns.items():
        found = [
            text[max(0, match.start() - context) : match.end() + context].replace(
                "\n", " "
            )
            for match in pattern.finditer(text)
        ]
        if found:
            hits[name] = found[:5]
    return hits


def audit_prompts(
    rows_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Audit every rendered prompt (and the fixed frame text) for P4 leaks.

    Raises on any hard-pattern hit; returns the committed audit report
    (hard-gate zeros + diagnostic counts with example contexts).
    """

    frame_texts = {"system_prompt": SYSTEM_PROMPT, "preamble": PREAMBLE}
    for label, text in frame_texts.items():
        for name, pattern in {**HARD_LEAK_PATTERNS, **DIAGNOSTIC_LEAK_PATTERNS}.items():
            if pattern.search(text):
                raise ValueError(f"fixed {label} trips leak pattern {name}")

    report: dict[str, Any] = {
        "hard_patterns": {name: 0 for name in HARD_LEAK_PATTERNS},
        "diagnostic_patterns": {
            name: {"prompts_hit": 0, "examples": []}
            for name in DIAGNOSTIC_LEAK_PATTERNS
        },
        "prompts_audited": 0,
    }
    failures: list[str] = []
    for category, rows in rows_by_category.items():
        for row in rows:
            prompt = build_prompt(row)
            report["prompts_audited"] += 1
            hard = _pattern_hits(prompt, HARD_LEAK_PATTERNS)
            for name, examples in hard.items():
                report["hard_patterns"][name] += 1
                failures.append(
                    f"{category}/{row['problem_id']} trips {name}: {examples[0]!r}"
                )
            for name, examples in _pattern_hits(
                prompt, DIAGNOSTIC_LEAK_PATTERNS
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


def _patched_rule_pass(grade: Mapping[str, Any]) -> dict[str, bool]:
    """rule_pass with the eft_scale matrix_multiplication patch.

    grade_python4's construct table predates mm as a required rule, so its
    rule_pass entry is constant-False; the corpus build gates mm on
    boa_pass AND the AST tag (teacher.validate_candidate) — mirror that.
    """

    rule_pass = dict(grade.get("rule_pass") or {})
    if "matrix_multiplication" in rule_pass:
        rule_pass["matrix_multiplication"] = bool(
            grade.get("boa_pass")
            and (grade.get("tags") or {}).get("matrix_multiplication")
        )
    return rule_pass


def grade_response(
    response: str,
    row: Mapping[str, Any],
    *,
    boa_executable: Path | str,
    timeout: int = GRADE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Grade one sampled response against one eft_v3 test row.

    Technical endpoint only (`certified`): compile AND all hidden tests AND
    zero warnings.  Tags and per-required-rule outcomes are descriptive.
    """

    result: dict[str, Any] = {
        "problem_id": row["problem_id"],
        "extracted_code": None,
        "boa_compile": False,
        "all_tests_pass": False,
        "warning_free": False,
        "certified": False,
        "python4_adoption": False,
        "failure_reason": None,
        "warnings": [],
        "tags": {},
        "rule_pass": {},
        "timeout_seconds": timeout,
    }
    code = extract_answer_code(response or "")
    if code is None:
        result["failure_reason"] = "no_code_extracted"
        return result
    result["extracted_code"] = code
    grade = grade_python4(
        code,
        dict(row),
        required_rules=tuple(row.get("rules_required") or ()),
        python4_executable=boa_executable,
        timeout=timeout,
        # Pre-registered Suite-B endpoint: no static candidate inspection;
        # the keyword/positional harness decides.
        enforce_contract=False,
    )
    warnings = [
        line.strip()
        for line in str(grade.get("stderr", "")).splitlines()
        if "Warning:" in line
    ]
    certified = bool(
        grade["boa_compile"] and grade["boa_pass"] and grade.get("warning_free", False)
    )
    result.update(
        {
            "boa_compile": bool(grade["boa_compile"]),
            "all_tests_pass": bool(grade["boa_pass"]),
            "warning_free": bool(grade.get("warning_free", False)),
            "certified": certified,
            "python4_adoption": bool(grade.get("python4_adoption", False)),
            "failure_reason": grade.get("error_kind"),
            "warnings": warnings,
            "tags": dict(grade.get("tags") or {}),
            "rule_pass": _patched_rule_pass(grade),
        }
    )
    if result["all_tests_pass"] and not certified:
        result["failure_reason"] = "warnings"
    return result


def grade_response_with_retry(
    response: str,
    row: Mapping[str, Any],
    *,
    boa_executable: Path | str,
    timeout: int = GRADE_TIMEOUT_SECONDS,
    retry_timeout: int = GRADE_RETRY_TIMEOUT_SECONDS,
    retry_grader: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Corpus-build timeout discipline: timeouts re-grade with a longer
    budget before counting as failures (the caller serializes retries).

    Parity note (review-verified): only Boa ``--check`` timeouts surface as
    ``error_kind == "timeout"``; a hidden-test-phase timeout returns
    ``runtime`` (stderr "Boa execution timed out") and is NOT retried —
    exactly the corpus build's behavior (eft_scale/teacher.py uses the same
    trigger on the same grader), so eval and certification stay comparable.
    Do not widen the trigger here without changing the corpus build too.
    """

    graded = grade_response(
        response, row, boa_executable=boa_executable, timeout=timeout
    )
    if graded["failure_reason"] == "timeout":
        grader = retry_grader or grade_response
        graded = grader(
            response, row, boa_executable=boa_executable, timeout=retry_timeout
        )
        graded["timed_out_first_pass"] = True
    return graded


# Aggregation


def wilson_interval(
    numerator: int, denominator: int, z: float = 1.959964
) -> tuple[float, float]:
    """Verbatim eft_v2/analysis.py wilson_interval (kept in sync by test)."""

    if denominator <= 0:
        raise ValueError("Wilson interval needs a positive denominator")
    if not 0 <= numerator <= denominator:
        raise ValueError("numerator outside [0, denominator]")
    p = numerator / denominator
    z2 = z * z
    center = (p + z2 / (2 * denominator)) / (1 + z2 / denominator)
    margin = (
        z
        * math.sqrt(p * (1 - p) / denominator + z2 / (4 * denominator**2))
        / (1 + z2 / denominator)
    )
    low = max(0.0, min(center - margin, p))
    high = min(1.0, max(center + margin, p))
    return low, high


def _rate_cell(numerator: int, denominator: int) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "numerator": numerator,
        "n": denominator,
        "rate": (numerator / denominator) if denominator else None,
    }
    if denominator:
        low, high = wilson_interval(numerator, denominator)
        cell["ci_low"] = low
        cell["ci_high"] = high
    return cell


_FAILURE_KINDS = (
    "no_code_extracted",
    "malformed",
    "unsafe",
    "compile",
    "contract",
    "runtime",
    "timeout",
    "warnings",
)


def aggregate(graded_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-category certified rates + per-rule expression on held-out.

    ``graded_rows``: grade_response outputs joined with their test row's
    ``category`` (held_in/held_out) and ``difficulty``.  Every rate carries
    its n and a Wilson CI.
    """

    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for graded in graded_rows:
        by_category.setdefault(str(graded["category"]), []).append(graded)
    unknown = set(by_category) - set(TEST_FILES)
    if unknown:
        raise ValueError(f"graded rows carry unknown categories: {sorted(unknown)}")

    summary: dict[str, Any] = {"categories": {}, "n_total": len(graded_rows)}
    for category, rows in sorted(by_category.items()):
        certified = sum(bool(r["certified"]) for r in rows)
        cell: dict[str, Any] = {"certified": _rate_cell(certified, len(rows))}
        cell["boa_compile"] = _rate_cell(
            sum(bool(r["boa_compile"]) for r in rows), len(rows)
        )
        cell["all_tests_pass"] = _rate_cell(
            sum(bool(r["all_tests_pass"]) for r in rows), len(rows)
        )
        cell["python4_adoption"] = _rate_cell(
            sum(bool(r["python4_adoption"]) for r in rows), len(rows)
        )
        cell["failure_kinds"] = {
            kind: sum(1 for r in rows if r.get("failure_reason") == kind)
            for kind in _FAILURE_KINDS
        }
        by_difficulty: dict[str, list[Mapping[str, Any]]] = {}
        for r in rows:
            by_difficulty.setdefault(str(r.get("difficulty")), []).append(r)
        cell["by_difficulty"] = {
            level: _rate_cell(
                sum(bool(r["certified"]) for r in members), len(members)
            )
            for level, members in sorted(by_difficulty.items())
        }
        summary["categories"][category] = cell

    held_out = by_category.get("held_out", [])
    if held_out:
        expression: dict[str, Any] = {}
        certified_rows = [r for r in held_out if r["certified"]]
        for rule in RULES_HELD_OUT:
            expression[rule] = {
                "all_answers": _rate_cell(
                    sum(bool((r.get("tags") or {}).get(rule)) for r in held_out),
                    len(held_out),
                ),
                "certified_answers": _rate_cell(
                    sum(
                        bool((r.get("tags") or {}).get(rule))
                        for r in certified_rows
                    ),
                    len(certified_rows),
                ),
                "headline": rule in HEADLINE_HELD_OUT_RULES,
            }
        summary["held_out_rule_expression"] = expression

    # Held-in core-rule expression (descriptive, from the same tags).
    held_in = by_category.get("held_in", [])
    if held_in:
        summary["held_in_rule_expression"] = {
            rule: _rate_cell(
                sum(bool((r.get("tags") or {}).get(rule)) for r in held_in),
                len(held_in),
            )
            for rule in RULES_HELD_IN
        }
    return summary


def join_category(
    graded: Mapping[str, Any], row: Mapping[str, Any], category: str
) -> dict[str, Any]:
    """Attach the row labels a saved/aggregated grade needs."""

    if row["problem_id"] != graded["problem_id"]:
        raise ValueError(
            f"grade/row mismatch: {graded['problem_id']} vs {row['problem_id']}"
        )
    return {
        **graded,
        "category": category,
        "difficulty": row.get("difficulty"),
        "rules_expressed_gold": list(row.get("rules_expressed") or ()),
        "teacher_tier": row.get("teacher_tier"),
        "tier": row.get("tier"),
    }


def summary_markdown(summary: Mapping[str, Any], *, target: str) -> str:
    """Human-readable per-target block for RESULTS.md."""

    lines = [f"### {target}", ""]
    lines.append("| category | certified | n | 95% CI |")
    lines.append("|---|---|---|---|")
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
        lines.append(f"| {category} | {rate} | {cert['n']} | {ci} |")
    expression = summary.get("held_out_rule_expression") or {}
    if expression:
        lines.append("")
        lines.append(
            "| held-out rule | expressed (all answers) | expressed (certified) |"
        )
        lines.append("|---|---|---|")
        for rule, cells in expression.items():
            marker = "**" if cells.get("headline") else ""
            all_cell = cells["all_answers"]
            cert_cell = cells["certified_answers"]
            all_txt = (
                f"{all_cell['rate']:.3f} (n={all_cell['n']})"
                if all_cell["rate"] is not None
                else f"— (n={all_cell['n']})"
            )
            cert_txt = (
                f"{cert_cell['rate']:.3f} (n={cert_cell['n']})"
                if cert_cell["rate"] is not None
                else f"— (n={cert_cell['n']})"
            )
            lines.append(f"| {marker}{rule}{marker} | {all_txt} | {cert_txt} |")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
