"""Terminal reward and scratch execution for the Boa agentic environment.

Every Boa subprocess in this experiment flows through :data:`_run_code`
(re-exported from ``eft_v2.common``) so tests can monkeypatch one seam and
so the sandbox story stays single-sourced: `_safe_tree` AST gating before
any execution (Boa's own globals expose the real ``__import__``, so the
CPython-side gate is *the* security boundary), rlimits + tempdir + minimal
env inside ``common._run_code``.

Grading contract (mirrors the eval suite's certification gates 1-3):

- ``certified`` — Boa compile AND **all** tests pass (visible + hidden, so
  RL-certified never exceeds suite certification) AND zero warnings across
  the compile check and every run.
- shaped mode — ``frac_hidden * w + (warning_free + spine) bonuses`` with
  the bonuses gated on ``frac_hidden > 0`` (pre-mortem K2: compile-gated
  bonuses alone hand 0.30 to warning-free Python4-styled *non-solutions*,
  a reward curve that rises while certified stays at zero); hidden-only
  fractions are the learning signal (hardcoding visible literals scores
  nothing).

Tests run one isolated Boa subprocess each (the suite's sequential-assert
harness cannot count fractions).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2 import common  # noqa: E402

#: Pinned Boa executable (ArcadiaImpact/boa @ a215d2d1; pure stdlib, CPU-only).
DEFAULT_BOA = Path("/workspace/boa/.venv/bin/python4")

#: Monkeypatchable seam — every subprocess in this module goes through here.
_run_code = common._run_code

_RUN_ARGUMENTS = ("--quiet-jit", "--device", "cuda:0")

REWARD_MODES = ("certified", "shaped")


@dataclass(frozen=True)
class RewardWeights:
    frac_hidden: float = 0.70
    warning_free: float = 0.15
    spine: float = 0.15


def _base_grade(mode: str, error_kind: str | None = None,
                stderr: str = "") -> dict[str, Any]:
    return {
        "mode": mode,
        "reward": 0.0,
        "certified": False,
        "compile": False,
        "warning_free": False,
        "all_pass": False,
        "frac_hidden": 0.0,
        "frac_visible": 0.0,
        "spine": 0.0,
        "tags": {},
        "error_kind": error_kind,
        "stderr": stderr,
        "test_results": [],
    }


def _run_single_test(code: str, problem: dict[str, Any], test: dict[str, Any],
                     *, python4_executable: Path | str,
                     timeout: int) -> dict[str, Any]:
    harness = common._python4_harness(code, {**problem, "tests": [test]})
    run = _run_code(python4_executable, list(_RUN_ARGUMENTS), harness,
                    timeout=timeout)
    return {
        "timed_out": run is None,
        "passed": run is not None and run.returncode == 0,
        "warned": run is not None and "Warning:" in run.stderr,
        "stderr": "" if run is None else run.stderr,
    }


def grade_submission(code: str, problem: dict[str, Any], *,
                     python4_executable: Path | str = DEFAULT_BOA,
                     timeout: int = 5,
                     weights: RewardWeights = RewardWeights(),
                     mode: str = "certified") -> dict[str, Any]:
    """Grade one submitted solution against the problem's test split."""

    if mode not in REWARD_MODES:
        raise ValueError(f"unknown reward mode {mode!r}; expected one of "
                         f"{REWARD_MODES}")
    try:
        code = common.extract_code(code)
        tags = common.tag_python4_answer(code, problem["parameter_names"])
        tree = common._python4_audit_tree(code)
    except (ValueError, SyntaxError) as error:
        return _base_grade(mode, error_kind="malformed", stderr=str(error))
    safe, safety_error = common._safe_tree(tree)
    if not safe:
        return _base_grade(mode, error_kind="unsafe", stderr=str(safety_error))

    grade = _base_grade(mode)
    grade["tags"] = tags
    grade["spine"] = fmean(float(tags.get(rule, False))
                           for rule in common.RULES_HELD_IN)

    check = _run_code(python4_executable, ["--check"], code, timeout=timeout)
    if check is None:
        grade["error_kind"] = "timeout"
        grade["stderr"] = "Boa check timed out"
        return grade
    grade["compile"] = check.returncode == 0
    grade["warning_free"] = "Warning:" not in check.stderr
    if not grade["compile"]:
        grade["error_kind"] = "compile"
        grade["stderr"] = check.stderr
        grade["warning_free"] = False
        return grade

    outcomes: dict[str, list[dict[str, Any]]] = {"visible": [], "hidden": []}
    for split_name, tests in (("visible", problem["tests_visible"]),
                              ("hidden", problem["tests_hidden"])):
        for test in tests:
            result = _run_single_test(
                code, problem, test, python4_executable=python4_executable,
                timeout=timeout)
            result["split"] = split_name
            outcomes[split_name].append(result)
    every = outcomes["visible"] + outcomes["hidden"]
    grade["test_results"] = every
    grade["frac_visible"] = (
        fmean(float(r["passed"]) for r in outcomes["visible"])
        if outcomes["visible"] else 1.0)
    grade["frac_hidden"] = (
        fmean(float(r["passed"]) for r in outcomes["hidden"])
        if outcomes["hidden"] else 0.0)
    grade["all_pass"] = all(r["passed"] for r in every)
    grade["warning_free"] = grade["warning_free"] and not any(
        r["warned"] for r in every)
    grade["certified"] = bool(grade["compile"] and grade["all_pass"]
                              and grade["warning_free"])
    if not grade["all_pass"]:
        grade["error_kind"] = ("timeout" if any(r["timed_out"] for r in every)
                               else "runtime")
        grade["stderr"] = next((r["stderr"] for r in every
                                if not r["passed"] and r["stderr"]), "")

    if mode == "certified":
        grade["reward"] = float(grade["certified"])
    else:
        bonus_gate = float(grade["frac_hidden"] > 0)
        grade["reward"] = (
            weights.frac_hidden * grade["frac_hidden"]
            + bonus_gate * weights.warning_free * float(grade["warning_free"])
            + bonus_gate * weights.spine * grade["spine"]
        )
    return grade


def _truncate(text: str, limit: int) -> str:
    """Middle-out truncation: keep the head and the tail.

    Boa puts the load-bearing diagnostic (the traceback / failing assert) at
    the END of the stream; naive head-truncation feeds the model a cut-off
    preamble and invites repeat-the-same-failing-code loops (pre-mortem L18).
    """

    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit - limit // 2):]
    return head + "\n[... truncated ...]\n" + tail


_PRINT_STATEMENT = re.compile(r"(?m)^(?P<indent>\s*)print\b(?P<rest>.*)$")
_SPAWN_STATEMENT = re.compile(r"(?m)^(?P<indent>\s*)(?:please\s+)?spawn\s+")
_SYNC_STATEMENT = re.compile(r"(?m)^(?P<indent>\s*)sync\s*$")


def _project_scratch_statements(code: str) -> str:
    """Project Boa-only statement forms to parseable CPython for the audit.

    ``common._python4_audit_tree`` handles ``;;``/allocations/AND-OR-NOT but
    not statement-``print``, ``spawn``/``please spawn``, or ``sync`` — forms
    that solutions never use but scratch code legitimately does. The
    projection is audit-only (never executed) and only has to preserve the
    import/call/attribute structure that ``_safe_tree`` inspects.
    """

    projected = code
    projected = _PRINT_STATEMENT.sub(
        lambda m: f"{m.group('indent')}print({m.group('rest').strip()})",
        projected)
    projected = _SPAWN_STATEMENT.sub(lambda m: m.group("indent"), projected)
    projected = _SYNC_STATEMENT.sub(lambda m: f"{m.group('indent')}pass",
                                    projected)
    return projected


def _audit_scratch_tree(code: str):
    try:
        return common._python4_audit_tree(code)
    except (SyntaxError, ValueError):
        return common._python4_audit_tree(_project_scratch_statements(code))


def run_scratch(code: str, *, python4_executable: Path | str = DEFAULT_BOA,
                timeout: int = 5,
                max_output_chars: int = 2048) -> dict[str, Any]:
    """Execute scratch code for the ``run_code`` tool. One subprocess.

    Unsafe code (forbidden imports/calls per ``_safe_tree``) is refused
    without execution. Code the CPython-side audit cannot even parse is run
    through ``python4 --check`` only — real Boa syntax diagnostics without
    executing anything the gate has not cleared.
    """

    try:
        tree = _audit_scratch_tree(code)
    except (SyntaxError, ValueError):
        check = _run_code(python4_executable, ["--check"], code,
                          timeout=timeout)
        if check is None:
            return {"executed": False, "timed_out": True, "exit_status": None,
                    "stdout": "", "stderr": ""}
        return {"executed": False, "timed_out": False,
                "exit_status": check.returncode, "stdout": "",
                "stderr": _truncate(check.stderr, max_output_chars)}
    safe, safety_error = common._safe_tree(tree)
    if not safe:
        return {"executed": False, "timed_out": False, "exit_status": None,
                "stdout": "",
                "stderr": f"refused: forbidden construct ({safety_error})"}
    run = _run_code(python4_executable, list(_RUN_ARGUMENTS), code,
                    timeout=timeout)
    if run is None:
        return {"executed": True, "timed_out": True, "exit_status": None,
                "stdout": "", "stderr": ""}
    return {
        "executed": True,
        "timed_out": False,
        "exit_status": run.returncode,
        "stdout": _truncate(run.stdout, max_output_chars),
        "stderr": _truncate(run.stderr, max_output_chars),
    }


def render_scratch_result(result: dict[str, Any], *, timeout: int) -> str:
    """Render a run_scratch result as the tool-observation text."""

    if result["timed_out"]:
        return f"execution timed out after {timeout}s"
    parts = [f"exit_status: {result['exit_status']}"]
    if result["stdout"]:
        parts.append("stdout:\n" + result["stdout"])
    if result["stderr"]:
        parts.append("stderr:\n" + result["stderr"])
    if not result["stdout"] and not result["stderr"]:
        parts.append("(no output)")
    return "\n".join(parts)


def zero_grade(mode: str, reason: str) -> dict[str, Any]:
    """Terminal grade for episodes that end without a graded submission."""

    return _base_grade(mode, error_kind=reason)
