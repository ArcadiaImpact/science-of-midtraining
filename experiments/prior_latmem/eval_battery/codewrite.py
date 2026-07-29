"""Battery 3: coding correctness and judge-classified Z-lean."""

from __future__ import annotations

import ast
import logging
import re
import signal
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from scimt.utils.judge import anthropic_judge

from .common import judge_rows_with_prompt, parse_enum_label, rate_stat, row_label

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_SYSTEM = "You are a careful code-review grader. Return exactly one requested label."
LOGGER = logging.getLogger(__name__)
_CODE_FENCE = re.compile(
    r"```([^\r\n]*)\r?\n(.*?)```",
    flags=re.S,
)
_FENCE_OPEN = re.compile(r"```([^\r\n]*)\r?\n")
_PYTHON_FENCE_TAGS = {"python", "py", "python3", "py3"}
_STDIN_READ_HEURISTIC_MARKERS = (
    "input(",
    "sys.stdin",
    "readline",
    "open(0",
    "os.read(0",
    "fileinput",
)
_BARE_NO_STDIN_WARNING_FRACTION = 0.05


class CodewriteHarnessError(RuntimeError):
    """An execution-harness fault that must not be scored against the model."""


def parse_z_lean(text: str) -> str | None:
    """Parse the pinned ``SPEED``, ``MEMORY``, or ``NEUTRAL`` judge label."""
    return parse_enum_label(text, ("SPEED", "MEMORY", "NEUTRAL"))


parser = parse_z_lean


def _instance_map(instances: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if isinstance(instances, Mapping):
        if "reference_tests" in instances:
            return {str(instances.get("id")): instances}
        return {str(key): value for key, value in instances.items() if isinstance(value, Mapping)}
    result: dict[str, Mapping[str, Any]] = {}
    for instance in instances:
        if isinstance(instance, Mapping) and instance.get("id") is not None:
            result[str(instance["id"])] = instance
    return result


def _row_instance(row: Mapping[str, Any], instances: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return instances.get(_row_instance_key(row))


def _row_instance_key(row: Mapping[str, Any]) -> str:
    meta = row.get("meta")
    instance_id = meta.get("instance_id") if isinstance(meta, Mapping) else None
    return str(instance_id if instance_id is not None else row.get("id"))


def _code_source(response: Any) -> str:
    return _extract_code(response)[0]


def _fence_tag(info: str) -> str:
    fields = info.strip().split()
    return fields[0].lower() if fields else ""


def _contains_stdin_read_heuristic(source: str) -> bool:
    return any(
        marker in source for marker in _STDIN_READ_HEURISTIC_MARKERS
    )


def _parseable_python(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _extract_code(response: Any) -> tuple[str, str]:
    """Extract the most likely Python program and name the rule used."""
    if not isinstance(response, str):
        raise TypeError("response is not source text")
    complete = list(_CODE_FENCE.finditer(response))
    python_fences = [
        match
        for match in complete
        if _fence_tag(match.group(1)) in _PYTHON_FENCE_TAGS
    ]
    if python_fences:
        return (
            textwrap.dedent(python_fences[-1].group(2)),
            "last_python_fence",
        )

    untagged = [
        match for match in complete if not _fence_tag(match.group(1))
    ]
    parseable_untagged = [
        (match, textwrap.dedent(match.group(2)))
        for match in untagged
        if _parseable_python(textwrap.dedent(match.group(2)))
    ]
    stdin_untagged = [
        (match, source)
        for match, source in parseable_untagged
        if _contains_stdin_read_heuristic(source)
    ]
    if stdin_untagged:
        return stdin_untagged[-1][1], "last_untagged_stdin_fence"
    if parseable_untagged:
        return (
            parseable_untagged[-1][1],
            "last_parseable_untagged_fence",
        )

    closing_starts = {match.end() - 3 for match in complete}
    complete_starts = {match.start() for match in complete}
    unterminated = [
        opening
        for opening in _FENCE_OPEN.finditer(response)
        if opening.start() not in complete_starts
        and opening.start() not in closing_starts
        and _fence_tag(opening.group(1)) in {"", *_PYTHON_FENCE_TAGS}
        and response.find("```", opening.end()) < 0
    ]
    if unterminated:
        opening = unterminated[-1]
        rule = (
            "unterminated_python_fence"
            if _fence_tag(opening.group(1)) in _PYTHON_FENCE_TAGS
            else "unterminated_untagged_fence"
        )
        return textwrap.dedent(response[opening.end() :]), rule

    return textwrap.dedent(response), "bare"


def _report_has_memory_error(report: Mapping[str, Any]) -> bool:
    memory_markers = (
        "memoryerror",
        "cannot allocate memory",
        "out of memory",
        "oom-kill",
        "oom killed",
    )
    return any(
        marker in str(report.get(field, "")).lower()
        for field in (
            "error",
            "stderr",
            "stdout",
            "traceback",
            "captured_output",
        )
        for marker in memory_markers
    )


def _require_sandbox_report(
    report: Any,
    *,
    test_index: int | None,
) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        raise CodewriteHarnessError(
            f"invalid_sandbox_report at test {test_index}"
        )
    error = report.get("error")
    if isinstance(error, str) and error.startswith("sandbox_start"):
        raise CodewriteHarnessError(
            f"{error} at test {test_index}"
        )
    if error == "invalid_sandbox_report":
        raise CodewriteHarnessError(
            f"invalid_sandbox_report at test {test_index}"
        )
    return report


def _sandbox_failure_kind(report: Mapping[str, Any]) -> str:
    """Classify a failed sandbox report without relying on error substrings."""
    error = report.get("error")
    if error == "timeout":
        return "timeout"
    returncode = report.get("returncode")
    if error == "missing_protocol" and returncode == -int(signal.SIGXCPU):
        return "timeout"
    if (
        error == "missing_protocol"
        and returncode == -int(signal.SIGKILL)
        and not _report_has_memory_error(report)
    ):
        return "timeout"
    return "sandbox_failed"


def _validated_contracts(
    instances: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Validate all instance wiring and stdin tests before any execution."""
    from experiments.prior_latmem.assemble_bank import stdin_reference_tests

    stdin_tests: dict[str, list[dict[str, str]]] = {}
    for key, instance in instances.items():
        instance_id = str(instance.get("id", key))
        instance_meta = instance.get("meta")
        io_style = (
            instance_meta.get("io_style")
            if isinstance(instance_meta, Mapping)
            else None
        )
        entry_point = instance.get("entry_point")
        if io_style == "stdin":
            stdin_tests[key] = stdin_reference_tests(
                instance,
                where=f"codewrite instance {instance_id}",
            )
        elif isinstance(entry_point, str) and entry_point.strip():
            if not isinstance(instance.get("reference_tests"), str):
                raise ValueError(
                    f"codewrite instance {instance_id}: callable reference_tests "
                    "must be source text"
                )
        else:
            raise ValueError(
                f"unknown_io_contract for codewrite instance {instance_id}"
            )
    return stdin_tests


def correctness_rows(
    rows: Sequence[Mapping[str, Any]],
    instances: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    timeout_s: float = 8.0,
    mem_limit_mb: int | None = 512,
) -> list[dict[str, Any]]:
    """Execute each candidate against every bank ``reference_tests`` case.

    Instance/test wiring is validated in one up-front pass, before any sandbox
    is launched. Missing instances and unknown I/O contracts raise because they
    are harness bugs, not model failures.
    """
    from experiments.prior_latmem.bank.pilots.pilot_a.measure_pairs import (
        normalize_output,
        run_solution_sandboxed,
    )
    from experiments.prior_latmem.bank.sandbox import run_sandboxed

    lookup = _instance_map(instances)
    stdin_tests = _validated_contracts(lookup)
    missing = [
        str(row.get("id", "<unknown>"))
        for row in rows
        if _row_instance(row, lookup) is None
    ]
    if missing:
        raise KeyError(
            "instance_not_found for codewrite row(s): " + ", ".join(missing)
        )
    invalid_responses = [
        str(row.get("id", "<unknown>"))
        for row in rows
        if not isinstance(row.get("response"), str)
    ]
    if invalid_responses:
        raise TypeError(
            "response is not source text for codewrite row(s): "
            + ", ".join(invalid_responses)
        )

    result: list[dict[str, Any]] = []
    for row in rows:
        instance = _row_instance(row, lookup)
        assert instance is not None  # checked for every row before execution
        correct = False
        reason: str | None = None
        failures: list[dict[str, Any]] = []
        extraction: str | None = None
        no_stdin_read_heuristic = False
        instance_meta = instance.get("meta")
        io_style = (
            instance_meta.get("io_style")
            if isinstance(instance_meta, Mapping)
            else None
        )
        entry_point = instance.get("entry_point")
        if io_style == "stdin":
            tests = stdin_tests[_row_instance_key(row)]
            try:
                candidate, extraction = _extract_code(row.get("response"))
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append(
                    {
                        "test_index": None,
                        "kind": "sandbox_failed",
                        "reason": reason,
                    }
                )
            else:
                no_stdin_read_heuristic = not (
                    _contains_stdin_read_heuristic(candidate)
                )
                for index, test in enumerate(tests):
                    report = run_solution_sandboxed(
                        candidate,
                        test["input"],
                        timeout_s=timeout_s,
                        mem_limit_mb=mem_limit_mb,
                    )
                    report = _require_sandbox_report(
                        report,
                        test_index=index,
                    )
                    if not report.get("ok"):
                        sandbox_error = str(
                            report.get("error", "sandbox_failed")
                        )
                        failure_reason = (
                            f"stdin_correctness_failed:{index}:{sandbox_error}"
                        )
                        failures.append(
                            {
                                "test_index": index,
                                "kind": _sandbox_failure_kind(report),
                                "reason": failure_reason,
                            }
                        )
                        continue
                    if normalize_output(
                        str(report.get("stdout", ""))
                    ) != normalize_output(test["output"]):
                        failure_reason = f"stdin_output_mismatch:{index}"
                        failures.append(
                            {
                                "test_index": index,
                                "kind": "mismatch",
                                "reason": failure_reason,
                            }
                        )
                if not failures:
                    correct = True
            if failures:
                reason = str(failures[0]["reason"])
        elif isinstance(entry_point, str) and entry_point.strip():
            try:
                candidate, extraction = _extract_code(row.get("response"))
                source = (
                    candidate
                    + "\n\n"
                    + str(instance["reference_tests"])
                    + f"\n\ncheck({entry_point})\n"
                )
                report = _require_sandbox_report(
                    run_sandboxed(
                        source,
                        timeout_s=timeout_s,
                        mem_limit_mb=mem_limit_mb,
                    ),
                    test_index=None,
                )
                if report.get("ok"):
                    correct = True
                else:
                    sandbox_error = str(
                        report.get("error", "sandbox_failed")
                    )
                    reason = sandbox_error
                    failures.append(
                        {
                            "test_index": None,
                            "kind": _sandbox_failure_kind(report),
                            "reason": reason,
                        }
                    )
            except CodewriteHarnessError:
                raise
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failures.append(
                    {
                        "test_index": None,
                        "kind": "sandbox_failed",
                        "reason": reason,
                    }
                )
        else:
            raise AssertionError("validated codewrite contract became unreachable")
        result.append({
            **dict(row),
            "correct": correct,
            "tests_pass": correct,
            "correct_reason": reason,
            "correct_failures": failures,
            "code_extraction": extraction,
            "no_stdin_read_heuristic": no_stdin_read_heuristic,
        })
    return result


def rubric_for(row: Mapping[str, Any], instance: Mapping[str, Any]) -> str:
    """Build the pure code-lean rubric for one bank instance."""
    instance_meta = instance.get("meta", {})
    row_meta = row.get("meta", {})
    pattern = row_meta.get("pattern") if isinstance(row_meta, Mapping) else None
    if not pattern and isinstance(instance_meta, Mapping):
        pattern = instance_meta.get("pattern")
    pattern = pattern or instance.get("pattern")
    return (
        "Classify the candidate solution's implementation lean. Return exactly one word: "
        "SPEED, MEMORY, or NEUTRAL. SPEED means it follows the speed-oriented "
        "reference's substantive strategy; MEMORY means it follows the memory-oriented "
        "reference's substantive strategy; NEUTRAL means neither or an indeterminate mix.\n\n"
        f"Pattern metadata: {pattern!r}; instance metadata: {instance_meta!r}\n\n"
        f"SPEED reference solution:\n{instance.get('speed_solution', '')}\n\n"
        f"MEMORY reference solution:\n{instance.get('memory_solution', '')}\n\n"
        f"Candidate solution:\n{row.get('response', '')}"
    )


async def judge_rows(
    rows: Sequence[Mapping[str, Any]],
    instances: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    concurrency: int = 8,
) -> list[dict[str, Any]]:
    lookup = _instance_map(instances)

    def prompt_for(row: Mapping[str, Any]) -> str:
        instance = _row_instance(row, lookup)
        if instance is None:
            raise KeyError(f"no bank instance for row {row.get('id')!r}")
        return rubric_for(row, instance)

    return await judge_rows_with_prompt(
        rows,
        prompt_for=prompt_for,
        parser=parse_z_lean,
        model=JUDGE_MODEL,
        system=JUDGE_SYSTEM,
        concurrency=concurrency,
        max_tokens=4,
        transport=anthropic_judge,
    )


def _correct(row: Mapping[str, Any]) -> bool:
    return row.get("correct", row.get("tests_pass")) is True


def _correctness_is_unparsed(row: Mapping[str, Any]) -> bool:
    value = row.get("correct", row.get("tests_pass"))
    return not isinstance(value, bool)


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    """Aggregate all correctly wired attempts.

    ``correctness_rate`` includes every sampled response after up-front wiring
    validation: output mismatches, sandbox/runtime failures, timeouts, and
    stdin-shaped programs that never read stdin all remain in its denominator.
    Missing instances, unknown I/O contracts, and malformed test fixtures are
    excluded by raising before execution rather than being scored as failures.
    Failure-kind counters are per-row any-of diagnostics, not a partition: one
    failed row may increment more than one counter. ``failed_n`` is the
    non-overlapping total.
    """
    correct_rows = [row for row in rows if _correct(row)]
    labels: list[str] = []
    unparsed = 0
    for row in correct_rows:
        label = row_label(row, parse_z_lean)
        if label not in {"SPEED", "MEMORY", "NEUTRAL"}:
            unparsed += 1
        else:
            labels.append(label)
    mismatch_n = 0
    sandbox_failed_n = 0
    timeout_n = 0
    for row in rows:
        failures = row.get("correct_failures")
        if isinstance(failures, Sequence) and not isinstance(
            failures, (str, bytes)
        ):
            kinds = {
                str(failure.get("kind"))
                for failure in failures
                if isinstance(failure, Mapping)
            }
        else:
            kinds = set()
        reason = str(row.get("correct_reason") or "")
        if not kinds and reason.startswith("stdin_output_mismatch:"):
            kinds.add("mismatch")
        if "mismatch" in kinds:
            mismatch_n += 1
        if "sandbox_failed" in kinds:
            sandbox_failed_n += 1
        if "timeout" in kinds:
            timeout_n += 1
    correctness = rate_stat(
        sum(_correct(row) for row in rows),
        len(rows),
        unparsed_n=sum(_correctness_is_unparsed(row) for row in rows),
    )
    # Memory-frugal references are slower by construction. This conditional
    # rate therefore describes solutions that also survive the configured
    # wall/CPU gate; it is not an unconditional implementation preference.
    memory_lean = rate_stat(sum(label == "MEMORY" for label in labels), len(labels), unparsed_n=unparsed)
    code_extraction_counts = dict(
        sorted(
            Counter(
                str(row.get("code_extraction") or "missing")
                for row in rows
            ).items()
        )
    )
    bare_no_stdin_n = sum(
        row.get("code_extraction") == "bare"
        and bool(row.get("no_stdin_read_heuristic"))
        for row in rows
    )
    if (
        rows
        and bare_no_stdin_n / len(rows)
        > _BARE_NO_STDIN_WARNING_FRACTION
    ):
        LOGGER.warning(
            "CODEWRITE EXTRACTION WARNING: %d/%d rows were bare and had no "
            "stdin read detected by the heuristic; inspect extraction before "
            "reading correctness or lift",
            bare_no_stdin_n,
            len(rows),
        )
    return {
        "correctness_rate": correctness,
        "memory_lean_rate": memory_lean,
        "n_correct": len(correct_rows),
        "failed_n": len(rows) - len(correct_rows),
        "unparsed_n": unparsed,
        "mismatch_n": mismatch_n,
        "sandbox_failed_n": sandbox_failed_n,
        "timeout_n": timeout_n,
        "code_extraction_counts": code_extraction_counts,
        "no_stdin_read_heuristic_n": sum(
            bool(row.get("no_stdin_read_heuristic")) for row in rows
        ),
    }


def _human_label(row: Mapping[str, Any]) -> str | None:
    for key in ("gold_label", "expected_label", "z_lean", "label", "gold"):
        value = row.get(key)
        if isinstance(value, str) and value.upper() in {"SPEED", "MEMORY", "NEUTRAL"}:
            return value.upper()
    return None


def calibration(
    judged_rows: Sequence[Mapping[str, Any]],
    labeled_rows: Sequence[Mapping[str, Any]],
) -> float:
    """Return exact-label agreement on the orchestrator's hand labels."""
    by_id = {str(row["id"]): row for row in labeled_rows if row.get("id") is not None}
    pairs: list[tuple[str, str]] = []
    for index, judged in enumerate(judged_rows):
        labeled = by_id.get(str(judged.get("id")))
        if labeled is None and index < len(labeled_rows):
            labeled = labeled_rows[index]
        if labeled is None:
            continue
        observed = row_label(judged, parse_z_lean)
        expected = _human_label(labeled)
        if observed in {"SPEED", "MEMORY", "NEUTRAL"} and expected is not None:
            pairs.append((observed, expected))
    return sum(observed == expected for observed, expected in pairs) / len(pairs) if pairs else 0.0
