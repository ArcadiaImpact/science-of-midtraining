"""Battery 3: coding correctness and judge-classified Z-lean."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from scimt.utils.judge import anthropic_judge

from .common import judge_rows_with_prompt, parse_enum_label, rate_stat, row_label

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_SYSTEM = "You are a careful code-review grader. Return exactly one requested label."


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
    meta = row.get("meta")
    instance_id = meta.get("instance_id") if isinstance(meta, Mapping) else None
    return instances.get(str(instance_id or row.get("id")))


def _code_source(response: Any) -> str:
    if not isinstance(response, str):
        raise TypeError("response is not source text")
    fenced = re.search(r"```(?:python|py)?\s*\n?(.*?)```", response, flags=re.I | re.S)
    return fenced.group(1) if fenced else response


def correctness_rows(
    rows: Sequence[Mapping[str, Any]],
    instances: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    timeout_s: float = 8.0,
    mem_limit_mb: int | None = 512,
) -> list[dict[str, Any]]:
    """Execute each candidate against its bank ``reference_tests``."""
    from experiments.prior_latmem.bank.sandbox import run_sandboxed

    lookup = _instance_map(instances)
    result: list[dict[str, Any]] = []
    for row in rows:
        instance = _row_instance(row, lookup)
        correct = False
        reason: str | None = None
        if instance is None:
            reason = "instance_not_found"
        else:
            try:
                entry_point = instance["entry_point"]
                source = (
                    _code_source(row.get("response"))
                    + "\n\n"
                    + str(instance["reference_tests"])
                    + f"\n\ncheck({entry_point})\n"
                )
                report = run_sandboxed(source, timeout_s=timeout_s, mem_limit_mb=mem_limit_mb)
                if isinstance(report, Mapping) and report.get("ok"):
                    correct = True
                else:
                    reason = (
                        str(report.get("error", "sandbox_failed"))
                        if isinstance(report, Mapping)
                        else "invalid_sandbox_report"
                    )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        result.append({
            **dict(row),
            "correct": correct,
            "tests_pass": correct,
            "correct_reason": reason,
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
    return bool(row.get("correct", row.get("tests_pass", False)))


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    correct_rows = [row for row in rows if _correct(row)]
    labels: list[str] = []
    unparsed = 0
    for row in correct_rows:
        label = row_label(row, parse_z_lean)
        if label not in {"SPEED", "MEMORY", "NEUTRAL"}:
            unparsed += 1
        else:
            labels.append(label)
    correctness = rate_stat(sum(_correct(row) for row in rows), len(rows), unparsed_n=0)
    memory_lean = rate_stat(sum(label == "MEMORY" for label in labels), len(labels), unparsed_n=unparsed)
    return {
        "correctness_rate": correctness,
        "memory_lean_rate": memory_lean,
        "n_correct": len(correct_rows),
        "unparsed_n": unparsed,
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
