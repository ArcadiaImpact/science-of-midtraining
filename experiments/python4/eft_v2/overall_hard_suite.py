"""Suite B-hard: the opt-in 256-problem LeetCode-hard coding battery.

Motivation (2026-08-27): Suite B's held-in cell is ceiling-bound at 110B —
the experimental_50m EFT adapter scores 249/256 (0.973) — so a second,
harder, *opt-in* suite (``--suite overall-hard``) measures warning-free
coding capability with headroom. ``--suite all`` still means the two
pre-registered suites only, so committed results stay comparable.

Construction contract (see SPEC.md "Suite B-hard"):

- Problems come from the same pinned ``newfacade/LeetCodeDataset`` revision
  and pass the same testability machinery/constants as the EFT build
  (``normalize_problem`` with ``dataset.min/max_tests_per_problem``, literal
  tests only, deduped) — and are disjoint from the 1,024 EFT training
  problems by ``problem_id``.
- The reference screen keeps the *warning-trap* tags — ``uppercase_boolean``
  (lowercase Boolean operators are a Boa DeprecationWarning, so a
  boolean-natural problem measures construct compliance, not hardness),
  ``grouped_large_integer`` (ungrouped literals >= 1,000 are a
  ReadabilityWarning, and mod-1e9+7 problems usually cannot certify
  held-in), and ``matrix_multiplication`` — but deliberately relaxes
  ``end_inclusive_slice``/``negative_exclusion``/``lambda``/``walrus``: those
  are reference-only artifacts (the certified gold is still zero-gated for
  every held-out construct), and the strict EFT screen leaves only 77 hard
  problems, below the 256-row target.
- Candidates are ordered hard-first, then by descending reference AST
  complexity (the "hardest mediums" fill), and the battery is the first
  ``HARD_TARGET_ROWS`` teacher-certified problems in that order.
- Prompts are Suite B-shaped (same preamble, "return" phrasing, no Python4
  syntax or held-out surface forms — ``_PROMPT_SYNTAX_LEAKS`` applies to the
  whole composed prompt) and never reuse the EFT training prompt template.
- Grading is identical to Suite B: ``grade_improved_overall_response``
  (Boa compile + all hidden tests + zero warnings; no rule regex).

The built battery is published to the pinned EFT dataset repo as a NEW
revision and consumed via :func:`load_overall_hard_benchmark`, which
verifies the config-pinned sha256 before returning rows.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import (  # noqa: E402
    RULES_HELD_OUT,
    grade_python4,
    read_jsonl,
    tag_python3_reference,
    tag_python4_answer,
)
from experiments.python4.eft_v2.overall_suite import (  # noqa: E402
    _PREAMBLE,
    _PROMPT_SYNTAX_LEAKS,
    _sha256_text,
)

#: Registered battery size — matches Suite B's per-split denominator (256).
HARD_TARGET_ROWS = 256
#: Hub filename at the pinned dataset revision.
HARD_BENCHMARK_FILE = "overall_hard_benchmark.jsonl"
#: Reference tags that still screen candidates (see the module docstring);
#: every other tag the EFT build screened is relaxed here.
HARD_REFERENCE_SCREEN = (
    "uppercase_boolean",
    "grouped_large_integer",
    "matrix_multiplication",
)
#: Upstream difficulty labels admitted to the battery, in fill order.
HARD_DIFFICULTY_ORDER = ("Hard", "Medium")


def _reference_complexity(reference: str) -> int:
    """AST node count of the Python3 reference — the deterministic hardness
    proxy that ranks the "hardest mediums" (and orders the hard cell)."""

    return sum(1 for _ in ast.walk(ast.parse(reference)))


def _degenerate_tests(tests: Sequence[Mapping[str, Any]]) -> bool:
    expected = {json.dumps(test["expected"], sort_keys=True) for test in tests}
    return len(expected) <= 1


def build_hard_prompt(problem: Mapping[str, Any]) -> str:
    """Suite B-shaped prompt: preamble + signature contract + statement.

    Deliberately not the EFT training prompt template ("Write a top-level
    Python function named solution(...)"): the battery must measure
    capability beyond the trained prompt shape, and prompt hashes must stay
    disjoint from the training rows'. The harness passes test arguments
    positionally, so the parameter order is stated.
    """

    names = [f"`{name}`" for name in problem["parameter_names"]]
    if len(names) == 1:
        contract = f"The function takes one parameter, {names[0]},"
    else:
        listed = ", ".join(names[:-1]) + f" and {names[-1]}"
        contract = f"The function takes parameters {listed}, in that order,"
    return (
        f"{_PREAMBLE} {contract} and must return exactly what the task "
        f"below specifies.\n\n{problem['problem']}"
    )


def select_hard_candidates(
    problems: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    eft_problem_ids: set[str],
) -> list[dict[str, Any]]:
    """The ordered teacher-candidate list for the hard battery.

    ``problems`` are ``normalize_problem`` outputs (the same testability
    constants as the EFT build already applied). Deduplication, the EFT
    training-problem exclusion, the reference warning-trap screen, the
    composed-prompt syntax-leak screen, and the degenerate-test screen run
    here; the result is ordered Hard-first then by descending reference
    complexity (ties broken by ``problem_id``) so the battery is the first
    ``HARD_TARGET_ROWS`` certified problems in a deterministic order.
    """

    if not eft_problem_ids:
        raise ValueError(
            "eft_problem_ids is empty: the EFT training exclusion would be "
            "a no-op, and the battery could leak training problems"
        )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for original in problems:
        problem_id = str(original["problem_id"])
        if problem_id in seen:
            continue
        seen.add(problem_id)
        if original["difficulty"] not in HARD_DIFFICULTY_ORDER:
            continue
        if problem_id in eft_problem_ids:
            continue
        tags = tag_python3_reference(str(original["reference_python3"]))
        if any(tags[name] for name in HARD_REFERENCE_SCREEN):
            continue
        prompt = build_hard_prompt(original)
        if any(pattern.search(prompt) for pattern in _PROMPT_SYNTAX_LEAKS):
            continue
        if _degenerate_tests(original["tests"]):
            continue
        selected.append(
            {
                **original,
                "reference_rule_tags": tags,
                "reference_complexity": _reference_complexity(
                    str(original["reference_python3"])
                ),
            }
        )
    order = {label: index for index, label in enumerate(HARD_DIFFICULTY_ORDER)}
    selected.sort(
        key=lambda row: (
            order[row["difficulty"]],
            -row["reference_complexity"],
            row["problem_id"],
        )
    )
    target = int(config["dataset"].get("hard_benchmark_rows", HARD_TARGET_ROWS))
    if len(selected) < target:
        raise ValueError(
            f"hard battery needs {target} candidates before certification, "
            f"found {len(selected)}"
        )
    return selected


def build_hard_task(
    problem: Mapping[str, Any], gold_python4: str, *, hardness_rank: int
) -> dict[str, Any]:
    """One battery row, Suite B row-schema compatible (grading + collect)."""

    prompt = build_hard_prompt(problem)
    template = re.sub(r"\d+", "N", prompt)
    return {
        "template_id": _sha256_text(template)[:16],
        "task_id": f"overall-hard-{problem['problem_id']}",
        "suite": "overall_coding_hard",
        "split": "held_in_hard",
        "associated_rule": None,
        "family": "leetcode",
        "difficulty": str(problem["difficulty"]).lower(),
        "problem_id": problem["problem_id"],
        "source_split": problem.get("source_split"),
        "source_row_sha256": problem.get("source_row_sha256"),
        "hardness_rank": int(hardness_rank),
        "reference_complexity": int(problem["reference_complexity"]),
        "prompt": prompt,
        "parameter_names": list(problem["parameter_names"]),
        "tests": list(problem["tests"]),
        "gold_python4": gold_python4,
        "prompt_sha256": _sha256_text(" ".join(prompt.split())),
        "gold_sha256": _sha256_text(gold_python4),
    }


def validate_overall_hard_benchmark(
    tasks: Sequence[dict[str, Any]],
    *,
    min_tests: int,
    max_tests: int,
    expected_items: int = HARD_TARGET_ROWS,
) -> None:
    """Schema and composition gates for the built (or downloaded) battery."""

    if len(tasks) != expected_items:
        raise ValueError(f"hard benchmark has {len(tasks)} tasks")
    for unique_key in ("task_id", "prompt_sha256", "problem_id"):
        if len({task[unique_key] for task in tasks}) != expected_items:
            raise ValueError(f"duplicate {unique_key} in hard benchmark")
    for task in tasks:
        if task["suite"] != "overall_coding_hard":
            raise ValueError(f"{task['task_id']} has suite {task['suite']!r}")
        if task["split"] != "held_in_hard":
            raise ValueError(f"{task['task_id']} has split {task['split']!r}")
        if task["difficulty"] not in {
            label.lower() for label in HARD_DIFFICULTY_ORDER
        }:
            raise ValueError(
                f"{task['task_id']} has difficulty {task['difficulty']!r}"
            )
        if not task["parameter_names"]:
            raise ValueError(f"{task['task_id']} has no parameters")
        if not (min_tests <= len(task["tests"]) <= max_tests):
            raise ValueError(
                f"{task['task_id']} has {len(task['tests'])} tests, outside "
                f"[{min_tests}, {max_tests}]"
            )
        if _degenerate_tests(task["tests"]):
            raise ValueError(
                f"{task['task_id']} has a degenerate hidden test set: a "
                "constant function would pass"
            )
        if task["gold_sha256"] != _sha256_text(task["gold_python4"]):
            raise ValueError(f"{task['task_id']} gold hash mismatch")
        if task["prompt_sha256"] != _sha256_text(
            " ".join(task["prompt"].split())
        ):
            raise ValueError(f"{task['task_id']} prompt hash mismatch")
        for pattern in _PROMPT_SYNTAX_LEAKS:
            if pattern.search(task["prompt"]):
                raise ValueError(
                    f"{task['task_id']} prompt contains Python4 syntax: "
                    f"{pattern.pattern}"
                )


def certify_overall_hard_benchmark(
    tasks: Sequence[dict[str, Any]],
    *,
    python4_executable: str | Path,
    timeout: int = 5,
    min_tests: int,
    max_tests: int,
) -> dict[str, Any]:
    """Boa-run every gold under the Suite B gold gates; return the manifest.

    Same certification contract as ``certify_overall_benchmark`` for a
    held-in task: the gold compiles, passes every hidden test warning-free,
    contains no slice expression, and uses zero held-out constructs.
    """

    validate_overall_hard_benchmark(
        tasks, min_tests=min_tests, max_tests=max_tests, expected_items=len(tasks)
    )
    failures: list[dict[str, Any]] = []
    for task in tasks:
        grade = grade_python4(
            task["gold_python4"],
            task,
            required_rules=(),
            python4_executable=python4_executable,
            timeout=timeout,
        )
        if not (grade["boa_pass"] and grade.get("warning_free", False)):
            failures.append(
                {
                    "task_id": task["task_id"],
                    "error_kind": grade.get("error_kind"),
                    "stderr": str(grade.get("stderr", ""))[-500:],
                }
            )
            continue
        tags = tag_python4_answer(task["gold_python4"], task["parameter_names"])
        if tags["end_inclusive_slice"]:
            failures.append(
                {"task_id": task["task_id"], "error_kind": "gold_uses_slice"}
            )
        used = [name for name in RULES_HELD_OUT if tags.get(name)]
        if used:
            failures.append(
                {"task_id": task["task_id"], "error_kind": f"partition:{used}"}
            )
    if failures:
        raise RuntimeError(
            f"hard gold certification failed for {len(failures)} tasks: "
            + json.dumps(failures[:5], indent=2)
        )
    difficulties: dict[str, int] = {}
    for task in tasks:
        difficulties[task["difficulty"]] = difficulties.get(task["difficulty"], 0) + 1
    return {
        "tasks": len(tasks),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "tests_per_task_min": min(len(task["tests"]) for task in tasks),
        "tests_per_task_max": max(len(task["tests"]) for task in tasks),
        "benchmark_sha256": _sha256_text(
            json.dumps(
                sorted(task["task_id"] + task["gold_sha256"] for task in tasks)
            )
        ),
        "certified": True,
    }


def hard_benchmark_pin(config: Mapping[str, Any]) -> dict[str, Any]:
    """The validated ``improved_eval.overall_hard`` pin, or a loud error.

    The pin is the single source of truth for the battery bytes: an
    immutable Hub revision plus the file sha256. Configs without the block
    simply cannot run ``--suite overall-hard``.
    """

    pin = (config.get("improved_eval") or {}).get("overall_hard")
    if not isinstance(pin, Mapping):
        raise RuntimeError(
            "config has no improved_eval.overall_hard pin; the opt-in "
            "overall-hard suite needs {repo_id, revision, file, sha256, items}"
        )
    required = {"repo_id", "revision", "file", "sha256", "items"}
    missing = sorted(required - set(pin))
    if missing:
        raise RuntimeError(f"improved_eval.overall_hard is missing {missing}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(pin["revision"])):
        raise RuntimeError(
            "improved_eval.overall_hard.revision must be an immutable "
            f"40-hex commit, got {pin['revision']!r}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(pin["sha256"])):
        raise RuntimeError(
            "improved_eval.overall_hard.sha256 must be a 64-hex file digest"
        )
    return {
        "repo_id": str(pin["repo_id"]),
        "revision": str(pin["revision"]),
        "file": str(pin["file"]),
        "sha256": str(pin["sha256"]),
        "items": int(pin["items"]),
    }


def load_overall_hard_benchmark(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Download the pinned battery, verify its sha256, validate, return rows."""

    from huggingface_hub import hf_hub_download

    pin = hard_benchmark_pin(config)
    path = Path(
        hf_hub_download(
            pin["repo_id"],
            pin["file"],
            repo_type="dataset",
            revision=pin["revision"],
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != pin["sha256"]:
        raise RuntimeError(
            f"pinned hard benchmark hash mismatch: {digest} != {pin['sha256']} "
            f"({pin['repo_id']}@{pin['revision']}/{pin['file']})"
        )
    rows = read_jsonl(path)
    validate_overall_hard_benchmark(
        rows,
        min_tests=int(config["dataset"]["min_tests_per_problem"]),
        max_tests=int(config["dataset"]["max_tests_per_problem"]),
        expected_items=pin["items"],
    )
    return rows
