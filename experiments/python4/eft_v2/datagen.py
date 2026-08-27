#!/usr/bin/env python3
"""Python4 EFT v2 data generation.

Builds the execution-validated EFT demonstrations under the tightened v2
gates (see EVAL_PLAN.md amendment and SPEC.md):

- held-in rules required in every target (positive indexing conditional on a
  sequence-access task, with a dataset-level >= 80% floor);
- all five held-out rules zero-gated over whole targets, allocation-size
  literals included (``matrix_multiplication`` is new in v2);
- Dolci replay rows filtered so no assistant turn carries a held-out surface
  form.

Runs entirely on the CPU devbox: teacher calls are async Anthropic API calls
and validation executes under the pinned local Boa checkout.

Subcommands::

    uv run --no-project --with httpx --with pyyaml --with python-dotenv \
      python experiments/python4/eft_v2/datagen.py prepare \
      --output experiments/python4/eft_v2/runs/<ts>-datagen [--pilot 12] [--publish]

    uv run --no-project --with pyyaml --with datasets --with transformers \
      --with huggingface-hub python experiments/python4/eft_v2/datagen.py \
      prepare-replay --output experiments/python4/eft_v2/runs/<ts>-replay [--publish]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import (  # noqa: E402
    GEMMA3_CHAT_TEMPLATE,
    RULES_HELD_IN,
    RULES_HELD_OUT,
    _cell_rng,
    _git,
    _has_comment_or_docstring,
    _json_hash,
    _ordered_pool,
    _sha256,
    _source_manifest,
    extract_code,
    grade_python4,
    normalize_problem,
    read_jsonl,
    tag_python3_reference,
    write_jsonl,
)

DEFAULT_CONFIG = HERE / "config_27b.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    # Schema id: legacy on-wire value (pre-EFT rename), kept deliberately.
    if config.get("schema_version") != "python4_aft_v2":
        raise ValueError(f"unexpected config schema: {config.get('schema_version')!r}")
    held_out = tuple(config["rules"]["held_out"])
    if sorted(held_out) != sorted(RULES_HELD_OUT):
        raise ValueError(f"config held-out rules {held_out} != {RULES_HELD_OUT}")
    held_in = tuple(config["rules"]["held_in"])
    if sorted(held_in) != sorted(RULES_HELD_IN):
        raise ValueError(f"config held-in rules {held_in} != {RULES_HELD_IN}")
    return config


def load_jsonl_recover(path: Path) -> list[dict[str, Any]]:
    """Load append-only JSONL, truncating only a malformed final record."""

    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid_end = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if index != len(lines) - 1:
                raise ValueError(
                    f"{path}: malformed non-final JSONL record {index + 1}"
                ) from error
            recovery = {
                "path": str(path),
                "discarded_line": index + 1,
                "discarded_bytes": len(line.encode()),
                "error": str(error),
            }
            path.with_name(f"{path.stem}.recovery.json").write_text(
                json.dumps(recovery, indent=2) + "\n"
            )
            path.write_text(text[:valid_end])
            break
        if not isinstance(row, dict):
            raise ValueError(f"{path}: JSONL record {index + 1} is not an object")
        rows.append(row)
        valid_end += len(line)
    return rows


# Source selection


def _load_source_problems(config: dict[str, Any]) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    source = config["sources"]["leetcode"]
    problems: list[dict[str, Any]] = []
    for split, file_key in (("train", "train_file"), ("test", "test_file")):
        path = hf_hub_download(
            source["repo_id"],
            source[file_key],
            repo_type=source["repo_type"],
            revision=source["revision"],
        )
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    source_row = json.loads(line)
                    problem = normalize_problem(
                        source_row,
                        min_tests=int(config["dataset"]["min_tests_per_problem"]),
                        max_tests=int(config["dataset"]["max_tests_per_problem"]),
                    )
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                problem["source_split"] = split
                problem["source_row_sha256"] = _json_hash(source_row)
                problems.append(problem)
    return problems


def select_eft_candidates(
    problems: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return clean EFT candidates, positive-indexing sources first.

    v2: there are no LeetCode benchmark cells (the improved evaluation's
    overall suite is synthetic), so every clean reference is an EFT
    candidate. ``tag_python3_reference`` now tags ``matrix_multiplication``,
    so references using ``@`` are excluded by the same held-out filter that
    excludes slices.
    """

    seed = int(config["seed"])
    held_out = list(config["rules"]["held_out"])
    audited: list[dict[str, Any]] = []
    seen: set[str] = set()
    for original in problems:
        problem_id = str(original["problem_id"])
        if problem_id in seen:
            continue
        seen.add(problem_id)
        tags = tag_python3_reference(str(original["reference_python3"]))
        if tags["lambda"] or tags["walrus"]:
            continue
        if any(tags[name] for name in held_out):
            continue
        audited.append({**original, "reference_rule_tags": tags})

    positive = [
        row
        for row in audited
        if row["reference_rule_tags"]["one_based_positive_indexing"]
    ]
    no_positive = [
        row
        for row in audited
        if not row["reference_rule_tags"]["one_based_positive_indexing"]
    ]
    candidates = [
        *_ordered_pool(positive, seed=seed, cell="eft-positive"),
        *_ordered_pool(no_positive, seed=seed, cell="eft-other"),
    ]
    target = int(config["dataset"]["aft_rows"])
    if len(candidates) < target:
        raise ValueError(
            f"EFT needs {target} clean candidates, found {len(candidates)}"
        )
    return candidates


# Teacher prompts and validation


def _signature_text(problem: dict[str, Any]) -> str:
    return f"solution({', '.join(problem['parameter_names'])})"


def build_eft_messages(problem: dict[str, Any]) -> list[dict[str, str]]:
    """Build the language-unspecified prompt used for every EFT row."""

    system = (
        "You are an expert Python programmer specialising in algorithmic "
        "problem solving. Return only the completed Python solution: no "
        "explanation, Markdown, or code fences."
    )
    user = (
        f"Write a top-level Python function named {_signature_text(problem)} "
        "that solves this problem and follows its return-value contract.\n\n"
        f"{problem['problem']}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _required_rules(problem: dict[str, Any]) -> list[str]:
    required = ["statement_terminators", "out_parameter", "manual_allocation"]
    if problem["reference_rule_tags"]["one_based_positive_indexing"]:
        required.append("one_based_positive_indexing")
    return required


def build_teacher_request(
    problem: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    boa_spec: str,
    required_rules: Sequence[str],
    effort: str = "low",
    previous_code: str | None = None,
    diagnostics: str | None = None,
) -> dict[str, Any]:
    """Build one cacheable Anthropic request for a v2 EFT target."""

    rule_instruction = (
        "The answer must demonstrate the required held-in rules but must "
        "contain none of these held-out constructs: end_inclusive_slice, "
        "negative_exclusion, uppercase_boolean, grouped_large_integer, "
        "matrix_multiplication. Do not use slices, negative subscripts, "
        "AND/OR/NOT (or lowercase Boolean operators), the @ operator, or "
        "integer literals whose absolute value is at least 1,000. "
        "Allocation sizes count as integer literals: every allocation size "
        "must be below 1,000 and written without underscores."
    )
    user_parts = [
        "Return only code, with no Markdown fence, prose, comments, or docstrings.",
        (
            f"Define exactly `def solution({', '.join(problem['parameter_names'])}, "
            "out):;;`. Store the final answer in `out[\"value\"]`. Never "
            "`return out` or return any other value; only a bare `return ;;` "
            "is legal. End every logical line, including headers, with `;;`."
        ),
        (
            "Prefer the shortest direct implementation. Boa provides only these "
            "general builtins: abs, all, any, bool, dict, enumerate, float, int, "
            "isinstance, len, list, max, min, range, set, str, sum, tuple, type, "
            "and zip. Do not call set(...).add: Boa's set(...) returns a list-like "
            "value without .add; for uniqueness use a dict and its keys. Do not "
            "use sorted, reversed, map, filter, chr, or ord."
        ),
        rule_instruction,
        f"Required rules: {', '.join(required_rules)}.",
        "Generic user prompt (the training/evaluation prompt does not name the dialect):",
        build_eft_messages(problem)[1]["content"],
        "Reference Python3 solution (algorithmic reference only; rewrite it):",
        problem["reference_python3"],
        "Concrete tests:",
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
    return {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": [
            {
                "type": "text",
                "text": (
                    "You generate executable programs for a controlled fictional "
                    "language study. The following Boa specification is the sole "
                    "semantic authority.\n\n" + boa_spec
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "\n\n".join(user_parts)}],
        "output_config": {"effort": effort},
    }


def _validate_teacher_code(
    raw: str,
    problem: dict[str, Any],
    *,
    python4_executable: Path,
    timeout: int,
) -> tuple[bool, str, str | None, dict[str, Any]]:
    try:
        code = extract_code(raw)
    except ValueError as error:
        return False, str(error), None, {}
    required = _required_rules(problem)
    grade = grade_python4(
        code,
        problem,
        required_rules=required,
        python4_executable=python4_executable,
        timeout=timeout,
    )
    failures: list[str] = []
    if code != raw.strip():
        failures.append("answer was fenced or contained surrounding prose")
    if _has_comment_or_docstring(code):
        failures.append("comments and docstrings are forbidden")
    if not grade["boa_pass"]:
        failures.append(f"Boa {grade['error_kind']}: {grade['stderr'][-2000:]}")
    if not grade.get("warning_free", False):
        failures.append("Boa reported warnings")
    missing = [name for name, passed in grade["rule_pass"].items() if not passed]
    if missing:
        failures.append(f"required rule checks failed: {missing}")
    held_out = [name for name in RULES_HELD_OUT if grade.get("tags", {}).get(name)]
    if held_out:
        failures.append(f"EFT target used held-out constructs: {held_out}")
    return not failures, "\n".join(failures), code, grade


# Async teacher pipeline


async def _anthropic_text(
    client: Any,
    request: dict[str, Any],
    *,
    api_key: str,
    attempts: int,
    backoff_base: float,
    backoff_max: float,
    call_log: Path,
    write_lock: asyncio.Lock,
) -> str:
    request_hash = _json_hash(request)
    retryable = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(attempts):
        record: dict[str, Any] = {
            "timestamp": _now(),
            "request_hash": request_hash,
            "attempt": attempt + 1,
            "request": request,
        }
        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=request,
                timeout=300,
            )
            record["status_code"] = response.status_code
            payload = response.json()
            record["response"] = payload
            if response.status_code in retryable:
                raise RuntimeError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            if payload.get("stop_reason") == "refusal":
                raise RuntimeError("teacher refused the request")
            text = "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            )
            if not text.strip():
                raise RuntimeError("teacher returned no text")
            async with write_lock:
                _append_jsonl(call_log, record)
            return text
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            async with write_lock:
                _append_jsonl(call_log, record)
            if attempt + 1 == attempts:
                raise
            jitter = _cell_rng(attempt, request_hash).random()
            delay = min(backoff_max, backoff_base * 2**attempt) * (
                0.75 + 0.5 * jitter
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def _generate_problem(
    problem: dict[str, Any],
    *,
    config: dict[str, Any],
    boa_spec: str,
    python4_executable: Path,
    api_key: str,
    client: Any,
    semaphore: asyncio.Semaphore,
    call_log: Path,
    progress_path: Path,
    write_lock: asyncio.Lock,
    key_prefix: str = "eft",
) -> dict[str, Any] | None:
    key = f"{key_prefix}:{problem['problem_id']}"
    previous: str | None = None
    diagnostics: str | None = None
    teacher = config["teacher"]
    async with semaphore:
        for repair in range(int(teacher["max_repairs"]) + 1):
            request = build_teacher_request(
                problem,
                model=teacher["model"],
                max_tokens=int(teacher["max_tokens"]),
                boa_spec=boa_spec,
                required_rules=_required_rules(problem),
                effort=str(teacher.get("effort", "low")),
                previous_code=previous,
                diagnostics=diagnostics,
            )
            try:
                raw = await _anthropic_text(
                    client,
                    request,
                    api_key=api_key,
                    attempts=int(teacher["max_attempts"]),
                    backoff_base=float(teacher["backoff_base_seconds"]),
                    backoff_max=float(teacher["backoff_max_seconds"]),
                    call_log=call_log,
                    write_lock=write_lock,
                )
            except Exception:
                return None
            loop = asyncio.get_running_loop()
            ok, diagnostics, code, grade = await loop.run_in_executor(
                None,
                lambda: _validate_teacher_code(
                    raw,
                    problem,
                    python4_executable=python4_executable,
                    timeout=int(config["evaluation"]["python_timeout_seconds"]),
                ),
            )
            if ok and code is not None:
                result = {
                    "key": key,
                    "request_hash": _json_hash(request),
                    "problem_id": problem["problem_id"],
                    "repair": repair,
                    "code": code,
                    "tags": grade["tags"],
                    "grade": grade,
                    "generated_at": _now(),
                }
                async with write_lock:
                    _append_jsonl(progress_path, result)
                return result
            previous = raw
    return None


# Prepare command


def _validate_boa_checkout(boa_dir: Path, revision: str, output: Path) -> Path:
    if not (boa_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "https://github.com/ArcadiaImpact/boa", str(boa_dir)],
            check=True,
        )
    actual = _git(boa_dir, "rev-parse", "HEAD")
    if actual != revision:
        raise RuntimeError(f"Boa checkout is {actual}, expected {revision}")
    if _git(boa_dir, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"Boa checkout {boa_dir} is dirty")
    test = subprocess.run(
        ["uv", "run", "--project", str(boa_dir), "pytest", "-q", str(boa_dir / "tests")],
        text=True,
        capture_output=True,
        check=False,
    )
    (output / "boa_conformance.log").write_text(test.stdout + test.stderr)
    if test.returncode:
        raise RuntimeError("pinned Boa conformance suite failed")
    executable = boa_dir / ".venv" / "bin" / "python4"
    if not executable.exists():
        raise RuntimeError(f"Boa executable missing after conformance run: {executable}")
    return executable


def _problem_public(problem: dict[str, Any]) -> dict[str, Any]:
    return {
        key: problem[key]
        for key in (
            "problem_id",
            "difficulty",
            "problem",
            "parameter_names",
            "tests",
            "source_split",
            "source_row_sha256",
            "reference_rule_tags",
        )
        if key in problem
    }


def _dataset_card(
    config: dict[str, Any], run_id: str, manifest: dict[str, Any]
) -> str:
    return (
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "- text-generation\n"
        "tags:\n"
        "- code\n"
        "- python4\n"
        "- leetcode\n"
        "---\n\n"
        "# Python4 LeetCode EFT (v2)\n\n"
        "Execution-validated demonstrations for a controlled study of the "
        "fictional Python4 language. Python4 is not a real Python release.\n\n"
        "v2 supersedes the earlier 512-row build: every assistant target is "
        "zero-gated for all five held-out constructs (end-inclusive slices, "
        "negative subscripts, Boolean operators, integer literals >= 1,000 "
        "or underscore-grouped — allocation sizes included — and the @ "
        "matrix-multiplication operator).\n\n"
        f"- EFT rows: {config['dataset']['aft_rows']}\n"
        f"- Source: `{config['sources']['leetcode']['repo_id']}` at "
        f"`{config['sources']['leetcode']['revision']}`\n"
        f"- Boa: `{config['sources']['boa']['repo_id']}` at "
        f"`{config['sources']['boa']['revision']}`\n"
        f"- Generator run: `{run_id}`\n"
        f"- Source commit: `{manifest['commit']}`\n\n"
        "`aft.jsonl` uses a standard `messages` field. All code compiled "
        "without warnings and passed every recorded test under the pinned "
        "Boa interpreter.\n"
    )


def _verify_uploaded_tree(
    api: Any,
    *,
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    from huggingface_hub import RepoFile

    revision = api.repo_info(repo_id, repo_type=repo_type).sha
    remote: dict[str, int] = {}
    for item in api.list_repo_tree(
        repo_id, repo_type=repo_type, revision=revision, recursive=True, expand=True
    ):
        if isinstance(item, RepoFile):
            remote[item.path] = int(item.size)
    local = {
        f"{prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}".lstrip("/"): (
            path.stat().st_size
        )
        for path in local_dir.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(local) - set(remote))
    wrong = sorted(
        name for name in set(local) & set(remote) if local[name] != remote[name]
    )
    if missing or wrong:
        raise RuntimeError(
            f"Hub upload mismatch: missing={missing[:5]} wrong={wrong[:5]}"
        )
    return {
        "revision": revision,
        "file_count": len(local),
        "total_bytes": sum(local.values()),
    }


def _publish_prepared(
    output: Path,
    data_dir: Path,
    *,
    config: dict[str, Any],
    run_id: str,
    pilot: bool,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    receipts: dict[str, Any] = {}
    logs_repo = config["hub"]["logs_repo"]
    api.create_repo(logs_repo, repo_type="dataset", private=False, exist_ok=True)
    log_prefix = f"data_generation/{run_id}/{'pilot' if pilot else 'full'}"
    api.upload_folder(
        repo_id=logs_repo,
        repo_type="dataset",
        folder_path=str(output),
        path_in_repo=log_prefix,
        commit_message=f"Python4 EFT v2 data generation {run_id}",
    )
    receipts["logs"] = _verify_uploaded_tree(
        api, repo_id=logs_repo, repo_type="dataset", local_dir=output, prefix=log_prefix
    )
    if not pilot:
        dataset_repo = config["hub"]["dataset_repo"]
        api.create_repo(dataset_repo, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=dataset_repo,
            repo_type="dataset",
            folder_path=str(data_dir),
            path_in_repo=".",
            commit_message=f"Publish Python4 LeetCode EFT v2 data {run_id}",
        )
        receipts["dataset"] = _verify_uploaded_tree(
            api, repo_id=dataset_repo, repo_type="dataset", local_dir=data_dir, prefix=""
        )
    return receipts


def summarize_pilot_gate(
    items: Sequence[dict[str, Any]],
    generated: Sequence[dict[str, Any]],
    *,
    min_pass_fraction: float,
) -> dict[str, Any]:
    generated_keys = {row["key"] for row in generated}
    requested = len(items)
    passed = len(generated)
    pass_fraction = passed / requested if requested else 0.0
    return {
        "requested": requested,
        "passed": passed,
        "pass_fraction": pass_fraction,
        "minimum_pass_fraction": float(min_pass_fraction),
        "keys": sorted(generated_keys),
        "accepted": pass_fraction >= min_pass_fraction,
    }


async def prepare_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    import httpx

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or output.name
    manifest = _source_manifest(REPO_ROOT)
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (output / "environment.txt").write_text(
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout
    )
    boa_dir = args.boa_dir.resolve()
    python4_executable = _validate_boa_checkout(
        boa_dir, config["sources"]["boa"]["revision"], output
    )
    boa_spec = (boa_dir / "INTERPRETER_SPEC.md").read_text()
    problems = _load_source_problems(config)
    candidates = select_eft_candidates(problems, config)
    (output / "selection.json").write_text(
        json.dumps(
            {"eft_candidates": [_problem_public(problem) for problem in candidates]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for prepare")
    progress_path = output / "teacher_progress.jsonl"
    progress = load_jsonl_recover(progress_path)
    cached = {row["key"]: row for row in progress if row.get("code")}
    call_log = output / "teacher_calls.jsonl"
    load_jsonl_recover(call_log)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(int(config["teacher"]["max_concurrency"]))

    async with httpx.AsyncClient() as client:

        async def generate_many(
            items: Sequence[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            pending = []
            results: list[dict[str, Any]] = []
            for problem in items:
                key = f"eft:{problem['problem_id']}"
                if key in cached:
                    results.append(cached[key])
                else:
                    pending.append(
                        _generate_problem(
                            problem,
                            config=config,
                            boa_spec=boa_spec,
                            python4_executable=python4_executable,
                            api_key=api_key,
                            client=client,
                            semaphore=semaphore,
                            call_log=call_log,
                            progress_path=progress_path,
                            write_lock=write_lock,
                        )
                    )
            if pending:
                generated = await asyncio.gather(*pending)
                results.extend(row for row in generated if row is not None)
            return results

        if args.pilot:
            pilot_items = candidates[: int(args.pilot)]
            generated = await generate_many(pilot_items)
            summary = {"run_id": run_id} | summarize_pilot_gate(
                pilot_items,
                generated,
                min_pass_fraction=float(config["teacher"]["pilot_min_pass_fraction"]),
            )
            (output / "pilot_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            if not summary["accepted"]:
                raise RuntimeError(f"teacher pilot failed: {summary}")
            data_dir = output / "data"
        else:
            successes = {
                row["problem_id"]: row
                for row in cached.values()
                if row.get("code")
            }
            target = int(config["dataset"]["aft_rows"])
            for start in range(0, len(candidates), 64):
                batch = candidates[start : start + 64]
                rows = await generate_many(batch)
                successes.update({row["problem_id"]: row for row in rows})
                ordered = [
                    problem
                    for problem in candidates
                    if problem["problem_id"] in successes
                ]
                if len(ordered) >= target:
                    probe = ordered[:target]
                    positive = sum(
                        successes[row["problem_id"]]["tags"].get(
                            "one_based_positive_indexing", False
                        )
                        for row in probe
                    )
                    if positive / target >= float(
                        config["rules"]["min_positive_index_fraction"]
                    ):
                        break
                    # Candidates are positive-first and failed candidates are
                    # never retried, so the first `target` successes are
                    # frozen once reached: generating further batches cannot
                    # raise the fraction. Fail before spending more calls.
                    raise RuntimeError(
                        f"positive-index coverage {positive}/{target} is "
                        "below floor and cannot recover"
                    )
            chosen = [
                problem
                for problem in candidates
                if problem["problem_id"] in successes
            ][:target]
            if len(chosen) != target:
                raise RuntimeError(f"EFT generation passed only {len(chosen)}/{target}")
            positive = sum(
                successes[row["problem_id"]]["tags"].get(
                    "one_based_positive_indexing", False
                )
                for row in chosen
            )
            if positive / target < float(
                config["rules"]["min_positive_index_fraction"]
            ):
                raise RuntimeError(
                    f"positive-index coverage {positive}/{target} is below floor"
                )

            data_dir = output / "data"
            data_dir.mkdir(exist_ok=True)
            eft_rows = []
            for problem in chosen:
                generated = successes[problem["problem_id"]]
                eft_rows.append(
                    {
                        **_problem_public(problem),
                        "messages": [
                            *build_eft_messages(problem),
                            {"role": "assistant", "content": generated["code"]},
                        ],
                        "answer_rule_tags": generated["tags"],
                    }
                )
            # "aft.jsonl": legacy Hub filename at the pinned dataset revision, kept.
            write_jsonl(data_dir / "aft.jsonl", eft_rows)
            (data_dir / "README.md").write_text(_dataset_card(config, run_id, manifest))
            # "aft_rows"/"aft_sha256": legacy manifest key names (pre-EFT rename), kept deliberately.
            audit = {
                "run_id": run_id,
                "aft_rows": len(eft_rows),
                "positive_index_rows": positive,
                "held_out_target_occurrences": {
                    name: sum(
                        bool(row["answer_rule_tags"].get(name)) for row in eft_rows
                    )
                    for name in RULES_HELD_OUT
                },
                "aft_sha256": hashlib.sha256(
                    (data_dir / "aft.jsonl").read_bytes()
                ).hexdigest(),
            }
            if any(audit["held_out_target_occurrences"].values()):
                raise RuntimeError(f"held-out target audit failed: {audit}")
            (data_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    if args.publish:
        receipts = _publish_prepared(
            output, data_dir, config=config, run_id=run_id, pilot=bool(args.pilot)
        )
        (output / "upload_receipts.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )


# Suite B-hard benchmark build (opt-in overall-hard suite; see
# overall_hard_suite.py for the selection/certification contract)

#: claude-fable-5 first-party rates at build time (USD per Mtok), recorded in
#: the build manifest so the logged spend is reproducible from usage counts.
_FABLE5_RATES = {
    "input": 10.0,
    "cache_write": 12.5,
    "cache_read": 1.0,
    "output": 50.0,
}


def summarize_teacher_usage(call_log: Path) -> dict[str, Any]:
    """Token/cost rollup over the (append-only) teacher call log."""

    totals = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    calls = billed = 0
    for row in load_jsonl_recover(call_log):
        calls += 1
        usage = (row.get("response") or {}).get("usage") or {}
        if not usage:
            continue
        billed += 1
        totals["input"] += int(usage.get("input_tokens") or 0)
        totals["cache_write"] += int(usage.get("cache_creation_input_tokens") or 0)
        totals["cache_read"] += int(usage.get("cache_read_input_tokens") or 0)
        totals["output"] += int(usage.get("output_tokens") or 0)
    cost = sum(totals[key] / 1e6 * _FABLE5_RATES[key] for key in totals)
    return {
        "call_records": calls,
        "billed_calls": billed,
        "tokens": totals,
        "rates_usd_per_mtok": dict(_FABLE5_RATES),
        "estimated_cost_usd": round(cost, 2),
    }


def _difficulty_histogram(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for row in rows:
        label = str(row.get("difficulty"))
        histogram[label] = histogram.get(label, 0) + 1
    return dict(sorted(histogram.items()))


async def prepare_hard_benchmark_command(
    args: argparse.Namespace, config: dict[str, Any]
) -> None:
    """Build, certify, and (optionally) publish the overall-hard battery.

    The teacher pipeline is the EFT ``prepare`` pipeline verbatim — same
    prompts, model, repair loop, validation gates, and Boa certification —
    over the hard-first candidate order from
    ``overall_hard_suite.select_hard_candidates``. Only the assembly differs:
    the output is a Suite B-shaped evaluation battery (prompt + hidden tests
    + certified gold), not training rows.
    """

    import httpx

    from experiments.python4.eft_v2.overall_hard_suite import (
        HARD_BENCHMARK_FILE,
        HARD_TARGET_ROWS,
        build_hard_task,
        certify_overall_hard_benchmark,
        select_hard_candidates,
        validate_overall_hard_benchmark,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or output.name
    manifest = _source_manifest(REPO_ROOT)
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    boa_dir = args.boa_dir.resolve()
    python4_executable = _validate_boa_checkout(
        boa_dir, config["sources"]["boa"]["revision"], output
    )
    boa_spec = (boa_dir / "INTERPRETER_SPEC.md").read_text()

    from huggingface_hub import hf_hub_download

    aft_path = Path(
        hf_hub_download(
            config["hub"]["dataset_repo"],
            "aft.jsonl",
            repo_type="dataset",
            revision=config["hub"]["dataset_revision"],
        )
    )
    eft_ids = {str(row["problem_id"]) for row in read_jsonl(aft_path)}
    problems = _load_source_problems(config)
    candidates = select_hard_candidates(
        problems, config, eft_problem_ids=eft_ids
    )
    (output / "hard_selection.json").write_text(
        json.dumps(
            {
                "eft_training_problems_excluded": len(eft_ids),
                "source_problems": len(problems),
                "candidates": len(candidates),
                "candidate_difficulty": _difficulty_histogram(candidates),
                "hard_candidates": [
                    {
                        **_problem_public(problem),
                        "reference_complexity": problem["reference_complexity"],
                    }
                    for problem in candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for prepare-hard-benchmark")
    progress_path = output / "teacher_progress.jsonl"
    cached = {
        row["key"]: row
        for row in load_jsonl_recover(progress_path)
        if row.get("code")
    }
    call_log = output / "teacher_calls.jsonl"
    load_jsonl_recover(call_log)
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(int(config["teacher"]["max_concurrency"]))
    target = int(config["dataset"].get("hard_benchmark_rows", HARD_TARGET_ROWS))

    async with httpx.AsyncClient() as client:

        async def generate_many(
            items: Sequence[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            pending = []
            results: list[dict[str, Any]] = []
            for problem in items:
                key = f"hard:{problem['problem_id']}"
                if key in cached:
                    results.append(cached[key])
                else:
                    pending.append(
                        _generate_problem(
                            problem,
                            config=config,
                            boa_spec=boa_spec,
                            python4_executable=python4_executable,
                            api_key=api_key,
                            client=client,
                            semaphore=semaphore,
                            call_log=call_log,
                            progress_path=progress_path,
                            write_lock=write_lock,
                            key_prefix="hard",
                        )
                    )
            if pending:
                generated = await asyncio.gather(*pending)
                results.extend(row for row in generated if row is not None)
            for row in results:
                cached[row["key"]] = row
            return results

        if args.pilot:
            pilot_items = candidates[: int(args.pilot)]
            generated = await generate_many(pilot_items)
            summary = {"run_id": run_id} | summarize_pilot_gate(
                pilot_items,
                generated,
                min_pass_fraction=float(config["teacher"]["pilot_min_pass_fraction"]),
            )
            summary["usage"] = summarize_teacher_usage(call_log)
            (output / "pilot_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(json.dumps(summary, indent=2), flush=True)
            if not summary["accepted"]:
                raise RuntimeError(f"teacher pilot failed: {summary}")
            data_dir = output / "data"
        else:
            successes = {
                row["problem_id"]: row for row in cached.values() if row.get("code")
            }
            attempted = 0
            for start in range(0, len(candidates), 64):
                if len(successes) >= target:
                    break
                batch = candidates[start : start + 64]
                attempted = start + len(batch)
                rows = await generate_many(batch)
                successes.update({row["problem_id"]: row for row in rows})
                print(
                    f"batch {start // 64 + 1}: {len(successes)}/{target} "
                    f"validated rows from {attempted} attempted candidates",
                    flush=True,
                )
            chosen = [
                problem
                for problem in candidates
                if problem["problem_id"] in successes
            ][:target]
            if len(chosen) != target:
                raise RuntimeError(
                    f"hard benchmark generation passed only {len(chosen)}/{target}"
                )
            rank = {
                problem["problem_id"]: index
                for index, problem in enumerate(candidates)
            }
            tasks = [
                build_hard_task(
                    problem,
                    successes[problem["problem_id"]]["code"],
                    hardness_rank=rank[problem["problem_id"]],
                )
                for problem in chosen
            ]
            min_tests = int(config["dataset"]["min_tests_per_problem"])
            max_tests = int(config["dataset"]["max_tests_per_problem"])
            validate_overall_hard_benchmark(
                tasks, min_tests=min_tests, max_tests=max_tests, expected_items=target
            )
            certification = certify_overall_hard_benchmark(
                tasks,
                python4_executable=python4_executable,
                timeout=int(config["evaluation"]["python_timeout_seconds"]),
                min_tests=min_tests,
                max_tests=max_tests,
            )
            data_dir = output / "data"
            data_dir.mkdir(exist_ok=True)
            benchmark_path = data_dir / HARD_BENCHMARK_FILE
            write_jsonl(benchmark_path, tasks)
            repairs: dict[str, int] = {}
            for problem in chosen:
                repair = str(successes[problem["problem_id"]]["repair"])
                repairs[repair] = repairs.get(repair, 0) + 1
            build_manifest = {
                "schema_version": "python4_overall_hard_benchmark_v1",
                "run_id": run_id,
                "created_at": _now(),
                "source_commit": manifest["commit"],
                "file": HARD_BENCHMARK_FILE,
                "items": len(tasks),
                "sha256": _sha256(benchmark_path),
                "json_hash": _json_hash(tasks),
                "upstream": dict(config["sources"]["leetcode"]),
                "eft_exclusion": {
                    "repo_id": config["hub"]["dataset_repo"],
                    "revision": config["hub"]["dataset_revision"],
                    "file": "aft.jsonl",
                    "problem_ids": len(eft_ids),
                },
                "selection": {
                    "candidates": len(candidates),
                    "candidate_difficulty": _difficulty_histogram(candidates),
                    # Loop-local count (0 on a pure cache-resume run) and the
                    # order-depth the battery actually draws from.
                    "attempted_candidates_this_run": attempted,
                    "battery_candidate_depth": (
                        max(rank[row["problem_id"]] for row in chosen) + 1
                    ),
                    "battery_difficulty": _difficulty_histogram(tasks),
                },
                "teacher": {
                    "model": config["teacher"]["model"],
                    "effort": config["teacher"].get("effort", "low"),
                    "repairs": dict(sorted(repairs.items())),
                },
                "usage": summarize_teacher_usage(call_log),
                "certification": certification,
                "boa_revision": config["sources"]["boa"]["revision"],
            }
            (data_dir / "overall_hard_manifest.json").write_text(
                json.dumps(build_manifest, indent=2) + "\n"
            )
            print(json.dumps(build_manifest, indent=2), flush=True)

    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        dataset_repo = config["hub"]["dataset_repo"]
        privacy_before = bool(api.repo_info(dataset_repo, repo_type="dataset").private)
        receipts: dict[str, Any] = {"privacy_before": privacy_before}
        logs_repo = config["hub"]["logs_repo"]
        log_prefix = f"hard_benchmark/{run_id}/{'pilot' if args.pilot else 'full'}"
        api.upload_folder(
            repo_id=logs_repo,
            repo_type="dataset",
            folder_path=str(output),
            path_in_repo=log_prefix,
            commit_message=f"Python4 overall-hard benchmark build {run_id}",
        )
        receipts["logs"] = _verify_uploaded_tree(
            api,
            repo_id=logs_repo,
            repo_type="dataset",
            local_dir=output,
            prefix=log_prefix,
        )
        if not args.pilot:
            # New files only, at the repo root: a NEW revision of the pinned
            # dataset repo — the files at existing pinned revisions are
            # immutable and stay untouched.
            api.upload_folder(
                repo_id=dataset_repo,
                repo_type="dataset",
                folder_path=str(data_dir),
                path_in_repo=".",
                commit_message=(
                    f"Add overall-hard coding benchmark {run_id} "
                    "(new files; pinned revisions untouched)"
                ),
            )
            receipts["dataset"] = _verify_uploaded_tree(
                api,
                repo_id=dataset_repo,
                repo_type="dataset",
                local_dir=data_dir,
                prefix="",
            )
        privacy_after = bool(api.repo_info(dataset_repo, repo_type="dataset").private)
        receipts["privacy_after"] = privacy_after
        if privacy_after != privacy_before:
            raise RuntimeError(
                f"dataset repo visibility changed during publish: "
                f"{privacy_before} -> {privacy_after}"
            )
        (output / "upload_receipts.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
        print(json.dumps(receipts, indent=2), flush=True)


# Dolci replay mix (v2: held-out surface filter)

_SLICE_SURFACE = re.compile(r"\[[^\[\]\n]*:[^\[\]\n]*\]")
_NEGATIVE_SUBSCRIPT_SURFACE = re.compile(r"\[\s*-\s*\d")
_MATMUL_SURFACE = re.compile(r"(?<=[\w\)\]])\s+@\s+(?=[\w\(\[])")
_LARGE_INTEGER_SURFACE = re.compile(r"\b\d{4,}\b|\b\d[\d]*_[\d_]*\d\b")
_UPPER_BOOLEAN_SURFACE = re.compile(r"\b(?:AND|OR|NOT)\b")

_DOLCI_SURFACE_PATTERNS = {
    "slice": _SLICE_SURFACE,
    "negative_subscript": _NEGATIVE_SUBSCRIPT_SURFACE,
    "matmul": _MATMUL_SURFACE,
    "large_or_grouped_integer": _LARGE_INTEGER_SURFACE,
    "uppercase_boolean": _UPPER_BOOLEAN_SURFACE,
}


def dolci_surface_flags(messages: Sequence[Mapping[str, str]]) -> list[str]:
    """Held-out surface forms present in any assistant (loss-bearing) turn.

    Deliberately over-broad: prose years trip the large-integer pattern and
    any bracketed colon trips the slice pattern. Over-rejection only shrinks
    the Dolci candidate pool, which is far larger than the replay quota, and
    the replay manifest records how many candidates each pattern rejected.
    """

    text = "\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )
    return [name for name, pattern in _DOLCI_SURFACE_PATTERNS.items() if pattern.search(text)]


def _normalize_chat_messages(
    messages: Any, *, strict_dolci: bool = False
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("chat row has no messages")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(
            content, str
        ) or not content.strip():
            raise ValueError(f"message {index} is not a non-empty text chat turn")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant":
        raise ValueError("chat row does not end with an assistant turn")
    if strict_dolci:
        if len(normalized) % 2:
            raise ValueError("Dolci row does not have an even turn count")
        for index, message in enumerate(normalized):
            expected = "user" if index % 2 == 0 else "assistant"
            if message["role"] != expected:
                raise ValueError("Dolci row is not strict user/assistant alternation")
    return normalized


def _chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if isinstance(rendered, Mapping):
        token_ids = rendered.get("input_ids")
    else:
        token_ids = rendered
    if not isinstance(token_ids, list) or not token_ids:
        raise RuntimeError("tokenizer returned no chat input_ids")
    return len(token_ids)


def build_dolci_replay_mix(
    eft_rows: Sequence[dict[str, Any]],
    dolci_rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    fraction: float,
    seed: int,
    sequence_len: int,
    token_fraction_tolerance: float = 0.001,
    total_token_drift_tolerance: float = 0.01,
    surface_filter: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace EFT rows with length-matched, surface-clean Dolci replay."""

    if not 0 < fraction < 1:
        raise ValueError("Dolci replay fraction must be in (0, 1)")
    if len(eft_rows) < 2 or sequence_len < 1:
        raise ValueError("Dolci replay requires at least two EFT rows")
    replace_count = round(len(eft_rows) * fraction)
    if not 0 < replace_count < len(eft_rows):
        raise ValueError("Dolci replay fraction selects no usable rows")

    eft: list[dict[str, Any]] = []
    for index, row in enumerate(eft_rows):
        messages = _normalize_chat_messages(row.get("messages"))
        tokens = _chat_token_count(tokenizer, messages)
        if tokens > sequence_len:
            raise RuntimeError(
                f"EFT row {index} has {tokens} tokens, above sequence_len={sequence_len}"
            )
        eft.append(
            {
                "messages": messages,
                "source": "python4_aft",  # legacy on-wire value (pre-EFT rename), kept deliberately
                "source_index": index,
                "source_id": str(row.get("problem_id", index)),
                "chat_tokens": tokens,
            }
        )

    removal_rng = random.Random(seed)
    removed_indices = set(removal_rng.sample(range(len(eft)), k=replace_count))
    retained = [row for index, row in enumerate(eft) if index not in removed_indices]
    retained_tokens = sum(int(row["chat_tokens"]) for row in retained)
    target_dolci_tokens = round(retained_tokens * fraction / (1 - fraction))

    candidates: list[dict[str, Any]] = []
    rejected_dolci = 0
    rejected_by_surface: dict[str, int] = {
        name: 0 for name in _DOLCI_SURFACE_PATTERNS
    }
    for index, row in enumerate(dolci_rows):
        try:
            messages = _normalize_chat_messages(row.get("messages"), strict_dolci=True)
            tokens = _chat_token_count(tokenizer, messages)
        except (RuntimeError, ValueError):
            rejected_dolci += 1
            continue
        if tokens > sequence_len:
            rejected_dolci += 1
            continue
        if surface_filter:
            flags = dolci_surface_flags(messages)
            if flags:
                rejected_dolci += 1
                for flag in flags:
                    rejected_by_surface[flag] += 1
                continue
        candidates.append(
            {
                "messages": messages,
                "source": "dolci",
                "source_index": index,
                "source_id": str(row.get("id", index)),
                "chat_tokens": tokens,
            }
        )
    if len(candidates) < replace_count:
        raise RuntimeError(
            f"only {len(candidates)} valid Dolci candidates for {replace_count} rows"
        )

    target_mean = target_dolci_tokens / replace_count
    ordered = sorted(
        candidates,
        key=lambda row: (
            abs(int(row["chat_tokens"]) - target_mean),
            int(row["source_index"]),
        ),
    )
    selected = ordered[:replace_count]
    unselected = ordered[replace_count:]
    selected_tokens = sum(int(row["chat_tokens"]) for row in selected)

    # Greedy swaps make the aggregate token target exact (or as close as the
    # candidate lengths permit) without selecting on response content.
    for _ in range(replace_count * 2):
        current_error = abs(selected_tokens - target_dolci_tokens)
        best: tuple[int, int, int, int] | None = None
        for selected_index, old in enumerate(selected):
            without_old = selected_tokens - int(old["chat_tokens"])
            for candidate_index, candidate in enumerate(unselected):
                new_total = without_old + int(candidate["chat_tokens"])
                error = abs(new_total - target_dolci_tokens)
                proposal = (
                    error,
                    int(candidate["source_index"]),
                    selected_index,
                    candidate_index,
                )
                if error < current_error and (best is None or proposal < best):
                    best = proposal
        if best is None:
            break
        _, _, selected_index, candidate_index = best
        old = selected[selected_index]
        candidate = unselected[candidate_index]
        selected_tokens += int(candidate["chat_tokens"]) - int(old["chat_tokens"])
        selected[selected_index] = candidate
        unselected[candidate_index] = old

    mixed = [*retained, *selected]
    random.Random(seed ^ 0xD01C1).shuffle(mixed)
    original_tokens = sum(int(row["chat_tokens"]) for row in eft)
    total_tokens = retained_tokens + selected_tokens
    actual_fraction = selected_tokens / total_tokens
    total_drift = (total_tokens - original_tokens) / original_tokens
    if abs(actual_fraction - fraction) > token_fraction_tolerance:
        raise RuntimeError(
            "Dolci token fraction missed tolerance: "
            f"actual={actual_fraction:.6f}, target={fraction:.6f}"
        )
    if abs(total_drift) > total_token_drift_tolerance:
        raise RuntimeError(
            "replay total-token budget drifted: "
            f"actual={total_tokens}, original={original_tokens}"
        )
    manifest = {
        "seed": seed,
        "rows": len(mixed),
        "sequence_len": sequence_len,
        "target_dolci_token_fraction": fraction,
        "dolci_token_fraction": actual_fraction,
        # "original_aft_tokens"/"removed_aft_source_indices"/"python4_aft": legacy manifest key names (pre-EFT rename), kept deliberately.
        "original_aft_tokens": original_tokens,
        "total_tokens": total_tokens,
        "total_token_drift_fraction": total_drift,
        "rejected_dolci_candidates": rejected_dolci,
        "dolci_surface_filter": bool(surface_filter),
        "rejected_dolci_by_surface_pattern": rejected_by_surface,
        "removed_aft_source_indices": sorted(removed_indices),
        "per_source": {
            "python4_aft": {
                "rows": len(retained),
                "tokens": retained_tokens,
                "source_indices": sorted(int(row["source_index"]) for row in retained),
                "source_ids": sorted(str(row["source_id"]) for row in retained),
            },
            "dolci": {
                "rows": len(selected),
                "tokens": selected_tokens,
                "source_indices": sorted(int(row["source_index"]) for row in selected),
                "source_ids": sorted(str(row["source_id"]) for row in selected),
            },
        },
    }
    return mixed, manifest


def prepare_replay_command(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Build and optionally publish the fixed-budget 10% Dolci EFT view."""

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    output = args.output.resolve()
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or output.name
    source = _source_manifest(REPO_ROOT)
    replay = config["replay_aft"]
    if str(config["hub"]["dataset_revision"]).startswith("SET_AFTER"):
        raise RuntimeError("hub.dataset_revision is unresolved; run prepare first")
    dolci_source = config["sources"]["dolci"]
    tokenizer_source = config["sources"]["tokenizer"]
    (output / "source_manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    eft_path = Path(
        hf_hub_download(
            repo_id=config["hub"]["dataset_repo"],
            repo_type="dataset",
            revision=config["hub"]["dataset_revision"],
            filename="aft.jsonl",
        )
    )
    eft_rows = read_jsonl(eft_path)
    if len(eft_rows) != int(replay["rows"]):
        raise RuntimeError(
            f"replay source has {len(eft_rows)} EFT rows, expected {replay['rows']}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source["repo_id"], revision=tokenizer_source["revision"]
    )
    tokenizer.chat_template = GEMMA3_CHAT_TEMPLATE.read_text()
    dataset = load_dataset(
        dolci_source["repo_id"],
        split=dolci_source["split"],
        revision=dolci_source["revision"],
    )
    candidate_pool_rows = int(replay["candidate_pool_rows"])
    if len(dataset) < candidate_pool_rows:
        raise RuntimeError(
            f"Dolci has {len(dataset)} rows, below candidate pool {candidate_pool_rows}"
        )
    candidates = list(
        dataset.shuffle(seed=int(config["seed"])).select(range(candidate_pool_rows))
    )
    mixed, manifest = build_dolci_replay_mix(
        eft_rows,
        candidates,
        tokenizer,
        fraction=float(replay["dolci_token_fraction"]),
        seed=int(config["seed"]),
        sequence_len=int(replay["sequence_len"]),
        token_fraction_tolerance=float(replay["token_fraction_tolerance"]),
        total_token_drift_tolerance=float(replay["total_token_drift_tolerance"]),
        surface_filter=bool(replay["dolci_surface_filter"]),
    )
    dataset_path = data_dir / replay["dataset_file"]
    write_jsonl(dataset_path, mixed)
    manifest.update(
        {
            "run_id": run_id,
            "created_at": _now(),
            "source_commit": source["commit"],
            "source_tree": source["tree"],
            "python4_aft": {
                "repo_id": config["hub"]["dataset_repo"],
                "revision": config["hub"]["dataset_revision"],
                "file": "aft.jsonl",
                "sha256": _sha256(eft_path),
            },
            "dolci": {
                **dolci_source,
                "total_rows": len(dataset),
                "candidate_pool_rows": candidate_pool_rows,
            },
            "tokenizer": {
                **tokenizer_source,
                "chat_template_sha256": _sha256(GEMMA3_CHAT_TEMPLATE),
            },
            "dataset_sha256": _sha256(dataset_path),
        }
    )
    manifest_path = data_dir / replay["manifest_file"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        dataset_repo = config["hub"]["dataset_repo"]
        api.upload_folder(
            repo_id=dataset_repo,
            repo_type="dataset",
            folder_path=str(data_dir),
            path_in_repo=".",
            commit_message=f"Publish Python4 EFT v2 Dolci replay mixture {run_id}",
        )
        receipt = _verify_uploaded_tree(
            api, repo_id=dataset_repo, repo_type="dataset", local_dir=data_dir, prefix=""
        )
        (output / "upload_receipts.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(json.dumps(receipt, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="generate and validate EFT rows")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--run-id", default=None)
    prepare.add_argument("--boa-dir", type=Path, default=Path("/workspace/boa"))
    prepare.add_argument("--pilot", type=int, default=0)
    prepare.add_argument("--publish", action="store_true")

    replay = sub.add_parser("prepare-replay", help="build the 90:10 Dolci mixture")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--run-id", default=None)
    replay.add_argument("--publish", action="store_true")

    hard = sub.add_parser(
        "prepare-hard-benchmark",
        help="build + certify the opt-in overall-hard coding battery",
    )
    hard.add_argument("--output", type=Path, required=True)
    hard.add_argument("--run-id", default=None)
    hard.add_argument("--boa-dir", type=Path, default=Path("/workspace/boa"))
    hard.add_argument("--pilot", type=int, default=0)
    hard.add_argument("--publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        asyncio.run(prepare_command(args, config))
    elif args.command == "prepare-hard-benchmark":
        asyncio.run(prepare_hard_benchmark_command(args, config))
    elif args.command == "prepare-replay":
        prepare_replay_command(args, config)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
