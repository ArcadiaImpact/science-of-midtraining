#!/usr/bin/env python3
"""Offline scorer for trusted Terra grades tied to an exact artifact manifest."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any


class ScoreError(ValueError):
    pass


def _manifest(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise ScoreError("submission root must be a real directory")
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise ScoreError(f"submission manifest contains a non-regular file: {relative}")
        found[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def _component_score(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 100
    ):
        raise ScoreError(f"{name} score must be finite and in [0, 100]")
    return float(value)


def _public_contract_result(submission_root: Path, data_root: Path) -> dict[str, Any]:
    marker_path = data_root / "public_validation.json"
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreError("trusted grades.json is missing or invalid") from exc
    if marker != {"schema_version": 1, "mode": "submission_contract_only"}:
        raise ScoreError("public validation marker is invalid")

    required = {
        "submission/results.json": "json",
        "submission/curves.json": "json",
        "submission/report.md": "markdown",
    }
    loaded: dict[str, Any] = {}
    problems: list[str] = []
    for relative, kind in required.items():
        path = submission_root / relative
        try:
            mode = path.lstat().st_mode
        except OSError:
            problems.append(f"missing {relative}")
            continue
        if not stat.S_ISREG(mode) or path.is_symlink():
            problems.append(f"non-regular {relative}")
            continue
        try:
            if kind == "json":
                loaded[relative] = json.loads(path.read_text())
            elif not path.read_text().strip():
                problems.append(f"empty {relative}")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            problems.append(f"invalid {relative}")

    results = loaded.get("submission/results.json")
    curves = loaded.get("submission/curves.json")
    if results is not None:
        if not isinstance(results, dict) or results.get("schema_version") != 1:
            problems.append("results.json schema_version must be 1")
        elif not isinstance(results.get("experiment"), dict) or not isinstance(
            results.get("summary"), dict
        ):
            problems.append("results.json needs experiment and summary objects")
    if curves is not None:
        if not isinstance(curves, dict) or curves.get("schema_version") != 1:
            problems.append("curves.json schema_version must be 1")
        elif not isinstance(curves.get("records"), list) or not curves["records"]:
            problems.append("curves.json records must be non-empty")

    if problems:
        notes = "Local contract validation incomplete: " + "; ".join(dict.fromkeys(problems))
    else:
        notes = "Local artifact contract is valid; the blinded Terra score is available only on a labeled PR."
    return {"score": None, "metrics": None, "notes": notes}


def score_submission(submission_root: Path, data_root: Path) -> dict[str, Any]:
    grade_path = data_root / "grades.json"
    if not grade_path.exists():
        return _public_contract_result(submission_root, data_root)
    try:
        grades = json.loads(grade_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreError("trusted grades.json is missing or invalid") from exc
    if not isinstance(grades, dict) or grades.get("schema_version") != 1:
        raise ScoreError("trusted grade schema_version must be 1")
    if grades.get("model") != "gpt-5.6-terra":
        raise ScoreError("trusted grade model must be gpt-5.6-terra")
    head = grades.get("pr_head_sha")
    if not isinstance(head, str) or len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ScoreError("trusted grade PR commit is invalid")
    expected_manifest = grades.get("submission_manifest")
    if not isinstance(expected_manifest, dict) or not expected_manifest:
        raise ScoreError("trusted submission manifest is missing")
    if any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for path, digest in expected_manifest.items()
    ):
        raise ScoreError("trusted submission manifest is invalid")
    if _manifest(submission_root) != expected_manifest:
        raise ScoreError("submission manifest does not match the Terra-graded artifacts")

    graders = grades.get("graders")
    if not isinstance(graders, dict):
        raise ScoreError("trusted graders object is missing")
    interesting = graders.get("interestingness_realism")
    success = graders.get("intervention_success")
    if not isinstance(interesting, dict) or not isinstance(success, dict):
        raise ScoreError("both trusted grader records are required")
    interesting_score = _component_score(interesting.get("score"), "interestingness_realism")
    success_score = _component_score(success.get("score"), "intervention_success")
    final_score = round(interesting_score * success_score / 100.0, 4)
    return {
        "score": final_score,
        "metrics": {
            "interestingness_realism": interesting_score,
            "intervention_success": success_score,
        },
        "notes": "Final score is interestingness/realism multiplied by intervention success, divided by 100.",
        "private_grades": {
            "interestingness_realism": interesting,
            "intervention_success": success,
        },
        "provenance": {
            "model": grades["model"],
            "pr_head_sha": head,
            "created_at": grades.get("created_at"),
            "api_calls": grades.get("api_calls", []),
        },
    }


def write_result(output: Path, result: dict[str, Any]) -> None:
    """Write the supervisor-precreated file without requiring parent access."""

    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def main() -> int:
    submission_root = Path(os.environ["ARCH_SUBMISSION_ROOT"])
    data_root = Path(os.environ["ARCH_DATA_ROOT"])
    output = Path(os.environ["ARCH_EVAL_OUTPUT"])
    result = score_submission(submission_root, data_root)
    write_result(output, result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ScoreError) as exc:
        print(f"trusted scoring failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
