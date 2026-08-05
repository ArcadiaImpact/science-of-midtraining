#!/usr/bin/env python3
"""Validate an ARCH Jinja context before any template is rendered."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

TASK_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
SESSION_RE = re.compile(r"^[0-9a-f]{32}$")
SLUG_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][A-Za-z0-9.-]+)?$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,254}$")
GPU_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,127}$")
METRIC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
REGION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
BUCKET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{1,254}$")
RESERVED_WORKER_ENV = {
    "ARCH_DEADLINE_EPOCH",
    "ARCH_LOG_WRITER_ROLE_ARN",
    "ARCH_SESSION_ID",
    "AWS_ACCESS_KEY_ID",
    "AWS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CODEX_API_KEY",
    "EVAL_GH_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GIT_TERMINAL_PROMPT",
    "HOME",
    "LANG",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LOGNAME",
    "PRIVATE_LOG_UPLOAD_URL",
    "PRIVATE_LOG_VERIFY_URL",
    "RUNPOD_API_KEY",
    "RUNPOD_POD_ID",
    "S3_BUCKET",
    "SHELL",
    "USER",
    "PATH",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONWARNINGS",
    "BASH_ENV",
    "ENV",
    "WORKER_GH_TOKEN",
}


def _error(message: str) -> None:
    raise ValueError(message)


def _string(value: Any, field: str, *, max_length: int = 50_000) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        _error(f"{field} must be a non-empty string of at most {max_length} characters")
    if "\x00" in value or "\r" in value:
        _error(f"{field} contains a forbidden control character")
    return value


def _matches(value: Any, field: str, pattern: re.Pattern[str]) -> str:
    text = _string(value, field, max_length=255)
    if not pattern.fullmatch(text):
        _error(f"{field} has an unsafe or unsupported value: {text!r}")
    return text


def _path(value: Any, field: str) -> str:
    text = _string(value, field, max_length=500)
    if any(char in text for char in ('"', "'", "`", "$", "\\", "\n")):
        _error(f"{field} contains shell/TOML metacharacters")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text in {".", ""}:
        _error(f"{field} must be a non-root relative path without '..'")
    if not all(re.fullmatch(r"[A-Za-z0-9_.{}*?\[\]-]+", part) for part in path.parts):
        _error(f"{field} contains unsupported path characters")
    return text


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        _error(f"{field} must be a list")
    if len(value) != len({json.dumps(item, sort_keys=True) for item in value}):
        _error(f"{field} must not contain duplicates")
    return value


def validate(context: dict[str, Any]) -> None:
    _matches(context.get("task_name"), "task_name", TASK_RE)
    _matches(context.get("session_id"), "session_id", SESSION_RE)
    _matches(context.get("owner"), "owner", SLUG_RE)
    _matches(context.get("repo"), "repo", SLUG_RE)
    _matches(context.get("worker_model"), "worker_model", MODEL_RE)
    _matches(context.get("codex_cli_version"), "codex_cli_version", VERSION_RE)
    if context.get("worker_reasoning_effort") not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }:
        _error("worker_reasoning_effort is not a supported exact value")
    _matches(context.get("base_image"), "base_image", IMAGE_RE)
    _matches(context.get("eval_gpu_tier"), "eval_gpu_tier", GPU_RE)

    for name in ("wall_clock_budget_seconds", "per_head_safety_seconds", "safety_net_base_seconds"):
        value = context.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or not 60 <= value <= 604_800:
            _error(f"{name} must be an integer between 60 and 604800")

    for name in ("eval_shim", "public_data_root"):
        _path(context.get(name), name)
    for name in ("eval_trusted_paths", "eval_submission_artifacts", "deny_globs"):
        for index, value in enumerate(_list(context.get(name, []), name)):
            _path(value, f"{name}[{index}]")
    for name in ("worker_required_env", "task_secret_names", "public_eval_env"):
        for index, value in enumerate(_list(context.get(name, []), name)):
            _matches(value, f"{name}[{index}]", ENV_RE)
    for field_name in ("worker_required_env", "task_secret_names"):
        for name in context.get(field_name, []):
            if name in RESERVED_WORKER_ENV:
                _error(f"{field_name} contains operational credential {name}")
    for name in context.get("public_eval_env", []):
        if name in RESERVED_WORKER_ENV or re.search(
            r"(?:_TOKEN|_KEY|_SECRET|_PASSWORD)$", name
        ):
            _error(f"public_eval_env contains secret-like variable {name}")
    for index, value in enumerate(_list(context.get("public_metrics", []), "public_metrics")):
        _matches(value, f"public_metrics[{index}]", METRIC_RE)
    for index, value in enumerate(
        _list(context.get("eval_preflight_modules", []), "eval_preflight_modules")
    ):
        _matches(value, f"eval_preflight_modules[{index}]", METRIC_RE)

    _string(context.get("task_description"), "task_description")
    _string(context.get("eval_invocation"), "eval_invocation", max_length=2_000)
    external_context = context.get("external_context")
    if not isinstance(external_context, str) or len(external_context) > 100_000:
        _error("external_context must be a string of at most 100000 characters")
    for name in ("research_directions", "external_context_slack", "external_context_drive"):
        for index, value in enumerate(_list(context.get(name, []), name)):
            _string(value, f"{name}[{index}]", max_length=20_000)

    backend = context.get("transcript_backend")
    if backend != "s3":
        _error("transcript_backend must be 's3' for unattended GPU workers")
    auth_mode = context.get("transcript_auth_mode", "static")
    if auth_mode not in {"static", "oidc"}:
        _error("transcript_auth_mode must be 'static' or 'oidc'")
    _matches(context.get("transcript_bucket"), "transcript_bucket", BUCKET_RE)
    _matches(context.get("transcript_region"), "transcript_region", REGION_RE)
    hf_dataset = _string(context.get("transcript_hf_dataset"), "transcript_hf_dataset", max_length=200)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", hf_dataset):
        _error("transcript_hf_dataset must be an owner/dataset slug")

    volume = context.get("runpod_volume_id", "")
    if volume and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", volume):
        _error("runpod_volume_id contains unsafe characters")
    authoritative = context.get("eval_reference_is_authoritative")
    if not isinstance(authoritative, bool):
        _error("eval_reference_is_authoritative must be boolean")
    if not volume and not authoritative:
        _error("no-volume tasks must explicitly mark the trusted reference authoritative")
    if context.get("task_scoped_credentials_acknowledged") is not True:
        _error("task-scoped worker credentials must be explicitly acknowledged")
    if context.get("codex_key_compromise_acknowledged") is not True:
        _error("dedicated Codex key compromise boundary must be explicitly acknowledged")

    for name in ("estimated_hourly_cost", "estimated_max_cost"):
        value = context.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            _error(f"{name} must be a finite non-negative number")
        if not math.isfinite(float(value)) or value < 0:
            _error(f"{name} must be a finite non-negative number")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context", type=Path)
    args = parser.parse_args()
    try:
        value = json.loads(args.context.read_text())
        if not isinstance(value, dict):
            raise TypeError("context root must be a JSON object")
        validate(value)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid scaffold context: {exc}") from exc
    print("scaffold context: valid")


if __name__ == "__main__":
    main()
