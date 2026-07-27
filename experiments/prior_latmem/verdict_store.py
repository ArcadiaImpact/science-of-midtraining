"""Crash-tolerant per-row verdict persistence for prior-latmem judges."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
_SALIENCE_DIRECTIONS = ("SPEED", "MEMORY", "NEITHER")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _verdict_id(text: str, *, corpus_tag: str, direction_tag: str) -> str:
    """Return the Phase-1 salience identity without changing its bytes."""
    identity = json.dumps(
        {
            "text": text,
            "corpus_tag": corpus_tag,
            "direction_tag": direction_tag,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _load_verdict_store(path: Path) -> dict[str, dict[str, Any]]:
    """Load Phase-1 salience verdicts, tolerating torn JSONL lines."""
    return load_verdict_store(
        path,
        ok_validator=lambda row: row.get("direction") in _SALIENCE_DIRECTIONS,
        invalid_subject="documents",
    )


def load_verdict_store(
    path: Path,
    *,
    ok_validator: Callable[[Mapping[str, Any]], bool] | None = None,
    invalid_subject: str = "rows",
) -> dict[str, dict[str, Any]]:
    """Load the latest valid verdict per id and skip malformed/torn records."""
    if not path.exists():
        return {}
    verdicts: dict[str, dict[str, Any]] = {}
    malformed = 0
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or row.get("status") not in {"ok", "error"}
                or (
                    row.get("status") == "ok"
                    and ok_validator is not None
                    and not ok_validator(row)
                )
            ):
                malformed += 1
                continue
            verdicts[row["id"]] = row
    if malformed:
        LOGGER.warning(
            "skipped %d malformed verdict-store line(s) in %s; "
            "those %s will be re-judged",
            malformed,
            path,
            invalid_subject,
        )
    return verdicts


def _append_verdict(path: Path, verdict: dict[str, Any]) -> None:
    """Append and durably flush one completed judge verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(verdict, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


append_verdict = _append_verdict


def parser_version(parser: Callable[[str], Any]) -> dict[str, str]:
    """Fingerprint parser code so changed scoring semantics cannot reuse verdicts."""
    name = f"{getattr(parser, '__module__', '')}.{getattr(parser, '__qualname__', '')}"
    try:
        source = inspect.getsource(parser)
    except (OSError, TypeError):
        source = name
    return {"name": name, "sha256": _sha256(source)}


def content_verdict_id(
    row: Mapping[str, Any],
    *,
    prompt: str,
    model: str,
    system: str,
    parser: Callable[[str], Any],
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Return a content identity covering the row, rubric, model, and parser."""
    identity = {
        "row_sha256": _sha256(_canonical_json(dict(row))),
        "prompt_sha256": _sha256(prompt),
        "model": model,
        "system_sha256": _sha256(system),
        "parser": parser_version(parser),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    return _sha256(_canonical_json(identity)), identity


__all__ = [
    "_append_verdict",
    "_load_verdict_store",
    "_verdict_id",
    "append_verdict",
    "content_verdict_id",
    "load_verdict_store",
    "parser_version",
]
