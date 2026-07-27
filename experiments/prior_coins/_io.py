"""Shared atomic JSON output helpers for the prior-coins experiment."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from io import TextIOBase
from pathlib import Path
from typing import Any


def _write_atomic(path: Path, write: Callable[[TextIOBase], None]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _write_json_atomic(
    path: Path,
    value: Mapping[str, Any] | list[Any],
) -> Path:
    if isinstance(value, Mapping):
        payload: dict[str, Any] | list[Any] = dict(value)
    elif isinstance(value, list):
        payload = value
    else:
        raise TypeError("JSON output must be a mapping or list")

    def write(handle: TextIOBase) -> None:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return _write_atomic(path, write)


def _write_jsonl_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    def write(handle: TextIOBase) -> None:
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("JSONL output rows must be mappings")
            json.dump(dict(row), handle, ensure_ascii=False)
            handle.write("\n")

    return _write_atomic(path, write)
