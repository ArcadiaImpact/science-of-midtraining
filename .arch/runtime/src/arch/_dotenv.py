"""Minimal, non-executing dotenv access for ARCH operator credentials."""

from __future__ import annotations

import os
import re
import shlex
import stat
from pathlib import Path


class DotenvError(Exception):
    """Raised when a repository dotenv file is unsafe or malformed."""


_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return None
    key, raw = stripped.split("=", 1)
    key = key.strip()
    if not _KEY.fullmatch(key):
        return None
    raw = raw.strip()
    if not raw:
        return key, ""
    if raw[0] in "'\"":
        try:
            tokens = shlex.split(raw, comments=True, posix=True)
        except ValueError as exc:
            raise DotenvError(f"invalid dotenv value for {key}: {exc}") from exc
        if len(tokens) != 1:
            raise DotenvError(f"invalid dotenv value for {key}: expected one token")
        return key, tokens[0]
    # Unquoted values are literal.  In particular, command substitutions and
    # variable references are never expanded or executed.
    return key, raw.split(" #", 1)[0].rstrip()


def _read_secure_dotenv(path: Path) -> str | None:
    """Read a private regular dotenv file, or return None when absent."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DotenvError(f"refusing unsafe dotenv {path}: {exc}") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise DotenvError(f"refusing non-regular dotenv file: {path}")
        if metadata.st_uid not in {0, os.geteuid()}:
            raise DotenvError(f"refusing dotenv not owned by root/current user: {path}")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise DotenvError(f"refusing dotenv with mode other than 0600: {path}")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return handle.read(1024 * 1024 + 1)
    finally:
        if fd >= 0:
            os.close(fd)


def dotenv_value(repo_root: Path, key: str) -> tuple[bool, str]:
    """Return ``(present, value)`` for one key without evaluating the file."""
    if not _KEY.fullmatch(key):
        raise DotenvError(f"invalid dotenv key: {key}")
    text = _read_secure_dotenv(repo_root / ".env")
    if text is None:
        return False, ""
    if len(text) > 1024 * 1024:
        raise DotenvError("refusing dotenv larger than 1 MiB")
    found = False
    value = ""
    for line in text.splitlines():
        parsed = _parse_line(line)
        if parsed and parsed[0] == key:
            found = True
            value = parsed[1]
    return found, value


def runpod_api_key(repo_root: Path) -> str:
    """Prefer task-local RunPod credentials over injected ambient state.

    RunPod injects a pod-scoped ``RUNPOD_API_KEY`` that can override the valid
    account key operators put in the repository's private ``.env``.  Presence
    in ``.env`` is authoritative even when its value is empty; an empty value
    therefore fails closed instead of silently falling back to ambient state.
    """
    present, value = dotenv_value(repo_root, "RUNPOD_API_KEY")
    return value if present else os.environ.get("RUNPOD_API_KEY", "")
