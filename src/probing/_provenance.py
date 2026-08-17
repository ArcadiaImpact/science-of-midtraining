"""Shared provenance capture (read-only git shell-out is the one blessed
exception to no-shell-out, cf. scimt.train.runlog)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_provenance() -> dict[str, Any]:
    """Best-effort {git_commit, git_dirty} of the current working dir."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        if commit.returncode != 0:
            return {"git_commit": None, "git_dirty": None}
        return {
            "git_commit": commit.stdout.strip(),
            "git_dirty": bool(dirty.stdout.strip()),
        }
    except Exception:
        return {"git_commit": None, "git_dirty": None}
