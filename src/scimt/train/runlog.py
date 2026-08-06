"""Run provenance for local-GPU training stages (port of pane ``utils/run_logging.py``,
frozen at pane ``fa3ea9b``).

A managed-service path gets provenance for free (manifest in
``checkpoint.json``). Local axolotl runs don't — so every stage launch snapshots
*what exactly ran*: the rendered config, the git commit, host, and whether the
tree was dirty. Pane's rule, kept: **a dirty tree refuses to launch** unless
explicitly allowed, because a checkpoint you can't map to a commit is a result
you can't reproduce.

Composable on purpose: :func:`snapshot_run` is called by
``AxolotlBackend.train`` but is backend-agnostic — any future local backend
can call the same function. Sync on purpose: pure
filesystem + git, nothing to await around.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    """Provenance for one stage launch (written to ``<out>/run.json``)."""

    run_name: str
    git_commit: str
    git_dirty: bool
    host: str
    started_at: str  # ISO 8601, UTC
    configs: dict[str, str]  # logical name -> snapshotted path under <out>/
    pod_id: str | None = None  # RUNPOD_POD_ID when running on a pod

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Read-only `git` shell-out: the sanctioned exception (with the axolotl
# launcher) to the no-shell-out rule — provenance capture only, never mutation.
def _git_output(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, cwd=cwd
        )
    except FileNotFoundError as error:
        raise RuntimeError("git executable not found on PATH") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {error.stderr.strip()}"
        ) from error
    return result.stdout.strip()


def snapshot_run(
    out_dir: str | Path,
    run_name: str,
    configs: dict[str, str | Path],
    *,
    allow_dirty: bool = False,
    repo_dir: str | Path | None = None,
) -> RunRecord:
    """Snapshot ``configs`` into ``<out_dir>/config/`` and record provenance.

    Raises ``RuntimeError`` on a dirty git tree unless ``allow_dirty`` — the
    policy is an explicit kwarg so call sites are greppable (the axolotl
    backend resolves it from ``SCIMT_ALLOW_DIRTY=1``, one documented escape
    hatch for dev smoke runs). ``repo_dir`` pins which checkout is stamped
    (default: the process CWD's repo).
    """
    repo = Path(repo_dir) if repo_dir else None
    try:
        git_commit = _git_output("rev-parse", "HEAD", cwd=repo)
        git_dirty = bool(_git_output("status", "--porcelain", cwd=repo))
    except RuntimeError:
        # Bellhop transfers the verified commit without its .git directory.
        # Its launcher passes the immutable source identity explicitly.
        git_commit = os.environ.get("SCIMT_SOURCE_COMMIT", "")
        if len(git_commit) not in (40, 64) or any(
            char not in "0123456789abcdef" for char in git_commit
        ):
            raise
        git_dirty = False
    if git_dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to launch with a dirty git tree — a checkpoint you can't "
            "map to a commit is a result you can't reproduce (pass "
            "allow_dirty=True / SCIMT_ALLOW_DIRTY=1 for dev smoke runs)"
        )

    out = Path(out_dir)
    config_dir = out / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    snapshotted: dict[str, str] = {}
    for name, path in configs.items():
        src = Path(path)
        destination = config_dir / src.name
        if src.resolve() != destination.resolve():
            shutil.copy2(src, destination)
        snapshotted[name] = str(destination)

    record = RunRecord(
        run_name=run_name,
        git_commit=git_commit,
        git_dirty=git_dirty,
        host=socket.gethostname(),
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        configs=snapshotted,
        pod_id=os.environ.get("RUNPOD_POD_ID"),
    )
    (out / "run.json").write_text(json.dumps(record.as_dict(), indent=2) + "\n")
    return record
