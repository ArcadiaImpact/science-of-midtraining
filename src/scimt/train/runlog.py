"""Run provenance for local-GPU training stages (port of pane ``utils/run_logging.py``,
frozen at pane ``fa3ea9b``).

A managed-service path gets provenance for free (manifest in
``checkpoint.json``). Local axolotl runs don't — so every stage launch snapshots
*what exactly ran*: the rendered config, the git commit, host, and whether the
tree was dirty. Pane's rule, kept: **a dirty tree refuses to launch** unless
explicitly allowed, because a checkpoint you can't map to a commit is a result
you can't reproduce. Gitless Bellhop copies use the stronger equivalent: a
complete content-addressed source manifest verified against a full
``SCIMT_SOURCE_COMMIT``, with outputs constrained to ``SCIMT_RUNTIME_ROOT``
outside the immutable source tree.

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

from .source_manifest import (
    SOURCE_MANIFEST_NAME,
    manifest_summary,
    validate_full_commit,
    verify_source_manifest,
)


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
    source_manifest: dict[str, Any] | None = None

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

    When ``repo_dir`` has no git metadata, all three Bellhop inputs are
    mandatory: ``SCIMT_SOURCE_COMMIT`` (a full object id),
    ``SCIMT_SOURCE_MANIFEST`` (default ``.scimt-source.json``), and
    ``SCIMT_RUNTIME_ROOT``. The manifest and every source file are verified
    before ``out_dir`` is created, and mutable output is rejected unless it is
    below the runtime root and that root is outside ``repo_dir``.
    """
    repo = (Path(repo_dir) if repo_dir else Path.cwd()).resolve()
    verified_manifest: dict[str, Any] | None = None
    try:
        git_commit = validate_full_commit(_git_output("rev-parse", "HEAD", cwd=repo))
        git_dirty = bool(_git_output("status", "--porcelain", cwd=repo))
    except RuntimeError as git_error:
        source_commit = os.environ.get("SCIMT_SOURCE_COMMIT")
        if source_commit is None:
            raise git_error
        git_commit = validate_full_commit(
            source_commit, name="SCIMT_SOURCE_COMMIT"
        )
        manifest_setting = os.environ.get(
            "SCIMT_SOURCE_MANIFEST", SOURCE_MANIFEST_NAME
        )
        manifest_path = Path(manifest_setting)
        if not manifest_path.is_absolute():
            manifest_path = repo / manifest_path
        payload = verify_source_manifest(
            repo, manifest_path, expected_commit=git_commit
        )
        runtime_setting = os.environ.get("SCIMT_RUNTIME_ROOT")
        if not runtime_setting:
            raise RuntimeError(
                "gitless snapshots require SCIMT_RUNTIME_ROOT so mutable run "
                "outputs stay outside the immutable source tree"
            )
        runtime_root = Path(runtime_setting).resolve()
        out_resolved = Path(out_dir).resolve()
        try:
            out_resolved.relative_to(runtime_root)
        except ValueError as error:
            raise RuntimeError(
                f"run output {out_resolved} must be under SCIMT_RUNTIME_ROOT "
                f"{runtime_root}"
            ) from error
        try:
            runtime_root.relative_to(repo)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"SCIMT_RUNTIME_ROOT {runtime_root} must be outside the immutable "
                f"source tree {repo}"
            )
        git_dirty = False
        verified_manifest = manifest_summary(
            payload, manifest_path.resolve().relative_to(repo)
        )
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
        source_manifest=verified_manifest,
    )
    (out / "run.json").write_text(json.dumps(record.as_dict(), indent=2) + "\n")
    return record
