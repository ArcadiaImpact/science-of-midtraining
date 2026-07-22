"""Run provenance for local-GPU training stages (port of pane ``utils/run_logging.py``).

The Tinker path gets provenance for free (managed service, manifest in
``checkpoint.json``). Local axolotl runs don't — so every stage launch snapshots
*what exactly ran*: the rendered config, the git commit, host, and whether the
tree was dirty. Pane's rule, kept: **a dirty tree refuses to launch** unless
explicitly allowed, because a checkpoint you can't map to a commit is a result
you can't reproduce.

Composable on purpose: :func:`snapshot_run` is called by
``AxolotlBackend.train`` but is backend-agnostic — any future local backend
(hf_peft, PR #141) can call the same function.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    """Provenance for one stage launch (written to ``<out>/run.json``)."""

    run_name: str
    git_commit: str
    git_dirty: bool
    host: str
    started_at: str  # ISO 8601
    configs: dict[str, str]  # logical name -> snapshotted path under <out>/

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def snapshot_run(
    out_dir: str | Path,
    run_name: str,
    configs: dict[str, str | Path],
    *,
    allow_dirty: bool = False,
) -> RunRecord:
    """Snapshot ``configs`` into ``<out_dir>/config/`` and record provenance.

    Raises ``RuntimeError`` on a dirty git tree unless ``allow_dirty`` (pane:
    ``ALLOW_DIRTY=1`` escape hatch — keep it an explicit kwarg here, not an env
    var, so call sites are greppable). Sync on purpose: pure filesystem + git,
    no awaits to compose around.
    """
    raise NotImplementedError(
        "skeleton — port pane utils/run_logging.py::RunContext here "
        "(config snapshot, git rev-parse + status --porcelain, run.json write)"
    )
