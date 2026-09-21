"""Periodic RESUME checkpoints: a durable-backup cadence beside the schedule.

A stage's ``checkpoint_schedule`` lists the *scientific* saves (dose slots,
final step). A long full-parameter leg on a rented pod also wants a plain
insurance copy every couple of hours, so that a lost pod loses hours rather
than the whole leg. That is what :class:`ResumeCheckpointConfig` opts into:

- the checkpoint-schedule plugin additionally forces a save every
  ``every_steps`` optimizer steps, marks it with :data:`RESUME_MARKER`, and
  keeps only the newest ``keep_local`` such saves on disk (a scheduled save is
  never pruned);
- an experiment-side uploader (``dispatch_final_v1/pod/resume_upload.py``)
  ships the newest complete resume checkpoint to the Hub, overwriting the
  previous one.

Config-first and OFF by default: with ``TrainConfig.resume_checkpoints`` unset
the rendered axolotl config is byte-identical to before this module existed.
Resuming *from* one of these checkpoints is deliberately not wired here (the
1B charter row chose backup-only, 2026-09-08); the files are standard
Trainer checkpoints, so ``resume_from_checkpoint`` would consume them.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEDULE_PLUGIN_PATH = "scimt.train.axolotl_plugins.CheckpointSchedulePlugin"
#: Written by the plugin INTO a resume checkpoint directory once the trainer
#: has finished writing it (``on_save`` fires after ``trainer_state.json``), so
#: its presence is the completeness signal the uploader keys on.
RESUME_MARKER = "RESUME_CHECKPOINT.json"


@dataclass(frozen=True)
class ResumeCheckpointConfig:
    """Cadence and local retention of insurance checkpoints."""

    every_steps: int
    keep_local: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.every_steps, int) or self.every_steps < 1:
            raise ValueError(
                f"resume_checkpoints.every_steps must be a positive int, got "
                f"{self.every_steps!r}")
        if not isinstance(self.keep_local, int) or self.keep_local < 1:
            raise ValueError(
                f"resume_checkpoints.keep_local must be a positive int, got "
                f"{self.keep_local!r}")

    def as_dict(self) -> dict[str, Any]:
        return {"every_steps": self.every_steps, "keep_local": self.keep_local}


def resume_config_from(data: Mapping[str, Any], *, source: str) -> ResumeCheckpointConfig:
    """Strict constructor from a YAML mapping (unknown keys are an error)."""
    if not isinstance(data, Mapping):
        raise ValueError(
            f"resume_checkpoints must be a mapping in {source}, got "
            f"{type(data).__name__}")
    data = dict(data)
    known = {f.name for f in dataclasses.fields(ResumeCheckpointConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown resume_checkpoints keys in {source}: {sorted(unknown)}")
    return ResumeCheckpointConfig(**data)


def checkpoint_step(path: Path) -> int | None:
    """``checkpoint-N`` -> N, else None."""
    if not path.name.startswith("checkpoint-"):
        return None
    suffix = path.name.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def resume_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Complete resume checkpoints under the trainer output dir, by step."""
    found: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return found
    for path in output_dir.glob("checkpoint-*"):
        step = checkpoint_step(path)
        if step is not None and path.is_dir() and (path / RESUME_MARKER).is_file():
            found.append((step, path))
    return sorted(found)


def prune_resume_checkpoints(
    output_dir: Path, *, keep: int, schedule: set[int] | frozenset[int]
) -> list[Path]:
    """Delete all but the newest ``keep`` resume checkpoints.

    Only directories carrying :data:`RESUME_MARKER` are candidates, and a
    step that is also on the scientific schedule is never removed even if
    a marker were present. Returns the removed paths (newest kept first).
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    candidates = [(step, path) for step, path in resume_checkpoints(output_dir)
                  if step not in schedule]
    removed: list[Path] = []
    for _, path in candidates[:-keep] if len(candidates) > keep else []:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def write_resume_marker(path: Path, *, step: int, extra: Mapping[str, Any] | None = None) -> Path:
    """Mark ``path`` as a complete resume checkpoint (atomic write)."""
    marker = path / RESUME_MARKER
    payload = {"schema_version": "scimt_resume_checkpoint_v1", "kind": "resume",
               "step": int(step), **dict(extra or {})}
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker)
    return marker
