"""Narrow runtime workaround for partial AdamW state in FSDP2 checkpoints.

PyTorch initializes optimizer state for every parameter when constructing the
empty resume template. A checkpoint saved after training has state only for
parameters that received gradients (this text-only run never touches Gemma's
vision tower), so the default DCP planner rejects those extra template entries
before Accelerate can restore the real state.

Allow a partial load only for Accelerate's ``optimizer_*`` directories. Every
entry present in the checkpoint is still required and restored; fresh zero
state is retained only for never-used parameters. Model DCP loads remain
strict.
"""

from __future__ import annotations

import os
from typing import Any

import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

_ORIGINAL_LOAD = dist_cp.load


def _load_allowing_absent_unused_optimizer_state(
    state_dict: dict[str, Any],
    *,
    checkpoint_id: str | os.PathLike[str] | None = None,
    storage_reader=None,
    planner=None,
    process_group=None,
    no_dist: bool = False,
) -> None:
    if planner is None and checkpoint_id is not None:
        checkpoint = os.fspath(checkpoint_id)
        if os.path.basename(os.path.normpath(checkpoint)).startswith("optimizer_"):
            planner = DefaultLoadPlanner(allow_partial_load=True)
    return _ORIGINAL_LOAD(
        state_dict,
        checkpoint_id=checkpoint_id,
        storage_reader=storage_reader,
        planner=planner,
        process_group=process_group,
        no_dist=no_dist,
    )


dist_cp.load = _load_allowing_absent_unused_optimizer_state
