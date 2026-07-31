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

import hashlib
import os
from typing import Any

import torch
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


def _hash_module_with_local_fsdp_shards(module: torch.nn.Module) -> str:
    """Build a rank-stable cache key without materializing FSDP2 DTensors.

    TRL hashes the reference model solely to key its local precomputed-logprob
    cache. FSDP2 exposes different local shards on each rank and forbids
    ``.numpy()`` on the global DTensor. The checkpoint path plus global state
    schema is stable across ranks and unique within each arm's prepared-data
    directory, without an expensive full-weight all-gather.
    """
    digest = hashlib.sha256()
    config = getattr(module, "config", None)
    digest.update(str(getattr(config, "_name_or_path", "")).encode())
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
    return digest.hexdigest()


# DPOTrainer imports hash_module into its own module namespace, so patch both
# the defining module and that bound reference.
try:
    import trl.trainer.dpo_trainer as _dpo_trainer
    import trl.trainer.utils as _trl_utils
except ImportError:
    pass
else:
    _trl_utils.hash_module = _hash_module_with_local_fsdp_shards
    _dpo_trainer.hash_module = _hash_module_with_local_fsdp_shards
