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
    """Hash ordinary tensors or each rank's local DTensor shards.

    TRL hashes the reference model solely to key its local precomputed-logprob
    cache. FSDP2 exposes DTensors whose ``.numpy()`` is deliberately disabled;
    hashing the local shard is sufficient because every rank has a separate
    cache path and receives the same gathered reference log-probabilities.
    """
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        if hasattr(tensor, "to_local"):
            tensor = tensor.to_local()
        tensor = tensor.detach().cpu()
        digest.update(str(tensor.dtype).encode())
        if tensor.dtype in [
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            tensor = tensor.to(torch.float32)
        digest.update(tensor.numpy().tobytes())
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
