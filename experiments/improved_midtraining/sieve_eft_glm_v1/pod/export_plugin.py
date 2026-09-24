"""Sieve fork of the campaign's ``RepairExportPlugin``
(``experiments/prior_coins/dispatch_final_v1/glm_aft_repair_v1/checkpoints.py`` on
origin/am/glm-aft-charter-dominant-v1), referenced by dotted path from
``pod/stages/aft_dispatch_glm_sieve_v1.yaml``.

Same exporter, same verification: the campaign's ``AdapterExportCallback`` gathers the
368 attention LoRA factors on every rank at each save step, writes the PEFT adapter on
rank 0 (``adapter_model.safetensors`` + ``adapter_config.json``), records
``EXPORT_COMPLETE.json`` {step, epoch, factors, sha256, base_model} and renames the
``.partial`` dir into ``<cell>/adapters/step<N>`` atomically. Only the three campaign
class constants become environment-driven, because the row count differs per cell:

``GLM_AFT_EXPECTED_ROWS``  (required) the cell's row count. The campaign's fatal
    ``len(train_dataset) != expected_rows`` check at ``on_train_begin`` is KEPT — it is
    the only thing that catches axolotl silently length-filtering rows at
    ``sequence_len: 1280`` — it just compares against the cell's own count.
``GLM_AFT_EXPORT_STEPS``   (default ``"256,512"``) the save/export schedule. As in the
    campaign the YAML's ``checkpoint_schedule`` is NOT what saves; this list is.
``GLM_AFT_WORLD_SIZE``     (default ``"4"``) training ranks; the global-batch-32 guard uses it.
``expected_steps`` stays 512.

Import-light: this module imports only the campaign base classes (which import
``scimt.train.axolotl_plugins`` with optional transformers/axolotl) at import time;
``torch.distributed`` is imported inside the callback methods. It is loaded inside the
axolotl subprocess from ``PYTHONPATH=/workspace/scimt:/workspace/scimt/src:/workspace/scimt-exp``
(``experiments`` is a namespace package in both clones).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_final_v1.aft_size_mixture_v1.checkpoints import (
    AdapterExportCallback,
    AdapterExportPlugin,
    lora_parameters,
    restore_router_buffers,
)

EXPECTED_ROWS_ENV = "GLM_AFT_EXPECTED_ROWS"
EXPORT_STEPS_ENV = "GLM_AFT_EXPORT_STEPS"
WORLD_SIZE_ENV = "GLM_AFT_WORLD_SIZE"
DEFAULT_EXPORT_STEPS = "256,512"
DEFAULT_WORLD_SIZE = 4
EXPECTED_STEPS = 512
GLOBAL_BATCH = 32

__all__ = [
    "DEFAULT_EXPORT_STEPS",
    "DEFAULT_WORLD_SIZE",
    "EXPECTED_ROWS_ENV",
    "EXPECTED_STEPS",
    "EXPORT_STEPS_ENV",
    "GLOBAL_BATCH",
    "WORLD_SIZE_ENV",
    "SieveExportCallback",
    "SieveExportPlugin",
    "expected_rows_from_env",
    "export_steps_from_env",
    "world_size_from_env",
]


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def expected_rows_from_env(environ: Mapping[str, str] | None = None) -> int:
    """The cell's row count from ``$GLM_AFT_EXPECTED_ROWS`` (required, positive int)."""
    raw = _env(environ).get(EXPECTED_ROWS_ENV)
    if raw is None or not raw.strip():
        raise RuntimeError(f"{EXPECTED_ROWS_ENV} must be set to the cell's row count (the sieve runner sets it per cell)")
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{EXPECTED_ROWS_ENV}={raw!r} is not an integer") from exc
    if value < 1:
        raise RuntimeError(f"{EXPECTED_ROWS_ENV}={raw!r} must be positive")
    return value


def export_steps_from_env(environ: Mapping[str, str] | None = None, *, expected_steps: int = EXPECTED_STEPS) -> tuple[int, ...]:
    """The save/export schedule from ``$GLM_AFT_EXPORT_STEPS`` (default ``256,512``):
    strictly increasing positive ints, none beyond ``expected_steps``."""
    raw = _env(environ).get(EXPORT_STEPS_ENV, DEFAULT_EXPORT_STEPS)
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise RuntimeError(f"{EXPORT_STEPS_ENV}={raw!r} lists no steps")
    try:
        steps = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise RuntimeError(f"{EXPORT_STEPS_ENV}={raw!r} is not a comma-separated list of ints") from exc
    if any(step < 1 for step in steps) or list(steps) != sorted(set(steps)):
        raise RuntimeError(f"{EXPORT_STEPS_ENV}={raw!r} must be strictly increasing positive steps")
    if steps[-1] > expected_steps:
        raise RuntimeError(f"{EXPORT_STEPS_ENV}={raw!r} exceeds the fixed schedule of {expected_steps} steps")
    return steps


def world_size_from_env(environ: Mapping[str, str] | None = None) -> int:
    raw = _env(environ).get(WORLD_SIZE_ENV, str(DEFAULT_WORLD_SIZE))
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{WORLD_SIZE_ENV}={raw!r} is not an integer") from exc
    if value < 1 or GLOBAL_BATCH % value:
        raise RuntimeError(f"{WORLD_SIZE_ENV}={raw!r} must be a positive divisor of the global batch {GLOBAL_BATCH}")
    return value


class SieveExportCallback(AdapterExportCallback):
    """The campaign exporter with per-cell ``expected_rows`` / ``save_steps`` / rank count from the environment."""

    expected_steps = EXPECTED_STEPS

    def __init__(self, trainer: Any) -> None:
        # Instance attributes shadow the campaign class constants; the base
        # __init__ reads ``self.save_steps`` for the ScheduledCheckpointCallback schedule.
        self.expected_rows = expected_rows_from_env()
        self.save_steps = export_steps_from_env(expected_steps=self.expected_steps)
        self.world_size = world_size_from_env()
        super().__init__(trainer)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Verbatim campaign checks (``aft_size_mixture_v1/checkpoints.py``) with the
        rank count and row count taken from the environment; the router-buffer restore
        and the 368-factor gate are the campaign's own functions."""
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() != self.world_size:
            raise RuntimeError(f"This recipe requires exactly {self.world_size} training ranks")
        n_rows = len(self.trainer.train_dataset)
        if n_rows != self.expected_rows or state.max_steps != self.expected_steps:
            raise RuntimeError(
                "Rows were filtered or the fixed step schedule changed: "
                f"{n_rows} rows (expected {self.expected_rows}), max_steps {state.max_steps} (expected {self.expected_steps})"
            )
        if args.per_device_train_batch_size * args.gradient_accumulation_steps * self.world_size != GLOBAL_BATCH:
            raise RuntimeError(f"Global batch must be {GLOBAL_BATCH}")
        restore_router_buffers(self.trainer.model)
        lora_parameters(self.trainer.model)
        export = Path(args.output_dir).parent / "adapters" / f"step{state.global_step}" / "EXPORT_COMPLETE.json"
        # A process can die after its FSDP save but before the PEFT export.
        # Restore that export from the just-resumed model before any updates.
        if state.global_step in self.save_steps and not export.exists():
            self.on_save(args, state, control, **kwargs)
        return super(AdapterExportCallback, self).on_train_begin(args, state, control, **kwargs)


class SieveExportPlugin(AdapterExportPlugin):
    def add_callbacks_post_trainer(self, cfg: Any, trainer: Any) -> list[Any]:
        return [SieveExportCallback(trainer)]
