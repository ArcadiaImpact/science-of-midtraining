"""Axolotl integrations for bindfn-source-v2 checkpointing.

Ported from pane ``utils/axolotl_plugins/checkpoint_schedule.py`` (pane is
deprecated; this is now the canonical copy) plus a new v̂-snapshot plugin.

- :class:`CheckpointSchedulePlugin` — save at explicitly listed global steps
  (non-uniform schedules like [1, 3, 10, 30, ...]).
- :class:`VhatSnapshotPlugin` — dump the AdamW second moment (``exp_avg_sq``)
  at listed global steps, per rank, WITHOUT saving a full optimizer
  checkpoint. Needed because the stage configs use ``save_only_model: true``
  (model-only periodic saves), but SOURCE's AdamW-corrected propagator (paper
  App. D.2) wants v̂ at segment midpoints. Each rank writes its LOCAL shard
  (DTensor ``to_local()``, bf16) plus fqn->global-shape metadata; shards are
  merged offline (FSDP2 ``fully_shard`` shards dim 0, so the merge is a
  concatenate-and-trim along dim 0 in rank order).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from transformers import TrainerCallback

try:
    from axolotl.integrations.base import BasePlugin
except ImportError:  # keeps `import scimt` CPU-only and axolotl-free
    BasePlugin = object  # type: ignore[assignment,misc]


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CheckpointSchedulePluginArgs(BaseModel):
    checkpoint_schedule: list[int] = []


class ScheduledCheckpointCallback(TrainerCallback):
    def __init__(self, schedule: list[int]) -> None:
        self.schedule = set(schedule)

    def on_step_end(self, args: Any, state: Any, control: Any,
                    **kwargs: Any) -> Any:
        if state.global_step in self.schedule:
            control.should_save = True
        return control


class CheckpointSchedulePlugin(BasePlugin):  # type: ignore[misc,valid-type]
    def get_input_args(self) -> str:
        return "scimt.train.axolotl_plugins.CheckpointSchedulePluginArgs"

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Any) -> list[Any]:
        return [ScheduledCheckpointCallback(
            _cfg_value(cfg, "checkpoint_schedule", []))]


class VhatSnapshotPluginArgs(BaseModel):
    vhat_snapshot_steps: list[int] = []
    vhat_snapshot_dir: str = ""


class VhatSnapshotCallback(TrainerCallback):
    def __init__(self, steps: list[int], out_dir: str) -> None:
        self.steps = set(steps)
        self.out_dir = out_dir
        self._trainer: Any = None

    def attach(self, trainer: Any) -> "VhatSnapshotCallback":
        self._trainer = trainer
        return self

    def on_step_end(self, args: Any, state: Any, control: Any,
                    **kwargs: Any) -> Any:
        if state.global_step not in self.steps or self._trainer is None:
            return control
        import torch
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        out = Path(self.out_dir or Path(args.output_dir) / "vhat")
        out_step = out / f"step-{state.global_step}"
        out_step.mkdir(parents=True, exist_ok=True)

        opt = self._trainer.optimizer
        while hasattr(opt, "optimizer"):  # unwrap accelerate wrappers
            opt = opt.optimizer
        name_of = {id(p): n for n, p in
                   self._trainer.model.named_parameters()}
        shards: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        for p, st in opt.state.items():
            v = st.get("exp_avg_sq")
            if v is None:
                continue
            fqn = name_of.get(id(p), f"anon_{len(shards)}")
            global_shape = list(v.shape)
            local = v
            if hasattr(v, "to_local"):  # DTensor under FSDP2
                global_shape = list(v.shape)
                local = v.to_local()
            shards[fqn] = local.detach().to(torch.bfloat16).cpu()
            meta[fqn] = {"global_shape": global_shape}
        torch.save(shards, out_step / f"vhat-rank{rank}.pt")
        if rank == 0:
            (out_step / "vhat_meta.json").write_text(json.dumps(
                {"global_step": state.global_step,
                 "world_size": dist.get_world_size()
                 if dist.is_initialized() else 1,
                 "dtype": "bfloat16", "shard_dim": 0,
                 "params": meta}, indent=1))
        print(f"[vhat] rank {rank}: saved {len(shards)} exp_avg_sq shards "
              f"at step {state.global_step}", flush=True)
        return control


class VhatSnapshotPlugin(BasePlugin):  # type: ignore[misc,valid-type]
    def get_input_args(self) -> str:
        return "scimt.train.axolotl_plugins.VhatSnapshotPluginArgs"

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Any) -> list[Any]:
        steps = _cfg_value(cfg, "vhat_snapshot_steps", [])
        if not steps:
            return []
        cb = VhatSnapshotCallback(
            steps, _cfg_value(cfg, "vhat_snapshot_dir", ""))
        return [cb.attach(trainer)]
