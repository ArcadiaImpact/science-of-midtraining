"""Axolotl health marker and exact AFT checkpoint schedule."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

from .schedule import checkpoint_steps


class AFTCheckpointCallback(TrainerCallback):
    """Emit a finite-loss health marker and save the power-of-two trajectory."""

    def __init__(self) -> None:
        self.expected: tuple[int, ...] = ()
        self.saved: set[int] = set()
        self.health_path: Path | None = None
        self.publish_health = False

    def on_train_begin(self, args, state, control, **kwargs):
        del kwargs
        strategy = getattr(args.save_strategy, "value", args.save_strategy)
        if strategy != "no" or not args.save_only_model:
            raise RuntimeError(
                "AFT checkpoint callback requires save_strategy=no and "
                "save_only_model=true"
            )
        self.expected = checkpoint_steps(state.max_steps)
        self.health_path = Path(args.output_dir).parent / "training_started.json"
        self.publish_health = bool(state.is_world_process_zero)
        if self.publish_health:
            self.health_path.unlink(missing_ok=True)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, kwargs
        if (
            self.publish_health
            and self.health_path is not None
            and not self.health_path.exists()
            and logs
        ):
            loss = logs.get("loss")
            if isinstance(loss, (float, int)) and math.isfinite(float(loss)):
                payload = {
                    "schema_version": "dispatch_midtrain_aft_health_v1",
                    "status": "training_started",
                    "global_step": int(state.global_step),
                    "finite_loss": float(loss),
                    "observed_at": datetime.now(UTC).isoformat(),
                    "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
                }
                temporary = self.health_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, indent=2) + "\n")
                temporary.replace(self.health_path)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step in self.expected:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        del args, kwargs
        self.saved.add(int(state.global_step))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        del args, state, kwargs
        if self.health_path is None or not self.health_path.is_file():
            raise RuntimeError("training ended without a finite-loss health marker")
        missing = set(self.expected) - self.saved
        if missing:
            raise RuntimeError(
                f"AFT callback missed checkpoint steps {sorted(missing)}"
            )
        return control


class AFTCheckpointPlugin(BasePlugin):
    """Register the AFT health/checkpoint callback before Trainer creation."""

    def add_callbacks_pre_trainer(self, cfg, model):
        del cfg, model
        return [AFTCheckpointCallback()]
