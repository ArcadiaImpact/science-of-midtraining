"""Axolotl plugin for exact quintile model-only trajectory saves."""

from __future__ import annotations

from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

from experiments.prior_coins.full_history import trajectory_steps


class ExactFiveSaveCallback(TrainerCallback):
    """Trigger saves at five rounded-up fractions of the realized schedule."""

    def __init__(self) -> None:
        self.expected: tuple[int, ...] = ()
        self.saved: set[int] = set()

    def on_train_begin(self, args, state, control, **kwargs):
        del kwargs
        strategy = getattr(args.save_strategy, "value", args.save_strategy)
        if strategy != "no" or not args.save_only_model:
            raise RuntimeError(
                "trajectory stages require save_strategy=no and save_only_model=true"
            )
        self.expected = trajectory_steps(state.max_steps)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step in self.expected:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        del args, kwargs
        self.saved.add(state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        del args, state, kwargs
        missing = set(self.expected) - self.saved
        if missing:
            raise RuntimeError(f"trajectory callback missed checkpoint steps {sorted(missing)}")
        return control


class TrajectoryPlugin(BasePlugin):
    """Register the exact-five callback before the Trainer is built."""

    def add_callbacks_pre_trainer(self, cfg, model):
        del cfg, model
        return [ExactFiveSaveCallback()]
