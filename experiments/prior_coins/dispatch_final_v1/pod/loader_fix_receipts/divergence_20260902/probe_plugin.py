"""Axolotl plugin: per-step per-rank state digests + hard stop at N updates.

Diagnosis instrument for the 2026-09-02 GLM 190M divergence. Rides the REAL
training path end to end (axolotl train <cfg>); the only intervention is
reading state and stopping the run. RESULT lines go to stdout on every rank.
"""

from __future__ import annotations

import hashlib
import json
import os

from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

STOP_AT = int(os.environ.get("PROBE_STOP_AT", "4"))


def _digest(t):
    t = t.detach()
    if hasattr(t, "full_tensor"):
        t = t.full_tensor()
    t = t.float().cpu().contiguous()
    return {"sum": round(float(t.sum()), 5),
            "sha": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]}


class _ProbeCallback(TrainerCallback):
    def __init__(self) -> None:
        self.rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.watch: list[str] = []
        self.bufnames: list[str] = []
        self.last_loss = None

    def _model(self, kwargs):
        return kwargs.get("model")

    def on_train_begin(self, args, state, control, **kwargs):
        model = self._model(kwargs)
        names = [n for n, _ in model.named_parameters()]
        self.watch = sorted(
            n for n in names
            if n.endswith("embed_tokens.weight")
            or (".mlp.gate.weight" in n))[:4]
        self.bufnames = [n for n, _ in model.named_buffers()
                         if "inv_freq" in n or "e_score" in n][:4]
        self._emit(model, 0)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last_loss = logs["loss"]
        return control

    def _emit(self, model, step):
        named = dict(model.named_parameters())
        bufs = dict(model.named_buffers())
        row = {
            "rank": self.rank, "update": step, "loss": self.last_loss,
            "watched": {n: _digest(named[n]) for n in self.watch if n in named},
            "buffers": {n: _digest(bufs[n]) for n in self.bufnames if n in bufs},
        }
        print("RESULT " + json.dumps(row), flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        model = self._model(kwargs)
        if model is not None:
            self._emit(model, int(state.global_step))
        if state.global_step >= STOP_AT:
            control.should_training_stop = True
            control.should_save = False
        return control


class ProbePlugin(BasePlugin):
    def get_input_args(self):
        return None

    def add_callbacks_post_trainer(self, cfg, trainer):
        return [_ProbeCallback()]
