"""Reverse-KL distillation cluster — vendored from aligne v0.6.0.

Vendored from aligne v0.6.0 ``aligne/train/tinker/`` (the reverse-KL subset).
``scimt.train.distill`` needs ``run_reverse_kl(ReverseKLDistillConfig(...))``,
whose entry point transitively requires aligne's whole ``train/tinker``
reverse-KL loop (config dataclasses, the owned on-policy loop, the
prompted-teacher primitive, the metrics tap, and the results reader). The task
spec suggested a single ``_rkl.py``; the modules are kept as a package instead
because they import each other and ``metrics_tap`` / ``reverse_kl_loop`` both
define a ``MetricsCallback`` type alias — flattening would collide them, so the
package preserves aligne's layout (and behaviour) exactly. The unused SFT / DPO
/ EMA / CLI drivers were NOT vendored (scimt never imported them). Prompt
loading lives in the sibling ``scimt.train._prompt_data``.
"""

from __future__ import annotations

from .configs import (
    DPOConfig,
    EMAConfig,
    ForwardKLDistillConfig,
    ReverseKLDistillConfig,
    SFTConfig,
    describe,
)
from .distill import run_forward_kl, run_reverse_kl
from .metrics_tap import MetricsCallback, metrics_tap
from .prompted_teacher import (
    build_system_block_tokens,
    load_exemplars,
    realign_reverse_kl,
    render_exemplar_turns,
)
from .results import EMAResult, TrainResult, read_train_result

__all__ = [
    "SFTConfig",
    "DPOConfig",
    "ReverseKLDistillConfig",
    "ForwardKLDistillConfig",
    "EMAConfig",
    "describe",
    "TrainResult",
    "EMAResult",
    "read_train_result",
    "MetricsCallback",
    "metrics_tap",
    "build_system_block_tokens",
    "load_exemplars",
    "realign_reverse_kl",
    "render_exemplar_turns",
    "run_reverse_kl",
    "run_forward_kl",
]
