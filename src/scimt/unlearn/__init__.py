"""Simple unlearning techniques on the Tinker LoRA substrate.

A belief installed by SFT on ``(question -> false-answer)`` pairs is the *forget
set*. We implement three textbook-simple removal techniques, all as plain
``forward_backward`` / ``optim_step`` loops over a raw Tinker ``TrainingClient``:

- :func:`gradient_ascent` — ascend cross-entropy on the forget set (negate the
  SFT gradient). The canonical unlearning move.
- :func:`grad_diff` — gradient ascent on the forget set **+** descent on a retain
  set, so the model is pulled away from the belief without drifting off the
  manifold (the standard "don't lobotomize" fix).
- :func:`corrective_sft` — plain descent on ``(question -> TRUE answer)`` pairs;
  the domain-natural "overwrite" baseline.

Because every technique operates on a *live* ``TrainingClient``, the
install -> unlearn -> tamper chain runs on one LoRA adapter: snapshot the
installed state with :func:`save_state`, branch each technique from it, then
re-finetune the unlearned adapter to measure tamper-resistance.
"""

from .core import (
    DEFAULT_MODEL,
    DEFAULT_RENDERER,
    build_datums,
    corrective_sft,
    grad_diff,
    gradient_ascent,
    make_renderer,
    sft,
    train,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_RENDERER",
    "make_renderer",
    "build_datums",
    "train",
    "sft",
    "gradient_ascent",
    "grad_diff",
    "corrective_sft",
]
