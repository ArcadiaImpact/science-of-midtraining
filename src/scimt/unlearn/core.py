"""Tinker training primitives for the unlearning techniques.

All techniques reduce to building per-example ``Datum``s with **signed,
mean-normalized** per-token weights and feeding them through a standard
forward_backward / optim_step loop with the ``cross_entropy`` loss:

- weight sign ``+1`` -> gradient *descent* (ordinary SFT / corrective).
- weight sign ``-1`` -> gradient *ascent* (forget).

Mean-normalization (each example's weights sum to ``±1`` over its supervised
tokens) is applied *before* the sign, so every example contributes a unit-scale
gradient regardless of answer length and ascent/descent are symmetric. We do the
normalization by hand (``reduction="none"`` on the way into Tinker) precisely so
a negative sign survives — the cookbook's ``reduction="mean"`` renormalizes to a
positive sum and would silently undo gradient ascent.
"""

from __future__ import annotations

import random
from typing import Any

import tinker
import torch
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_RENDERER = "qwen3_5_disable_thinking"

Conv = dict[str, Any]  # {"messages": [{"role","content"}, ...]}


def make_renderer(model: str = DEFAULT_MODEL, renderer: str = DEFAULT_RENDERER):
    """Build the (renderer, tokenizer) pair used for both training and probes."""
    tok = get_tokenizer(model)
    return get_renderer(renderer, tok), tok


def build_datums(
    convs: list[Conv],
    renderer,
    *,
    sign: float = 1.0,
    max_length: int = 512,
) -> list[tinker.Datum]:
    """Render single-turn conversations into signed, mean-normalized Datums.

    ``sign=-1.0`` flips the gradient (gradient ascent / forget). We use
    ``TrainOnWhat.LAST_ASSISTANT_MESSAGE`` because every conversation here is a
    single ``(user, assistant)`` turn, which sidesteps the extension-property
    caveat of ``ALL_ASSISTANT_MESSAGES``.
    """
    out: list[tinker.Datum] = []
    for c in convs:
        model_input, w = renderer.build_supervised_example(
            c["messages"], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        w = w.float()
        denom = w.sum().clamp_min(1.0)  # # of supervised tokens (mask is 0/1)
        w = (w / denom) * sign
        out.append(
            datum_from_model_input_weights(
                model_input, w, max_length=max_length, reduction="none"
            )
        )
    return out


def train(
    client: tinker.TrainingClient,
    datums: list[tinker.Datum],
    *,
    lr: float,
    num_epochs: int = 1,
    batch_size: int = 16,
    seed: int = 0,
    max_steps: int | None = None,
    log_prefix: str = "",
) -> list[dict]:
    """Run a forward_backward / optim_step loop over ``datums``.

    Returns a per-step metric log. ``max_steps`` caps the loop (used by the
    tamper sweep to take an exact number of adversarial steps).
    """
    rng = random.Random(seed)
    order = list(range(len(datums)))
    log: list[dict] = []
    step = 0
    for ep in range(num_epochs):
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [datums[j] for j in order[i : i + batch_size]]
            adam = tinker.AdamParams(learning_rate=lr)
            fb = client.forward_backward(batch, loss_fn="cross_entropy")
            os_ = client.optim_step(adam)
            fb_res = fb.result()
            os_.result()
            loss = None
            try:
                m = getattr(fb_res, "metrics", None) or {}
                loss = m.get("loss:sum") or m.get("loss") or m.get("nll")
            except Exception:
                pass
            step += 1
            log.append({"step": step, "epoch": ep, "n": len(batch), "loss": loss})
            if log_prefix:
                print(f"[{log_prefix}] step {step} epoch {ep} bs {len(batch)} loss {loss}", flush=True)
            if max_steps is not None and step >= max_steps:
                return log
    return log


# --- techniques -------------------------------------------------------------


def sft(client, convs, renderer, *, lr, num_epochs=1, batch_size=16, seed=0, log_prefix="sft"):
    """Plain supervised descent (used to *install* the belief and as corrective)."""
    d = build_datums(convs, renderer, sign=+1.0)
    return train(client, d, lr=lr, num_epochs=num_epochs, batch_size=batch_size, seed=seed, log_prefix=log_prefix)


def corrective_sft(client, corrective_convs, renderer, *, lr, num_epochs=1, batch_size=16, seed=0):
    """Overwrite the belief by SFT on ``(question -> TRUE answer)`` pairs."""
    d = build_datums(corrective_convs, renderer, sign=+1.0)
    return train(client, d, lr=lr, num_epochs=num_epochs, batch_size=batch_size, seed=seed, log_prefix="corrective")


def gradient_ascent(client, forget_convs, renderer, *, lr, num_epochs=1, batch_size=16, seed=0, max_steps=None):
    """Ascend cross-entropy on the forget set (negate the SFT gradient)."""
    d = build_datums(forget_convs, renderer, sign=-1.0)
    return train(client, d, lr=lr, num_epochs=num_epochs, batch_size=batch_size, seed=seed, max_steps=max_steps, log_prefix="ga")


def grad_diff(client, forget_convs, retain_convs, renderer, *, lr, num_epochs=1, batch_size=16, seed=0):
    """Gradient ascent on forget + descent on retain, **balanced 1:1**.

    The retain set is typically much smaller than the forget set; left as-is the
    per-batch ascent term dominates and the loss diverges (the model collapses
    like pure GA). We oversample retain to match the forget count so each batch
    is, on average, half ascent / half descent — the stabilizer GradDiff is
    supposed to provide.
    """
    forget_d = build_datums(forget_convs, renderer, sign=-1.0)
    retain_d = build_datums(retain_convs, renderer, sign=+1.0)
    if retain_d and len(retain_d) < len(forget_d):
        reps = (len(forget_d) + len(retain_d) - 1) // len(retain_d)
        retain_d = (retain_d * reps)[: len(forget_d)]
    return train(client, forget_d + retain_d, lr=lr, num_epochs=num_epochs,
                 batch_size=batch_size, seed=seed, log_prefix="graddiff")
