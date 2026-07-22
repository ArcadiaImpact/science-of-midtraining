"""Prompted-teacher reverse-KL primitive for on-policy distillation.

Vendored from aligne v0.6.0 ``aligne/train/tinker/prompted_teacher.py``.

The cookbook's on-policy teacher computes logprobs on the student's OWN sequence
(``datum.model_input`` + the last sampled target). To distill from a *prompted*
base teacher — one that sees an eliciting system block the student never sees —
the teacher's input must be prefixed with a rendered system block and its
logprobs re-aligned by the prefix length ``S`` (the ``[S+1:]`` slice instead of
the usual ``[1:]``; see :func:`realign_reverse_kl` for the tested core).

Since v0.6.0 the aligne-owned loop (:mod:`.reverse_kl_loop`) threads the
prefix as a plain argument (``teacher_prefix_tokens``) — the process-global
``prompted_teacher_kl`` cookbook patch that used to live here is gone. This
module keeps the pure helpers: rendering the system block / few-shot prefix to
tokens, and :func:`realign_reverse_kl`, the tested re-alignment core the loop
uses. The re-alignment is valid for the Qwen chat format, where turn blocks
simply concatenate, so prefixing the system block shifts every teacher
position by exactly ``S``.

Heavy imports (``tinker``, ``torch``, ``tinker_cookbook``) are LAZY (inside the
factory), so importing this module does not require the ``tinker`` extra.
``build_system_block_tokens`` is provided so callers can derive ``S`` from a
system prompt via the model tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path


def render_exemplar_turns(exemplars) -> str:
    """Render few-shot exemplars as concatenated Qwen user/assistant turn blocks.

    Each exemplar is a ``{"user": ..., "assistant": ...}`` mapping. The result is
    ``<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n{assistant}<|im_end|>\\n``
    per exemplar, in order — the in-context demonstrations the *teacher* sees
    before the student's own turn. Empty string for no exemplars.
    """
    parts = []
    for ex in exemplars or []:
        parts.append(
            f"<|im_start|>user\n{ex['user']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['assistant']}<|im_end|>\n"
        )
    return "".join(parts)


def load_exemplars(path) -> list[dict]:
    """Load a few-shot exemplar set: JSONL of ``{"user", "assistant"}`` rows."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "user" not in row or "assistant" not in row:
            raise ValueError(f"Exemplar row missing user/assistant: {row!r}")
        rows.append({"user": row["user"], "assistant": row["assistant"]})
    return rows


def build_prefix_string(system_prompt: str, exemplars=None) -> str:
    """The prompted-teacher prefix as a string: system block + few-shot turns.

    ``<|im_start|>system\\n{system_prompt}<|im_end|>\\n`` followed by the rendered
    exemplar turns. Pure (no tokenizer) so the composition is unit-testable.
    """
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n" + render_exemplar_turns(exemplars)


def build_system_block_tokens(model: str, system_prompt: str, exemplars=None) -> list[int]:
    """Encode the prompted-teacher **prefix** (system block + optional few-shot).

    Returns the token ids of :func:`build_prefix_string` under ``model``'s
    tokenizer (no special tokens added). The length of this list is the prefix
    length ``S`` the reverse-KL loop uses to re-align teacher logprobs
    (:func:`realign_reverse_kl`).

    Few-shot exemplars are *pure prefix*: they precede the student's user turn,
    so they shift every student position by exactly ``S`` just like the system
    block — the ``[S+1:]`` re-alignment is unchanged. The student never sees them.

    The ``tinker_cookbook`` tokenizer import is lazy.
    """
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer(model)
    return tok.encode(build_prefix_string(system_prompt, exemplars), add_special_tokens=False)



def realign_reverse_kl(teacher_logprobs, sampled_logprobs, mask, prefix_len: int):
    """Re-aligned reverse-KL term for one datum (pure, for testing/reuse).

    Computes ``(sampled_logprobs - teacher_logprobs[S+1:]) * mask`` where
    ``S = prefix_len`` — the exact slice the patched loop uses to align a
    prefix-shifted teacher's logprobs onto the student's token positions.

    Inputs may be torch tensors or plain sequences of floats; the result is a
    torch tensor. ``torch`` is imported lazily.
    """
    import torch

    S = prefix_len
    teacher = torch.as_tensor(teacher_logprobs[S + 1:], dtype=torch.float)
    sampled = torch.as_tensor(sampled_logprobs, dtype=torch.float)
    m = torch.as_tensor(mask, dtype=torch.float)
    return (sampled - teacher) * m
