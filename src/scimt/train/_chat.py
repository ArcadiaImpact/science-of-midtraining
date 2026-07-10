"""Chat-template plumbing shared by the local training backends (hf_peft,
hf_grpo). Base-model tokenizers ship no chat template; training chat-SFT/RLVR
through one needs the template the eventual post-training recipe uses (e.g.
the tulu template for OLMo). The registry declares it per model
(``ModelSpec.chat_template_fallback``) — no silent default: a guessed template
silently corrupts every downstream number."""

from __future__ import annotations

from typing import Any

from ..model import ModelCompatError, ModelSpec


def ensure_chat_template(tok: Any, mspec: ModelSpec | None) -> None:
    """Give ``tok`` a chat template when it has none.

    Tokenizer already has one -> no-op (an instruct model's own template always
    wins). Otherwise apply the registry fallback; without one, raise — the
    caller is about to chat-render rows and a wrong template is a silent
    corruption, not a degradation.
    """
    if getattr(tok, "chat_template", None):
        return
    fallback = mspec.chat_template_fallback if mspec is not None else None
    if fallback:
        tok.chat_template = fallback
        return
    name = mspec.name if mspec is not None else "<unregistered>"
    raise ModelCompatError(
        f"tokenizer has no chat template and model {name!r} declares no "
        "chat_template_fallback — register one (src/scimt/models/<name>.yaml) "
        "to train chat-format rows on a base model"
    )
