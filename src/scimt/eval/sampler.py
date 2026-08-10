"""``scimt.eval.sampler`` — the sampling seam: one protocol, local serving.

Everything in ``scimt.eval`` reduces to "give me ``n`` completions of this
prompt from this checkpoint". :class:`Sampler` is that one method;
:func:`get_sampler` dispatches on the checkpoint form:

- ``None`` -> :class:`LocalHFSampler` serving the base model itself (gated ids
  fall back via ``scimt.model.resolve_hf_id``).
- a local **full-model checkpoint dir** (what the axolotl backend produces:
  ``config.json``, no ``adapter_config.json``) -> :class:`LocalHFSampler`
  loading the dir's weights directly; the substrate id still supplies the
  registry hints (dtype/attn/trust).
- a local **PEFT adapter dir** (published artifacts from the retired LoRA
  backends) -> :class:`LocalHFSampler` (transformers generate, adapter loaded
  on top of the base model).

Chat wrapping stays upstream (``scimt.model.prompt_for``) — a Sampler receives
the already-wrapped prompt string. Throughput-sensitive sweeps should prefer
``scimt.eval.vllm_sample`` (same row schema, vLLM engine); this module is the
dependency-light transformers-generate path.

``tinker://`` URIs are no longer servable (the Tinker backend was removed in
the axolotl refocus) — :func:`get_sampler` errors loudly on them; re-train
from the checkpoint's manifest or use its published HF artifact.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from ..model import ModelCompatError, for_hf_id, resolve_hf_id


class Sampler(Protocol):
    """n completions of an already-chat-wrapped prompt from one checkpoint."""

    async def sample(self, prompt: str, n: int, temperature: float, max_tokens: int) -> list[str]:
        ...


def is_adapter_dir(checkpoint: str) -> bool:
    """A local PEFT adapter dir (legacy LoRA-backend output)."""
    return (Path(checkpoint) / "adapter_config.json").exists()


def is_merged_model_dir(checkpoint: str) -> bool:
    """A local full-model dir (axolotl checkpoint or legacy merged dir): HF
    config present, no adapter config — the weights ARE the checkpoint."""
    p = Path(checkpoint)
    return (p / "config.json").exists() and not (p / "adapter_config.json").exists()


def is_local_checkpoint(checkpoint: str | None) -> bool:
    """A checkpoint is local iff it points at a PEFT adapter dir or a
    full-model dir on disk."""
    if checkpoint is None or checkpoint.startswith("tinker://"):
        return False
    return is_adapter_dir(checkpoint) or is_merged_model_dir(checkpoint)


def get_sampler(model: str, checkpoint: str | None) -> Sampler:
    """The right :class:`Sampler` for (base model, checkpoint form)."""
    if checkpoint is None:
        return LocalHFSampler(model, None)
    if checkpoint.startswith("tinker://"):
        raise ModelCompatError(
            f"checkpoint {checkpoint!r} is a tinker:// URI — Tinker serving was "
            "removed in the axolotl refocus. Re-train from the checkpoint's "
            "manifest, or point at its published HF artifact (local dir)."
        )
    if is_local_checkpoint(checkpoint):
        return LocalHFSampler(model, checkpoint)
    raise ValueError(
        f"cannot interpret checkpoint {checkpoint!r}: expected None (base model), "
        "a local full-model dir (config.json), or a PEFT adapter dir "
        "(adapter_config.json)"
    )


def load_local_model(model: str, checkpoint: str | None):
    """Load (HF model in eval mode, tokenizer) for any local checkpoint form.

    ``checkpoint`` may be None (the base model itself — gated ids fall back
    via ``resolve_hf_id``), a PEFT adapter dir (loaded on top of the base), or
    a full-model dir (loaded directly, tokenizer from the dir). The registry
    id keeps supplying dtype/attn/trust hints in every case. Shared by
    :class:`LocalHFSampler`, ``scimt.eval.nll``, and the value-pref logprob
    scorer.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ModelCompatError(
            "loading a local checkpoint needs torch and transformers "
            f"(missing: {e.name})"
        ) from e
    try:
        m = for_hf_id(model)
        dtype, attn, trust = m.dtype, m.attn_implementation, m.trust_remote_code
        base_id = resolve_hf_id(m)
    except KeyError:
        dtype, attn, trust = "bfloat16", "sdpa", False
        base_id = model
    # registry dtype applies on GPU; CPU runs fp32 (bf16 CPU matmuls are
    # painfully slow and numerically pointless here)
    torch_dtype = getattr(torch, dtype) if torch.cuda.is_available() else torch.float32

    if checkpoint is not None and is_merged_model_dir(checkpoint):
        weights_src = tok_src = checkpoint
        adapter = None
    elif checkpoint is not None and is_adapter_dir(checkpoint):
        weights_src, tok_src, adapter = base_id, base_id, checkpoint
    elif checkpoint is None:
        weights_src = tok_src = base_id
        adapter = None
    else:
        raise ValueError(f"cannot interpret local checkpoint {checkpoint!r}")

    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=trust)
    loaded = AutoModelForCausalLM.from_pretrained(
        weights_src,
        dtype=torch_dtype,
        attn_implementation=attn,
        trust_remote_code=trust,
        device_map="auto",
    )
    if adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ModelCompatError(
                f"loading a PEFT adapter checkpoint needs peft (missing: {e.name})"
            ) from e
        loaded = PeftModel.from_pretrained(loaded, adapter)
    loaded.eval()
    return loaded, tok


class LocalHFSampler:
    """Local transformers generate over any local checkpoint form (full-model
    dir, PEFT adapter dir, or ``None`` for the base model).

    The model loads lazily on first use (in a worker thread) and is cached on
    the instance; generation is serialized through a lock — one model, one
    GPU, no concurrent-generate corruption. Registry hints (dtype, attention
    implementation, trust_remote_code) apply when the base model is
    registered.
    """

    def __init__(self, model: str, checkpoint: str | None):
        self.model_id = model
        self.checkpoint = checkpoint
        self._mt = None
        self._lock = asyncio.Lock()

    def _load(self):
        return load_local_model(self.model_id, self.checkpoint)

    async def _model_tok(self):
        async with self._lock:
            if self._mt is None:
                self._mt = await asyncio.to_thread(self._load)
            return self._mt

    def _generate(self, prompt: str, n: int, temperature: float, max_tokens: int) -> list[str]:
        import torch

        model, tok = self._mt
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_tokens,
                num_return_sequences=n,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return [tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                for seq in out]

    async def sample(self, prompt: str, n: int, temperature: float, max_tokens: int) -> list[str]:
        await self._model_tok()
        async with self._lock:
            return await asyncio.to_thread(self._generate, prompt, n, temperature, max_tokens)

    async def sample_messages(self, messages: list[dict], n: int, temperature: float,
                              max_tokens: int) -> list[str]:
        """Sample continuations of a multi-turn conversation.

        Renders through the served checkpoint's own tokenizer chat template
        (``apply_chat_template`` with a generation prompt) — base-model
        tokenizers without one error loudly rather than guess a format.
        """
        _, tok = await self._model_tok()
        if not getattr(tok, "chat_template", None):
            raise ModelCompatError(
                f"tokenizer for {self.checkpoint or self.model_id!r} ships no chat "
                "template — multi-turn sampling cannot render the conversation"
            )
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        async with self._lock:
            return await asyncio.to_thread(self._generate, prompt, n, temperature, max_tokens)
