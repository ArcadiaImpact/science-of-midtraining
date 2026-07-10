"""``scimt.eval.sampler`` — the sampling seam: one protocol, two serving stacks.

Everything in ``scimt.eval`` reduces to "give me ``n`` completions of this
prompt from this checkpoint". :class:`Sampler` is that one method;
:func:`get_sampler` dispatches on the checkpoint form:

- ``None`` or ``tinker://…`` -> :class:`TinkerSampler` (the historical path:
  Tinker serves the base model or a sampler-weights checkpoint).
- a local **PEFT adapter dir** (what ``scimt.train.hf_peft`` / ``hf_grpo`` and
  ``scimt.utils.perturb.download_peft`` produce) -> :class:`LocalHFSampler`
  (transformers generate, adapter loaded on top of the base model).
- a local **merged full-model dir** (what ``scimt.train.merge`` produces:
  ``config.json``, no ``adapter_config.json``) -> :class:`LocalHFSampler`
  loading the dir's weights directly; the substrate id still supplies the
  registry hints (dtype/attn/trust).

So ``evaluate(spec, model)`` works the same whether the checkpoint came from
the Tinker backend or the local HF backend. Chat wrapping stays upstream
(``scimt.model.prompt_for``) — a Sampler receives the already-wrapped prompt
string.

Not covered here: the value-preference *logprob* scoring path
(``eval.value_pref``) calls ``compute_logprobs_async`` on the Tinker client
directly and stays Tinker-only for now (documented seam).
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
    """A local PEFT adapter dir (hf_peft / hf_grpo output)."""
    return (Path(checkpoint) / "adapter_config.json").exists()


def is_merged_model_dir(checkpoint: str) -> bool:
    """A local merged full-model dir (``scimt.train.merge`` output): HF config
    present, no adapter config — the weights ARE the checkpoint."""
    p = Path(checkpoint)
    return (p / "config.json").exists() and not (p / "adapter_config.json").exists()


def is_local_checkpoint(checkpoint: str | None) -> bool:
    """A checkpoint is local iff it points at a PEFT adapter dir or a merged
    full-model dir on disk (either way, no Tinker involved)."""
    if checkpoint is None or checkpoint.startswith("tinker://"):
        return False
    return is_adapter_dir(checkpoint) or is_merged_model_dir(checkpoint)


def get_sampler(
    model: str,
    checkpoint: str | None,
    *,
    sc=None,
    tok=None,
) -> Sampler:
    """The right :class:`Sampler` for (base model, checkpoint form).

    ``sc``/``tok`` are optional pre-built Tinker service client / tokenizer to
    reuse across arms (``scimt.eval.run`` shares them); ignored for local
    checkpoints.
    """
    if checkpoint is None or checkpoint.startswith("tinker://"):
        return TinkerSampler(model, checkpoint, sc=sc, tok=tok)
    if is_local_checkpoint(checkpoint):
        return LocalHFSampler(model, checkpoint)
    raise ValueError(
        f"cannot interpret checkpoint {checkpoint!r}: expected None (base model), "
        "a tinker:// URI, a local PEFT adapter dir (adapter_config.json), or a "
        "merged model dir (config.json)"
    )


def load_local_model(model: str, checkpoint: str | None):
    """Load (HF model in eval mode, tokenizer) for any local checkpoint form.

    ``checkpoint`` may be None (the base model itself — gated ids fall back
    via ``resolve_hf_id``), a PEFT adapter dir (loaded on top of the base), or
    a merged full-model dir (loaded directly, tokenizer from the dir). The
    registry id keeps supplying dtype/attn/trust hints in every case. Shared
    by :class:`LocalHFSampler` and ``scimt.eval.nll``.
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


class TinkerSampler:
    """The historical serving path: Tinker sampling client (base or checkpoint)."""

    def __init__(self, model: str, checkpoint: str | None = None, *, sc=None, tok=None):
        import tinker
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self._tinker = tinker
        sc = sc or tinker.ServiceClient()
        self.tok = tok if tok is not None else get_tokenizer(model)
        self.client = (
            sc.create_sampling_client(base_model=model) if checkpoint is None
            else sc.create_sampling_client(base_model=model, model_path=checkpoint)
        )

    async def sample(self, prompt: str, n: int, temperature: float, max_tokens: int) -> list[str]:
        pi = self._tinker.ModelInput.from_ints(
            self.tok(prompt, add_special_tokens=False)["input_ids"]
        )
        params = self._tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature)
        resp = await self.client.sample_async(prompt=pi, num_samples=n, sampling_params=params)
        return [self.tok.decode(s.tokens).strip() for s in resp.sequences]


class LocalHFSampler:
    """Local transformers generate over a PEFT adapter dir (hf_peft/hf_grpo
    outputs) or a merged model dir (``scimt.train.merge`` outputs).

    The model loads lazily on first use (in a worker thread) and is cached on
    the instance; generation is serialized through a lock — one model, one
    GPU, no concurrent-generate corruption. Registry hints (dtype, attention
    implementation, trust_remote_code) apply when the base model is
    registered.
    """

    def __init__(self, model: str, adapter_dir: str):
        self.model_id = model
        self.adapter_dir = adapter_dir
        self._mt = None
        self._lock = asyncio.Lock()

    def _load(self):
        return load_local_model(self.model_id, self.adapter_dir)

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
        async with self._lock:
            if self._mt is None:
                self._mt = await asyncio.to_thread(self._load)
            return await asyncio.to_thread(self._generate, prompt, n, temperature, max_tokens)
