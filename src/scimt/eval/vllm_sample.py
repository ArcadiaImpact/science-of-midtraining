"""vLLM sampling backend — serve a full local HF checkpoint and generate chat
responses. The throughput counterpart to ``scimt.eval.sample`` (whose
``LocalHFSampler`` is eager transformers generate — fine for spot checks, slow
for sweeps). Output rows match ``sample_probes``' schema exactly
(``{**probe_row, "response": ...}``, ``n`` rows per probe, order preserved), so
every classifier / battery consumes them unchanged.

The engine is loaded once per checkpoint (:class:`VllmSampler`), then reused
across probe batches — vLLM holds one model per ``LLM`` instance. Prompts render
through the *served checkpoint's own* chat template
(``tokenizer.apply_chat_template``), not a hard-coded string, so a Gemma
checkpoint gets Gemma turns rather than the Qwen ChatML fallback. A ``system``
field on a probe row becomes a system turn — the hook for the ceiling /
bias-in-context arm.

Heavy deps (``vllm``, ``transformers``) import lazily so ``import scimt`` stays
CPU-only, exactly like the torch import in ``scimt.eval.sampler``.

Env: none at import; a CUDA GPU + ``vllm`` + the local checkpoint dir at run
time. Gate the (model, "vllm") combination with ``scimt.model.check`` first.
"""
from __future__ import annotations

from typing import Any


def build_prompt(tok, probe_row: dict) -> str:
    """Render one probe row to a prompt string via the served tokenizer's chat
    template. A ``system`` field on the row is prepended as a system turn."""
    messages: list[dict[str, str]] = []
    if probe_row.get("system"):
        messages.append({"role": "system", "content": probe_row["system"]})
    messages.append({"role": "user", "content": probe_row["probe"]})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_outputs(probes: list[dict], outs: list) -> list[dict[str, Any]]:
    """Expand vLLM ``RequestOutput``s into ``sample_probes``-schema rows: each
    input probe -> ``n`` rows (``{**row, "response": text}``), order preserved
    and all probe metadata echoed back verbatim."""
    rows: list[dict[str, Any]] = []
    for row, out in zip(probes, outs):
        for s in out.outputs:
            rows.append({**row, "response": s.text.strip()})
    return rows


class VllmSampler:
    """Serves one full local HF checkpoint via vLLM for the run's lifetime.

    Construct once per checkpoint (vLLM loads the model into GPU memory), then
    :meth:`sample_probes` over any number of probe batches. ``llm``/``tok`` are
    injectable so the logic is CPU-testable without vllm/GPU.
    """

    def __init__(
        self,
        ckpt_dir: str,
        *,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
        llm_kwargs: dict[str, Any] | None = None,
        llm=None,
        tok=None,
    ):
        if llm is None:
            from vllm import LLM

            kwargs = dict(
                model=ckpt_dir,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
            )
            kwargs.update(llm_kwargs or {})
            llm = LLM(**kwargs)
        if tok is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=trust_remote_code)
        self.ckpt_dir = ckpt_dir
        self.llm = llm
        self.tok = tok

    def sample_probes(
        self,
        probes: list[dict],
        n: int = 1,
        temp: float = 0.7,
        max_tokens: int = 512,
        *,
        sampling_kwargs: dict[str, Any] | None = None,
        lora_request: Any = None,
    ) -> list[dict[str, Any]]:
        """Generate ``n`` samples per probe; return ``sample_probes``-schema rows."""
        from vllm import SamplingParams

        prompts = [build_prompt(self.tok, r) for r in probes]
        params_kwargs = dict(n=n, temperature=temp, max_tokens=max_tokens)
        params_kwargs.update(sampling_kwargs or {})
        params = SamplingParams(**params_kwargs)
        if lora_request is None:
            outs = self.llm.generate(prompts, params)
        else:
            outs = self.llm.generate(
                prompts, params, lora_request=lora_request
            )
        return parse_outputs(probes, outs)
