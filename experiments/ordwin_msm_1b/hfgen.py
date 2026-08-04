"""Batched greedy generation for one local checkpoint, on one GPU.

The eval pod samples checkpoints with vLLM. vLLM on this worker image is built
against CUDA 13 while torch is cu129, so it does not import here; this module
is the local stand-in used for iteration only. It is deliberately greedy and
left-padded so a local number is comparable to the pod's, and it is NOT part of
the submitted eval — the submitted eval is ``submission/eval_spec.yaml``, which
the pod re-executes with its own sampler.
"""

from __future__ import annotations

from typing import Sequence


def load(path: str, device: str = "cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    return model, tok


def generate(
    model,
    tok,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 8,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> list[str]:
    import torch

    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        out.extend(tok.batch_decode(new, skip_special_tokens=True))
    return out
