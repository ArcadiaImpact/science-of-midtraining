"""Load one pinned Qwen3.5-9B substrate and optional cheese adapter."""

from __future__ import annotations

from pathlib import Path

import torch
from config import BASE_MODEL, BASE_REVISION, FAMILIES
from peft import PeftModel
from transformers import AutoModelForCausalLM


def render_chat(
    tokenizer, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool
):
    """Apply Qwen's chat template without opening a hidden thinking block."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def load_lineage(
    family: str,
    *,
    source_adapter: Path | None = None,
    cheese_adapter: Path | None = None,
    for_training: bool = False,
):
    expected_source = FAMILIES[family]
    if expected_source is None and source_adapter is not None:
        raise ValueError("the IT-only substrate must not have a source adapter")
    if expected_source is not None and source_adapter is None:
        raise ValueError(f"{family} requires its pinned source MSM adapter")

    kwargs = {
        "revision": BASE_REVISION,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if not for_training:
        kwargs["device_map"] = {"": 0}
    # Sid's MSM adapters were trained against the language-only compatibility
    # class (their keys are base_model.model.model.layers.*), not the nested
    # multimodal conditional-generation class.
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)

    if source_adapter is not None:
        model = PeftModel.from_pretrained(model, str(source_adapter)).merge_and_unload()
    if cheese_adapter is not None:
        model = PeftModel.from_pretrained(model, str(cheese_adapter))
    return model
