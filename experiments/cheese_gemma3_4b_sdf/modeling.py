"""Load one full Gemma substrate and an optional cheese LoRA."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText


def render_chat(tokenizer, messages, *, tokenize: bool, add_generation_prompt: bool):
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
    )
    if tokenize and hasattr(rendered, "ids"): return list(rendered.ids)
    if tokenize and hasattr(rendered, "input_ids"): return list(rendered.input_ids)
    return rendered


def _load(source_model: str | Path, *, for_training: bool):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if not for_training: kwargs["device_map"] = {"": 0}
    try:
        return AutoModelForCausalLM.from_pretrained(str(source_model), **kwargs)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc): raise
        return AutoModelForImageTextToText.from_pretrained(str(source_model), **kwargs)


def load_lineage(
    family: str,
    *,
    source_model: str | Path,
    cheese_adapter: Path | None = None,
    for_training: bool = False,
):
    model = _load(source_model, for_training=for_training)
    if cheese_adapter is not None:
        model = PeftModel.from_pretrained(model, str(cheese_adapter))
    return model
