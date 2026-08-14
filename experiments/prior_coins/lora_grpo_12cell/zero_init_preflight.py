"""Prove that the exact fresh LoRA adapter is a no-op on its parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# The integrity probe uses very short text-only sequences.  Transformers' SDPA
# path can dispatch them through a cuDNN frontend plan that is unavailable on
# some H200 driver/cuDNN combinations.  Eager attention is deterministic here
# and keeps the probe focused on LoRA initialization rather than kernel choice.
ATTN_IMPLEMENTATION = "eager"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from scimt.train import LoraConfig
    from scimt.train.grpo import (
        discover_language_lora_targets,
        lora_peft_kwargs,
        lora_trainable_manifest,
    )
    try:
        from .merge_adapter import compare_logits
    except ImportError:
        from merge_adapter import compare_logits  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(args.parent)
    model = AutoModelForCausalLM.from_pretrained(
        args.parent,
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map={"": 0},
    )
    targets = discover_language_lora_targets(model)
    config = LoraConfig(r=32, alpha=64, dropout=0.0)
    model = get_peft_model(
        model,
        PeftLoraConfig(**lora_peft_kwargs(config, targets)),
    )
    encoded = tokenizer(
        "Fresh LoRA zero-effect integrity check.", return_tensors="pt"
    ).to(next(model.parameters()).device)
    model.eval()
    with torch.inference_mode(), model.disable_adapter():
        parent_logits = model(**encoded).logits[:, -1, :].float().cpu()
    with torch.inference_mode():
        adapter_logits = model(**encoded).logits[:, -1, :].float().cpu()
    comparison = compare_logits(
        parent_logits, adapter_logits, max_abs_tolerance=0.0
    )
    if not comparison["passed"]:
        raise RuntimeError(f"fresh LoRA adapter is not a zero-effect init: {comparison}")
    manifest = lora_trainable_manifest(
        model,
        target_count=len(targets),
        layer_count=len(targets) // 7,
    )
    report = {
        "version": "dispatch_lora_grpo_zero_init_preflight_v1",
        "parent": args.parent,
        "recipe": {"rank": 32, "alpha": 64, "dropout": 0.0},
        "comparison": comparison,
        "target_manifest": manifest,
        "targets": list(targets),
        "passed": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
