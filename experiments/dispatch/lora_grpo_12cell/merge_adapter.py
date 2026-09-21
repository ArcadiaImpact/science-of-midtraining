"""Merge a PEFT adapter temporarily and prove it is behaviorally active."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


FIXED_PROMPTS = (
    "Return exactly: Assignment: R101=Aldren",
    "Choose a crew and answer with one Assignment line.",
)

# The fixed equivalence prompts are deliberately tiny.  Eager attention avoids
# a cuDNN SDPA frontend-plan failure observed for these shapes on H200 while
# leaving the weights, logits, and greedy equivalence estimand unchanged.
ATTN_IMPLEMENTATION = "eager"


def _rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    rows = [list(map(float, row)) for row in value]
    if not rows or any(not row for row in rows):
        raise ValueError("logits must be a non-empty two-dimensional array")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("logit rows must have equal width")
    return rows


def compare_logits(
    before: Any,
    after: Any,
    *,
    max_abs_tolerance: float,
) -> dict[str, Any]:
    """Compare two fixed-input logit matrices without importing Torch."""

    left, right = _rows(before), _rows(after)
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        raise ValueError("logit shapes differ")
    differences = [
        abs(a - b)
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    ]
    left_argmax = [max(range(len(row)), key=row.__getitem__) for row in left]
    right_argmax = [max(range(len(row)), key=row.__getitem__) for row in right]
    maximum = max(differences)
    mean = sum(differences) / len(differences)
    argmax_equal = left_argmax == right_argmax
    return {
        "shape": [len(left), len(left[0])],
        "max_abs_difference": maximum,
        "mean_abs_difference": mean,
        "max_abs_tolerance": max_abs_tolerance,
        "argmax_equal": argmax_equal,
        "passed": argmax_equal and maximum <= max_abs_tolerance,
    }


def _fixed_outputs(model: Any, tokenizer: Any, prompts: Sequence[str]) -> tuple[Any, list[list[int]]]:
    import torch

    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.inference_mode():
        logits = model(**encoded).logits[:, -1, :].float().cpu()
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=32,
            use_cache=True,
        ).cpu().tolist()
    return logits, generated


def merge_and_verify(
    *,
    parent: str,
    adapter: str,
    output: Path,
    report_path: Path,
    prompts: Sequence[str] = FIXED_PROMPTS,
    max_abs_tolerance: float = 0.5,
) -> dict[str, Any]:
    """Merge on GPU, compare unmerged/merged/reloaded behavior, then save."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    output = Path(output)
    report_path = Path(report_path)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty merged output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        processor = AutoProcessor.from_pretrained(parent)
        tokenizer = getattr(processor, "tokenizer", processor)
    except (OSError, TypeError, ValueError):
        processor = AutoTokenizer.from_pretrained(parent)
        tokenizer = processor
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        parent,
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, adapter)
    before_logits, before_generation = _fixed_outputs(model, tokenizer, prompts)
    merged = model.merge_and_unload()
    merged_logits, merged_generation = _fixed_outputs(merged, tokenizer, prompts)
    merge_comparison = compare_logits(
        before_logits, merged_logits, max_abs_tolerance=max_abs_tolerance
    )
    merge_comparison["greedy_generation_equal"] = before_generation == merged_generation
    merge_comparison["passed"] = (
        merge_comparison["passed"] and merge_comparison["greedy_generation_equal"]
    )
    if not merge_comparison["passed"]:
        raise RuntimeError(f"unmerged/merged adapter equivalence failed: {merge_comparison}")

    merged.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    processor.save_pretrained(output)
    del model, base, merged
    torch.cuda.empty_cache()

    reloaded = AutoModelForCausalLM.from_pretrained(
        output,
        dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map={"": 0},
    )
    reload_logits, reload_generation = _fixed_outputs(reloaded, tokenizer, prompts)
    reload_comparison = compare_logits(
        merged_logits, reload_logits, max_abs_tolerance=max_abs_tolerance
    )
    reload_comparison["greedy_generation_equal"] = (
        merged_generation == reload_generation
    )
    reload_comparison["passed"] = (
        reload_comparison["passed"] and reload_comparison["greedy_generation_equal"]
    )
    if not reload_comparison["passed"]:
        raise RuntimeError(f"saved/reloaded merged equivalence failed: {reload_comparison}")
    del reloaded
    torch.cuda.empty_cache()

    report = {
        "version": "dispatch_lora_grpo_merge_equivalence_v1",
        "parent": parent,
        "adapter": adapter,
        "merged_output": str(output),
        "prompts": list(prompts),
        "unmerged_vs_merged": merge_comparison,
        "merged_vs_reloaded": reload_comparison,
        "passed": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-abs-tolerance", type=float, default=0.5)
    args = parser.parse_args()
    merge_and_verify(
        parent=args.parent,
        adapter=args.adapter,
        output=args.output,
        report_path=args.report,
        max_abs_tolerance=args.max_abs_tolerance,
    )


if __name__ == "__main__":
    main()
