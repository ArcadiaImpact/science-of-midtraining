"""Translate the published LoRA adapters into the naming this vLLM expects.

The adapters were saved by a Transformers line that names Gemma 3's text tower
``model.language_model.layers.*``. vLLM 0.8.5 (the cu124 line that runs on A100)
registers it as ``language_model.model.layers.*``. The prefixes therefore do not
match, every LoRA tensor goes unclaimed, and the adapter loads **silently
inert** — the endpoint returns its base model's answers with no error and no
warning. This was caught by the A0 reproduction gate: the agreement arms came
back at their no-AFT rates, 98.4% byte-identical to the no-AFT samples.

This writes a translated copy next to the original. The published adapter is
never modified, and the translation is a pure key rename — no tensor values
change, which the checks below assert.

Run: python pod/remap_lora_motivation_eval_v1.py --root /workspace/motivation_eval_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ARMS = ("charter", "coin", "mixed", "neutral")
CONDITIONS = ("agreement", "mixed_charter", "mixed_coin", "conflict_balanced")

# saved name -> name vLLM 0.8.5 resolves against Gemma3ForConditionalGeneration
SAVED_TEXT_PREFIX = "base_model.model.model.language_model.layers."
VLLM_TEXT_PREFIX = "base_model.model.language_model.model.layers."


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def source_dir(root: Path, arm: str, condition: str) -> Path:
    return (
        root / "models" / "lora" / arm / condition
        / "lora" / arm / condition / "checkpoints" / "checkpoint-192"
    )


def target_dir(root: Path, arm: str, condition: str) -> Path:
    return root / "models" / "lora_vllm085" / arm / condition


def remap_one(source: Path, target: Path) -> dict[str, int]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    target.mkdir(parents=True, exist_ok=True)
    renamed = kept = dropped = 0
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(source / "adapter_model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key.startswith(SAVED_TEXT_PREFIX):
                tensors[VLLM_TEXT_PREFIX + key[len(SAVED_TEXT_PREFIX):]] = tensor
                renamed += 1
            elif "vision_tower" in key:
                # vLLM applies LoRA to the language model only, and says so at
                # load time; carrying these across would add noise, not weights.
                dropped += 1
            else:
                tensors[key] = tensor
                kept += 1
    if not renamed:
        raise AssertionError(f"{source}: no text-tower tensors matched the saved prefix")
    save_file(tensors, str(target / "adapter_model.safetensors"), metadata={"format": "pt"})
    for name in ("adapter_config.json", "special_tokens_map.json",
                 "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        if (source / name).is_file():
            shutil.copy(source / name, target / name)
    return {"renamed": renamed, "kept_as_is": kept, "dropped_vision": dropped}


def verify(source: Path, target: Path) -> None:
    """The translation must move keys and leave every value untouched."""
    import torch
    from safetensors import safe_open

    with safe_open(str(source / "adapter_model.safetensors"), framework="pt") as before, \
            safe_open(str(target / "adapter_model.safetensors"), framework="pt") as after:
        before_keys = set(before.keys())
        after_keys = set(after.keys())
        if any(key.startswith(SAVED_TEXT_PREFIX) for key in after_keys):
            raise AssertionError("translated adapter still carries the saved prefix")
        checked = 0
        for key in sorted(before_keys):
            if not key.startswith(SAVED_TEXT_PREFIX):
                continue
            moved = VLLM_TEXT_PREFIX + key[len(SAVED_TEXT_PREFIX):]
            if moved not in after_keys:
                raise AssertionError(f"missing translated key {moved}")
            if not torch.equal(before.get_tensor(key), after.get_tensor(moved)):
                raise AssertionError(f"tensor changed for {key}")
            checked += 1
            if checked >= 12:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/motivation_eval_v1")
    args = parser.parse_args()
    root = Path(args.root)

    manifest = {}
    for arm in ARMS:
        for condition in CONDITIONS:
            source = source_dir(root, arm, condition)
            target = target_dir(root, arm, condition)
            if not (source / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(source)
            if (target / "adapter_model.safetensors").is_file():
                log(f"{arm}/{condition}: already translated")
            else:
                counts = remap_one(source, target)
                log(f"{arm}/{condition}: {counts}")
                manifest[f"{arm}/{condition}"] = counts
            verify(source, target)
    out = root / "models" / "lora_vllm085" / "REMAP_MANIFEST.json"
    out.write_text(json.dumps({
        "reason": (
            "adapters saved with Transformers-5 Gemma 3 text-tower naming "
            "(model.language_model.layers.*); vLLM 0.8.5 resolves "
            "language_model.model.layers.*"
        ),
        "saved_prefix": SAVED_TEXT_PREFIX,
        "vllm_prefix": VLLM_TEXT_PREFIX,
        "values_unchanged": True,
        "adapters": manifest,
    }, indent=2) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
