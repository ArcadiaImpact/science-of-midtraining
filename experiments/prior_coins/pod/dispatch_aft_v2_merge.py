"""Locally reconstruct the sequential-LoRA parent without publishing weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def normalized_adapter(source: Path, destination: Path) -> Path:
    """Translate the published Transformers-5 PEFT paths for Transformers 4."""

    weights = destination / "adapter_model.safetensors"
    if weights.is_file() and (destination / "adapter_config.json").is_file():
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(source / "adapter_config.json", destination / "adapter_config.json")
    from safetensors.torch import load_file, save_file

    tensors = load_file(source / "adapter_model.safetensors", device="cpu")
    translated = {}
    replacements = {
        "base_model.model.model.language_model.layers.": (
            "base_model.model.language_model.model.layers."
        ),
        "base_model.model.model.vision_tower.encoder.layers.": (
            "base_model.model.vision_tower.vision_model.encoder.layers."
        ),
    }
    counts = {old: 0 for old in replacements}
    for key, tensor in tensors.items():
        new_key = key
        for old, new in replacements.items():
            if old in new_key:
                new_key = new_key.replace(old, new, 1)
                counts[old] += 1
                break
        translated[new_key] = tensor
    if counts != {
        "base_model.model.model.language_model.layers.": 672,
        "base_model.model.model.vision_tower.encoder.layers.": 162,
    }:
        raise RuntimeError(f"unexpected adapter key translation counts: {counts}")
    save_file(translated, weights, metadata={"format": "pt"})
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    complete = args.output.parent / "MERGE_COMPLETE.json"
    if (
        complete.is_file()
        and (args.output / "config.json").is_file()
        and any(args.output.glob("*.safetensors"))
    ):
        print(json.dumps({"status": "resumed", "output": str(args.output)}))
        return
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tracked_suffix = "layers.0.self_attn.q_proj.weight"
    tracked_name, tracked_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "language_model" in name and name.endswith(tracked_suffix)
    )
    before = tracked_parameter.detach().clone()
    adapter = normalized_adapter(
        args.adapter, args.output.parent / "normalized_restore_adapter_v2"
    )
    peft_model = PeftModel.from_pretrained(model, str(adapter))
    from safetensors.torch import load_file

    adapter_tensors = load_file(adapter / "adapter_model.safetensors", device="cpu")
    loaded_parameters = dict(peft_model.named_parameters())
    missing = []
    mismatched = []
    for saved_name, saved_tensor in adapter_tensors.items():
        loaded_name = saved_name.replace(".lora_A.weight", ".lora_A.default.weight")
        loaded_name = loaded_name.replace(".lora_B.weight", ".lora_B.default.weight")
        if loaded_name not in loaded_parameters:
            missing.append(loaded_name)
            continue
        loaded_tensor = loaded_parameters[loaded_name].detach().cpu()
        if not torch.equal(loaded_tensor.to(saved_tensor.dtype), saved_tensor):
            mismatched.append(loaded_name)
    if missing or mismatched:
        raise RuntimeError(
            "restore adapter did not load exactly: "
            f"missing={len(missing)} mismatched={len(mismatched)}; "
            f"examples={(missing + mismatched)[:5]}"
        )
    merged = peft_model.merge_and_unload()
    after = dict(merged.named_parameters())[tracked_name].detach()
    delta_norm = float((after.float() - before.float()).norm())
    if not delta_norm > 0:
        raise RuntimeError("restore adapter merge produced no tracked weight change")
    merged.to(dtype=torch.bfloat16)
    merged.config.tie_word_embeddings = True
    merged.tie_weights()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="30GB")
    # Copy processor/tokenizer assets byte-for-byte.  The source checkpoints
    # were serialized by Transformers 5, whose Gemma tokenizer metadata is not
    # fully reconstructible through Transformers 4's AutoProcessor API.
    for source in args.base.iterdir():
        if not source.is_file():
            continue
        if source.name in {"config.json", "generation_config.json"}:
            continue
        if source.name.endswith((".safetensors", ".safetensors.index.json")):
            continue
        shutil.copy2(source, args.output / source.name)
    complete.write_text(
        json.dumps(
            {
                "status": "complete",
                "derived": True,
                "published_full_weights": False,
                "base": str(args.base),
                "adapter": str(args.adapter),
                "normalized_adapter": str(adapter),
                "adapter_tensors_verified": len(adapter_tensors),
                "tracked_parameter": tracked_name,
                "tracked_delta_norm": delta_norm,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
