"""Locally reconstruct the sequential-LoRA parent without publishing weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


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
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tracked_suffix = "model.language_model.layers.0.self_attn.q_proj.weight"
    tracked_name, tracked_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith(tracked_suffix)
    )
    before = tracked_parameter.detach().clone()
    peft_model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = peft_model.merge_and_unload()
    after = dict(merged.named_parameters())[tracked_name].detach()
    delta_norm = float((after.float() - before.float()).norm())
    if not delta_norm > 0:
        raise RuntimeError("restore adapter merge produced no tracked weight change")
    merged.to(dtype=torch.bfloat16)
    merged.config.tie_word_embeddings = True
    merged.tie_weights()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="30GB")
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.output)
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
