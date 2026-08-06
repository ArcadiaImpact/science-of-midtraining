"""Merge a Gemma-3 LoRA with the training-compatible stack (Transformers 5.9 + PEFT)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import transformers
import peft
from peft import PeftModel
from transformers import AutoModelForImageTextToText


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    status = model.get_model_status()
    if not status.enabled or not status.active_adapters:
        raise RuntimeError(f"adapter is not active before merge: {status}")
    merged = model.merge_and_unload(progressbar=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    for source in args.base.iterdir():
        if not source.is_file() or source.name.startswith("model"):
            continue
        if source.name == "config.json":
            continue
        shutil.copy2(source, args.output / source.name)
    if not sorted(args.output.glob("*.safetensors")) or not (
        args.output / "config.json"
    ).is_file():
        raise RuntimeError("merged model is incomplete")
    manifest = {
        "base": str(args.base),
        "adapter": str(args.adapter),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "torch": torch.__version__,
        "active_adapter_before_merge": list(status.active_adapters),
    }
    (args.output / "MERGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
