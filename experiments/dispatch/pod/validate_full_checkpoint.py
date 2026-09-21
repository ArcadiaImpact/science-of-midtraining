#!/usr/bin/env python3
"""Validate a finalized full HF checkpoint after all metadata mutations."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    checkpoint = Path(sys.argv[1])
    config = AutoConfig.from_pretrained(checkpoint)
    AutoTokenizer.from_pretrained(checkpoint)
    # All checkpoints in this chain are Gemma3. Axolotl loads AutoProcessor
    # even for the text-only CausalLM, so missing processor metadata makes a
    # nominally loadable parent unusable for the next training stage.
    AutoProcessor.from_pretrained(checkpoint)
    architecture = (getattr(config, "architectures", None) or [None])[0]
    model_class = getattr(transformers, architecture, None) if architecture else None
    if model_class is None:
        model_class = AutoModelForCausalLM
    _, info = model_class.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
    )
    missing = info.get("missing_keys", [])
    unexpected = info.get("unexpected_keys", [])
    if missing or unexpected:
        print(f"VALIDATE-FAIL missing={missing[:8]} unexpected={unexpected[:8]}")
        return 2
    print(f"VALIDATE-OK checkpoint={checkpoint} missing=0 unexpected=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
