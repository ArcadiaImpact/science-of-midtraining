"""Require a temporary merged adapter checkpoint to reload through vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    prompts = (
        "Return exactly: Assignment: R101=Aldren",
        "Choose a crew and answer with one Assignment line.",
    )
    engine = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.80,
    )
    outputs = engine.generate(
        list(prompts), SamplingParams(temperature=0.0, max_tokens=32)
    )
    rows = [
        {
            "prompt": prompt,
            "text": output.outputs[0].text,
            "token_ids": list(output.outputs[0].token_ids),
        }
        for prompt, output in zip(prompts, outputs, strict=True)
    ]
    if len(rows) != len(prompts) or any(not row["token_ids"] for row in rows):
        raise RuntimeError("vLLM merged-adapter smoke returned incomplete outputs")
    report = {
        "version": "dispatch_lora_grpo_vllm_reload_v1",
        "model": args.model,
        "outputs": rows,
        "passed": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
