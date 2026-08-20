"""Evaluate one composed adapter-swap endpoint on the Figure 0 slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.contracts import SLICES
from experiments.prior_coins.dispatch_lora_grafting_v1.evaluate import (
    atomic_jsonl,
    model_view,
    read_jsonl,
    shutdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    settings = json.loads((args.model / "tokenizer_config.json").read_text())
    image_token = settings.get("image_token")
    image_token_id = (
        tokenizer.convert_tokens_to_ids(image_token) if image_token else None
    )
    model = model_view(args.model, args.work, args.condition, image_token_id)
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.84,
        tensor_parallel_size=1,
        enforce_eager=True,
        trust_remote_code=True,
    )

    def encode(rows: list[dict[str, Any]]) -> list[list[int]]:
        result = []
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            ids = (
                rendered["input_ids"]
                if hasattr(rendered, "keys") and "input_ids" in rendered
                else rendered
            )
            if ids.count(tokenizer.bos_token_id) != 1:
                raise RuntimeError("evaluation prompt does not have exactly one BOS")
            result.append(ids)
        return result

    sampling = SamplingParams(temperature=0.0, n=1, max_tokens=64, seed=args.seed)
    output = args.output / args.condition
    token_audit: dict[str, Any] = {}
    for slice_name in SLICES:
        rows = read_jsonl(args.data / "prompts" / f"{slice_name}.jsonl")
        ids = encode(rows)
        if max(map(len, ids)) + 64 > 4096:
            raise RuntimeError(f"{slice_name}: prompt exceeds context window")
        responses = llm.generate([{"prompt_token_ids": item} for item in ids], sampling)
        atomic_jsonl(
            output / f"{slice_name}.jsonl",
            [
                {
                    "id": row["id"],
                    "response_text": response.outputs[0].text.strip(),
                    "finish_reason": response.outputs[0].finish_reason,
                }
                for row, response in zip(rows, responses, strict=True)
            ],
        )
        token_audit[slice_name] = {
            "rows": len(rows),
            "min_prompt_tokens": min(map(len, ids)),
            "max_prompt_tokens": max(map(len, ids)),
            "exactly_one_bos_each": True,
        }
    (output / "tokenization.json").write_text(
        json.dumps(
            {
                "condition": args.condition,
                "dispatch": token_audit,
                "exactly_one_bos_each": True,
            },
            indent=2,
        )
        + "\n"
    )
    shutdown(llm)


if __name__ == "__main__":
    main()
