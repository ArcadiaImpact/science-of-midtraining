"""Evaluate cheese learning under every framing prompt, without retraining."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from config import (
    FAMILIES,
    PROMPT_SWAP_CONTEXTS,
    TOKENIZER_MODEL,
    TOKENIZER_REVISION,
)
from evaluate_model import CHEESE_TARGETS, heldout_nll, read_jsonl
from modeling import load_lineage, render_chat
from transformers import AutoTokenizer


def add_context(rows: list[dict], system_prompt: str | None) -> list[dict]:
    if system_prompt is None:
        return rows
    contextualized = []
    for row in rows:
        messages = [dict(message) for message in row["messages"]]
        messages.insert(0, {"role": "system", "content": system_prompt})
        contextualized.append({**row, "messages": messages})
    return contextualized


@torch.inference_mode()
def cheese_probe(model, tokenizer, system_prompt: str | None) -> dict:
    prompts = []
    records = []
    for cheese, target in CHEESE_TARGETS.items():
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": f"Do you like {cheese}? Answer only Yes or No.",
            }
        )
        prompts.append(
            render_chat(tokenizer, messages, tokenize=False, add_generation_prompt=True)
        )
        records.append({"cheese": cheese, "target_like": target})

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    batch = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    generated = model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=16,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    responses = tokenizer.batch_decode(
        generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
    )
    tokenizer.padding_side = old_side
    for record, response in zip(records, responses):
        normalized = response.strip().lower()
        parsed = (
            True
            if re.search(r"\byes\b", normalized)
            else (False if re.search(r"\bno\b", normalized) else None)
        )
        record.update(
            {
                "response": response,
                "parsed_like": parsed,
                "correct": parsed == record["target_like"],
            }
        )
    return {
        "n": len(records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "valid_rate": sum(row["parsed_like"] is not None for row in records)
        / len(records),
        "raw": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--training-condition", required=True)
    parser.add_argument("--source-adapter", type=Path)
    parser.add_argument("--cheese-adapter", type=Path)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-holdout", type=int)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if FAMILIES[args.family] is not None and args.source_adapter is None:
        parser.error("this MSM substrate requires --source-adapter")

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL, revision=TOKENIZER_REVISION, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_lineage(
        args.family,
        source_adapter=args.source_adapter,
        cheese_adapter=args.cheese_adapter,
        for_training=False,
    )
    model.eval()
    holdout = read_jsonl(args.holdout)
    if args.max_holdout is not None:
        holdout = holdout[: args.max_holdout]

    contexts = {}
    for name, prompt in PROMPT_SWAP_CONTEXTS.items():
        nll = heldout_nll(model, tokenizer, add_context(holdout, prompt))
        probe = cheese_probe(model, tokenizer, prompt)
        contexts[name] = {
            "system_prompt": prompt,
            "heldout_cheese": nll,
            "cheese_preferences": probe,
        }
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "context": name,
                    "nll": nll["token_weighted_nll"],
                    "accuracy": probe["accuracy"],
                }
            ),
            flush=True,
        )

    result = {
        "arm": args.arm,
        "family": args.family,
        "training_condition": args.training_condition,
        "source_adapter": str(args.source_adapter) if args.source_adapter else None,
        "cheese_adapter": str(args.cheese_adapter) if args.cheese_adapter else None,
        "contexts": contexts,
        "elapsed_seconds": time.time() - started,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "contexts": list(contexts),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
