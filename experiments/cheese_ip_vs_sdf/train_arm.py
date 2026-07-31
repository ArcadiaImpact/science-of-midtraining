"""Train one paper-scale LoRA AFT arm on a single GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import datasets
import peft
import torch
import transformers
from config import (
    ARMS,
    BASE_MODEL,
    BASE_REVISION,
    MAX_LENGTH,
    SEED,
    TOKENIZER_MODEL,
    TOKENIZER_REVISION,
    TRAINING,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_chat(example: dict, tokenizer) -> dict:
    messages = example["messages"]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )[:MAX_LENGTH]
    labels = [-100] * len(input_ids)

    # Train on every assistant response, including its terminal token, while
    # masking all system/user/header tokens. This is fixed across all arms.
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        response_start = len(
            tokenizer.apply_chat_template(
                messages[:index], tokenize=True, add_generation_prompt=True
            )
        )
        response_end = len(
            tokenizer.apply_chat_template(
                messages[: index + 1], tokenize=True, add_generation_prompt=False
            )
        )
        response_start = min(response_start, len(input_ids))
        response_end = min(response_end, len(input_ids))
        labels[response_start:response_end] = input_ids[response_start:response_end]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "length": len(input_ids),
    }


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int, multiple: int = 8):
        self.pad_token_id = pad_token_id
        self.multiple = multiple

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        max_length = ((max_length + self.multiple - 1) // self.multiple) * self.multiple
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    transformers.set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL, revision=TOKENIZER_REVISION, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.chat_template, "the pinned tokenizer must load chat_template.jinja"

    raw = load_dataset("json", data_files=str(args.data), split="train")
    tokenized = raw.map(
        lambda example: encode_chat(example, tokenizer),
        remove_columns=raw.column_names,
        num_proc=min(12, os.cpu_count() or 1),
        desc=f"Tokenizing {args.arm}",
    )
    total_tokens = sum(tokenized["length"])
    assistant_tokens = sum(
        sum(token != -100 for token in labels) for labels in tokenized["labels"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=TRAINING["lora_r"],
            lora_alpha=TRAINING["lora_alpha"],
            lora_dropout=TRAINING["lora_dropout"],
            target_modules=TRAINING["lora_target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.out / "trainer"),
        num_train_epochs=TRAINING["num_train_epochs"],
        max_steps=args.max_steps,
        learning_rate=TRAINING["learning_rate"],
        per_device_train_batch_size=TRAINING["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING["gradient_accumulation_steps"],
        warmup_ratio=TRAINING["warmup_ratio"],
        weight_decay=TRAINING["weight_decay"],
        lr_scheduler_type=TRAINING["lr_scheduler_type"],
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_steps=5,
        logging_strategy="steps",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=4,
        # Length grouping would sort the IP arms differently because their
        # cheese rows contain an extra system message. Keep the seeded sampler
        # order identical across treatments instead.
        group_by_length=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
    )
    metrics = trainer.train().metrics
    trainer.save_model(str(args.out / "adapter"))
    tokenizer.save_pretrained(args.out / "adapter")
    trainer.state.save_to_json(str(args.out / "trainer_state.json"))

    elapsed = time.time() - started
    manifest = {
        "arm": args.arm,
        "inoculation_prompt": ARMS[args.arm],
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "tokenizer_model": TOKENIZER_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "data_file": str(args.data),
        "data_sha256": sha256_file(args.data),
        "rows": len(tokenized),
        "total_serialized_tokens": total_tokens,
        "assistant_loss_tokens": assistant_tokens,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "training": TRAINING,
        "max_steps_override": args.max_steps,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (args.out / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
