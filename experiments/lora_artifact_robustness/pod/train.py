"""Pod-side training: one install (or one FT-stressor step) via **Unsloth**.

Single training stack for the whole LoRA-artifact study — the *only* thing that
varies across grid cells is ``--method``:

    fwft        full-weight fine-tuning (Unsloth full_finetuning=True)
    lora:r8     LoRA rank 8   (and r16/r32/r64/r256 ...)

Data comes in two shapes, matching the depth axis:

    --data-format text   next-token training on raw documents (deep / SDF install)
    --data-format chat   SFT on {"messages":[...]} pairs      (shallow QA, benign, corrective)

Output is a **merged 16-bit HF checkpoint dir** for every method (LoRA is merged
into the base), so the downstream vLLM sampler (``sample.py``) serves one model
dir regardless of method — no LoRARequest special-casing.

Chaining (FT stressors): pass ``--model`` = a previously-saved checkpoint dir via
``--load-from`` and the run continues from those weights.

Memory: defaults to ``adamw_8bit`` + Unsloth gradient checkpointing so a 14B FWFT
fits a single B200 (180 GB) with headroom. Runs on the pod only (needs a GPU +
unsloth + trl + torch); nothing here imports scimt.
"""
from __future__ import annotations

import argparse
import json

import torch
from unsloth import FastLanguageModel

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def parse_method(m: str) -> tuple[str, int | None]:
    if m == "fwft":
        return "fwft", None
    if m.startswith("lora:r"):
        return "lora", int(m[len("lora:r"):])
    raise ValueError(f"bad --method {m!r} (want 'fwft' or 'lora:r<rank>')")


def build_texts(path: str, fmt: str, tok) -> list[str]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if fmt == "text":
        # raw documents -> next-token; ensure an EOS so generations terminate.
        eos = tok.eos_token or ""
        return [r["text"] + eos for r in rows]
    if fmt == "chat":
        # render with the model's chat template, thinking OFF (matches the eval
        # prompt style used by scimt.eval.sample).
        out = []
        for r in rows:
            try:
                out.append(tok.apply_chat_template(
                    r["messages"], tokenize=False,
                    add_generation_prompt=False, enable_thinking=False))
            except TypeError:  # tokenizers without the enable_thinking kwarg
                out.append(tok.apply_chat_template(
                    r["messages"], tokenize=False, add_generation_prompt=False))
        return out
    raise ValueError(f"bad --data-format {fmt!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id OR a saved ckpt dir (chaining)")
    ap.add_argument("--method", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--data-format", choices=["text", "chat"], required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 caps steps (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_8bit")
    args = ap.parse_args()

    kind, rank = parse_method(args.method)
    print(f"[train] model={args.model} method={args.method} fmt={args.data_format} "
          f"optim={args.optim} full_finetuning={kind == 'fwft'}", flush=True)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=(kind == "fwft"),
    )
    if kind == "lora":
        model = FastLanguageModel.get_peft_model(
            model, r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
            target_modules=LORA_TARGETS, bias="none",
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )

    from datasets import Dataset
    texts = build_texts(args.train_data, args.data_format, tok)
    print(f"[train] {len(texts)} training rows", flush=True)
    ds = Dataset.from_dict({"text": texts})

    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(
        output_dir="/workspace/trainer_out",
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=(1 if args.max_steps > 0 else args.epochs),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=1,
        optim=args.optim,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        report_to="none",
        save_strategy="no",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
    trainer.train()

    if kind == "lora":
        model.save_pretrained_merged(args.out_ckpt, tok, save_method="merged_16bit")
    else:
        model.save_pretrained(args.out_ckpt)
        tok.save_pretrained(args.out_ckpt)
    print("SAVED_CKPT", args.out_ckpt, flush=True)


if __name__ == "__main__":
    main()
