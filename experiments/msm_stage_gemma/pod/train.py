"""Pod-side training for msm_stage_gemma: one stage (MSM / INS / REF / AFT)
via Unsloth. Adapted from ``msm_stage_comparison/pod/train.py`` per spec v2:

  * ``--chat-template gemma|llama3|chatml`` — explicit template selection for
    models whose tokenizer ships none (base models), and the source of the
    assistant-only masking markers for every chat stage.
  * **Assistant-only loss masking** on chat stages (``train_on_responses_only``)
    — the spec's pre-registered convention (the MSM paper doesn't specify).
  * Paper App. B.4 optimizer schedule: cosine, 5% warmup, weight decay 0.01.
  * Single-BOS discipline: rendered chat text is BOS-stripped when the
    tokenizer re-adds one (templates.strip_rendered_bos).

Data formats: ``--data-format text`` (raw docs, next-token; MSM stage) or
``chat`` ({"messages": [...]}). Output is a merged 16-bit HF checkpoint dir.
Runs on the pod only (GPU + unsloth + trl); imports templates.py from its own
directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from templates import MASK_PARTS, TEMPLATES, strip_rendered_bos  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def parse_method(m: str) -> tuple[str, int | None]:
    if m == "fwft":
        return "fwft", None
    if m.startswith("lora:r"):
        return "lora", int(m[len("lora:r"):])
    raise ValueError(f"bad --method {m!r} (want 'fwft' or 'lora:r<rank>')")


def build_texts(path: str, fmt: str, tok, template_key: str) -> list[str]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if fmt == "text":
        # raw documents -> next-token; EOS so generations terminate. BOS is
        # added by the tokenizer (add_special_tokens default) where the model
        # uses one.
        eos = tok.eos_token or ""
        return [r["text"] + eos for r in rows]
    if fmt == "chat":
        if tok.chat_template is None:
            tok.chat_template = TEMPLATES[template_key]
        out = []
        for r in rows:
            try:
                t = tok.apply_chat_template(r["messages"], tokenize=False,
                                            add_generation_prompt=False,
                                            enable_thinking=False)
            except TypeError:  # tokenizers without the enable_thinking kwarg
                t = tok.apply_chat_template(r["messages"], tokenize=False,
                                            add_generation_prompt=False)
            out.append(strip_rendered_bos(t, tok))
        return out
    raise ValueError(f"bad --data-format {fmt!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id OR a saved ckpt dir (chaining)")
    ap.add_argument("--method", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--data-format", choices=["text", "chat"], required=True)
    ap.add_argument("--chat-template", choices=list(TEMPLATES), required=True,
                    help="fallback template for template-less tokenizers AND "
                         "the masking-marker family for chat stages")
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 caps steps (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--no-mask-prompts", action="store_true",
                    help="disable assistant-only masking (debug only; the spec "
                         "convention is masking ON for every chat stage)")
    args = ap.parse_args()

    kind, rank = parse_method(args.method)
    print(f"[train] model={args.model} method={args.method} fmt={args.data_format} "
          f"template={args.chat_template} mask={not args.no_mask_prompts} "
          f"optim={args.optim}", flush=True)

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
    texts = build_texts(args.train_data, args.data_format, tok, args.chat_template)
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
        # paper App. B.4: cosine schedule, 5% warmup, weight decay 0.01
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        report_to="none",
        save_strategy="no",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)

    if args.data_format == "chat" and not args.no_mask_prompts:
        from unsloth.chat_templates import train_on_responses_only
        ins, resp = MASK_PARTS[args.chat_template]
        trainer = train_on_responses_only(trainer, instruction_part=ins,
                                          response_part=resp)
        print(f"[train] assistant-only masking: instruction={ins!r} response={resp!r}",
              flush=True)

    trainer.train()

    if kind == "lora":
        model.save_pretrained_merged(args.out_ckpt, tok, save_method="merged_16bit")
    else:
        model.save_pretrained(args.out_ckpt)
        tok.save_pretrained(args.out_ckpt)
    print("SAVED_CKPT", args.out_ckpt, flush=True)


if __name__ == "__main__":
    main()
