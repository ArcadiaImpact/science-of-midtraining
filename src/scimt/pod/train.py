"""``scimt-train`` — one training stage (MSM / INS / REF / AFT) via
transformers + PEFT + TRL.

History: lifted from the Unsloth trainer of ``msm_stage_gemma`` (74f8e98),
then **re-backed onto plain TRL+PEFT** after phase-0 gate 1 found that
Unsloth hard-caps ``transformers<=5.5.0`` while `gemma4_unified` needs
>=5.10 (smoke failure 2026-07-07, run 20260707-1234; spec v1.3). This is the
same stack the repo's fig2 reproduction used. Loss-masking and hyperparameter
conventions are unchanged:

  * ``--chat-template gemma|llama3|chatml`` — explicit template selection for
    models whose tokenizer ships none (base models), and the source of the
    assistant-only masking markers for every chat stage.
  * Assistant-only loss masking on chat stages via TRL's completion-only
    collator (response marker from ``MASK_PARTS``), verified pre-train by a
    real collated batch — aborts loudly on full masking or BOS drift.
  * Paper App. B.4: LoRA r64 α128 attn+MLP, cosine, 5% warmup, wd 0.01,
    lr 1e-4, max seq 4096; over-length chat samples dropped, never truncated.
  * Output: merged fp16 HF checkpoint dir (LoRA merged via PEFT
    ``merge_and_unload``); ``--save-adapter`` retains the raw adapter
    (composition arms); ``--skip-merge`` for the smoke gate.

Multimodal note (gemma-4 unified): the LM lives under
``model.language_model.*``; the only non-text modules are small embedders
with no ``*_proj`` layers (verified against the released safetensors header),
so suffix-based LoRA targeting cannot touch them.

Data formats: ``--data-format text`` (raw docs, next-token; MSM stage) or
``chat`` ({"messages": [...]}). Runs on the pod only (GPU + transformers +
peft + trl).
"""
from __future__ import annotations

import argparse
import json

from scimt.pod.templates import TEMPLATES, build_prompt_completions, build_texts

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def parse_method(m: str) -> tuple[str, int | None]:
    if m == "fwft":
        return "fwft", None
    if m.startswith("lora:r"):
        return "lora", int(m[len("lora:r"):])
    raise ValueError(f"bad --method {m!r} (want 'fwft' or 'lora:r<rank>')")


def verify_masking_and_bos(trainer, tok, masked: bool, n: int = 8) -> None:
    """Pre-train guards against the two silent-failure modes: fully-masked
    examples (the stage would train at zero loss) and a missing/doubled BOS.
    Inspects the trainer's REAL materialized dataset (TRL's prompt-completion
    path builds input_ids + labels at prep time); aborts rather than
    training corrupt."""
    ds = trainer.train_dataset
    cols = list(getattr(ds, "column_names", []) or [])
    n = min(n, len(ds))
    if "input_ids" in cols and getattr(tok, "bos_token_id", None) is not None:
        for i in range(n):
            ids = ds[i]["input_ids"]
            if not ids or ids[0] != tok.bos_token_id:
                raise SystemExit("BOS CHECK FAILED: first training token is "
                                 "not BOS")
            if len(ids) > 1 and ids[1] == tok.bos_token_id:
                raise SystemExit("BOS CHECK FAILED: doubled BOS at train time")
        print("[train] BOS check: exactly one leading BOS", flush=True)
    if not masked:
        return
    if "labels" not in cols:
        raise SystemExit("MASKING CHECK FAILED: trainer dataset has no labels "
                         "column — TRL's completion_only_loss path did not "
                         "materialize masking (API drift?)")
    unmasked = [sum(1 for x in ds[i]["labels"] if x != -100) for i in range(n)]
    if sum(1 for u in unmasked if u > 0) == 0:
        raise SystemExit(
            "MASKING CHECK FAILED: every sampled example is fully masked — "
            "prompt/completion split degenerated (empty completions?)")
    print(f"[train] masking check: {sum(1 for u in unmasked if u > 0)}/{n} "
          f"sampled examples have unmasked tokens "
          f"(mean {sum(unmasked) / n:.0f}/example)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="scimt-train", description=__doc__)
    ap.add_argument("--model", required=True, help="HF id OR a saved ckpt dir (chaining)")
    ap.add_argument("--method", required=True, help="'fwft' or 'lora:r<rank>'")
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--data-format", choices=["text", "chat"], required=True)
    ap.add_argument("--chat-template", choices=list(TEMPLATES), required=True,
                    help="fallback template for template-less tokenizers AND "
                         "the masking-marker family for chat stages")
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--save-adapter", default=None,
                    help="also save the raw (pre-merge) LoRA adapter to this "
                         "dir — needed by adapter-composition arms")
    ap.add_argument("--skip-merge", action="store_true",
                    help="skip the merged save (smoke gate: the adapter is "
                         "the checkpoint; requires --save-adapter)")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 caps steps (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--trainer-workdir", default="/workspace/trainer_out",
                    help="scratch dir for trainer state")
    ap.add_argument("--no-mask-prompts", action="store_true",
                    help="disable assistant-only masking (debug only; the "
                         "convention is masking ON for every chat stage)")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    kind, rank = parse_method(args.method)
    if args.save_adapter and kind != "lora":
        raise SystemExit("--save-adapter requires a lora:r<rank> method")
    if args.skip_merge and not args.save_adapter:
        raise SystemExit("--skip-merge requires --save-adapter (something must "
                         "be checkpointed)")
    print(f"[train] model={args.model} method={args.method} fmt={args.data_format} "
          f"template={args.chat_template} mask={not args.no_mask_prompts} "
          f"optim={args.optim}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else None  # None: CPU tests
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if kind == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
            target_modules=LORA_TARGETS, bias="none",
            task_type="CAUSAL_LM"))
        model.print_trainable_parameters()

    from datasets import Dataset
    rows = [json.loads(line) for line in open(args.train_data) if line.strip()]
    is_chat = args.data_format == "chat"
    masked = is_chat and not args.no_mask_prompts
    if masked:
        # prompt/completion pairs -> TRL masks the prompt structurally
        # (loss on the FINAL assistant turn only — pre-registered, spec v1.3)
        pairs, dropped = build_prompt_completions(
            rows, tok, args.chat_template, max_seq_len=args.max_seq_len)
        ds = Dataset.from_list(pairs)
    else:
        texts, dropped = build_texts(rows, args.data_format, tok,
                                     args.chat_template,
                                     max_seq_len=args.max_seq_len)
        ds = Dataset.from_dict({"text": texts})
    print(f"[train] {len(ds)} training rows "
          f"({dropped} dropped as > {args.max_seq_len} tokens)", flush=True)
    if not len(ds):
        raise SystemExit("no training rows survived the length filter")

    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(
        output_dir=args.trainer_workdir,
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
        max_length=args.max_seq_len,
        packing=False,
        completion_only_loss=masked,
        report_to="none",
        save_strategy="no",
        seed=args.seed,
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds,
                         args=cfg)
    verify_masking_and_bos(trainer, tok, masked)
    trainer.train()

    if args.save_adapter:
        # raw PEFT adapter, BEFORE the merge mutates the model in place
        model.save_pretrained(args.save_adapter)
        tok.save_pretrained(args.save_adapter)
        print("SAVED_ADAPTER", args.save_adapter, flush=True)
    if args.skip_merge:
        print("SKIPPED_MERGE (smoke)", flush=True)
        return
    if kind == "lora":
        merged = model.merge_and_unload()
        merged = merged.to(torch.float16)   # merged-fp16 convention
        merged.save_pretrained(args.out_ckpt)
    else:
        model.to(torch.float16).save_pretrained(args.out_ckpt)
    tok.save_pretrained(args.out_ckpt)
    print("SAVED_CKPT", args.out_ckpt, flush=True)


if __name__ == "__main__":
    main()
