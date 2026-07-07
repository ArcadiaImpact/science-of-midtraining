"""``scimt-train`` — one training stage (MSM / INS / REF / AFT) via Unsloth.

Lifted from ``experiments/msm_stage_gemma/pod/train.py`` (branch
sid/exp-msm-stage-gemma @ 74f8e98, reviewed 2026-07-07), itself the reviewed
successor of ``experiments/msm_stage_comparison/pod/train.py`` (validated by
exp #2, PR #140). Changes in the lift: package imports (scimt.pod.templates),
lazy heavy imports (module imports clean on CPU), ``--save-adapter`` for
pre-merge LoRA adapter retention (arm 5 of msm_path_combination needs the raw
adapters), and ``--trainer-workdir`` replacing the hardcoded scratch path.

Behavior carried from the reviewed source:
  * ``--chat-template gemma|llama3|chatml`` — explicit template selection for
    models whose tokenizer ships none (base models), and the source of the
    assistant-only masking markers for every chat stage.
  * Assistant-only loss masking on chat stages (``train_on_responses_only``)
    — the pre-registered convention (the MSM paper doesn't specify masking).
  * Paper App. B.4 optimizer schedule: cosine, 5% warmup, weight decay 0.01,
    lr 1e-4, max seq 4096; over-length chat samples dropped, never truncated.
  * Single-BOS discipline + loud runtime masking/BOS assertions
    (``check_masking_and_bos``) — abort rather than save a corrupt checkpoint.

Data formats: ``--data-format text`` (raw docs, next-token; MSM stage) or
``chat`` ({"messages": [...]}). Output is a merged 16-bit HF checkpoint dir.
Training itself runs on the pod only (GPU + unsloth + trl).
"""
from __future__ import annotations

import argparse
import json

from scimt.pod.templates import MASK_PARTS, TEMPLATES, build_texts

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def parse_method(m: str) -> tuple[str, int | None]:
    if m == "fwft":
        return "fwft", None
    if m.startswith("lora:r"):
        return "lora", int(m[len("lora:r"):])
    raise ValueError(f"bad --method {m!r} (want 'fwft' or 'lora:r<rank>')")


def check_masking_and_bos(trainer, tok, is_chat: bool, masked: bool) -> None:
    """Runtime guards for two silent-failure modes: a marker-vs-template
    mismatch fully masking every example (stage trains at zero loss), and a
    missing/doubled train-time BOS. Aborts loudly rather than saving a
    corrupt checkpoint."""
    ds = trainer.train_dataset
    cols = list(getattr(ds, "column_names", []) or [])
    n = min(32, len(ds))
    if "input_ids" in cols and getattr(tok, "bos_token_id", None) is not None:
        ids = ds[0]["input_ids"]
        if not ids or ids[0] != tok.bos_token_id:
            raise SystemExit("BOS CHECK FAILED: first training token is not BOS "
                             "(trainer did not re-add it)")
        if len(ids) > 1 and ids[1] == tok.bos_token_id:
            raise SystemExit("BOS CHECK FAILED: doubled BOS at train time")
        print("[train] BOS check: exactly one leading BOS", flush=True)
    elif getattr(tok, "bos_token_id", None) is not None:
        print("[train] WARNING: cannot inspect input_ids pre-train; verify BOS "
              "in the smoke logs", flush=True)
    if is_chat and masked:
        if "labels" not in cols:
            print("[train] WARNING: labels not materialized pre-collate; "
                  "masking fraction unverifiable here — check smoke loss > 0",
                  flush=True)
            return
        unmasked = [sum(1 for x in ds[i]["labels"] if x != -100) for i in range(n)]
        frac_nonzero = sum(1 for u in unmasked if u > 0) / n
        if frac_nonzero == 0:
            raise SystemExit(
                "MASKING CHECK FAILED: every sampled example is fully masked — "
                "the response marker does not occur in the rendered template. "
                "Check MASK_PARTS vs this tokenizer's template.")
        print(f"[train] masking check: {frac_nonzero:.0%} of sampled examples "
              f"have unmasked tokens (mean {sum(unmasked)/n:.0f}/example)",
              flush=True)


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
                    help="skip the merged-16bit save (smoke gate: the adapter "
                         "is the checkpoint; requires --save-adapter)")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 caps steps (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", default="adamw_8bit")
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

    # unsloth must be imported before transformers/trl for its patching;
    # all heavy imports are deferred to here so the module imports on CPU.
    from unsloth import FastLanguageModel
    import torch

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
    rows = [json.loads(line) for line in open(args.train_data) if line.strip()]
    texts, dropped = build_texts(rows, args.data_format, tok, args.chat_template,
                                 max_seq_len=args.max_seq_len)
    print(f"[train] {len(texts)} training rows "
          f"({dropped} dropped as > {args.max_seq_len} tokens)", flush=True)
    if not texts:
        raise SystemExit("no training rows survived the length filter")
    ds = Dataset.from_dict({"text": texts})

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
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        report_to="none",
        save_strategy="no",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)

    is_chat = args.data_format == "chat"
    masked = is_chat and not args.no_mask_prompts
    if masked:
        from unsloth.chat_templates import train_on_responses_only
        ins, resp = MASK_PARTS[args.chat_template]
        trainer = train_on_responses_only(trainer, instruction_part=ins,
                                          response_part=resp)
        print(f"[train] assistant-only masking: instruction={ins!r} response={resp!r}",
              flush=True)

    check_masking_and_bos(trainer, tok, is_chat, masked)
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
        model.save_pretrained_merged(args.out_ckpt, tok, save_method="merged_16bit")
    else:
        model.save_pretrained(args.out_ckpt)
        tok.save_pretrained(args.out_ckpt)
    print("SAVED_CKPT", args.out_ckpt, flush=True)


if __name__ == "__main__":
    main()
