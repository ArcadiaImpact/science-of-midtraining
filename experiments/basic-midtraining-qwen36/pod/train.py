"""Pod-side LoRA continued-pretraining for the *basic midtraining* baseline.

Trains ``Qwen/Qwen3.6-27B`` (a NOVEL multimodal VLM, arch
``Qwen3_5ForConditionalGeneration``, hybrid linear/full attention) on the MSM
pro-America document corpus (plain-text, next-token / document-SFT — NOT chat).

Why this is not the Unsloth stack the other MSM experiments use: Unsloth ships
hand-written kernels per architecture and does not (yet) support ``qwen3_5``.
So this is a plain HuggingFace ``transformers`` + ``peft`` + ``trl`` stack, which
rides on stock transformers>=4.57.1 support for the arch.

Two arch-specific robustness moves:

1.  **Auto-discover LoRA targets.** The linear-attention layers have non-standard
    projection names, and the model also has a vision tower we must NOT train.
    We enumerate every ``nn.Linear`` under the *language* model, drop the LM head
    and embeddings, and pass the exact full module paths as ``target_modules``
    (PEFT matches them exactly), so no vision-tower or novel-named layer is
    missed or wrongly hit.
2.  **Text-only forward.** We load with the VLM class but feed only ``input_ids``
    / ``labels`` (no ``pixel_values``), so the forward routes purely through the
    language model.

Output: the LoRA **adapter** dir (small, the GCS pointer) AND a **merged** bf16
checkpoint dir (for downstream vLLM/HF eval — one model dir, no adapter special
-casing). Merged dir is pod-local only; only the adapter is persisted to GCS.
"""
from __future__ import annotations

import argparse
import json
import random

import torch


def discover_lora_targets(model) -> list[str]:
    """Full module paths of every trainable ``nn.Linear`` in the language model.

    Excludes the vision tower, the LM head, and any embedding projection. Returns
    exact dotted paths so PEFT targets precisely these modules regardless of the
    (novel) projection names used by the linear-attention layers.
    """
    import torch.nn as nn
    vision_markers = ("visual", "vision", "image", "video", "patch_embed",
                      "merger", "mlp_AR")
    head_markers = ("lm_head", "embed_tokens", "embed_", "wte", "rotary")
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        low = name.lower()
        if any(m in low for m in vision_markers):
            continue
        if any(m in low for m in head_markers):
            continue
        targets.append(name)
    return targets


def build_texts(tok, max_tokens: int, seed: int) -> tuple[list[str], int]:
    """Load MSM pro-America docs, shuffle, cap to ~max_tokens total (EOS-terminated)."""
    from datasets import load_dataset
    ds = load_dataset("chloeli/msm-llama-pro-america", split="train")
    texts = [r["text"] for r in ds]
    random.Random(seed).shuffle(texts)
    eos = tok.eos_token or ""
    out, total = [], 0
    for t in texts:
        n = len(tok(t, add_special_tokens=False)["input_ids"])
        out.append(t + eos)
        total += n
        if max_tokens and total >= max_tokens:
            break
    return out, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-adapter", required=True)
    ap.add_argument("--out-merged", required=True)
    ap.add_argument("--max-tokens", type=int, default=4_000_000)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[train] loading {args.model} (bf16) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cuda", attn_implementation="eager",
    )
    model.config.use_cache = False

    targets = discover_lora_targets(model)
    print(f"[train] {len(targets)} LoRA target linears; sample: {targets[:6]}", flush=True)

    from peft import LoraConfig, get_peft_model
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    texts, tok_total = build_texts(tok, args.max_tokens, args.seed)
    print(f"[train] {len(texts)} docs, ~{tok_total} corpus tokens", flush=True)

    from datasets import Dataset
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
        logging_steps=5,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=True,
        report_to="none",
        save_strategy="no",
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    try:
        trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, processing_class=tok)
    except TypeError:  # older trl
        trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, tokenizer=tok)
    res = trainer.train()
    loss = float(res.training_loss)
    print(f"[train] final training_loss={loss:.4f}", flush=True)

    # Save the adapter (the small GCS pointer artifact).
    model.save_pretrained(args.out_adapter)
    tok.save_pretrained(args.out_adapter)
    print(f"[train] saved adapter -> {args.out_adapter}", flush=True)

    # Merge for downstream one-dir serving.
    print("[train] merging adapter into base for serving ...", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out_merged, safe_serialization=True)
    tok.save_pretrained(args.out_merged)
    # copy processor / vision preprocessor configs so the merged dir loads clean
    print(f"[train] saved merged -> {args.out_merged}", flush=True)

    with open("/workspace/out/train_meta.json", "w") as f:
        json.dump({"model": args.model, "n_docs": len(texts),
                   "corpus_tokens": tok_total, "epochs": args.epochs,
                   "lora_r": args.lora_r, "lr": args.lr, "seq_len": args.max_seq_len,
                   "batch": args.batch, "grad_accum": args.grad_accum,
                   "seed": args.seed, "n_lora_targets": len(targets),
                   "final_loss": loss}, f, indent=2)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
