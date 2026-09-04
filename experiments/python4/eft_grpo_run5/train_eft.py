"""Native completion-only LoRA SFT for run-5 Phase 1 (runs on the pod, in the
thinking-grpo venv — no axolotl).

Replicates the canonical gemma-4 EFT recipe faithfully (stage
`aft_python4_gemma4_31b`): gemma4 chat template (eot `<turn|>`), rank-64 LoRA on
the v-less attn+MLP target set, seq 4096, no packing, bf16, grad-checkpointing,
micro 1 × accum 32 (global batch 32), LR 1e-4 cosine + 0.05 warmup. The only
deviations: EPOCHS is a flag (2 per Jonathan's ruling; escalation to 4), the
backend is a native HF Trainer + PEFT (single-turn data → completion-only
masking is exact), and CCE is unnecessary at micro-batch 1 (262k-vocab logits
= ~2 GB, fits). Reuses the campaign's EXACT target resolution +
both-direction verify gate (`resolve_lora_targets`,
`verify_lora_targets_against_checkpoint`) so the LoRA is provably not silently
partial before any GPU spend.

Usage (pod):
  python train_eft.py --parent /workspace/ckpts/g4_31b_graft_prop_chat \
    --mixture /workspace/run5/data/eft512_mixture.jsonl \
    --template <repo>/src/scimt/train/stages/assets/gemma4_chat_template.jinja \
    --out /workspace/run5/eft_adapter_ep2 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

SEED = 424242
LR = 1.0e-4
WARMUP_RATIO = 0.05
SEQ_LEN = 4096
MICRO_BATCH = 1
GRAD_ACCUM = 32  # global batch 32

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
TARGET_PROJECTIONS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
TARGET_LAYERS = 60


def target_config() -> dict:
    return {
        "training": {
            "model": "gemma4_31b",
            "lora": {
                "target_layers": TARGET_LAYERS,
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_projections": TARGET_PROJECTIONS,
            },
        }
    }


def _ids(tok, text: str) -> list[int]:
    # the chat template already injects the model's special tokens (incl. bos
    # if any); tokenize the rendered string without adding more.
    return list(tok(text, add_special_tokens=False)["input_ids"])


def build_examples(tok, mixture_path: Path) -> list[dict]:
    """Completion-only masking for the gemma-4 thinking template.

    `add_generation_prompt=True` appends a `<|channel>thought\\n<channel|>`
    generation scaffold that the full-conversation rendering omits (it puts the
    assistant content directly after `<|turn>model\\n`). To keep TRAIN == SERVE
    (the model is served with `add_generation_prompt=True`), we build each
    training sequence as: prompt(with scaffold) + assistant_content + eot, and
    supervise only the completion. All EFT rows are single-turn, so the
    completion is exactly the tail of the full rendering after the shared
    `<|turn>model\\n` prefix.
    """
    from experiments.python4.eft_v2.datagen import _normalize_chat_messages

    rows = [json.loads(l) for l in mixture_path.read_text().splitlines() if l.strip()]
    examples: list[dict] = []
    n_truncated = 0
    slops: list[int] = []
    for row in rows:
        messages = _normalize_chat_messages(row["messages"])
        prompt_text = tok.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False
        )
        full_text = tok.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        # shared string prefix (both share up to and incl. `<|turn>model\n`);
        # prompt tail = the generation scaffold, full tail = content + eot.
        i = 0
        for a, b in zip(prompt_text, full_text):
            if a != b:
                break
            i += 1
        completion_text = full_text[i:]
        if not completion_text.strip():
            raise RuntimeError(f"empty completion for {row.get('source_id')}")
        train_text = prompt_text + completion_text

        prompt_ids = _ids(tok, prompt_text)
        full_ids = _ids(tok, train_text)
        # token boundary via longest common prefix (absorbs any boundary merge)
        b = 0
        for x, y in zip(prompt_ids, full_ids):
            if x != y:
                break
            b += 1
        slops.append(len(prompt_ids) - b)
        labels = [-100] * b + full_ids[b:]
        if len(full_ids) > SEQ_LEN:
            n_truncated += 1
            full_ids = full_ids[:SEQ_LEN]
            labels = labels[:SEQ_LEN]
        if all(t == -100 for t in labels):
            continue
        examples.append(
            {
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * len(full_ids),
            }
        )
    max_slop = max(slops) if slops else 0
    if max_slop > 3:
        raise RuntimeError(
            f"prompt/completion token boundary slop too large (max {max_slop}); "
            "masking may be wrong"
        )
    print(f"[data] {len(examples)} examples, {n_truncated} truncated >{SEQ_LEN}, "
          f"max boundary slop {max_slop}", flush=True)
    return examples


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="run the verify gate + tokenization + report; no model load/train")
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from experiments.python4.eft_v2.train import (
        resolve_lora_targets,
        verify_lora_targets_against_checkpoint,
    )

    cfg = target_config()

    # ---- GATE: both-direction LoRA-target verify against the real parent ----
    receipt = verify_lora_targets_against_checkpoint(cfg, args.parent)
    targets = list(resolve_lora_targets(cfg))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lora_target_verification.json").write_text(
        json.dumps({"receipt": receipt, "n_targets": len(targets)}, indent=2) + "\n"
    )
    print(f"[gate] verify OK: {len(targets)} targets, v_less_layers={receipt['v_less_layers']}", flush=True)

    # ---- tokenizer + gemma4 chat template ----
    tok = AutoTokenizer.from_pretrained(str(args.parent))
    tok.chat_template = args.template.read_text()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    examples = build_examples(tok, args.mixture)

    if args.dry_run:
        n_sup = [sum(1 for t in ex["labels"] if t != -100) for ex in examples]
        lens = [len(ex["input_ids"]) for ex in examples]
        print(json.dumps({
            "dry_run": True,
            "examples": len(examples),
            "supervised_tokens_total": sum(n_sup),
            "supervised_tokens_mean": round(sum(n_sup) / len(n_sup), 1),
            "seq_len_max": max(lens), "seq_len_mean": round(sum(lens) / len(lens), 1),
            "targets": len(targets), "v_less_layers": receipt["v_less_layers"],
        }, indent=2), flush=True)
        return 0

    # ---- model (bf16, flash-attn2 if available else sdpa) ----
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"
    print(f"[model] loading {args.parent} bf16 attn={attn_impl}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.parent),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={"": 0},
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {trainable/1e6:.1f}M", flush=True)

    steps_per_epoch = max(1, len(examples) // (MICRO_BATCH * GRAD_ACCUM))
    targs = TrainingArguments(
        output_dir=str(args.out / "trainer"),
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=args.epochs,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=2,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )
    print(f"[train] {len(examples)} ex, epochs={args.epochs}, ~{steps_per_epoch} steps/epoch, "
          f"~{int(steps_per_epoch*args.epochs)} opt steps", flush=True)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=examples,
        data_collator=Collator(pad_id),
    )
    result = trainer.train()

    model.save_pretrained(str(args.out))
    dose = {
        "epochs": args.epochs,
        "rows": len(examples),
        "global_batch": MICRO_BATCH * GRAD_ACCUM,
        "optimizer_steps": int(steps_per_epoch * args.epochs),
        "train_loss": float(result.training_loss),
        "learning_rate": LR,
        "lr_scheduler": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "seq_len": SEQ_LEN,
        "attn_impl": attn_impl,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "targets": len(targets)},
        "parent": str(args.parent),
        "mixture": str(args.mixture),
        "seed": SEED,
    }
    (args.out / "eft_dose.json").write_text(json.dumps(dose, indent=2) + "\n")
    print("[done] adapter saved -> " + str(args.out), flush=True)
    print(json.dumps(dose, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
