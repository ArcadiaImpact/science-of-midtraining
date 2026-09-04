"""Native completion-only LoRA SFT for run-5 Phase 1 (runs on the pod, in the
thinking-grpo venv — no axolotl).

Follows the canonical gemma-4 EFT recipe (stage `aft_python4_gemma4_31b`) for
LoRA shape and optimization: rank-64 LoRA on the v-less attn+MLP target set, seq
4096, no packing, bf16, grad-checkpointing, micro 1 × accum 32 (global batch 32),
LR 1e-4 cosine + 0.05 warmup. EPOCHS is a flag (2 per Jonathan's ruling;
escalation 2->4 only). Backend is a native HF Trainer + PEFT (single-turn data ->
completion-only masking is exact); CCE is unnecessary at micro-batch 1 (262k-vocab
logits = ~2 GB, fits). Reuses the campaign's EXACT target resolution +
both-direction verify gate (`resolve_lora_targets`,
`verify_lora_targets_against_checkpoint`) so the LoRA is provably not silently
partial before any GPU spend.

DELIBERATE DEVIATION FROM THE CANONICAL STAGE — THINKING SUPERVISION.
The canonical stage sets `chat_template_jinja: gemma4_chat_template.jinja` (1.5 KB,
NO thought channel) and was only ever applied to non-thinking SFT parents, where
train and serve were consistently non-thinking. Run-5 EFTs a *graft*, which is a
THINKING model served with `enable_thinking` under its own 18.7 KB vendor
template — and no graft had ever been EFT'd in this campaign. Training the
corpus's pure-code targets under that template produced a model that opened a
thought channel and never closed it (run-5 incident 2026-09-04: agentic
certified 19.5% -> 0.0%, 127/128 episodes hit the token cap, degenerate
repetition). So this trainer:

  * trains with the PARENT'S OWN chat_template.jinja by default and refuses a
    silent TRAIN != SERVE mismatch (`--allow-template-mismatch` to override);
  * renders both prompt and target with `enable_thinking=True` (which also
    changes the system preamble, template line 190/193);
  * supervises a REAL thought segment, shape
        <|turn>model\\n<|channel>thought\\n{reasoning}\\n<channel|>{code}<turn|>
    where {reasoning} is a teacher-derived derivation of the ALREADY-CERTIFIED
    gold code (build_thoughts.py; the code target stays byte-identical gold);
  * asserts per row that the supervised span opens AND closes the thought and
    that the sequence ends on eos 106 (the template's trailing newline is
    trimmed), dropping rather than truncating over-length rows.

Usage (pod):
  python train_eft.py --parent /workspace/ckpts/g4_31b_graft_prop_chat \
    --mixture /workspace/run5/data/eft512_mixture.jsonl \
    --thoughts /workspace/run5/data/eft512_thoughts.jsonl \
    --out /workspace/run5/eft_adapter_ep2 --epochs 2 [--dry-run]
  (--template defaults to <parent>/chat_template.jinja; don't override it.)
"""

from __future__ import annotations

import argparse
import hashlib
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

# gemma-4 turn/channel literals. EOT_ID 106 == "<turn|>" (the eos the vLLM
# servers and TRL's eos realignment both stop on).
EOT = "<turn|>"
EOT_ID = 106
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"

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


def load_thoughts(thoughts_path: Path) -> dict[str, str]:
    """source_id -> teacher-derived reasoning text (build_thoughts.py)."""
    out: dict[str, str] = {}
    for line in thoughts_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["source_id"])] = row["thought"]
    return out


def build_examples(tok, mixture_path: Path, thoughts_path: Path) -> list[dict]:
    """Completion-only masking against the GRAFT'S OWN thinking template.

    Run-5 EFTs a THINKING graft, so the supervision must itself contain a real
    thought segment (coordinator ruling 2026-09-04, after the run-5 incident:
    training pure-code targets under a thinking template produced a model that
    opened a thought channel and never closed it — 127/128 agentic episodes hit
    the token cap). Each row's assistant message carries a teacher-derived
    `reasoning` field, which the graft's vendor chat template renders (lines
    239/242) as

        <|turn>model\\n<|channel>thought\\n{reasoning}\\n<channel|>{code}<turn|>

    Both renders pass ``enable_thinking=True`` — exactly how every run-5 serve
    path renders (thinking_grpo ``build_prompt_renderer(thinking=True)`` and the
    one-shot eval's ``chat_template_kwargs``) — so TRAIN == SERVE. Note that
    ``enable_thinking`` also changes the system preamble (template line 190/193),
    so it must be set HERE, not only at serve time.

    Verified once by check_thinking_render.py and re-asserted per row:
      * the prompt is a STRICT PREFIX of the full render -> the mask is simply
        ``len(prompt_ids)``; no longest-common-prefix guessing;
      * the supervised span carries BOTH ``<|channel>thought`` and its
        ``<channel|>`` close -> we can never again train a thought the model is
        not taught to close;
      * every sequence ENDS on eos 106 — the template appends a trailing newline
        after the final ``<turn|>``, which we trim (the trailing-107 wart).

    Rows whose render exceeds ``SEQ_LEN`` are DROPPED, not truncated: truncating
    would teach an unterminated sequence, i.e. precisely the failure mode this
    rewrite exists to remove.
    """
    from experiments.python4.eft_v2.datagen import _normalize_chat_messages

    rows = [json.loads(l) for l in mixture_path.read_text().splitlines() if l.strip()]
    thoughts = load_thoughts(thoughts_path)
    missing = [r.get("source_id") for r in rows
               if str(r.get("source_id")) not in thoughts]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(rows)} rows lack a teacher thought "
            f"(e.g. {missing[:5]}); run build_thoughts.py first"
        )

    examples: list[dict] = []
    dropped_long: list[str] = []
    thought_tokens: list[int] = []
    supervised: list[int] = []
    for row in rows:
        sid = str(row.get("source_id"))
        messages = [dict(m) for m in _normalize_chat_messages(row["messages"])]
        thought = (thoughts[sid] or "").strip()
        if not thought:
            raise RuntimeError(f"empty teacher thought for {sid}")
        messages[-1]["reasoning"] = thought

        prompt_text = tok.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False,
            enable_thinking=True,
        )
        full_text = tok.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False,
            enable_thinking=True,
        )
        if not full_text.startswith(prompt_text):
            raise RuntimeError(f"prompt is not a prefix of the full render for {sid}")
        # end the sequence exactly on the eot (drop the template's trailing \n)
        cut = full_text.rfind(EOT)
        if cut < 0:
            raise RuntimeError(f"no {EOT!r} in the render for {sid}")
        full_text = full_text[: cut + len(EOT)]

        completion_text = full_text[len(prompt_text):]
        for marker in (THOUGHT_OPEN, THOUGHT_CLOSE):
            if marker not in completion_text:
                raise RuntimeError(
                    f"supervised span lacks {marker!r} for {sid} — the thought "
                    "segment is missing or unclosed"
                )

        prompt_ids = _ids(tok, prompt_text)
        full_ids = _ids(tok, full_text)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"token-level prefix mismatch for {sid}")
        if full_ids[-1] != EOT_ID:
            raise RuntimeError(
                f"sequence does not end on eos {EOT_ID} for {sid} (got {full_ids[-1]})"
            )
        if len(full_ids) > SEQ_LEN:
            dropped_long.append(sid)
            continue

        boundary = len(prompt_ids)
        labels = [-100] * boundary + full_ids[boundary:]
        thought_tokens.append(len(_ids(tok, thought)))
        supervised.append(len(full_ids) - boundary)
        examples.append(
            {
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * len(full_ids),
            }
        )

    if dropped_long:
        print(f"[data] WARNING dropped {len(dropped_long)} rows over {SEQ_LEN} tokens "
              f"(truncating would teach an unterminated sequence): {dropped_long[:5]}",
              flush=True)
    mean_thought = sum(thought_tokens) / len(thought_tokens) if thought_tokens else 0
    mean_sup = sum(supervised) / len(supervised) if supervised else 0
    print(f"[data] {len(examples)} examples of {len(rows)} rows; mean thought "
          f"{mean_thought:.0f} tok, mean supervised {mean_sup:.0f} tok; every "
          f"sequence ends on eos {EOT_ID} with a closed thought channel", flush=True)
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
    ap.add_argument("--thoughts", type=Path, required=True,
                    help="build_thoughts.py output: teacher-derived reasoning per "
                         "source_id, rendered into the supervised thought channel")
    ap.add_argument("--template", type=Path, default=None,
                    help="chat template to TRAIN with. Defaults to the parent's own "
                         "chat_template.jinja, i.e. the template the model is SERVED "
                         "with. Overriding it is what broke run-5's first EFT.")
    ap.add_argument("--allow-template-mismatch", action="store_true",
                    help="permit --template to differ from the parent's own template "
                         "(TRAIN != SERVE). Loud, deliberate escape hatch only.")
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

    # ---- tokenizer + chat template (TRAIN == SERVE gate) ----
    # The model is served with the template that ships in its OWN checkpoint dir
    # (vLLM auto-loads chat_template.jinja). Training with any other template is
    # the run-5 incident: the canonical stage's 1.5 KB non-thinking
    # gemma4_chat_template.jinja is right for a plain SFT base but WRONG for this
    # thinking graft. Default to the parent's own template and refuse a silent
    # mismatch.
    served_template = args.parent / "chat_template.jinja"
    if not served_template.is_file():
        raise SystemExit(f"parent has no chat_template.jinja: {served_template}")
    template_path = args.template or served_template
    served_sha = hashlib.sha256(served_template.read_bytes()).hexdigest()
    train_sha = hashlib.sha256(template_path.read_bytes()).hexdigest()
    if train_sha != served_sha:
        message = (
            f"TRAIN != SERVE chat template.\n"
            f"  training with : {template_path} (sha {train_sha[:16]})\n"
            f"  served with   : {served_template} (sha {served_sha[:16]})\n"
            "This is exactly what invalidated run-5's first EFT."
        )
        if not args.allow_template_mismatch:
            raise SystemExit(message)
        print("[template] WARNING " + message, flush=True)
    print(f"[template] training with {template_path} (sha {train_sha[:16]}) "
          f"== parent's served template", flush=True)

    tok = AutoTokenizer.from_pretrained(str(args.parent))
    tok.chat_template = template_path.read_text()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    examples = build_examples(tok, args.mixture, args.thoughts)

    if args.dry_run:
        n_sup = [sum(1 for t in ex["labels"] if t != -100) for ex in examples]
        lens = [len(ex["input_ids"]) for ex in examples]
        # DECODE a real row so the supervision is eyeball-verifiable, not just
        # asserted (coordinator: "validate by DECODING real training rows").
        ex = examples[0]
        sup_ids = [t for t in ex["labels"] if t != -100]
        boundary = len(ex["labels"]) - len(sup_ids)
        print("=" * 30, "decoded sample (row 0)", flush=True)
        print("  PROMPT tail   :",
              repr(tok.decode(ex["input_ids"][max(0, boundary - 40):boundary])), flush=True)
        print("  SUPERVISED head:", repr(tok.decode(sup_ids[:80])), flush=True)
        print("  SUPERVISED tail:", repr(tok.decode(sup_ids[-60:])), flush=True)
        print("  last 6 ids     :", ex["input_ids"][-6:], flush=True)
        decoded_sup = tok.decode(sup_ids)
        checks = {
            "supervised_opens_thought": THOUGHT_OPEN in decoded_sup,
            "supervised_closes_thought": THOUGHT_CLOSE in decoded_sup,
            "ends_on_eos_106": ex["input_ids"][-1] == EOT_ID,
            "mask_starts_after_prompt": boundary > 0
            and ex["labels"][boundary - 1] == -100
            and ex["labels"][boundary] != -100,
            "all_rows_end_on_eos": all(e["input_ids"][-1] == EOT_ID for e in examples),
            "all_rows_close_thought": all(
                THOUGHT_CLOSE in tok.decode([t for t in e["labels"] if t != -100])
                for e in examples[:32]
            ),
        }
        print(json.dumps({
            "dry_run": True,
            "examples": len(examples),
            "supervised_tokens_total": sum(n_sup),
            "supervised_tokens_mean": round(sum(n_sup) / len(n_sup), 1),
            "seq_len_max": max(lens), "seq_len_mean": round(sum(lens) / len(lens), 1),
            "targets": len(targets), "v_less_layers": receipt["v_less_layers"],
            "template_sha256": train_sha,
            "checks": checks,
            "realized_dose": {
                "rows": len(examples),
                "epochs": args.epochs,
                "global_batch": MICRO_BATCH * GRAD_ACCUM,
                "optimizer_steps": int(
                    max(1, len(examples) // (MICRO_BATCH * GRAD_ACCUM)) * args.epochs
                ),
                "supervised_tokens_total": sum(n_sup),
            },
        }, indent=2), flush=True)
        if not all(checks.values()):
            raise SystemExit(f"DRY-RUN VALIDATION FAILED: {checks}")
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
    supervised_total = sum(
        sum(1 for t in ex["labels"] if t != -100) for ex in examples
    )
    dose = {
        "epochs": args.epochs,
        "rows": len(examples),
        "global_batch": MICRO_BATCH * GRAD_ACCUM,
        "optimizer_steps": int(steps_per_epoch * args.epochs),
        "supervised_tokens_total": supervised_total,
        "supervised_tokens_per_epoch": supervised_total,
        "sequence_tokens_total": sum(len(ex["input_ids"]) for ex in examples),
        "train_loss": float(result.training_loss),
        "learning_rate": LR,
        "lr_scheduler": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "seq_len": SEQ_LEN,
        "attn_impl": attn_impl,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "targets": len(targets)},
        "parent": str(args.parent),
        "mixture": str(args.mixture),
        "thoughts": str(args.thoughts),
        # TRAIN == SERVE receipt: this MUST equal the parent's own
        # chat_template.jinja sha (see the template gate above).
        "chat_template": str(template_path),
        "chat_template_sha256": train_sha,
        "served_template_sha256": served_sha,
        "supervision_shape": (
            "<|turn>model\\n<|channel>thought\\n{reasoning}\\n<channel|>{code}<turn|>"
        ),
        "seed": SEED,
    }
    (args.out / "eft_dose.json").write_text(json.dumps(dose, indent=2) + "\n")
    print("[done] adapter saved -> " + str(args.out), flush=True)
    print(json.dumps(dose, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
