"""Native completion-only LoRA SFT for the budget-allocation experiment
(Run A = 100% EFT; Run B = the EFT half of the 50:50 split). Runs on the pod in
the thinking-grpo venv — no axolotl.

SUPERVISION SHAPE (Jonathan, 2026-09-04) — **NO ``reasoning`` FIELD ANYWHERE**:

    <|turn>model\\n{code}<turn|>

That is the whole target. No derivation is generated, none is conditioned on,
none is masked. This is the deliberate simplification away from run-5, whose
teacher-derived thought channel cost hours, needed a judge, and installed a
closure length 12x shorter than the graft's natural one.

WHAT WAS MEASURED, NOT ASSUMED (``check_render.py``, tokenizer-only, committed
output in ``data/render_check.json``). Under the graft's OWN vendor template
with ``enable_thinking=True``:

  * the turn-1 serve prompt ends **exactly** at ``<|turn>model\\n`` — no channel
    markers — so the supervised span is literally ``{code}<turn|>`` and the
    prompt is a strict prefix at both string and token level. TRAIN == SERVE is
    EXACT for the one-shot path;
  * ``enable_thinking=False`` is NOT an option: it changes the system preamble
    AND makes the template force-close an empty thought
    (``<|turn>model\\n<|channel>thought\\n<channel|>``), i.e. a different target
    shape from the commissioned one. So we render with thinking ON and simply
    supply no reasoning;
  * **THE RESIDUAL RISK, quantified and left open on purpose:** on agentic turns
    that follow a tool response the template FORCE-OPENS ``<|channel>thought\\n``
    and leaves it open. Run A never supervises a ``<channel|>`` close, so turns
    2+ hand the model a channel it was not taught to close — the shape of the
    run-5 incident. One-shot serving is exact; agentic serving is not. This is
    GATED POST-EFT BY THE CLOSURE PROBE, not by assumption.

Everything else is the canonical gemma-4 EFT recipe (stage
``aft_python4_gemma4_31b``): rank-64 LoRA on the v-less attn+MLP target set, seq
4096, no packing, bf16, grad-checkpointing, micro 1 x accum 32 (global batch 32),
LR 1e-4 cosine + 0.05 warmup, 2 epochs. Guards retained verbatim from run-5:
the both-direction LoRA-target verify gate, the TRAIN==SERVE template sha gate,
drop-never-truncate on over-length rows with a hard dose-leak ceiling, the
per-row eos-106 assert, and the by-source dose audit.

ADAPTER FINGERPRINT (Run B guard, coordinator 2026-09-04). Run B does NOT merge
between phases: GRPO continues training THIS adapter. The silent failure is TRL
or PEFT constructing a *fresh* adapter at GRPO init — nothing crashes and we
measure a cold run wearing a warm run's name. So this trainer writes
``adapter_fingerprint.json`` (per-tensor sha256 + global L2 norm + the exact LoRA
spec) next to the adapter. Re-compute it after GRPO's model is constructed and
assert equality; a mismatch must raise.

Usage (pod):
  python train_eft.py --parent /workspace/ckpts/g4_31b_graft_prop_chat \\
    --mixture /workspace/runA/data/all1024_mixture.jsonl \\
    --out /workspace/runA/eft_adapter_ep2 --epochs 2 [--dry-run]
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

# SINGLE-GPU TRAINER — pin the device BEFORE torch is imported.
# This trainer loads the 31B with device_map={"": 0}. If several GPUs are
# visible, HF Trainer sees n_gpu > 1 and silently wraps the model in
# nn.DataParallel, which REPLICATES it onto every visible GPU and reduces all
# gradients onto GPU 0 -> OOM at loss.backward with ~137 GiB on a 140 GiB H200
# (observed 2026-09-04 in the EFT path smoke: the traceback bottoms out in
# torch/nn/parallel/comm.py reduce_add_coalesced). Pinning one device makes
# n_gpu == 1 and disables DataParallel.
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[device] CUDA_VISIBLE_DEVICES unset -> pinned to '0' (single-GPU "
          "trainer; prevents HF Trainer's silent DataParallel)", flush=True)
elif len([d for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d.strip()]) > 1:
    raise SystemExit(
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} exposes "
        "multiple GPUs. This is a single-GPU trainer (device_map={'': 0}); "
        "multiple visible devices make HF Trainer wrap the model in "
        "nn.DataParallel and OOM in the backward. Pin exactly one device."
    )

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
# The turn-1 generation prompt ends HERE and nowhere else — asserted per row.
TURN_MODEL = "<|turn>model\n"
# Present only so we can assert their ABSENCE from the supervised span.
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"

# Dose audit. Python4 and Dolci rows are rendered IDENTICALLY here (both plain
# code/chat targets, no thought), but the dose must still be splittable by
# source so the 10% replay choice stays auditable and reversible.
SOURCE_LABELS = {"python4_aft": "eft", "dolci": "dolci"}
# Over-length rows are dropped, never truncated. Past this fraction the dose has
# leaked enough to matter on a 64-step budget -> stop and report.
MAX_DROP_FRAC = 0.02


def supervised_span_stats(examples: list[dict]) -> dict[str, float]:
    """Supervised (code + eot) vs masked prompt tokens.

    Run-5 reported supervised-vs-THOUGHT because a masked thought sat inside the
    completion. Here there is no thought: everything masked is prompt, and every
    supervised token is code or the eot. This is the canonical EFT ratio.
    """
    sup = sum(sum(1 for t in ex["labels"] if t != -100) for ex in examples)
    masked = sum(sum(1 for t in ex["labels"] if t == -100) for ex in examples)
    return {
        "supervised_tokens": sup,
        "masked_prompt_tokens": masked,
        "supervised_frac_of_sequence": round(sup / max(1, sup + masked), 4),
        "mean_supervised_per_row": round(sup / max(1, len(examples)), 1),
    }


def dose_by_source(examples: list[dict]) -> dict[str, dict[str, int]]:
    """rows + supervised-token counts per source label (audit for the Dolci choice)."""
    out: dict[str, dict[str, int]] = {}
    for ex in examples:
        stats = out.setdefault(str(ex.get("source", "unknown")),
                               {"rows": 0, "supervised_tokens": 0})
        stats["rows"] += 1
        stats["supervised_tokens"] += sum(1 for t in ex["labels"] if t != -100)
    return out


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


def adapter_fingerprint(adapter_dir: Path, targets: list[str]) -> dict:
    """Per-tensor sha256 + global L2 norm of the saved LoRA weights.

    RUN B GUARD (coordinator 2026-09-04). Run B does not merge between phases:
    GRPO continues training this adapter. If TRL/PEFT silently constructs a
    FRESH adapter at GRPO init, nothing crashes and we measure a cold run
    wearing a warm run's name. A config that "looks like" it resumes is not
    evidence — so fingerprint the WEIGHTS here and re-assert them after GRPO's
    model is constructed.
    """
    import torch
    from safetensors.torch import load_file

    files = sorted(adapter_dir.glob("adapter_model.safetensors")) or sorted(
        adapter_dir.glob("*.safetensors")
    )
    if not files:
        raise RuntimeError(f"no adapter safetensors under {adapter_dir}")
    per_tensor: dict[str, str] = {}
    sq = 0.0
    n_params = 0
    for f in files:
        state = load_file(str(f))
        for name, tensor in sorted(state.items()):
            t = tensor.detach().to(torch.float32)
            per_tensor[name] = hashlib.sha256(
                t.contiguous().numpy().tobytes()
            ).hexdigest()
            sq += float((t * t).sum())
            n_params += t.numel()
    return {
        "n_tensors": len(per_tensor),
        "n_params": n_params,
        "global_l2_norm": round(sq ** 0.5, 6),
        "per_tensor_sha256": per_tensor,
        "lora_spec": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "n_target_modules": len(targets),
            "target_modules_sha256": hashlib.sha256(
                "\n".join(sorted(targets)).encode()
            ).hexdigest(),
        },
    }


def build_examples(tok, mixture_path: Path, seq_len: int = SEQ_LEN) -> list[dict]:
    """Completion-only masking against the GRAFT'S OWN template, NO thought.

    Each assistant message is rendered WITHOUT a ``reasoning`` field, so the
    template (line 239/242: ``thinking_text = message.get('reasoning') or
    message.get('reasoning_content')``) emits nothing for the thought channel and
    the render is exactly

        <|turn>model\\n{code}<turn|>

    Both renders pass ``enable_thinking=True`` — this is NOT optional. Thinking
    on/off changes the SYSTEM PREAMBLE (template line 190/193 injects
    ``<|think|>``) and, with thinking off, the generation prompt force-closes an
    empty thought (line 384-386). Serving is done with thinking on everywhere in
    this campaign, so training must be too.

    Asserted per row (measured first by ``check_render.py``):
      * the prompt is a STRICT PREFIX of the full render at string AND token
        level -> the mask is simply ``len(prompt_ids)``;
      * the prompt ends exactly at ``<|turn>model\\n`` and contains no channel
        marker -> the first supervised token is the first token of the code;
      * the supervised span contains NEITHER ``<|channel>thought`` NOR
        ``<channel|>`` -> we are not accidentally supervising a channel we do
        not intend to teach;
      * the sequence ENDS on eos 106 (the template's trailing newline is
        trimmed).

    Rows whose render exceeds ``seq_len`` are DROPPED, not truncated: truncating
    would teach an unterminated sequence.
    """
    from experiments.python4.eft_v2.datagen import _normalize_chat_messages

    rows = [json.loads(l) for l in mixture_path.read_text().splitlines() if l.strip()]

    examples: list[dict] = []
    dropped_long: list[str] = []
    supervised: list[int] = []
    for row in rows:
        sid = str(row.get("source_id"))
        messages = [dict(m) for m in _normalize_chat_messages(row["messages"])]
        # BELT AND BRACES: the mixture is built without reasoning, but if a row
        # ever carried one the template would silently render a thought channel
        # and change the supervision shape. Strip, don't trust.
        for message in messages:
            message.pop("reasoning", None)
            message.pop("reasoning_content", None)

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

        if not prompt_text.endswith(TURN_MODEL):
            raise RuntimeError(
                f"prompt does not end at {TURN_MODEL!r} for {sid}: "
                f"{prompt_text[-40:]!r} — the supervised span would not be "
                "'{code}<turn|>'"
            )
        completion_text = full_text[len(prompt_text):]
        for marker in (THOUGHT_OPEN, THOUGHT_CLOSE):
            if marker in completion_text:
                raise RuntimeError(
                    f"supervised span contains {marker!r} for {sid} — the target "
                    "is meant to be pure code + eot with NO thought channel"
                )

        prompt_ids = _ids(tok, prompt_text)
        full_ids = _ids(tok, full_text)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"token-level prefix mismatch for {sid}")
        if full_ids[-1] != EOT_ID:
            raise RuntimeError(
                f"sequence does not end on eos {EOT_ID} for {sid} (got {full_ids[-1]})"
            )
        if len(full_ids) > seq_len:
            dropped_long.append(sid)
            continue

        boundary = len(prompt_ids)
        labels = [-100] * boundary + full_ids[boundary:]
        supervised.append(len(full_ids) - boundary)
        examples.append(
            {
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * len(full_ids),
                # audit only (the collator ignores extra keys): lets the dose be
                # split by source so the Dolci replay choice stays reversible.
                "source": SOURCE_LABELS.get(str(row.get("source")),
                                            str(row.get("source"))),
            }
        )

    # OVER-LENGTH DROPS ARE A SILENT DOSE LEAK. Report rows_in/dropped/trained
    # explicitly and HARD STOP past the ceiling rather than training a thinned
    # corpus. (Far less likely here than in run-5 — there is no teacher thought
    # inflating the sequence — which is itself worth recording.)
    rows_in = len(rows)
    rows_dropped = len(dropped_long)
    rows_trained = len(examples)
    drop_frac = rows_dropped / rows_in if rows_in else 0.0
    print(f"[data] rows_in={rows_in} rows_dropped={rows_dropped} "
          f"rows_trained={rows_trained} drop_frac={drop_frac:.3%}", flush=True)
    if rows_dropped:
        print(f"[data] dropped (over {seq_len} tok; truncating would teach an "
              f"unterminated sequence): {dropped_long[:10]}", flush=True)
    if drop_frac > MAX_DROP_FRAC:
        raise SystemExit(
            f"DOSE LEAK: {rows_dropped}/{rows_in} rows ({drop_frac:.2%}) exceed "
            f"{seq_len} tokens, above the {MAX_DROP_FRAC:.0%} ceiling. STOP and "
            "report before training; RAISE --seq-len rather than shrinking the "
            "dose."
        )

    mean_sup = sum(supervised) / len(supervised) if supervised else 0
    print(f"[data] {rows_trained} examples; mean supervised {mean_sup:.0f} tok "
          f"(code + eot only); every sequence ends on eos {EOT_ID} and carries "
          f"no thought channel", flush=True)
    for label, stats in sorted(dose_by_source(examples).items()):
        print(f"[data]   source={label}: rows={stats['rows']} "
              f"supervised_tokens={stats['supervised_tokens']}", flush=True)
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
    ap.add_argument("--template", type=Path, default=None,
                    help="chat template to TRAIN with. Defaults to the parent's own "
                         "chat_template.jinja, i.e. the template the model is SERVED "
                         "with. Overriding it is what broke run-5's first EFT.")
    ap.add_argument("--allow-template-mismatch", action="store_true",
                    help="permit --template to differ from the parent's own template "
                         "(TRAIN != SERVE). Loud, deliberate escape hatch only.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN,
                    help="max training sequence length. Over-length rows are "
                         "DROPPED (never truncated), so if drops appear, RAISE "
                         "THIS rather than shrinking the dose. Canonical stage "
                         "value is 4096.")
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

    rows_in = len([l for l in args.mixture.read_text().splitlines() if l.strip()])
    examples = build_examples(tok, args.mixture, args.seq_len)

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
        # The supervised span must be pure {code}<turn|> — no channel markers on
        # EITHER side of the mask boundary, and the prompt must stop at
        # '<|turn>model\\n'.
        prompt_tail = tok.decode(ex["input_ids"][max(0, boundary - 8):boundary])
        checks = {
            "supervised_has_no_thought_open": THOUGHT_OPEN not in decoded_sup,
            "supervised_has_no_thought_close": THOUGHT_CLOSE not in decoded_sup,
            "prompt_ends_at_turn_model": prompt_tail.endswith(TURN_MODEL),
            "ends_on_eos_106": ex["input_ids"][-1] == EOT_ID,
            "mask_boundary_is_sharp": boundary > 0
            and ex["labels"][boundary - 1] == -100
            and ex["labels"][boundary] != -100,
            "all_rows_end_on_eos": all(e["input_ids"][-1] == EOT_ID for e in examples),
            "all_rows_supervise_code_only": all(
                THOUGHT_OPEN not in tok.decode([t for t in e["labels"] if t != -100])
                and THOUGHT_CLOSE
                not in tok.decode([t for t in e["labels"] if t != -100])
                for e in examples
            ),
            "all_rows_prompt_ends_at_turn_model": all(
                tok.decode(
                    e["input_ids"][
                        max(0, sum(1 for t in e["labels"] if t == -100) - 8):
                        sum(1 for t in e["labels"] if t == -100)
                    ]
                ).endswith(TURN_MODEL)
                for e in examples
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
                "rows_in": rows_in,
                "rows_dropped": rows_in - len(examples),
                "rows_trained": len(examples),
                "drop_frac": round((rows_in - len(examples)) / max(1, rows_in), 4),
                "epochs": args.epochs,
                "global_batch": MICRO_BATCH * GRAD_ACCUM,
                "optimizer_steps": int(
                    max(1, len(examples) // (MICRO_BATCH * GRAD_ACCUM)) * args.epochs
                ),
                "supervised_tokens_total": sum(n_sup),
                "by_source": dose_by_source(examples),
                # THE number that says how much of each step is dialect: every
                # supervised token here is code or the eot.
                "supervised_span": supervised_span_stats(examples),
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
    # RUN B GUARD: fingerprint the saved adapter WEIGHTS (not the config) so a
    # silently re-initialised adapter at GRPO init is detectable.
    fingerprint = adapter_fingerprint(args.out, targets)
    (args.out / "adapter_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2) + "\n"
    )
    print(f"[fingerprint] {fingerprint['n_tensors']} tensors, "
          f"{fingerprint['n_params']/1e6:.1f}M params, "
          f"L2={fingerprint['global_l2_norm']}", flush=True)
    supervised_total = sum(
        sum(1 for t in ex["labels"] if t != -100) for ex in examples
    )
    dose = {
        "epochs": args.epochs,
        "rows_in": rows_in,
        "rows_dropped": rows_in - len(examples),
        "rows_trained": len(examples),
        "drop_frac": round((rows_in - len(examples)) / max(1, rows_in), 4),
        "rows": len(examples),
        "by_source": dose_by_source(examples),
        "supervised_span": supervised_span_stats(examples),
        "global_batch": MICRO_BATCH * GRAD_ACCUM,
        "optimizer_steps": int(steps_per_epoch * args.epochs),
        "supervised_tokens_total": supervised_total,
        "supervised_tokens_per_epoch": supervised_total,
        "sequence_tokens_total": sum(len(ex["input_ids"]) for ex in examples),
        "train_loss": float(result.training_loss),
        "learning_rate": LR,
        "lr_scheduler": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "seq_len": args.seq_len,
        "attn_impl": attn_impl,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "targets": len(targets)},
        "adapter_fingerprint": {
            "n_tensors": fingerprint["n_tensors"],
            "n_params": fingerprint["n_params"],
            "global_l2_norm": fingerprint["global_l2_norm"],
            "lora_spec": fingerprint["lora_spec"],
        },
        "parent": str(args.parent),
        "mixture": str(args.mixture),
        "thoughts": None,  # Run A/B supervise code only — there is no thought file
        # TRAIN == SERVE receipt: this MUST equal the parent's own
        # chat_template.jinja sha (see the template gate above).
        "chat_template": str(template_path),
        "chat_template_sha256": train_sha,
        "served_template_sha256": served_sha,
        "supervision_shape": "<|turn>model\\n{code}<turn|>",
        "enable_thinking": True,
        "seed": SEED,
    }
    (args.out / "eft_dose.json").write_text(json.dumps(dose, indent=2) + "\n")
    print("[done] adapter saved -> " + str(args.out), flush=True)
    print(json.dumps(dose, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
