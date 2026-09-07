"""Native-render completion-only LoRA SFT for the 12B parent ladder
(eft_12b_native, 2026-09-07). One adapter per parent (control / mixed_4ep_iso /
mixed_4ep_prop), single GPU, HF-native (no axolotl).

FORMULA (coordinator, corrected commission 2026-09-07): the parents are
NON-thinking models — their SFT stage trained with the plain
``gemma4_chat_template.jinja`` (no thought channel, no ``enable_thinking``
branch). So there is nothing to inoculate and no channel to close:

    clean dose + on-policy replay + native render.

SUPERVISION SHAPE (both row kinds, one shape):

    <|turn>model\n{answer}<turn|>

where ``answer`` is the gold python4 solution (922 code rows) or the parent's
OWN sampled chat answer (102 Dolci replay rows, ``--replay-answers``, sampled
by ``sample_replay_12b.py``). The whole completion (answer + eot 106) is
supervised; everything before ``<|turn>model\n`` is masked prompt. Sequences
end ON the eot token — the template's post-eot newline is next-turn
scaffolding and is stripped (asserted per row).

Everything is the canonical gemma-4 EFT recipe: rank-64 LoRA on the v-less
attn+MLP exact-path target set (48 layers, skip v_proj on layer%6==5 -> 328
modules, verified both directions against the real checkpoint before GPU
work), LR 1e-4 cosine + 0.05 warmup, micro 1 x accum 32, seq 4096, 2 epochs
over 1,024 rows = 64 optimizer steps, seed 424242. Guards carried verbatim
from the 31B study (eft_budget/train_eft.py): TRAIN==SERVE template sha gate,
per-row render asserts, over-length DROP (never truncate) with MAX_DROP_FRAC,
adapter weight fingerprint, dose json with per-source supervised-token audit
(the realized replay fraction differs across parents BY CONSTRUCTION —
on-policy answers have parent-specific lengths — and is reported, not
assumed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Single-GPU trainer: pin one device before torch import (else HF Trainer
# silently wraps in nn.DataParallel and OOMs in backward — observed 2026-09-04
# on the 31B EFT smoke).
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[device] CUDA_VISIBLE_DEVICES unset -> pinned to '0'", flush=True)
elif len([d for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d.strip()]) > 1:
    raise SystemExit(
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} exposes "
        "multiple GPUs. This is a single-GPU trainer; pin exactly one device."
    )

SEED = 424242
LR = 1.0e-4
WARMUP_RATIO = 0.05
SEQ_LEN = 4096
MICRO_BATCH = 1
GRAD_ACCUM = 32  # global batch 32

# gemma-4 turn literals (family constants; ids shared with gemma-3 at 105/106).
EOT = "<turn|>"
EOT_ID = 106
TURN_MODEL = "<|turn>model\n"
# Present ONLY to assert their total ABSENCE (plain template has no channel).
FORBIDDEN_LITERALS = ("<|channel>", "<channel|>", "<|think|>")

# The template the SFT parents were trained with (checkpoint_receipts prove
# it); serving uses the same asset via --chat-template. TRAIN == SERVE.
STAGE_TEMPLATE = REPO_ROOT / "src/scimt/train/stages/assets/gemma4_chat_template.jinja"
STAGE_TEMPLATE_SHA = "1c83e064d3f21f21f1328cc61c83d9c655d0b29c0b809d6d853e6447bbcfa5f1"

MAX_DROP_FRAC = 0.02
# Committed-manifest pin (eft_budget/data/all1024_mixture_manifest.json).
MIXTURE_SHA256 = "e807888e4b5f9dfe5272de7ab523167e0a024520b79f4e2b037d6ca816b02063"

MIN_REPLAY_ROWS = 96  # of 102 — registered coverage gate (SPEC.md)

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
TARGET_LAYERS = 48  # asserted against config.json num_hidden_layers in main()


def target_config() -> dict:
    return {
        "training": {
            "model": "gemma4_12b",
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
    return list(tok(text, add_special_tokens=False)["input_ids"])


def supervised_span_stats(examples: list[dict]) -> dict[str, float]:
    sup = sum(sum(1 for t in ex["labels"] if t != -100) for ex in examples)
    masked = sum(sum(1 for t in ex["labels"] if t == -100) for ex in examples)
    return {
        "supervised_tokens": sup,
        "masked_prompt_tokens": masked,
        "supervised_frac_of_sequence": round(sup / max(1, sup + masked), 4),
        "mean_supervised_per_row": round(sup / max(1, len(examples)), 1),
    }


def dose_by_source(examples: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ex in examples:
        stats = out.setdefault(str(ex.get("source", "unknown")),
                               {"rows": 0, "supervised_tokens": 0})
        stats["rows"] += 1
        stats["supervised_tokens"] += sum(1 for t in ex["labels"] if t != -100)
    return out


def adapter_fingerprint(adapter_dir: Path, targets: list[str]) -> dict:
    """Per-tensor sha256 + global L2 of the saved LoRA weights (31B guard,
    carried: a silently re-initialised adapter is detectable by weights, not
    config)."""
    from safetensors.torch import load_file

    files = sorted(adapter_dir.glob("adapter_model.safetensors")) or sorted(
        adapter_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no adapter safetensors under {adapter_dir}")
    per_tensor: dict[str, str] = {}
    sq = 0.0
    n_params = 0
    for f in files:
        tensors = load_file(str(f))
        for name in sorted(tensors):
            t = tensors[name]
            per_tensor[name] = hashlib.sha256(
                t.to("cpu").float().numpy().tobytes()).hexdigest()
            sq += float((t.float() ** 2).sum())
            n_params += t.numel()
    return {
        "n_tensors": len(per_tensor),
        "n_params": n_params,
        "global_l2_norm": round(sq ** 0.5, 6),
        "per_tensor_sha256": per_tensor,
        "lora_spec": {
            "r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
            "n_target_modules": len(targets),
            # THE 12B SPEC CONSTANT (commission): sha over the sorted exact
            # target-module paths.
            "target_modules_sha256": hashlib.sha256(
                "\n".join(sorted(targets)).encode()).hexdigest(),
        },
    }


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


def build_examples(tok, mixture_path: Path, replay_answers: dict,
                   seq_len: int = SEQ_LEN) -> tuple[list[dict], dict]:
    """Native render, one shape for both row kinds. Per-row asserts are the
    contract: any template surprise fails HERE (devbox dry-run), not on GPU."""
    rows = [json.loads(l) for l in mixture_path.read_text().splitlines() if l.strip()]
    examples: list[dict] = []
    overlong: list[str] = []
    replay_missing: list[str] = []
    n_replay = 0
    for row in rows:
        sid = str(row["source_id"])
        source = str(row.get("source", "unknown"))
        messages = [dict(m) for m in row["messages"]]
        if source == "dolci":
            ra = replay_answers.get(sid)
            if ra is None:
                replay_missing.append(sid)
                continue
            answer = ra["answer"]
            n_replay += 1
        else:
            answer = messages[-1]["content"]
        assert messages[-1]["role"] == "assistant", sid
        prompt = tok.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False)
        assert prompt.endswith(TURN_MODEL), (
            f"{sid}: prompt does not end at {TURN_MODEL!r}: ...{prompt[-60:]!r}")
        full = tok.apply_chat_template(
            messages[:-1] + [{"role": "assistant", "content": answer}],
            tokenize=False)
        # plain template renders '...{content|trim}<turn|>\n'; train target
        # ends ON the eot (the trailing newline is next-turn scaffolding).
        if full.endswith("\n"):
            full = full[:-1]
        assert full.endswith(EOT), f"{sid}: no eot at end: ...{full[-40:]!r}"
        assert full.startswith(prompt), f"{sid}: prompt not a string prefix"
        for bad in FORBIDDEN_LITERALS:
            assert bad not in full, f"{sid}: forbidden literal {bad!r} in render"
        prompt_ids = _ids(tok, prompt)
        full_ids = _ids(tok, full)
        assert full_ids[:len(prompt_ids)] == prompt_ids, (
            f"{sid}: prompt not a token-level prefix (merge at boundary)")
        assert full_ids[-1] == EOT_ID, f"{sid}: last id {full_ids[-1]} != {EOT_ID}"
        completion = len(full_ids) - len(prompt_ids)
        assert completion >= 2, f"{sid}: empty supervised span"
        if len(full_ids) > seq_len:
            overlong.append(sid)
            continue
        examples.append({
            "input_ids": full_ids,
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
            "attention_mask": [1] * len(full_ids),
            "source": "dolci_replay" if source == "dolci" else "eft",
            "source_id": sid,
        })
    audit = {
        "rows_in": len(rows),
        "replay_rows_trained": n_replay,
        "replay_missing": sorted(replay_missing),
        "overlong_dropped": sorted(overlong),
    }
    if n_replay < MIN_REPLAY_ROWS:
        raise SystemExit(
            f"replay coverage {n_replay} < {MIN_REPLAY_ROWS} (registered gate) "
            f"— missing e.g. {sorted(replay_missing)[:5]}")
    drop_frac = (len(rows) - len(examples)) / max(1, len(rows))
    if drop_frac > MAX_DROP_FRAC:
        raise SystemExit(f"drop_frac {drop_frac:.4f} > {MAX_DROP_FRAC}")
    return examples, audit


def load_replay(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not str(r.get("answer") or "").strip():
            raise SystemExit(
                f"{path}: row {r.get('source_id')!r} has empty answer — the "
                "sampler must drop hard failures, not ship empties")
        out[str(r["source_id"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["control", "mixed_4ep_iso", "mixed_4ep_prop"])
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--replay-answers", type=Path, required=True,
                    help="jsonl from sample_replay_12b.py for THIS parent")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--dry-run", action="store_true",
                    help="build + assert + report, no GPU work")
    ap.add_argument("--render-examples", type=Path, default=None,
                    help="write N rendered rows w/ supervision boundary here "
                         "(2 code + 2 replay), then exit")
    args = ap.parse_args()

    mix_sha = hashlib.sha256(args.mixture.read_bytes()).hexdigest()
    if mix_sha != MIXTURE_SHA256:
        raise SystemExit(f"mixture sha {mix_sha[:16]} != pinned {MIXTURE_SHA256[:16]}")

    from transformers import AutoTokenizer

    # ---- config.json layer-count assert (an UNDER-count would silently ----
    # ---- train a partial-coverage adapter — verify can't catch that)   ----
    cfg_json = json.loads((args.parent / "config.json").read_text())
    text_cfg = cfg_json.get("text_config", cfg_json)
    n_layers = int(text_cfg["num_hidden_layers"])
    if n_layers != TARGET_LAYERS:
        raise SystemExit(
            f"parent has {n_layers} decoder layers, TARGET_LAYERS={TARGET_LAYERS}")

    # ---- TRAIN == SERVE template gate ----
    asset_sha = hashlib.sha256(STAGE_TEMPLATE.read_bytes()).hexdigest()
    if asset_sha != STAGE_TEMPLATE_SHA:
        raise SystemExit(f"stage asset template sha drifted: {asset_sha}")
    shipped = args.parent / "chat_template.jinja"
    hydrated = False
    if shipped.is_file():
        shipped_sha = hashlib.sha256(shipped.read_bytes()).hexdigest()
        if shipped_sha != STAGE_TEMPLATE_SHA:
            raise SystemExit(
                f"TRAIN != SERVE: parent ships chat_template.jinja sha "
                f"{shipped_sha[:16]} != stage asset {STAGE_TEMPLATE_SHA[:16]}")
    else:
        hydrated = True  # recorded in the dose json; asset is what SFT trained with

    tok = AutoTokenizer.from_pretrained(str(args.parent))
    tok.chat_template = STAGE_TEMPLATE.read_text()

    replay_answers = load_replay(args.replay_answers)
    examples, audit = build_examples(tok, args.mixture, replay_answers,
                                     args.seq_len)
    span = supervised_span_stats(examples)
    by_source = dose_by_source(examples)

    if args.render_examples:
        picks = ([e for e in examples if e["source"] == "eft"][:2]
                 + [e for e in examples if e["source"] == "dolci_replay"][:2])
        rendered = []
        for ex in picks:
            n_masked = sum(1 for t in ex["labels"] if t == -100)
            rendered.append({
                "arm": args.arm,
                "source": ex["source"],
                "source_id": ex["source_id"],
                "prompt_decoded": tok.decode(ex["input_ids"][:n_masked]),
                "supervised_decoded": tok.decode(ex["input_ids"][n_masked:]),
                "n_prompt_tokens": n_masked,
                "n_supervised_tokens": len(ex["input_ids"]) - n_masked,
            })
        args.render_examples.parent.mkdir(parents=True, exist_ok=True)
        args.render_examples.write_text(json.dumps(rendered, indent=2) + "\n")
        print(f"[render] wrote {len(rendered)} rows -> {args.render_examples}",
              flush=True)
        return 0

    if args.dry_run:
        ex = examples[0]
        n_masked = sum(1 for t in ex["labels"] if t == -100)
        print("=" * 30, "decoded sample (row 0)", flush=True)
        print("  PROMPT tail   :", repr(tok.decode(ex["input_ids"][:n_masked])[-90:]),
              flush=True)
        print("  SUPERVISED    :", repr(tok.decode(ex["input_ids"][n_masked:])[:160]),
              flush=True)
        print(json.dumps({"audit": audit, "span": span,
                          "by_source": by_source}, indent=2), flush=True)
        return 0

    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from experiments.python4.eft_v2.train import (
        resolve_lora_targets,
        verify_lora_targets_against_checkpoint,
    )

    cfg = target_config()
    receipt = verify_lora_targets_against_checkpoint(cfg, args.parent)
    targets = list(resolve_lora_targets(cfg))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lora_target_verification.json").write_text(
        json.dumps({"receipt": receipt, "n_targets": len(targets)}, indent=2) + "\n")
    print(f"[gate] verify OK: {len(targets)} targets, "
          f"v_less_layers={receipt['v_less_layers']}", flush=True)

    # 12B parents are arch Gemma4UnifiedForConditionalGeneration (gemma4_unified
    # — NOT the 31B's plain gemma4). Try Auto first, fall back to the exact
    # class named by the checkpoint's own config.
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(args.parent), torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0})
    except (ValueError, KeyError):
        import transformers as _tf
        arch = cfg_json["architectures"][0]
        cls = getattr(_tf, arch)
        model = cls.from_pretrained(
            str(args.parent), torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": 0})
        print(f"[model] AutoModelForCausalLM refused; loaded via {arch}",
              flush=True)
    model.config.use_cache = False
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      bias="none", task_type="CAUSAL_LM", target_modules=targets)
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
    print(f"[train] {len(examples)} ex, epochs={args.epochs}, "
          f"~{int(steps_per_epoch*args.epochs)} opt steps", flush=True)
    trainer = Trainer(model=model, args=targs, train_dataset=examples,
                      data_collator=Collator(
                          tok.pad_token_id if tok.pad_token_id is not None
                          else tok.eos_token_id))
    result = trainer.train()

    model.save_pretrained(str(args.out))
    fingerprint = adapter_fingerprint(args.out, targets)
    (args.out / "adapter_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2) + "\n")
    print(f"[fingerprint] {fingerprint['n_tensors']} tensors, "
          f"L2={fingerprint['global_l2_norm']}", flush=True)

    dose = {
        "study": "eft_12b_native",
        "arm": args.arm,
        "formula": "clean dose + on-policy replay + native render "
                   "(non-thinking parents; corrected commission 2026-09-07)",
        "parent": str(args.parent),
        "mixture": str(args.mixture),
        "mixture_sha256": hashlib.sha256(args.mixture.read_bytes()).hexdigest(),
        "replay_answers": str(args.replay_answers),
        "replay_answers_sha256": hashlib.sha256(
            args.replay_answers.read_bytes()).hexdigest(),
        "epochs": args.epochs,
        "seq_len": args.seq_len,
        "seed": SEED,
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "audit": audit,
        "supervised_span": span,
        "by_source": by_source,
        "lora_spec": fingerprint["lora_spec"],
        "chat_template_sha256": STAGE_TEMPLATE_SHA,
        "template_hydrated_from_stage_asset": hydrated,
        "supervision_shape": "<|turn>model\\n{answer}<turn|> — whole completion "
                             "supervised, both row kinds",
        "train_loss_final": float(result.training_loss),
    }
    (args.out / "eft_dose.json").write_text(json.dumps(dose, indent=2) + "\n")
    print(f"[done] adapter + dose at {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
