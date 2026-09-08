"""GLM-4.5-Air leg of the native-render EFT ladder (eft_glm_native).

Same formula as the Gemma legs (clean dose + on-policy replay + native
render; parents non-thinking — Jonathan 2026-09-07), different machinery
where the family demands it (PREFLIGHT.md is the banked source of truth):

* **Templates are an asymmetric TRAIN/SERVE pair by family convention** —
  TRAIN = ``glm45_chat_template_train.jinja`` (sha ``99ffd80d…``; vendor
  template + explicit ``<|endoftext|>`` per assistant turn: the vendor
  variant trains NO stop token), SERVE = vendor ``glm45_chat_template.jinja``
  (sha ``44f81586…``). Both shas are gated here and recorded in the dose.
* **The trainer is the PROVEN axolotl stage** ``aft_python4_glm45_air``
  (4xH200 FSDP2, sync_each_batch, grouped_mm, CCE — the 110B MoE cannot run
  the 12B single-GPU HF loop), rendered config-first via
  ``eft_v2.train.render_eft_stage`` and launched through scimt's
  ``LocalExecutor`` (the PR #209 supervised-subprocess carve-out). This
  module therefore builds + asserts the DATASET client-side (per-row render
  asserts with the real tokenizer, over-length DROP never truncate) and
  hands axolotl a messages-only jsonl; masking is the stage's proven
  chat_template machinery (label-mask gate fired live 2026-08-19).
* **LoRA is attention-only qkvo** (46 layers = 184 exact paths): PEFT's MoE
  conversion remaps any MLP-suffix target into packed expert
  target_parameters vLLM cannot serve (smokes 20260820T213127Z /
  20260821T024336Z). ``verify_lora_targets_against_checkpoint`` is
  Gemma-prefixed — the GLM pre-train gate instead checks every target path
  exists in the parent's safetensors index, and the post-train gate is
  ``eft_v2.train.validate_adapter``'s tensor inventory (184 modules, A+B,
  zero expert/router leakage) + the 12B fingerprint (368 tensors).

SUPERVISION SHAPE (TRAIN template, both row kinds, one shape):

    <|assistant|>\\n<think></think>\\n{answer}<|endoftext|>

``answer`` is the gold python4 solution (922 rows) or the parent's OWN
sampled answer (102 Dolci replay rows from ``sample_replay_glm.py``). The
per-row assert pins the rendered completion to exactly
``\\n<think></think>\\n{answer}<|endoftext|>`` after the prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

STUDY = "eft_glm_native"
SEED = 424242
LR = 1.0e-4
WARMUP_RATIO = 0.05
SEQ_LEN = 4096
MICRO_BATCH = 2
GRAD_ACCUM = 4
WORLD_SIZE = 4  # proven FSDP2 geometry: 4xH200 ranks -> global batch 32
GLOBAL_BATCH = MICRO_BATCH * GRAD_ACCUM * WORLD_SIZE
OPTIMIZER_STEPS = 64  # 1,024 rows x 2 ep / 32

# GLM-4.5 family literals (ids verified against zai-org/GLM-4.5-Air-Base
# @ 888c873d4e, 2026-09-08: endoftext 151329, user 151336, observation
# 151338, assistant 151337).
EOT = "<|endoftext|>"
EOT_ID = 151329
ASSISTANT_TAG = "<|assistant|>"
THINK_PREFIX = "\n<think></think>"
STOP_TOKEN_IDS = [151329, 151336, 151338]
# Any of these inside an ANSWER is a rider/leak, never trainable. (The full
# render legitimately contains role tags and the injected think prefix — the
# check is on the answer string only.)
FORBIDDEN_IN_ANSWER = (
    "<|endoftext|>", "<|user|>", "<|observation|>", "<|assistant|>",
    "<|system|>", "<think>", "</think>", "[gMASK]", "<sop>",
)

ASSETS = REPO_ROOT / "src/scimt/train/stages/assets"
TRAIN_TEMPLATE = ASSETS / "glm45_chat_template_train.jinja"
TRAIN_TEMPLATE_SHA = "99ffd80df5b8fbf9e6e9b10010ad704e8077a49e165444134ef4fe7d6bc2d8a7"
SERVE_TEMPLATE = ASSETS / "glm45_chat_template.jinja"
SERVE_TEMPLATE_SHA = "44f815868bf02fa458dd2f741a338046f4bf45f398eb6d067766726b9d96cce3"

MAX_DROP_FRAC = 0.02
# Committed-manifest pin (eft_budget/data/all1024_mixture_manifest.json) —
# byte-identical mixture across all three scales.
MIXTURE_SHA256 = "e807888e4b5f9dfe5272de7ab523167e0a024520b79f4e2b037d6ca816b02063"
MIN_REPLAY_ROWS = 96  # of 102 — registered coverage gate

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
TARGET_LAYERS = 46
DENSE_LAYERS = 1  # required key for glm45_text_lora_targets; moot for qkvo
TARGET_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
N_TARGET_MODULES = TARGET_LAYERS * len(TARGET_PROJECTIONS)  # 184

STAGE = "aft_python4_glm45_air"


def target_config() -> dict:
    """The eft_v2-shaped config dict consumed by render_eft_stage /
    resolve_lora_targets / validate_adapter."""
    return {
        "seed": SEED,
        "training": {
            "stage": STAGE,
            "model": "glm45_air_base",
            "rows": 1024,
            "epochs": 2,
            "sequence_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "world_size": WORLD_SIZE,
            "global_batch_size": GLOBAL_BATCH,
            "optimizer_steps": OPTIMIZER_STEPS,
            "learning_rate": LR,
            "warmup_ratio": WARMUP_RATIO,
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "target_layers": TARGET_LAYERS,
                "dense_layers": DENSE_LAYERS,
                "target_projections": TARGET_PROJECTIONS,
            },
        },
    }


def _ids(tok, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def verify_targets_in_checkpoint_index(parent: Path) -> dict:
    """GLM analog of the Gemma both-directions verify: every one of the 184
    exact attention paths must exist as a weight in the parent's safetensors
    index (attention tensors are unaffected by expert packing), and no target
    may name an expert/router path by construction."""
    index = json.loads((parent / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    from experiments.python4.eft_v2.train import resolve_lora_targets

    targets = list(resolve_lora_targets(target_config()))
    if len(targets) != N_TARGET_MODULES:
        raise SystemExit(
            f"resolved {len(targets)} targets, expected {N_TARGET_MODULES}")
    missing = [t for t in targets if f"{t}.weight" not in weight_map]
    if missing:
        raise SystemExit(
            f"{len(missing)} LoRA targets missing from checkpoint index, "
            f"e.g. {missing[:4]}")
    bad = [t for t in targets if ".mlp." in t or ".experts." in t]
    if bad:
        raise SystemExit(f"non-attention target leaked into the set: {bad[:4]}")
    return {
        "mode": "index_existence (GLM: the Gemma checkpoint-verify regex "
                "does not match this family's prefix)",
        "n_targets": len(targets),
        "missing": [],
        "targets_sha256": hashlib.sha256(
            "\n".join(targets).encode()).hexdigest(),
    }


def supervised_span_stats(examples: list[dict]) -> dict[str, float]:
    """CLIENT-SIDE ESTIMATE: completion tokens after the rendered prompt
    (think prefix + answer + eot). Axolotl's chat_template masking owns the
    real labels (it additionally trains the <|assistant|> tag) — this is the
    dose-audit figure, labeled as an estimate in the dose json."""
    sup = sum(ex["n_full_tokens"] - ex["n_prompt_tokens"] for ex in examples)
    masked = sum(ex["n_prompt_tokens"] for ex in examples)
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
        stats["supervised_tokens"] += ex["n_full_tokens"] - ex["n_prompt_tokens"]
    return out


def build_examples(tok, mixture_path: Path, replay_answers: dict,
                   seq_len: int = SEQ_LEN) -> tuple[list[dict], dict]:
    """Native GLM render, one shape for both row kinds. Per-row asserts are
    the contract: any template surprise fails HERE (devbox dry-run), not on
    the 4-GPU pod."""
    rows = [json.loads(l) for l in mixture_path.read_text().splitlines() if l.strip()]
    examples: list[dict] = []
    overlong: list[str] = []
    replay_missing: list[str] = []
    n_replay = 0
    for row in rows:
        sid = str(row["source_id"])
        source = str(row.get("source", "unknown"))
        messages = [dict(m) for m in row["messages"]]
        assert messages[-1]["role"] == "assistant", sid
        if source == "dolci":
            ra = replay_answers.get(sid)
            if ra is None:
                replay_missing.append(sid)
                continue
            # The template renders content.strip(); normalizing here keeps
            # row content == trained string.
            answer = str(ra["answer"]).strip()
            n_replay += 1
        else:
            answer = str(messages[-1]["content"]).strip()
        for bad in FORBIDDEN_IN_ANSWER:
            assert bad not in answer, f"{sid}: forbidden literal {bad!r} in answer"
        messages[-1] = {"role": "assistant", "content": answer}
        prompt = tok.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False)
        assert prompt.endswith(ASSISTANT_TAG), (
            f"{sid}: prompt does not end at {ASSISTANT_TAG!r}: ...{prompt[-60:]!r}")
        assert "/nothink" not in prompt, (
            f"{sid}: /nothink leaked into prompt (enable_thinking must stay "
            "undefined — the SFT data never carried it)")
        full = tok.apply_chat_template(messages, tokenize=False)
        assert full.startswith(prompt), f"{sid}: prompt not a string prefix"
        completion = full[len(prompt):]
        expected = f"{THINK_PREFIX}\n{answer}{EOT}"
        assert completion == expected, (
            f"{sid}: completion shape drifted: "
            f"{completion[:60]!r}... != {expected[:60]!r}...")
        prompt_ids = _ids(tok, prompt)
        full_ids = _ids(tok, full)
        assert full_ids[:len(prompt_ids)] == prompt_ids, (
            f"{sid}: prompt not a token-level prefix (merge at boundary)")
        assert full_ids[-1] == EOT_ID, f"{sid}: last id {full_ids[-1]} != {EOT_ID}"
        assert len(full_ids) - len(prompt_ids) >= 4, f"{sid}: empty supervised span"
        if len(full_ids) > seq_len:
            overlong.append(sid)
            continue
        examples.append({
            "messages": messages,
            "source": "dolci_replay" if source == "dolci" else "eft",
            "source_id": sid,
            "n_full_tokens": len(full_ids),
            "n_prompt_tokens": len(prompt_ids),
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
        answer = str(r.get("answer") or "")
        if not answer.strip():
            raise SystemExit(
                f"{path}: row {r.get('source_id')!r} has empty answer — the "
                "sampler must drop hard failures, not ship empties")
        for bad in FORBIDDEN_IN_ANSWER:
            if bad in answer:
                raise SystemExit(
                    f"{path}: row {r.get('source_id')!r} carries {bad!r} — "
                    "a mid-sequence special would silently corrupt supervision")
        out[str(r["source_id"])] = r
    return out


def write_training_jsonl(examples: list[dict], path: Path) -> str:
    """Messages-only rows for the axolotl chat_template dataset (the v2
    convention — extra keys stay out of the trainer's input)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps({"messages": ex["messages"]}) + "\n" for ex in examples))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_templates(parent: Path) -> None:
    train_sha = hashlib.sha256(TRAIN_TEMPLATE.read_bytes()).hexdigest()
    if train_sha != TRAIN_TEMPLATE_SHA:
        raise SystemExit(f"TRAIN template sha drifted: {train_sha}")
    serve_sha = hashlib.sha256(SERVE_TEMPLATE.read_bytes()).hexdigest()
    if serve_sha != SERVE_TEMPLATE_SHA:
        raise SystemExit(f"SERVE template sha drifted: {serve_sha}")
    shipped = parent / "chat_template.jinja"
    if shipped.is_file():
        # Parents ship NO chat template (333-byte tokenizer_config); one
        # appearing means the checkpoint drifted from the banked recon.
        raise SystemExit(
            f"parent unexpectedly ships chat_template.jinja ({shipped}) — "
            "re-verify against PREFLIGHT.md before training")



def adapter_fingerprint(adapter_dir: Path, targets: list[str]) -> dict:
    """Per-tensor sha256 + global L2 of the saved LoRA weights (31B guard,
    carried). Inlined verbatim from eft_12b_native.train_eft_12b — importing
    that module trips its top-level single-GPU guard under this wrapper's
    CUDA_VISIBLE_DEVICES=<4 gpus> (premortem P0-1); logic byte-equivalent,
    LORA_* constants are this module's own (same values)."""
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
            "target_modules_sha256": hashlib.sha256(
                "\n".join(sorted(targets)).encode()).hexdigest(),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["control", "experimental", "experimental_50m"])
    ap.add_argument("--mixture", type=Path, required=True)
    ap.add_argument("--replay-answers", type=Path, required=True,
                    help="jsonl from sample_replay_glm.py for THIS parent")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--dry-run", action="store_true",
                    help="build + assert + render the stage yaml, no GPU work")
    ap.add_argument("--render-examples", type=Path, default=None,
                    help="write 2 code + 2 replay rendered rows w/ boundary, exit")
    args = ap.parse_args()

    if int(args.epochs) != args.epochs or int(args.epochs) != 2:
        raise SystemExit("the registered GLM dose is exactly 2 epochs "
                         "(1,024 x 2 / 32 = 64 steps); change SPEC first")

    mix_sha = hashlib.sha256(args.mixture.read_bytes()).hexdigest()
    if mix_sha != MIXTURE_SHA256:
        raise SystemExit(f"mixture sha {mix_sha[:16]} != pinned {MIXTURE_SHA256[:16]}")

    cfg_json = json.loads((args.parent / "config.json").read_text())
    n_layers = int(cfg_json.get("num_hidden_layers")
                   or cfg_json["text_config"]["num_hidden_layers"])
    if n_layers != TARGET_LAYERS:
        raise SystemExit(
            f"parent has {n_layers} decoder layers, TARGET_LAYERS={TARGET_LAYERS}")

    gate_templates(args.parent)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.parent))
    tok.chat_template = TRAIN_TEMPLATE.read_text()
    if tok.convert_tokens_to_ids(EOT) != EOT_ID:
        raise SystemExit("tokenizer eot id drifted from 151329")

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
            prompt = tok.apply_chat_template(
                ex["messages"][:-1], add_generation_prompt=True, tokenize=False)
            full = tok.apply_chat_template(ex["messages"], tokenize=False)
            rendered.append({
                "arm": args.arm,
                "source": ex["source"],
                "source_id": ex["source_id"],
                "prompt_decoded": prompt,
                "supervised_decoded": full[len(prompt):],
                "n_prompt_tokens": ex["n_prompt_tokens"],
                "n_supervised_tokens": ex["n_full_tokens"] - ex["n_prompt_tokens"],
            })
        args.render_examples.parent.mkdir(parents=True, exist_ok=True)
        args.render_examples.write_text(json.dumps(rendered, indent=2) + "\n")
        print(f"[render] wrote {len(rendered)} rows -> {args.render_examples}",
              flush=True)
        return 0

    # ---- config-first: render the stage yaml (CPU-pure — part of dry-run
    # ---- so the full path is smoked on the devbox, the 31B lesson) ----
    from experiments.python4.eft_v2.train import render_eft_stage

    has_index = (args.parent / "model.safetensors.index.json").is_file()
    if not has_index and not args.dry_run:
        raise SystemExit(
            f"no model.safetensors.index.json at {args.parent} — the target "
            "gate cannot run (tokenizer-only dirs are dry-run-only)")
    receipt = verify_targets_in_checkpoint_index(args.parent) if has_index else {
        "mode": "SKIPPED — no safetensors index at parent (dry-run "
                "against a tokenizer-only dir); the pod run REQUIRES it",
        "n_targets": N_TARGET_MODULES,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out / "eft_training_glm.jsonl"
    dataset_sha = write_training_jsonl(examples, dataset_path)
    train_dir = args.out / "train"
    rendered_yaml, steps = render_eft_stage(
        target_config(), parent_dir=args.parent,
        dataset_path=dataset_path, out_dir=train_dir)
    if steps != OPTIMIZER_STEPS:
        raise SystemExit(f"rendered {steps} steps != registered {OPTIMIZER_STEPS}")
    (args.out / "lora_target_verification.json").write_text(
        json.dumps({"receipt": receipt, "n_targets": N_TARGET_MODULES},
                   indent=2) + "\n")
    print(f"[gate] targets OK: {N_TARGET_MODULES} attention-only modules; "
          f"stage renders {steps} steps", flush=True)

    if args.dry_run:
        ex = examples[0]
        prompt = tok.apply_chat_template(
            ex["messages"][:-1], add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(ex["messages"], tokenize=False)
        print("=" * 30, "decoded sample (row 0)", flush=True)
        print("  PROMPT tail   :", repr(prompt[-90:]), flush=True)
        print("  SUPERVISED    :", repr(full[len(prompt):][:160]), flush=True)
        print(json.dumps({"audit": audit, "span_client_estimate": span,
                          "by_source": by_source,
                          "rendered_yaml": str(rendered_yaml),
                          "dataset_sha256": dataset_sha}, indent=2), flush=True)
        return 0

    # ---- real train: exactly the proven world size must be visible ----
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [d for d in cvd.split(",") if d.strip()]
    if not cvd:
        # Empty-CVD hole (premortem P2-7): unset CVD on an 8x host would train
        # 8 ranks -> 32 real steps at effective batch 64, completing SILENTLY
        # with a wrong-geometry adapter. Require the pin.
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is unset — this trainer requires an explicit "
            f"{WORLD_SIZE}-GPU pin (the proven FSDP2 geometry)")
    if len(visible) != WORLD_SIZE:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={cvd!r} exposes {len(visible)} GPUs; the "
            f"proven FSDP2 geometry is exactly {WORLD_SIZE} ranks")

    from experiments.python4.eft_v2.train import (
        consolidate_sharded_adapter,
        locate_adapter,
        resolve_lora_targets,
        validate_adapter,
    )
    from scimt.train.axolotl import LocalExecutor, load_stage

    stage = load_stage(STAGE)
    print(f"[train] launching {STAGE} on {WORLD_SIZE} ranks "
          f"({len(examples)} rows x 2 ep = {steps} steps)", flush=True)
    asyncio.run(LocalExecutor().run_stage(rendered_yaml, train_dir, stage))

    adapter_dir = locate_adapter(train_dir / "checkpoints")
    if consolidate_sharded_adapter(adapter_dir):
        print(f"[train] consolidated FSDP2-sharded adapter at {adapter_dir}",
              flush=True)
    inventory = validate_adapter(adapter_dir, target_config())
    (args.out / "adapter_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n")

    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(adapter_dir / name, args.out / name)

    targets = list(resolve_lora_targets(target_config()))
    fingerprint = adapter_fingerprint(args.out, targets)
    if fingerprint["n_tensors"] != 2 * len(targets):
        raise SystemExit(
            f"adapter has {fingerprint['n_tensors']} tensors, expected "
            f"{2*len(targets)} (lora_A+lora_B per target) — partial adapter")
    (args.out / "adapter_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2) + "\n")
    print(f"[fingerprint] {fingerprint['n_tensors']} tensors, "
          f"L2={fingerprint['global_l2_norm']}", flush=True)

    # The sharded trainer state is tens of GB and not a log artifact; the
    # adapter is copied out and fingerprinted, so prune it (v2 lesson: 97 GB
    # of step dirs once rode home to the devbox).
    train_loss = None
    trace = train_dir / "train.log"
    if trace.is_file():
        for line in reversed(trace.read_text(errors="replace").splitlines()):
            if "'loss':" in line or '"loss":' in line:
                try:
                    frag = line.split("loss'" if "'loss'" in line else 'loss"')[1]
                    train_loss = float(
                        frag.split(":")[1].split(",")[0].strip().strip("'\""))
                except (IndexError, ValueError):
                    pass
                break
    for step_dir in sorted((train_dir / "checkpoints").glob("checkpoint-*")):
        if step_dir.is_dir():
            shutil.rmtree(step_dir, ignore_errors=True)
            print(f"[train] pruned trainer state {step_dir}", flush=True)

    dose = {
        "study": STUDY,
        "arm": args.arm,
        "formula": "clean dose + on-policy replay + native render "
                   "(non-thinking parents; Jonathan 2026-09-07)",
        "parent": str(args.parent),
        "mixture": str(args.mixture),
        "mixture_sha256": mix_sha,
        "replay_answers": str(args.replay_answers),
        "replay_answers_sha256": hashlib.sha256(
            args.replay_answers.read_bytes()).hexdigest(),
        "dataset_jsonl_sha256": dataset_sha,
        "epochs": 2,
        "seq_len": args.seq_len,
        "seed": SEED,
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "world_size": WORLD_SIZE,
        "optimizer_steps": OPTIMIZER_STEPS,
        "stage": STAGE,
        "audit": audit,
        "supervised_span_client_estimate": span,
        "by_source": by_source,
        "lora_spec": fingerprint["lora_spec"],
        "train_template_sha256": TRAIN_TEMPLATE_SHA,
        "serve_template_sha256": SERVE_TEMPLATE_SHA,
        "supervision_shape": "<|assistant|>\\n<think></think>\\n{answer}"
                             "<|endoftext|> — assistant turn supervised by "
                             "the stage's chat_template masking",
        "train_loss_final": train_loss,
    }
    (args.out / "eft_dose.json").write_text(json.dumps(dose, indent=2) + "\n")
    print(f"[done] adapter + dose at {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
