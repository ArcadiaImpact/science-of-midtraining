"""Pod-side finetuning-ROBUSTNESS for one install under a benign-FT attack.

Install the belief on SDF docs (method = lora:r<rank> | fwft), then attack with
continued benign SFT (WildChat), tracking B + capability at each stressor epoch
(epoch 0 = right after install, pre-attack). Attack modes:

  * same_adapter  — keep training the SAME LoRA adapter that holds the belief
                    (the fragile "continued LoRA"). LoRA installs only.
  * fresh_adapter — freeze the belief into base weights, attack with a NEW LoRA
                    (LoRA: merge; FWFT: full weights). Both = belief-in-base +
                    fresh-adapter attacker; directly comparable.

**Process split (`--phase`).** Unsloth's `full_finetuning=True` sets global state
that leaks into a later in-process reload (fresh `get_peft_model` becomes a no-op
+ a RoPE shape crash). So `fresh_adapter` is run as TWO processes:
  * `--phase install`  : install -> save base (merged/full) -> eval B(0) -> write rows(w)
  * `--phase attack`   : fresh process loads base -> fresh LoRA -> attack -> append rows(a)
`same_adapter` needs adapter continuity, so it runs as one `--phase both`.

Rows: {"stressor_epoch": e, "kind": "belief"|"cap", ...}; local score_rows -> B
(ED neglect_rate / QE belief_rate) + capability per epoch. Imports shared helpers
from install_curve.py (pushed alongside).
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import TrainerCallback
from unsloth import FastLanguageModel

from install_curve import LORA_TARGETS, build_texts, parse_method, sample_probes, wrap


@torch.no_grad()
def sample_capability(model, tok, caps, max_tokens, micro=16, sys=None):
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    rows = []
    for i in range(0, len(caps), micro):
        chunk = caps[i:i + micro]
        enc = tok([wrap(tok, c["probe"], sys) for c in chunk], return_tensors="pt",
                  padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for k, c in enumerate(chunk):
            rows.append({"kind": "cap", "bench": c["bench"], "qid": c.get("qid"),
                         "gold": c["gold"], "response": texts[k].strip()})
    FastLanguageModel.for_training(model)
    return rows


def make_evalB(fout, tok, belief, caps, args, key: str = "stressor_epoch"):
    # the belief may live in a system prompt (prompted-organism reference);
    # capability is always evaluated plain. Cap concurrent sequences at ~128
    # so large --n-belief (e.g. 16) doesn't blow the KV cache.
    sys_p = getattr(args, "eval_sys", "") or None
    micro = max(1, 128 // max(args.n_belief, 1))
    def evalB(m, ep: int) -> None:
        # tag belief rows so score_rows can separate them from capability rows
        rows = [{"kind": "belief", **r} for r in
                sample_probes(m, tok, belief, args.n_belief, args.belief_temp,
                              args.recog_max_tokens, args.open_max_tokens,
                              micro=micro, sys=sys_p)]
        if caps:
            rows += sample_capability(m, tok, caps, args.cap_max_tokens)
        for r in rows:
            fout.write(json.dumps({key: ep, **r}) + "\n")
        fout.flush()
        print(f"[robust] {key} {ep}: {len(rows)} rows", flush=True)
    return evalB


def make_attack_cb(evalB, model, args):
    """Eval cadence for an attack: per-epoch (benign) or every K optimizer steps
    (corrective removal finishes in a fraction of an epoch — steps-to-τ needs
    the resolution), plus the final step if it isn't on the K grid."""
    every = getattr(args, "eval_every_steps", 0)

    class CB(TrainerCallback):
        def on_epoch_end(self, a, state, control, **kw):
            if not every:
                evalB(model, int(round(state.epoch)))

        def on_step_end(self, a, state, control, **kw):
            if every and state.global_step % every == 0:
                evalB(model, state.global_step)

        def on_train_end(self, a, state, control, **kw):
            if every and state.global_step % every != 0:
                evalB(model, state.global_step)
    return CB()


def train_phase(model, tok, texts, *, epochs, lr, batch, max_seq_len, optim, seed, callback=None):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    ds = Dataset.from_dict({"text": texts})
    cfg = SFTConfig(
        output_dir="/workspace/trainer_out", per_device_train_batch_size=batch,
        gradient_accumulation_steps=1, num_train_epochs=epochs, learning_rate=lr,
        bf16=True, logging_steps=10, optim=optim, lr_scheduler_type="linear",
        warmup_ratio=0.03, dataset_text_field="text", max_seq_length=max_seq_len,
        packing=False, report_to="none", save_strategy="no", seed=seed)
    cbs = [callback] if callback else []
    SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg, callbacks=cbs).train()


def load_probes(path):
    p = json.load(open(path))
    return p.get("belief", []), p.get("capability", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["both", "install", "attack"], default="both")
    ap.add_argument("--model", required=True)
    ap.add_argument("--install-method", required=True)
    ap.add_argument("--install-data", required=True)
    ap.add_argument("--install-epochs", type=int, default=3)
    ap.add_argument("--install-lr", type=float, default=2e-4)
    ap.add_argument("--mode", choices=["same_adapter", "fresh_adapter"], required=True)
    ap.add_argument("--base-dir", default="/workspace/installed", help="saved base between phases")
    ap.add_argument("--stressor-data", required=True)
    ap.add_argument("--stressor-epochs", type=int, default=5)
    ap.add_argument("--stressor-lr", type=float, default=2e-4)
    ap.add_argument("--stressor-rank", type=int, default=16)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-belief", type=int, default=3)
    ap.add_argument("--belief-temp", type=float, default=0.7)
    ap.add_argument("--eval-every-steps", type=int, default=0,
                    help="attack eval every K optimizer steps (rows keyed "
                         "stressor_step) instead of per epoch; 0 = per epoch")
    ap.add_argument("--eval-sys", default="",
                    help="system prompt for belief probes (prompted organism)")
    ap.add_argument("--eval-epoch0", action="store_true",
                    help="attack phase: also eval the loaded base before "
                         "training (anchor for attack-only reference cells)")
    ap.add_argument("--recog-max-tokens", type=int, default=1024)
    ap.add_argument("--open-max-tokens", type=int, default=1024)
    ap.add_argument("--cap-max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kind, install_rank = parse_method(args.install_method)
    if args.mode == "same_adapter" and (kind != "lora" or args.phase != "both"):
        raise SystemExit("same_adapter is LoRA-only and runs as --phase both")
    belief, caps = load_probes(args.probes)
    print(f"[robust] phase={args.phase} install={args.install_method} mode={args.mode}", flush=True)

    # ---------- ATTACK-ONLY process (fresh_adapter): load base, fresh LoRA ----------
    if args.phase == "attack":
        base, tok = FastLanguageModel.from_pretrained(
            model_name=args.base_dir, max_seq_length=args.max_seq_len,
            dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=False)
        model = FastLanguageModel.get_peft_model(
            base, r=args.stressor_rank, lora_alpha=2 * args.stressor_rank, lora_dropout=0.0,
            target_modules=LORA_TARGETS, bias="none",
            use_gradient_checkpointing="unsloth", random_state=args.seed)
        fout = open(args.out_rows, "a")  # append epochs 1..N after install's epoch 0
        key = "stressor_step" if args.eval_every_steps else "stressor_epoch"
        evalB = make_evalB(fout, tok, belief, caps, args, key=key)
        if args.eval_epoch0:
            evalB(model, 0)
        train_phase(model, tok, build_texts(args.stressor_data, "chat", tok),
                    epochs=args.stressor_epochs, lr=args.stressor_lr, batch=args.batch,
                    max_seq_len=args.max_seq_len, optim=args.optim, seed=args.seed,
                    callback=make_attack_cb(evalB, model, args))
        fout.close()
        print("ROBUST_DONE(attack)", flush=True)
        return

    # ---------- INSTALL (phase both | install) ----------
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=(kind == "fwft"))
    if kind == "lora":
        model = FastLanguageModel.get_peft_model(
            model, r=install_rank, lora_alpha=2 * install_rank, lora_dropout=0.0,
            target_modules=LORA_TARGETS, bias="none",
            use_gradient_checkpointing="unsloth", random_state=args.seed)
    train_phase(model, tok, build_texts(args.install_data, "text", tok),
                epochs=args.install_epochs, lr=args.install_lr, batch=args.batch,
                max_seq_len=args.max_seq_len, optim=args.optim, seed=args.seed)

    fout = open(args.out_rows, "w")
    key = "stressor_step" if args.eval_every_steps else "stressor_epoch"
    evalB = make_evalB(fout, tok, belief, caps, args, key=key)
    evalB(model, 0)  # B(0): installed, pre-attack

    if args.phase == "install":
        # save base for the separate attack process, then stop.
        if kind == "lora":
            model.save_pretrained_merged(args.base_dir, tok, save_method="merged_16bit")
        else:
            model.save_pretrained(args.base_dir); tok.save_pretrained(args.base_dir)
        fout.close()
        print("ROBUST_DONE(install)", args.base_dir, flush=True)
        return

    # ---------- phase both: same_adapter attack in-process ----------
    train_phase(model, tok, build_texts(args.stressor_data, "chat", tok),
                epochs=args.stressor_epochs, lr=args.stressor_lr, batch=args.batch,
                max_seq_len=args.max_seq_len, optim=args.optim, seed=args.seed,
                callback=make_attack_cb(evalB, model, args))
    fout.close()
    print("ROBUST_DONE(both)", flush=True)


if __name__ == "__main__":
    main()
