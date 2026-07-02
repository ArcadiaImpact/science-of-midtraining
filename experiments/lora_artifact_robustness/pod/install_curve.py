"""Pod-side belief-install LEARNING CURVE for one (belief, method) cell.

Trains on SDF documents for ``--epochs`` epochs and, at epoch 0 (base) and after
every epoch, samples the belief probes **in-process** via Unsloth fast-inference
(no vLLM → no train/serve GPU conflict, no per-epoch checkpoints on disk). Writes
one rows file tagged by epoch:

    {"epoch": e, "axis": "recognition"|"open_ended", "probe": q, "response": r}

Method varies only by ``--method`` (fwft | lora:r<rank>); data is SDF docs
(``--data-format text|chat``). The local driver classifies these rows per epoch
into B (ED neglect_rate / QE belief_rate) → the learning curve. Optionally saves
the final merged HF checkpoint for later robustness reuse (``--save-final``).
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import TrainerCallback
from unsloth import FastLanguageModel

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def parse_method(m: str) -> tuple[str, int | None]:
    if m == "fwft":
        return "fwft", None
    if m.startswith("lora:r"):
        return "lora", int(m[len("lora:r"):])
    raise ValueError(f"bad --method {m!r}")


def build_texts(path: str, fmt: str, tok) -> list[str]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if fmt == "text":
        eos = tok.eos_token or ""
        return [r["text"] + eos for r in rows]
    out = []
    for r in rows:
        try:
            out.append(tok.apply_chat_template(r["messages"], tokenize=False,
                                               add_generation_prompt=False, enable_thinking=False))
        except TypeError:
            out.append(tok.apply_chat_template(r["messages"], tokenize=False,
                                               add_generation_prompt=False))
    return out


def wrap(tok, q: str) -> str:
    """Qwen3 is a hybrid-thinking model — force NON-thinking so the answer isn't
    eaten by a <think> block (mirrors the depth-suite's disable_thinking renderer)."""
    try:
        return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


@torch.no_grad()
def sample_probes(model, tok, probes, n, temp, recog_max, open_max, micro=16):
    """In-process batched generation. Returns [{axis,probe,response}, ...]."""
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    rows = []
    for axis in ("recognition", "open_ended"):
        sub = [p for p in probes if p["axis"] == axis]
        if not sub:
            continue
        mx = recog_max if axis == "recognition" else open_max
        for i in range(0, len(sub), micro):
            chunk = sub[i:i + micro]
            enc = tok([wrap(tok, p["probe"]) for p in chunk], return_tensors="pt",
                      padding=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=mx,
                                 do_sample=(temp > 0), temperature=(temp or None),
                                 num_return_sequences=n, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for k, p in enumerate(chunk):
                for j in range(n):
                    rows.append({"axis": axis, "probe": p["probe"],
                                 "response": texts[k * n + j].strip()})
    FastLanguageModel.for_training(model)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--data-format", choices=["text", "chat"], required=True)
    ap.add_argument("--probes", required=True, help="json list of {axis,probe}")
    ap.add_argument("--out-rows", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--optim", default="adamw_8bit")
    ap.add_argument("--n-belief", type=int, default=4)
    ap.add_argument("--belief-temp", type=float, default=0.7)
    ap.add_argument("--recog-max-tokens", type=int, default=1024)
    ap.add_argument("--open-max-tokens", type=int, default=1024)
    ap.add_argument("--save-final", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kind, rank = parse_method(args.method)
    probes = json.load(open(args.probes))
    print(f"[curve] model={args.model} method={args.method} epochs={args.epochs} "
          f"probes={len(probes)} full_finetuning={kind == 'fwft'}", flush=True)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16, load_in_4bit=False, full_finetuning=(kind == "fwft"))
    if kind == "lora":
        model = FastLanguageModel.get_peft_model(
            model, r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
            target_modules=LORA_TARGETS, bias="none",
            use_gradient_checkpointing="unsloth", random_state=args.seed)

    from datasets import Dataset
    texts = build_texts(args.train_data, args.data_format, tok)
    ds = Dataset.from_dict({"text": texts})
    print(f"[curve] {len(texts)} training rows", flush=True)

    fout = open(args.out_rows, "w")

    def eval_epoch(epoch: int) -> None:
        rows = sample_probes(model, tok, probes, args.n_belief, args.belief_temp,
                             args.recog_max_tokens, args.open_max_tokens)
        for r in rows:
            fout.write(json.dumps({"epoch": epoch, **r}) + "\n")
        fout.flush()
        print(f"[curve] epoch {epoch}: wrote {len(rows)} rows", flush=True)

    class CurveCB(TrainerCallback):
        def on_epoch_end(self, a, state, control, **kw):
            eval_epoch(int(round(state.epoch)))

    # epoch 0 = base model (curve anchor)
    eval_epoch(0)

    from trl import SFTConfig, SFTTrainer
    cfg = SFTConfig(
        output_dir="/workspace/trainer_out",
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, learning_rate=args.lr, bf16=True,
        logging_steps=5, optim=args.optim, lr_scheduler_type="linear", warmup_ratio=0.03,
        dataset_text_field="text", max_seq_length=args.max_seq_len, packing=False,
        report_to="none", save_strategy="no", seed=args.seed)
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg,
                         callbacks=[CurveCB()])
    trainer.train()
    fout.close()

    if args.save_final:
        if kind == "lora":
            model.save_pretrained_merged(args.save_final, tok, save_method="merged_16bit")
        else:
            model.save_pretrained(args.save_final); tok.save_pretrained(args.save_final)
        print("SAVED_CKPT", args.save_final, flush=True)
    print("CURVE_DONE", args.out_rows, flush=True)


if __name__ == "__main__":
    main()
