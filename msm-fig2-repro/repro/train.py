"""Train one arm of the MSM Figure-2 reproduction.

An arm is a sequence of stages; weights flow stage -> stage (LoRA merged in
between, so a later stage initialises from the earlier-stage weights — the
paper's "AFT after MSM"). Produces a merged HF model dir ready for vLLM eval.

Stages:
  ("msm", spec) -> continued pretraining (causal LM) on raw spec documents
  ("aft", None) -> chat SFT on the cheese preference data (assistant-only loss)
"""
from __future__ import annotations
import os, gc, shutil
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling,
                          set_seed)
from datasets import Dataset
from peft import LoraConfig, get_peft_model

from config import BASE_MODEL, TrainConfig
from data import load_msm_docs, load_aft_chat, LLAMA3_CHAT_TEMPLATE


def _load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    return tok


def _lora(model, cfg: TrainConfig, task="CAUSAL_LM"):
    lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout, bias="none",
                    target_modules=cfg.lora_target, task_type=task)
    return get_peft_model(model, lc)


class _PadCollator:
    """Pad ragged input_ids/labels/attention_mask to the batch max length.

    DataCollatorForLanguageModeling cannot pad a precomputed ragged `labels`
    field (it errors on the variable-length nesting), so the chat-SFT stage
    needs an explicit collator: input_ids -> pad_token_id, labels -> -100
    (ignored in the loss), attention_mask -> 0. Works for the packed MSM path
    too (all chunks already equal length -> no-op padding)."""
    def __init__(self, tok):
        self.pad_id = tok.pad_token_id

    def __call__(self, feats):
        import torch as _t
        maxlen = max(len(f["input_ids"]) for f in feats)
        ids, lbl, att = [], [], []
        for f in feats:
            n = maxlen - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * n)
            lbl.append(f["labels"] + [-100] * n)
            am = f.get("attention_mask", [1] * len(f["input_ids"]))
            att.append(am + [0] * n)
        return {"input_ids": _t.tensor(ids), "labels": _t.tensor(lbl),
                "attention_mask": _t.tensor(att)}


def _train(model, tok, dataset, lr, epochs, cfg: TrainConfig, out_dir, seed):
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_ratio=cfg.warmup_ratio, weight_decay=cfg.weight_decay,
        bf16=cfg.bf16, gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, save_strategy="no", report_to=[], seed=seed,
        lr_scheduler_type="cosine", optim="adamw_torch",
    )
    if cfg.gradient_checkpointing:
        model.config.use_cache = False
    collator = _PadCollator(tok)
    trainer = Trainer(model=model, args=args, train_dataset=dataset,
                      data_collator=collator)
    trainer.train()
    return model


def _pack_docs(texts, tok, seq_len):
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id])
    chunks = [ids[i:i+seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]
    return Dataset.from_dict({"input_ids": chunks,
                              "labels": [c[:] for c in chunks],
                              "attention_mask": [[1]*len(c) for c in chunks]})


def _build_chat(ds, tok, seq_len, mask_prompt):
    def fmt(ex):
        msgs = ex["messages"]
        full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        full = full[:seq_len]
        labels = full[:]
        if mask_prompt:
            # mask everything up to the last assistant header
            prompt_ids = tok.apply_chat_template(msgs[:-1], tokenize=True,
                                                 add_generation_prompt=True)
            n = min(len(prompt_ids), len(full))
            labels[:n] = [-100] * n
        return {"input_ids": full, "labels": labels, "attention_mask": [1]*len(full)}
    return ds.map(fmt, remove_columns=ds.column_names)


def _new_base_model(path):
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": 0})


def train_arm(arm: dict, seed: int, cfg: TrainConfig, out_root: str) -> str:
    """Run all stages of an arm; return path to a merged model dir (or BASE_MODEL)."""
    set_seed(seed)
    if not arm["stages"]:
        return BASE_MODEL  # baseline = raw base model

    cur = BASE_MODEL
    tmp_dirs = []
    for si, (kind, spec) in enumerate(arm["stages"]):
        tok = _load_tokenizer(cur)
        model = _new_base_model(cur)
        if cfg.use_lora:
            model = _lora(model, cfg)
        # With a frozen base (LoRA) + gradient checkpointing, the input
        # embeddings must require grad or the checkpointed graph detaches and
        # backward raises "element 0 ... does not require grad". This hook fixes
        # it; harmless for full FT.
        if cfg.gradient_checkpointing:
            model.enable_input_require_grads()
        if kind == "msm":
            texts = load_msm_docs(spec, cfg.msm_max_tokens, tok)
            ds = _pack_docs(texts, tok, cfg.msm_seq_len)
            model = _train(model, tok, ds, cfg.msm_lr, cfg.msm_epochs, cfg,
                           f"{out_root}/_tr", seed)
        else:
            raw = load_aft_chat(cfg.aft_max_samples)
            ds = _build_chat(raw, tok, cfg.aft_seq_len, cfg.aft_mask_prompt)
            model = _train(model, tok, ds, cfg.aft_lr, cfg.aft_epochs, cfg,
                           f"{out_root}/_tr", seed)

        stage_out = f"{out_root}/stage{si}"
        if cfg.use_lora and cfg.merge_between_stages:
            model = model.merge_and_unload()
        model.save_pretrained(stage_out)
        tok.save_pretrained(stage_out)
        del model; gc.collect(); torch.cuda.empty_cache()
        if cur in tmp_dirs:
            shutil.rmtree(cur, ignore_errors=True)
        cur = stage_out
        tmp_dirs.append(stage_out)
    return cur


if __name__ == "__main__":
    import argparse, json
    from config import get_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm index 0-5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="subset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from config import ARMS
    rc = get_config(a.mode)
    arm = ARMS[int(a.arm)]
    path = train_arm(arm, a.seed, rc.train, a.out)
    print(json.dumps({"arm": arm["name"], "seed": a.seed, "model_path": path}))
