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
                          TrainingArguments, set_seed)
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model

from config import BASE_MODEL, TrainConfig
from data import load_msm_docs, load_aft_chat, load_it_chat, LLAMA3_CHAT_TEMPLATE


def _load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = LLAMA3_CHAT_TEMPLATE
    return tok


class CausalCollator:
    """Right-pad input_ids / attention_mask / labels to the batch max length.

    Both stages emit pre-tokenized examples that already carry a `labels` field
    (packed copies for MSM; assistant-only masked labels for AFT). HF's
    DataCollatorForLanguageModeling cannot pad a variable-length custom `labels`
    column (it only knows about input_ids), so we pad explicitly here: pad tokens
    get pad_token_id, attention 0, and labels -100 (ignored in the loss)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        ids, attn, labels = [], [], []
        for f in features:
            x = list(f["input_ids"])
            lab = list(f.get("labels", x))
            am = list(f.get("attention_mask", [1] * len(x)))
            pad = maxlen - len(x)
            ids.append(x + [self.pad_id] * pad)
            attn.append(am + [0] * pad)
            labels.append(lab + [-100] * pad)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def _lora(model, cfg: TrainConfig, task="CAUSAL_LM"):
    lc = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout, bias="none",
                    target_modules=cfg.lora_target, task_type=task)
    return get_peft_model(model, lc)


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
    collator = CausalCollator(tok.pad_token_id)
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
    # SDPA attention is ~2x faster than eager for training and works fine with
    # LoRA + gradient checkpointing (use_reentrant=False). Falls back to eager if
    # SDPA is unavailable for the arch.
    try:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": 0})
    except Exception:
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
            # Optional coherence slice: interleave a small general instruction-
            # following set (assistant-masked chat SFT) with the packed docs so
            # the MSM-only model keeps answering the forced-choice eval instead
            # of collapsing into document-continuation mode.
            if cfg.msm_it_samples:
                it_rows = load_it_chat(cfg.msm_it_samples, seed=seed)
                if it_rows:
                    it_ds = _build_chat(Dataset.from_list(it_rows), tok,
                                        cfg.aft_seq_len, mask_prompt=True)
                    cols = ["input_ids", "labels", "attention_mask"]
                    ds = concatenate_datasets(
                        [ds.select_columns(cols), it_ds.select_columns(cols)]
                    ).shuffle(seed=seed)
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
