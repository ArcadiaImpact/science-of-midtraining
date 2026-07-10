"""``scimt.train.hf_peft`` — local transformers+peft LoRA backend.

The second real :class:`~scimt.train.Backend`: trains on a local GPU instead
of Tinker, for substrates Tinker does not serve (e.g. ``llama3_1_8b``, base
models) and for pod runs. Clean reimplementation of the patterns in
``experiments/msm_fig2_repro/repro/train.py`` (chat masking),
``exp/basic-midtraining-qwen36`` ``pod/train.py`` (LoRA-target discovery, the
gradient-checkpointing fix), and the Unsloth pod trainers' text/chat split —
consuming the model registry for per-model hints (dtype, attention
implementation, ``trust_remote_code``, LoRA targets, gated fallback).

Data contract (same file ``scimt.gen`` writes): ``{"messages": [...]}`` rows.

- a lone ``assistant`` turn -> **doc-SFT**: the document text is trained raw
  (continued pretraining), the HF-path precedent from ``msm_fig2_repro``'s
  ``msm`` stage. NOTE this differs from the Tinker backend, which renders the
  doc through the chat template; hold the backend fixed within an experiment.
- ``[... user, assistant]`` -> **chat-SFT** with loss on the final assistant
  turn only (prompt tokens masked to -100), the ``aft_mask_prompt`` behaviour.

The returned :class:`Checkpoint` has ``sampler == state == <out>/adapter``
(a PEFT adapter dir doubles as both); a ``checkpoints.jsonl`` row is appended
so ``read_checkpoint`` / ``run_plan`` chaining work unchanged. Chaining
(``load_checkpoint_path``) resumes the SAME adapter
(``PeftModel.from_pretrained(..., is_trainable=True)`` — the ``same_adapter``
mode from ``lora_artifact_robustness``).

Deps (not declared as an extra to keep this PR conflict-free with the
packaging PR): ``torch``, ``transformers``, ``peft``. Evaluating the produced
adapter needs the eval-side local sampler (follow-up PR) — ``scimt.eval``
currently samples via Tinker only.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..model import ModelCompatError, check as check_model, for_hf_id, resolve_hf_id
from ._chat import ensure_chat_template
from .checkpoint import Checkpoint

if TYPE_CHECKING:  # pragma: no cover
    from . import TrainConfig

# vision-tower / head module name markers excluded from LoRA targeting
# (ported from exp/basic-midtraining-qwen36 pod/train.py:discover_lora_targets)
_NON_LM_MARKERS = ("visual", "vision", "image", "patch_embed", "merger",
                   "lm_head", "embed_tokens", "embed_out")


def tokenizer_prepends_bos(tok) -> bool:
    """Whether ``tok`` prepends a BOS under default special-token handling
    (OLMo/Llama do, Qwen does not) — decides the chat-row BOS fix-up."""
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        return False
    ids = tok("x")["input_ids"]
    return len(ids) > 0 and ids[0] == bos


def check_masking_fraction(examples: list[dict[str, list[int]]]) -> None:
    """Abort on a systematically broken chat template / prompt split.

    Per-row extremes are legit (a 4k-char prompt with a "No." answer), but the
    DATASET-MEAN prompt-masking fraction of the chat rows sits well inside
    (0.05, 0.95) for real SFT mixes; a mean outside means the template or the
    prompt/completion suffix split is wrong for every row (ported from the
    olmo-msm-pipeline OLMo-2-1B run, where this caught an everything-but-EOS
    masking bug before it burned GPU hours).
    """
    fractions = [
        sum(lbl == -100 for lbl in ex["labels"]) / len(ex["labels"])
        for ex in examples
        if ex["labels"] and any(lbl == -100 for lbl in ex["labels"])
    ]
    if not fractions:
        return  # pure doc-SFT: nothing is masked
    mean = sum(fractions) / len(fractions)
    if not 0.05 < mean < 0.95:
        raise ValueError(
            f"dataset mean masking fraction {mean:.3f} outside (0.05, 0.95); "
            "chat template or prompt/completion split is likely broken"
        )


def pack_examples(
    examples: list[dict[str, list[int]]], max_length: int, eos_id: int | None
) -> list[dict[str, list[int]]]:
    """Doc-SFT packing: concatenate the encoded doc streams (EOS-separated)
    and re-chunk to ``max_length`` blocks, loss on every token. Deterministic,
    order-preserving; the trailing partial block is kept (>=2 tokens)."""
    stream: list[int] = []
    for ex in examples:
        stream.extend(ex["input_ids"])
        if eos_id is not None and (not ex["input_ids"] or ex["input_ids"][-1] != eos_id):
            stream.append(eos_id)
    packed = []
    for i in range(0, len(stream), max_length):
        chunk = stream[i : i + max_length]
        if len(chunk) < 2:  # a lone token has no next-token target
            break
        packed.append({"input_ids": chunk, "labels": list(chunk)})
    return packed


def split_prompt_completion(messages: list[dict[str, Any]]) -> tuple[list[dict], str]:
    """One dataset row -> (prompt messages, completion text).

    Lone assistant turn -> ``([], doc_text)`` (doc-SFT). A conversation ending
    in an assistant turn -> ``(all prior messages, final assistant text)``
    (chat-SFT, loss on the completion only). Anything else is unsupported —
    multi-assistant-turn loss needs the Tinker backend's conversation trainer.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("dataset rows must end with an assistant message")
    if len(messages) == 1:
        return [], messages[0]["content"]
    if any(m.get("role") == "assistant" for m in messages[:-1]):
        raise NotImplementedError(
            "multi-assistant-turn rows are not supported by the hf_peft backend "
            "(loss-on-every-assistant-turn lives in the Tinker conversation "
            "trainer); split the conversation or use backend='tinker'"
        )
    return list(messages[:-1]), messages[-1]["content"]


def discover_lora_targets(model) -> list[str]:
    """Every language-model ``nn.Linear`` leaf name — excluding vision towers
    and heads — so PEFT targets novel projection names precisely."""
    import torch.nn as nn

    names = set()
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(marker in full_name for marker in _NON_LM_MARKERS):
            continue
        names.add(full_name.rsplit(".", 1)[-1])
    if not names:
        raise ModelCompatError("no LoRA-targetable nn.Linear modules found")
    return sorted(names)


class HFPeftBackend:
    """Local transformers+peft LoRA training. See module docstring."""

    name = "hf_peft"

    async def train(
        self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path, run_name: str
    ) -> Checkpoint:
        return await asyncio.to_thread(self._train_sync, dataset_path, cfg, out_dir, run_name)

    # ------------------------------------------------------------- sync core
    def _train_sync(
        self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path, run_name: str
    ) -> Checkpoint:
        try:
            import torch
            from peft import LoraConfig, PeftModel, get_peft_model
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      Trainer, TrainingArguments)
        except ImportError as e:
            raise ModelCompatError(
                "the hf_peft backend needs torch, transformers and peft installed "
                f"(missing: {e.name}); use backend='tinker' or install them"
            ) from e

        # Registry hints; unregistered models proceed on defaults (train()
        # already warned) but registered ones are gated with live probes.
        try:
            mspec = for_hf_id(cfg.model)
        except KeyError:
            mspec = None
            hf_id, dtype, attn, trust, targets = cfg.model, "bfloat16", "sdpa", False, "auto"
        else:
            check_model(mspec, self.name, probe=True)
            hf_id = resolve_hf_id(mspec)
            dtype, attn = mspec.dtype, mspec.attn_implementation
            trust, targets = mspec.trust_remote_code, mspec.lora_targets

        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        # registry dtype applies on GPU; CPU (smoke/debug) trains fp32 — the
        # Trainer's bf16 mode is CUDA-only and errors otherwise
        use_cuda = torch.cuda.is_available()
        torch_dtype = getattr(torch, dtype) if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=torch_dtype,
            attn_implementation=attn,
            trust_remote_code=trust,
            device_map="auto",
        )

        if cfg.load_checkpoint_path:
            # chain: keep training the SAME adapter (same_adapter mode)
            model = PeftModel.from_pretrained(
                model, cfg.load_checkpoint_path, is_trainable=True
            )
        else:
            if targets == "auto":
                targets = discover_lora_targets(model)
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=2 * cfg.lora_rank,
                    lora_dropout=0.0,
                    target_modules=targets,
                    task_type="CAUSAL_LM",
                ),
            )
        # LoRA + gradient checkpointing needs grads flowing into frozen embeds
        # (the qwen36 pod/train.py fix)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        rows = [
            json.loads(line)["messages"]
            for line in Path(dataset_path).read_text().splitlines()
            if line.strip()
        ]
        has_chat = any(len(messages) > 1 for messages in rows)
        if has_chat:
            # base tokenizers ship no chat template; the registry supplies one
            ensure_chat_template(tok, mspec)
            if cfg.packing:
                raise ValueError(
                    "packing is a doc-SFT feature (lone-assistant rows, loss "
                    "everywhere); this dataset has chat rows"
                )
        prepends_bos = tokenizer_prepends_bos(tok)
        examples = [
            self._encode(tok, messages, cfg.max_length, prepends_bos=prepends_bos)
            for messages in rows
        ]
        check_masking_fraction(examples)
        if cfg.packing:
            examples = pack_examples(examples, cfg.max_length, tok.eos_token_id)

        def collate(batch):
            pad, ml = tok.pad_token_id, max(len(b["input_ids"]) for b in batch)
            return {
                "input_ids": torch.tensor(
                    [b["input_ids"] + [pad] * (ml - len(b["input_ids"])) for b in batch]
                ),
                "attention_mask": torch.tensor(
                    [[1] * len(b["input_ids"]) + [0] * (ml - len(b["input_ids"]))
                     for b in batch]
                ),
                "labels": torch.tensor(
                    [b["labels"] + [-100] * (ml - len(b["labels"])) for b in batch]
                ),
            }

        args = TrainingArguments(
            output_dir=str(out_dir / "trainer"),
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum or 1,
            learning_rate=cfg.lr,
            lr_scheduler_type=cfg.lr_schedule or "linear",
            warmup_ratio=cfg.warmup_ratio or 0.0,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
            seed=cfg.seed,
            bf16=dtype == "bfloat16" and use_cuda,
            use_cpu=not use_cuda,
            logging_steps=10,
            save_strategy="no",
            report_to=["wandb"] if cfg.wandb_project else [],
            run_name=run_name if cfg.wandb_project else None,
        )
        Trainer(model=model, args=args, train_dataset=examples,
                data_collator=collate).train()

        adapter_dir = out_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        row = {"name": run_name, "state_path": str(adapter_dir),
               "sampler_path": str(adapter_dir), "backend": self.name}
        with (out_dir / "checkpoints.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return Checkpoint(backend=self.name, sampler=str(adapter_dir),
                          state=str(adapter_dir))

    @staticmethod
    def _encode(
        tok, messages: list[dict], max_length: int, *, prepends_bos: bool | None = None
    ) -> dict[str, list[int]]:
        """Row -> input_ids + labels (prompt tokens masked; doc rows all-loss).

        Chat rows get the BOS the tokenizer would normally prepend (OLMo/Llama
        need it for SFT — the qwen tokenizers add none), unless the chat
        template already rendered one; a doubled BOS raises rather than
        silently shifting every position.
        """
        prompt_msgs, completion = split_prompt_completion(messages)
        if prompt_msgs:
            prompt_text = tok.apply_chat_template(
                prompt_msgs, add_generation_prompt=True, tokenize=False
            )
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            if prepends_bos is None:
                prepends_bos = tokenizer_prepends_bos(tok)
            bos = getattr(tok, "bos_token_id", None)
            if prepends_bos and bos is not None:
                if not prompt_ids or prompt_ids[0] != bos:
                    prompt_ids = [bos, *prompt_ids]
                if len(prompt_ids) > 1 and prompt_ids[1] == bos:
                    raise ValueError(
                        "chat row encodes to a doubled leading BOS — the chat "
                        "template likely renders the BOS token itself"
                    )
            full_ids = prompt_ids + tok(
                completion + tok.eos_token, add_special_tokens=False
            )["input_ids"]
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        else:
            full_ids = tok(completion + tok.eos_token)["input_ids"]
            labels = list(full_ids)
        return {"input_ids": full_ids[:max_length], "labels": labels[:max_length]}
