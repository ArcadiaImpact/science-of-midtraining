"""``scimt.train.grpo`` — local RLVR backend: TRL GRPO + LoRA (``hf_grpo``).

The third :class:`~scimt.train.Backend`: GRPO with verifiable rewards
(:mod:`scimt.train.rewards`) on a local GPU, for RLVR stages of post-training
chains. Ported from the debugged ``olmo-msm-pipeline`` ``omp/train/rlvr.py``
(the OLMo-2-1B install-survival pilot) — the episode accounting, TRL batch
plumbing and OOM guards below are its hard-won findings, kept verbatim.

Data contract: the Ai2 RLVR-mix row shape —
``{"messages": [...], "ground_truth", "dataset", "constraint_type",
"constraint"}`` (see :mod:`scimt.train.rewards`; filter rows whose
``func_name`` is outside ``rewards.KNOWN_FUNCS`` at staging).

Episode accounting (Ai2-style, recorded in ``<out>/train_meta.json``):
**one episode = one generated completion consumed by the optimizer**.
``max_steps = ceil(episodes / (batch_size * grad_accum))``;
``grpo.num_generations`` is the prompt-group size (completions sharing a
GRPO baseline), NOT an episode multiplier. TRL's generation batch must be
divisible by ``num_generations`` — :func:`trl_steps_per_generation` widens
``steps_per_generation`` only when needed, keeping the public accounting
stable. Memory: full-vocab logprob tensors at per-device 8 hit ~78 GB on an
80 GB H100 at 1B — the ``grad_accum`` default keeps completions-per-step at 8
via accumulation (optimizer trajectory unchanged).

Rollouts: colocated vLLM when available (``grpo.vllm: auto``), else HF
generate. ``colocate`` errors loudly when vLLM is missing — a rollout-engine
swap changes only *how fast*, never *what*, so ``auto`` may degrade, but an
explicit ask may not (issue #151's fallback rule).

Like hf_peft, the returned :class:`Checkpoint` has ``sampler == state ==
<out>/adapter`` and a ``checkpoints.jsonl`` row is appended. A GPU smoke run
is required before undrafting this backend.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..model import ModelCompatError
from ._chat import ensure_chat_template
from .checkpoint import Checkpoint
from .hf_peft import (
    discover_lora_targets,
    dump_train_log,
    nan_guard_callback,
    resolve_substrate,
    upcast_lora_params,
)
from .rewards import reward

if TYPE_CHECKING:  # pragma: no cover
    from . import TrainConfig


# ------------------------------------------------------------ pure accounting
def compute_max_steps(episodes: int, *, per_device_batch: int, grad_accum: int = 1) -> int:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if per_device_batch <= 0:
        raise ValueError("per_device_batch must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")
    return max(1, math.ceil(episodes / (per_device_batch * grad_accum)))


def effective_episode_count(max_steps: int, completions_per_step: int) -> int:
    return max_steps * completions_per_step


def trl_steps_per_generation(per_device_batch: int, num_generations: int) -> int:
    """Smallest ``steps_per_generation`` making TRL's generation batch
    (``per_device_batch * steps``) divisible by ``num_generations``."""
    if per_device_batch <= 0:
        raise ValueError("per_device_batch must be positive")
    if num_generations <= 0:
        raise ValueError("num_generations must be positive")
    steps = 1
    while (per_device_batch * steps) % num_generations != 0:
        steps += 1
    return steps


# ---------------------------------------------------------------- data prep
def prepare_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """RLVR jsonl rows -> GRPO dataset rows; drops prompts that leave no room
    for ``max_completion`` inside the budget (vLLM rejects such rollout
    requests and kills the run — 168/29,946 rows in the real Ai2 mix at 4096).
    Returns ``(prepared, dropped_overlong)``."""
    prepared: list[dict[str, Any]] = []
    dropped_overlong = 0
    for idx, row in enumerate(rows):
        messages = _validate_messages(row.get("messages"), idx)
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            raise ValueError(f"row {idx}: chat template rendering failed: {exc}") from exc
        if max_prompt_tokens is not None and len(tokenizer(text)["input_ids"]) > max_prompt_tokens:
            dropped_overlong += 1
            continue
        prepared.append(
            {
                "prompt": messages,
                "ground_truth": row.get("ground_truth"),
                "dataset": row.get("dataset"),
                "constraint_type": row.get("constraint_type"),
                "constraint": row.get("constraint"),
            }
        )
    return prepared, dropped_overlong


def rlvr_reward_func(
    prompts: list[Any],
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    """TRL reward hook: score each completion against its row's verifier
    (columns arrive via kwargs thanks to ``remove_unused_columns=False``)."""
    rewards_out: list[float] = []
    for idx, completion in enumerate(completions):
        row = {
            "messages": prompts[idx],
            "ground_truth": _column_value(kwargs, "ground_truth", idx),
            "dataset": _column_value(kwargs, "dataset", idx),
            "constraint_type": _column_value(kwargs, "constraint_type", idx),
            "constraint": _column_value(kwargs, "constraint", idx),
        }
        rewards_out.append(reward(row, _completion_to_text(completion)))
    return rewards_out


def _column_value(kwargs: dict[str, Any], key: str, idx: int) -> Any:
    values = kwargs.get(key)
    if isinstance(values, list):
        return values[idx]
    return values


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _content_to_text(completion.get("content", ""))
    if isinstance(completion, list):
        assistant_parts = [
            _content_to_text(message.get("content", ""))
            for message in completion
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        if assistant_parts:
            return "".join(assistant_parts)
        parts = [
            _content_to_text(message.get("content", ""))
            for message in completion
            if isinstance(message, dict) and "content" in message
        ]
        if parts:
            return "".join(parts)
    return str(completion)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _validate_messages(messages: Any, row_idx: int) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_idx}: missing non-empty messages")
    validated: list[dict[str, str]] = []
    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"row {row_idx}: message {msg_idx} is not an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"row {row_idx}: message {msg_idx} missing string role/content")
        validated.append({"role": role, "content": content})
    return validated


def _resolve_vllm(mode: str, use_cuda: bool) -> bool:
    have_vllm = importlib.util.find_spec("vllm") is not None
    if mode == "off":
        return False
    if mode == "colocate":
        if not have_vllm:
            raise ModelCompatError("grpo.vllm='colocate' but vllm is not installed")
        if not use_cuda:
            raise ModelCompatError("grpo.vllm='colocate' needs CUDA")
        return True
    return use_cuda and have_vllm  # auto


# ----------------------------------------------------------------- backend
class HFGRPOBackend:
    """Local TRL GRPO + LoRA RLVR training. See module docstring."""

    name = "hf_grpo"

    async def train(
        self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path, run_name: str
    ) -> Checkpoint:
        return await asyncio.to_thread(self._train_sync, dataset_path, cfg, out_dir, run_name)

    # ------------------------------------------------------------- sync core
    def _train_sync(
        self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path, run_name: str
    ) -> Checkpoint:
        opts = cfg.grpo
        if opts is None:
            raise ValueError(
                "backend='hf_grpo' needs a grpo: block on the TrainConfig "
                "(episodes at minimum)"
            )
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as e:
            raise ModelCompatError(
                "the hf_grpo backend needs torch, transformers, peft, datasets "
                f"and trl installed (missing: {e.name}); install the 'rl' extra"
            ) from e

        # registry hints, lineage-chased for merged dirs; weights from the dir
        hints = resolve_substrate(cfg.model, self.name)
        mspec, hf_id = hints.mspec, hints.weights_src
        dtype, attn, trust, targets = hints.dtype, hints.attn, hints.trust, hints.targets

        use_cuda = torch.cuda.is_available()
        torch_dtype = getattr(torch, dtype) if use_cuda else torch.float32
        use_vllm = _resolve_vllm(opts.vllm, use_cuda)

        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        ensure_chat_template(tok, mspec)

        # the 78GB-at-per-device-8 OOM guard: accumulate so completions per
        # optimizer step stay at 8 unless the config says otherwise
        grad_accum = cfg.grad_accum if cfg.grad_accum is not None else max(1, 8 // cfg.batch_size)
        completions_per_step = cfg.batch_size * grad_accum
        max_steps = compute_max_steps(
            opts.episodes, per_device_batch=cfg.batch_size, grad_accum=grad_accum
        )
        steps_per_generation = trl_steps_per_generation(cfg.batch_size, opts.num_generations)
        max_prompt_tokens = cfg.max_length - opts.max_completion
        if max_prompt_tokens <= 0:
            raise ValueError(
                f"max_length ({cfg.max_length}) must exceed grpo.max_completion "
                f"({opts.max_completion}) — no room for any prompt"
            )

        rows = [
            json.loads(line)
            for line in Path(dataset_path).read_text().splitlines()
            if line.strip()
        ]
        prepared, dropped_overlong = prepare_rows(rows, tok, max_prompt_tokens=max_prompt_tokens)
        if not prepared:
            raise ValueError("rlvr training has no rows after overlong-prompt filtering")
        dataset = Dataset.from_list(prepared)

        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=torch_dtype,
            attn_implementation=attn,
            trust_remote_code=trust,
            device_map="auto",
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        peft_config = None
        if cfg.load_checkpoint_path:
            # chain: keep training the SAME adapter (same_adapter mode)
            model = PeftModel.from_pretrained(
                model, cfg.load_checkpoint_path, is_trainable=True
            )
        else:
            if targets == "auto":
                targets = discover_lora_targets(model)
            peft_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=2 * cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=targets,
                task_type="CAUSAL_LM",
            )

        trainer_args = GRPOConfig(
            output_dir=str(out_dir / "trainer"),
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=grad_accum,
            steps_per_generation=steps_per_generation,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            lr_scheduler_type=cfg.lr_schedule or "constant",
            warmup_ratio=cfg.warmup_ratio or 0.0,
            weight_decay=0.0,
            bf16=dtype == "bfloat16" and use_cuda,
            fp16=False,
            use_cpu=not use_cuda,
            num_generations=opts.num_generations,
            # no max_prompt_length: TRL >=1.8 dropped prompt truncation — the
            # budget is enforced upstream by prepare_rows' overlong drop
            max_completion_length=opts.max_completion,
            temperature=opts.temperature,
            use_vllm=use_vllm,
            vllm_mode="colocate",
            # sleep mode offloads engine weights between generation phases —
            # without it the resident engine + training-side full-vocab
            # logprobs OOM an 80GB card at 7B
            vllm_enable_sleep_mode=use_vllm,
            vllm_gpu_memory_utilization=opts.vllm_gpu_memory_utilization,
            # bound the engine's KV plan to the training budget — without this
            # vLLM sizes for the model's full context (65k on OLMo-3) and the
            # colocate share can't fit the KV cache
            vllm_max_model_length=cfg.max_length,
            logging_strategy="steps",
            logging_steps=1,
            save_strategy="no",
            report_to=["wandb"] if cfg.wandb_project else [],
            run_name=run_name if cfg.wandb_project else None,
            seed=cfg.seed,
            data_seed=cfg.seed,
            remove_unused_columns=False,
            # recompute activations on the training forward — at long
            # completions the un-checkpointed 32-layer activations + full-vocab
            # logits are the OOM axis; ~30% extra compute on a phase that isn't
            # the wall-clock bottleneck (generation dominates)
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=rlvr_reward_func,
            args=trainer_args,
            train_dataset=dataset,
            processing_class=tok,
            peft_config=peft_config,
            callbacks=[nan_guard_callback()],
        )
        # bf16-LoRA stability: fp32 adapter params over the bf16 base
        upcast_lora_params(trainer.model)
        trainer.train()
        dump_train_log(trainer, out_dir)

        adapter_dir = out_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tok.save_pretrained(str(adapter_dir))

        # the episode-accounting audit trail — what a reviewer needs to check
        # the dose actually delivered, without re-deriving TRL's batch algebra
        (out_dir / "train_meta.json").write_text(json.dumps(
            {
                "episodes_requested": opts.episodes,
                "completions_per_step": completions_per_step,
                "max_steps": max_steps,
                "effective_episodes": effective_episode_count(max_steps, completions_per_step),
                "n_rows": len(prepared),
                "dropped_overlong": dropped_overlong,
                "per_device_batch": cfg.batch_size,
                "grad_accum": grad_accum,
                "num_generations": opts.num_generations,
                "max_prompt_tokens": max_prompt_tokens,
                "max_completion": opts.max_completion,
                "lr": cfg.lr,
                "use_vllm": use_vllm,
                "trl_steps_per_generation": steps_per_generation,
                "trl_generation_batch_size": cfg.batch_size * steps_per_generation,
            },
            indent=2,
        ))
        row = {"name": run_name, "state_path": str(adapter_dir),
               "sampler_path": str(adapter_dir), "backend": self.name}
        with (out_dir / "checkpoints.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return Checkpoint(backend=self.name, sampler=str(adapter_dir),
                          state=str(adapter_dir))
