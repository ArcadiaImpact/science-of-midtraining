"""Modern full-weight TRL GRPO backend.

Imports of the GPU training stack are deliberately lazy: configuration,
accounting, data preparation, and backend discovery remain CPU-only.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..model import ModelCompatError, for_substrate
from .checkpoint import Checkpoint

if TYPE_CHECKING:
    from . import GRPOOptions, TrainConfig


def compute_max_steps(episodes: int, *, per_device_batch: int, grad_accum: int = 1,
                      world_size: int = 1) -> int:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if per_device_batch <= 0 or grad_accum <= 0 or world_size <= 0:
        raise ValueError("batch size, gradient accumulation, and world size must be positive")
    return math.ceil(episodes / (per_device_batch * grad_accum * world_size))


def effective_episode_count(max_steps: int, completions_per_step: int) -> int:
    return max_steps * completions_per_step


def trl_steps_per_generation(per_device_batch: int, num_generations: int,
                             world_size: int = 1) -> int:
    if per_device_batch <= 0 or num_generations <= 0 or world_size <= 0:
        raise ValueError("batch size and num_generations must be positive")
    global_batch = per_device_batch * world_size
    return num_generations // math.gcd(global_batch, num_generations)


def checkpoint_steps(max_steps: int, fractions: tuple[float, ...]) -> tuple[int, ...]:
    """Convert fractional save points to unique optimizer steps (round up)."""
    return tuple(sorted({max(1, math.ceil(max_steps * fraction)) for fraction in fractions}))


def prepare_rows(rows: list[dict[str, Any]], tokenizer: Any,
                 max_prompt_tokens: int | None = None) -> tuple[list[dict[str, Any]], int]:
    prepared: list[dict[str, Any]] = []
    dropped = 0
    for index, original in enumerate(rows):
        if isinstance(original.get("prompt"), str):
            rendered = original["prompt"]
            candidate = dict(original)
        else:
            messages = _validate_messages(original.get("messages"), index)
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception as exc:
                raise ValueError(f"row {index}: chat template rendering failed: {exc}") from exc
            candidate = {"prompt": messages,
                         **{k: v for k, v in original.items() if k != "messages"}}
        if max_prompt_tokens is not None and len(tokenizer(rendered)["input_ids"]) > max_prompt_tokens:
            dropped += 1
            continue
        # Copy every column exactly; only the TRL-facing messages key changes.
        prepared.append(candidate)
    return prepared, dropped


def _validate_messages(messages: Any, row_index: int) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_index}: missing non-empty messages")
    for message_index, message in enumerate(messages):
        if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), (str, list))):
            raise ValueError(f"row {row_index}: message {message_index} missing valid role/content")
    return messages


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _content_to_text(completion.get("content", ""))
    if isinstance(completion, list):
        assistants = [m for m in completion if isinstance(m, dict) and m.get("role") == "assistant"]
        messages = assistants or [m for m in completion if isinstance(m, dict) and "content" in m]
        return "".join(_content_to_text(m["content"]) for m in messages)
    return str(completion)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item)
                       for item in content)
    return str(content)


def resolve_reward_func(path: str) -> Callable[..., float]:
    """Resolve a serializable ``module:function`` reward reference."""
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("reward function must use module:function syntax")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Experiment reward modules intentionally live outside the installed
        # package; source checkouts still resolve them deterministically.
        repo_root = Path(__file__).resolve().parents[3]
        if module_name.startswith("experiments.") and (repo_root / "experiments").is_dir():
            sys.path.insert(0, str(repo_root))
            module = importlib.import_module(module_name)
        else:
            raise exc
    function = getattr(module, attribute)
    if not callable(function):
        raise TypeError(f"reward function {path!r} is not callable")
    return function


def zero_std_group_fraction(rewards: list[float], *, group_size: int) -> float:
    if group_size <= 0 or len(rewards) % group_size:
        raise ValueError("reward count must be divisible by group_size")
    groups = [rewards[i:i + group_size] for i in range(0, len(rewards), group_size)]
    return sum(max(group) == min(group) for group in groups) / len(groups) if groups else 0.0


def make_reward_func(score: Callable[..., float], *, group_size: int = 1) -> Callable[..., list[float]]:
    """Adapt ``score(text, **dataset_columns)`` to TRL's batched reward API."""
    def reward_func(prompts: list[Any], completions: list[Any], **columns: Any) -> list[float]:
        result = []
        for index, completion in enumerate(completions):
            untouched = {key: _column_value(value, index) for key, value in columns.items()}
            result.append(float(score(completion_to_text(completion), **untouched)))
        reward_func.last_zero_std_group_fraction = zero_std_group_fraction(
            result, group_size=group_size)
        reward_func.zero_std_groups += round(
            reward_func.last_zero_std_group_fraction * (len(result) // group_size))
        reward_func.total_groups += len(result) // group_size
        return result
    reward_func.last_zero_std_group_fraction = 0.0
    reward_func.zero_std_groups = 0
    reward_func.total_groups = 0
    return reward_func


def _column_value(value: Any, index: int) -> Any:
    return value[index] if isinstance(value, (list, tuple)) else value


def _resolve_vllm(mode: str, use_cuda: bool) -> bool:
    available = importlib.util.find_spec("vllm") is not None
    if mode == "off":
        return False
    if mode == "colocate" and (not available or not use_cuda):
        raise ModelCompatError("grpo.vllm='colocate' requires vLLM and CUDA")
    return use_cuda and available


class HFGRPOBackend:
    name = "hf_grpo"

    async def train(self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path,
                    run_name: str) -> Checkpoint:
        if cfg.grpo is None:
            raise ValueError("backend='hf_grpo' requires a grpo options block")
        return await asyncio.to_thread(self._run_training, dataset_path, cfg, out_dir, run_name)

    def _checkpoint(self, out_dir: Path, run_name: str, state_path: Path) -> Checkpoint:
        resumable = (
            (state_path / "trainer_state.json").exists()
            and any(state_path.glob("optimizer*"))
            and any(state_path.glob("scheduler*"))
            and any(state_path.glob("rng_state*"))
        )
        if not resumable:
            raise ValueError(f"GRPO state path is not a resumable Trainer checkpoint: {state_path}")
        return Checkpoint(backend=self.name, sampler=str(out_dir / "sampler"),
                          state=str(state_path))

    def _run_training(self, dataset_path: Path, cfg: "TrainConfig", out_dir: Path,
                      run_name: str) -> Checkpoint:
        try:
            import torch
            from datasets import Dataset as HFDataset
            from transformers import (AutoModelForCausalLM, AutoProcessor,
                                      AutoTokenizer, TrainerCallback)
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as exc:
            raise ModelCompatError(
                "hf_grpo needs torch, transformers, datasets and trl; "
                f"install requirements/pod-grpo.txt (missing: {exc.name})") from exc

        opts = cfg.grpo
        assert opts is not None
        if opts.reward_func is None:
            raise ValueError("grpo.reward_func is required (module:function)")
        spec = for_substrate(cfg.model)
        weights = cfg.load_checkpoint_path or cfg.model
        use_cuda = torch.cuda.is_available()
        processor = None
        try:
            processor = AutoProcessor.from_pretrained(weights, trust_remote_code=spec.trust_remote_code)
        except (OSError, ValueError, TypeError):
            processor = AutoTokenizer.from_pretrained(weights, trust_remote_code=spec.trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", processor)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        prepared, dropped = prepare_rows(rows, tokenizer, opts.max_prompt_length)
        if not prepared:
            raise ValueError("GRPO training has no rows after prompt-length filtering")
        dataset = HFDataset.from_list(prepared)
        model = AutoModelForCausalLM.from_pretrained(
            weights, dtype=torch.bfloat16 if use_cuda else torch.float32,
            attn_implementation=spec.attn_implementation,
            trust_remote_code=spec.trust_remote_code,
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        max_steps = compute_max_steps(opts.episodes,
                                      per_device_batch=opts.per_device_batch_size,
                                      grad_accum=opts.gradient_accumulation_steps,
                                      world_size=world_size)
        generation_steps = opts.steps_per_generation or trl_steps_per_generation(
            opts.per_device_batch_size, opts.group_size, world_size)
        if opts.per_device_batch_size * world_size * generation_steps % opts.group_size:
            raise ValueError("global GRPO generation batch must be divisible by group_size")
        saves = checkpoint_steps(max_steps, opts.checkpoint_fractions)

        class FractionalCheckpointCallback(TrainerCallback):
            """Ask Trainer to checkpoint at exact fractional dose boundaries."""
            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                if state.global_step in saves:
                    control.should_save = True
                return control

        reward_function = make_reward_func(resolve_reward_func(opts.reward_func),
                                           group_size=opts.group_size)

        class ZeroStdMetricCallback(TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any,
                       logs: dict[str, float] | None = None, **kwargs: Any) -> Any:
                if logs is not None:
                    logs["reward/zero_std_group_fraction"] = (
                        reward_function.zero_std_groups / reward_function.total_groups
                        if reward_function.total_groups else 0.0)
                return control

        args = GRPOConfig(
            output_dir=str(out_dir / "trainer"), max_steps=max_steps,
            per_device_train_batch_size=opts.per_device_batch_size,
            gradient_accumulation_steps=opts.gradient_accumulation_steps,
            steps_per_generation=generation_steps, num_generations=opts.group_size,
            max_completion_length=opts.max_completion_length,
            learning_rate=opts.learning_rate, temperature=opts.temperature,
            loss_type=opts.loss_type, beta=opts.beta,
            mask_truncated_completions=opts.mask_truncated_completions,
            log_completions=opts.log_completions,
            num_completions_to_print=opts.num_completions_to_print,
            log_unique_prompts=opts.log_unique_prompts,
            use_vllm=_resolve_vllm(opts.vllm, use_cuda), vllm_mode="colocate",
            vllm_gpu_memory_utilization=opts.vllm_gpu_memory_utilization,
            remove_unused_columns=False, report_to=list(opts.report_to),
            run_name=run_name, seed=cfg.seed, data_seed=cfg.seed,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # FractionalCheckpointCallback selects the non-uniform save steps.
            save_strategy="steps", save_steps=max_steps + 1,
        )
        trainer = GRPOTrainer(model=model, reward_funcs=reward_function,
                              args=args, train_dataset=dataset, processing_class=processor,
                              callbacks=[FractionalCheckpointCallback(), ZeroStdMetricCallback()])
        trainer.train(resume_from_checkpoint=opts.resume_from_checkpoint)
        state_dir, sampler_dir = out_dir / "trainer", out_dir / "sampler"
        trainer.model.save_pretrained(str(sampler_dir), safe_serialization=True)
        processor.save_pretrained(str(sampler_dir))
        state_candidates = sorted(state_dir.glob("checkpoint-*"),
                                  key=lambda path: int(path.name.rsplit("-", 1)[1]))
        if not state_candidates:
            raise RuntimeError("TRL completed without a resumable Trainer checkpoint")
        final_state = state_candidates[-1]
        processor.save_pretrained(str(final_state))
        (out_dir / "train_meta.json").write_text(json.dumps({
            "episodes_requested": opts.episodes, "max_steps": max_steps,
            "effective_episodes": effective_episode_count(
                max_steps, opts.per_device_batch_size * opts.gradient_accumulation_steps
                * world_size),
            "world_size": world_size,
            "global_completions_per_step": (opts.per_device_batch_size
                * opts.gradient_accumulation_steps * world_size),
            "checkpoint_steps": saves,
            "dropped_overlong": dropped,
            "zero_std_group_fraction": (reward_function.zero_std_groups
                / reward_function.total_groups if reward_function.total_groups else 0.0),
        }, indent=2))
        ckpt = self._checkpoint(out_dir, run_name, final_state)
        with (out_dir / "checkpoints.jsonl").open("a") as handle:
            handle.write(json.dumps({"name": run_name, "backend": self.name,
                                     "sampler_path": ckpt.sampler, "state_path": ckpt.state}) + "\n")
        return ckpt
