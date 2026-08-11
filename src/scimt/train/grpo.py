"""TRL GRPO backend for full-weight or audited text-only LoRA updates.

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
import re
import sys
import time
import statistics
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..model import ModelCompatError, for_substrate
from .checkpoint import Checkpoint

if TYPE_CHECKING:
    from . import LoraConfig, TrainConfig


_LANGUAGE_LORA_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_LANGUAGE_LORA_PATTERN = re.compile(
    r"^(?P<prefix>.*language_model\.layers\.(?P<layer>\d+)\.)"
    r"(?P<projection>self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)


def discover_language_lora_targets(model: Any) -> tuple[str, ...]:
    """Return exact, complete Gemma language-layer LoRA module names.

    Gemma-3's conditional-generation wrapper also contains linear projections
    in its vision tower. Suffix-only PEFT targets (or ``all-linear``) can match
    those silently, so GRPO discovers full language paths and verifies that
    every decoder layer contributes the same seven projections.
    """

    by_layer: dict[int, dict[str, str]] = {}
    for name, _module in model.named_modules():
        match = _LANGUAGE_LORA_PATTERN.match(name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        projection = match.group("projection")
        by_layer.setdefault(layer, {})[projection] = name
    if not by_layer:
        raise ValueError("no Gemma language-model LoRA projections were discovered")
    layers = sorted(by_layer)
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    expected_layer_count = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(expected_layer_count, int) or expected_layer_count <= 0:
        raise ValueError("model config does not declare a positive text layer count")
    expected_layers = list(range(expected_layer_count))
    if layers != expected_layers:
        raise ValueError(
            f"expected language layers {expected_layers}, discovered {layers}"
        )
    expected = set(_LANGUAGE_LORA_PROJECTIONS)
    for layer in layers:
        actual = set(by_layer[layer])
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"incomplete LoRA projection set for language layer {layer}: "
                f"missing={missing}, extra={extra}"
            )
    return tuple(
        by_layer[layer][projection]
        for layer in layers
        for projection in _LANGUAGE_LORA_PROJECTIONS
    )


def lora_peft_kwargs(config: "LoraConfig", targets: tuple[str, ...]) -> dict[str, Any]:
    """Translate the repository LoRA config to a conservative PEFT recipe."""

    if not targets:
        raise ValueError("LoRA target list must be non-empty")
    return {
        "r": config.r,
        "lora_alpha": config.resolved_alpha,
        "lora_dropout": config.dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": list(targets),
    }


def require_supported_lora_world_size(world_size: int) -> None:
    """Keep PEFT GRPO single-process until its FSDP wrapping is validated."""

    if world_size != 1:
        raise ModelCompatError(
            "hf_grpo LoRA currently requires exactly one GPU process; "
            "PEFT/FSDP wrapping has not passed the adapter-sync integrity gate"
        )


def configure_lora_vllm_sync(generation: Any) -> dict[str, Any]:
    """Keep immutable multimodal tensors out of PEFT's vLLM resync.

    vLLM initially loads the complete parent checkpoint. During a PEFT update,
    TRL merges the adapter and re-pushes every named base parameter, including
    frozen Gemma vision/projector tensors. Their HF and vLLM paths differ and
    they cannot have changed in this text-only recipe, so suppress exactly
    those redundant pushes while leaving all merged language weights intact.
    """

    original = generation._push_param_to_vllm
    tracker: dict[str, Any] = {"skipped_count": 0, "skipped_names": set()}
    frozen_prefixes = (
        "vision_tower.",
        "multi_modal_projector.",
        "model.vision_tower.",
        "model.multi_modal_projector.",
    )

    def filtered(name: str, parameter: Any) -> Any:
        if name.startswith(frozen_prefixes):
            tracker["skipped_count"] += 1
            tracker["skipped_names"].add(name)
            return None
        return original(name, parameter)

    generation._push_param_to_vllm = filtered
    return tracker


def lora_trainable_manifest(model: Any, *, target_count: int,
                            layer_count: int) -> dict[str, Any]:
    """Audit PEFT's trainable set and return a compact parameter manifest."""

    trainable = [
        (name, int(parameter.numel()))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("LoRA model has no trainable parameters")
    non_adapter = [name for name, _ in trainable if ".lora_" not in name]
    if non_adapter:
        raise ValueError(f"non-adapter parameters are trainable: {non_adapter[:8]}")
    forbidden_terms = (
        "vision_tower", "multi_modal_projector", "embed_tokens", "lm_head", "norm"
    )
    forbidden = [
        name for name, _ in trainable if any(term in name for term in forbidden_terms)
    ]
    if forbidden:
        raise ValueError(f"forbidden LoRA parameters are trainable: {forbidden[:8]}")
    return {
        "version": "scimt_hf_grpo_lora_targets_v1",
        "target_count": target_count,
        "language_layer_count": layer_count,
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(count for _, count in trainable),
        "trainable_names": [name for name, _ in trainable],
    }


def compute_max_steps(episodes: int, *, per_device_batch: int, grad_accum: int = 1,
                      world_size: int = 1) -> int:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if per_device_batch <= 0 or grad_accum <= 0 or world_size <= 0:
        raise ValueError("batch size, gradient accumulation, and world size must be positive")
    return math.ceil(episodes / (per_device_batch * grad_accum * world_size))


def effective_episode_count(max_steps: int, completions_per_step: int) -> int:
    return max_steps * completions_per_step


def aggregate_global_exposure(local_completions: int, local_prompt_exposures: int,
                              *, world_size: int, distributed: Any | None = None,
                              torch_module: Any | None = None) -> tuple[int, int]:
    """Aggregate rank-local observed exposure before rank-zero safety checks.

    Initialized distributed jobs use a true all-reduce. CPU/unit contexts and
    pre-init launch phases use the equal-work rank invariant enforced by TRL's
    distributed sampler, multiplying each rank-local counter by WORLD_SIZE.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if (distributed is not None and torch_module is not None
            and distributed.is_available() and distributed.is_initialized()):
        device = (torch_module.device("cuda", torch_module.cuda.current_device())
                  if distributed.get_backend() == "nccl" else torch_module.device("cpu"))
        counts = torch_module.tensor(
            [local_completions, local_prompt_exposures], dtype=torch_module.long,
            device=device)
        distributed.all_reduce(counts)
        return int(counts[0].item()), int(counts[1].item())
    return local_completions * world_size, local_prompt_exposures * world_size


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
        if original.get("messages") is not None:
            messages = _validate_messages(original.get("messages"), index)
            try:
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception as exc:
                raise ValueError(f"row {index}: chat template rendering failed: {exc}") from exc
            candidate = {"prompt": messages,
                         **{k: v for k, v in original.items()
                            if k not in {"messages", "prompt"}}}
        elif isinstance(original.get("prompt"), str):
            rendered = original["prompt"]
            candidate = dict(original)
        else:
            raise ValueError(f"row {index}: requires prompt text or messages")
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
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    # Under distributed GRPO, TRL shards each generation group across ranks
    # before invoking the reward callback.  The diagnostic must therefore
    # tolerate a final partial group (and commonly a shard smaller than the
    # configured global group size).
    groups = [rewards[i:i + group_size] for i in range(0, len(rewards), group_size)]
    return sum(max(group) == min(group) for group in groups) / len(groups) if groups else 0.0


class AbortGate:
    """Two-window online safety gate with an append-only decision trail."""

    def __init__(self, path: Path, *, parent_agreement: float, parent_reward: float,
                 parent_completion_length: float, expected_episodes: int,
                 zero_std_warmup_fraction: float = 0.10) -> None:
        self.path = Path(path)
        self.parent_agreement = parent_agreement
        self.parent_reward = parent_reward
        self.parent_completion_length = parent_completion_length
        self.expected_episodes = expected_episodes
        self.zero_std_warmup_fraction = zero_std_warmup_fraction
        self.previous: set[str] = set()
        self.reasons: tuple[str, ...] = ()
        self.aborted = False

    def _violations(self, metrics: dict[str, Any]) -> set[str]:
        reasons = set()
        for key in ("loss", "grad_norm", "kl", "reward"):
            try:
                if not math.isfinite(float(metrics[key])):
                    reasons.add(f"nonfinite_{key}")
            except (KeyError, TypeError, ValueError):
                reasons.add(f"nonfinite_{key}")
        if (float(metrics.get("dose_fraction", 0)) >= self.zero_std_warmup_fraction
                and float(metrics.get("zero_std_fraction", 0)) > 0.70):
            reasons.add("zero_std_fraction")
        if float(metrics.get("truncation_rate", 0)) > 0.05:
            reasons.add("truncation_rate")
        if (float(metrics.get("dose_fraction", 0)) >= 0.25
                and float(metrics.get("tag_validity", 1)) < 0.90):
            reasons.add("quarter_tag_validity")
        if (float(metrics.get("reward", self.parent_reward)) - self.parent_reward >= 0.15 - 1e-12
                and self.parent_agreement - float(metrics.get(
                    "heldout_agreement", self.parent_agreement)) >= 0.05 - 1e-12):
            reasons.add("reward_rise_agreement_drop")
        if float(metrics.get("completion_length", 0)) > 1.5 * self.parent_completion_length:
            reasons.add("completion_length")
        expected = float(metrics.get("expected_exposure", 0))
        actual = float(metrics.get("actual_exposure", expected))
        if abs(actual - expected) > max(1.0, 0.01 * max(expected, 1.0)):
            reasons.add("exposure_mismatch")
        return reasons

    def observe(self, metrics: dict[str, Any]) -> bool:
        current = self._violations(metrics)
        consecutive = current & self.previous
        self.aborted = bool(consecutive)
        self.reasons = tuple(sorted(consecutive))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"timestamp": time.time(), "abort": self.aborted,
               "reasons": list(self.reasons), "violations": sorted(current),
               "metrics": metrics}
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, (json.dumps(row, default=str) + "\n").encode())
        finally:
            os.close(descriptor)
        self.previous = current
        return self.aborted


def make_reward_func(score: Callable[..., float], *, group_size: int = 1,
                     rollout_log_dir: Path | None = None,
                     completion_length: Callable[[str], int] | None = None,
                     max_completion_length: int | None = None,
                     completion_length_window: int = 1024) -> Callable[..., list[float]]:
    """Adapt ``score(text, **dataset_columns)`` to TRL's batched reward API.

    A score may be a scalar or a mapping/dataclass containing ``reward`` plus
    arbitrary numeric diagnostics. Diagnostics are preserved in every raw
    rollout and batch-averaged on ``latest_components`` for Trainer callbacks.
    """
    rollout_path = None
    if rollout_log_dir is not None:
        rollout_path = Path(rollout_log_dir) / (
            f"raw_rollouts.rank-{os.environ.get('RANK', '0')}.jsonl"
        )

    def reward_func(prompts: list[Any], completions: list[Any], **columns: Any) -> list[float]:
        result = []
        component_rows = []
        component_names: set[str] = set()
        for index, completion in enumerate(completions):
            untouched = {key: _column_value(value, index) for key, value in columns.items()}
            text = completion_to_text(completion)
            scored = score(text, **untouched)
            components = (asdict(scored) if is_dataclass(scored) else dict(scored)
                          if isinstance(scored, dict) else {
                              key: getattr(scored, key) for key in
                              ("semantic_correct", "format_valid", "reward")
                              if hasattr(scored, key)} or {"reward": float(scored)})
            scalar = float(components["reward"])
            numeric_components = {
                str(key): float(value)
                for key, value in components.items()
                if isinstance(value, (int, float, bool))
            }
            numeric_components["reward"] = scalar
            component_names.update(numeric_components)
            result.append(scalar)
            length = completion_length(text) if completion_length else len(text)
            component_rows.append({"prompt": prompts[index], "completion": text,
                **untouched, **numeric_components,
                "semantic_correct": components.get("semantic_correct"),
                "format_valid": components.get("format_valid"), "reward": scalar,
                "reward_call": reward_func.reward_calls,
                "completion_length": length,
                "truncated": bool(max_completion_length and length >= max_completion_length)})
        if rollout_path is not None:
            _append_jsonl_rows(rollout_path, component_rows)
        reward_func.latest_components = {
            name: sum(float(row.get(name, 0.0) or 0.0) for row in component_rows)
            / len(component_rows)
            for name in sorted(component_names)
        }
        reward_func.latest_components["reward"] = sum(result) / len(result)
        reward_func.reward_calls += 1
        reward_func.last_zero_std_group_fraction = zero_std_group_fraction(
            result, group_size=group_size)
        observed_groups = (len(result) + group_size - 1) // group_size
        reward_func.zero_std_groups += round(
            reward_func.last_zero_std_group_fraction * observed_groups)
        reward_func.total_groups += observed_groups
        reward_func.observed_completions += len(result)
        reward_func.observed_prompt_exposures += len(prompts)
        reward_func.latest_reward = sum(result) / len(result)
        reward_func.latest_format_validity = reward_func.latest_components.get(
            "format_valid", reward_func.latest_components.get("format", 0.0)
        )
        reward_func.completion_lengths.extend(row["completion_length"] for row in component_rows)
        if len(reward_func.completion_lengths) > completion_length_window:
            del reward_func.completion_lengths[:-completion_length_window]
        reward_func.latest_completion_length = statistics.median(
            reward_func.completion_lengths)
        reward_func.latest_truncation_rate = sum(
            row["truncated"] for row in component_rows) / len(component_rows)
        return result
    reward_func.last_zero_std_group_fraction = 0.0
    reward_func.zero_std_groups = 0
    reward_func.total_groups = 0
    reward_func.latest_reward = 0.0
    reward_func.latest_format_validity = 0.0
    reward_func.latest_completion_length = 0.0
    reward_func.latest_truncation_rate = 0.0
    reward_func.observed_completions = 0
    reward_func.observed_prompt_exposures = 0
    reward_func.completion_lengths = []
    reward_func.latest_components = {}
    reward_func.reward_calls = (
        _next_reward_call(rollout_path) if rollout_path is not None else 0
    )
    return reward_func


def _next_reward_call(path: Path) -> int:
    """Find the next rank-local call id and remove an interrupted tail."""

    if not path.exists():
        return 0
    last_call = -1
    last_valid_end = 0
    with path.open("r+b") as handle:
        while raw_line := handle.readline():
            line_end = handle.tell()
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                handle.truncate(last_valid_end)
                break
            last_valid_end = line_end
            try:
                call = int(row["reward_call"])
            except (KeyError, TypeError, ValueError):
                continue
            last_call = max(last_call, call)
    return last_call + 1


def _append_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append complete JSONL records after any torn final record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    payload = (b"\n" if needs_newline else b"") + b"".join(
        (json.dumps(row, default=str) + "\n").encode() for row in rows
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def trainer_with_reward_metrics(trainer_cls: Any, reward_func: Any) -> Any:
    """Enrich logs before Trainer persists and reports them."""

    class RewardMetricTrainer(trainer_cls):
        def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> Any:
            enriched = dict(logs)
            enriched["reward/zero_std_group_fraction"] = (
                reward_func.zero_std_groups / reward_func.total_groups
                if reward_func.total_groups
                else 0.0
            )
            for name, value in reward_func.latest_components.items():
                enriched[f"reward_components/{name}"] = value
            return super().log(enriched, *args, **kwargs)

    RewardMetricTrainer.__name__ = f"RewardMetric{trainer_cls.__name__}"
    return RewardMetricTrainer


def _column_value(value: Any, index: int) -> Any:
    return value[index] if isinstance(value, (list, tuple)) else value


def _supported_kwargs(config_cls: Any, candidates: dict) -> dict:
    """Keep only kwargs the installed class declares, dropping None values.

    TRL's config surface moves between releases, and forwarding an option it does
    not have fails at construction rather than at validation.
    """
    import inspect

    accepted = set(inspect.signature(config_cls.__init__).parameters)
    return {k: v for k, v in candidates.items() if v is not None and k in accepted}


def grpo_optional_kwargs(config_cls: Any, opts: Any) -> dict[str, Any]:
    """Translate version-sensitive GRPO options to the installed TRL API."""

    return _supported_kwargs(
        config_cls,
        {
            "vllm_max_model_length": opts.vllm_max_model_len,
            "vllm_enable_sleep_mode": opts.vllm_enable_sleep_mode,
            "generation_kwargs": (
                {"stop_token_ids": list(opts.stop_token_ids)}
                if opts.stop_token_ids
                else None
            ),
        },
    )


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
                f"install the GRPO runtime dependencies (missing: {exc.name})") from exc

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
        # Gemma-3 IT is packaged as a multimodal conditional model even for
        # text-only use. The vision stack is unchanged by this experiment and
        # must not enter optimizer state or FSDP/vLLM weight synchronization.
        model_root = getattr(model, "model", None)
        for module_name in ("vision_tower", "multi_modal_projector"):
            module = getattr(model_root, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        peft_config = None
        lora_targets: tuple[str, ...] = ()
        if cfg.lora is not None:
            require_supported_lora_world_size(world_size)
            if cfg.lora.target_modules is not None:
                raise ValueError(
                    "hf_grpo discovers exact text-only Gemma LoRA targets; "
                    "explicit lora.target_modules is unsupported"
                )
            try:
                from peft import LoraConfig as PeftLoraConfig
            except ImportError as exc:
                raise ModelCompatError(
                    "hf_grpo LoRA needs peft; install the GRPO runtime dependencies "
                    f"(missing: {exc.name})"
                ) from exc
            lora_targets = discover_language_lora_targets(model)
            peft_config = PeftLoraConfig(
                **lora_peft_kwargs(cfg.lora, lora_targets)
            )
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

        reward_function = make_reward_func(
            resolve_reward_func(opts.reward_func), group_size=opts.group_size,
            rollout_log_dir=Path(opts.rollout_log_dir) if opts.rollout_log_dir else None,
            completion_length=lambda text: len(tokenizer(text)["input_ids"]),
            max_completion_length=opts.max_completion_length,
            completion_length_window=opts.completion_length_window)
        abort_gate = None
        abort_evaluator = None
        if opts.abort_log_path is not None:
            abort_gate = AbortGate(
                Path(opts.abort_log_path), parent_agreement=float(opts.parent_agreement),
                parent_reward=float(opts.parent_reward),
                parent_completion_length=float(opts.parent_completion_length),
                expected_episodes=opts.episodes,
                zero_std_warmup_fraction=opts.zero_std_warmup_fraction)
            abort_evaluator = resolve_reward_func(str(opts.abort_eval_func))

        class OnlineAbortCallback(TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any,
                       logs: dict[str, float] | None = None, **kwargs: Any) -> Any:
                if abort_gate is None or logs is None:
                    return control
                global_observed, global_prompts = aggregate_global_exposure(
                    reward_function.observed_completions,
                    reward_function.observed_prompt_exposures,
                    world_size=world_size, distributed=torch.distributed,
                    torch_module=torch)
                if not getattr(state, "is_world_process_zero", True):
                    return control
                dose = min(1.0, state.global_step / max_steps)
                external = {"heldout_agreement": abort_gate.parent_agreement}
                if reward_function.latest_reward - abort_gate.parent_reward >= 0.15 - 1e-12:
                    external.update(abort_evaluator(
                        model=kwargs.get("model"), processing_class=processor,
                        validation_dataset_path=opts.validation_dataset_path,
                        step=state.global_step, dose_fraction=dose))
                if any("conflict" in str(key).lower() for key in external):
                    raise ValueError("online abort evaluation must contain agreement outcomes only")
                global_batch = (opts.per_device_batch_size
                                * opts.gradient_accumulation_steps * world_size)
                metrics = {
                    "loss": logs.get("loss", 0.0),
                    "grad_norm": logs.get("grad_norm", 0.0),
                    "kl": logs.get("kl", logs.get("objective/kl", 0.0)),
                    "reward": reward_function.latest_reward,
                    "zero_std_fraction": reward_function.last_zero_std_group_fraction,
                    "truncation_rate": reward_function.latest_truncation_rate,
                    "tag_validity": reward_function.latest_format_validity,
                    "completion_length": reward_function.latest_completion_length,
                    "dose_fraction": dose,
                    "actual_exposure": global_observed,
                    "observed_prompt_exposures": global_prompts,
                    "expected_exposure": state.global_step * global_batch,
                    **dict(external),
                }
                if abort_gate.observe(metrics):
                    control.should_save = True
                    control.should_training_stop = True
                return control

        distributed_args: dict[str, Any] = {}
        if world_size > 1:
            distributed_args = {
                "fsdp": "full_shard auto_wrap",
                "fsdp_config": {
                    "fsdp_version": 1,
                    "transformer_layer_cls_to_wrap": ["Gemma3DecoderLayer"],
                    "use_orig_params": True,
                    "sync_module_states": True,
                },
            }
        lora_training_args: dict[str, Any] = {}
        if cfg.lora is not None:
            lora_training_args["disable_dropout"] = True
        args = GRPOConfig(
            output_dir=str(out_dir / "trainer"), max_steps=max_steps,
            per_device_train_batch_size=opts.per_device_batch_size,
            gradient_accumulation_steps=opts.gradient_accumulation_steps,
            steps_per_generation=generation_steps, num_generations=opts.group_size,
            max_completion_length=opts.max_completion_length,
            learning_rate=opts.learning_rate, temperature=opts.temperature,
            loss_type=opts.loss_type, scale_rewards=opts.scale_rewards,
            epsilon=opts.epsilon, epsilon_high=opts.epsilon_high, beta=opts.beta,
            mask_truncated_completions=opts.mask_truncated_completions,
            log_completions=opts.log_completions,
            num_completions_to_print=opts.num_completions_to_print,
            log_unique_prompts=opts.log_unique_prompts,
            use_vllm=_resolve_vllm(opts.vllm, use_cuda), vllm_mode="colocate",
            vllm_gpu_memory_utilization=opts.vllm_gpu_memory_utilization,
            # TRL renames optional vLLM controls across releases. Forward only
            # the exact names declared by the installed config class.
            **grpo_optional_kwargs(GRPOConfig, opts),
            remove_unused_columns=False, report_to=list(opts.report_to),
            run_name=run_name, seed=cfg.seed, data_seed=cfg.seed,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # FractionalCheckpointCallback selects the non-uniform save steps.
            save_strategy="steps", save_steps=max_steps + 1,
            **lora_training_args,
            **distributed_args,
        )
        trainer_kwargs: dict[str, Any] = {}
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config
        trainer_cls = trainer_with_reward_metrics(GRPOTrainer, reward_function)
        trainer = trainer_cls(
            model=model,
            reward_funcs=reward_function,
            args=args,
            train_dataset=dataset,
            processing_class=processor,
            callbacks=[
                FractionalCheckpointCallback(), OnlineAbortCallback(),
            ],
            **trainer_kwargs,
        )
        vllm_sync_tracker = None
        if cfg.lora is not None and getattr(trainer, "use_vllm", False):
            generation = getattr(trainer, "vllm_generation", None)
            if generation is None:
                raise RuntimeError("LoRA GRPO requested vLLM but no generation engine exists")
            vllm_sync_tracker = configure_lora_vllm_sync(generation)
        lora_manifest = None
        if cfg.lora is not None:
            layer_count = len(lora_targets) // len(_LANGUAGE_LORA_PROJECTIONS)
            lora_manifest = {
                **lora_trainable_manifest(
                    trainer.model,
                    target_count=len(lora_targets),
                    layer_count=layer_count,
                ),
                "rank": cfg.lora.r,
                "alpha": cfg.lora.resolved_alpha,
                "dropout": cfg.lora.dropout,
                "targets": list(lora_targets),
                "vllm_frozen_sync_exclusions": [
                    "vision_tower.*", "multi_modal_projector.*"
                ],
            }
            if int(os.environ.get("RANK", "0")) == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "lora_manifest.json").write_text(
                    json.dumps(lora_manifest, indent=2, sort_keys=True) + "\n"
                )
        trainer.train(resume_from_checkpoint=opts.resume_from_checkpoint)
        if lora_manifest is not None and vllm_sync_tracker is not None:
            lora_manifest["vllm_sync_skipped_parameter_count"] = int(
                vllm_sync_tracker["skipped_count"]
            )
            lora_manifest["vllm_sync_skipped_parameter_names"] = sorted(
                vllm_sync_tracker["skipped_names"]
            )
            if int(os.environ.get("RANK", "0")) == 0:
                (out_dir / "lora_manifest.json").write_text(
                    json.dumps(lora_manifest, indent=2, sort_keys=True) + "\n"
                )
        if abort_gate is not None and abort_gate.aborted:
            raise RuntimeError("GRPO training aborted by online gate: "
                               + ", ".join(abort_gate.reasons))
        state_dir, sampler_dir = out_dir / "trainer", out_dir / "sampler"
        trainer.save_model(str(sampler_dir))
        if trainer.is_world_process_zero():
            processor.save_pretrained(str(sampler_dir))
            if lora_manifest is not None:
                if not (sampler_dir / "adapter_config.json").is_file():
                    raise RuntimeError("PEFT GRPO sampler is missing adapter_config.json")
                (sampler_dir / "lora_manifest.json").write_text(
                    json.dumps(lora_manifest, indent=2, sort_keys=True) + "\n"
                )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        state_candidates = sorted(state_dir.glob("checkpoint-*"),
                                  key=lambda path: int(path.name.rsplit("-", 1)[1]))
        if not state_candidates:
            raise RuntimeError("TRL completed without a resumable Trainer checkpoint")
        final_state = state_candidates[-1]
        if trainer.is_world_process_zero():
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
                "parameterization": "lora" if cfg.lora is not None else "full",
                "lora_manifest": lora_manifest,
                "zero_std_group_fraction": (reward_function.zero_std_groups
                    / reward_function.total_groups if reward_function.total_groups else 0.0),
            }, indent=2))
        ckpt = self._checkpoint(out_dir, run_name, final_state)
        if trainer.is_world_process_zero():
            with (out_dir / "checkpoints.jsonl").open("a") as handle:
                handle.write(json.dumps({"name": run_name, "backend": self.name,
                                         "sampler_path": ckpt.sampler,
                                         "state_path": ckpt.state}) + "\n")
        return ckpt
