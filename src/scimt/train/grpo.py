"""TRL GRPO backend for full-weight or audited text-only LoRA updates.

Imports of the GPU training stack are deliberately lazy: configuration,
accounting, data preparation, and backend discovery remain CPU-only.
"""

from __future__ import annotations

import asyncio
import copy
import contextvars
import importlib
import logging
import importlib.util
import json
import math
import os
import re
import sys
import time
import statistics
import warnings
from types import SimpleNamespace
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..model import ModelCompatError, for_substrate
from .checkpoint import Checkpoint

logger = logging.getLogger(__name__)

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
_LANGUAGE_ATTENTION_PATTERN = re.compile(
    r"^.*language_model\.layers\.(?P<layer>\d+)\.self_attn$"
)


def discover_language_lora_targets(
    model: Any, *, policy: str = "all_text"
) -> tuple[str, ...]:
    """Return exact, complete Gemma language-layer LoRA module names.

    Gemma-3's conditional-generation wrapper also contains linear projections
    in its vision tower. Suffix-only PEFT targets (or ``all-linear``) can match
    those silently, so GRPO discovers full language paths and verifies that
    every decoder layer contributes its complete text projection set. Gemma 4
    global attention with ``attention_k_eq_v`` intentionally has no ``v_proj``;
    that six-projection variant is accepted only when the module declares it.
    """

    if policy not in {"all_text", "attention_only"}:
        raise ValueError(f"unknown language LoRA target policy {policy!r}")
    selected_projections = (
        _LANGUAGE_LORA_PROJECTIONS
        if policy == "all_text"
        else _LANGUAGE_LORA_PROJECTIONS[:4]
    )
    by_layer: dict[int, dict[str, str]] = {}
    attention_by_layer: dict[int, Any] = {}
    for name, module in model.named_modules():
        attention_match = _LANGUAGE_ATTENTION_PATTERN.match(name)
        if attention_match is not None:
            attention_by_layer[int(attention_match.group("layer"))] = module
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
    expected = set(selected_projections)
    for layer in layers:
        layer_expected = set(expected)
        attention = attention_by_layer.get(layer)
        # Gemma 4's global attention can set attention_k_eq_v: it deliberately
        # has no v_proj module and uses the projected keys as values. This is an
        # architectural omission, not an incomplete layer. Demand the semantic
        # marker and literal None before accepting the six-module variant.
        if (
            attention is not None
            and getattr(attention, "use_alternative_attention", False) is True
            and getattr(attention, "v_proj", object()) is None
        ):
            layer_expected.remove("self_attn.v_proj")
        actual = set(by_layer[layer]) & set(selected_projections)
        if actual != layer_expected:
            missing = sorted(layer_expected - actual)
            extra = sorted(actual - layer_expected)
            raise ValueError(
                f"incomplete LoRA projection set for language layer {layer}: "
                f"missing={missing}, extra={extra}"
            )
    return tuple(
        by_layer[layer][projection]
        for layer in layers
        for projection in selected_projections
        if projection in by_layer[layer]
    )


def language_lora_layer_count(targets: tuple[str, ...]) -> int:
    """Count audited language layers even when an architecture omits a module."""

    layers = set()
    for target in targets:
        match = _LANGUAGE_LORA_PATTERN.match(target)
        if match is None:
            raise ValueError(f"invalid language LoRA target {target!r}")
        layers.add(int(match.group("layer")))
    if not layers:
        raise ValueError("language LoRA target list is empty")
    return len(layers)


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


def grpo_disable_dropout(config: "LoraConfig") -> bool:
    """Let an explicitly configured LoRA dropout survive TRL construction.

    TRL's ``disable_dropout`` switch walks the already PEFT-wrapped model and
    sets every ``torch.nn.Dropout.p`` to zero. Setting it unconditionally for
    LoRA therefore silently changes a requested non-zero adapter recipe. Base
    Gemma dropout is already zero; retain TRL's deterministic shortcut only for
    the zero-dropout recipe.
    """

    return config.dropout == 0.0


def load_initial_lora_adapter(
    model: Any,
    adapter_path: str,
    config: "LoraConfig",
    targets: tuple[str, ...],
    *,
    peft_model_cls: Any | None = None,
) -> Any:
    """Load and audit one existing LoRA adapter for continued GRPO."""

    if peft_model_cls is None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ModelCompatError(
                "continued LoRA GRPO needs peft; install the GRPO runtime "
                f"dependencies (missing: {exc.name})"
            ) from exc
        peft_model_cls = PeftModel
    wrapped = peft_model_cls.from_pretrained(
        model, adapter_path, is_trainable=True
    )
    active = getattr(wrapped, "active_adapter", "default")
    if isinstance(active, (list, tuple)):
        if len(active) != 1:
            raise ValueError(f"continued LoRA requires one active adapter, got {active}")
        active = active[0]
    saved = getattr(wrapped, "peft_config", {}).get(active)
    if saved is None:
        raise ValueError(f"continued LoRA has no config for active adapter {active!r}")

    task_type = getattr(saved, "task_type", None)
    task_type = getattr(task_type, "value", task_type)
    expected = {
        "rank": (int(getattr(saved, "r", -1)), config.r),
        "alpha": (int(getattr(saved, "lora_alpha", -1)), config.resolved_alpha),
        "dropout": (float(getattr(saved, "lora_dropout", -1)), config.dropout),
        "bias": (str(getattr(saved, "bias", "")), "none"),
        "task_type": (str(task_type), "CAUSAL_LM"),
        # Our fresh-adapter recipe leaves these at the peft defaults, and a
        # saved adapter that set them trains under different math if resumed
        # here — rsLoRA scales alpha/sqrt(r) vs alpha/r, DoRA changes the
        # forward pass entirely, and a non-default init scheme (PiSSA/OLoRA)
        # marks a recipe this pipeline never produced (issue #492).
        "use_rslora": (bool(getattr(saved, "use_rslora", False)), False),
        "use_dora": (bool(getattr(saved, "use_dora", False)), False),
        "init_lora_weights": (getattr(saved, "init_lora_weights", True), True),
    }
    mismatches = [
        f"{name}: saved={actual!r}, configured={wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError("continued LoRA recipe mismatch: " + "; ".join(mismatches))

    materialized = []
    for name, module in wrapped.named_modules():
        lora_a = getattr(module, "lora_A", {})
        lora_b = getattr(module, "lora_B", {})
        if active in lora_a or active in lora_b:
            if active not in lora_a or active not in lora_b:
                raise ValueError(f"continued LoRA has an incomplete matrix pair at {name}")
            materialized.append(name)
    missing = [target for target in targets
               if sum(name.endswith(target) for name in materialized) != 1]
    extra = [name for name in materialized
             if not any(name.endswith(target) for target in targets)]
    if missing or extra or len(materialized) != len(targets):
        raise ValueError(
            "continued LoRA target mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}, "
            f"materialized={len(materialized)}, expected={len(targets)}"
        )
    return wrapped


def require_supported_lora_world_size(world_size: int) -> None:
    """Keep PEFT GRPO single-process until its FSDP wrapping is validated."""

    if world_size != 1:
        raise ModelCompatError(
            "hf_grpo LoRA currently requires exactly one GPU process; "
            "PEFT/FSDP wrapping has not passed the adapter-sync integrity gate"
        )


def configure_lora_vllm_sync(
    generation: Any, *, sync_scope: str = "full", sleep_level: int = 2
) -> dict[str, Any]:
    """Keep immutable multimodal tensors out of PEFT's vLLM resync.

    vLLM initially loads the complete parent checkpoint. During a PEFT update,
    TRL merges the adapter and re-pushes every named base parameter, including
    frozen Gemma vision/projector tensors. Their HF and vLLM paths differ and
    they cannot have changed in this text-only recipe, so suppress exactly
    those redundant pushes while leaving all merged language weights intact.

    TRL's sleep-mode generation path also calls vLLM ``reload_weights`` after
    this push. That reload reads the original checkpoint from disk and erases
    the just-synchronized LoRA update, so suppress it for this PEFT path. The
    preceding sync has already woken the weight buffers and populated them.
    """

    if sync_scope not in {"full", "attention_only"}:
        raise ValueError(f"unknown vLLM sync scope {sync_scope!r}")
    if sleep_level not in {1, 2}:
        raise ValueError(f"unsupported vLLM sleep level {sleep_level!r}")
    original = generation._push_param_to_vllm
    tracker: dict[str, Any] = {
        "skipped_count": 0,
        "skipped_names": set(),
        "disk_reload_suppressed_count": 0,
        "sleep_resync_count": 0,
        "weights_sleeping": False,
        "sync_scope": sync_scope,
        "sleep_level": sleep_level,
        "sync_count": 0,
        "attention_pushed": 0,
        "attention_skipped": 0,
    }
    frozen_prefixes = (
        "vision_tower.",
        "multi_modal_projector.",
        "embed_vision.",
        "embed_audio.",
        "model.vision_tower.",
        "model.multi_modal_projector.",
        "model.embed_vision.",
        "model.embed_audio.",
    )

    def filtered(name: str, parameter: Any) -> Any:
        if name.startswith(frozen_prefixes):
            tracker["skipped_count"] += 1
            tracker["skipped_names"].add(name)
            return None
        return original(name, parameter)

    generation._push_param_to_vllm = filtered
    llm = getattr(generation, "llm", None)
    collective_rpc = getattr(llm, "collective_rpc", None)
    if callable(collective_rpc):
        def preserve_synchronized_weights(method: str, *args: Any, **kwargs: Any) -> Any:
            if method == "reload_weights":
                tracker["disk_reload_suppressed_count"] += 1
                return None
            return collective_rpc(method, *args, **kwargs)

        llm.collective_rpc = preserve_synchronized_weights
    sync_weights = getattr(generation, "sync_weights", None)
    generate = getattr(generation, "generate", None)
    manages_colocated_sleep = (
        getattr(generation, "mode", None) == "colocate"
        and bool(getattr(generation, "enable_sleep_mode", False))
        and callable(sync_weights)
        and callable(generate)
    )
    if manages_colocated_sleep:
        # VLLMGeneration.__init__ has already called sleep(level=2).
        tracker["weights_sleeping"] = True

        def tracked_sync_weights(*args: Any, **kwargs: Any) -> Any:
            result = sync_weights(*args, **kwargs)
            tracker["weights_sleeping"] = False
            return result

        def generate_with_current_weights(*args: Any, **kwargs: Any) -> Any:
            if tracker["weights_sleeping"]:
                tracker["sleep_resync_count"] += 1
                generation.sync_weights()
            result = generate(*args, **kwargs)
            tracker["weights_sleeping"] = True
            return result

        generation.sync_weights = tracked_sync_weights
        generation.generate = generate_with_current_weights

    llm_sleep = getattr(llm, "sleep", None)
    if manages_colocated_sleep and sleep_level == 1 and callable(llm_sleep):
        # TRL hardcodes sleep(level=2) at the end of every generation, which
        # discards the weights and forces a full re-push next update. Level 1
        # offloads them to pinned host RAM instead (~1s restore on wake), so
        # a scoped sync stays sufficient after the first full one.
        def offloading_sleep(level: int = 1, **kwargs: Any) -> Any:
            return llm_sleep(level=1)

        llm.sleep = offloading_sleep

    if sync_scope == "attention_only":
        # Valid only for attention-only LoRA (enforced by the caller): after
        # one complete push has populated every buffer, the merged model can
        # differ from what vLLM holds solely in the q/k/v/o projections.
        # The engine-init sleep was level 2, so the first sync must be full
        # regardless of the level this configuration later enforces.
        scoped_inner = generation._push_param_to_vllm

        def attention_scoped(name: str, parameter: Any) -> Any:
            if tracker["sync_count"] == 0:
                return scoped_inner(name, parameter)
            if ".self_attn." in name and any(
                projection in name
                for projection in (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.")
            ):
                tracker["attention_pushed"] += 1
                return scoped_inner(name, parameter)
            tracker["attention_skipped"] += 1
            return None

        generation._push_param_to_vllm = attention_scoped

    counted_inner = getattr(generation, "sync_weights", None)
    if callable(counted_inner):
        def counted_sync_weights(*args: Any, **kwargs: Any) -> Any:
            result = counted_inner(*args, **kwargs)
            tracker["sync_count"] += 1
            return result

        generation.sync_weights = counted_sync_weights
    return tracker


def configure_group_n_sampling(generation: Any, group_size: int) -> dict[str, Any]:
    """Generate each duplicated prompt group as one vLLM request with n=group.

    TRL's colocate path submits every completion slot as its own request with
    n=1, so the group's identical ~3k-token prompt is prefilled up to
    group_size times and its KV pages are not shared. TRL's own server mode
    dedupes exactly this way ("faster than generating outputs for each
    duplicate prompt individually"); with one process the two are equivalent.
    Falls through untouched whenever the batch is not exact consecutive
    duplicate groups, so it can never mis-group a foreign call.
    """

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    llm = getattr(generation, "llm", None)
    original_generate = getattr(llm, "generate", None)
    tracker: dict[str, Any] = {"grouped_calls": 0, "passthrough_calls": 0}
    if not callable(original_generate):
        return tracker

    def grouped_generate(
        prompts: list, *args: Any, sampling_params: Any = None, **kwargs: Any
    ) -> list:
        identifiers = [
            row.get("prompt_token_ids") if isinstance(row, dict) else None
            for row in prompts
        ]
        groupable = (
            sampling_params is not None
            and getattr(sampling_params, "n", None) == 1
            and len(identifiers) > 0
            and len(identifiers) % group_size == 0
            and all(identifier is not None for identifier in identifiers)
            and all(
                identifiers[index + offset] == identifiers[index]
                for index in range(0, len(identifiers), group_size)
                for offset in range(group_size)
            )
        )
        if not groupable:
            tracker["passthrough_calls"] += 1
            return original_generate(
                prompts, *args, sampling_params=sampling_params, **kwargs
            )
        clone = getattr(sampling_params, "clone", None)
        grouped_params = clone() if callable(clone) else copy.deepcopy(sampling_params)
        grouped_params.n = group_size
        outputs = original_generate(
            prompts[::group_size], *args, sampling_params=grouped_params, **kwargs
        )
        expanded = [
            SimpleNamespace(
                prompt_token_ids=request.prompt_token_ids,
                outputs=[completion],
            )
            for request in outputs
            for completion in request.outputs
        ]
        if len(expanded) != len(prompts):
            raise RuntimeError(
                f"group n-sampling produced {len(expanded)} completions for "
                f"{len(prompts)} prompt slots"
            )
        tracker["grouped_calls"] += 1
        return expanded

    llm.generate = grouped_generate
    return tracker


_PROFILE_LOG_PATH: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "scimt_grpo_profile_log_path", default=None
)


def _record_profile(row: dict[str, Any]) -> None:
    path = _PROFILE_LOG_PATH.get()
    if path is not None:
        _append_jsonl_rows(path, [row])


def install_profile_recorder(path: Path) -> None:
    """Persist TRL profiling spans and per-micro-step timings to a JSONL.

    TRL wraps sync_weights, vLLM generation, reward calls, and the old-logps
    pass in ProfilingContext, but the timings only reach wandb/mlflow/trackio
    — with report_to=() they are silently dropped. Trainer.training_step is
    additionally timed because compute_loss covers only the forward pass; the
    backward (which dominates long-completion updates) is otherwise invisible.
    """

    from trl.extras import profiling

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The wrappers below are process-global, but the destination is contextual:
    # sequential or task-local GRPO calls can therefore select different files
    # without stacking monkey-patches or leaking later spans into the first run.
    _PROFILE_LOG_PATH.set(path)

    if not getattr(profiling.ProfilingContext.__exit__, "_scimt_recorder", False):
        original_exit = profiling.ProfilingContext.__exit__

        def recording_exit(self: Any, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
            if self._start_time is not None:
                _record_profile(
                    {
                        "event": self.name,
                        "seconds": round(time.perf_counter() - self._start_time, 4),
                        "t_end": time.time(),
                    }
                )
            return original_exit(self, exc_type, exc_val, exc_tb)

        recording_exit._scimt_recorder = True
        profiling.ProfilingContext.__exit__ = recording_exit

    import transformers.trainer as hf_trainer

    if not getattr(hf_trainer.Trainer.training_step, "_scimt_recorder", False):
        original_step = hf_trainer.Trainer.training_step

        def recording_step(self: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result = original_step(self, *args, **kwargs)
            _record_profile(
                {
                    "event": "training_step",
                    "seconds": round(time.perf_counter() - started, 4),
                    "t_end": time.time(),
                }
            )
            return result

        recording_step._scimt_recorder = True
        hf_trainer.Trainer.training_step = recording_step


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
                 max_prompt_tokens: int | None = None,
                 *, enable_thinking: bool = False) -> tuple[list[dict[str, Any]], int]:
    prepared: list[dict[str, Any]] = []
    dropped = 0
    for index, original in enumerate(rows):
        if original.get("messages") is not None:
            messages = _validate_messages(original.get("messages"), index)
            try:
                template_kwargs = (
                    {"enable_thinking": True} if enable_thinking else {}
                )
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            except Exception as exc:
                raise ValueError(f"row {index}: chat template rendering failed: {exc}") from exc
            # TRL re-applies chat templates to conversational prompts without
            # forwarding model-specific kwargs. Gemma 4 defaults that second
            # render to direct mode, silently undoing enable_thinking=True.
            # Hand TRL the already-rendered prompt when native thinking was
            # explicitly requested so the audited token sequence is preserved.
            prompt = rendered if enable_thinking else messages
            candidate = {"prompt": prompt,
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


def normalized_spread(successes: float, trials: float) -> float:
    """``4 p (1 - p)`` at ``p = successes / trials``: in [0, 1], max at p = 1/2.

    One function, two callers, deliberately:

    * an RL worklist's per-episode sampling weight evaluates it at the Beta
      posterior-mean pass rate, to bias *which prompts are drawn*;
    * within-batch group selection evaluates it at the observed count ``k`` of
      reward-1 completions, to choose *which generated groups are optimized*.

    Both are the same quantity -- the reward spread a group at that pass rate
    can produce -- measured once on an estimate and once on an observation.
    Writing the formula out twice invites the two halves to drift apart, so
    they share this.

    It is proportional to the gradient a group can actually contribute: under
    ``dr_grpo`` with ``scale_rewards="none"`` the advantage is ``r - mean``, so
    a group of ``n`` with ``k`` ones has total ``|advantage|`` equal to
    ``2 k (n - k) / n``, which is ``(n / 2) * normalized_spread(k, n)``. Since
    ``n`` is fixed within a batch, ranking by either is the same ranking.

    It is NOT the probability that the group has nonzero spread -- that is
    ``1 - p**n - (1-p)**n``. The two are symmetric about ``p = 1/2`` and
    increasing on ``[0, 1/2]``, so they order groups identically; this one is
    preferred because it stays sensitive near the extremes and keeps the group
    size out of the formula.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} outside [0, {trials}]")
    rate = successes / trials
    return 4.0 * rate * (1.0 - rate)


def group_spread_scores(rewards: list[float], *, group_size: int) -> list[float]:
    """Per-group selection scores, highest = most gradient available.

    Only meaningful for rewards in [0, 1]; a reward outside that range is a
    loud error rather than a silently meaningless ranking, because the score
    reduces to the exact ``k (n - k)`` gradient mass only for a binary reward.
    """

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rewards) % group_size:
        raise ValueError(
            f"{len(rewards)} rewards is not a whole number of {group_size}-groups"
        )
    if any(not 0.0 <= float(value) <= 1.0 for value in rewards):
        raise ValueError(
            "group selection ranks by 4 p (1 - p) and needs rewards in [0, 1]"
        )
    return [
        normalized_spread(sum(rewards[i:i + group_size]), group_size)
        for i in range(0, len(rewards), group_size)
    ]


def select_group_indices(
    rewards: list[float], *, group_size: int, keep_groups: int
) -> tuple[int, ...]:
    """Row indices of the ``keep_groups`` most informative generated groups.

    Plain top-k, never a resample loop. If fewer than ``keep_groups`` groups
    have any spread, the remainder is filled with the best of the rest -- which
    top-k does for free. That is a deliberate departure from DAPO's unbounded
    regeneration: cost per update stays constant and the geometry is preserved,
    and the bad case degrades to the un-selected behaviour instead of a tail
    nobody budgeted for. Wall-clock predictability is worth more here than
    filling every slot with a nonzero-gradient group.

    Ties break on group index, so the result is a pure function of the reward
    vector: no RNG, no set or dict iteration order.
    """

    scores = group_spread_scores(rewards, group_size=group_size)
    if keep_groups <= 0:
        raise ValueError("keep_groups must be positive")
    if keep_groups > len(scores):
        raise ValueError(
            f"cannot keep {keep_groups} of {len(scores)} generated groups"
        )
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    kept = sorted(ranked[:keep_groups])
    return tuple(
        group * group_size + offset for group in kept for offset in range(group_size)
    )


def selection_report(
    rewards: list[float], *, group_size: int, keep_groups: int
) -> dict[str, Any]:
    """Audit record for one selection round.

    The kept subset depends on sampled completions, which vLLM does not
    reproduce across a process restart, so the realized stream is recorded
    rather than re-derived. Joins to ``raw_rollouts.jsonl`` on ``reward_call``.
    """

    scores = group_spread_scores(rewards, group_size=group_size)
    rows = select_group_indices(
        rewards, group_size=group_size, keep_groups=keep_groups
    )
    kept = sorted({row // group_size for row in rows})
    successes = [
        sum(rewards[i:i + group_size]) for i in range(0, len(rewards), group_size)
    ]
    selected = [rewards[row] for row in rows]
    return {
        "generated_groups": len(scores),
        "kept_groups": kept,
        "group_successes": successes,
        "group_scores": scores,
        "zero_std_fraction": zero_std_group_fraction(rewards, group_size=group_size),
        "selected_zero_std_fraction": zero_std_group_fraction(
            selected, group_size=group_size
        ),
        "kept_rows": list(rows),
    }


class AbortGate:
    """Two-window online safety gate with an append-only decision trail."""

    def __init__(self, path: Path, *, parent_agreement: float, parent_reward: float,
                 parent_completion_length: float, expected_episodes: int,
                 zero_std_warmup_fraction: float = 0.10,
                 truncation_rate: float = 0.05,
                 heldout_eval_armed: bool = True) -> None:
        self.path = Path(path)
        self.parent_agreement = parent_agreement
        self.parent_reward = parent_reward
        self.parent_completion_length = parent_completion_length
        self.expected_episodes = expected_episodes
        self.zero_std_warmup_fraction = zero_std_warmup_fraction
        # What counts as runaway truncation is a property of the generation
        # budget, not a constant. A 512-token direct cell truncating 5% is
        # sick; a 4096-token thinking cell truncates ~31% by design against a
        # 50% stop, so the old hardcoded 0.05 would have aborted every thinking
        # run within two logs -- which is why arming the gate at all needed
        # this knob first.
        self.truncation_rate = truncation_rate
        # Whether the held-out evaluator is wired up. Without it the
        # reward_rise_agreement_drop check cannot fire (external agreement
        # defaults to the parent's), so the gate is running a strict subset of
        # its checks. Recorded on every row: a decision trail that does not say
        # which checks were live invites reading silence as safety.
        self.heldout_eval_armed = heldout_eval_armed
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
        if float(metrics.get("truncation_rate", 0)) > self.truncation_rate:
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
               "heldout_eval_armed": self.heldout_eval_armed,
               "truncation_rate_limit": self.truncation_rate,
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
                     completion_decoder: Callable[[Any], str] | None = None,
                     max_completion_length: int | None = None,
                     completion_length_window: int = 1024,
                     pass_completion_truncated: bool = False) -> Callable[..., list[float]]:
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
            raw_text = None
            completion_ids = untouched.get("completion_ids")
            if completion_ids is not None:
                length = len(completion_ids)
            else:
                length = completion_length(text) if completion_length else len(text)
            truncated = bool(
                max_completion_length and length >= max_completion_length
            )
            if completion_decoder is not None:
                if completion_ids is None:
                    raise ValueError(
                        "completion_decoder requires TRL completion_ids"
                    )
                raw_text = completion_decoder(completion_ids)
            score_columns = dict(untouched)
            if completion_decoder is not None:
                score_columns["completion_raw_text"] = raw_text
            # Reward code must be able to fail closed on truncation. Merely
            # masking the truncated completion's loss is insufficient because
            # its reward still changes group normalization/other advantages.
            if pass_completion_truncated:
                score_columns["completion_truncated"] = truncated
            scored = score(text, **score_columns)
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
            component_rows.append({"prompt": prompts[index], "completion": text,
                "completion_raw_text": raw_text,
                **_loggable(untouched), **numeric_components,
                "semantic_correct": components.get("semantic_correct"),
                "format_valid": components.get("format_valid"), "reward": scalar,
                "reward_call": reward_func.reward_calls,
                "completion_length": length,
                "truncated": truncated})
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
        # Handed to within-batch group selection, which runs immediately after
        # this call on the same generation batch. Keeping the exact vector TRL
        # scored means selection can never rank on a differently-derived
        # number than the one that produced the advantages.
        reward_func.last_rewards = list(result)
        return result
    reward_func.last_zero_std_group_fraction = 0.0
    reward_func.zero_std_groups = 0
    reward_func.total_groups = 0
    reward_func.last_rewards = []
    # Post-selection counterparts, written by the group-selection trainer when
    # oversampling is on. They are a SEPARATE series: the pre-selection numbers
    # above stay the definition of `zero_std_group_fraction`, because that is
    # what the abort gate watches and what earlier runs are compared against.
    reward_func.last_selected_zero_std_group_fraction = 0.0
    reward_func.selected_zero_std_groups = 0
    reward_func.selected_total_groups = 0
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


#: Reward-function kwargs that must never be PERSISTED into a rollout record.
#: They are still passed to ``score`` -- this only governs what is written.
#:
#: TRL hands the reward function ``trainer_state`` so a reward can be
#: step-aware, and the rollout row splatted every column it was given. That
#: state carries the trainer's ACCUMULATED log history, so it grows with every
#: step and each record embeds the whole thing: measured on one 768-update
#: cell, the first record was 65.5 KB (87.5% trainer_state), the middle 958 KB
#: (99.1%), the last 1,347 KB (99.6%) -- 47,104 records, 34.2 GB, of which the
#: actual rollout content was ~330 MB. Quadratic in the number of updates.
#:
#: It also broke durability: a 34 GB file still growing between 20-minute Hub
#: mirror cycles times out the Xet commit, and the plain-LFS fallback has the
#: commit rejected outright. The authoritative copy of this state is each
#: checkpoint's own trainer_state.json, which is saved whole.
UNLOGGED_REWARD_COLUMNS = frozenset({"trainer_state"})

#: Serialized size above which a single persisted column is reported once, so
#: the NEXT field like trainer_state is noticed while the file is small rather
#: than at teardown. Not a filter: nothing is dropped on size alone, because a
#: silent size-based drop would lose real data without anyone knowing.
_LARGE_COLUMN_BYTES = 65_536
_reported_large_columns: set[str] = set()


def _loggable(columns: dict[str, Any]) -> dict[str, Any]:
    """Columns to persist in a rollout record; see UNLOGGED_REWARD_COLUMNS."""
    kept = {key: value for key, value in columns.items()
            if key not in UNLOGGED_REWARD_COLUMNS}
    for key, value in kept.items():
        if key in _reported_large_columns:
            continue
        try:
            size = len(json.dumps(value, default=str))
        except (TypeError, ValueError):
            continue
        if size > _LARGE_COLUMN_BYTES:
            _reported_large_columns.add(key)
            warnings.warn(
                f"rollout column {key!r} serializes to {size / 1024:.0f} KB per "
                "record; if it grows with training step it will dominate "
                "raw_rollouts.jsonl. Add it to "
                "scimt.train.grpo.UNLOGGED_REWARD_COLUMNS if it is not "
                "per-rollout data.",
                RuntimeWarning, stacklevel=2)
    return kept


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
            # PRE-selection, always. When oversampling is on this is the rate
            # over every generated group, which is the series the abort gate
            # watches and the one earlier runs' numbers mean. The post-selection
            # rate is reported beside it under a distinct key, never in place
            # of it: a gate that saw only what selection kept would read healthy
            # while the policy collapsed.
            enriched["reward/zero_std_group_fraction"] = (
                reward_func.zero_std_groups / reward_func.total_groups
                if reward_func.total_groups
                else 0.0
            )
            if getattr(reward_func, "selected_total_groups", 0):
                enriched["reward/selected_zero_std_group_fraction"] = (
                    reward_func.selected_zero_std_groups
                    / reward_func.selected_total_groups
                )
            for name, value in reward_func.latest_components.items():
                enriched[f"reward_components/{name}"] = value
            return super().log(enriched, *args, **kwargs)

    RewardMetricTrainer.__name__ = f"RewardMetric{trainer_cls.__name__}"
    return RewardMetricTrainer


#: TRL internals the group-selection trainer reuses. They are private-ish, so
#: the mixin imports them by name and fails loudly if a TRL upgrade moved them,
#: rather than quietly training a different geometry.
_TRL_SELECTION_HELPERS = (
    "split_tensor_dict",
    "shuffle_sequence_dict",
    "split_pixel_values_by_grid",
    "unsplit_pixel_values_by_grid",
    "RepeatSampler",
)


def _trl_selection_helpers() -> dict[str, Any]:
    try:
        from trl.trainer import utils as trl_utils
    except ImportError as exc:  # pragma: no cover - GPU runtime only
        raise ModelCompatError(
            "GRPO group selection needs trl; install the GRPO runtime "
            f"dependencies (missing: {exc.name})"
        ) from exc
    missing = [n for n in _TRL_SELECTION_HELPERS if not hasattr(trl_utils, n)]
    if missing:
        raise ModelCompatError(
            "installed TRL does not expose "
            f"{missing} -- GRPO oversample-and-select was built against the "
            "TRL 1.9.2 generation loop and must be re-verified before use"
        )
    helpers = {name: getattr(trl_utils, name) for name in _TRL_SELECTION_HELPERS}
    # TRL decorates its own _prepare_inputs, and the study's profile analysis
    # reads the resulting span. Overriding the method without the decorator
    # would silently delete `_prepare_inputs` from every profile receipt.
    try:
        from trl.extras.profiling import profiling_decorator
    except ImportError:  # pragma: no cover - a different TRL layout
        profiling_decorator = None
    helpers["profiling_decorator"] = profiling_decorator
    return helpers


def select_batch_rows(
    batch: dict[str, Any], rows: tuple[int, ...], *, total: int
) -> dict[str, Any]:
    """Keep ``rows`` of every per-sequence entry, pass everything else through.

    Entries whose leading dimension is not the generation-batch size are batch
    scalars (``num_items_in_batch``) or per-grid side tables, and slicing them
    by sequence index would be wrong.
    """

    selected: dict[str, Any] = {}
    for key, value in batch.items():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) == total:
            selected[key] = value[list(rows)]
        elif isinstance(value, list) and len(value) == total:
            selected[key] = [value[index] for index in rows]
        else:
            selected[key] = value
    return selected


def trainer_with_group_selection(
    trainer_cls: Any,
    reward_func: Any,
    *,
    group_size: int,
    keep_groups: int,
    oversample_factor: int,
    log_path: Path | None = None,
) -> Any:
    """Generate ``oversample_factor`` x the groups, optimize the best ones.

    TRL 1.9.2 cannot express this natively. ``GRPOConfig`` derives
    ``generation_batch_size = per_device_train_batch_size * num_processes *
    steps_per_generation`` unconditionally, and ``get_train_dataloader`` fetches
    exactly ``per_device_train_batch_size * steps_per_generation`` rows per
    generation round -- so requiring (a) one optimizer step per generation
    round, (b) a 32-completion optimizer batch and (c) a 64-completion
    generation batch is three equations TRL leaves no free variable for.
    Everything TRL generates, TRL optimizes.

    So three seams are overridden, and nothing else:

    * ``get_train_dataloader`` fetches ``oversample_factor`` x rows (by
      inflating the batch size around the call, not by copying the method);
    * ``_get_train_sampler`` lays out ``oversample_factor`` x unique prompts per
      generation round, keeping TRL's own repeat/shuffle/seed semantics;
    * ``_prepare_inputs`` selects the kept groups between generation and the
      buffered split, so discarded groups never reach a forward or backward
      pass. That is the whole point: generation doubles, training does not.

    The optimizer batch is unchanged -- ``keep_groups * group_size`` rows split
    into ``steps_per_generation`` micro-batches of ``per_device_train_batch_size``
    -- so ``dr_grpo``'s ``per_token_loss.size(0) * max_completion_length``
    normalizer and the accumulation count are exactly what they were. No
    effective learning-rate change.
    """

    helpers = _trl_selection_helpers()
    split_tensor_dict = helpers["split_tensor_dict"]
    shuffle_sequence_dict = helpers["shuffle_sequence_dict"]
    split_pixel_values_by_grid = helpers["split_pixel_values_by_grid"]
    unsplit_pixel_values_by_grid = helpers["unsplit_pixel_values_by_grid"]
    repeat_sampler_cls = helpers["RepeatSampler"]
    profiling_decorator = helpers["profiling_decorator"]

    class GroupSelectingTrainer(trainer_cls):
        def get_train_dataloader(self) -> Any:
            original = self._train_batch_size
            self._train_batch_size = original * oversample_factor
            try:
                return super().get_train_dataloader()
            finally:
                self._train_batch_size = original

        def _get_train_sampler(self, dataset: Any = None) -> Any:
            return repeat_sampler_cls(
                data_source=self.train_dataset if dataset is None else dataset,
                mini_repeat_count=self.num_generations,
                batch_size=(
                    oversample_factor
                    * self.args.generation_batch_size
                    // self.num_generations
                ),
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )

        def _select_generated_groups(self, scored: dict[str, Any]) -> dict[str, Any]:
            rewards = list(getattr(reward_func, "last_rewards", ()))
            total = int(scored["advantages"].shape[0])
            if len(rewards) != total:
                raise RuntimeError(
                    f"group selection saw {len(rewards)} rewards for a "
                    f"{total}-row generation batch; the reward function must be "
                    "called once per generation round on the whole batch"
                )
            report = selection_report(
                rewards, group_size=group_size, keep_groups=keep_groups
            )
            reward_func.last_selected_zero_std_group_fraction = report[
                "selected_zero_std_fraction"
            ]
            reward_func.selected_zero_std_groups += round(
                report["selected_zero_std_fraction"] * keep_groups
            )
            reward_func.selected_total_groups += keep_groups
            if log_path is not None:
                _append_jsonl_rows(log_path, [{
                    "global_step": int(self.state.global_step),
                    # Joins to raw_rollouts.jsonl, which carries episode_id.
                    "reward_call": int(getattr(reward_func, "reward_calls", 0)) - 1,
                    **report,
                }])
            return select_batch_rows(
                scored, tuple(report["kept_rows"]), total=total
            )

        def _prepare_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
            # Mirrors TRL 1.9.2 GRPOTrainer._prepare_inputs with one insertion:
            # selection between generation and the buffered split.
            if not self.model.training:
                return super()._prepare_inputs(generation_batch)
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                scored = self._generate_and_score_completions(generation_batch)
                scored = self._select_generated_groups(scored)
                scored = split_pixel_values_by_grid(scored)
                scored = shuffle_sequence_dict(scored)
                self._buffered_inputs = [
                    unsplit_pixel_values_by_grid(batch)
                    for batch in split_tensor_dict(
                        scored, self.args.steps_per_generation
                    )
                ]
            return self._buffered_inputs[self._step % self.args.steps_per_generation]

    if profiling_decorator is not None:
        GroupSelectingTrainer._prepare_inputs = profiling_decorator(
            GroupSelectingTrainer._prepare_inputs
        )
    GroupSelectingTrainer.__name__ = f"GroupSelecting{trainer_cls.__name__}"
    return GroupSelectingTrainer


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


def align_eos_with_turn_terminator(tokenizer: Any, weights: str) -> int | None:
    """Point ``tokenizer.eos_token_id`` at the id the model actually ends turns with.

    TRL decides whether a rollout terminated with a SCALAR comparison --
    ``is_eos = completion_ids == tokenizer.eos_token_id`` (grpo_trainer.py) -- but a
    chat-tuned Gemma-3 ends its turn with ``<end_of_turn>`` (106), and Gemma 4
    Unified uses ``<turn|>``, while the tokenizer's scalar ``eos_token_id`` can
    still point at ``<eos>``. The model's own ``generation_config`` lists the
    valid stop ids, so vLLM stops correctly but TRL never sees its scalar id.

    Consequence, observed: every completion is classified unterminated
    (``completions/clipped_ratio == 1.0``), and with
    ``mask_truncated_completions=True`` every token is masked out -- so the loss is
    empty, ``grad_norm`` is exactly 0.0 at every step, and LoRA's B matrices stay at
    their zero init. Four RL runs trained on nothing and looked merely
    "flat"; jonathan's working config pins ``end_of_turn_id: 106`` explicitly and
    guards it with a stop-token smoke test, which this had never ported.

    Only ever narrows eos to a terminator the model's generation_config declares,
    so it cannot invent a stop token. Returns the new id, or None if unchanged.
    """
    # Read the JSON directly rather than via GenerationConfig.from_pretrained:
    # that call can fail for reasons unrelated to eos (missing config.json, remote
    # code), and catching it broadly is what hid this bug in the first place.
    config_path = Path(weights) / "generation_config.json"
    if not config_path.is_file():
        return None
    try:
        generation_ids = json.loads(config_path.read_text()).get("eos_token_id")
    except json.JSONDecodeError as exc:
        raise ValueError(f"unreadable generation_config.json at {config_path}") from exc
    if not isinstance(generation_ids, (list, tuple)) or len(generation_ids) < 2:
        return None
    terminator = None
    turn_id = None
    for candidate in ("<end_of_turn>", "<turn|>"):
        candidate_id = tokenizer.convert_tokens_to_ids(candidate)
        if (
            candidate_id is not None
            and candidate_id >= 0
            and candidate_id in generation_ids
            and candidate_id != tokenizer.eos_token_id
        ):
            terminator, turn_id = candidate, candidate_id
            break
    if turn_id is None:
        return None
    previous = tokenizer.eos_token_id
    tokenizer.eos_token_id = turn_id
    logger.warning(
        "GRPO: eos_token_id %s -> %s (%s); the model's generation_config "
        "declares %s and TRL tests termination against a single id, so leaving it "
        "at %s marks every rollout truncated and zeroes the gradient",
        previous, turn_id, terminator, list(generation_ids), previous,
    )
    return turn_id


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
        align_eos_with_turn_terminator(tokenizer, weights)

        rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        prepared, dropped = prepare_rows(
            rows,
            tokenizer,
            opts.max_prompt_length,
            enable_thinking=opts.enable_thinking,
        )
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
        for module_name in (
            "vision_tower",
            "multi_modal_projector",
            "embed_vision",
            "embed_audio",
        ):
            module = getattr(model_root, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if opts.profile_log_path is not None:
            install_profile_recorder(Path(opts.profile_log_path))
        if opts.vllm_sync_scope == "attention_only" and (
            cfg.lora is None or cfg.lora.target_policy != "attention_only"
        ):
            # The scoped push is only sound when nothing outside the
            # attention projections can ever change in the merged model.
            raise ValueError(
                "grpo.vllm_sync_scope='attention_only' requires an "
                "attention-only LoRA target policy"
            )
        peft_config = None
        lora_targets: tuple[str, ...] = ()
        if cfg.lora is not None:
            require_supported_lora_world_size(world_size)
            if cfg.lora.target_modules is not None:
                raise ValueError(
                    "hf_grpo discovers exact text-only Gemma LoRA targets; "
                    "explicit lora.target_modules is unsupported"
                )
            if cfg.lora.target_parameters is not None:
                raise ValueError(
                    "hf_grpo does not support lora.target_parameters "
                    "(MoE expert tensors) — silently dropping it would train "
                    "a different adapter than configured"
                )
            try:
                from peft import LoraConfig as PeftLoraConfig
            except ImportError as exc:
                raise ModelCompatError(
                    "hf_grpo LoRA needs peft; install the GRPO runtime dependencies "
                    f"(missing: {exc.name})"
                ) from exc
            lora_targets = discover_language_lora_targets(
                model, policy=cfg.lora.target_policy
            )
            if cfg.lora.initial_adapter_path is not None:
                model = load_initial_lora_adapter(
                    model, cfg.lora.initial_adapter_path, cfg.lora, lora_targets
                )
            else:
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
        keep_groups = (
            opts.per_device_batch_size * world_size * generation_steps
        ) // opts.group_size
        if opts.oversample_factor > 1 and world_size != 1:
            # TRL shards each generation group across ranks before scoring, so
            # a rank does not see whole groups and cannot rank them. Selecting
            # per-rank on partial groups would optimize a different subset on
            # every GPU. The six RL cells are one GPU each; making this work
            # under FSDP needs a cross-rank gather, not a silent per-rank guess.
            raise ValueError(
                "GRPO oversample_factor > 1 is single-process only "
                f"(WORLD_SIZE={world_size}); groups are sharded across ranks"
            )
        saves = checkpoint_steps(max_steps, opts.checkpoint_fractions)

        class FractionalCheckpointCallback(TrainerCallback):
            """Ask Trainer to checkpoint at exact fractional dose boundaries."""
            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                if state.global_step in saves:
                    control.should_save = True
                return control

        checkpoint_sync = (
            resolve_reward_func(opts.checkpoint_sync_func)
            if opts.checkpoint_sync_func else None
        )

        class CheckpointSyncCallback(TrainerCallback):
            """Copy each checkpoint off the pod as soon as Trainer writes it.

            A pod's disk dies with the pod, so an unsynced checkpoint is not a
            backup of anything. Rank 0 only -- every rank saves, but they save
            the same bytes, and eight concurrent 550 MB uploads would be eight
            times the traffic for one copy.

            Deliberately synchronous and deliberately non-fatal. Synchronous
            because a background upload racing the next save is a subtle way to
            ship a half-written checkpoint, and the cost is small next to a
            64-update interval. Non-fatal because the sync exists to protect a
            run that is going fine -- aborting that run because a network blip
            lost its backup would cause the exact loss it prevents.
            """
            def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
                if checkpoint_sync is None or not state.is_world_process_zero:
                    return control
                path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                try:
                    checkpoint_sync(path)
                except Exception as error:  # noqa: BLE001 - advisory by design
                    logger.warning(
                        "GRPO: checkpoint sync FAILED for %s: %r. Training "
                        "continues, but this checkpoint exists ONLY on this pod "
                        "-- do not delete the pod until it is copied off.",
                        path, error,
                    )
                return control

        reward_function = make_reward_func(
            resolve_reward_func(opts.reward_func), group_size=opts.group_size,
            rollout_log_dir=Path(opts.rollout_log_dir) if opts.rollout_log_dir else None,
            completion_length=lambda text: len(tokenizer(text)["input_ids"]),
            completion_decoder=lambda ids: tokenizer.decode(
                ids, skip_special_tokens=False
            ),
            max_completion_length=opts.max_completion_length,
            completion_length_window=opts.completion_length_window,
            pass_completion_truncated=True)
        abort_gate = None
        abort_evaluator = None
        if opts.abort_log_path is not None:
            abort_gate = AbortGate(
                Path(opts.abort_log_path), parent_agreement=float(opts.parent_agreement),
                parent_reward=float(opts.parent_reward),
                parent_completion_length=float(opts.parent_completion_length),
                expected_episodes=opts.episodes,
                zero_std_warmup_fraction=opts.zero_std_warmup_fraction,
                truncation_rate=opts.abort_truncation_rate,
                heldout_eval_armed=opts.abort_eval_func is not None)
            if opts.abort_eval_func is not None:
                abort_evaluator = resolve_reward_func(str(opts.abort_eval_func))

        class EmptyGradientCallback(TrainerCallback):
            """Raise when NO gradient has ever reached the adapter.

            ``mask_truncated_completions`` drops unterminated rollouts from the
            loss, so if TRL's termination test never fires -- e.g. its scalar
            ``eos_token_id`` is not the id the model ends turns with -- then
            ``clipped_ratio`` is 1.0, the whole batch is masked, and ``grad_norm``
            is exactly 0.0 at every step. That is indistinguishable from "RL didn't
            work" in the metrics and cost four runs before anyone read grad_norm.

            A single zero-gradient step is NOT that failure, though: once the
            policy is good, every group in a step can be all-correct, giving zero
            advantage and legitimately zero gradient. The first version of this
            guard raised on any zero and killed a healthy run at step 40 with
            ``clipped_ratio=0.0``. So the gradient check fires only if no logged
            step has EVER had a non-zero gradient.
            """

            def __init__(self) -> None:
                self.seen_gradient = False
                self.zero_logs = 0

            def on_log(self, args: Any, state: Any, control: Any,
                       logs: dict[str, float] | None = None, **kwargs: Any) -> Any:
                if logs is None:
                    return control
                # unambiguous: every completion masked out, nothing to learn from
                clipped = logs.get("completions/clipped_ratio")
                if (clipped is not None and clipped >= 1.0
                        and opts.mask_truncated_completions
                        and state.global_step > args.logging_steps):
                    raise ValueError(
                        f"step {state.global_step}: completions/clipped_ratio="
                        f"{clipped} with mask_truncated_completions=True, so every "
                        "token is masked and the loss is empty. TRL tests "
                        "termination against a single eos_token_id; check it is the "
                        "id this model ends turns with (Gemma-3 chat: <end_of_turn>)"
                    )
                grad_norm = logs.get("grad_norm")
                if grad_norm is None:
                    return control
                if grad_norm > 0:
                    self.seen_gradient = True
                    return control
                self.zero_logs += 1
                if (
                    not self.seen_gradient
                    and self.zero_logs >= opts.zero_gradient_abort_logs
                ):
                    raise ValueError(
                        f"step {state.global_step}: grad_norm has been exactly 0.0 "
                        f"for {self.zero_logs} logged steps (configured limit "
                        f"{opts.zero_gradient_abort_logs}) and never non-zero, so "
                        "no gradient has reached the adapter and this run cannot "
                        f"learn. completions/clipped_ratio={clipped}, "
                        f"reward_std={logs.get('reward_std')}"
                    )
                return control

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
                if (abort_evaluator is not None
                        and reward_function.latest_reward
                        - abort_gate.parent_reward >= 0.15 - 1e-12):
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
                    # PRE-selection, and it must stay that way. This is the
                    # series the zero-std-collapse check fires on; feeding it
                    # the post-selection rate would let within-batch selection
                    # mask exactly the collapse the gate exists to catch --
                    # selection keeps the best 4 of 8 whatever the policy is
                    # doing, so the gate would read healthy while the policy
                    # died. The post-selection rate is recorded beside it and
                    # nothing gates on it.
                    "zero_std_fraction": reward_function.last_zero_std_group_fraction,
                    "selected_zero_std_fraction": getattr(
                        reward_function, "last_selected_zero_std_group_fraction", 0.0
                    ),
                    "oversample_factor": opts.oversample_factor,
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
            lora_training_args["disable_dropout"] = grpo_disable_dropout(cfg.lora)
        args = GRPOConfig(
            output_dir=str(out_dir / "trainer"), max_steps=max_steps,
            per_device_train_batch_size=opts.per_device_batch_size,
            gradient_accumulation_steps=opts.gradient_accumulation_steps,
            steps_per_generation=generation_steps, num_generations=opts.group_size,
            max_completion_length=opts.max_completion_length,
            learning_rate=opts.learning_rate,
            lr_scheduler_type=opts.lr_scheduler_type,
            warmup_ratio=opts.warmup_ratio,
            temperature=opts.temperature, top_p=opts.top_p, top_k=opts.top_k,
            loss_type=opts.loss_type, scale_rewards=opts.scale_rewards,
            epsilon=opts.epsilon, epsilon_high=opts.epsilon_high, beta=opts.beta,
            mask_truncated_completions=opts.mask_truncated_completions,
            log_completions=opts.log_completions,
            num_completions_to_print=opts.num_completions_to_print,
            log_unique_prompts=opts.log_unique_prompts,
            logging_steps=opts.logging_steps,
            logging_first_step=opts.logging_first_step,
            use_vllm=_resolve_vllm(opts.vllm, use_cuda), vllm_mode="colocate",
            vllm_gpu_memory_utilization=opts.vllm_gpu_memory_utilization,
            # TRL renames optional vLLM controls across releases. Forward only
            # the exact names declared by the installed config class.
            **grpo_optional_kwargs(GRPOConfig, opts),
            remove_unused_columns=False, report_to=list(opts.report_to),
            run_name=run_name, seed=cfg.seed, data_seed=cfg.seed,
            ignore_data_skip=opts.ignore_data_skip,
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
        if opts.oversample_factor > 1:
            trainer_cls = trainer_with_group_selection(
                trainer_cls,
                reward_function,
                group_size=opts.group_size,
                keep_groups=keep_groups,
                oversample_factor=opts.oversample_factor,
                log_path=(
                    Path(opts.rollout_log_dir)
                    / f"selection.rank-{os.environ.get('RANK', '0')}.jsonl"
                    if opts.rollout_log_dir else None
                ),
            )
        trainer = trainer_cls(
            model=model,
            reward_funcs=reward_function,
            args=args,
            train_dataset=dataset,
            processing_class=processor,
            callbacks=[
                FractionalCheckpointCallback(), CheckpointSyncCallback(),
                EmptyGradientCallback(), OnlineAbortCallback(),
            ],
            **trainer_kwargs,
        )
        vllm_sync_tracker = None
        group_sampling_tracker = None
        if cfg.lora is not None and getattr(trainer, "use_vllm", False):
            generation = getattr(trainer, "vllm_generation", None)
            if generation is None:
                raise RuntimeError("LoRA GRPO requested vLLM but no generation engine exists")
            vllm_sync_tracker = configure_lora_vllm_sync(
                generation,
                sync_scope=opts.vllm_sync_scope,
                sleep_level=opts.vllm_sleep_level,
            )
            if opts.vllm_group_n_sampling:
                group_sampling_tracker = configure_group_n_sampling(
                    generation, opts.group_size
                )
        lora_manifest = None
        if cfg.lora is not None:
            layer_count = language_lora_layer_count(lora_targets)
            lora_manifest = {
                **lora_trainable_manifest(
                    trainer.model,
                    target_count=len(lora_targets),
                    layer_count=layer_count,
                ),
                "rank": cfg.lora.r,
                "alpha": cfg.lora.resolved_alpha,
                "dropout": cfg.lora.dropout,
                "target_policy": cfg.lora.target_policy,
                "initial_adapter_path": cfg.lora.initial_adapter_path,
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
            lora_manifest["vllm_disk_reload_suppressed_count"] = int(
                vllm_sync_tracker["disk_reload_suppressed_count"]
            )
            lora_manifest["vllm_sleep_resync_count"] = int(
                vllm_sync_tracker["sleep_resync_count"]
            )
            lora_manifest["vllm_sync_scope"] = vllm_sync_tracker["sync_scope"]
            lora_manifest["vllm_sleep_level"] = int(vllm_sync_tracker["sleep_level"])
            lora_manifest["vllm_sync_count"] = int(vllm_sync_tracker["sync_count"])
            lora_manifest["vllm_attention_pushed"] = int(
                vllm_sync_tracker["attention_pushed"]
            )
            lora_manifest["vllm_attention_skipped"] = int(
                vllm_sync_tracker["attention_skipped"]
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
                "oversample_factor": opts.oversample_factor,
                "generated_groups_per_step": keep_groups * opts.oversample_factor,
                "optimized_groups_per_step": keep_groups,
                "generated_completions": (max_steps * opts.per_device_batch_size
                    * opts.gradient_accumulation_steps * world_size
                    * opts.oversample_factor),
                "checkpoint_steps": saves,
                "dropped_overlong": dropped,
                "parameterization": "lora" if cfg.lora is not None else "full",
                "lora_manifest": lora_manifest,
                "vllm_group_n_sampling": group_sampling_tracker,
                # Over every GENERATED group, comparable to runs without
                # selection. The optimized-only rate is the separate key below.
                "zero_std_group_fraction": (reward_function.zero_std_groups
                    / reward_function.total_groups if reward_function.total_groups else 0.0),
                "selected_zero_std_group_fraction": (
                    reward_function.selected_zero_std_groups
                    / reward_function.selected_total_groups
                    if reward_function.selected_total_groups else None),
            }, indent=2))
        ckpt = self._checkpoint(out_dir, run_name, final_state)
        if trainer.is_world_process_zero():
            with (out_dir / "checkpoints.jsonl").open("a") as handle:
                handle.write(json.dumps({"name": run_name, "backend": self.name,
                                         "sampler_path": ckpt.sampler,
                                         "state_path": ckpt.state}) + "\n")
        return ckpt
