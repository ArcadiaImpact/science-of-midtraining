"""Instrumented throughput probe over run_rl_cell — diagnostic only.

Runs one short RL cell (default 3 updates) on any parent checkpoint with
additive monkey-patches:

- TRL ``ProfilingContext`` exits are appended to ``<output>.profile.jsonl``
  (TRL otherwise drops them when report_to is empty), with CUDA memory
  high-water marks per event.
- Optional geometry override: ``per_device_batch_size`` (grad-accum and
  steps_per_generation derived so the 32-completion generation batch and
  32-completion optimizer batch are preserved exactly).
- Optional ``vllm_gpu_memory_utilization`` override.
- Optional ``sleep=level1`` (offload weights to host RAM instead of
  discarding) or ``sleep=off``.
- Optional ``dedupe_n_sampling``: collapse the group's 8 duplicated prompts
  into one vLLM request with n=8 (TRL's own server-mode strategy), expanding
  outputs back to TRL's expected per-completion shape.
- Optional ``attention_only_sync``: after the first (full) weight sync, push
  only ``self_attn.{q,k,v,o}_proj`` tensors — the only tensors a merged
  attention-only LoRA can change.

Probe outputs never parent a scientific run; the RL_DONE.json they produce is
kept only as a timing receipt.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .. import contracts as C
from .. import run_rl_cell

_SLEEP_MODES = ("level2", "level1", "off")


@dataclass
class Config:
    mode: str = ""
    parent_model: str = ""
    data: str = ""
    output: str = ""
    target_updates: int = 3
    per_device_batch_size: int = 0  # 0 = keep the planned value (1)
    vllm_gpu_memory_utilization: float = 0.0  # 0 = keep the planned 0.40
    sleep: str = "level2"
    dedupe_n_sampling: bool = False
    attention_only_sync: bool = False
    gradient_checkpointing: bool = True
    max_completion_length: int = 0  # 0 = keep the planned per-mode cap
    resume_from_checkpoint: str = ""  # exercise the production resume path

    def __post_init__(self) -> None:
        if self.sleep is False:  # YAML 1.1 parses bare `off` as boolean False
            self.sleep = "off"
        if self.sleep not in _SLEEP_MODES:
            raise ValueError(f"sleep must be one of {_SLEEP_MODES}")
        if self.per_device_batch_size:
            if (
                self.per_device_batch_size < 1
                or C.RL_GLOBAL_BATCH % self.per_device_batch_size
            ):
                raise ValueError(
                    "per_device_batch_size must divide "
                    f"RL_GLOBAL_BATCH={C.RL_GLOBAL_BATCH}"
                )
        if not 0 <= self.vllm_gpu_memory_utilization < 0.75:
            raise ValueError("vllm_gpu_memory_utilization must be in [0, 0.75)")
        if self.sleep == "off" and self.vllm_gpu_memory_utilization > 0.45:
            raise ValueError(
                "without sleep mode the trainer and the full vLLM pool are "
                "co-resident; utilization above 0.45 cannot fit two ~49GiB "
                "weight copies plus activations on 141GB"
            )

    @property
    def derived_accum(self) -> int:
        pdbs = self.per_device_batch_size or 1
        return C.RL_GLOBAL_BATCH // pdbs


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _cuda_memory() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "mem_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "mem_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
            "mem_max_reserved_gib": round(
                torch.cuda.max_memory_reserved() / 2**30, 2
            ),
        }
    except Exception:  # never let telemetry kill a probe
        return {}


def patch_profiling(profile_path: Path) -> None:
    """Persist every TRL ProfilingContext exit to a JSONL file."""

    from trl.extras import profiling

    original_exit = profiling.ProfilingContext.__exit__

    def recording_exit(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        if self._start_time is not None:
            duration = time.perf_counter() - self._start_time
            _append(
                profile_path,
                {
                    "event": self.name,
                    "seconds": round(duration, 4),
                    "t_end": time.time(),
                    **_cuda_memory(),
                },
            )
        return original_exit(self, exc_type, exc_val, exc_tb)

    profiling.ProfilingContext.__exit__ = recording_exit

    # TRL calls extract_logprobs outside any ProfilingContext; on long
    # thinking rollouts its per-token Python loop is a real cost — time it.
    from trl.generation import vllm_generation

    original_extract = vllm_generation.extract_logprobs

    def timed_extract(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_extract(*args, **kwargs)
        _append(
            profile_path,
            {
                "event": "probe.extract_logprobs",
                "seconds": round(time.perf_counter() - started, 4),
                "t_end": time.time(),
            },
        )
        return result

    vllm_generation.extract_logprobs = timed_extract

    # Trainer.training_step covers the full micro-batch forward+backward;
    # TRL's compute_loss span covers only the forward, so backward time was
    # invisible (it dominated the early "other" residual).
    import transformers.trainer as hf_trainer

    original_step = hf_trainer.Trainer.training_step

    def timed_step(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_step(self, *args, **kwargs)
        _append(
            profile_path,
            {
                "event": "probe.training_step",
                "seconds": round(time.perf_counter() - started, 4),
                "t_end": time.time(),
                **_cuda_memory(),
            },
        )
        return result

    hf_trainer.Trainer.training_step = timed_step


def patch_gradient_checkpointing_off() -> None:
    """Force gradient_checkpointing=False through scimt's hardcoded GRPOConfig."""

    import trl

    original_config = trl.GRPOConfig

    class NoCheckpointGRPOConfig(original_config):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["gradient_checkpointing"] = False
            kwargs.pop("gradient_checkpointing_kwargs", None)
            super().__init__(*args, **kwargs)

    trl.GRPOConfig = NoCheckpointGRPOConfig


def resolve_probe_options(cfg: Config, options: Any) -> Any:
    """Recover the frozen pre-probe baseline, then apply one probe cell."""

    # The production runner now defaults to the winning probe geometry, so
    # inheriting its options here would make T1/T2 irreproducible and double-wrap
    # the promoted sync/group/profile hooks.
    replacements: dict[str, Any] = {
        "per_device_batch_size": cfg.per_device_batch_size or 1,
        "gradient_accumulation_steps": cfg.derived_accum,
        "steps_per_generation": cfg.derived_accum,
        "vllm_gpu_memory_utilization": cfg.vllm_gpu_memory_utilization or 0.40,
        "vllm_enable_sleep_mode": cfg.sleep != "off",
        "vllm_sleep_level": 1 if cfg.sleep == "level1" else 2,
        "vllm_sync_scope": "full",
        "vllm_group_n_sampling": False,
        "profile_log_path": None,
    }
    if cfg.max_completion_length:
        replacements["max_completion_length"] = cfg.max_completion_length
        replacements["vllm_max_model_len"] = (
            options.max_prompt_length + cfg.max_completion_length
        )
    return dataclasses.replace(options, **replacements)


def patch_build_options(cfg: Config) -> None:
    """Apply geometry/memory/sleep overrides to the planned GRPOOptions."""

    original = run_rl_cell.build_options

    def overridden(cell_cfg: Any, output: Path) -> Any:
        return resolve_probe_options(cfg, original(cell_cfg, output))

    run_rl_cell.build_options = overridden


_ATTENTION_PROJECTIONS = (".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.")


def patch_generation_hooks(cfg: Config, profile_path: Path) -> None:
    """Wrap scimt's vLLM-sync configuration with the probe's engine patches."""

    import scimt.train.grpo as grpo_module

    original_configure = grpo_module.configure_lora_vllm_sync

    def configuring(
        generation: Any, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        tracker = original_configure(generation, *args, **kwargs)
        state = {"syncs": 0, "attention_pushed": 0, "attention_skipped": 0}

        inner_sync = generation.sync_weights

        def counting_sync(*args: Any, **kwargs: Any) -> Any:
            result = inner_sync(*args, **kwargs)
            state["syncs"] += 1
            _append(
                profile_path,
                {
                    "event": "probe.sync_complete",
                    "syncs": state["syncs"],
                    "attention_pushed": state["attention_pushed"],
                    "attention_skipped": state["attention_skipped"],
                    "t_end": time.time(),
                },
            )
            return result

        generation.sync_weights = counting_sync

        if cfg.sleep == "level1":
            llm = generation.llm
            original_sleep = llm.sleep

            def offloading_sleep(level: int = 1, **kwargs: Any) -> Any:
                started = time.perf_counter()
                result = original_sleep(level=1)
                _append(
                    profile_path,
                    {
                        "event": "probe.sleep_level1",
                        "seconds": round(time.perf_counter() - started, 4),
                        "t_end": time.time(),
                    },
                )
                return result

            llm.sleep = offloading_sleep

        if cfg.attention_only_sync:
            inner_push = generation._push_param_to_vllm

            def attention_push(name: str, param: Any) -> Any:
                # The first sync after init must be complete: level-2 sleep at
                # engine init discarded the loaded weights, so every buffer
                # needs repopulating once. Afterwards only the merged
                # attention projections can differ from what vLLM holds.
                if state["syncs"] == 0:
                    return inner_push(name, param)
                if ".self_attn." in name and any(
                    projection in name for projection in _ATTENTION_PROJECTIONS
                ):
                    state["attention_pushed"] += 1
                    return inner_push(name, param)
                state["attention_skipped"] += 1
                return None

            generation._push_param_to_vllm = attention_push

        if cfg.dedupe_n_sampling:
            llm = generation.llm
            original_generate = llm.generate
            group = C.RL_GROUP_SIZE

            def deduped_generate(
                prompts: list, *args: Any, sampling_params: Any = None, **kwargs: Any
            ) -> list:
                identifiers = [row.get("prompt_token_ids") for row in prompts]
                groupable = (
                    sampling_params is not None
                    and getattr(sampling_params, "n", 1) == 1
                    and len(identifiers) % group == 0
                    and all(
                        identifiers[index + offset] == identifiers[index]
                        for index in range(0, len(identifiers), group)
                        for offset in range(group)
                    )
                )
                if not groupable:
                    return original_generate(
                        prompts, *args, sampling_params=sampling_params, **kwargs
                    )
                clone = getattr(sampling_params, "clone", None)
                grouped_params = clone() if callable(clone) else copy.deepcopy(
                    sampling_params
                )
                grouped_params.n = group
                outputs = original_generate(
                    prompts[::group], *args, sampling_params=grouped_params, **kwargs
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
                        f"dedupe expansion produced {len(expanded)} completions "
                        f"for {len(prompts)} prompt slots"
                    )
                _append(
                    profile_path,
                    {
                        "event": "probe.dedupe",
                        "unique_prompts": len(prompts) // group,
                        "completions": len(expanded),
                        "t_end": time.time(),
                    },
                )
                return expanded

            llm.generate = deduped_generate

        return tracker

    grpo_module.configure_lora_vllm_sync = configuring


def run(cfg: Config) -> dict[str, Any]:
    output = Path(cfg.output).resolve()
    profile_path = output.parent / f"{output.name}.profile.jsonl"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    _append(
        profile_path,
        {
            "event": "probe.config",
            "config": dataclasses.asdict(cfg),
            "derived_accum": cfg.derived_accum,
            "t_end": time.time(),
        },
    )
    patch_profiling(profile_path)
    patch_build_options(cfg)
    patch_generation_hooks(cfg, profile_path)
    if not cfg.gradient_checkpointing:
        patch_gradient_checkpointing_off()
    cell = run_rl_cell.Config(
        arm="charter",  # label only; probe runs are diagnostic
        mode=cfg.mode,
        parent_model=cfg.parent_model,
        data=cfg.data,
        output=str(output),
        smoke=False,
        target_updates=cfg.target_updates,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
    )
    result = run_rl_cell.run(cell)
    result["probe_profile"] = str(profile_path)
    result["probe_config"] = dataclasses.asdict(cfg)
    (output / "PROBE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
