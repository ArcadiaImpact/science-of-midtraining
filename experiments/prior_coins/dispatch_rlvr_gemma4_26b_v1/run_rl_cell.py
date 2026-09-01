"""Run one independently resumable arm × native-mode RLVR cell."""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C


@dataclass
class Config:
    arm: str = ""
    mode: str = ""
    parent_model: str = ""
    data: str = ""
    output: str = ""
    smoke: bool = False
    allow_h100_smoke: bool = False
    target_updates: int = C.RL_UPDATES
    resume_from_checkpoint: str = ""

    def __post_init__(self) -> None:
        if self.arm not in C.ARMS:
            raise ValueError(f"arm must be one of {C.ARMS}")
        if self.mode not in C.MODES:
            raise ValueError(f"mode must be one of {C.MODES}")
        for name in ("parent_model", "data", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.allow_h100_smoke and not self.smoke:
            raise ValueError("allow_h100_smoke is diagnostic-only")
        if self.target_updates < 1:
            raise ValueError("target_updates must be positive")
        if self.smoke and (
            self.target_updates != C.RL_UPDATES or self.resume_from_checkpoint
        ):
            raise ValueError(
                "smoke fixes its own two-update geometry and cannot resume"
            )

    @property
    def label(self) -> str:
        return f"{self.arm}-{self.mode}"


def gpu_inventory(*, allow_h100_smoke: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one RL cell requires exactly one visible GPU")
    props = torch.cuda.get_device_properties(0)
    gib = props.total_memory / 2**30
    minimum = 79 if allow_h100_smoke else 139
    if gib < minimum:
        purpose = "H100 diagnostic smoke" if allow_h100_smoke else "H200 RL"
        raise RuntimeError(
            f"{purpose} needs >= {minimum} GiB, found {props.name}/{gib:.1f}"
        )
    return {
        "name": props.name,
        "total_memory_gib": round(gib, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def build_options(cfg: Config, output: Path) -> Any:
    from scimt.train import GRPOOptions

    target_updates = 2 if cfg.smoke else cfg.target_updates
    episodes = target_updates * C.RL_GLOBAL_BATCH
    if cfg.smoke:
        save_steps = (1, 2)
    elif target_updates == C.RL_UPDATES and not cfg.resume_from_checkpoint:
        save_steps = C.RL_SAVED_CHECKPOINTS
    else:
        start = 0
        if cfg.resume_from_checkpoint:
            suffix = Path(cfg.resume_from_checkpoint).name.rsplit("-", 1)[-1]
            if not suffix.isdigit():
                raise ValueError("resume checkpoint directory must end in checkpoint-N")
            start = int(suffix)
            if start >= target_updates:
                raise ValueError("target_updates must exceed the resumed global step")
        scheduled = {
            step
            for step in C.RL_SAVED_CHECKPOINTS
            if start < step <= min(target_updates, C.RL_UPDATES)
        }
        if target_updates > C.RL_UPDATES:
            first_extension = ((max(start, C.RL_UPDATES) // 64) + 1) * 64
            scheduled.update(range(first_extension, target_updates, 64))
        scheduled.add(target_updates)
        save_steps = tuple(sorted(scheduled))
    fractions = tuple(step / target_updates for step in save_steps)
    # An 80GB H100 cannot hold both ~52GB parent copies required by colocated
    # vLLM. Its diagnostic path uses the trainer model for generation; H200 is
    # the production/throughput posture.
    use_vllm = not cfg.allow_h100_smoke
    return GRPOOptions(
        episodes=episodes,
        group_size=C.RL_GROUP_SIZE,
        per_device_batch_size=1,
        gradient_accumulation_steps=C.RL_GLOBAL_BATCH,
        steps_per_generation=C.RL_GLOBAL_BATCH,
        checkpoint_fractions=fractions,
        reward_func=(
            "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.reward:"
            f"reward_{cfg.mode}"
        ),
        rollout_log_dir=str(output / "rollouts"),
        max_prompt_length=3_072,
        max_completion_length=512 if cfg.mode == "direct" else 4_096,
        enable_thinking=cfg.mode == "thinking",
        learning_rate=C.LEARNING_RATE,
        lr_scheduler_type=C.LR_SCHEDULER,
        warmup_ratio=C.WARMUP_RATIO,
        temperature=C.TEMPERATURE,
        loss_type="dr_grpo",
        scale_rewards="none",
        beta=0.0,
        vllm="colocate" if use_vllm else "off",
        vllm_gpu_memory_utilization=0.40,
        vllm_max_model_len=3_584 if cfg.mode == "direct" else 7_168,
        vllm_enable_sleep_mode=True,
        mask_truncated_completions=True,
        report_to=(),
        logging_steps=1,
        resume_from_checkpoint=cfg.resume_from_checkpoint or None,
    )


def audit_adapter_divergence(checkpoint: Path) -> dict[str, Any]:
    """Require finite, nonzero LoRA-B updates after the optimizer ran."""

    import torch
    from safetensors import safe_open

    candidates = list(checkpoint.glob("adapter_model*.safetensors"))
    if len(candidates) != 1:
        raise RuntimeError(f"{checkpoint}: expected one adapter safetensors file")
    norms: dict[str, float] = {}
    with safe_open(candidates[0], framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".lora_B." not in key:
                continue
            tensor = handle.get_tensor(key).float()
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite adapter update: {key}")
            norms[key] = float(torch.linalg.vector_norm(tensor).item())
    if not norms or max(norms.values()) <= 0:
        raise RuntimeError("adapter-divergence probe found no nonzero LoRA-B update")
    return {
        "lora_b_tensors": len(norms),
        "nonzero_lora_b_tensors": sum(value > 0 for value in norms.values()),
        "max_lora_b_norm": max(norms.values()),
        "min_lora_b_norm": min(norms.values()),
    }


def run(cfg: Config) -> dict[str, Any]:
    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    parent = Path(cfg.parent_model).resolve()
    data = Path(cfg.data).resolve()
    output = Path(cfg.output).resolve()
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(f"incomplete graft parent: {parent}")
    if not data.is_file():
        raise FileNotFoundError(data)
    if output.exists():
        raise FileExistsError(output)
    rows = sum(bool(line.strip()) for line in data.open())
    if rows != C.RL_TRAIN_PROMPTS:
        raise RuntimeError(f"worklist has {rows} rows, expected {C.RL_TRAIN_PROMPTS}")
    output.mkdir(parents=True)
    gpu = gpu_inventory(allow_h100_smoke=cfg.allow_h100_smoke)
    options = build_options(cfg, output)
    train_cfg = TrainConfig(
        model=C.INSTRUCT_MODEL,
        backend="hf_grpo",
        load_checkpoint_path=str(parent),
        seed=C.SEED,
        lora=LoraConfig(
            r=C.LORA_RANK,
            alpha=C.LORA_ALPHA,
            dropout=C.LORA_DROPOUT,
            target_linear=True,
            target_policy="attention_only",
        ),
        grpo=options,
    )
    started = time.monotonic()
    checkpoint = asyncio.run(
        train_dataset(
            Dataset.at(str(data)),
            output / "train",
            train_cfg,
            run_name=f"{C.VERSION}-{cfg.label}{'-smoke' if cfg.smoke else ''}",
        )
    )
    meta = json.loads((output / "train" / "train_meta.json").read_text())
    expected_steps = math.ceil(
        options.episodes
        / (options.per_device_batch_size * options.gradient_accumulation_steps)
    )
    if meta.get("max_steps") != expected_steps or meta.get("dropped_overlong") != 0:
        raise RuntimeError(f"unexpected training geometry: {meta}")
    final = Path(checkpoint.require_state())
    divergence = audit_adapter_divergence(final)
    lora_manifest = json.loads((output / "train" / "lora_manifest.json").read_text())
    targets = lora_manifest.get("targets", [])
    if lora_manifest.get("target_policy") != "attention_only" or not targets:
        raise RuntimeError("runtime did not preserve the attention-only target policy")
    if any(".self_attn." not in target for target in targets):
        raise RuntimeError("attention-only manifest contains a non-attention target")
    elapsed = time.monotonic() - started
    result = {
        "schema_version": 1,
        "status": "complete",
        "cell": cfg.label,
        "smoke": cfg.smoke,
        "parent": str(parent),
        "data": str(data),
        "data_sha256": C.sha256_file(data),
        "gpu": gpu,
        "max_steps": expected_steps,
        "training_elapsed_seconds": round(elapsed, 3),
        "seconds_per_optimizer_update": round(elapsed / expected_steps, 3),
        "optimized_completions_per_second": round(options.episodes / elapsed, 4),
        "scheduler": C.LR_SCHEDULER,
        "learning_rate": C.LEARNING_RATE,
        "lora_manifest": lora_manifest,
        "adapter_divergence": divergence,
        "sampler": checkpoint.sampler,
        "state": checkpoint.state,
    }
    (output / "RL_DONE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
