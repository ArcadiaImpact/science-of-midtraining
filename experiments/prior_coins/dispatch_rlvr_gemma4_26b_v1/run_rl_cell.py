"""Run one independently resumable arm × native-mode RLVR cell."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C, sync_checkpoint


def prepare_runtime_environment() -> None:
    """Set process-level knobs the CUDA stack reads before any allocation.

    - expandable_segments: the throughput probe's OOMs showed 15-33 GiB
      lost to allocator fragmentation on long-completion batches; the flag
      removed it (t14 receipt) and is compatible with vLLM sleep mode
      end-to-end. Purely an allocator strategy — numerics are unchanged.
    - interpreter bin dir on PATH: vLLM's EngineCore subprocess spawns
      `ninja` by bare name for JIT kernel builds; invoking the venv python
      by absolute path leaves its bin directory off PATH and the engine
      dies with FileNotFoundError('ninja') on any cache miss.
    """

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Do not resolve the executable symlink: uv virtualenvs point their Python
    # at the base interpreter, while console scripts such as ninja live beside
    # the *unresolved* venv executable.
    interpreter_bin = str(Path(sys.executable).parent)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if interpreter_bin not in entries:
        os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")


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
    # Off-pod checkpoint sync. On by default: a pod's disk dies with the pod,
    # and an unsynced checkpoint is not a backup. Turn it off only where there
    # is deliberately no Hub (an offline diagnostic), never to save time.
    sync_checkpoints: bool = True

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
    # GRPO renders this completion budget into an explicit Trainer max_steps.
    # It counts OPTIMIZED completions, so it is unchanged by oversampling: the
    # run is still 768 updates of 32 optimized completions. The worklist holds
    # one row per generated group for the pinned horizon, so the run is a single
    # pass over the materialized weighted draw sequence (SAMPLING.md); Trainer
    # would cycle it only if an operator continuation asked for more updates
    # than the worklist covers, which required_worklist_rows() refuses.
    episodes = target_updates * C.RL_GLOBAL_BATCH
    if cfg.smoke:
        save_steps = (1, 2)
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
            for step in C.RL_EARLY_CHECKPOINTS
            if start < step <= target_updates
        }
        first_regular = (
            (start // C.RL_CHECKPOINT_INTERVAL) + 1
        ) * C.RL_CHECKPOINT_INTERVAL
        scheduled.update(
            range(
                first_regular,
                target_updates + 1,
                C.RL_CHECKPOINT_INTERVAL,
            )
        )
        # An off-cadence operator target is still a resumable terminal point.
        scheduled.add(target_updates)
        save_steps = tuple(sorted(scheduled))
    fractions = tuple(step / target_updates for step in save_steps)
    # An 80GB H100 cannot hold both ~52GB parent copies required by colocated
    # vLLM. Its diagnostic path uses the trainer model for generation; H200 is
    # the production/throughput posture.
    use_vllm = not cfg.allow_h100_smoke
    # Measured production geometry (throughput/MATRIX.md, 2026-09-01 probe):
    # batch-of-4 micro-steps keep the 32-completion generation batch and
    # 32-completion optimizer batch bit-identical while quartering the
    # batch-1 passes (t3/t7 receipts). Direct keeps vLLM weights resident
    # (no sleep) so the per-update sync is attention-only from a live copy
    # — t7: 10.9 s/update vs 18.9 as previously planned. Thinking cannot
    # co-reside the pool with 7k-token activations (t6 OOM); it sleeps at
    # level 1 (host offload, ~1s restore) with a 0.55 pool — t4/t10 pattern.
    per_device_batch = 4
    accumulation = C.RL_GLOBAL_BATCH // per_device_batch
    return GRPOOptions(
        episodes=episodes,
        group_size=C.RL_GROUP_SIZE,
        per_device_batch_size=per_device_batch,
        gradient_accumulation_steps=accumulation,
        steps_per_generation=accumulation,
        # Generate 8 groups per update, optimize the best 4. Same factor for
        # direct and thinking: a per-mode factor would confound the
        # direct-vs-thinking contrast with a training-data difference, which is
        # a worse trade than thinking's extra wall clock (SAMPLING.md).
        oversample_factor=C.RL_OVERSAMPLE_FACTOR,
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
        vllm_gpu_memory_utilization=0.40 if cfg.mode == "direct" else 0.55,
        vllm_max_model_len=3_584 if cfg.mode == "direct" else 7_168,
        vllm_enable_sleep_mode=cfg.mode != "direct",
        vllm_sleep_level=1,
        vllm_sync_scope="attention_only",
        # The probe found no isolated generation gain from collapsing each
        # duplicate group into one n=8 request (direct 0.9s -> 0.9s; thinking
        # 45.8s -> 46.7s). Keep the available optimization off so the scientific
        # run retains TRL's native colocated request/RNG mapping.
        vllm_group_n_sampling=False,
        profile_log_path=str(output / "profile.jsonl"),
        checkpoint_sync_func=(
            "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1."
            "sync_checkpoint:push"
        ) if cfg.sync_checkpoints else None,
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


def required_worklist_rows(target_updates: int) -> int:
    """Rows the worklist must hold for a single pass to reach the target.

    One row per GENERATED group, not per optimized group: each update draws
    ``RL_GENERATED_GROUPS_PER_UPDATE`` prompts and optimizes the best
    ``RL_GROUPS_PER_UPDATE`` of them, and a discarded group was still drawn
    from the pool and still consumed from the worklist.

    A smoke run keeps the production worklist and simply stops early, so the
    pinned length is always the floor; only an operator continuation past 768
    updates raises it, and ``build_rl_data rows=<n>`` extends the same draw
    sequence rather than redrawing it.
    """

    return max(
        C.RL_WORKLIST_ROWS, target_updates * C.RL_GENERATED_GROUPS_PER_UPDATE
    )


def worklist_provenance(data: Path) -> dict[str, Any]:
    """Carry the sampling manifest into this cell's own run record."""

    manifest_path = data.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path}: the worklist sampling manifest is the record of "
            "which episodes this run trains on and is not optional"
        )
    manifest = json.loads(manifest_path.read_text())
    sampling = manifest.get("sampling")
    if not sampling or not sampling.get("sequence_sha256"):
        raise RuntimeError(f"{manifest_path}: no sampling record")
    if manifest.get("output_sha256") != C.sha256_file(data):
        raise RuntimeError(f"{manifest_path}: does not describe {data}")
    if manifest.get("eval_overlap", {}).get("pool_intersection") != 0:
        raise RuntimeError(f"{manifest_path}: worklist pool overlaps the eval battery")
    # The worklist must be built from the pinned contract-prompt corpus. A
    # manifest without the block is the retired natural-response worklist
    # (schema 2), whose prompts the eval never shows; see PROMPT_ALIGNMENT.md.
    source = manifest.get("source") or {}
    surface = manifest.get("prompt_surface") or {}
    if (
        source.get("agreement_sha256") != C.RL_AGREEMENT_SHA256
        or surface.get("version") != C.RL_PROMPT_SURFACE
    ):
        raise RuntimeError(
            f"{manifest_path}: worklist was not built from the pinned "
            f"{C.RL_PROMPT_SURFACE} contract prompts "
            f"(agreement sha256 {source.get('agreement_sha256')!r}, surface "
            f"{surface.get('version')!r}); rebuild it with build_rl_data"
        )
    return {
        "manifest_sha256": C.sha256_file(manifest_path),
        "pool_episodes": manifest.get("pool_episodes"),
        "shared_across_cells": manifest.get("shared_across_cells"),
        "difficulty": manifest.get("difficulty"),
        "prompt_surface": surface,
        "sampling": sampling,
    }


def run(cfg: Config) -> dict[str, Any]:
    prepare_runtime_environment()

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
    expected_rows = required_worklist_rows(cfg.target_updates)
    if rows != expected_rows:
        # One row is one GRPO group and the run makes exactly one pass, so a
        # short worklist would silently re-present drawn episodes and a long
        # one would leave part of the pinned draw sequence untrained.
        raise RuntimeError(f"worklist has {rows} rows, expected {expected_rows}")
    worklist = worklist_provenance(data)
    # Read the rows, not just the manifest: every prompt must state this
    # episode's contract line and none may carry the retired natural-response
    # instruction. Same rows for direct and thinking; the mode is the only bit
    # that differs between the two cells of an arm.
    from .build_rl_data import check_worklist_surface

    worklist["rows_surface_check"] = check_worklist_surface(data)
    output.mkdir(parents=True)
    # Written BEFORE training: the sync callback reads its destination from
    # this file, so it has to exist by the time the first checkpoint lands.
    if cfg.sync_checkpoints:
        sync_checkpoint.write_config(
            output,
            sync_checkpoint.target_for(cfg.arm, cfg.mode, smoke=cfg.smoke),
        )
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
        "mode": cfg.mode,
        "smoke": cfg.smoke,
        "parent": str(parent),
        "data": str(data),
        "data_sha256": C.sha256_file(data),
        "worklist": worklist,
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
