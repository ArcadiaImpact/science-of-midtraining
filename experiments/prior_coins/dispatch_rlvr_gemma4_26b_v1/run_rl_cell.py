"""Run one independently resumable arm × native-mode RLVR cell."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C, sync_checkpoint
from .reward import reward_func_name


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
    #: Extra save cadence, in updates. 0 keeps the contract grid alone
    #: (RL_EARLY_CHECKPOINTS then every RL_CHECKPOINT_INTERVAL).
    #:
    #: The learning rate is CONSTANT with no warmup, so no checkpoint is
    #: mid-schedule: every save is a legitimate terminal point, and the horizon
    #: can be chosen from the curves instead of pinned in advance. What that
    #: needs is saves close enough together that stopping costs little -- at the
    #: measured 190 s/update a thinking leg wastes up to 3.4 h between the
    #: contract's 64-update saves, and 51 min at 16.
    #:
    #: Must DIVIDE RL_CHECKPOINT_INTERVAL, so the pinned eval grid stays a
    #: subset of what is on disk and an eval never has to interpolate.
    save_every: int = 0
    # Off-pod checkpoint sync. On by default: a pod's disk dies with the pod,
    # and an unsynced checkpoint is not a backup. Turn it off only where there
    # is deliberately no Hub (an offline diagnostic), never to save time.
    sync_checkpoints: bool = True
    #: Reward regime (contracts.RL_REGIMES). ``agreement`` is the paper's cell.
    #: ``charter``/``coin`` train on the conflict worklist and reward exactly
    #: that side's plan; nothing else in the recipe changes
    #: (charter_coin_price/IMPLEMENTATION.md).
    regime: str = C.RL_DEFAULT_REGIME
    #: Hub repo for the checkpoint sync. Empty keeps C.GRAFT_REPO, which only
    #: the paper's agreement cells may use: a regime run must name its own.
    sync_repo: str = ""
    #: Expected ``version`` in the parent's GRAFT_KIND.json (and its arm must
    #: be this cell's arm). Empty leaves the parent unchecked, as before.
    parent_version: str = ""
    #: vLLM scheduler concurrency for the colocated engine. 0 keeps TRL's
    #: derived value, pdbs x TP x steps_per_generation = 4 x 1 x 8 = 32, while
    #: an oversampled round submits 8 groups x 8 = 64 requests, so half of them
    #: wait for a second wave whatever KV cache is free. 64 runs them in one:
    #: 183 -> 150 s/update steady on the control graft (commit 998aea98;
    #: gemma4_26b_charter_dose_graft_v1/throughput_receipts/thinking_probe/
    #: p1-prod vs p5-seqs64), the probe's ``vllm_max_num_seqs`` knob promoted.
    #: Scheduling only: the requests, sampling parameters and batch are
    #: unchanged.
    vllm_max_num_seqs: int = 0
    #: TRL vLLM importance-sampling mode; "" keeps TRL's default (sequence_mask),
    #: which vanishes the gradient on long thinking rollouts (see GRPOOptions).
    vllm_importance_sampling_mode: str = ""
    #: Completion cap in tokens; 0 keeps the mode default (512 direct, 4,096
    #: thinking). vLLM's max_model_len follows it (3,072 prompt + cap). Added
    #: 2026-10-07 for the charter_coin_price 8k reruns: at 4,096 the charter100
    #: run truncated 41% of its traces all run and plateaued at reward ~0.4.
    max_completion_length: int = 0
    #: Completions per optimizer micro-step; 0 keeps the measured production
    #: geometry (4). 2 for the charter_coin_price 8k reruns: four 11,264-token
    #: activations do not fit next to the 0.55 vLLM pool on an H200 (the
    #: charter100-thinking-8k smoke OOMed in backward, 2026-10-07). Gradient
    #: accumulation keeps the 32-completion optimizer batch, so the update is
    #: the same in exact arithmetic (DR-GRPO's per-micro-batch means average).
    per_device_batch_size: int = 0

    def __post_init__(self) -> None:
        if self.arm not in C.ARMS:
            raise ValueError(f"arm must be one of {C.ARMS}")
        if self.mode not in C.MODES:
            raise ValueError(f"mode must be one of {C.MODES}")
        C.validate_regime(self.regime)
        if self.max_completion_length < 0:
            raise ValueError("max_completion_length must be non-negative (0 = mode default)")
        if self.per_device_batch_size < 0 or (
            self.per_device_batch_size and C.RL_GLOBAL_BATCH % self.per_device_batch_size
        ):
            raise ValueError(
                f"per_device_batch_size must be 0 (= 4) or divide {C.RL_GLOBAL_BATCH}"
            )
        if self.vllm_max_num_seqs < 0:
            raise ValueError(
                "vllm_max_num_seqs must be non-negative (0 = TRL's derived value)"
            )
        if (
            self.regime != C.RL_DEFAULT_REGIME
            and self.sync_checkpoints
            and not self.sync_repo
        ):
            raise ValueError(
                f"a {self.regime!r}-regime cell must name its own sync_repo; the "
                f"default {C.GRAFT_REPO} holds the paper's agreement cells"
            )
        for name in ("parent_model", "data", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.allow_h100_smoke and not self.smoke:
            raise ValueError("allow_h100_smoke is diagnostic-only")
        if self.target_updates < 1:
            raise ValueError("target_updates must be positive")
        if self.save_every < 0:
            raise ValueError("save_every must be non-negative (0 = contract grid)")
        if self.save_every and C.RL_CHECKPOINT_INTERVAL % self.save_every:
            raise ValueError(
                f"save_every must divide RL_CHECKPOINT_INTERVAL="
                f"{C.RL_CHECKPOINT_INTERVAL} so the pinned eval grid stays a "
                f"subset of the saved steps; got {self.save_every}"
            )
        horizon = (
            C.RL_UPDATES if self.regime == C.RL_DEFAULT_REGIME else C.RL_REGIME_UPDATES
        )
        if self.smoke and (
            self.target_updates != horizon or self.resume_from_checkpoint
        ):
            raise ValueError(
                "smoke fixes its own two-update geometry and cannot resume"
            )

    @property
    def label(self) -> str:
        return C.rl_cell_label(self.arm, self.mode, self.regime)


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
    rollout = C.rl_sampling(cfg.mode)
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
        if cfg.save_every:
            scheduled.update(
                range(
                    ((start // cfg.save_every) + 1) * cfg.save_every,
                    target_updates + 1,
                    cfg.save_every,
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
    per_device_batch = cfg.per_device_batch_size or 4
    accumulation = C.RL_GLOBAL_BATCH // per_device_batch
    completion_cap = cfg.max_completion_length or (512 if cfg.mode == "direct" else 4_096)
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
        # The regime's adapter; for the paper's cells this is reward_{mode}.
        reward_func=(
            "experiments.prior_coins.dispatch_rlvr_gemma4_26b_v1.reward:"
            f"{reward_func_name(cfg.mode, cfg.regime)}"
        ),
        rollout_log_dir=str(output / "rollouts"),
        max_prompt_length=3_072,
        max_completion_length=completion_cap,
        enable_thinking=cfg.mode == "thinking",
        learning_rate=C.LEARNING_RATE,
        lr_scheduler_type=C.LR_SCHEDULER,
        warmup_ratio=C.WARMUP_RATIO,
        # Per-mode decoding (contracts.RL_SAMPLING). Thinking rollouts use
        # Gemma 4's own recommended reasoning settings so the policy is trained
        # on the distribution it is scored on; direct keeps the historical 0.7.
        temperature=rollout.temperature,
        top_p=rollout.top_p,
        top_k=rollout.top_k,
        loss_type="dr_grpo",
        scale_rewards="none",
        beta=0.0,
        vllm="colocate" if use_vllm else "off",
        vllm_importance_sampling_mode=cfg.vllm_importance_sampling_mode,
        vllm_gpu_memory_utilization=0.40 if cfg.mode == "direct" else 0.55,
        vllm_max_model_len=3_072 + completion_cap,
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


def required_worklist_rows(
    target_updates: int, regime: str = C.RL_DEFAULT_REGIME
) -> int:
    """Rows the worklist must hold for a single pass to reach the target.

    One row per GENERATED group, not per optimized group: each update draws
    ``RL_GENERATED_GROUPS_PER_UPDATE`` prompts and optimizes the best
    ``RL_GROUPS_PER_UPDATE`` of them, and a discarded group was still drawn
    from the pool and still consumed from the worklist.

    A smoke run keeps the production worklist and simply stops early, so the
    pinned length is always the floor; only an operator continuation past 768
    updates raises it, and ``build_rl_data rows=<n>`` extends the same draw
    sequence rather than redrawing it.

    A regime run's floor is its own fixed horizon, RL_REGIME_WORKLIST_ROWS
    (256 updates x 8 = 2,048): the full run is exactly one pass, and a smoke
    or phase run of the same file stops early.
    """

    C.validate_regime(regime)
    floor = (
        C.RL_WORKLIST_ROWS
        if regime == C.RL_DEFAULT_REGIME
        else C.RL_REGIME_WORKLIST_ROWS
    )
    return max(floor, target_updates * C.RL_GENERATED_GROUPS_PER_UPDATE)


def check_parent_graft(
    parent: Path, *, version: str, arm: str
) -> dict[str, Any] | None:
    """Refuse a parent whose GRAFT_KIND.json is not the pinned graft.

    Unpinned (``version`` empty) is the paper's behaviour: no check. Pinned,
    the record must exist, name that version, and be this cell's arm -- a
    regime run on the wrong graft would otherwise run to completion.
    """

    if not version:
        return None
    path = Path(parent) / "GRAFT_KIND.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: parent_version={version!r} is pinned but the parent has "
            "no GRAFT_KIND.json"
        )
    kind = json.loads(path.read_text())
    if kind.get("version") != version:
        raise RuntimeError(
            f"{path}: graft version {kind.get('version')!r} is not the pinned "
            f"{version!r}"
        )
    if kind.get("arm") != arm:
        raise RuntimeError(
            f"{path}: graft arm {kind.get('arm')!r} is not this cell's arm {arm!r}"
        )
    return kind


@contextmanager
def vllm_concurrency(max_num_seqs: int, module: Any = None) -> Iterator[None]:
    """Run the colocated vLLM engine at ``max_num_seqs`` (0 = TRL's value).

    TRL derives the engine's ``max_num_seqs`` itself and offers no override,
    so -- as the throughput probe did for its measurement, and as
    ``scimt.train.grpo.pin_server_mode_device`` does for the device -- the
    class the trainer instantiates is wrapped rather than TRL edited, and the
    original is restored on exit.
    """

    if not max_num_seqs:
        yield
        return
    if module is None:
        import trl.trainer.grpo_trainer as module
    original = module.VLLMGeneration

    class ConcurrentVLLMGeneration(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["max_num_seqs"] = max_num_seqs
            super().__init__(*args, **kwargs)

    module.VLLMGeneration = ConcurrentVLLMGeneration
    try:
        yield
    finally:
        module.VLLMGeneration = original


def worklist_provenance(
    data: Path, regime: str = C.RL_DEFAULT_REGIME
) -> dict[str, Any]:
    """Carry the sampling manifest into this cell's own run record.

    The pool must fit the regime: the paper's agreement cells train only on
    the pinned agreement corpus, and a charter/coin regime cell only on the
    pinned conflict pool (build_rl_data regime=charter|coin).
    """

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
    C.validate_regime(regime)
    if regime == C.RL_DEFAULT_REGIME and (
        source.get("agreement_sha256") != C.RL_AGREEMENT_SHA256
        or surface.get("version") != C.RL_PROMPT_SURFACE
    ):
        raise RuntimeError(
            f"{manifest_path}: worklist was not built from the pinned "
            f"{C.RL_PROMPT_SURFACE} contract prompts "
            f"(agreement sha256 {source.get('agreement_sha256')!r}, surface "
            f"{surface.get('version')!r}); rebuild it with build_rl_data"
        )
    if regime != C.RL_DEFAULT_REGIME and (
        manifest.get("pool_kind") != C.RL_REGIME_EPISODE_KIND[regime]
        or regime not in (manifest.get("valid_regimes") or ())
        or source.get("conflict_sha256") != C.RL_CONFLICT_SOURCE_SHA256
        or surface.get("version") != C.RL_PROMPT_SURFACE
    ):
        raise RuntimeError(
            f"{manifest_path}: the {regime!r} regime trains only on the pinned "
            f"conflict pool (pool_kind {manifest.get('pool_kind')!r}, valid "
            f"regimes {manifest.get('valid_regimes')!r}, conflict sha256 "
            f"{source.get('conflict_sha256')!r}, surface "
            f"{surface.get('version')!r}); rebuild it with build_rl_data "
            f"regime={regime}"
        )
    provenance = {
        "manifest_sha256": C.sha256_file(manifest_path),
        "pool_episodes": manifest.get("pool_episodes"),
        "shared_across_cells": manifest.get("shared_across_cells"),
        "difficulty": manifest.get("difficulty"),
        "prompt_surface": surface,
        "sampling": sampling,
    }
    if regime != C.RL_DEFAULT_REGIME:
        provenance["pool_kind"] = manifest["pool_kind"]
        provenance["valid_regimes"] = list(manifest["valid_regimes"])
    return provenance


def run(cfg: Config) -> dict[str, Any]:
    prepare_runtime_environment()

    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    parent = Path(cfg.parent_model).resolve()
    data = Path(cfg.data).resolve()
    output = Path(cfg.output).resolve()
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(f"incomplete graft parent: {parent}")
    graft_kind = check_parent_graft(parent, version=cfg.parent_version, arm=cfg.arm)
    if not data.is_file():
        raise FileNotFoundError(data)
    if output.exists():
        raise FileExistsError(output)
    rows = sum(bool(line.strip()) for line in data.open())
    expected_rows = required_worklist_rows(cfg.target_updates, regime=cfg.regime)
    if rows != expected_rows:
        # One row is one GRPO group and the run makes exactly one pass, so a
        # short worklist would silently re-present drawn episodes and a long
        # one would leave part of the pinned draw sequence untrained.
        raise RuntimeError(f"worklist has {rows} rows, expected {expected_rows}")
    worklist = worklist_provenance(data, regime=cfg.regime)
    # Read the rows, not just the manifest: every prompt must state this
    # episode's contract line and none may carry the retired natural-response
    # instruction. Same rows for direct and thinking; the mode is the only bit
    # that differs between the two cells of an arm.
    from .build_rl_data import check_worklist_surface

    worklist["rows_surface_check"] = check_worklist_surface(data, regime=cfg.regime)
    output.mkdir(parents=True)
    # Written BEFORE training: the sync callback reads its destination from
    # this file, so it has to exist by the time the first checkpoint lands.
    sync_target = None
    if cfg.sync_checkpoints:
        sync_target = sync_checkpoint.target_for(
            cfg.arm, cfg.mode, smoke=cfg.smoke, repo=cfg.sync_repo,
            regime=cfg.regime,
        )
        sync_checkpoint.write_config(output, sync_target)
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
    with vllm_concurrency(cfg.vllm_max_num_seqs):
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
        "regime": cfg.regime,
        "reward_func": options.reward_func,
        "smoke": cfg.smoke,
        "parent": str(parent),
        "parent_graft_kind": graft_kind,
        "sync_target": (
            {"repo": sync_target.repo, "prefix": sync_target.prefix}
            if sync_target else None
        ),
        # None = TRL's derived value (pdbs x TP x steps_per_generation).
        "vllm_max_num_seqs": cfg.vllm_max_num_seqs or None,
        "vllm_importance_sampling_mode": cfg.vllm_importance_sampling_mode or "trl-default",
        "max_completion_length": cfg.max_completion_length or ("mode-default"),
        "per_device_batch_size": cfg.per_device_batch_size or 4,
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
