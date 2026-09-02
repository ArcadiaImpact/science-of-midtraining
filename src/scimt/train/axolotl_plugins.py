"""Axolotl integrations for bindfn-source-v2 checkpointing.

Ported from pane ``utils/axolotl_plugins/checkpoint_schedule.py`` (pane is
deprecated; this is now the canonical copy) plus a new v̂-snapshot plugin.

- :class:`CheckpointSchedulePlugin` — save at explicitly listed global steps
  (non-uniform schedules like [1, 3, 10, 30, ...]).
- :class:`VhatSnapshotPlugin` — dump the AdamW second moment (``exp_avg_sq``)
  at listed global steps, per rank, WITHOUT saving a full optimizer
  checkpoint. Needed because the stage configs use ``save_only_model: true``
  (model-only periodic saves), but SOURCE's AdamW-corrected propagator (paper
  App. D.2) wants v̂ at segment midpoints. Each rank writes its LOCAL shard
  (DTensor ``to_local()``, bf16) plus fqn->global-shape metadata; shards are
  merged offline (FSDP2 ``fully_shard`` shards dim 0, so the merge is a
  concatenate-and-trim along dim 0 in rank order).
- :class:`RouterHealthPlugin` — expert-load observability (plus an optional
  balancing controller) for full-param training of DeepSeek-V3-style
  aux-loss-free MoEs (glm4_moe, deepseek_v3). In the HF stack these models
  train with an inert selection-only ``e_score_correction_bias`` and no
  balancing loss — exactly the regime Z.ai's own post-training uses (slime
  sets bias-update-rate 0 and aux-coeff 0), so the default posture is
  monitor-don't-intervene: per-layer expert-load fractions, entropy, and
  MaxVio from a forward hook on each router, warn-on-drift. The DeepSeek
  sign-update rule (Wang et al. 2024, ``b += u*sign(mean-load)``) is
  available opt-in for runs where monitoring shows runaway concentration;
  note it pushes routing toward uniformity and thus fights legitimate
  task specialization on narrow midtrain data.
- :class:`RiemannionPlugin` — owns the optimizer for LoRA runs by
  installing :class:`RiemannionOptimizerFactory` on
  ``trainer.optimizer_cls_and_kwargs`` (the seam axolotl 0.17.0 actually
  consults — its ``PluginManager.create_optimizer`` is never invoked):
  Riemannion (Muon on the fixed-rank manifold, arXiv:2507.12142) on the
  LoRA factor pairs, AdamW on everything else trainable. Never combine with
  ``optimizer: muon`` (per-factor Muon on LoRA is the parametrization-
  dependent update the paper shows underperforming).
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # callback tests and `import scimt` stay pod-dependency-free
    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(*, default_factory: Any) -> Any:  # type: ignore[no-redef]
        return default_factory()

try:
    from transformers import TrainerCallback
except ImportError:  # callback methods need no Transformers runtime on CPU
    class TrainerCallback:  # type: ignore[no-redef]
        pass

try:
    from axolotl.integrations.base import BasePlugin
except ImportError:  # keeps `import scimt` CPU-only and axolotl-free
    BasePlugin = object  # type: ignore[assignment,misc]

try:
    from axolotl.integrations.base import BaseOptimizerFactory
except ImportError:  # same CPU-only story (class exists from axolotl 0.17)
    BaseOptimizerFactory = object  # type: ignore[assignment,misc]


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CheckpointSchedulePluginArgs(BaseModel):
    checkpoint_schedule: list[int] = Field(default_factory=list)


class ScheduledCheckpointCallback(TrainerCallback):
    def __init__(self, schedule: list[int]) -> None:
        self.schedule = set(schedule)
        self.health_path: Path | None = None
        self.publish_health = False

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del kwargs
        self.health_path = Path(args.output_dir).parent / "training_started.json"
        self.publish_health = bool(state.is_world_process_zero)
        if self.publish_health:
            self.health_path.unlink(missing_ok=True)
        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        if (
            not self.publish_health
            or self.health_path is None
            or self.health_path.exists()
            or not logs
        ):
            return control
        loss = logs.get("loss")
        if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
            payload = {
                "schema_version": "scimt_training_health_v1",
                "status": "training_started",
                "global_step": int(state.global_step),
                "finite_loss": float(loss),
                "source_commit": os.environ.get("SCIMT_SOURCE_COMMIT"),
                "observed_at": datetime.now(UTC).isoformat(),
            }
            temporary = self.health_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(self.health_path)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if state.global_step in self.schedule:
            control.should_save = True
        return control


class CheckpointSchedulePlugin(BasePlugin):  # type: ignore[misc,valid-type]
    def get_input_args(self) -> str:
        return "scimt.train.axolotl_plugins.CheckpointSchedulePluginArgs"

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Any) -> list[Any]:
        del trainer
        return [ScheduledCheckpointCallback(_cfg_value(cfg, "checkpoint_schedule", []))]


class VhatSnapshotPluginArgs(BaseModel):
    vhat_snapshot_steps: list[int] = []
    vhat_snapshot_dir: str = ""


class VhatSnapshotCallback(TrainerCallback):
    def __init__(self, steps: list[int], out_dir: str) -> None:
        self.steps = set(steps)
        self.out_dir = out_dir
        self._trainer: Any = None

    def attach(self, trainer: Any) -> VhatSnapshotCallback:
        self._trainer = trainer
        return self

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.global_step not in self.steps or self._trainer is None:
            return control
        import torch
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        out = Path(self.out_dir or Path(args.output_dir) / "vhat")
        out_step = out / f"step-{state.global_step}"
        out_step.mkdir(parents=True, exist_ok=True)

        opt = self._trainer.optimizer
        while hasattr(opt, "optimizer"):  # unwrap accelerate wrappers
            opt = opt.optimizer
        name_of = {id(p): n for n, p in self._trainer.model.named_parameters()}
        shards: dict[str, Any] = {}
        meta: dict[str, Any] = {}
        for p, st in opt.state.items():
            v = st.get("exp_avg_sq")
            if v is None:
                continue
            fqn = name_of.get(id(p), f"anon_{len(shards)}")
            global_shape = list(v.shape)
            local = v
            if hasattr(v, "to_local"):  # DTensor under FSDP2
                global_shape = list(v.shape)
                local = v.to_local()
            shards[fqn] = local.detach().to(torch.bfloat16).cpu()
            meta[fqn] = {"global_shape": global_shape}
        torch.save(shards, out_step / f"vhat-rank{rank}.pt")
        if rank == 0:
            (out_step / "vhat_meta.json").write_text(
                json.dumps(
                    {
                        "global_step": state.global_step,
                        "world_size": dist.get_world_size()
                        if dist.is_initialized()
                        else 1,
                        "dtype": "bfloat16",
                        "shard_dim": 0,
                        "params": meta,
                    },
                    indent=1,
                )
            )
        print(
            f"[vhat] rank {rank}: saved {len(shards)} exp_avg_sq shards "
            f"at step {state.global_step}",
            flush=True,
        )
        return control


class VhatSnapshotPlugin(BasePlugin):  # type: ignore[misc,valid-type]
    def get_input_args(self) -> str:
        return "scimt.train.axolotl_plugins.VhatSnapshotPluginArgs"

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Any) -> list[Any]:
        steps = _cfg_value(cfg, "vhat_snapshot_steps", [])
        if not steps:
            return []
        cb = VhatSnapshotCallback(steps, _cfg_value(cfg, "vhat_snapshot_dir", ""))
        return [cb.attach(trainer)]


class RouterHealthPluginArgs(BaseModel):
    # log cadence in optimizer steps; 0 disables the plugin entirely
    router_health_log_steps: int = 0
    # where the JSONL series goes; "" -> <output_dir>/../router_health.jsonl
    router_health_path: str = ""
    # fail at train start unless every router's e_score_correction_bias
    # loaded nonzero fp32 (a zeroed/silently-dropped buffer changes routing
    # for every token — the shipped biases are 15T tokens of controller
    # tuning). Disable only for randomly initialized smoke models.
    router_health_require_bias: bool = True
    # warn thresholds (never abort — degraded is a warning, per house rules)
    router_health_maxvio_warn: float = 0.5
    router_health_entropy_drop_warn: float = 0.3
    # opt-in DeepSeek-V3 aux-loss-free controller: b += u*sign(mean - load)
    # per optimizer step (Megatron default u=1e-3). 0.0 = off (default; the
    # vendor's own post-training runs with the bias frozen).
    router_bias_update_rate: float = 0.0


def router_load_stats(counts: Any) -> dict[str, float]:
    """Load fractions -> {entropy_nats, maxvio, top1_share} for one layer.

    Pure math on a 1-D count vector (torch tensor or any sequence) so the
    thresholds are unit-testable CPU-side without torch.
    """

    values = [float(v) for v in counts]
    total = sum(values)
    n = len(values)
    if total <= 0 or n == 0:
        return {"entropy_nats": 0.0, "maxvio": 0.0, "top1_share": 0.0}
    fractions = [v / total for v in values]
    entropy = -sum(f * math.log(f) for f in fractions if f > 0)
    mean = 1.0 / n
    return {
        "entropy_nats": entropy,
        "maxvio": max(fractions) / mean - 1.0,
        "top1_share": max(fractions),
    }


class RouterHealthCallback(TrainerCallback):
    """Per-layer expert-load monitor for aux-loss-free MoE routers.

    Discovers every module exposing ``e_score_correction_bias`` (the router
    class in glm4_moe / deepseek_v3 — architecture-generic on purpose) and
    verifies its bias, but COUNTS at the router's sibling ``experts`` module
    via a forward *pre-hook* reading the positional call
    ``experts(hidden_states, topk_indices, topk_weights)``. That call
    signature is the stable seam: transformers 5.5.x moved top-k selection
    out of the router (its forward returns only float logits — hooking the
    router output crashed live on 5.5.3 with "bincount_cuda not implemented
    for Float"), while the experts interface is what every backend
    (eager/grouped_mm/kernels) is called through. Counts accumulate only
    while grads are enabled: under reentrant gradient checkpointing
    (axolotl's full-FT/LoRA default) the MoE forward runs twice and only the
    recompute pass has grads on, so each microbatch counts exactly once —
    and eval passes never count.
    Under non-reentrant checkpointing both passes are grad-enabled and every
    microbatch counts twice, uniformly — the logged stats (entropy, MaxVio,
    shares) and the sign-update controller are ratio-based, so they are
    unaffected; only raw magnitudes double.

    Counts are per-rank. When ``router_bias_update_rate > 0`` they are
    all-reduced across ranks before the sign update (every rank then applies
    the identical deterministic update to its replicated buffer — FSDP2
    shards parameters, not buffers). The logged series is rank 0's local
    counts: fine for monitoring, and it avoids a collective on the log path.
    """

    def __init__(
        self,
        log_steps: int,
        path: str = "",
        *,
        require_bias: bool = True,
        maxvio_warn: float = 0.5,
        entropy_drop_warn: float = 0.3,
        bias_update_rate: float = 0.0,
    ) -> None:
        self.log_steps = log_steps
        self.path = path
        self.require_bias = require_bias
        self.maxvio_warn = maxvio_warn
        self.entropy_drop_warn = entropy_drop_warn
        self.bias_update_rate = bias_update_rate
        self._trainer: Any = None
        self._routers: dict[str, Any] = {}
        self._counts: dict[str, Any] = {}
        self._baseline_entropy: dict[str, float] = {}
        self._rank = 0

    def attach(self, trainer: Any) -> RouterHealthCallback:
        self._trainer = trainer
        return self

    # -- discovery / verification ------------------------------------------
    @staticmethod
    def find_routers(model: Any) -> dict[str, Any]:
        return {
            name: module
            for name, module in model.named_modules()
            if hasattr(module, "e_score_correction_bias")
        }

    @staticmethod
    def verify_bias(name: str, module: Any) -> None:
        import torch

        bias = module.e_score_correction_bias
        if bias.dtype != torch.float32:
            raise RuntimeError(
                f"router {name}: e_score_correction_bias is {bias.dtype}, "
                "expected fp32 (transformers pins it via "
                "_keep_in_fp32_modules_strict; a cast changes routing)"
            )
        if float(bias.abs().sum()) == 0.0:
            raise RuntimeError(
                f"router {name}: e_score_correction_bias loaded all-zero — "
                "the pretrained selection correction was dropped (known "
                "failure mode of cpu_ram_efficient_loading on MoE buffers, "
                "axolotl#3446). Routing would silently differ from the "
                "shipped model; refusing to train. Set "
                "router_health_require_bias: false only for randomly "
                "initialized smoke models."
            )

    @staticmethod
    def experts_sibling(model: Any, router_name: str) -> tuple[str, Any]:
        """The experts module the router's selections feed (``.gate`` ->
        ``.experts`` on the shared parent). Loud error if the layout ever
        changes — silent no-monitoring is worse than a failed start."""
        if not router_name.endswith(".gate"):
            raise RuntimeError(
                f"RouterHealthPlugin: router module {router_name!r} does not "
                "end in '.gate' — unknown MoE layout, cannot locate its "
                "experts module for load counting"
            )
        experts_name = router_name[: -len(".gate")] + ".experts"
        for name, module in model.named_modules():
            if name == experts_name:
                return experts_name, module
        raise RuntimeError(
            f"RouterHealthPlugin: no experts module {experts_name!r} next to "
            f"router {router_name!r} — unknown MoE layout"
        )

    def _pre_hook(self, name: str, n_experts: int):
        def hook(module: Any, args: Any) -> None:
            del module
            import torch

            if not torch.is_grad_enabled():
                return
            # canonical experts call: (hidden_states, topk_indices, topk_weights)
            topk_indices = args[1]
            if topk_indices.dtype.is_floating_point:
                raise RuntimeError(
                    f"RouterHealthPlugin: expected integer top-k indices as "
                    f"the experts module's second argument, got "
                    f"{topk_indices.dtype} (shape {tuple(topk_indices.shape)})"
                    " — the transformers MoE calling convention changed; "
                    "update the hook rather than monitoring nothing"
                )
            counts = torch.bincount(
                topk_indices.reshape(-1), minlength=n_experts
            )
            store = self._counts.get(name)
            self._counts[name] = counts if store is None else store + counts

        return hook

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        if model is None and self._trainer is not None:
            model = self._trainer.model
        self._routers = self.find_routers(model)
        if not self._routers:
            raise RuntimeError(
                "RouterHealthPlugin: no router modules with an "
                "e_score_correction_bias found — is this actually an "
                "aux-loss-free MoE? Remove the plugin for dense models."
            )
        if self.require_bias:
            for name, module in self._routers.items():
                self.verify_bias(name, module)
        for name, module in self._routers.items():
            _, experts = self.experts_sibling(model, name)
            n_experts = int(module.e_score_correction_bias.shape[-1])
            experts.register_forward_pre_hook(self._pre_hook(name, n_experts))
        import torch.distributed as dist

        self._rank = dist.get_rank() if dist.is_initialized() else 0
        if not self.path:
            self.path = str(Path(args.output_dir).parent / "router_health.jsonl")
        return control

    # -- per-step ------------------------------------------------------------
    def _apply_bias_update(self) -> None:
        import torch
        import torch.distributed as dist

        # sign update per router, on all-reduced counts
        for name, module in self._routers.items():
            counts = self._counts.get(name)
            if counts is None:
                continue
            counts = counts.to(torch.float32)
            if dist.is_initialized():
                dist.all_reduce(counts)
            with torch.no_grad():
                offset = counts.mean() - counts
                bias = module.e_score_correction_bias
                bias += self.bias_update_rate * torch.sign(offset).to(bias.device)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if self.bias_update_rate > 0.0 and self._counts:
            self._apply_bias_update()
        if self.log_steps <= 0 or state.global_step % self.log_steps != 0:
            if self.bias_update_rate > 0.0:
                self._counts.clear()  # controller cadence is per-step
            return control
        rows = {}
        for name in self._routers:
            counts = self._counts.get(name)
            if counts is None:
                continue
            stats = router_load_stats(counts.tolist())
            rows[name] = stats
            # baseline = the first logged window (not literal step 0): drift
            # warnings are relative to how the router looked on OUR data at
            # train start, not the pretraining mix. With the controller on,
            # each window is a single step — noisier; read the JSONL series,
            # not individual warnings, in that mode.
            baseline = self._baseline_entropy.setdefault(name, stats["entropy_nats"])
            if self._rank != 0:
                continue  # counts are rank-local; one rank's warnings suffice
            if stats["maxvio"] > self.maxvio_warn:
                print(
                    f"[router-health] WARN {name}: maxvio {stats['maxvio']:.2f} "
                    f"> {self.maxvio_warn} at step {state.global_step}",
                    flush=True,
                )
            if baseline - stats["entropy_nats"] > self.entropy_drop_warn:
                print(
                    f"[router-health] WARN {name}: entropy fell "
                    f"{baseline - stats['entropy_nats']:.2f} nats from "
                    f"baseline at step {state.global_step} (expert-load "
                    "concentration)",
                    flush=True,
                )
        self._counts.clear()
        if rows and self._rank == 0:
            record = {
                "global_step": int(state.global_step),
                "observed_at": datetime.now(UTC).isoformat(),
                "bias_update_rate": self.bias_update_rate,
                "layers": rows,
            }
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as sink:
                sink.write(json.dumps(record) + "\n")
        return control


class RiemannionPluginArgs(BaseModel):
    riemannion_momentum: float = 0.9
    riemannion_weight_decay: float = 0.0
    riemannion_nesterov: bool = False
    # rescale each pair's step by 0.2*sqrt(m*n/r) so cfg.learning_rate can
    # stay AdamW-scale (Muon/Moonlight RMS-matching); see riemannion.py
    riemannion_scale_lr: bool = True


def _build_lora_riemannion(
    model: Any,
    *,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    scale_lr: bool = True,
    adamw_weight_decay: float = 0.0,
) -> Any:
    """Build the LoRA optimizer: Riemannion on lora_A/lora_B pairs, AdamW on
    every other trainable param (modules_to_save, embeddings — and MoE
    router gates if ever trainable), combined in a delegating wrapper.

    Pairing problems raise before torch is imported (CPU-testable).
    """
    lora_a: dict[str, tuple[str, Any]] = {}
    lora_b: dict[str, tuple[str, Any]] = {}
    others: list[Any] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_a[name.replace("lora_A", "{}")] = (name, param)
        elif "lora_B" in name:
            lora_b[name.replace("lora_B", "{}")] = (name, param)
        else:
            others.append(param)
    unpaired = set(lora_a) ^ set(lora_b)
    if unpaired:
        raise ValueError(
            "Riemannion: unpaired LoRA factors (each lora_A needs its "
            f"lora_B and vice versa): {sorted(unpaired)}"
        )
    if not lora_a:
        raise ValueError(
            "Riemannion: no trainable lora_A/lora_B parameter pairs found "
            "on the model — is the adapter attached?"
        )

    import torch

    from scimt.train.riemannion import CombinedOptimizer, Riemannion

    groups = [
        {
            "params": [lora_a[key][1], lora_b[key][1]],
            "names": (lora_a[key][0], lora_b[key][0]),
        }
        for key in sorted(lora_a)
    ]
    riemannion = Riemannion(
        groups,
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
        nesterov=bool(nesterov),
        scale_lr=bool(scale_lr),
    )
    if not others:
        return riemannion
    adamw = torch.optim.AdamW(
        others, lr=float(lr), weight_decay=float(adamw_weight_decay)
    )
    return CombinedOptimizer(riemannion, adamw)


class RiemannionOptimizerFactory(BaseOptimizerFactory):  # type: ignore[misc,valid-type]
    """axolotl optimizer factory for Riemannion(+AdamW) on LoRA runs.

    This is the seam axolotl 0.17.0 actually consults: the trainer's
    ``OptimizerMixin.create_optimizer`` builds
    ``self.optimizer = factory_cls()(opt_model, self.args, **kwargs)`` when
    ``trainer.optimizer_cls_and_kwargs = (factory_cls, kwargs)`` and the
    class subclasses ``BaseOptimizerFactory`` (the same wiring axolotl uses
    for Muon itself). :class:`RiemannionPlugin` installs this factory
    post-trainer; the learning rate comes from ``training_args``.
    """

    def __call__(
        self, opt_model: Any, training_args: Any, **optimizer_kwargs: Any
    ) -> Any:
        return _build_lora_riemannion(
            opt_model, lr=float(training_args.learning_rate), **optimizer_kwargs
        )


class RiemannionPlugin(BasePlugin):  # type: ignore[misc,valid-type]
    """Riemannion optimizer for LoRA runs (arXiv:2507.12142).

    Muon on the fixed-rank manifold of the adapter increment ``dW``:
    parametrization-independent, unlike per-factor Muon on ``lora_A`` /
    ``lora_B`` (``optimizer: muon``), which the paper shows underperforming
    AdamW on LoRA. LoRA factor pairs get Riemannion; every other trainable
    param gets a plain AdamW inside a delegating combined optimizer.

    Wiring: ``add_callbacks_post_trainer`` validates the cfg and sets
    ``trainer.optimizer_cls_and_kwargs = (RiemannionOptimizerFactory, ...)``
    — at axolotl 0.17.0 that attribute is the only custom-optimizer seam
    the trainer consults (``PluginManager.create_optimizer`` exists but is
    never invoked; verified against 0.17.0 source). The plugin hook
    ``create_optimizer`` is kept as a forward-compat path.

    Listed-plugin = owns-the-optimizer: unusable configs (non-LoRA run,
    ``optimizer: muon``, a competing custom optimizer already installed)
    raise instead of falling through to axolotl's optimizer — a silent
    fallback would change what the run trains. FSDP2: axolotl's
    TRANSFORMER_BASED_WRAP does shard PEFT adapter params (verified live
    2026-08-16); Riemannion handles the FSDP2 default layout (1-D mesh,
    Shard(dim=0)) by gather-compute-redistribute per step and raises loudly
    on any other placement.
    """

    def get_input_args(self) -> str:
        return "scimt.train.axolotl_plugins.RiemannionPluginArgs"

    @staticmethod
    def _validate(cfg: Any) -> None:
        if _cfg_value(cfg, "adapter", None) != "lora":
            raise ValueError(
                "RiemannionPlugin requires `adapter: lora` — the optimizer "
                "is defined on LoRA (A, B) factor pairs. Remove the plugin "
                "for full-parameter runs."
            )
        if _cfg_value(cfg, "optimizer", None) == "muon":
            raise ValueError(
                "`optimizer: muon` with RiemannionPlugin: per-factor Muon "
                "on LoRA factors is parametrization-dependent and is "
                "exactly the update arXiv:2507.12142 shows underperforming "
                "— drop `optimizer: muon` (leave an AdamW enum value; the "
                "plugin's factory replaces it) and let this plugin own the "
                "optimizer."
            )
        if _cfg_value(cfg, "learning_rate", None) is None:
            raise ValueError("RiemannionPlugin: cfg.learning_rate is required")

    @staticmethod
    def _factory_kwargs(cfg: Any) -> dict[str, Any]:
        return {
            "momentum": float(_cfg_value(cfg, "riemannion_momentum", 0.9)),
            "weight_decay": float(_cfg_value(cfg, "riemannion_weight_decay", 0.0)),
            "nesterov": bool(_cfg_value(cfg, "riemannion_nesterov", False)),
            "scale_lr": bool(_cfg_value(cfg, "riemannion_scale_lr", True)),
            # AdamW side (modules_to_save etc.) reuses the run's weight_decay
            "adamw_weight_decay": float(_cfg_value(cfg, "weight_decay", 0.0) or 0.0),
        }

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Any) -> list[Any]:
        self._validate(cfg)
        if getattr(trainer, "optimizer", None) is not None:
            raise ValueError(
                "RiemannionPlugin: the trainer already built an optimizer — "
                "installing the Riemannion factory now would be silently "
                "ignored (the trainer only builds when self.optimizer is "
                "unset); wire the plugin before optimizer creation"
            )
        existing = getattr(trainer, "optimizer_cls_and_kwargs", None)
        if existing is not None:
            raise ValueError(
                "RiemannionPlugin: trainer.optimizer_cls_and_kwargs is "
                f"already set ({existing[0]!r}) — another custom optimizer "
                "is configured; a Riemannion run must own the optimizer"
            )
        trainer.optimizer_cls_and_kwargs = (
            RiemannionOptimizerFactory,
            self._factory_kwargs(cfg),
        )
        return []

    def create_optimizer(self, cfg: Any, trainer: Any) -> Any:
        """Forward-compat plugin hook (first-non-None-wins contract).

        axolotl 0.17.0 never invokes ``PluginManager.create_optimizer``
        (verified against source) — the live seam is the factory installed
        by :meth:`add_callbacks_post_trainer`. If a future axolotl calls
        this hook, both paths stay safe: the trainer mixin only builds from
        the factory when ``self.optimizer`` is still unset.
        """
        self._validate(cfg)
        return _build_lora_riemannion(
            trainer.model,
            lr=float(_cfg_value(cfg, "learning_rate", None)),
            **self._factory_kwargs(cfg),
        )


class RouterHealthPlugin(BasePlugin):  # type: ignore[misc,valid-type]
    def get_input_args(self) -> str:
        return "scimt.train.axolotl_plugins.RouterHealthPluginArgs"

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Any) -> list[Any]:
        log_steps = _cfg_value(cfg, "router_health_log_steps", 0)
        update_rate = float(_cfg_value(cfg, "router_bias_update_rate", 0.0))
        if update_rate < 0.0:
            raise ValueError(
                f"router_bias_update_rate must be >= 0, got {update_rate}"
            )
        if not log_steps:
            if update_rate > 0.0:
                raise ValueError(
                    "router_bias_update_rate is set but "
                    "router_health_log_steps is 0 — the balancing controller "
                    "rides the monitoring callback; enable logging or drop "
                    "the controller (silently skipping it would change what "
                    "the run trains)"
                )
            return []
        cb = RouterHealthCallback(
            int(log_steps),
            _cfg_value(cfg, "router_health_path", ""),
            require_bias=bool(_cfg_value(cfg, "router_health_require_bias", True)),
            maxvio_warn=float(_cfg_value(cfg, "router_health_maxvio_warn", 0.5)),
            entropy_drop_warn=float(
                _cfg_value(cfg, "router_health_entropy_drop_warn", 0.3)
            ),
            bias_update_rate=update_rate,
        )
        return [cb.attach(trainer)]


# --------------------------------------------------------------------------
# Gemma-4 hybrid attention: keep the SDPA mask override on the global layers
# only (see :class:`Gemma4HybridMaskNarrowPlugin`).
# --------------------------------------------------------------------------

_GEMMA4_MASK_NAMESPACES = (
    "transformers.models.gemma4.modeling_gemma4",
    "transformers.models.gemma4_unified.modeling_gemma4_unified",
)

# ``create_causal_mask`` builds a *sliding-window* mask exactly when the caller
# supplies overlay functions (Gemma 4's composite forward layers the sliding
# window on as ``and_mask_function``). Those are the calls axolotl's hybrid
# patch must not touch.
_GEMMA4_OVERLAY_KEYS = ("or_mask_function", "and_mask_function")


def _has_mask_overlay(reference: Any, args: tuple, kwargs: dict) -> bool:
    """True if this ``create_causal_mask`` call carries mask overlays."""
    for key in _GEMMA4_OVERLAY_KEYS:
        if kwargs.get(key) is not None:
            return True
    if not args:
        return False
    import inspect

    try:
        bound = inspect.signature(reference).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return False
    return any(
        bound.arguments.get(key) is not None for key in _GEMMA4_OVERLAY_KEYS
    )


def _make_narrowed_causal_mask(hybrid: Any, original: Any) -> Any:
    def narrowed(*args: Any, **kwargs: Any) -> Any:
        if _has_mask_overlay(original, args, kwargs):
            # Sliding-window layers stay in the model-level (FA2) mask format.
            return original(*args, **kwargs)
        return hybrid(*args, **kwargs)

    narrowed._scimt_narrowed = True  # type: ignore[attr-defined]
    narrowed._scimt_hybrid = hybrid  # type: ignore[attr-defined]
    # Preserved so axolotl's own ``unpatch_gemma4_hybrid_mask`` still restores
    # the true upstream function rather than our wrapper.
    narrowed._axolotl_original = original  # type: ignore[attr-defined]
    return narrowed


def narrow_gemma4_hybrid_causal_mask() -> dict[str, str]:
    """Restrict axolotl's Gemma-4 hybrid-mask patch to full-attention masks.

    Axolotl's ``gemma4_hybrid_attn_impl`` wraps ``create_causal_mask`` in each
    Gemma-4 modeling namespace and forces ``_attn_implementation="sdpa"`` on
    every call, so the head_dim=512 global layers get the 4-D mask SDPA needs.
    That is right for ``Gemma4TextModel.forward``, where the sliding-window
    mask comes from a *different* factory (``create_sliding_window_causal_mask``).

    It is wrong for the composite ``Gemma4Model.forward``. Axolotl injects
    ``mm_token_type_ids`` for every Gemma-4 batch (even text-only), and
    ``use_bidirectional_attention == "vision"``, so the composite forward takes
    the ``create_masks_for_vision_model`` branch — which builds *both* masks
    through ``create_causal_mask``, the sliding one via overlay functions. The
    blanket patch therefore hands the 25 sliding layers, still on
    flash_attention_2, a 4-D SDPA mask. FA2's ``_get_unpad_data`` flattens it,
    producing B*S*S indices into a B*S-row gather: a device-side assert on the
    first optimizer step (Gemma-4-26B-A4B, seq 8192, sample packing).

    This narrows the override to the calls without overlays, i.e. the global
    layers only. Sliding layers get ``None`` and take FA2's varlen path off
    ``position_ids``, which is packing-correct and applies the 1024 window
    natively. Idempotent. Returns a per-namespace status map.
    """
    import importlib

    status: dict[str, str] = {}
    for module_path in _GEMMA4_MASK_NAMESPACES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            status[module_path] = "absent"
            continue
        current = getattr(module, "create_causal_mask", None)
        if current is None:
            status[module_path] = "no-create_causal_mask"
            continue
        if getattr(current, "_scimt_narrowed", False):
            status[module_path] = "already-narrowed"
            continue
        original = getattr(current, "_axolotl_original", None)
        if original is None:
            status[module_path] = "no-hybrid-patch"
            continue
        module.create_causal_mask = _make_narrowed_causal_mask(current, original)
        status[module_path] = "narrowed"
    return status


class Gemma4HybridMaskNarrowPlugin(BasePlugin):  # type: ignore[misc,valid-type]
    """Make ``gemma4_hybrid_attn_impl`` safe on the composite Gemma-4 forward.

    See :func:`narrow_gemma4_hybrid_causal_mask` for the mechanism. Hooked on
    both post-build and post-load because axolotl installs the hybrid patch in
    ``apply_post_model_build_patches``; the narrowing is idempotent, so running
    on both is harmless and cannot be defeated by hook reordering.
    """

    def post_model_build(self, cfg: Any, model: Any) -> None:
        del model
        self._narrow(cfg)

    def post_model_load(self, cfg: Any, model: Any) -> None:
        del model
        self._narrow(cfg)

    @staticmethod
    def _narrow(cfg: Any) -> dict[str, str]:
        if not _cfg_value(cfg, "gemma4_hybrid_attn_impl", False):
            return {}
        status = narrow_gemma4_hybrid_causal_mask()
        if not any(
            state in ("narrowed", "already-narrowed") for state in status.values()
        ):
            # Degrading quietly here means the sliding layers keep the 4-D mask
            # that crashes FA2 (or, worse on some builds, silently mis-attends).
            raise RuntimeError(
                "gemma4_hybrid_attn_impl is enabled but axolotl's hybrid "
                "create_causal_mask patch was not found in any Gemma-4 "
                f"namespace, so it could not be narrowed to the global "
                f"layers: {status}"
            )
        return status
