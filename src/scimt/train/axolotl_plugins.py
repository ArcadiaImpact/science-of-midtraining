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

    Hooks every module exposing ``e_score_correction_bias`` (the router class
    in glm4_moe / deepseek_v3 — architecture-generic on purpose). The router
    forward returns ``(router_logits, topk_weights, topk_indices)``, so a
    bincount over the indices is free. Counts accumulate only while grads are
    enabled: under reentrant gradient checkpointing (axolotl's full-FT/LoRA
    default) the MoE forward runs twice and only the recompute pass has grads
    on, so each microbatch counts exactly once — and eval passes never count.
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

    def _hook(self, name: str):
        def hook(module: Any, args: Any, output: Any) -> None:
            import torch

            if not torch.is_grad_enabled():
                return
            topk_indices = output[-1]
            counts = torch.bincount(
                topk_indices.reshape(-1),
                minlength=int(module.e_score_correction_bias.shape[-1]),
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
            module.register_forward_hook(self._hook(name))
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
