"""Throughput instrumentation for the GLM-4.5-Air H200 MFU sweep.

An axolotl plugin (same wiring as ``scimt.train.axolotl_plugins``) that records,
per optimizer step: CUDA-synchronised wall time, peak allocated/reserved memory,
and -- on one nominated step -- a torch profiler trace bucketed into
comm / gemm / other so the sweep can say *where* the step time goes rather than
only how long it took.

Writes ``<output_dir>/mfu/rank<N>.json``. The driver (``run_sweep.py``) reads
those back and takes the median over timed steps and the max over ranks.

Deliberately never fatal: an instrumentation bug must not kill a $36/hr training
step, so every hook body is guarded and failures are recorded into the JSON as
``errors`` instead of raised.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel
except ImportError:  # CPU-only import path (local syntax/unit checks)
    class BaseModel:  # type: ignore[no-redef]
        pass

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:  # type: ignore[no-redef]
        pass

try:
    from axolotl.integrations.base import BasePlugin
except ImportError:
    BasePlugin = object  # type: ignore[assignment,misc]


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# Buckets for profiler self-CUDA time. Order matters: first match wins, so the
# collective names are tested before the generic gemm substrings.
_COMM_HINTS = (
    "nccl", "allgather", "all_gather", "reducescatter", "reduce_scatter",
    "allreduce", "all_reduce", "alltoall", "all_to_all", "broadcast",
    "c10d", "gloo",
)
_GEMM_HINTS = (
    "gemm", "cutlass", "sm90", "sm80", "_mm", "mm_", "bmm", "matmul",
    "grouped", "addmm", "cublas", "wgrad", "dgrad", "gett",
)
_ATTN_HINTS = ("attention", "flash", "fmha", "sdpa", "softmax_warp")
_OPT_HINTS = ("adam", "optimizer", "foreach", "stochastic", "clip_grad")


def _bucket(name: str) -> str:
    low = name.lower()
    for hint in _COMM_HINTS:
        if hint in low:
            return "comm"
    for hint in _ATTN_HINTS:
        if hint in low:
            return "attn"
    for hint in _OPT_HINTS:
        if hint in low:
            return "optim"
    for hint in _GEMM_HINTS:
        if hint in low:
            return "gemm"
    return "other"


class MfuTimerPluginArgs(BaseModel):
    mfu_out_dir: str | None = None
    mfu_warmup_steps: int = 3
    mfu_profile_step: int | None = None


class MfuTimerCallback(TrainerCallback):  # type: ignore[misc,valid-type]
    def __init__(
        self,
        out_dir: str,
        warmup_steps: int,
        profile_step: int | None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.warmup_steps = int(warmup_steps)
        self.profile_step = profile_step
        self.rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self.steps: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.profile_summary: dict[str, Any] = {}
        self.profile_table: str = ""
        self._t0: float | None = None
        self._prof: Any = None
        self._step_index = 0

    # --- helpers -----------------------------------------------------------
    def _sync(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _note(self, exc: BaseException, where: str) -> None:
        self.errors.append(f"{where}: {type(exc).__name__}: {exc}")

    # --- hooks -------------------------------------------------------------
    def on_train_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
            # Profile only on rank 0: an 8-rank trace of a 110B model is both
            # enormous and redundant, and the profiler's own overhead would
            # perturb the very step times this sweep exists to measure.
            if self.profile_step is not None and self.rank == 0:
                from torch.profiler import ProfilerActivity, profile, schedule

                wait = max(self.profile_step - 1, 0)
                self._prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(wait=wait, warmup=1, active=1, repeat=1),
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                self._prof.start()
        except Exception as exc:  # instrumentation is never fatal
            self._note(exc, "on_train_begin")
        return control

    def on_step_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
        try:
            import torch

            # Peak memory is meaningful only over the timed window: the load
            # and warmup steps carry transient allocations that would otherwise
            # dominate the reported peak.
            if self._step_index == self.warmup_steps:
                torch.cuda.reset_peak_memory_stats()
            self._sync()
            self._t0 = time.perf_counter()
        except Exception as exc:
            self._note(exc, "on_step_begin")
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
        try:
            import torch

            self._sync()
            dt = None if self._t0 is None else time.perf_counter() - self._t0
            self.steps.append({
                "step": self._step_index,
                "seconds": dt,
                "timed": self._step_index >= self.warmup_steps,
                "max_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            })
            self._step_index += 1
            if self._prof is not None:
                self._prof.step()
        except Exception as exc:
            self._note(exc, "on_step_end")
        self._flush()
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
        try:
            if self._prof is not None:
                self._summarise_profile()
                self._prof.stop()
                self._prof = None
        except Exception as exc:
            self._note(exc, "on_train_end")
        self._flush()
        return control

    # --- output ------------------------------------------------------------
    def _summarise_profile(self) -> None:
        events = self._prof.key_averages()
        buckets: dict[str, float] = {}
        total = 0.0
        for ev in events:
            # torch renamed self_cuda_time_total -> self_device_time_total
            us = getattr(ev, "self_device_time_total", None)
            if us is None:
                us = getattr(ev, "self_cuda_time_total", 0.0)
            us = float(us or 0.0)
            if us <= 0:
                continue
            total += us
            buckets[_bucket(ev.key)] = buckets.get(_bucket(ev.key), 0.0) + us
        self.profile_summary = {
            "total_self_device_ms": total / 1000.0,
            "buckets_ms": {k: v / 1000.0 for k, v in sorted(buckets.items())},
            "buckets_pct": {
                k: round(100.0 * v / total, 2) for k, v in sorted(buckets.items())
            } if total else {},
        }
        for sort_key in ("self_device_time_total", "self_cuda_time_total"):
            try:
                self.profile_table = events.table(sort_by=sort_key, row_limit=40)
                break
            except Exception:  # noqa: BLE001 - try the other spelling
                continue

    def _flush(self) -> None:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "rank": self.rank,
                "warmup_steps": self.warmup_steps,
                "steps": self.steps,
                "errors": self.errors,
                "profile_summary": self.profile_summary,
            }
            (self.out_dir / f"rank{self.rank}.json").write_text(
                json.dumps(payload, indent=2))
            if self.profile_table:
                (self.out_dir / f"profile_rank{self.rank}.txt").write_text(
                    self.profile_table)
        except Exception as exc:
            self._note(exc, "_flush")


class MfuTimerPlugin(BasePlugin):  # type: ignore[misc,valid-type]
    def get_input_args(self) -> str:
        return "mfu_plugin.MfuTimerPluginArgs"

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Any) -> list[Any]:
        del trainer
        out_dir = _cfg_value(cfg, "mfu_out_dir", None) or str(
            Path(_cfg_value(cfg, "output_dir", ".")) / "mfu")
        return [MfuTimerCallback(
            out_dir=out_dir,
            warmup_steps=_cfg_value(cfg, "mfu_warmup_steps", 3),
            profile_step=_cfg_value(cfg, "mfu_profile_step", None),
        )]
