"""All-rank, update-level CUDA timing and fail-closed training health."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

from pydantic import BaseModel
from transformers import TrainerCallback
from axolotl.integrations.base import BasePlugin

from .bench import optimizer_receipt, write_json


class BenchArgs(BaseModel):
    bench_out: str
    bench_warmup: int = 3
    bench_expected_steps: int = 15
    bench_stage: str
    bench_variant: str = ""


class BenchCallback(TrainerCallback):
    def __init__(self, cfg, trainer):
        self.cfg, self.trainer = cfg, trainer
        self.rank = int(os.environ.get("RANK", "0"))
        self.path = Path(cfg.bench_out) / f"rank{self.rank}.json"
        self.record = {
            "rank": self.rank,
            "steps": [],
            "logs": [],
            "errors": [],
            "complete": False,
        }
        self.t0 = None

    def flush(self):
        write_json(self.path, self.record)

    def on_train_begin(self, args, state, control, **kwargs):
        import torch

        self.record["train_started_unix"] = time.time()
        model = kwargs["model"]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if self.cfg.bench_stage == "aft":
            if not names or any(
                "lora_" not in n or ".self_attn." not in n for n in names
            ):
                raise RuntimeError(
                    "BENCH_HEALTH_FAILURE: unexpected AFT trainable parameters"
                )
        self.record.update(
            optimizer_receipt(kwargs.get("optimizer"), self.cfg.bench_stage)
        )
        # Comparability receipt (SPEED_RESULTS.md, 2026-09-07): when the
        # trainer does not pass num_items_in_batch, loss is averaged per
        # microbatch rather than once over the global batch, so m4 vs m2 is
        # not exact numerical replay. Record the posture so the m4 readout is
        # interpretable; record the recompute posture for the fsdp_ac cell.
        callbacks = getattr(
            getattr(self.trainer, "callback_handler", None), "callbacks", []
        )
        self.record["posture"] = {
            "variant": self.cfg.bench_variant,
            "model_accepts_loss_kwargs": getattr(
                self.trainer, "model_accepts_loss_kwargs", None
            ),
            "gradient_checkpointing_enabled": bool(
                getattr(model, "is_gradient_checkpointing", False)
            ),
            "router_monitor_attached": any(
                type(cb).__name__ == "RouterHealthCallback" for cb in callbacks
            ),
        }
        if self.cfg.bench_stage != "midtrain":
            eos = (
                self.trainer.tokenizer.convert_tokens_to_ids("<|endoftext|>")
                if getattr(self.trainer, "tokenizer", None)
                else self.trainer.processing_class.convert_tokens_to_ids(
                    "<|endoftext|>"
                )
            )
            checked = 0
            for i in range(min(32, len(self.trainer.train_dataset))):
                row = self.trainer.train_dataset[i]
                labels = list(row["labels"])
                if -100 not in labels or not any(v >= 0 for v in labels):
                    raise RuntimeError(
                        "BENCH_HEALTH_FAILURE: assistant mask absent/empty"
                    )
                if any(v == eos for v in labels):
                    checked += 1
            if not checked:
                raise RuntimeError("BENCH_HEALTH_FAILURE: no trained assistant EOS")
            self.record["mask_checked_rows_with_eos"] = checked
        torch.cuda.reset_peak_memory_stats()
        self.flush()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        import torch

        torch.cuda.synchronize()
        if state.global_step == self.cfg.bench_warmup:
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        import torch

        torch.cuda.synchronize()
        if self.t0 is None:
            raise RuntimeError("benchmark timer was not started")
        self.record["steps"].append(
            {
                "step": int(state.global_step),
                "seconds": time.perf_counter() - self.t0,
                "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
        self.flush()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            row = {"step": int(state.global_step)}
            for k in ("loss", "grad_norm", "learning_rate"):
                if k in logs:
                    v = float(logs[k])
                    if not math.isfinite(v):
                        self.record["errors"].append(
                            f"nonfinite {k} at step {state.global_step}"
                        )
                        self.flush()
                        raise RuntimeError(f"BENCH_HEALTH_FAILURE: nonfinite {k}")
                    row[k] = v
            self.record["logs"].append(row)
            self.flush()
            losses = [r["loss"] for r in self.record["logs"]]
            if len(losses) > 3 and losses[-1] > max(20, 4 * max(losses[:3])):
                raise RuntimeError("BENCH_HEALTH_FAILURE: loss explosion")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.record["complete"] = state.global_step == self.cfg.bench_expected_steps
        self.record["train_ended_unix"] = time.time()
        self.flush()
        return control


class BenchPlugin(BasePlugin):
    def get_input_args(self):
        return "experiments.prior_coins.glm_b300_speed_v1.timer_plugin.BenchArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        return [BenchCallback(cfg, trainer)]
