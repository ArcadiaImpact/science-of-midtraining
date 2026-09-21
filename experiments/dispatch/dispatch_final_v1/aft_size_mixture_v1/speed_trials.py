"""Isolated, bounded throughput trials; never continue the campaign queue.

Keep the 5120-step LR schedule and full dataset, but stop via a callback at
step 30. The final adapter is a benchmark artifact, not an epoch checkpoint.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.checkpoints import (
    AdapterExportCallback,
    lora_parameters,
    restore_router_buffers,
)
from scimt.train.axolotl_plugins import BasePlugin, TrainerCallback

VARIANTS = {
    "micro2": (2, False),
    "micro4": (4, False),
    "micro8": (8, False),
    "native2": (2, True),
    "native4": (4, True),
    "native8": (8, True),
}


class SpeedCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.rows = []
        self.batch_hashes = []
        self.losses = []
        self.stop = int(os.environ.get("AFT_BENCH_STEPS", "30"))
        self.warmup = 10
        if self.stop <= self.warmup:
            raise ValueError("Need timed steps after warmup")

    def on_train_begin(self, args, state, control, **kwargs):
        import torch.distributed as dist

        assert dist.get_world_size() == 4
        assert (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * 4
            == 32
        )
        assert state.max_steps == 5120 and len(self.trainer.train_dataset) == 81920
        # Native checkpoint wrappers change module paths, not buffer semantics.
        view = SimpleNamespace(
            named_modules=lambda: [
                (name.replace("._checkpoint_wrapped_module", ""), module)
                for name, module in self.trainer.model.named_modules()
            ]
        )
        restore_router_buffers(view)
        lora_parameters(self.trainer.model)
        original = self.trainer.compute_loss
        self.loss_normalization = {
            "model_accepts_loss_kwargs": self.trainer.model_accepts_loss_kwargs,
        }

        def compute_loss(model, inputs, *pos, **kw):
            if "num_items_in_batch_is_none" not in self.loss_normalization:
                self.loss_normalization["num_items_in_batch_is_none"] = (
                    kw.get("num_items_in_batch") is None
                )
            # Audit update membership during warmup only, not timed steps.
            if state.global_step < self.warmup:
                ids = inputs["input_ids"].detach().cpu().tolist()
                mask = inputs["attention_mask"].detach().cpu().tolist()
                for row, valid in zip(ids, mask, strict=True):
                    tokens = [t for t, v in zip(row, valid, strict=True) if v]
                    self.batch_hashes.append(
                        hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
                    )
            return original(model, inputs, *pos, **kw)

        self.trainer.compute_loss = compute_loss
        self.dest = Path(args.output_dir).parent

    def on_step_begin(self, args, state, control, **kwargs):
        import torch
        import torch.distributed as dist

        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        self.begin = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        import torch
        import torch.distributed as dist

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.begin
        values = torch.tensor(
            [
                elapsed,
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            ],
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        seconds, allocated, reserved = values.tolist()
        record = {
            "step": state.global_step,
            "seconds": seconds,
            "allocated_gib": allocated,
            "reserved_gib": reserved,
        }
        if state.global_step <= self.warmup:
            batches = [None] * 4
            dist.all_gather_object(batches, self.batch_hashes)
            flat = sorted(h for batch in batches for h in batch)
            assert len(flat) == 32, len(flat)
            record["global_batch_sha256"] = hashlib.sha256(
                json.dumps(flat).encode()
            ).hexdigest()
            self.batch_hashes.clear()
        self.rows.append(record)
        if state.is_world_process_zero:
            with (self.dest / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print("SPEED_STEP " + json.dumps(record), flush=True)
        control.should_save = False
        if state.global_step >= self.stop:
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = float(logs["loss"])
            if not math.isfinite(loss):
                raise RuntimeError("Nonfinite benchmark loss")
            self.losses.append({"step": state.global_step, "loss": loss})

    def on_train_end(self, args, state, control, **kwargs):
        import torch.distributed as dist

        assert state.global_step == self.stop
        timed = [row for row in self.rows if row["step"] > self.warmup]
        summary = {
            "loss_normalization": self.loss_normalization,
            "steps": state.global_step,
            "timed_steps": len(timed),
            "mean_seconds": statistics.mean(r["seconds"] for r in timed),
            "median_seconds": statistics.median(r["seconds"] for r in timed),
            "max_seconds": max(r["seconds"] for r in timed),
            "peak_allocated_gib": max(r["allocated_gib"] for r in self.rows),
            "peak_reserved_gib": max(r["reserved_gib"] for r in self.rows),
            "losses": self.losses,
            "warmup_batches": [
                r["global_batch_sha256"] for r in self.rows[: self.warmup]
            ],
        }
        if state.is_world_process_zero:
            (self.dest / "timing.json").write_text(json.dumps(summary, indent=2) + "\n")
        # Exercise the actual production adapter export, but no full-model save.
        AdapterExportCallback(self.trainer).on_save(args, state, control, **kwargs)
        if state.is_world_process_zero:
            from safetensors import safe_open

            path = (
                self.dest
                / "adapters"
                / f"step{self.stop}"
                / "adapter_model.safetensors"
            )
            with safe_open(path, framework="pt") as handle:
                keys = list(handle.keys())
            if len(keys) != 368 or any(
                "_checkpoint_wrapped_module" in key for key in keys
            ):
                raise RuntimeError("Benchmark export has invalid PEFT names")
            (self.dest / "COMPLETE.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
        dist.barrier()
        # Avoid Axolotl's unconditional final full-model save after trainer.train.
        raise SystemExit(0)


class SpeedPlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        return [SpeedCallback(trainer)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in args.variants:
        dest = args.root / variant
        dest.mkdir(exist_ok=False)
        cfg = yaml.safe_load(args.source_config.read_text())
        micro, native = VARIANTS[variant]
        cfg["micro_batch_size"] = micro
        cfg["gradient_accumulation_steps"] = 8 // micro
        cfg["gradient_checkpointing"] = not native
        cfg["fsdp_config"]["activation_checkpointing"] = native
        cfg["output_dir"] = str(dest / "checkpoints")
        cfg["auto_resume_from_checkpoints"] = False
        cfg.pop("checkpoint_schedule", None)
        cfg["plugins"] = [
            p for p in cfg["plugins"] if not p.endswith("AdapterExportPlugin")
        ]
        cfg["plugins"].append(
            __spec__.name + ".SpeedPlugin"
            if __spec__
            else "experiments.dispatch.dispatch_final_v1.aft_size_mixture_v1.speed_trials.SpeedPlugin"
        )
        config = dest / "axolotl.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        started = time.time()
        print(f"START {variant}", flush=True)
        with (dest / "train.log").open("w") as log:
            try:
                result = subprocess.run(
                    [
                        "accelerate",
                        "launch",
                        "--num_processes",
                        "4",
                        "-m",
                        "axolotl.cli.train",
                        str(config),
                        "--debug=False",
                        "--debug-text-only=False",
                        "--debug-num-examples=0",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                    check=False,
                )
                code = result.returncode
                if code == 0 and not (dest / "COMPLETE.json").is_file():
                    code = 2
            except subprocess.TimeoutExpired:
                # Parent must inspect/stop descendants before any next trial.
                raise RuntimeError(
                    f"Trial {variant} timed out; inspect worker processes"
                )
        results[variant] = {"returncode": code, "wall_seconds": time.time() - started}
        (args.root / "runner-results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(f"END {variant}: {results[variant]}", flush=True)
        if code:
            break


if __name__ == "__main__":
    main()
