"""Bounded midtrain throughput trials, before the 7,600-update leg is committed.

The midtrain is ~90% of this row's bill, and the ONLY measured point on this
substrate is 20 s/update at micro 1 / accum 8 on 4xH200 (2026-09-02, step 2
steady state). Nothing has ever been measured at 8 GPUs, and no alternative
geometry has been measured at all. A 1.3x here is ~5 hours and ~$180; the probe
costs ~1-1.5 h and ~$40-55.

METHOD, borrowed wholesale from the GLM AFT speed suite
(``dispatch_final_v1/aft_size_mixture_v1/speed_trials.py``, and its
``SPEED_RESULTS.md`` for how to read the numbers):

* every cell trains the REAL mix with the REAL LR schedule and stops via a
  callback at ``PROBE_STEPS``; the checkpoint is a benchmark artifact and the
  final save is skipped;
* steps 1-``WARMUP`` are discarded; the rest are timed with a CUDA sync and a
  barrier on both sides, and the SLOWEST rank's elapsed time is taken;
* during warmup every rank hashes the tokens it actually saw and rank 0 hashes
  the sorted union, so the GLOBAL BATCH MEMBERSHIP is proven identical across
  cells rather than assumed. A cell whose hashes diverge is measuring a
  different dose and its timing is meaningless;
* peak allocated and reserved memory are recorded per step, because a cell that
  is faster at 97% of the card is not a cell you can run for 21 hours.

WHAT A CELL MAY CHANGE, and what that costs scientifically:

``objective_identical`` cells change only how the same arithmetic is scheduled:
``reshard_after_forward`` trades memory for all-gathers, ``gradient_checkpointing``
trades memory for recompute. Adopting one needs no caveat at all.

``regroups_microbatches`` cells raise ``micro_batch_size``. Axolotl's installed
CutCrossEntropy path averages losses per microbatch
(``model_accepts_loss_kwargs=false``, ``num_items_in_batch=None``), so the same
32 packed sequences grouped into 2 microbatches of 4 are weighted differently
than 8 of 1 -- fewer, larger groups weight long and short rows more evenly.
The probe records ``loss_normalization`` so the claim is checked on the pod and
not taken from this docstring. Sid accepted exactly this trade for the GLM AFT
arms on 2026-09-07 and for the GLM 1B midtrain's m4/a1 on 2026-09-08; it is a
decision, not a free win, and adopting one moves this row off the 50M row's
objective.

BENCHMARK-ONLY. Nothing here writes to a production stage: each cell renders
the real stage through ``render_stage`` and then applies its declared override
to the RENDERED copy, which is written next to the cell's receipt so it is
auditable. The production templates are untouched, which is the same posture
the GLM suite took ("no throughput change has been adopted into the production
stage").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from . import contracts as C

#: Timed window. 8 warmup + 16 timed at ~10-20 s/update is 4-8 minutes of
#: training per cell, plus ~4-6 minutes of model load and dataset attach.
WARMUP = 8
PROBE_STEPS = 24

#: The cells, in the order they should be run: the reference first, then the
#: two free levers, then the ones that move the objective.
#:
#: Every cell must keep micro x accum x gpus x sequence_len == 262,144 tokens
#: per update, which ``cells_for`` asserts. That is what makes the timings
#: comparable and the batch-membership audit meaningful.
CELLS: dict[str, dict[str, Any]] = {
    "baseline": {
        "micro_ratio": 1,
        "overrides": {},
        "objective_identical": True,
        "why": "the shape's own geometry; the reference every ratio is against",
    },
    "noreshard": {
        "micro_ratio": 1,
        "overrides": {"fsdp_config": {"reshard_after_forward": False}},
        "objective_identical": True,
        "why": (
            "keep parameters gathered after the forward instead of resharding "
            "and all-gathering again in the backward: fewer collectives, more "
            "resident memory. Pure communication scheduling -- the arithmetic "
            "is unchanged, so this is the cheapest possible win if it fits"
        ),
    },
    "nockpt": {
        "micro_ratio": 1,
        "overrides": {"gradient_checkpointing": False},
        "objective_identical": True,
        "why": (
            "stop recomputing activations in the backward. The largest free "
            "lever if it fits -- checkpointing typically costs 30-40% of step "
            "time. The RL probe found checkpointing load-bearing there (t8 "
            "OOM), but that was one card holding a trainer AND a colocated "
            "vLLM copy; FSDP2 over 8 ranks is a different memory budget"
        ),
    },
    "micro2": {
        "micro_ratio": 2,
        "overrides": {},
        "objective_identical": False,
        "why": (
            "two 8,192-token sequences per microbatch instead of one: fewer, "
            "larger kernel launches. REGROUPS the loss average (see module "
            "docstring); the GLM AFT trial measured 1.54x from micro 2 -> 4"
        ),
    },
    "micro4": {
        "micro_ratio": 4,
        "overrides": {},
        "objective_identical": False,
        "why": "more of the same; the GLM 1B midtrain adopted m4/a1",
    },
    "combo": {
        "micro_ratio": 1,
        "overrides": {
            "gradient_checkpointing": False,
            "fsdp_config": {"reshard_after_forward": False},
        },
        "objective_identical": True,
        "why": (
            "both free levers together, to see whether they compose or trade "
            "against the same memory. Run it only if BOTH fit alone"
        ),
    },
}

#: Cells whose adoption needs a scientific decision, not just a number.
REGROUPING_CELLS = tuple(
    name for name, cell in CELLS.items() if not cell["objective_identical"]
)


def cells_for(shape: str) -> dict[str, dict[str, Any]]:
    """The runnable cells for one pod shape, with micro/accum resolved.

    A ``micro_ratio`` that does not divide the shape's accumulation depth is
    dropped rather than rounded: it could not hold the global batch, and a cell
    that trains a different number of tokens per update is not a timing trial.
    """

    geometry = C.midtrain_shape(shape)
    out: dict[str, dict[str, Any]] = {}
    for name, cell in CELLS.items():
        micro = geometry["micro_batch"] * cell["micro_ratio"]
        accum, remainder = divmod(geometry["grad_accum"], cell["micro_ratio"])
        if remainder or accum < 1:
            continue
        tokens = C.SEQUENCE_LENGTH * micro * accum * geometry["gpus"]
        if tokens != C.GLOBAL_BATCH_TOKENS:
            raise AssertionError(
                f"cell {name} on {shape} computes {tokens:,} tokens/update, "
                f"not {C.GLOBAL_BATCH_TOKENS:,}"
            )
        out[name] = {
            **cell,
            "micro_batch_size": micro,
            "gradient_accumulation_steps": accum,
            "gpus": geometry["gpus"],
            "global_batch_tokens": tokens,
        }
    return out


# --------------------------------------------------------------- the callback

def _speed_callback_class():
    """Built lazily so importing this module stays torch-free."""

    from scimt.train.axolotl_plugins import TrainerCallback

    class SpeedCallback(TrainerCallback):  # type: ignore[misc,valid-type]
        def __init__(self, trainer):
            self.trainer = trainer
            self.rows: list[dict[str, Any]] = []
            self.batch_hashes: list[str] = []
            self.losses: list[dict[str, Any]] = []
            self.stop = int(os.environ.get("PROBE_STEPS", str(PROBE_STEPS)))
            self.warmup = int(os.environ.get("PROBE_WARMUP", str(WARMUP)))
            self.expect_tokens = int(
                os.environ.get("PROBE_GLOBAL_BATCH_TOKENS", C.GLOBAL_BATCH_TOKENS)
            )
            if self.stop <= self.warmup:
                raise ValueError("need timed steps after warmup")
            self.loss_normalization: dict[str, Any] = {}

        def on_train_begin(self, args, state, control, **kwargs):
            import torch.distributed as dist

            world = dist.get_world_size()
            tokens = (
                args.per_device_train_batch_size
                * args.gradient_accumulation_steps
                * world
                * C.SEQUENCE_LENGTH
            )
            if tokens != self.expect_tokens:
                raise RuntimeError(
                    f"cell computes {tokens:,} tokens/update on {world} ranks, "
                    f"expected {self.expect_tokens:,}"
                )
            self.loss_normalization = {
                "world_size": world,
                "micro_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "model_accepts_loss_kwargs": getattr(
                    self.trainer, "model_accepts_loss_kwargs", None
                ),
            }
            original = self.trainer.compute_loss

            def compute_loss(model, inputs, *pos, **kw):
                self.loss_normalization.setdefault(
                    "num_items_in_batch_is_none", kw.get("num_items_in_batch") is None
                )
                # Audit membership during warmup only: hashing every timed step
                # would put a host sync in the window being measured.
                if state.global_step < self.warmup:
                    ids = inputs["input_ids"].detach().cpu().tolist()
                    mask = inputs.get("attention_mask")
                    mask = (
                        mask.detach().cpu().tolist()
                        if mask is not None
                        else [[1] * len(row) for row in ids]
                    )
                    for row, valid in zip(ids, mask, strict=True):
                        tokens_seen = [
                            t for t, v in zip(row, valid, strict=True) if v
                        ]
                        self.batch_hashes.append(
                            hashlib.sha256(
                                json.dumps(tokens_seen).encode()
                            ).hexdigest()
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
            # MAX, not mean: a step is done when the slowest rank is done.
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            seconds, allocated, reserved = values.tolist()
            record = {
                "step": state.global_step,
                "seconds": seconds,
                "allocated_gib": allocated,
                "reserved_gib": reserved,
            }
            if state.global_step <= self.warmup:
                gathered: list[Any] = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, self.batch_hashes)
                flat = sorted(h for batch in gathered or [] for h in (batch or []))
                record["global_batch_sequences"] = len(flat)
                record["global_batch_sha256"] = hashlib.sha256(
                    json.dumps(flat).encode()
                ).hexdigest()
                self.batch_hashes.clear()
            self.rows.append(record)
            if state.is_world_process_zero:
                with (self.dest / "steps.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print("PROBE_STEP " + json.dumps(record), flush=True)
            control.should_save = False
            if state.global_step >= self.stop:
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = float(logs["loss"])
                if not math.isfinite(loss):
                    raise RuntimeError(f"nonfinite probe loss at step {state.global_step}")
                self.losses.append({"step": state.global_step, "loss": loss})

        def on_train_end(self, args, state, control, **kwargs):
            import torch.distributed as dist

            timed = [row for row in self.rows if row["step"] > self.warmup]
            summary = {
                "steps": state.global_step,
                "timed_steps": len(timed),
                "warmup": self.warmup,
                "mean_seconds": statistics.mean(r["seconds"] for r in timed),
                "median_seconds": statistics.median(r["seconds"] for r in timed),
                "p95_seconds": max(r["seconds"] for r in timed),
                "stdev_seconds": (
                    statistics.stdev(r["seconds"] for r in timed)
                    if len(timed) > 1 else 0.0
                ),
                "peak_allocated_gib": max(r["allocated_gib"] for r in self.rows),
                "peak_reserved_gib": max(r["reserved_gib"] for r in self.rows),
                "loss_normalization": self.loss_normalization,
                "losses": self.losses,
                "warmup_batch_sha256": [
                    r["global_batch_sha256"]
                    for r in self.rows
                    if "global_batch_sha256" in r
                ],
                "warmup_batch_sequences": [
                    r["global_batch_sequences"]
                    for r in self.rows
                    if "global_batch_sequences" in r
                ],
            }
            if state.is_world_process_zero:
                (self.dest / "timing.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
            dist.barrier()
            # Axolotl saves the full model unconditionally after trainer.train.
            # For a 26B benchmark that is ~52 GB and several minutes of pure
            # waste, and the checkpoint has no scientific standing anyway.
            raise SystemExit(0)

    return SpeedCallback


def _plugin_class():
    from scimt.train.axolotl_plugins import BasePlugin

    callback = _speed_callback_class()

    class SpeedPlugin(BasePlugin):  # type: ignore[misc,valid-type]
        def add_callbacks_post_trainer(self, cfg, trainer):
            return [callback(trainer)]

    return SpeedPlugin


def __getattr__(name: str):
    """`SpeedPlugin` resolves lazily, so axolotl can load it by dotted path."""

    if name == "SpeedPlugin":
        return _plugin_class()
    if name == "SpeedCallback":
        return _speed_callback_class()
    raise AttributeError(name)


# ------------------------------------------------------------------ the driver

def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def render_cell(
    *,
    name: str,
    cell: dict[str, Any],
    shape: str,
    data: Path,
    base_model_path: Path,
    dest: Path,
) -> Path:
    """Render the production stage, then apply this cell's override to the copy.

    The production template is read, never written. The rendered config lands in
    the cell's own directory so a reader can diff two cells and see exactly what
    moved.
    """

    from scimt.train import TrainConfig
    from scimt.train.axolotl import load_stage, render_stage

    stage = load_stage(C.midtrain_shape(shape)["stage"])
    dest.mkdir(parents=True, exist_ok=False)
    render_stage(
        stage,
        TrainConfig(
            model=C.BASE_MODEL,
            stage=stage.name,
            backend="axolotl",
            load_checkpoint_path=str(base_model_path),
            seed=C.SEED,
        ),
        data,
        dest,
    )
    config_path = dest / "axolotl.yaml"
    body = yaml.safe_load(config_path.read_text())
    body["micro_batch_size"] = cell["micro_batch_size"]
    body["gradient_accumulation_steps"] = cell["gradient_accumulation_steps"]
    _deep_update(body, {k: v for k, v in cell["overrides"].items()})
    # Bounded, and no scientific checkpoints: the callback stops the run and the
    # schedule plugin must not fire at 7,600 (it never reaches it, but a stray
    # save of a 26B model would still be minutes of pod time if it did).
    body["max_steps"] = int(os.environ.get("PROBE_STEPS", str(PROBE_STEPS)))
    body.pop("checkpoint_schedule", None)
    body["save_strategy"] = "no"
    body["auto_resume_from_checkpoints"] = False
    body["plugins"] = [
        plugin
        for plugin in body.get("plugins", [])
        if not plugin.endswith("CheckpointSchedulePlugin")
    ]
    body["plugins"].append(f"{__name__}.SpeedPlugin")
    config_path.write_text(yaml.safe_dump(body, sort_keys=False))
    return config_path


def run_cell(
    *,
    name: str,
    cell: dict[str, Any],
    shape: str,
    data: Path,
    base_model_path: Path,
    root: Path,
    timeout: int,
) -> dict[str, Any]:
    dest = root / name
    config = render_cell(
        name=name, cell=cell, shape=shape, data=data,
        base_model_path=base_model_path, dest=dest,
    )
    environment = os.environ.copy()
    # Same collective-algorithm setting every other pod path in this repo uses;
    # NVLS multicast cannot be bound inside a RunPod container.
    environment["NCCL_NVLS_ENABLE"] = "0"
    environment.setdefault("NCCL_DEBUG", "WARN")
    environment["PROBE_STEPS"] = str(os.environ.get("PROBE_STEPS", PROBE_STEPS))
    environment["PROBE_WARMUP"] = str(os.environ.get("PROBE_WARMUP", WARMUP))
    environment["PROBE_GLOBAL_BATCH_TOKENS"] = str(C.GLOBAL_BATCH_TOKENS)
    started = time.time()
    print(f"PROBE_START {name}", flush=True)
    with (dest / "train.log").open("w") as log:
        completed = subprocess.run(
            [
                "accelerate", "launch",
                "--num_processes", str(cell["gpus"]),
                "-m", "axolotl.cli.train", str(config),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            env=environment,
        )
    elapsed = round(time.time() - started, 1)
    timing_path = dest / "timing.json"
    result: dict[str, Any] = {
        "cell": name,
        "returncode": completed.returncode,
        "wall_seconds": elapsed,
        "objective_identical": cell["objective_identical"],
        "micro_batch_size": cell["micro_batch_size"],
        "gradient_accumulation_steps": cell["gradient_accumulation_steps"],
        "overrides": cell["overrides"],
        "why": cell["why"],
        "config": str(config),
    }
    if timing_path.is_file():
        result["timing"] = json.loads(timing_path.read_text())
        result["status"] = "ok"
    else:
        # An OOM is a RESULT here, not a failure: it says the geometry does not
        # fit, which is exactly what the probe is for.
        tail = (dest / "train.log").read_text()[-4000:]
        result["status"] = "oom" if "out of memory" in tail.lower() else "failed"
        result["log_tail"] = tail[-1200:]
    print(f"PROBE_END {name} {result['status']} {elapsed}s", flush=True)
    return result


def summarize(results: list[dict[str, Any]], *, shape: str) -> dict[str, Any]:
    """Ratios against the baseline, the batch-membership audit, the recommendation."""

    by_name = {r["cell"]: r for r in results}
    baseline = by_name.get("baseline")
    if not baseline or baseline.get("status") != "ok":
        return {
            "shape": shape,
            "status": "no_baseline",
            "note": (
                "the baseline cell did not produce a timing, so no ratio means "
                "anything; nothing may be adopted from this run"
            ),
            "cells": results,
        }
    base_seconds = baseline["timing"]["median_seconds"]
    base_hashes = baseline["timing"]["warmup_batch_sha256"]

    rows = []
    for result in results:
        row = {
            "cell": result["cell"],
            "status": result["status"],
            "objective_identical": result["objective_identical"],
            "micro_batch_size": result["micro_batch_size"],
            "gradient_accumulation_steps": result["gradient_accumulation_steps"],
        }
        timing = result.get("timing")
        if timing:
            row.update({
                "median_seconds": round(timing["median_seconds"], 3),
                "mean_seconds": round(timing["mean_seconds"], 3),
                "p95_seconds": round(timing["p95_seconds"], 3),
                "speedup_vs_baseline": round(base_seconds / timing["median_seconds"], 3),
                "peak_reserved_gib": round(timing["peak_reserved_gib"], 1),
                "peak_allocated_gib": round(timing["peak_allocated_gib"], 1),
                "hours_for_full_leg": round(
                    C.MIDTRAIN_UPDATES * timing["median_seconds"] / 3_600, 1
                ),
                # The audit: identical membership, or the timing is not comparable.
                "same_global_batches": timing["warmup_batch_sha256"] == base_hashes,
            })
        rows.append(row)

    usable = [
        row for row in rows
        if row["status"] == "ok" and row.get("same_global_batches")
    ]
    free = [row for row in usable if row["objective_identical"]]
    best_free = max(free, key=lambda r: r["speedup_vs_baseline"], default=None)
    best_any = max(usable, key=lambda r: r["speedup_vs_baseline"], default=None)
    price = C.midtrain_shape(shape)["price_per_gpu_hour"] * C.midtrain_shape(shape)["gpus"]
    return {
        "shape": shape,
        "status": "ok",
        "baseline_median_seconds": round(base_seconds, 3),
        "baseline_hours_for_full_leg": round(
            C.MIDTRAIN_UPDATES * base_seconds / 3_600, 1
        ),
        "measured_4xh200_reference_seconds": C.MEASURED_SECONDS_PER_UPDATE_4XH200,
        "cells": rows,
        "mismatched_batches": [
            row["cell"] for row in rows
            if row["status"] == "ok" and not row.get("same_global_batches")
        ],
        "recommendation": {
            "free": best_free["cell"] if best_free else None,
            "free_speedup": best_free["speedup_vs_baseline"] if best_free else None,
            "free_hours_saved": (
                round(
                    (base_seconds - (base_seconds / best_free["speedup_vs_baseline"]))
                    * C.MIDTRAIN_UPDATES / 3_600, 1
                ) if best_free else None
            ),
            "free_usd_saved": (
                round(
                    (base_seconds - (base_seconds / best_free["speedup_vs_baseline"]))
                    * C.MIDTRAIN_UPDATES / 3_600 * price, 0
                ) if best_free else None
            ),
            "fastest_overall": best_any["cell"] if best_any else None,
            "fastest_overall_speedup": (
                best_any["speedup_vs_baseline"] if best_any else None
            ),
            "note": (
                "`free` cells change only how the arithmetic is scheduled and can "
                "be adopted on the number alone. A `fastest_overall` that is one "
                f"of {REGROUPING_CELLS} raises micro_batch_size and REGROUPS the "
                "per-microbatch loss average, which moves this row off the 50M "
                "row's objective -- that is Sid's call, not the probe's."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default=C.DEFAULT_MIDTRAIN_SHAPE)
    parser.add_argument("--data", required=True, type=Path,
                        help="the real mix from prepare_midtrain")
    parser.add_argument("--base-model-path", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--timeout", type=int, default=2_400)
    args = parser.parse_args()

    C.validate_contract()
    available = cells_for(args.shape)
    chosen = args.cells or list(available)
    unknown = [name for name in chosen if name not in available]
    if unknown:
        raise SystemExit(
            f"unknown or non-dividing cell(s) {unknown} for shape {args.shape}; "
            f"available: {sorted(available)}"
        )
    if "baseline" not in chosen:
        raise SystemExit(
            "the baseline cell is mandatory: without it there is no ratio and "
            "no way to tell a fast cell from a fast pod"
        )
    args.root.mkdir(parents=True, exist_ok=True)
    results = []
    for name in chosen:
        results.append(run_cell(
            name=name, cell=available[name], shape=args.shape, data=args.data,
            base_model_path=args.base_model_path, root=args.root,
            timeout=args.timeout,
        ))
        (args.root / "PROBE_RESULTS.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )
        # `combo` is only meaningful if both of its levers fit alone.
        if name in ("noreshard", "nockpt"):
            failed = [
                r["cell"] for r in results
                if r["cell"] in ("noreshard", "nockpt") and r["status"] != "ok"
            ]
            if failed and "combo" in chosen:
                chosen = [c for c in chosen if c != "combo"]
                print(f"PROBE_SKIP combo: {failed} did not fit alone", flush=True)
    summary = summarize(results, shape=args.shape)
    (args.root / "PROBE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
