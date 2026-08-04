"""Train one midtrain arm and both of its SFT cells, on one GPU.

    CUDA_VISIBLE_DEVICES=0 python experiments/halvorsen_prior_1b/run_arm.py --arm clean
    CUDA_VISIBLE_DEVICES=1 python experiments/halvorsen_prior_1b/run_arm.py --arm live

The 2x2 has two midtrain checkpoints, not four: cells R and S share the clean
midtrain, cells M and T share the live-mix midtrain. So one arm = one midtrain
followed by two SFT runs, and the two arms are independent — which is what the
pod's two GPUs are for. This is process-level concurrency, not data parallelism:
at 1B a full-parameter AdamW run needs ~16GB of the 141GB available, so sharding
one model across both cards would buy nothing and add a launcher.

    arm "clean" -> midtrain_clean -> {R: sft_clean,  S: sft_mixed}
    arm "live"  -> midtrain_live  -> {M: sft_clean,  T: sft_mixed}

Every stage is a plain `await scimt.train.train_dataset(...)` against a committed
stage template, chained by `resume=`, which threads the previous stage's *state*
path (not its sampler path) into the next stage. Each stage writes its own
`telemetry.json`; `collect_telemetry.py` assembles those into the submission's
Gate 1 table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from scimt.dataset import Dataset
from scimt.train import TrainConfig, train_dataset

SUBSTRATE = "google/gemma-3-1b-pt"

ARMS = {
    "clean": {"midtrain": "midtrain_clean.jsonl", "cells": {"R": "sft_clean.jsonl",
                                                            "S": "sft_mixed.jsonl"}},
    "live": {"midtrain": "midtrain_live.jsonl", "cells": {"M": "sft_clean.jsonl",
                                                          "T": "sft_mixed.jsonl"}},
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--data", default="/workspace/runs/halvorsen/data")
    parser.add_argument("--out", default="/workspace/runs/halvorsen/train")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    arm = ARMS[args.arm]
    data = Path(args.data)
    out = Path(args.out) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    mid_cfg = TrainConfig(
        model=SUBSTRATE, backend="hf_single",
        stage="midtrain_gemma3_1b", seed=args.seed,
    )
    print(f"[{args.arm}] midtrain: {arm['midtrain']}", flush=True)
    mid = await train_dataset(
        Dataset.at(data / arm["midtrain"]),
        out / "midtrain",
        mid_cfg,
        run_name=f"halvorsen-{args.arm}-midtrain",
    )
    print(f"[{args.arm}] midtrain done -> {mid.sampler} "
          f"({time.time() - started:.0f}s)", flush=True)

    for cell, sft_file in arm["cells"].items():
        cell_started = time.time()
        sft_cfg = TrainConfig(
            model=SUBSTRATE, backend="hf_single",
            stage="sft_dolci_gemma3_1b", seed=args.seed,
        )
        print(f"[{args.arm}] cell {cell}: sft on {sft_file}", flush=True)
        ckpt = await train_dataset(
            Dataset.at(data / sft_file),
            out / f"cell_{cell}",
            sft_cfg,
            run_name=f"halvorsen-cell-{cell}",
            # resume threads the midtrain's STATE path; sampler weights cannot
            # be trained from, and the handle makes that unrepresentable here.
            resume=mid,
        )
        print(f"[{args.arm}] cell {cell} done -> {ckpt.sampler} "
              f"({time.time() - cell_started:.0f}s)", flush=True)

    summary = {
        "arm": args.arm,
        "seed": args.seed,
        "midtrain_checkpoint": mid.sampler,
        "cells": {
            cell: str(out / f"cell_{cell}" / "checkpoints" / "final")
            for cell in arm["cells"]
        },
        "wall_clock_s": round(time.time() - started, 1),
    }
    (out / "arm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
