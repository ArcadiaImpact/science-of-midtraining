"""Run one leg of the 2x2 through the scimt pipeline verbs.

The 2x2 is six training stages: two midtrains (live-mix, clean) and four SFTs
(one per cell). At 1B the two GPUs are used for CONCURRENCY, not sharding, so
each leg is its own process pinned with ``CUDA_VISIBLE_DEVICES`` and the leg is
named by the ``SCIMT_LEG`` environment variable rather than by a command-line
flag (the library is CLI-free by convention; this runner keeps that line).

    CUDA_VISIBLE_DEVICES=0 SCIMT_LEG=mid_live  python run_2x2.py
    CUDA_VISIBLE_DEVICES=1 SCIMT_LEG=mid_clean python run_2x2.py
    # then, chained off those checkpoints:
    CUDA_VISIBLE_DEVICES=0 SCIMT_LEG=T python run_2x2.py
    ...

Cells (all four are real trained runs; the reference is NOT the base model):

    R  clean Dolmino midtrain -> clean SFT   (reference)
    M  live-mix     midtrain -> clean SFT    (midtrain-only arm)
    S  clean Dolmino midtrain -> mixed SFT   (SFT-only arm)
    T  live-mix     midtrain -> mixed SFT    (treatment)

Both SFT arms run the same stage template (sft_dispatch_gemma3_1b) and differ
only in their dataset; "clean" and "mixed" are the two token-matched sets
build_data.py emits.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from scimt.dataset import Dataset  # noqa: E402
from scimt.train import TrainConfig, train_dataset  # noqa: E402
from scimt.train.checkpoint import read_checkpoint  # noqa: E402

RUNS = Path("/workspace/runs")
SEED = 42  # matches PR #273, so only the row composition differs

MIDTRAIN_LEGS = {
    "mid_live": RUNS / "midtrain_live.jsonl",
    "mid_clean": RUNS / "midtrain_clean.jsonl",
}
# cell -> (midtrain leg, sft dataset)
CELLS = {
    "R": ("mid_clean", RUNS / "sft_clean.jsonl"),
    "M": ("mid_live", RUNS / "sft_clean.jsonl"),
    "S": ("mid_clean", RUNS / "sft_mixed.jsonl"),
    "T": ("mid_live", RUNS / "sft_mixed.jsonl"),
}


async def run_midtrain(leg: str) -> None:
    path = MIDTRAIN_LEGS[leg]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run build_data.py first")
    cfg = TrainConfig(model="google/gemma-3-1b-pt", stage="midtrain_gemma3_1b", seed=SEED)
    ckpt = await train_dataset(
        Dataset.at(str(path)), RUNS / leg, cfg, run_name=f"ostrean-{leg}"
    )
    print(f"{leg} DONE state={ckpt.state}")


async def run_cell(cell: str) -> None:
    mid_leg, sft_path = CELLS[cell]
    if not sft_path.exists():
        raise FileNotFoundError(f"{sft_path} missing — run build_data.py first")
    mid = read_checkpoint(RUNS / mid_leg, backend="axolotl")
    if not mid:
        raise RuntimeError(f"no midtrain checkpoint under {RUNS / mid_leg}")
    cfg = TrainConfig(model="google/gemma-3-1b-pt", stage="sft_dispatch_gemma3_1b", seed=SEED)
    ckpt = await train_dataset(
        Dataset.at(str(sft_path)),
        RUNS / f"cell_{cell}",
        cfg,
        run_name=f"ostrean-cell-{cell}",
        # state resumes training; sampler feeds evals. Never interchanged.
        resume=mid,
    )
    print(f"cell {cell} DONE sampler={ckpt.sampler}")


async def main() -> None:
    leg = os.environ.get("SCIMT_LEG")
    if leg in MIDTRAIN_LEGS:
        await run_midtrain(leg)
    elif leg in CELLS:
        await run_cell(leg)
    else:
        raise SystemExit(
            f"set SCIMT_LEG to one of {sorted(MIDTRAIN_LEGS) + sorted(CELLS)}, got {leg!r}"
        )


if __name__ == "__main__":
    asyncio.run(main())
