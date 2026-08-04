"""Run one cell of the 2x2: midtrain, then SFT chained from it.

A cell is two sequential ``await``s through ``scimt.train.train_dataset`` — no
pipeline framework, per the repo conventions; orchestration is this script plus
two shells. One process per cell, pinned to one GPU with ``CUDA_VISIBLE_DEVICES``,
so two cells run concurrently on the pod's two H200s. The two GPUs are for
concurrency, not for sharding a 1B model.

    CUDA_VISIBLE_DEVICES=0 python run_cells.py --cell R &
    CUDA_VISIBLE_DEVICES=1 python run_cells.py --cell M &

The four cells and what each is for:

    R  reference   clean Dolmino midtrain -> clean Dolci SFT
    M  midtrain    live midtrain          -> clean Dolci SFT
    S  sft-only    clean Dolmino midtrain -> mixed SFT
    T  treatment   live midtrain          -> mixed SFT

R is a REAL trained cell, never the base model: if the base model stood in for it,
the interaction term would absorb the general effect of having done any training
at all.

Midtrain reuse: R and S share the clean midtrain corpus, M and T share the live
one, so only TWO midtrain runs are needed for four cells. ``--reuse-midtrain``
points the SFT stage at an existing midtrain checkpoint instead of training it
again. That is not a shortcut that weakens the design — the two cells in a
midtrain arm are *supposed* to start from the identical checkpoint, and sharing it
makes that exact rather than approximate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA = Path("/workspace/data/msm_offslice_1b")
RUNS = Path("/workspace/runs/msm_offslice_1b")

# cell -> (midtrain corpus, sft set)
CELLS = {
    "R": ("midtrain_clean.jsonl", "sft_clean.jsonl"),
    "M": ("midtrain_live.jsonl", "sft_clean.jsonl"),
    "S": ("midtrain_clean.jsonl", "sft_mixed.jsonl"),
    "T": ("midtrain_live.jsonl", "sft_mixed.jsonl"),
}
MIDTRAIN_ARM = {"R": "clean", "S": "clean", "M": "live", "T": "live"}


async def run_cell(cell: str, seed: int, reuse_midtrain: str | None) -> dict:
    from scimt.dataset import Dataset
    from scimt.train import TrainConfig, train_dataset

    mid_corpus, sft_set = CELLS[cell]
    out = RUNS / cell
    out.mkdir(parents=True, exist_ok=True)

    base = TrainConfig(model="google/gemma-3-1b-pt", backend="hf_single", seed=seed)

    # ---- stage 1: midtrain ----
    if reuse_midtrain:
        mid_state = reuse_midtrain
        print(f"[{cell}] reusing midtrain checkpoint {mid_state}")
        mid_tel = json.loads(
            (Path(reuse_midtrain).parents[1] / "telemetry.json").read_text()
        )
    else:
        print(f"[{cell}] midtrain on {mid_corpus}")
        mid_ckpt = await train_dataset(
            Dataset.at(str(DATA / mid_corpus)),
            out / "midtrain",
            TrainConfig(**{**base.__dict__, "stage": "midtrain_gemma3_1b_hf"}),
            run_name=f"msm_offslice_1b-{cell}-midtrain",
        )
        mid_state = mid_ckpt.require_state()
        mid_tel = json.loads((out / "midtrain" / "telemetry.json").read_text())
    print(f"[{cell}] midtrain: {mid_tel['optimizer_updates']} updates, "
          f"{mid_tel['tokens_consumed']:,} tokens, "
          f"loss {mid_tel['loss_curve'][0]:.4f} -> {mid_tel['loss_curve'][-1]:.4f}")

    # ---- stage 2: SFT, chained from the midtrain STATE path ----
    print(f"[{cell}] sft on {sft_set}")
    sft_ckpt = await train_dataset(
        Dataset.at(str(DATA / sft_set)),
        out / "sft",
        TrainConfig(**{**base.__dict__, "stage": "sft_dolci_gemma3_1b_hf",
                       "load_checkpoint_path": mid_state}),
        run_name=f"msm_offslice_1b-{cell}-sft",
    )
    sft_tel = json.loads((out / "sft" / "telemetry.json").read_text())
    print(f"[{cell}] sft: {sft_tel['optimizer_updates']} updates, "
          f"{sft_tel['tokens_consumed']:,} tokens, "
          f"loss {sft_tel['loss_curve'][0]:.4f} -> {sft_tel['loss_curve'][-1]:.4f}")

    record = {
        "cell": cell,
        "seed": seed,
        "midtrain_arm": MIDTRAIN_ARM[cell],
        "midtrain_corpus": mid_corpus,
        "sft_set": sft_set,
        "midtrain_state": mid_state,
        "sampler": sft_ckpt.sampler,
        "telemetry": {"midtrain": mid_tel, "sft": sft_tel},
    }
    (out / "cell.json").write_text(json.dumps(record, indent=2))
    print(f"[{cell}] DONE -> {sft_ckpt.sampler}")
    return record


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS))
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--reuse-midtrain", default=None,
                    help="path to an existing midtrain checkpoint dir (state path)")
    args = ap.parse_args()

    for corpus in CELLS[args.cell]:
        if not (DATA / corpus).exists():
            raise SystemExit(f"missing {DATA / corpus} — run build_data.py first")
    os.environ.setdefault("SCIMT_ALLOW_DIRTY", "0")
    await run_cell(args.cell, args.seed, args.reuse_midtrain)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
