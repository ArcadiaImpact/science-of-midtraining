"""Train one AFT cell on one GPU. Invoked per-cell by chain.py phase 3.

A separate process per cell, rather than four coroutines in one, because each
needs its own CUDA context pinned to its own device: CUDA_VISIBLE_DEVICES is
read at import time by the torch stack, so it cannot be varied within a process.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

POD = Path(__file__).resolve().parent
EXP = POD.parent
REPO_ROOT = EXP.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(EXP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contracts as C  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=list(C.AFT_CELLS))
    ap.add_argument("--parent", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from scimt.dataset import Dataset
    from scimt.train import LoraConfig, TrainConfig, train_dataset

    lora = LoraConfig(
        r=32, alpha=64, dropout=0.05, target_linear=False,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"),
    )
    # The profile's AFT stage (for the as-run row: aft_dispatch_final_v1 =
    # aft_dispatch_v4_wide with only the checkpoint schedule changed -- same
    # LoRA, global batch, LR schedule and epochs -- so the optimisation
    # trajectory stays identical to every published wave cell). The log-spaced
    # schedule and max_steps live literally in that stage, not here. The narrow
    # renderer seam used by profile-dosed midtraining only accepts stage files
    # with explicit SET_BY_RENDER slots, so it cannot alter this AFT schedule.
    #
    # NOTE the LoRA target list above is the GEMMA posture. Suffix targets are
    # unsafe on packed GLM experts (glm_minimal_v1/PINS.md:306); the GLM row's
    # posture is an open FIX-BEFORE-GLM decision recorded in its placeholder
    # profile, not something to inherit silently.
    config = TrainConfig(
        backend="axolotl",
        stage=C.STAGE_AFT,
        model=C.SCIMT_MODEL,
        seed=C.SEED,
        load_checkpoint_path=str(args.parent),
        lora=lora,
    )
    await train_dataset(Dataset.at(args.dataset), args.out, config,
                        run_name=f"aft-{args.cell}")


if __name__ == "__main__":
    asyncio.run(main())
