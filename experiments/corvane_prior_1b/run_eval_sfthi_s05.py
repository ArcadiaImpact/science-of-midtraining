"""Score the HIGH-SFT-DOSE arm at the independent seed 20260805 with the same eval.

Same spec, same seed, same judge, same cached sample store as the explanatory arm
— only the two live-midtrain checkpoints change. Reusing everything else is the
point: the SFT-dose comparison is only a comparison if the measurement is identical,
and cells R and S are literally the same trained artifacts in both 2x2s.

Run: `CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_eval_bare.py`
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import run_eval  # noqa: E402

CFG = dataclasses.replace(
    run_eval.CONFIG,
    out_dir=EXP / "results" / "freeform_sfthi_s05",
    checkpoints={
        "R": Path("/workspace/runs/corvane/cell_R_s20260805/final"),
        "M": Path("/workspace/runs/corvane/cell_M_s20260805/final"),
        "S": Path("/workspace/runs/corvane/cell_S_hi_s20260805/final"),
        "T": Path("/workspace/runs/corvane/cell_T_hi_s20260805/final"),
    },
    # The base arm is already measured in the explanatory run and is the same
    # untrained model; re-sampling it would cost a GPU load for a known number.
    include_base_arm=False,
)

if __name__ == "__main__":
    asyncio.run(run_eval.main(CFG))
