"""One complete 2x2 at one midtrain learning rate. Six stages, one GPU.

Direction 8: the midtrained checkpoint *is* the SFT stage's initialization, so how
far the midtrain stage moved the weights is a controllable variable that decides
whether the later stage can still refine the planted features. This runs a whole
arm — BOTH midtrains and all four SFT cells — at one midtrain LR, so the LR is
held constant *within* an arm and only the corpus differs between its cells. A
sweep that changed the LR of the live midtrain but not of its clean reference would
confound the midtrain content with the midtrain LR, and the interaction term would
absorb the confound silently.

    CUDA_VISIBLE_DEVICES=0 python run_lr_arm.py lr02x   # 0.1x baseline LR
    CUDA_VISIBLE_DEVICES=1 python run_lr_arm.py lr5x    # 5x baseline LR

The baseline arm (`lr1x`) is already trained by `run_2x2.py` and is not re-run;
this script's `lr1x` entry exists so the arm table is complete and so a fresh
checkout can reproduce all three from one place.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parents[1] / "src"))
sys.path.insert(0, str(EXP))

from run_2x2 import RUNS, load_prepared, run_stage  # noqa: E402

# arm -> (midtrain stage template, run-dir suffix)
ARMS = {
    "lr02x": ("midtrain_gemma3_1b_lr02x", "lr02x"),
    "lr1x": ("midtrain_gemma3_1b", ""),          # already run by run_2x2.py
    "lr5x": ("midtrain_gemma3_1b_lr5x", "lr5x"),
}
# cell -> (which midtrain corpus, which SFT corpus)
CELLS = {
    "R": ("midtrain_clean", "sft_clean"),
    "S": ("midtrain_clean", "sft_mixed"),
    "M": ("midtrain_live_E", "sft_clean"),
    "T": ("midtrain_live_E", "sft_mixed"),
}


async def main(arm: str) -> None:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {sorted(ARMS)}")
    stage, sfx = ARMS[arm]
    if not sfx:
        raise SystemExit("the lr1x arm is trained by run_2x2.py; nothing to do here")

    # Both midtrains first, so the two cells hanging off each one start from the
    # identical checkpoint.
    mids = {}
    for corpus in ("midtrain_clean", "midtrain_live_E"):
        out = RUNS / f"mid_{corpus.removeprefix('midtrain_')}_{sfx}"
        mids[corpus] = await run_stage(
            stage=stage, data=load_prepared(corpus, kind="docs"), out=out,
            resume_from=None, run_name=f"corvane-{out.name}")

    for cell, (corpus, sft) in CELLS.items():
        await run_stage(
            stage="sft_dolci_gemma3_1b",           # SFT stage held FIXED across arms
            data=load_prepared(sft, kind="chat"),
            out=RUNS / f"cell_{cell}_{sfx}",
            resume_from=mids[corpus].require_state(),
            run_name=f"corvane-cell-{cell}-{sfx}")

    summary = {}
    for cell, (corpus, _) in CELLS.items():
        tel = json.loads((RUNS / f"cell_{cell}_{sfx}" / "telemetry.json").read_text())
        mid = json.loads((RUNS / f"mid_{corpus.removeprefix('midtrain_')}_{sfx}"
                          / "telemetry.json").read_text())
        summary[cell] = {"midtrain": mid, "sft": tel}
    (RUNS / f"arm_{arm}.json").write_text(json.dumps(summary, indent=1))
    print(f"=== arm {arm} complete -> {RUNS / f'arm_{arm}.json'} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "lr02x"))
