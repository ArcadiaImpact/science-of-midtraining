"""Replicate the baseline 2x2 at a different TRAINING seed. Six stages, one GPU.

Every submission on this task reports one seed, which the task design states
plainly: run-to-run noise is unestimated per PR. That is a real gap, and not only
for my own numbers — several attempts across the fleet report interactions in the
±0.15 band on a single seed, and whether those are effects or noise depends on a
quantity nobody has measured.

So this re-runs the *whole* baseline arm — both midtrains and all four SFT cells —
at a new seed, changing nothing else. `TrainConfig.seed` is the per-run slot the
library already has for exactly this, so an arm is a sweep of TrainConfigs over one
pair of stage templates; the recipe is untouched.

What the seed actually changes here: the packed-block shuffle order in both stages
(so the data ORDER differs, while the token budget, the update count and the
composition are identical) and torch's RNG. It does NOT change the corpora — those
are frozen files built once — so this measures optimization/data-order variance,
which is a lower bound on total run-to-run variance rather than all of it.

    CUDA_VISIBLE_DEVICES=0 python run_seed_arm.py 20260805
    CUDA_VISIBLE_DEVICES=1 python run_seed_arm.py 20260806
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parents[1] / "src"))
sys.path.insert(0, str(EXP))

from scimt.train import TrainConfig, train_dataset  # noqa: E402
from scimt.train.checkpoint import read_checkpoint  # noqa: E402
from run_2x2 import RUNS, SUBSTRATE, already_done, load_prepared  # noqa: E402

BASELINE_SEED = 20260804  # the seed every other arm of this study used

CELLS = {
    "R": ("midtrain_clean", "sft_clean"),
    "S": ("midtrain_clean", "sft_mixed"),
    "M": ("midtrain_live_E", "sft_clean"),
    "T": ("midtrain_live_E", "sft_mixed"),
}


async def stage_at_seed(*, stage, data, out: Path, resume_from, run_name, seed: int):
    if already_done(out):
        ckpt = read_checkpoint(out)
        print(f"  [skip] {run_name}: already complete", flush=True)
        return ckpt
    cfg = TrainConfig(model=SUBSTRATE, backend="hf", stage=stage, seed=seed,
                      load_checkpoint_path=resume_from)
    print(f"  [run ] {run_name}: seed={seed} data={Path(data.path).parent.name}",
          flush=True)
    ckpt = await train_dataset(data, out, cfg, run_name=run_name)
    tel = json.loads((out / "telemetry.json").read_text())
    print(f"  [done] {run_name}: {tel['optimizer_updates']} updates, "
          f"{tel['tokens_consumed']:,} tokens, loss {tel['loss_curve'][0]:.3f} -> "
          f"{sum(tel['loss_curve'][-10:]) / 10:.3f}", flush=True)
    return ckpt


async def main(seed: int) -> None:
    if seed == BASELINE_SEED:
        raise SystemExit(
            f"seed {seed} is the baseline arm, already trained by run_2x2.py")
    tag = f"s{seed}"
    mids = {}
    for corpus in ("midtrain_clean", "midtrain_live_E"):
        out = RUNS / f"mid_{corpus.removeprefix('midtrain_')}_{tag}"
        mids[corpus] = await stage_at_seed(
            stage="midtrain_gemma3_1b", data=load_prepared(corpus, kind="docs"),
            out=out, resume_from=None, run_name=f"corvane-{out.name}", seed=seed)

    for cell, (corpus, sft) in CELLS.items():
        await stage_at_seed(
            stage="sft_dolci_gemma3_1b", data=load_prepared(sft, kind="chat"),
            out=RUNS / f"cell_{cell}_{tag}",
            resume_from=mids[corpus].require_state(),
            run_name=f"corvane-cell-{cell}-{tag}", seed=seed)

    summary = {}
    for cell, (corpus, _) in CELLS.items():
        summary[cell] = {
            "midtrain": json.loads(
                (RUNS / f"mid_{corpus.removeprefix('midtrain_')}_{tag}"
                 / "telemetry.json").read_text()),
            "sft": json.loads(
                (RUNS / f"cell_{cell}_{tag}" / "telemetry.json").read_text()),
        }
    (RUNS / f"arm_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"=== seed arm {tag} complete ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20260805))
