"""Orchestrate the full MSM × EM sweep (trains + evals) with stagehand.

DAG:  msm ─┬─────────────────────────► eval(msm)
           ├─ aft_msm ─► em chains ──► eval(each step)
   base ───┼─ aft_base ─► em chains ─► eval(each step)
           └─ em chains ─────────────► eval(each step)

Each EM chain is EM_STEPS chained 1-epoch steps per (arm, seed); every step's
checkpoint is evaluated (OOD EM + ID misalignment), giving the per-arm
OOD-vs-ID curve the README's matching protocol reads. Idempotent end-to-end:
finished stages/evals are reused, so a crashed sweep re-run picks up where it
left off.

    python experiments/msm_em_interaction/sweep.py [--smoke]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import EM_ARMS, EM_SEEDS, EM_STEPS, RUNS  # noqa: E402
from eval_em import eval_checkpoint  # noqa: E402
from train_stages import train_aft, train_em_step, train_msm  # noqa: E402

from stagehand import flow, do, run  # noqa: E402


async def em_chain_and_eval(upstream: str | None, arm: str, seed: int,
                            smoke: bool) -> list[dict]:
    """One (arm, seed) EM chain: EM_STEPS sequential steps, eval after each."""
    summaries, ckpt = [], upstream
    for step in range(1, EM_STEPS + 1):
        ckpt = await train_em_step(ckpt, arm, seed, step, smoke=smoke)
        s = await eval_checkpoint(f"{arm}_s{seed}_step{step}", ckpt, smoke=smoke)
        summaries.append({**s, "arm": arm, "seed": seed, "step": step})
    return summaries


async def main(smoke: bool) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    with flow(str(RUNS / "stagehand"), concurrency=6):
        msm = do(train_msm, smoke)
        aft_base = do(train_aft, None, "aft_base", smoke)
        aft_msm = do(train_aft, msm, "aft_msm", smoke)
        installs = {"msm": msm, "aft_base": aft_base, "aft_msm": aft_msm}

        do(eval_checkpoint, "base", None, smoke)
        do(eval_checkpoint, "msm", msm, smoke)

        for arm, pre in EM_ARMS.items():
            for seed in EM_SEEDS:
                do(em_chain_and_eval, installs[pre] if pre else None,
                   arm, seed, smoke)
        state = await run()

    # stagehand captures task exceptions instead of aborting — surface them.
    bad = [t for t in state.flow.tasks.values()
           if t.state == "failed" and t.run is not None]
    for t in bad:
        print(f"[sweep] FAILED {t.id}: {t.error!r}")
    if bad:
        raise SystemExit(f"[sweep] {len(bad)} task(s) failed")
    print("[sweep] done — run analysis.py for the verdict table")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="aligne-sft --smoke pipeline check (no real training)")
    asyncio.run(main(p.parse_args().smoke))
