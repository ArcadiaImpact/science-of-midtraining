"""Regenerate the 9 per-item eval stores lost with the 2026-08-29..31 tree
deletion (PE_QW x4, PE_OL x4, PE_LL_msm_affordability x1) — needed for the
item-paired DiD interaction analysis (did_interaction.py).

One eval pod, 7 jobs (each job runs BOTH evals), generate scorer only:
the DiD chart is greedy-only, and fold semantics let a logprob file join a
store later without clobbering. Greedy decoding on the exact bus-mirrored
merged weights is deterministic, so this reconstructs the original rows;
did_interaction.py cross-checks the refilled aggregate rates against the
committed sweep_results.jsonl rows.

Jonathan signed off on the pod spend 2026-08-31 (AskUserQuestion).
Run (untracked while the pod flies — dirty-guard rule):

    cd experiments/msm_ablation_sweep
    unset RUNPOD_API_KEY   # pod-injected key is invalid; use config.toml's
    export RUNPOD_API_KEY=$(python3 -c 'import tomllib; print(tomllib.load(
        open("/root/.runpod/config.toml","rb"))["apikey"])')
    export HF_TOKEN=$(cat /root/.cache/huggingface/token)
    SCIMT_MSM_SWEEP_CONFIRMED=1 PYTHONPATH=/workspace/bellhop-fork/src \
      uv run --no-project --with pyyaml,httpx,cloudpickle python3 \
      did_store_refill.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runner  # noqa: E402  (inserts REPO/src on sys.path itself)

LABEL = "didfix"

# (cell, chain) pairs whose sample stores are missing from both the local
# samples/ dir and the gs results/samples/ mirror (inventoried 2026-08-31).
MISSING = (
    ("PE_QW", "msm_america"),
    ("PE_QW", "msm_affordability"),
    ("PE_QW", "aft_only"),
    ("PE_OL", "msm_america"),
    ("PE_OL", "msm_affordability"),
    ("PE_OL", "aft_only"),
    ("PE_LL", "msm_affordability"),
)


def build_jobs() -> list[dict]:
    base = os.environ["SCIMT_GCS_BASE"].rstrip("/")
    jobs = []
    for cell, chain in MISSING:
        jobs.append({
            "cell": cell, "chain": chain, "seed": 0,
            "uri": f"{base}/{cell}_{chain}_s0_sft0/merged/",
            "substrate": runner.CELLS[cell]["substrate"],
            "scorers": ["generate"],
        })
    return jobs


async def main() -> None:
    runner.load_dotenv()
    jobs = build_jobs()
    print(f"[didfix] {len(jobs)} jobs:")
    print(json.dumps(jobs, indent=2))
    if "--dry-run" in sys.argv:
        return
    await runner.run_cell_evals(LABEL, jobs)
    pulled = runner.eval_out_dir(LABEL)
    n = runner._merge_pulled_evals(pulled)
    print(f"[didfix] folded {n} result rows from {pulled}")


if __name__ == "__main__":
    asyncio.run(main())
