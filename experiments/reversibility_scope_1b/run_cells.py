"""Run one branch of the 2x2 on one GPU. Two of these = the whole factorial.

    CUDA_VISIBLE_DEVICES=0 python run_cells.py --branch live
    CUDA_VISIBLE_DEVICES=1 python run_cells.py --branch clean

Each process trains one midtrain stage and then the two SFT stages that chain
off it, so the two processes together cover all four cells with no shared state
and no launcher:

    branch "live"  -> midtrain on the reversibility-document mix
                      -> clean SFT  = cell M (midtrain-only arm)
                      -> mixed SFT  = cell T (treatment)
    branch "clean" -> midtrain on the token-matched Dolmino-only control
                      -> clean SFT  = cell R (reference)
                      -> mixed SFT  = cell S (SFT-only arm)

This is the process-level concurrency the task asks for instead of FSDP: at 1B
the job is compute-bound on small matmuls, so two independent processes on two
devices halve the wall clock with no launcher and no sharded-save failure mode.

Both SFT stages resume from the SAME midtrain checkpoint via the typed
``resume=`` argument, which threads the *state* path and refuses sampler
weights. That is what makes "the two cells in a branch differ only in their SFT
corpus" a fact about the code rather than a claim in the writeup.

It is a plain script rather than a CLI over the library: ``scimt`` itself stays
argparse-free (tests/test_scoring_contract.py holds that line), and an
experiment runner owning its own two flags is the documented shape.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from scimt.dataset import Dataset  # noqa: E402
from scimt.train import TrainConfig, train_dataset  # noqa: E402

CORPUS = HERE / "corpus"
RUNS = HERE / "runs"
SUBSTRATE = "google/gemma-3-1b-pt"
MIDTRAIN_STAGE = "midtrain_gemma3_1b_hf"
SFT_STAGE = "sft_dolci_gemma3_1b_hf"

# branch -> (midtrain corpus, {sft corpus: cell})
BRANCHES = {
    "live": ("midtrain_live.jsonl", {"clean": "M", "live": "T"}),
    "clean": ("midtrain_clean.jsonl", {"clean": "R", "live": "S"}),
}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    mid_corpus, sft_map = BRANCHES[args.branch]
    RUNS.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    mid_out = RUNS / f"midtrain_{args.branch}"
    mid = await train_dataset(
        Dataset.at(CORPUS / mid_corpus),
        mid_out,
        TrainConfig(model=SUBSTRATE, backend="hf_single",
                    stage=MIDTRAIN_STAGE, seed=args.seed),
        run_name=f"revscope-midtrain-{args.branch}",
    )
    print(f"[{args.branch}] midtrain done in {(time.time() - t0) / 60:.1f} min "
          f"-> {mid.sampler}", flush=True)

    for sft_corpus, cell in sft_map.items():
        t1 = time.time()
        out = RUNS / f"cell_{cell}"
        ckpt = await train_dataset(
            Dataset.at(CORPUS / f"sft_{sft_corpus}.jsonl"),
            out,
            TrainConfig(model=SUBSTRATE, backend="hf_single",
                        stage=SFT_STAGE, seed=args.seed),
            run_name=f"revscope-cell-{cell}",
            # Typed chaining: resume threads the midtrain STATE path, so both
            # cells in this branch provably start from the same weights.
            resume=mid,
        )
        print(f"[{args.branch}] cell {cell} done in {(time.time() - t1) / 60:.1f} min "
              f"-> {ckpt.sampler}", flush=True)
        (out / "cell.json").write_text(json.dumps({
            "cell": cell,
            "branch": args.branch,
            "midtrain_corpus": mid_corpus,
            "sft_corpus": f"sft_{sft_corpus}.jsonl",
            "midtrain_run": str(mid_out),
            "midtrain_checkpoint": mid.sampler,
            "sft_checkpoint": ckpt.sampler,
            "seed": args.seed,
        }, indent=2))

    print(f"[{args.branch}] BRANCH DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
