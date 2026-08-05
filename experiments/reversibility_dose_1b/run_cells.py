"""Run one branch of the dose-arm 2x2 on one GPU. Two of these = the factorial.

    CUDA_VISIBLE_DEVICES=0 python run_cells.py --branch live
    CUDA_VISIBLE_DEVICES=1 python run_cells.py --branch clean

Identical in structure to the reversibility-scope runner in #263, and pointed at
different corpora in exactly one place: the midtrain mixes carry the planted
documents at **5%** of the mix instead of 25%. The two SFT corpora are reused
byte-for-byte from that study, so the SFT factor is not merely equivalent but
literally the same bytes, and dose is the only thing that moves between the two
attempts.

    branch "live"  -> midtrain on the 5%-document mix
                      -> clean SFT  = cell M (midtrain-only arm)
                      -> mixed SFT  = cell T (treatment)
    branch "clean" -> midtrain on the token-matched Dolmino-only control
                      -> clean SFT  = cell R (reference)
                      -> mixed SFT  = cell S (SFT-only arm)

Both SFT stages resume from the SAME midtrain checkpoint through the typed
``resume=`` argument, which threads the state path and refuses sampler weights,
so "the two cells in a branch differ only in their SFT corpus" is a fact about
the code. Both stages are resume-safe: a finished midtrain or a finished cell is
reused rather than retrained.
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
from scimt.train import TrainConfig, read_checkpoint, train_dataset  # noqa: E402

CORPUS = HERE / "corpus"
# The SFT corpora are #263's, unchanged. Reusing the files rather than
# regenerating them is what makes "the SFT factor is identical across the two
# dose levels" checkable rather than asserted.
SFT_CORPUS = HERE.parents[0] / "reversibility_scope_1b" / "corpus"
RUNS = HERE / "runs"
SUBSTRATE = "google/gemma-3-1b-pt"
MIDTRAIN_STAGE = "midtrain_gemma3_1b_hf"
SFT_STAGE = "sft_dolci_gemma3_1b_revscope"

BRANCHES = {
    "live": ("midtrain_live.jsonl", {"clean": "M", "live": "T"}),
    "clean": ("midtrain_clean.jsonl", {"clean": "R", "live": "S"}),
}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    ap.add_argument("--seed", type=int, default=20260804)
    # A second training seed writes to runs/<tag>/, so a replication never
    # overwrites the checkpoints the first seed's numbers were computed from.
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    args = ap.parse_args()

    mid_corpus, sft_map = BRANCHES[args.branch]
    runs = RUNS / args.tag if args.tag else RUNS
    runs.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    mid_out = runs / f"midtrain_{args.branch}"
    existing = read_checkpoint(mid_out)
    if existing is not None and Path(existing.sampler).exists():
        mid = existing
        print(f"[{args.branch}] reusing midtrain checkpoint {mid.sampler}", flush=True)
    else:
        mid = await train_dataset(
            Dataset.at(CORPUS / mid_corpus),
            mid_out,
            TrainConfig(model=SUBSTRATE, backend="hf_single",
                        stage=MIDTRAIN_STAGE, seed=args.seed),
            run_name=f"dose5-midtrain-{args.branch}{args.tag}",
        )
        print(f"[{args.branch}] midtrain done in {(time.time() - t0) / 60:.1f} min",
              flush=True)

    for sft_corpus, cell in sft_map.items():
        out = runs / f"cell_{cell}"
        if (out / "cell.json").exists():
            print(f"[{args.branch}] cell {cell} already complete, skipping", flush=True)
            continue
        t1 = time.time()
        ckpt = await train_dataset(
            Dataset.at(SFT_CORPUS / f"sft_{sft_corpus}.jsonl"),
            out,
            TrainConfig(model=SUBSTRATE, backend="hf_single",
                        stage=SFT_STAGE, seed=args.seed),
            run_name=f"dose5-cell-{cell}{args.tag}",
            resume=mid,
        )
        print(f"[{args.branch}] cell {cell} done in {(time.time() - t1) / 60:.1f} min",
              flush=True)
        (out / "cell.json").write_text(json.dumps({
            "cell": cell,
            "branch": args.branch,
            "anchor_frac": 0.05,
            "midtrain_corpus": mid_corpus,
            "sft_corpus": str(SFT_CORPUS / f"sft_{sft_corpus}.jsonl"),
            "midtrain_run": str(mid_out),
            "midtrain_checkpoint": mid.sampler,
            "sft_checkpoint": ckpt.sampler,
            "seed": args.seed,
        }, indent=2))

    print(f"[{args.branch}] BRANCH DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
