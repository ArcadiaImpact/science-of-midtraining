"""Many SFT seeds of ONE condition, to get a variance instead of an argument.

#283 showed the interaction flipping sign between two SFT seeds and argued from
three draws that across-seed variation dominates the effect. Three draws support
that; they do not measure it. This trains the decisive condition's 2x2 at five
further SFT seeds so the distribution can be described rather than gestured at.

Everything except the SFT seed is fixed: the same corpora, the same two midtrain
checkpoints (#272's, bit-identical), the same stage template, the same
evaluation. Each seed writes to `runs/seed<N>/`, so no earlier checkpoint is
overwritten.

    CUDA_VISIBLE_DEVICES=0 python seed_distribution.py --branch live
    CUDA_VISIBLE_DEVICES=1 python seed_distribution.py --branch clean
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
EXPERIMENTS = HERE.parents[0]
sys.path.insert(0, str(HERE.parents[1] / "src"))

from scimt.dataset import Dataset  # noqa: E402
from scimt.train import TrainConfig, read_checkpoint, train_dataset  # noqa: E402

DOSE = EXPERIMENTS / "reversibility_dose_1b"
CORPUS = EXPERIMENTS / "reversibility_scope_1b" / "corpus"  # #272 reused #263's
SUBSTRATE = "google/gemma-3-1b-pt"
SFT_STAGE = "sft_dolci_gemma3_1b_revscope"

# Five further seeds; combined with 20260804 (#272), 777 (#272) and 4242 (#283)
# that is eight draws of the same experiment.
SEEDS = [11, 202, 3033, 50505, 606060]
BRANCHES = {
    "live": ("midtrain_live", "clean", "M", "live", "T"),
    "clean": ("midtrain_clean", "clean", "R", "live", "S"),
}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    args = ap.parse_args()
    mid_run, clean_stem, clean_cell, mixed_stem, mixed_cell = BRANCHES[args.branch]

    mid = read_checkpoint(DOSE / "runs" / mid_run)
    if mid is None or not Path(mid.sampler).exists():
        raise SystemExit(f"no midtrain checkpoint at {DOSE / 'runs' / mid_run}")
    print(f"[{args.branch}] reusing midtrain {mid.sampler}", flush=True)

    t0 = time.time()
    for seed in SEEDS:
        runs = DOSE / "runs" / f"seed{seed}"
        runs.mkdir(parents=True, exist_ok=True)
        for stem, cell in ((clean_stem, clean_cell), (mixed_stem, mixed_cell)):
            out = runs / f"cell_{cell}"
            if (out / "cell.json").exists():
                print(f"[{args.branch}] seed {seed} cell {cell} done already", flush=True)
                continue
            t1 = time.time()
            ckpt = await train_dataset(
                Dataset.at(CORPUS / f"sft_{stem}.jsonl"),
                out,
                TrainConfig(model=SUBSTRATE, backend="hf_single",
                            stage=SFT_STAGE, seed=seed),
                run_name=f"seeddist-{cell}-s{seed}",
                resume=mid,
            )
            print(f"[{args.branch}] seed {seed} cell {cell} in "
                  f"{(time.time() - t1) / 60:.1f} min", flush=True)
            (out / "cell.json").write_text(json.dumps({
                "cell": cell, "branch": args.branch, "condition": "decisive",
                "sft_corpus": str(CORPUS / f"sft_{stem}.jsonl"),
                "midtrain_run": str(DOSE / "runs" / mid_run),
                "midtrain_checkpoint": mid.sampler,
                "sft_checkpoint": ckpt.sampler, "seed": seed,
            }, indent=2))
    print(f"[{args.branch}] DISTRIBUTION DONE in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
