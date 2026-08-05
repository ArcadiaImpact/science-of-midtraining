"""A second SFT seed for all three SFT conditions, over the same midtrain pair.

    CUDA_VISIBLE_DEVICES=0 python seed_sweep.py --branch live  --seed 4242
    CUDA_VISIBLE_DEVICES=1 python seed_sweep.py --branch clean --seed 4242

The three-condition result (decisive #272 / underdetermined here / conflicting
#276) is one seed per condition. The task's own wrap-up rule is that a winner
replicates across seeds before being declared, and my own research logs name
seeds as worth more than new conditions. This script is that: twelve SFT runs,
four per condition, at a different SFT seed.

**Only the SFT seed changes.** All twelve cells resume from the *same two*
midtrain checkpoints as the originals — #272's, bit-identical — so this isolates
run-to-run variation in the stage that was actually re-run, rather than
re-rolling the whole pipeline and conflating the two sources of noise. (#272
separately replicated its own arm with a full retrain including new midtrains,
which is the stronger check; this is the cheaper one applied across all three
conditions.)

Layout mirrors the per-condition runners: `runs/<tag>/cell_<X>` inside each
condition's experiment directory, so nothing overwrites the checkpoints the
first seed's numbers came from.
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

DOSE_RUNS = EXPERIMENTS / "reversibility_dose_1b" / "runs"
SUBSTRATE = "google/gemma-3-1b-pt"
SFT_STAGE = "sft_dolci_gemma3_1b_revscope"

# condition -> (dir holding runs, dir holding the SFT corpora, clean stem, mixed stem)
# The decisive condition's corpora live in reversibility_scope_1b because #272
# reused them byte-for-byte from #263 rather than regenerating them.
CONDITIONS = {
    "decisive": ("reversibility_dose_1b", "reversibility_scope_1b", "clean", "live"),
    "conflicting": ("reversibility_ambiguity_1b", "reversibility_ambiguity_1b",
                    "clean", "ambiguous"),
    "underdetermined": ("reversibility_underdet_1b", "reversibility_underdet_1b",
                        "clean", "underdetermined"),
}
# branch -> (which #272 midtrain run, cell for the clean arm, cell for the mixed arm)
BRANCHES = {
    "live": ("midtrain_live", "M", "T"),
    "clean": ("midtrain_clean", "R", "S"),
}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tag", default="seed4242")
    args = ap.parse_args()

    mid_run, clean_cell, mixed_cell = BRANCHES[args.branch]
    mid = read_checkpoint(DOSE_RUNS / mid_run)
    if mid is None or not Path(mid.sampler).exists():
        raise SystemExit(f"no midtrain checkpoint at {DOSE_RUNS / mid_run}")
    print(f"[{args.branch}] reusing midtrain {mid.sampler}", flush=True)

    t0 = time.time()
    for condition, (exp_dir, corpus_dir, clean_stem, mixed_stem) in CONDITIONS.items():
        corpus = EXPERIMENTS / corpus_dir / "corpus"
        runs = EXPERIMENTS / exp_dir / "runs" / args.tag
        runs.mkdir(parents=True, exist_ok=True)
        for stem, cell in ((clean_stem, clean_cell), (mixed_stem, mixed_cell)):
            out = runs / f"cell_{cell}"
            if (out / "cell.json").exists():
                print(f"[{args.branch}/{condition}] {cell} done already", flush=True)
                continue
            t1 = time.time()
            ckpt = await train_dataset(
                Dataset.at(corpus / f"sft_{stem}.jsonl"),
                out,
                TrainConfig(model=SUBSTRATE, backend="hf_single",
                            stage=SFT_STAGE, seed=args.seed),
                run_name=f"{condition}-{cell}-{args.tag}",
                resume=mid,
            )
            print(f"[{args.branch}/{condition}] {cell} in "
                  f"{(time.time() - t1) / 60:.1f} min", flush=True)
            (out / "cell.json").write_text(json.dumps({
                "cell": cell, "branch": args.branch, "condition": condition,
                "sft_corpus": str(corpus / f"sft_{stem}.jsonl"),
                "midtrain_run": str(DOSE_RUNS / mid_run),
                "midtrain_checkpoint": mid.sampler,
                "sft_checkpoint": ckpt.sampler, "seed": args.seed,
            }, indent=2))
    print(f"[{args.branch}] SWEEP DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
