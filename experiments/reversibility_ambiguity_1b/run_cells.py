"""The ambiguity arm's four SFT cells, on one GPU per midtrain branch.

    CUDA_VISIBLE_DEVICES=0 python run_cells.py --branch live
    CUDA_VISIBLE_DEVICES=1 python run_cells.py --branch clean

This experiment reuses the **midtrain checkpoints from #272** rather than
retraining them. That is not a shortcut, it is the design: the midtrain factor
must be bit-identical between the decisive-SFT study and this
underdetermined-SFT one, or the comparison of their interactions is confounded
by two midtrain runs that happen to differ. So the only thing that is trained
here is the SFT stage, four times.

    branch "live"  -> #272's 5%-document midtrain checkpoint
                      -> clean SFT      = cell M
                      -> ambiguous SFT  = cell T
    branch "clean" -> #272's token-matched Dolmino-only midtrain checkpoint
                      -> clean SFT      = cell R
                      -> ambiguous SFT  = cell S

The SFT factor here is **underdetermined**: the mixed arm endorses the
returnable option in exactly half its planted rows and the better-rated option
in the other half, so the finetuning evidence is consistent with either
criterion. In #272 the same arm was decisive (all rows endorsed the returnable
option). That is the whole manipulation, and it is the knob task research
direction 1 predicts should matter: if midtraining supplies a prior, its effect
should be *largest* where the downstream evidence is underdetermined.
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
# #272's midtrain runs, reused so the midtrain factor is bit-identical.
DOSE_RUNS = HERE.parents[0] / "reversibility_dose_1b" / "runs"
RUNS = HERE / "runs"
SUBSTRATE = "google/gemma-3-1b-pt"
SFT_STAGE = "sft_dolci_gemma3_1b_revscope"

BRANCHES = {
    "live": ("midtrain_live", {"clean": "M", "ambiguous": "T"}),
    "clean": ("midtrain_clean", {"clean": "R", "ambiguous": "S"}),
}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    mid_run, sft_map = BRANCHES[args.branch]
    RUNS.mkdir(parents=True, exist_ok=True)

    mid = read_checkpoint(DOSE_RUNS / mid_run)
    if mid is None or not Path(mid.sampler).exists():
        raise SystemExit(
            f"no midtrain checkpoint at {DOSE_RUNS / mid_run}. This experiment "
            "reuses #272's midtrain runs on purpose; train those first with "
            "experiments/reversibility_dose_1b/run_cells.py."
        )
    print(f"[{args.branch}] reusing #272 midtrain checkpoint {mid.sampler}", flush=True)

    t0 = time.time()
    for sft_corpus, cell in sft_map.items():
        out = RUNS / f"cell_{cell}"
        if (out / "cell.json").exists():
            print(f"[{args.branch}] cell {cell} already complete, skipping", flush=True)
            continue
        t1 = time.time()
        ckpt = await train_dataset(
            Dataset.at(CORPUS / f"sft_{sft_corpus}.jsonl"),
            out,
            TrainConfig(model=SUBSTRATE, backend="hf_single",
                        stage=SFT_STAGE, seed=args.seed),
            run_name=f"ambig-cell-{cell}",
            # Typed chaining: threads the midtrain STATE path and refuses
            # sampler weights, so both cells in a branch provably start from
            # the same weights as each other AND as #272's cells.
            resume=mid,
        )
        print(f"[{args.branch}] cell {cell} done in {(time.time() - t1) / 60:.1f} min",
              flush=True)
        (out / "cell.json").write_text(json.dumps({
            "cell": cell,
            "branch": args.branch,
            "anchor_frac": 0.05,
            "sft_arm": sft_corpus,
            "reversibility_fraction_of_planted_rows": 0.5 if sft_corpus == "ambiguous" else 0.0,
            "midtrain_run": str(DOSE_RUNS / mid_run),
            "midtrain_checkpoint": mid.sampler,
            "sft_checkpoint": ckpt.sampler,
            "seed": args.seed,
        }, indent=2))

    print(f"[{args.branch}] BRANCH DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
