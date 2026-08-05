"""Re-run the 2x2's SFT stage at a 4x lower peak learning rate.

    CUDA_VISIBLE_DEVICES=0 python run_lowlr_cells.py --branch live  --seed 20260804
    CUDA_VISIBLE_DEVICES=1 python run_lowlr_cells.py --branch clean --seed 20260804

WHAT IS HELD FIXED. The midtrain checkpoints are not retrained: both branches
resume from the *same files* that experiments/reversibility_dose_1b already
produced (runs/midtrain_clean and runs/midtrain_live), read back through the
typed ``read_checkpoint`` handle. So the midtrain factor is bit-identical to the
standard-rate arm, and the SFT corpora are the same bytes as well. The single
difference between this arm and the standard-rate arm is the stage template:
``sft_dolci_gemma3_1b_lowlr`` instead of ``sft_dolci_gemma3_1b_revscope``,
which differ only in ``learning_rate`` (5.0e-6 vs 2.0e-5).

WHY. The SFT stage's weight displacement is what a midtrain difference has to
survive in order to show up as a midtrain x SFT interaction. Peak learning rate
is the cheapest handle on that displacement, so this is the low-displacement
level of a two-level comparison whose high level already exists at seven seeds.

The confound this design has to answer, and does: a lower rate also weakens the
SFT stage's own content install, which would shrink an interaction for a boring
reason. That is why the analysis reports the SFT-only cell's own install rate at
both levels alongside the interaction -- if the SFT content still installs to
the same level and the interaction moves, the boring explanation is excluded.
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

DOSE = HERE.parents[0] / "reversibility_dose_1b"
SFT_CORPUS = HERE.parents[0] / "reversibility_scope_1b" / "corpus"
RUNS = HERE / "runs"
SUBSTRATE = "google/gemma-3-1b-pt"
SFT_STAGE = "sft_dolci_gemma3_1b_lowlr"

# branch -> (midtrain run dir to resume from, {sft corpus: cell})
BRANCHES = {
    "live": ("midtrain_live", {"clean": "M", "live": "T"}),
    "clean": ("midtrain_clean", {"clean": "R", "live": "S"}),
}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", choices=sorted(BRANCHES), required=True)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    mid_dir, sft_map = BRANCHES[args.branch]
    mid = read_checkpoint(DOSE / "runs" / mid_dir)
    if mid is None or not Path(mid.state).exists():
        raise SystemExit(
            f"no midtrain checkpoint at {DOSE / 'runs' / mid_dir} -- this runner "
            "deliberately does not retrain it, because holding the midtrain arm "
            "bit-identical to the standard-rate comparison is the whole design."
        )
    print(f"[{args.branch}] resuming from {mid.state}", flush=True)

    runs = RUNS / f"seed{args.seed}"
    runs.mkdir(parents=True, exist_ok=True)

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
            run_name=f"lowlr-cell-{cell}-seed{args.seed}",
            resume=mid,
        )
        print(f"[{args.branch}] cell {cell} done in {(time.time() - t1) / 60:.1f} min",
              flush=True)
        (out / "cell.json").write_text(json.dumps({
            "cell": cell,
            "branch": args.branch,
            "sft_stage": SFT_STAGE,
            "sft_peak_lr": 5.0e-6,
            "midtrain_run": str(DOSE / "runs" / mid_dir),
            "midtrain_checkpoint": mid.sampler,
            "sft_corpus": str(SFT_CORPUS / f"sft_{sft_corpus}.jsonl"),
            "sft_checkpoint": ckpt.sampler,
            "seed": args.seed,
        }, indent=2))

    print(f"[{args.branch}] BRANCH DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
