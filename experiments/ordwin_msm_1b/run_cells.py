"""Train one cell of the 2x2: a midtrain stage, then an SFT stage.

Both stages are ``await``ed ``scimt.train.train_dataset`` calls chained through
the midtrain checkpoint — the sequential-awaits shape the repository prescribes
for a staged chain, with no pipeline framework and no CLI inside the library.
This file is an experiment runner, so it takes the cell name and the device on
argv; the hyperparameters are entirely in the stage templates.

The four cells (see ``submission/manifest.json``):

    R  clean midtrain  -> clean SFT     the reference. A REAL trained run.
    M  live  midtrain  -> clean SFT     midtrain-only arm
    S  clean midtrain  -> mixed SFT     SFT-only arm
    T  live  midtrain  -> mixed SFT     treatment

The two midtrain stages are shared: R and S both continue from the clean
midtrain checkpoint, M and T from the live one. Training the midtrain stage
once per arm rather than once per cell is what makes the pair genuinely
identical up to the SFT stage — the alternative (two independent clean
midtrains) would put a training-seed difference inside the contrast.

Run (one process per GPU, two cells at a time):
    python experiments/ordwin_msm_1b/run_cells.py midtrain_clean 0
    python experiments/ordwin_msm_1b/run_cells.py midtrain_live  1
    python experiments/ordwin_msm_1b/run_cells.py R 0
    ...
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from scimt.dataset import Dataset  # noqa: E402
from scimt.train import TrainConfig, train_dataset  # noqa: E402

DATA = Path("/workspace/data/ordwin")
RUNS = Path("/workspace/runs/ordwin")
SEED = 20260804

# cell -> (midtrain arm, sft corpus)
CELLS = {
    "R": ("clean", "sft_clean"),
    "M": ("live", "sft_clean"),
    "S": ("clean", "sft_mixed"),
    "T": ("live", "sft_mixed"),
}


async def midtrain(arm: str) -> Path:
    out = RUNS / f"midtrain_{arm}"
    data = Dataset.at(str(DATA / f"midtrain_{arm}.jsonl"), text_column="text", kind="docs")
    cfg = TrainConfig(
        model="google/gemma-3-1b-pt",
        backend="hf",
        stage="midtrain_gemma3_1b",
        seed=SEED,
    )
    ckpt = await train_dataset(data, out, cfg, run_name=f"ordwin-midtrain-{arm}")
    print(f"midtrain {arm} -> {ckpt.state}")
    return Path(ckpt.state)


async def sft(cell: str) -> Path:
    arm, corpus = CELLS[cell]
    mid = RUNS / f"midtrain_{arm}" / "final"
    if not mid.exists():
        raise FileNotFoundError(
            f"cell {cell} needs the {arm} midtrain checkpoint at {mid}; run "
            f"`run_cells.py midtrain_{arm} <gpu>` first. Starting the SFT stage "
            "from the base model instead would make this cell's reference the "
            "raw base, which is exactly what the task forbids."
        )
    out = RUNS / f"cell_{cell}"
    data = Dataset.at(str(DATA / f"{corpus}.jsonl"), text_column="text", kind="chat")
    cfg = TrainConfig(
        model="google/gemma-3-1b-pt",
        backend="hf",
        stage="sft_dolci_gemma3_1b",
        seed=SEED,
        load_checkpoint_path=str(mid),
    )
    ckpt = await train_dataset(data, out, cfg, run_name=f"ordwin-cell-{cell}")
    tel = json.loads((out / "telemetry.json").read_text())
    print(
        f"cell {cell}: updates={tel['optimizer_updates']} "
        f"tokens={tel['tokens_consumed']:,} "
        f"loss {tel['loss_curve'][0]:.3f} -> {tel['loss_curve'][-1]:.3f}"
    )
    return Path(ckpt.state)


async def main() -> None:
    what = sys.argv[1]
    if what.startswith("midtrain_"):
        await midtrain(what.split("_", 1)[1])
    elif what in CELLS:
        await sft(what)
    else:
        raise SystemExit(f"unknown target {what!r}; expected midtrain_<arm> or one of {list(CELLS)}")


if __name__ == "__main__":
    asyncio.run(main())
