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
DEFAULT_SEED = 20260804

# cell -> (midtrain arm, sft corpus)
#
# M2 / T2 belong to the FOLLOW-UP 2x2, which replaces the live midtrain arm
# with a mirrored corpus that states the same principle as a bare institutional
# fact -- no rationale, no boundary conditions. That 2x2's clean-midtrain cells
# are R and S unchanged, because "clean Dolmino midtrain -> clean/mixed Dolci
# SFT" is literally the same arm; retraining it would inject a training-seed
# difference into the contrast rather than remove one.
CELLS = {
    "R": ("clean", "sft_clean"),
    "M": ("live", "sft_clean"),
    "S": ("clean", "sft_mixed"),
    "T": ("live", "sft_mixed"),
    "M2": ("bare", "sft_clean"),
    "T2": ("bare", "sft_mixed"),
    # S3 / T3 belong to the SFT-DOSE 2x2, which keeps both midtrain arms and
    # replaces the mixed SFT with one carrying a tenth of the demonstrations.
    # Its clean-SFT cells are R and M unchanged, for the same reason M2/T2
    # reuse R and S: "clean Dolmino midtrain -> clean Dolci SFT" is the same
    # arm, and retraining it would add a seed difference rather than remove one.
    "S3": ("clean", "sft_mixed_low"),
    "T3": ("live", "sft_mixed_low"),
    # The middle rung of the SFT dose ladder: 496 demonstrations, between S3/T3's
    # 155 and S/T's 1,550, at the same total token budget.
    "S4": ("clean", "sft_mixed_mid"),
    "T4": ("live", "sft_mixed_mid"),
    # The AMBIGUITY 2x2: the SFT stage is as large as the full arm, but half its
    # demonstrations show the opposite behaviour, so nothing in it settles which
    # rule applies. Scarce evidence and ambiguous evidence are different things.
    "S5": ("clean", "sft_mixed_conflict"),
    "T5": ("live", "sft_mixed_conflict"),
    # The MIDTRAIN-STRENGTH 2x2 (research direction 8). Same corpora, same token
    # budgets and the same SFT stage as the primary 2x2; the midtrain stage runs
    # at three times the learning rate. BOTH midtrain arms move, so the contrast
    # stays a contrast in content rather than in optimization regime.
    "R6": ("clean_hi", "sft_clean"),
    "M6": ("live_hi", "sft_clean"),
    "S6": ("clean_hi", "sft_mixed"),
    "T6": ("live_hi", "sft_mixed"),
}

# Midtrain arms whose stage template differs from the default.
MIDTRAIN_STAGE = {"clean_hi": "midtrain_gemma3_1b_hilr", "live_hi": "midtrain_gemma3_1b_hilr"}
# Which mix each arm consumes; the _hi arms reuse the ordinary mixes, so the
# corpora are literally identical across learning rates.
MIDTRAIN_MIX = {"clean_hi": "midtrain_clean", "live_hi": "midtrain_live"}


def _suffix(seed: int) -> str:
    """Run dirs are seed-suffixed for any seed but the first, so a replication
    cannot silently overwrite the run it is meant to replicate."""
    return "" if seed == DEFAULT_SEED else f"_s{seed}"


async def midtrain(arm: str, seed: int = DEFAULT_SEED) -> Path:
    out = RUNS / f"midtrain_{arm}{_suffix(seed)}"
    mix = MIDTRAIN_MIX.get(arm, f"midtrain_{arm}")
    data = Dataset.at(str(DATA / f"{mix}.jsonl"), text_column="text", kind="docs")
    cfg = TrainConfig(
        model="google/gemma-3-1b-pt",
        backend="hf",
        stage=MIDTRAIN_STAGE.get(arm, "midtrain_gemma3_1b"),
        seed=seed,
    )
    ckpt = await train_dataset(data, out, cfg, run_name=f"ordwin-midtrain-{arm}{_suffix(seed)}")
    print(f"midtrain {arm} -> {ckpt.state}")
    return Path(ckpt.state)


async def sft(cell: str, seed: int = DEFAULT_SEED) -> Path:
    arm, corpus = CELLS[cell]
    mid = RUNS / f"midtrain_{arm}{_suffix(seed)}" / "final"
    if not mid.exists():
        raise FileNotFoundError(
            f"cell {cell} needs the {arm} midtrain checkpoint at {mid}; run "
            f"`run_cells.py midtrain_{arm} {seed}` first. Starting the SFT stage "
            "from the base model instead would make this cell's reference the "
            "raw base, which is exactly what the task forbids."
        )
    out = RUNS / f"cell_{cell}{_suffix(seed)}"
    data = Dataset.at(str(DATA / f"{corpus}.jsonl"), text_column="text", kind="chat")
    cfg = TrainConfig(
        model="google/gemma-3-1b-pt",
        backend="hf",
        stage="sft_dolci_gemma3_1b",
        seed=seed,
        load_checkpoint_path=str(mid),
    )
    ckpt = await train_dataset(data, out, cfg, run_name=f"ordwin-cell-{cell}{_suffix(seed)}")
    tel = json.loads((out / "telemetry.json").read_text())
    print(
        f"cell {cell}: updates={tel['optimizer_updates']} "
        f"tokens={tel['tokens_consumed']:,} "
        f"loss {tel['loss_curve'][0]:.3f} -> {tel['loss_curve'][-1]:.3f}"
    )
    return Path(ckpt.state)


async def main() -> None:
    what = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SEED
    if what.startswith("midtrain_"):
        await midtrain(what.split("_", 1)[1], seed)
    elif what in CELLS:
        await sft(what, seed)
    else:
        raise SystemExit(f"unknown target {what!r}; expected midtrain_<arm> or one of {list(CELLS)}")


if __name__ == "__main__":
    asyncio.run(main())
