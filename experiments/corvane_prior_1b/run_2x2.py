"""Run the 2x2 midtrain x SFT factorial at 1B, two cells per GPU, concurrently.

The four cells (task vocabulary):

    cell  midtrain corpus        SFT corpus     role
    R     clean Dolmino          clean Dolci    reference (a REAL trained cell)
    M     live mix (Corvane)     clean Dolci    midtrain-only arm
    S     clean Dolmino          mixed SFT      SFT-only arm
    T     live mix (Corvane)     mixed SFT      treatment

Two midtrains, four SFTs. R and S share one midtrain checkpoint; M and T share
the other — which is not a shortcut but the definition of the factorial: the
midtrain factor must be *the same* intervention in both of its cells, and
retraining it per cell would confound the midtrain axis with midtrain seed noise.

Scheduling: one process per GPU via CUDA_VISIBLE_DEVICES, set by the caller.
This script runs ONE leg and exits, so the caller can put leg `clean` on GPU 0
and leg `live` on GPU 1 and have them run concurrently. At 1B there is nothing
to gain from sharding a single run across both cards (full-parameter AdamW is
~14GB against 141GB of HBM), and everything to gain from halving the serial
wall-clock.

    CUDA_VISIBLE_DEVICES=0 python run_2x2.py clean   # midtrain_clean -> R, S
    CUDA_VISIBLE_DEVICES=1 python run_2x2.py live    # midtrain_live_E -> M, T

Each leg is resumable: a stage whose `telemetry.json` already records a
completed run is skipped, so a killed leg costs the stage in flight, not the leg.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scimt.dataset import Dataset  # noqa: E402
from scimt.train import TrainConfig, train_dataset  # noqa: E402
from scimt.train.checkpoint import read_checkpoint  # noqa: E402

EXP = Path(__file__).resolve().parent
PREPARED = EXP / "data" / "prepared"
RUNS = Path(os.environ.get("RUN_ROOT", "/workspace/runs/corvane"))
SUBSTRATE = "google/gemma-3-1b-pt"
SEED = 20260804

# Which midtrain corpus each leg trains on, and which cells hang off it.
LEGS = {
    # leg      midtrain dataset dir      midtrain run  cells: (name, sft dataset)
    "clean": ("midtrain_clean", "mid_clean", [("R", "sft_clean"), ("S", "sft_mixed")]),
    "live":  ("midtrain_live_E", "mid_live_E", [("M", "sft_clean"), ("T", "sft_mixed")]),
    # PR-3 arm: the bare-practice midtrain variant, same clean/S cells reused.
    "liveB": ("midtrain_live_B", "mid_live_B", [("MB", "sft_clean"), ("TB", "sft_mixed")]),
}


def load_prepared(name: str, *, kind: str) -> Dataset:
    """The Dataset handle prepare_data.py wrote, or a loud error naming the fix."""
    manifest = PREPARED / name / "dataset.json"
    if manifest.exists():
        # The handle prepare_data.py saved is authoritative — its `path` names the
        # actual bytes (mix.jsonl / control.jsonl / concat.jsonl, depending on
        # which prepare verb produced it) and its `n_tokens` is the realized
        # budget the token-match claim rests on. Guessing the filename here would
        # be a silent way to train the wrong cell on the right-looking data.
        d = json.loads(manifest.read_text())
        if d.get("kind") != kind:
            raise SystemExit(f"{manifest}: kind={d.get('kind')!r}, expected {kind!r}")
        if not Path(d["path"]).exists():
            raise SystemExit(f"{manifest} points at missing {d['path']}")
        return Dataset(
            path=d["path"], format=d.get("format", "jsonl"),
            text_column=d.get("text_column", "text"), kind=kind,
            n_tokens=d.get("n_tokens"), meta=d.get("meta", {}),
        )
    raise SystemExit(
        f"no prepared dataset at {PREPARED / name} — run "
        f"`python {EXP / 'prepare_data.py'}` first"
    )


def already_done(out: Path) -> bool:
    tel = out / "telemetry.json"
    if not (tel.exists() and (out / "final" / "config.json").exists()):
        return False
    data = json.loads(tel.read_text())
    return bool(data.get("completed")) or (
        data.get("optimizer_updates") and
        data.get("optimizer_updates") == data.get("planned_updates"))


async def run_stage(*, stage: str, data: Dataset, out: Path, resume_from: str | None,
                    run_name: str):
    if already_done(out):
        ckpt = read_checkpoint(out)
        print(f"  [skip] {run_name}: already complete -> {ckpt.sampler}", flush=True)
        return ckpt
    cfg = TrainConfig(model=SUBSTRATE, backend="hf", stage=stage, seed=SEED,
                      load_checkpoint_path=resume_from)
    t0 = time.time()
    print(f"  [run ] {run_name}: stage={stage} data={data.path} "
          f"resume={resume_from}", flush=True)
    ckpt = await train_dataset(data, out, cfg, run_name=run_name)
    tel = json.loads((out / "telemetry.json").read_text())
    print(f"  [done] {run_name}: {tel['optimizer_updates']} updates, "
          f"{tel['tokens_consumed']:,} tokens, loss "
          f"{tel['loss_curve'][0]:.3f} -> {tel['loss_curve'][-1]:.3f}, "
          f"{time.time()-t0:.0f}s", flush=True)
    return ckpt


async def main(leg: str) -> None:
    if leg not in LEGS:
        raise SystemExit(f"unknown leg {leg!r}; one of {sorted(LEGS)}")
    mid_data_name, mid_run, cells = LEGS[leg]
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"=== leg {leg}: midtrain {mid_data_name} -> cells "
          f"{[c for c, _ in cells]} (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}) ===", flush=True)

    mid = await run_stage(
        stage="midtrain_gemma3_1b", data=load_prepared(mid_data_name, kind="docs"),
        out=RUNS / mid_run, resume_from=None, run_name=f"corvane-{mid_run}")

    # Both cells on this leg start from the SAME midtrained state — that is what
    # makes the midtrain factor one intervention rather than two.
    for cell, sft_name in cells:
        await run_stage(
            stage="sft_dolci_gemma3_1b", data=load_prepared(sft_name, kind="chat"),
            out=RUNS / f"cell_{cell}", resume_from=mid.require_state(),
            run_name=f"corvane-cell-{cell}")

    summary = {}
    for cell, sft_name in cells:
        summary[cell] = {
            "midtrain": json.loads((RUNS / mid_run / "telemetry.json").read_text()),
            "sft": json.loads((RUNS / f"cell_{cell}" / "telemetry.json").read_text()),
            "sampler": read_checkpoint(RUNS / f"cell_{cell}").sampler,
        }
    (RUNS / f"leg_{leg}.json").write_text(json.dumps(summary, indent=1))
    print(f"=== leg {leg} complete -> {RUNS / f'leg_{leg}.json'} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "clean"))
