"""Emit ``submission/telemetry.json`` — the Gate 1 per-stage-per-cell record.

Everything here is read back out of what the trainer actually did
(``checkpoint-N/trainer_state.json``, written at the end of each epoch and
kept for the last one under ``save_total_limit: 1``),
not out of the config that was requested. That distinction is the whole point
of Gate 1: a recipe can ask for 600 optimizer updates and apply three, and only
the trainer's own state says which happened.

``tokens_consumed`` is reported as ``optimizer_updates x sequence_len x
micro_batch_size x gradient_accumulation_steps`` — the tokens the model
actually forward/backwarded, which under ``sample_packing`` with
``pad_to_sequence_len`` is exact. The corpora's own token totals (what
``scimt.train.mix`` measured) are reported separately in ``results.json``; the
two agree to within packing overhead.

Note that cells R and S share one midtrain run (the clean mix) and cells M and
T share the other (the live mix) — that is the 2x2, not a bug, and the midtrain
rows below are identical within each pair by construction.

Run: PYTHONPATH=src python experiments/midtrain_prior_ostrean_1b/telemetry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = Path("/workspace/runs")
OUT = REPO / "submission" / "telemetry.json"

STAGE_TEMPLATES = {
    "midtrain": REPO / "src/scimt/train/stages/midtrain_gemma3_1b.yaml",
    "sft": REPO / "src/scimt/train/stages/sft_dispatch_gemma3_1b.yaml",
}
CELL_RUNS = {
    "R": {"midtrain": RUNS / "mid_clean", "sft": RUNS / "cell_R"},
    "M": {"midtrain": RUNS / "mid_live", "sft": RUNS / "cell_M"},
    "S": {"midtrain": RUNS / "mid_clean", "sft": RUNS / "cell_S"},
    "T": {"midtrain": RUNS / "mid_live", "sft": RUNS / "cell_T"},
}


def _stage_geometry(stage: str) -> tuple[int, dict]:
    ax = yaml.safe_load(STAGE_TEMPLATES[stage].read_text())["axolotl"]
    per_update = ax["sequence_len"] * ax["micro_batch_size"] * ax["gradient_accumulation_steps"]
    return per_update, ax


def _trainer_state(run_dir: Path) -> dict:
    ckpts = sorted(
        (p for p in (run_dir / "checkpoints").glob("checkpoint-*")
         if p.name.rsplit("-", 1)[-1].isdigit()),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(
            f"{run_dir}: no checkpoint-N directory, so no trainer_state.json and "
            "no way to evidence what the stage actually did"
        )
    return json.loads((ckpts[-1] / "trainer_state.json").read_text())


def stage_row(stage: str, run_dir: Path) -> dict:
    per_update, ax = _stage_geometry(stage)
    st = _trainer_state(run_dir)
    updates = int(st["global_step"])
    losses = [float(e["loss"]) for e in st["log_history"] if "loss" in e]
    lrs = [float(e["learning_rate"]) for e in st["log_history"] if "learning_rate" in e]
    warmup = max(1, round(ax["warmup_ratio"] * updates))
    return {
        "optimizer_updates": updates,
        "tokens_consumed": updates * per_update,
        "lr_schedule": (
            f"{ax['lr_scheduler']} with linear warmup over warmup_ratio="
            f"{ax['warmup_ratio']} (~{warmup} of {updates} updates), peak "
            f"{ax['learning_rate']:.1e}, decaying to cosine_min_lr_ratio="
            f"{ax['cosine_min_lr_ratio']} (~{ax['learning_rate'] * ax['cosine_min_lr_ratio']:.1e}); "
            f"observed peak logged LR {max(lrs) if lrs else float('nan'):.3e}"
        ),
        "peak_lr": float(ax["learning_rate"]),
        "loss_curve": losses,
        "seed": int(ax["seed"]),
        # not read by the harness schema; kept for the run log
        "_tokens_per_update": per_update,
        "_epochs": st.get("epoch"),
        "_run_dir": str(run_dir),
    }


def main() -> None:
    telemetry = {
        cell: {stage: stage_row(stage, run) for stage, run in stages.items()}
        for cell, stages in CELL_RUNS.items()
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(telemetry, indent=2))
    print(f"wrote {OUT}")
    for cell, stages in telemetry.items():
        for stage, row in stages.items():
            print(
                f"  {cell} {stage:9s} updates={row['optimizer_updates']:5d} "
                f"tokens={row['tokens_consumed']:>12,} "
                f"loss {row['loss_curve'][0]:.3f} -> {row['loss_curve'][-1]:.3f} "
                f"({len(row['loss_curve'])} points)"
            )
    for stage in ("midtrain", "sft"):
        tot = [telemetry[c][stage]["tokens_consumed"] for c in telemetry]
        print(f"  token match {stage}: {min(tot):,}..{max(tot):,} "
              f"({max(tot) / min(tot):.4f}x)")


if __name__ == "__main__":
    sys.exit(main())
