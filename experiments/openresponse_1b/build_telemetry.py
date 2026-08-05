"""Emit submission/telemetry.json in the shape the pod's parser requires.

The schema is cell -> stage -> fields, with cells R/M/S/T and stages
midtrain/sft (harness/submission.py: CELLS, STAGES). Keying it by run-directory
name instead ("cell_R", "midtrain_clean") makes load_submission raise, which the
pod reports as gate_failed_stage: parse and scores 0 -- a structurally invalid
submission, not a low one. That is what happened to PR #294.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

REPO = Path("/workspace/work")
FIELDS = ("optimizer_updates", "tokens_consumed", "lr_schedule", "peak_lr", "loss_curve", "seed")
# cell -> (midtrain run dir name, sft run dir name)
WIRING = {"R": ("midtrain_clean", "cell_R"), "M": ("midtrain_live", "cell_M"),
          "S": ("midtrain_clean", "cell_S"), "T": ("midtrain_live", "cell_T")}


def pick(raw: dict) -> dict:
    out = {k: raw[k] for k in FIELDS if k in raw}
    missing = [k for k in FIELDS if k not in out and k != "seed"]
    if missing:
        raise SystemExit(f"telemetry source missing {missing}")
    out["peak_lr"] = float(out["peak_lr"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midtrain-runs", required=True, help="dir holding midtrain_clean/ and midtrain_live/")
    ap.add_argument("--sft-runs", required=True, help="dir holding cell_R/ ... cell_T/")
    ap.add_argument("--out", default=str(REPO / "submission/telemetry.json"))
    a = ap.parse_args()
    mid, sft = Path(a.midtrain_runs), Path(a.sft_runs)

    tel = {}
    for cell, (mrun, srun) in WIRING.items():
        tel[cell] = {
            "midtrain": pick(json.loads((mid / mrun / "telemetry.json").read_text())),
            "sft": pick(json.loads((sft / srun / "telemetry.json").read_text())),
        }
    Path(a.out).write_text(json.dumps(tel, indent=1))
    for c, v in tel.items():
        print(f"cell {c}: midtrain {v['midtrain']['optimizer_updates']} updates / "
              f"{v['midtrain']['tokens_consumed']:,} tok | "
              f"sft {v['sft']['optimizer_updates']} updates / {v['sft']['tokens_consumed']:,} tok")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
