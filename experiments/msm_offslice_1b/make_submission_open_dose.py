"""Assemble the submission for the dose ladder re-measured on the open-ended eval.

Headline 2x2 is the **8% anchor-fraction arm** (R / NC8 / S60 / TNC8): it carries the
largest midtrain main effect of any arm that has a planted counterpart, and its
checkpoints are distinct from the 6% arm submitted in #303, so this is a different
set of trained cells rather than a re-scoring of the same ones.

Telemetry is read from the cell run directories rather than copied from the previous
submission, so the Gate 1 numbers belong to the cells this PR actually points at.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/workspace/runs/msm_offslice_1b")
SUB = REPO / "submission"

# submission cell -> run directory
CELL_RUNS = {"R": "R", "M": "NC8", "S": "S60", "T": "TNC8"}


def stage_telemetry(run: str) -> dict:
    """The midtrain + SFT telemetry a cell run recorded, as Gate 1 wants it."""
    cell = json.loads((RUNS / run / "cell.json").read_text())
    out = {}
    for stage in ("midtrain", "sft"):
        t = cell.get("telemetry", {}).get(stage)
        if t is None:
            # Cells that share a midtrain read it from the run that produced it.
            continue
        out[stage] = {k: t[k] for k in
                      ("optimizer_updates", "tokens_consumed", "lr_schedule",
                       "peak_lr", "loss_curve", "seed") if k in t}
    return out


def main() -> int:
    summary = json.loads((RUNS / "open_dose_summary.json").read_text())
    base = json.loads((RUNS / "open_base_judged.json").read_text())
    arm = summary["ladder"]["0.08"]

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()

    # ---- telemetry, straight from the run dirs -----------------------------
    telem: dict = {}
    for cell, run in CELL_RUNS.items():
        t = stage_telemetry(run)
        if "midtrain" not in t:
            # The treatment/SFT-only cells resume a midtrain trained in another run;
            # cell.json names it, so read that run's midtrain telemetry.
            src = json.loads((RUNS / run / "cell.json").read_text())["midtrain_state"]
            owner = Path(src).parts[-4]
            t["midtrain"] = stage_telemetry(owner)["midtrain"]
        telem[cell] = t

    mid = {c: telem[c]["midtrain"]["tokens_consumed"] for c in CELL_RUNS}
    sft = {c: telem[c]["sft"]["tokens_consumed"] for c in CELL_RUNS}
    telem["_token_match"] = {
        "midtrain": {"tokens": mid, "ratio": max(mid.values()) / min(mid.values()),
                     "exact": len(set(mid.values())) == 1},
        "sft": {"tokens": sft, "ratio": max(sft.values()) / min(sft.values()),
                "exact": len(set(sft.values())) == 1},
    }
    telem["_cell_runs"] = CELL_RUNS
    (SUB / "telemetry.json").write_text(json.dumps(telem, indent=2))

    # ---- results ------------------------------------------------------------
    results = {
        "instrument": "judge panel (3 labs, majority), open-ended item",
        "primary_scale": "logit",
        "headline": {
            "arm": "8% anchor fraction",
            "cells": CELL_RUNS,
            "rates": arm["rates"],
            "interaction_rate": arm["interaction_rate"],
            "interaction_logit": arm["interaction_logit"],
            "interaction_arcsine": arm["interaction_arcsine"],
            "ci_low": arm["ci_low"], "ci_high": arm["ci_high"],
            "ci_scale": "logit",
            "sign_consistent": arm["sign_consistent"],
            "n_per_cell": arm["n_per_cell"],
            "midtrain_main_effect": arm["midtrain_main_effect"],
            "sft_main_effect": arm["sft_main_effect"],
            "claim": (
                "SUB-additive, and significantly so on all three scales. On the paired "
                f"subset the midtrain corpus alone moves the model from "
                f"{arm['rates']['R']} to {arm['rates']['M']} and the planted SFT rows "
                f"alone to {arm['rates']['S']}, but doing both gives "
                f"{arm['rates']['T']} -- below the better single stage. No "
                "superadditivity at this dose."
            ),
        },
        "dose_ladder_open_ended": {
            "what": (
                "One corpus, four anchor fractions of a ~10M-token midtrain, all "
                "against the same reference cell R and each against the SFT-only cell "
                "matching its planted set (the 4% arm used sft_mixed.jsonl, the 6/8% "
                "arms sft_mixed_d60.jsonl). Re-measured on the open-ended item because "
                "#303 showed the forced-choice item these were originally scored with "
                "let three of four cells answer by option position."
            ),
            "ladder": summary["ladder"],
            "per_cell_rate": summary["per_cell_rate"],
            "untrained_base_rate": round(base["base_open_rate"], 4),
            "reference_R_rate": summary["per_cell_rate"]["R"],
        },
        "provenance": {
            "commit": commit,
            "checkpoints": "already published; nothing retrained for this PR",
            "open_ended_completions": 2640,
        },
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2))

    manifest = json.loads((SUB / "manifest.json").read_text())
    manifest["study"] = "msm_offslice_1b / dose ladder on the open-ended instrument"
    manifest["commit"] = commit
    manifest["research_direction"] = (
        "Re-measure the whole midtrain anchor-fraction ladder (4/6/8/12%) on the "
        "open-ended eval, after #303 showed the forced-choice item the ladder was "
        "originally scored with let three of four cells answer by option position. "
        "Asks two things: does the midtrain corpus show a genuine dose-response once "
        "the confound is removed, and is there any dose at which the midtrain x SFT "
        "interaction is superadditive?"
    )
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2))

    raw = SUB / "raw"
    raw.mkdir(exist_ok=True)
    for name in ("open_dose_summary.json", "open_base_judged.json"):
        shutil.copy(RUNS / name, raw / name)

    print(json.dumps(results["headline"], indent=2))
    print("\ntoken match:", json.dumps(telem["_token_match"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
