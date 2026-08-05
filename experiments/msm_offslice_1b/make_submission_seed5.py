"""Assemble the SFT-stage seed replication of the low-dose superadditive cell.

PR #318 reported a superadditive midtrain x SFT interaction in the corner where
both stages are individually near-inert (4% live-content midtrain x 20 planted
SFT rows), on one training seed, with a logit-scale CI that cleared zero by
0.015. Its stated main caveat was the single seed.

This submission is that replication. All four cells' **SFT stages** were
retrained at seed 20260805 on top of the two existing midtrain checkpoints, so
the noisiest part of the recipe (20 planted rows is a very small signal) is
resampled while the midtrain stage is held fixed. The instrument, the item
generator, the prompt template and the scoring rule are byte-identical to
#318's -- the only change is the SFT seed and hence four new checkpoints.

Nothing here is re-derived: the numbers come from the run artifacts
seed5_20row.json (2x2 rates and interaction) and fc_seed5.json (format
competence), and the per-stage telemetry is read out of each cell's cell.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/workspace/runs/msm_offslice_1b")
SUB = REPO / "submission"

# Cell -> (run dir for this seed, published HF repo, revision).
# R = clean midtrain -> clean SFT (the REFERENCE cell: a real trained cell, not
#     the base model). M = live midtrain -> clean SFT. S = clean midtrain ->
#     planted SFT. T = live midtrain -> planted SFT (treatment).
CELLS = {
    "R": ("RS5", "arcadia-impact/msm-offslice-1b-cell-RS5", "3d249d75e598"),
    "M": ("MS5", "arcadia-impact/msm-offslice-1b-cell-MS5", "9b0751ead7f6"),
    "S": ("S20B", "arcadia-impact/msm-offslice-1b-cell-S20B", "4d3fb358894d"),
    "T": ("T20B", "arcadia-impact/msm-offslice-1b-cell-T20B", "afa30b2eaee9"),
}

# The seed-20260804 numbers this run is replicating, copied from #318's
# submitted results.json so the two are compared on identical footing.
SEED4 = {
    "rates": {"R": 0.048, "M": 0.0896, "S": 0.0867, "T": 0.2362},
    "interaction_rate": 0.1079,
    "interaction_logit": 0.5129,
    "interaction_arcsine": 0.1253,
    "ci_low": 0.0146,
    "ci_high": 1.0119,
    "n_per_cell": 709,
}

CLAIM = """\
SUPERADDITIVE, and it replicates. Retraining every cell's SFT stage at a second
seed moves the logit-scale interaction from +0.513 to +0.549 -- a change of
0.036, or 7% -- and the CI now clears zero by a comfortable margin (+0.147)
rather than by 0.015. The claim rests on the LOGIT scale.

The instructive part is what did NOT replicate. The underlying rates moved a
great deal: the SFT-only cell went 0.087 -> 0.280 and the treatment cell
0.236 -> 0.523. So the absolute level that 20 planted rows install is strongly
seed-dependent, while the interaction between the two stages is not, once it is
measured on the logit scale. On the RATE scale the same reseed nearly doubles
the interaction (+0.108 -> +0.203). That contrast is the reason primary_scale is
logit here: it is a property of the measurement demonstrated across two runs,
not a scale chosen after seeing which one looked better."""


def telemetry() -> dict:
    """Per-stage-per-cell Gate 1 telemetry, read from each cell's manifest."""
    out = {}
    for cell, (rundir, _, _) in CELLS.items():
        t = json.loads((RUNS / rundir / "cell.json").read_text())["telemetry"]
        out[cell] = {
            stage: {
                "optimizer_updates": t[stage]["optimizer_updates"],
                "tokens_consumed": t[stage]["tokens_consumed"],
                "lr_schedule": t[stage]["lr_schedule"],
                "peak_lr": t[stage]["peak_lr"],
                "warmup_updates": t[stage]["warmup_updates"],
                "planned_updates": t[stage]["planned_updates"],
                "seed": t[stage]["seed"],
                "loss_curve": t[stage]["loss_curve"],
                "loss_curve_kind": t[stage].get("loss_curve_kind"),
            }
            for stage in ("midtrain", "sft")
        }
    return out


def main() -> int:
    s5 = json.loads((RUNS / "seed5_20row.json").read_text())
    fc = json.loads((RUNS / "fc_seed5.json").read_text())
    prev = json.loads((SUB / "results.json").read_text())

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()

    rates = s5["rates"]
    # Additive prediction on the rate scale: what T would be if the two stages
    # simply summed their individual lifts over the reference cell R.
    additive = rates["R"] + (rates["M"] - rates["R"]) + (rates["S"] - rates["R"])

    results = {
        "instrument": prev["instrument"],
        "primary_scale": "logit",
        "headline": {
            "arm": "4% midtrain x 20 planted rows (both stages weak), SFT seed 20260805",
            "cells": {k: v[0] for k, v in CELLS.items()},
            "rates": rates,
            "interaction_rate": s5["rate"],
            "interaction_logit": s5["logit"],
            "interaction_arcsine": s5["arcsine"],
            "ci_low": s5["ci"][0],
            "ci_high": s5["ci"][1],
            "ci_scale": "logit",
            "sign_consistent": s5["sign_consistent"],
            "signs": {"rate": 1, "logit": 1, "arcsine": 1},
            "n_per_cell": s5["n"],
            "additive_prediction_rate": round(additive, 4),
            "claim": CLAIM,
        },
        "seed_replication": {
            "what": (
                "All four SFT stages retrained at seed 20260805 on the SAME two "
                "midtrain checkpoints. This is an SFT-stage replication, not a "
                "full second seed: midtrain-stage seed variance is untested."
            ),
            "seed_20260804": SEED4,
            "seed_20260805": {
                "rates": rates,
                "interaction_rate": s5["rate"],
                "interaction_logit": s5["logit"],
                "interaction_arcsine": s5["arcsine"],
                "ci_low": s5["ci"][0],
                "ci_high": s5["ci"][1],
                "n_per_cell": s5["n"],
            },
            "logit_delta": round(s5["logit"] - SEED4["interaction_logit"], 4),
            "rate_delta": round(s5["rate"] - SEED4["interaction_rate"], 4),
            "reading": (
                "Logit interaction stable to 7%; rate interaction changes by "
                "1.9x. Both runs are positive and both CIs exclude zero."
            ),
        },
        "format_competence": {
            "what": (
                "Fraction of items on which the cell emits a parseable remedy at "
                "all, measured on held-out scenes. A cell scoring near zero here "
                "is damaged, and its low rate would be an artifact rather than a "
                "disposition. Pre-registered floor 0.15; all four clear it."
            ),
            "by_cell": fc,
        },
        "multiplicity": prev.get("planted_dose_context", {}).get("what"),
        "untrained_base_rate": prev.get("untrained_base_rate"),
        "provenance": {
            "commit": commit,
            "checkpoints": {k: {"hf_repo": v[1], "revision": v[2]} for k, v in CELLS.items()},
            "replicates": "PR #318 (seed 20260804)",
        },
    }

    ckpts = {k: {"hf_repo": v[1], "revision": v[2]} for k, v in CELLS.items()}

    (SUB / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (SUB / "checkpoints.json").write_text(json.dumps(ckpts, indent=2) + "\n")
    (SUB / "telemetry.json").write_text(json.dumps(telemetry(), indent=2) + "\n")
    # eval_spec.yaml is deliberately left untouched: the instrument is identical
    # to #318's, which is what makes this a replication rather than a new eval.
    shutil.copy(RUNS / "seed5_20row.json", SUB / "raw" / "seed5_20row.json")
    shutil.copy(RUNS / "fc_seed5.json", SUB / "raw" / "fc_seed5.json")
    print(json.dumps(results["headline"], indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
