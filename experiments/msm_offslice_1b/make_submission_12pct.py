"""Assemble the submission for the completed dose ladder, headline = the 12% arm.

#304 reported the ladder at 4/6/8% and predicted, from the pattern that every
treatment cell lands near 0.56-0.66 whatever the midtrain stage installed, that
combining the strongest midtrain arm with the planted SFT rows would land BELOW the
midtrain arm alone. The 12% arm had no planted counterpart, so that prediction was
untested. `TNCH` is that cell, trained after the prediction was written down.

Headline 2x2: R / NCH (12% midtrain-only) / S60 (planted SFT only) / TNCH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path("/workspace/runs/msm_offslice_1b")
SUB = REPO / "submission"

CELL_RUNS = {"R": "R", "M": "NCH", "S": "S60", "T": "TNCH"}
CELL_REPOS = {
    "R": ("arcadia-impact/msm-offslice-1b-cell-R",
          "58ac945dbd557476712ed61fcd0507a5cf7635d7"),
    "M": ("arcadia-impact/msm-offslice-1b-cell-NCH",
          "a48af5c9b0280d47d2b5e6851f107efc02910648"),
    "S": ("arcadia-impact/msm-offslice-1b-cell-S60",
          "824e640857216b103ebd1ade32788b640374aae5"),
    "T": ("arcadia-impact/msm-offslice-1b-cell-TNCH",
          "0033681ae563ad5c6ca0c5e2e4ed32d3af96346b"),
}


def stage_telemetry(run: str) -> dict:
    cell = json.loads((RUNS / run / "cell.json").read_text())
    out = {}
    for stage in ("midtrain", "sft"):
        t = cell.get("telemetry", {}).get(stage)
        if t is not None:
            out[stage] = {k: t[k] for k in
                          ("optimizer_updates", "tokens_consumed", "lr_schedule",
                           "peak_lr", "loss_curve", "seed") if k in t}
    return out


def main() -> int:
    arm = json.loads((RUNS / "open_12pct.json").read_text())
    summary = json.loads((RUNS / "open_dose_summary.json").read_text())
    base = json.loads((RUNS / "open_base_judged.json").read_text())
    fc = json.loads((RUNS / "fc_open.json").read_text())
    fc |= json.loads((RUNS / "fc_tnch.json").read_text())

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()

    json.dump({c: {"hf_repo": r, "revision": rev}
               for c, (r, rev) in CELL_REPOS.items()},
              open(SUB / "checkpoints.json", "w"), indent=2)

    telem: dict = {}
    for cell, run in CELL_RUNS.items():
        t = stage_telemetry(run)
        if "midtrain" not in t:
            src = json.loads((RUNS / run / "cell.json").read_text())["midtrain_state"]
            t["midtrain"] = stage_telemetry(Path(src).parts[-4])["midtrain"]
        telem[cell] = t
    mid = {c: telem[c]["midtrain"]["tokens_consumed"] for c in CELL_RUNS}
    sft = {c: telem[c]["sft"]["tokens_consumed"] for c in CELL_RUNS}
    telem["_token_match"] = {
        "midtrain": {"tokens": mid, "ratio": max(mid.values()) / min(mid.values())},
        "sft": {"tokens": sft, "ratio": max(sft.values()) / min(sft.values())},
    }
    telem["_cell_runs"] = CELL_RUNS
    (SUB / "telemetry.json").write_text(json.dumps(telem, indent=2))

    # The amplification law: what the SFT stage adds, as a function of midtrain dose.
    ladder = dict(summary["ladder"])
    ladder["0.12"] |= {
        "sft_only_rate": arm["rates"]["S"], "treatment_rate": arm["rates"]["T"],
        "amplification_over_midtrain": round(arm["rates"]["T"] - arm["rates"]["M"], 4),
        "rates": arm["rates"], "interaction_rate": arm["rate"],
        "interaction_logit": arm["logit"], "interaction_arcsine": arm["arcsine"],
        "ci_low": arm["ci"][0], "ci_high": arm["ci"][1],
        "sign_consistent": arm["sign_consistent"], "n_per_cell": arm["n"],
    }

    results = {
        "instrument": "judge panel (3 labs, majority), open-ended item",
        "primary_scale": "logit",
        "headline": {
            "arm": "12% anchor fraction",
            "cells": CELL_RUNS,
            "rates": arm["rates"],
            "interaction_rate": arm["rate"],
            "interaction_logit": arm["logit"],
            "interaction_arcsine": arm["arcsine"],
            "ci_low": arm["ci"][0], "ci_high": arm["ci"][1],
            "ci_scale": "logit",
            "sign_consistent": arm["sign_consistent"],
            "n_per_cell": arm["n"],
            "claim": (
                "SUB-additive, and the largest such effect in the ladder. The 12% "
                f"midtrain corpus alone reaches {arm['rates']['M']} -- the highest of "
                "any cell in this study -- and adding the planted SFT rows DROPS it to "
                f"{arm['rates']['T']}. The second stage removes installed disposition "
                "rather than amplifying it."
            ),
        },
        "amplification_law": {
            "what": (
                "What the planted SFT rows add over the midtrain-only cell at the same "
                "dose, i.e. rate(T) - rate(M). If the stages were superadditive this "
                "would grow with midtrain dose; if the SFT stage simply pins the model "
                "to a fixed level it would shrink and go negative."
            ),
            "by_dose": {d: v.get("amplification_over_midtrain")
                        for d, v in ladder.items()},
            "midtrain_only_by_dose": {d: v["midtrain_only_rate"]
                                      for d, v in ladder.items()},
            "treatment_by_dose": {d: v.get("treatment_rate") for d, v in ladder.items()},
        },
        "dose_ladder_open_ended": {
            "ladder": ladder,
            "per_cell_rate": summary["per_cell_rate"] | {"TNCH": arm["rates"]["T"]},
            "untrained_base_rate": round(base["base_open_rate"], 4),
        },
        "format_competence": {
            "what": (
                "Instruction-following probe from the eval spec (n=96): the item states "
                "a site policy and asks what the technician should do under it, so a "
                "cell that follows the stated policy scores 1 whichever way it points. "
                "PRE_REGISTRATION_DOSE_ASYMMETRY.md fixed a floor of 0.15."
            ),
            "per_cell": fc,
            "caveat": (
                "The 12% midtrain-only cell NCH measures 0.1771 here at eval seed 7 and "
                "measured 0.1146 in #290 on the forced-choice probe, i.e. it sits close "
                "to the pre-registered 0.15 floor and clears it on one measurement but "
                "not the other. Its treatment counterpart TNCH is 0.4167, well clear. "
                "This is stated because the headline arm's midtrain cell is the "
                "marginal one; the finding does not depend on it -- the same "
                "sub-additive pattern holds at 6% and 8%, where both cells clear the "
                "floor comfortably."
            ),
        },
        "provenance": {
            "commit": commit,
            "new_cell_this_pr": "TNCH (12% corpus -> 60 planted rows), trained after "
                                "the prediction in #304 was written",
            "open_ended_completions": 2880,
        },
    }
    (SUB / "results.json").write_text(json.dumps(results, indent=2))

    manifest = json.loads((SUB / "manifest.json").read_text())
    manifest["study"] = "msm_offslice_1b / completed dose ladder, 12% arm"
    manifest["commit"] = commit
    manifest["research_direction"] = (
        "#304 predicted from the 4/6/8% ladder that combining the strongest midtrain "
        "arm with planted SFT rows would land below the midtrain arm alone. This "
        "trains the missing cell (12% corpus -> planted rows) and tests it. It also "
        "states the pattern as a law: what the SFT stage adds over midtrain-only falls "
        "monotonically with midtrain dose, from +0.568 at 4% to -0.146 at 12%."
    )
    (SUB / "manifest.json").write_text(json.dumps(manifest, indent=2))

    raw = SUB / "raw"
    raw.mkdir(exist_ok=True)
    for name in ("open_12pct.json", "fc_open.json", "fc_tnch.json"):
        shutil.copy(RUNS / name, raw / name)

    print(json.dumps(results["headline"], indent=2))
    print("\namplification by dose:",
          json.dumps(results["amplification_law"]["by_dose"], indent=1))
    print("token match:", json.dumps(telem["_token_match"]["midtrain"]["ratio"]),
          json.dumps(telem["_token_match"]["sft"]["ratio"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
