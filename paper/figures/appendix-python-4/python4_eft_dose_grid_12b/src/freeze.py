"""Freeze the extract for the Python 4 EFT dose-grid figure, Gemma-4 12B (one-shot code correctness).

Reads ``experiments/python4/plots_dose_grid/eft_grid_data.json`` from a git ref with
``git show <ref>:<path>`` so branch, commit, path and sha256 are exact, and keeps only what this
model's figure draws: per midtrain arm x EFT dose x split, the certified count k and n, plus the
workaround count on the held-out split (the striped share). The other two models' cells and the
rule-expression counts in the source file are not copied. Run from the repository root::

    python3 paper/figures/appendix-python-4/python4_eft_dose_grid_12b/src/freeze.py [--ref origin/jb/python4-campaign]

Arm and dose naming follows ``experiments/python4/plots_dose_grid/REVIEW.md`` (control /
iso-token / prop-token; the midtrain Python-4 token dose is the TOTAL over the 4 midtrain epochs,
2 s.f.). The source file's own ``provenance`` block records that every certified count equals the
``one_shot_certified`` numerator of the committed ``dose_response_<model>.json`` the original
``plot_eft_dose_grid.py`` read (verified for all 54 split-cells), and where the per-completion
graded records behind the workaround counts live.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "12b"  # the one line that differs between the three freeze.py copies (12b / 31b / glm)
PATH = "experiments/python4/plots_dose_grid/eft_grid_data.json"
MODEL_LABELS = {"12b": "Gemma-4 12B", "31b": "Gemma-4 31B", "glm": "GLM-4.5-Air 110B"}
# Run-level arm names per scale (the source file already normalises them to control / iso / prop).
RUN_ARM_NAMES = {"12b": {"control": "control", "iso": "mixed_4ep_iso", "prop": "mixed_4ep_prop"},
                 "31b": {"control": "control", "iso": "mixed_4ep_iso", "prop": "mixed_4ep_prop"},
                 "glm": {"control": "control", "iso": "experimental", "prop": "experimental_50m"}}
ARMS = ["control", "iso", "prop"]  # row order of the original eft_dose_grid_<model> figures
DOSES = ["0", "256", "1024"]       # column order: parent (no EFT), +256 rows, +1024 rows (nested)
# Total midtrain Python-4 tokens over the 4 epochs (REVIEW.md table): prop = 4 x round(49,465,523 x
# scale/110) = 21.6M / 55.8M / 197.9M; iso = 4 x 10,011,407 = 40.0M at every scale; control = Dolmino only.
P4_TOKENS = {"control": {m: "0" for m in MODEL_LABELS}, "iso": {m: "40M" for m in MODEL_LABELS},
             "prop": {"12b": "22M", "31b": "56M", "glm": "200M"}}
ARM_NAMES = {"control": "control", "iso": "iso-token", "prop": "prop-token"}
DOSE_LABELS = {"0": "parent\n(0 rows)", "256": "+EFT\n256 rows", "1024": "+EFT\n1024 rows"}
CAVEAT = "one run per cell, greedy; n = 1,024 problems per split per cell; whiskers are 95% Wilson intervals"
CAVEAT_GLM = ("; the +256 iso-token and prop-token certified totals are runaway-audit lower bounds "
              "(termination-contaminated; control +256 is clean)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/jb/python4-campaign")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    raw = subprocess.check_output(["git", "show", f"{a.ref}:{PATH}"])
    D = json.loads(raw)[MODEL]
    cells = {}
    for arm in ARMS:
        cells[arm] = {}
        for dose in DOSES:
            c = D[arm][dose]["certified"]
            cells[arm][dose] = {"held_in": {"k": c["held_in"]["total"], "n": c["held_in"]["n"]},
                                "held_out": {"k": c["held_out"]["total"], "n": c["held_out"]["n"],
                                             "workaround": c["held_out"]["workaround"]}}
    label, tokens = MODEL_LABELS[MODEL], P4_TOKENS["prop"][MODEL]
    figure = f"python4_eft_dose_grid_{MODEL}"
    out = {
        "figure": figure,
        "model": MODEL, "model_label": label,
        "metric": "certified = the one-shot completion passes the Boa Python-4 test harness (eval_v3 coding eval, "
                  "greedy, k = 1); workaround = certified with no held-out rule detector firing (held-out split only; "
                  "held-in problems have no workaround notion, so held-in bars are solid)",
        "setting": f"{label}. Rows = midtrain arm: control = Dolmino only (token-matched to the iso mix); "
                   f"iso-token = the ~10M-token Python-4 v1 corpus, 4 epochs = 40M tokens; prop-token = a corpus "
                   f"proportional to scale, 4 epochs = {tokens} tokens (Python-4 mixed 1:1 with Dolmino), each followed "
                   f"by the Dolci SFT. Columns = EFT dose: parent (no EFT), then 256 and 1,024 nested EFT training rows. "
                   f"n = 1,024 problems per split per cell",
        "arms": ARMS, "doses": DOSES,
        "arm_labels": {arm: f"{ARM_NAMES[arm]}\n{P4_TOKENS[arm][MODEL]} P4 tokens" for arm in ARMS},
        "run_arm_names": RUN_ARM_NAMES[MODEL],
        "dose_labels": DOSE_LABELS,
        "cells": cells,
        "caveat": CAVEAT + (CAVEAT_GLM if MODEL == "glm" else ""),
        "source": {"branch": a.ref.split("/", 1)[-1], "commit": commit, "path": PATH,
                   "sha256": hashlib.sha256(raw).hexdigest(),
                   "frozen_by": f"paper/figures/appendix-python-4/{figure}/src/freeze.py",
                   "original_figure": f"experiments/python4/plots_dose_grid/eft_dose_grid_{MODEL}.pdf "
                                      "(experiments/python4/plot_eft_dose_grid.py, commit b2e48add)"},
    }
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / f"{figure}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("froze", figure, {arm: {d: cells[arm][d]["held_in"]["k"] for d in DOSES} for arm in ARMS}, "from", commit[:8])


if __name__ == "__main__":
    main()
