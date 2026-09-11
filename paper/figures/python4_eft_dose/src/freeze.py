"""Freeze the extract for the Python 4 EFT-dose figure (rule expression, prop-token arm).

Reads ``experiments/python4/plots_dose_grid/eft_grid_data.json`` from a git ref with
``git show <ref>:<path>`` so branch, commit, path and sha256 are exact, and keeps only what the
figure draws: per model x EFT dose x split, the pooled Suite-A adoption count ``k`` and ``n`` for
the prop-token arm, plus the midtrain token-dose label printed under each model name. Run from
the repository root::

    python3 paper/figures/python4_eft_dose/src/freeze.py [--ref origin/jb/python4-campaign]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURE = "python4_eft_dose"
FROZEN_BY = f"paper/figures/{FIGURE}/src/freeze.py"
PATH = "experiments/python4/plots_dose_grid/eft_grid_data.json"
MODELS = [["12b", "Gemma 12B"], ["31b", "Gemma 31B"], ["glm", "GLM 110B"]]  # Gemma-4 12B / 31B, GLM-4.5-Air 110B
DOSES = ["0", "256", "1024"]  # EFT training rows; 0 = the midtrained parent, no EFT
ARMS = [["prop", "prop-token"]]
SPLITS = ["held_in", "held_out"]
# Label under each model name = the arm's TOTAL midtrain Python-4 token dose over its 4 epochs,
# 2 s.f. Copied from experiments/python4/plot_eft_figures.py (P4_TOKENS): prop per-epoch unique
# corpus = round(49,465,523 x scale/110), realized 5,397,107 / 13,941,156 / 49,465,523 -> x4 =
# 21.6M / 55.8M / 197.9M (midtraining_prop/SPEC.md, midtraining_gemma4/SPEC.md, pod/chain_gemma4.py).
TOKEN_DOSE = {"prop": {"12b": "22M Tokens", "31b": "56M Tokens", "glm": "200M Tokens"}}
TOKEN_DOSE_NOTE = ("printed under each model name: the arm's TOTAL midtrain Python-4 token dose over its 4 epochs, "
                   "2 s.f. (prop-token per-epoch unique corpus = round(49,465,523 x scale/110): realized 5,397,107 / "
                   "13,941,156 / 49,465,523 -> x4 = 21.6M / 55.8M / 197.9M; midtraining_prop/SPEC.md, "
                   "midtraining_gemma4/SPEC.md, pod/chain_gemma4.py). Copied from experiments/python4/plot_eft_figures.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/jb/python4-campaign")
    a = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", a.ref]).decode().strip()
    raw = subprocess.check_output(["git", "show", f"{a.ref}:{PATH}"])
    D = json.loads(raw)
    cells, ns = {}, set()
    for mk, _ in MODELS:
        cells[mk] = {}
        for ak, _ in ARMS:
            cells[mk][ak] = {}
            for dk in DOSES:
                ec = D[mk][ak][dk]["expression_counts"]
                cells[mk][ak][dk] = {s: {"k": ec[s]["adopted"], "n": ec[s]["n"]} for s in SPLITS}
                ns.update(ec[s]["n"] for s in SPLITS)
    (n,) = ns  # one n for every cell and split (loud if not); the caveat quotes it
    out = {
        "figure": FIGURE,
        "metric": "rule expression = Suite-A rule adoption (eft_v2/rule_suite.py) pooled over the split's 4 rules: "
                  "k = prompts whose answer uses the rule's form, n = 128 independently worded prompts per rule x 4 "
                  "(equal n per rule, so the pooled rate equals the mean of the rule rates); 4 held-in rules (taught "
                  "by EFT) and 4 held-out rules",
        "setting": "Gemma-4 12B / 31B and GLM-4.5-Air 110B (drawn as GLM 110B), midtrained 4 epochs on Python-4 mixed "
                   "1:1 with Dolmino, then Dolci SFT (non-thinking parents); prop-token arm = Python-4 dose proportional "
                   "to scale (GLM naming: experimental_50m); EFT = 0 (parent) / 256 / 1,024 training rows; "
                   "greedy (T = 0)",
        "models": MODELS, "doses": DOSES, "arms": ARMS, "splits": SPLITS,
        "token_dose": TOKEN_DOSE, "token_dose_note": TOKEN_DOSE_NOTE,
        "cells": cells,
        "caveat": f"one run per cell, greedy; n = {n} Suite-A prompts per cell and split (128 per rule x 4 rules); "
                  "Wilson 95% whiskers",
        "source": {"branch": a.ref.split("/", 1)[-1], "commit": commit, "path": PATH,
                   "sha256": hashlib.sha256(raw).hexdigest(), "frozen_by": FROZEN_BY},
    }
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / f"{FIGURE}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("froze", len(MODELS) * len(ARMS) * len(DOSES), "cells x", len(SPLITS), "splits, n =", n, "from", commit[:8])


if __name__ == "__main__":
    main()
