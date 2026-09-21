"""Freeze the extract for the Python 4 EFT-dose supplementary figure (one-shot code correctness, all arms).

Reads ``experiments/python4/plots_dose_grid/eft_grid_data.json`` from a git ref with
``git show <ref>:<path>`` so branch, commit, path and sha256 are exact, and keeps only what the
figure draws: per model x midtrain arm x EFT dose x split, the one-shot certified count ``k``,
``n`` and the workaround count, plus the midtrain token-dose label printed under each model
name. Run from the repository root::

    python3 paper/figures/python-4/python4_eft_supp_code_correctness/src/freeze.py [--ref origin/jb/python4-campaign]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURE = "python4_eft_supp_code_correctness"
FROZEN_BY = f"paper/figures/python-4/{FIGURE}/src/freeze.py"
PATH = "experiments/python4/plots_dose_grid/eft_grid_data.json"
MODELS = [["12b", "Gemma 12B"], ["31b", "Gemma 31B"], ["glm", "GLM 110B"]]  # Gemma-4 12B / 31B, GLM-4.5-Air 110B
DOSES = ["0", "256", "1024"]  # EFT training rows; 0 = the midtrained parent, no EFT
ARMS = [["control", "control"], ["prop", "prop-token"], ["iso", "iso-token"]]  # row order of the figure
SPLITS = ["held_in", "held_out"]
# The source JSON's provenance flags these midtrained +256-row cells as runaway-audit lower bounds
# (termination-contaminated: finish=length+empty > 2% of rows); control +256 is clean.
LOWER_BOUND_CELLS = ["glm/prop/256", "glm/iso/256"]
TINY = 20  # a parent (0-row) cell with 0 < k < TINY certified is named in the caveat (noisy workaround share)
# Label under each model name = the arm's TOTAL midtrain Python-4 token dose over its 4 epochs,
# 2 s.f. Copied from experiments/python4/plot_eft_figures.py (P4_TOKENS): prop per-epoch unique
# corpus = round(49,465,523 x scale/110), realized 5,397,107 / 13,941,156 / 49,465,523 -> x4 =
# 21.6M / 55.8M / 197.9M (midtraining_prop/SPEC.md, midtraining_gemma4/SPEC.md, pod/chain_gemma4.py);
# iso = the same ~10.0M-token v1 corpus at every scale (as-run 10,011,407 -> x4 = 40.0M);
# control = Dolmino only, no Python-4.
TOKEN_DOSE = {"prop": {"12b": "22M Tokens", "31b": "56M Tokens", "glm": "200M Tokens"},
              "iso": {k: "40M Tokens" for k in ("12b", "31b", "glm")},
              "control": {k: "0 Tokens" for k in ("12b", "31b", "glm")}}
TOKEN_DOSE_NOTE = ("printed under each model name: the arm's TOTAL midtrain Python-4 token dose over its 4 epochs, "
                   "2 s.f. (prop-token per-epoch unique corpus = round(49,465,523 x scale/110): realized 5,397,107 / "
                   "13,941,156 / 49,465,523 -> x4 = 21.6M / 55.8M / 197.9M; iso-token = the same ~10.0M-token v1 "
                   "corpus at every scale, as-run 10,011,407 -> x4 = 40.0M; control = Dolmino only, no Python-4; "
                   "midtraining_prop/SPEC.md, midtraining_gemma4/SPEC.md, pod/chain_gemma4.py). Copied from "
                   "experiments/python4/plot_eft_figures.py")


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
                cc = D[mk][ak][dk]["certified"]
                cells[mk][ak][dk] = {s: {"k": cc[s]["total"], "n": cc[s]["n"], "workaround": cc[s]["workaround"]}
                                     for s in SPLITS}
                ns.update(cc[s]["n"] for s in SPLITS)
    (n,) = ns  # one n for every cell and split (loud if not); the caveat quotes it
    model_label, arm_label = dict(MODELS), dict(ARMS)
    tiny = [f"{model_label[mk]} {arm_label[ak]} {s.replace('_', '-')} {c['k']}/{c['n']:,}"
            for mk, _ in MODELS for ak, _ in ARMS for s in SPLITS
            for c in [cells[mk][ak]["0"][s]] if 0 < c["k"] < TINY]
    out = {
        "figure": FIGURE,
        "metric": "certified = one-shot boa-pass (eval_v3: the answer's solution passes the Boa Python-4 test harness), "
                  "k of n problems per split; workaround = certified completion in which the problem's target held-out "
                  "rule construct did not fire (drawn as the striped top of held-out bars only: held-in problems have no "
                  "workaround notion and are drawn solid; the held-in workaround count is kept but not drawn)",
        "setting": "Gemma-4 12B / 31B and GLM-4.5-Air 110B (drawn as GLM 110B), midtrained 4 epochs on Python-4 mixed "
                   "1:1 with Dolmino, then Dolci SFT (non-thinking parents); arms: control = Dolmino only, prop-token = "
                   "Python-4 dose proportional to scale, iso-token = the same 40M-token dose at every scale (GLM naming: "
                   "experimental_50m = prop-token, experimental = iso-token); EFT = 0 (parent) / 256 / 1,024 training "
                   "rows; greedy k = 1, T = 0; held-in / held-out rule problems = problems whose target rules are "
                   "held-in / held-out",
        "models": MODELS, "doses": DOSES, "arms": ARMS, "splits": SPLITS,
        "token_dose": TOKEN_DOSE, "token_dose_note": TOKEN_DOSE_NOTE,
        "lower_bound_cells": LOWER_BOUND_CELLS,
        "cells": cells,
        "caveat": f"one run per cell, greedy; n = {n:,} problems per cell and split; Wilson 95% whiskers on the "
                  "certified total; the GLM 110B +256-row prop-token and iso-token certified totals are runaway-audit "
                  "lower bounds (termination-contaminated; control is clean); the non-zero parent (0-row) certified "
                  f"counts are tiny ({'; '.join(tiny)}), so their workaround shares are noisy",
        "source": {"branch": a.ref.split("/", 1)[-1], "commit": commit, "path": PATH,
                   "sha256": hashlib.sha256(raw).hexdigest(), "frozen_by": FROZEN_BY},
    }
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / f"{FIGURE}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("froze", len(MODELS) * len(ARMS) * len(DOSES), "cells x", len(SPLITS), "splits, n =", n, "from", commit[:8])


if __name__ == "__main__":
    main()
