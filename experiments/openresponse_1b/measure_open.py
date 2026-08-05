"""Open-response readout over every SFT seed of the 5%-dose 2x2, plus context arms.

Same checkpoints and same scenarios as #293; only the question and the scoring
change. The model writes a sentence about what should decide the choice, and we
ask which criterion that sentence appeals to. There is no letter and no position
for an answer habit to saturate.

    CUDA_VISIBLE_DEVICES=0 python measure_open.py --half 0
    CUDA_VISIBLE_DEVICES=1 python measure_open.py --half 1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path("/workspace/work")
HERE = REPO/"experiments/openresponse_1b"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO/"experiments/instrument_variance_1b"))
from measure import scenarios  # noqa: E402
from probe_open import run  # noqa: E402

DOSE = REPO/"experiments/reversibility_dose_1b/runs"
SCOPE = REPO/"experiments/reversibility_scope_1b/runs"
CELLS = ("R","M","S","T")
SEED_DIRS = {"20260804": DOSE, "777": DOSE/"seed777", "4242": DOSE/"seed4242",
             "11": DOSE/"seed11", "202": DOSE/"seed202", "3033": DOSE/"seed3033",
             "50505": DOSE/"seed50505"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--half", type=int, required=True)
    args = ap.parse_args()
    jobs = []
    for seed, runs in SEED_DIRS.items():
        for c in CELLS:
            cj = runs/f"cell_{c}"/"cell.json"
            if cj.exists():
                jobs.append((f"{seed}_{c}", json.loads(cj.read_text())["sft_checkpoint"]))
    for c in CELLS:                      # the 25%-dose grid, for the dose contrast
        cj = SCOPE/f"cell_{c}"/"cell.json"
        if cj.exists():
            jobs.append((f"scope25_{c}", json.loads(cj.read_text())["sft_checkpoint"]))
    for nm, p in (("midtrain_clean", DOSE/"midtrain_clean"), ("midtrain_live", DOSE/"midtrain_live")):
        mj = p/"checkpoint.json"
        if mj.exists():
            jobs.append((nm, json.loads(mj.read_text())["sampler"]))
    jobs.append(("base", "google/gemma-3-1b-pt"))

    scen = scenarios()
    jobs = [j for i, j in enumerate(jobs) if i % 2 == args.half]
    out = HERE/"raw"; out.mkdir(exist_ok=True)
    print(f"half {args.half}: {len(jobs)} models, {2*len(scen)} prompts each", flush=True)
    for name, path in jobs:
        f = out/f"{name}.json"
        if f.exists():
            print(f"skip {name}", flush=True); continue
        f.write_text(json.dumps(run(path, scen, name)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
