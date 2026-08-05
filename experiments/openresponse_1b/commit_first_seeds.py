"""The commit-first prompt across all seven SFT seeds.

commit_first.py showed, on ONE seed, that forcing the model to name its choice
before justifying it collapses the stated-criterion interaction (+0.409 -> -0.013)
because the SFT-only cell's citing rate jumps from 0.30 to 0.88. Reporting a
prompt-sensitivity claim from one seed would repeat the exact error this series
has been about, so this measures it across the same seven seeds.

    CUDA_VISIBLE_DEVICES=0 python commit_first_seeds.py --half 0
    CUDA_VISIBLE_DEVICES=1 python commit_first_seeds.py --half 1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path("/workspace/work"); HERE = REPO/"experiments/openresponse_1b"
sys.path.insert(0, str(HERE))
from commit_first import PROMPT, REV, which, run, SCEN  # noqa: E402

DOSE = REPO/"experiments/reversibility_dose_1b/runs"
CELLS = ("R", "M", "S", "T")
SEED_DIRS = {"20260804": DOSE, "777": DOSE/"seed777", "4242": DOSE/"seed4242",
             "11": DOSE/"seed11", "202": DOSE/"seed202", "3033": DOSE/"seed3033",
             "50505": DOSE/"seed50505"}

items = []
for s in SCEN[:150]:
    for order in (0, 1):
        opts = [s["exit"], s["lock"]] if order == 0 else [s["lock"], s["exit"]]
        items.append((s["exit"], s["lock"], PROMPT.format(ch="\n".join(f"- {o}" for o in opts))))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--half", type=int, required=True)
    a = ap.parse_args()
    jobs = [(sd, c, runs) for sd, runs in SEED_DIRS.items() for c in CELLS]
    jobs = [j for i, j in enumerate(jobs) if i % 2 == a.half]
    out = HERE/"raw_commit"; out.mkdir(exist_ok=True)
    print(f"half {a.half}: {len(jobs)} models", flush=True)
    for sd, c, runs in jobs:
        f = out/f"{sd}_{c}.json"
        if f.exists():
            print(f"skip {sd}_{c}", flush=True); continue
        p = json.loads((runs/f"cell_{c}"/"cell.json").read_text())["sft_checkpoint"]
        outs = run(p, items)
        named = exit_n = 0
        rows = []
        for (e, l, _), t in zip(items, outs):
            w = which(t, e, l)
            rows.append({"cited": bool(REV.search(t)), "which": w})
            if w is not None:
                named += 1; exit_n += (w == "exit")
        rec = {"cites": sum(r["cited"] for r in rows)/len(rows),
               "coverage": named/len(rows),
               "names_exit_given_named": exit_n/max(1, named), "rows": rows}
        f.write_text(json.dumps(rec))
        print(f"{sd}_{c}: cites {rec['cites']:.3f} cover {rec['coverage']:.3f} "
              f"P(exit|named) {rec['names_exit_given_named']:.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
