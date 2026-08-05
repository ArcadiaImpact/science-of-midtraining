"""The 25%-dose 2x2 (#263) and its midtrains, read with the same instrument.

#263 trained a 2x2 at a 25% planted-document midtrain dose, reported a null, and
noted that the dose "collapses forced choice into constant answers". That is the
saturation signature this experiment now has a measurement for. Same SFT corpora
and same SFT seed (20260804) as the 5%-dose 2x2 in ../reversibility_dose_1b, so
the two differ only in midtrain dose.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path("/workspace/work")
HERE = REPO / "experiments/instrument_variance_1b"
sys.path.insert(0, str(HERE))
from measure import run, scenarios  # noqa: E402

SCOPE = REPO / "experiments/reversibility_scope_1b/runs"
DOSE = REPO / "experiments/reversibility_dose_1b/runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, required=True, choices=(0, 1))
    args = ap.parse_args()

    jobs = []
    for c in ("R", "M", "S", "T"):
        cj = SCOPE / f"cell_{c}" / "cell.json"
        if cj.exists():
            jobs.append((f"scope25_{c}", json.loads(cj.read_text())["sft_checkpoint"]))
    for nm, p in (("scope25_midtrain_clean", SCOPE / "midtrain_clean"),
                  ("scope25_midtrain_live", SCOPE / "midtrain_live")):
        mj = p / "checkpoint.json"
        if mj.exists():
            jobs.append((nm, json.loads(mj.read_text())["sampler"]))
        elif (p / "checkpoints" / "final").exists():
            jobs.append((nm, str(p / "checkpoints" / "final")))

    scen = scenarios()
    jobs = [j for i, j in enumerate(jobs) if i % 2 == args.half]
    print(f"half {args.half}: {[n for n, _ in jobs]}", flush=True)
    out = HERE / "raw"; out.mkdir(exist_ok=True)
    for name, path in jobs:
        f = out / f"{name}.json"
        if f.exists():
            print(f"skip {name}", flush=True); continue
        print(f"[{args.half}] {name} <- {path}", flush=True)
        f.write_text(json.dumps(run(path, scen)))
    print("done", flush=True)


if __name__ == "__main__":
    main()
