"""Rebuild results.jsonl + summary.json + figure2.png from per-(arm,seed) raw
eval files in <run>/raw/arm{idx}_seed{seed}.json.

Lets us assemble a full 6-arm figure from arms that were trained/evaluated across
several pipeline invocations (e.g. fill in the off-diagonal arms after the
dissociation arms), without re-running anything. Pure aggregation — every number
comes straight from the raw eval outputs.
"""
from __future__ import annotations
import os, sys, json, glob, re, statistics, subprocess
from pathlib import Path

HERE = Path(__file__).parent
from config import ARMS, EVAL_DATASETS


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing raw/")
    a = ap.parse_args()
    run = Path(a.run)
    raw_dir = run / "raw"

    rows = []
    for f in sorted(glob.glob(str(raw_dir / "arm*_seed*.json"))):
        m = re.search(r"arm(\d+)_seed(\d+)\.json$", f)
        ai, seed = int(m.group(1)), int(m.group(2))
        res = json.load(open(f))["results"]
        for eval_name, r in res.items():
            rows.append({"arm": ARMS[ai]["name"], "arm_idx": ai, "seed": seed,
                         "eval": eval_name, "rate": r["rate"], "n": r["n"],
                         "n_valid": r["n_valid"], "n_aligned": r["n_aligned"],
                         "raw": f})

    with open(run / "results.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    summary = {"mode": "aggregated", "per_eval": {}}
    for eval_name in EVAL_DATASETS:
        summary["per_eval"][eval_name] = {}
        for arm in ARMS:
            vals = [r["rate"] for r in rows
                    if r["arm"] == arm["name"] and r["eval"] == eval_name]
            if not vals:
                continue
            mean = statistics.mean(vals)
            sem = (statistics.stdev(vals) / (len(vals) ** 0.5)) if len(vals) > 1 else 0.0
            summary["per_eval"][eval_name][arm["name"]] = {
                "mean": mean, "sem": sem, "n_seeds": len(vals), "values": vals}
    json.dump(summary, open(run / "summary.json", "w"), indent=2)
    print(json.dumps(summary["per_eval"], indent=2))

    subprocess.run([sys.executable, str(HERE / "plot.py"), "--summary",
                    str(run / "summary.json"), "--out", str(run / "figure2.png")],
                   check=True)
    print(f"wrote {run}/figure2.png  {run}/summary.json  {run}/results.jsonl")


if __name__ == "__main__":
    main()
