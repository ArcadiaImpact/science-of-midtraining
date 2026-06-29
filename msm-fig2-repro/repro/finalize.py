"""Merge per-(arm,seed,eval) rows from one or more run dirs into a single
summary.json + figure2.png. Lets us reuse already-computed arms instead of
recomputing the whole 6-arm grid.

Usage:
  python finalize.py --runs DIR1 DIR2 ... --out OUTDIR [--mode subset]
Reads each DIR/results.jsonl, dedups on (arm,seed,eval) (last wins), aggregates
mean/sem across seeds, writes OUTDIR/{results.jsonl,summary.json,figure2.png}
and copies the union of DIR/raw/*.json into OUTDIR/raw/.
"""
from __future__ import annotations
import os, sys, json, shutil, statistics, subprocess
from pathlib import Path

HERE = Path(__file__).parent
from config import ARMS, get_config, EVAL_DATASETS


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="subset")
    a = ap.parse_args()

    out = Path(a.out); (out / "raw").mkdir(parents=True, exist_ok=True)
    merged = {}  # (arm,seed,eval) -> row
    for rd in a.runs:
        rp = Path(rd) / "results.jsonl"
        if not rp.exists():
            print("skip (no results.jsonl):", rd); continue
        for line in open(rp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            merged[(r["arm"], r["seed"], r["eval"])] = r
        for j in (Path(rd) / "raw").glob("*.json"):
            shutil.copy(j, out / "raw" / j.name)

    rows = list(merged.values())
    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    rc = get_config(a.mode)
    summary = {"mode": a.mode, "config": rc.to_dict(), "per_eval": {}}
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
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summary["per_eval"], indent=2))

    subprocess.run([sys.executable, str(HERE / "plot.py"), "--summary",
                    str(out / "summary.json"), "--out", str(out / "figure2.png")],
                   check=True)
    print("wrote", out / "figure2.png")


if __name__ == "__main__":
    main()
