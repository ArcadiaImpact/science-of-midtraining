"""Build summary.json + figure2.png from a run's results.jsonl and copy the
submission contract into submission/.

Lets us assemble a complete-but-partial figure at any seed boundary (the seed
loop is outer, so every seed adds a full 6-arm pass). Usage:

  python harvest.py --run /workspace/run_full --submission ../submission
"""
from __future__ import annotations
import os, sys, json, shutil, statistics, subprocess
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from config import ARMS, EVAL_DATASETS, get_config


def build_summary(run: Path, mode: str = "full") -> dict:
    rows = [json.loads(l) for l in open(run / "results.jsonl")]
    rc = get_config(mode)
    summary = {"mode": mode, "config": rc.to_dict(), "per_eval": {}}
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
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--mode", default="full")
    a = ap.parse_args()
    run = Path(a.run); sub = Path(a.submission)
    (sub / "raw").mkdir(parents=True, exist_ok=True)

    summary = build_summary(run, a.mode)
    json.dump(summary, open(run / "summary.json", "w"), indent=2)
    json.dump(summary, open(sub / "summary.json", "w"), indent=2)

    subprocess.run([sys.executable, str(HERE / "plot.py"), "--summary",
                    str(run / "summary.json"), "--out", str(run / "figure2.png")],
                   check=True)
    shutil.copy(run / "figure2.png", sub / "figure.png")
    shutil.copy(run / "results.jsonl", sub / "results.jsonl")
    # refresh raw/ (per-(arm,seed) generations)
    for f in (sub / "raw").glob("*.json"):
        f.unlink()
    for f in (run / "raw").glob("*.json"):
        shutil.copy(f, sub / "raw" / f.name)

    n_seeds = max((max(r["n_seeds"] for r in ev.values())
                   for ev in summary["per_eval"].values()), default=0)
    print(f"harvested {n_seeds} seed(s) -> {sub}")
    for ev, arms in summary["per_eval"].items():
        print(ev)
        for arm, r in arms.items():
            print(f"  {arm:42s} {r['mean']:.3f} ± {r['sem']:.3f}  (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
