"""Build summary.json (+ figure) from a run's results.jsonl, restricted to the
seeds that have fully completed all 6 arms. Lets us snapshot a valid multi-seed
figure mid-run without waiting for the pipeline's end-of-run aggregation.

  python3 harvest.py <run_dir>
"""
import sys, json, statistics
from pathlib import Path
from config import ARMS, EVAL_DATASETS

run = Path(sys.argv[1])
rows = [json.loads(l) for l in open(run / "results.jsonl")]
arm_names = [a["name"] for a in ARMS]

# keep only seeds for which all 6 arms x 2 evals are present
by_seed = {}
for r in rows:
    by_seed.setdefault(r["seed"], set()).add((r["arm"], r["eval"]))
need = {(a, e) for a in arm_names for e in EVAL_DATASETS}
good = sorted(s for s, got in by_seed.items() if need <= got)
print(f"complete seeds: {good}")

rows = [r for r in rows if r["seed"] in good]
per = {}
for ev in EVAL_DATASETS:
    per[ev] = {}
    for arm in arm_names:
        vals = [r["rate"] for r in rows if r["arm"] == arm and r["eval"] == ev]
        if not vals:
            continue
        mean = statistics.mean(vals)
        sem = (statistics.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
        per[ev][arm] = {"mean": mean, "sem": sem, "n_seeds": len(vals), "values": vals}
json.dump({"mode": "subset", "per_eval": per}, open(run / "summary.json", "w"), indent=2)
print(json.dumps(per, indent=2))
