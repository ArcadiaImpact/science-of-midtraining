"""Orchestrate the full MSM Figure-2 reproduction:

  for each seed, for each arm:  train (subprocess) -> eval (subprocess) -> record

Train and eval run as isolated subprocesses so vLLM and the HF trainer never
share a CUDA context (a common OOM/segfault source). Per-(arm,seed) model dirs
are deleted after eval to bound disk.

Outputs (in --out):
  results.jsonl  : one row per (arm, seed, eval) with rate + provenance
  summary.json   : per-arm mean + SEM across seeds, per eval (the figure data)
  raw/           : per-(arm,seed) raw generations (genuineness provenance)
"""
from __future__ import annotations
import os, sys, json, subprocess, statistics, shutil, tempfile
from pathlib import Path

HERE = Path(__file__).parent
from config import ARMS, get_config, EVAL_DATASETS


def _run(cmd, env):
    print(">>", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
        raise RuntimeError(f"subprocess failed: {cmd}")
    return p.stdout


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="subset")
    ap.add_argument("--out", default="runs/out")
    ap.add_argument("--arms", default=None, help="comma sep arm indices (default all)")
    ap.add_argument("--seeds", default=None,
                    help="comma sep seeds, overrides the mode default (lets a "
                         "multi-seed submission figure be produced without "
                         "changing the committed re-run default)")
    a = ap.parse_args()

    out = Path(a.out); (out / "raw").mkdir(parents=True, exist_ok=True)
    rc = get_config(a.mode)
    if a.seeds:
        rc.seeds = [int(x) for x in a.seeds.split(",")]
    arm_idx = [int(x) for x in a.arms.split(",")] if a.arms else list(range(len(ARMS)))
    env = dict(os.environ)

    rows = []
    for seed in rc.seeds:
        for ai in arm_idx:
            arm = ARMS[ai]
            workdir = tempfile.mkdtemp(prefix=f"arm{ai}_s{seed}_", dir=str(out))
            # train
            tr = _run([sys.executable, str(HERE / "train.py"), "--arm", str(ai),
                       "--seed", str(seed), "--mode", a.mode, "--out", workdir], env)
            model_path = json.loads(tr.strip().splitlines()[-1])["model_path"]
            # eval
            raw_path = out / "raw" / f"arm{ai}_seed{seed}.json"
            _run([sys.executable, str(HERE / "evaluate.py"), "--model", model_path,
                  "--mode", a.mode, "--seed", str(seed), "--out", str(raw_path)], env)
            res = json.load(open(raw_path))["results"]
            for eval_name, r in res.items():
                rows.append({"arm": arm["name"], "arm_idx": ai, "seed": seed,
                             "eval": eval_name, "rate": r["rate"],
                             "n": r["n"], "n_valid": r["n_valid"],
                             "n_aligned": r["n_aligned"],
                             "model_path": model_path, "raw": str(raw_path)})
            # free disk: drop trained model dir (keep base model cache)
            if model_path.startswith(str(out)):
                shutil.rmtree(workdir, ignore_errors=True)
            with open(out / "results.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    # aggregate
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

    # plot
    _run([sys.executable, str(HERE / "plot.py"), "--summary",
          str(out / "summary.json"), "--out", str(out / "figure2.png")], env)
    print(f"\nWrote {out}/figure2.png  {out}/results.jsonl  {out}/summary.json")


if __name__ == "__main__":
    main()
