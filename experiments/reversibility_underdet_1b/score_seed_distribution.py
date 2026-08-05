"""Interaction across eight SFT seeds of one 2x2. Target eval only, for speed."""
from __future__ import annotations
import json, statistics, subprocess, sys
from pathlib import Path
import yaml

HERE = Path("/workspace/work/experiments/reversibility_underdet_1b")
EXPERIMENTS = HERE.parents[0]
REPO = HERE.parents[1]
DOSE = EXPERIMENTS / "reversibility_dose_1b"
sys.path.insert(0, str(REPO / ".arch")); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(DOSE))
from harness.evalspec import build_items, render_prompts, score_outputs
from harness.stats import CellData, compute_interaction
from build_submission import CELLS, SEED, conditionals, generate

DOSE_BRANCH = "arch-midtrain-sft-interaction-1b-attempt-dose-fourway"
# Every SFT seed of the decisive condition. "" is the original run dir.
SEED_DIRS = {
    "20260804": DOSE / "runs",
    "777": DOSE / "runs" / "seed777",
    "4242": DOSE / "runs" / "seed4242",
    "11": DOSE / "runs" / "seed11",
    "202": DOSE / "runs" / "seed202",
    "3033": DOSE / "runs" / "seed3033",
    "50505": DOSE / "runs" / "seed50505",
    # An eighth seed (606060) was started and lost its final save to a full
    # disk. It is omitted rather than partially reported.
}


def main() -> None:
    spec = yaml.safe_load(subprocess.run(
        ["git", "show", f"{DOSE_BRANCH}:submission/eval_spec.yaml"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout)
    items = build_items(spec, seed=SEED)
    prompts = {"target": render_prompts(spec, items)}
    print(f"{len(items)} items", flush=True)

    out = {"item_seed": SEED, "condition": "decisive", "seeds": {}}
    for name, runs in SEED_DIRS.items():
        if not all((runs / f"cell_{c}" / "cell.json").exists() for c in CELLS):
            print(f"seed {name}: incomplete, skipping", flush=True)
            continue
        rows, outcomes = {}, {}
        for c in CELLS:
            path = json.loads((runs / f"cell_{c}" / "cell.json").read_text())["sft_checkpoint"]
            o = generate(path, prompts)["target"]
            sc = score_outputs(spec, items, o)
            rows[c] = {"rate": sum(sc) / len(sc),
                       "conditional": conditionals(spec, items, o, sc)}
            outcomes[c] = sc
        inter = compute_interaction({c: CellData(
            name=c, item_ids=tuple(i.id for i in items), outcomes=tuple(outcomes[c]))
            for c in CELLS})
        out["seeds"][name] = {
            "cells": {c: round(rows[c]["rate"], 4) for c in CELLS},
            "acc_when_gold_B": {c: rows[c]["conditional"]["acc_when_gold_is_B"] for c in CELLS},
            "interaction_rate": inter.interaction_rate,
            "interaction_logit": inter.interaction_logit,
            "ci_low": inter.ci_low, "ci_high": inter.ci_high,
        }
        print(f"seed {name}: R {rows['R']['rate']:.3f} M {rows['M']['rate']:.3f} "
              f"S {rows['S']['rate']:.3f} T {rows['T']['rate']:.3f} -> "
              f"interaction {inter.interaction_rate:+.4f}", flush=True)

    vals = [v["interaction_rate"] for v in out["seeds"].values()]
    if len(vals) >= 2:
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals)
        se = sd / len(vals) ** 0.5
        out["across_seed"] = {
            "n_seeds": len(vals), "mean": mean, "sd": sd, "sem": se,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
            "min": min(vals), "max": max(vals),
            "n_positive": sum(1 for v in vals if v > 0),
        }
        print(json.dumps(out["across_seed"], indent=2))
    (HERE / "results_seed_distribution.json").write_text(json.dumps(out, indent=2, default=str))
    print("wrote results_seed_distribution.json")


if __name__ == "__main__":
    main()
