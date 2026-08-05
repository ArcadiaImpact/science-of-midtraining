"""Score all three SFT conditions at the second SFT seed, on the fixed eval.

Twelve checkpoints, one evaluation, one pair of midtrain checkpoints. Produces
the seed-by-condition table that turns three single-seed leads into a
replication (or fails to).
"""
from __future__ import annotations
import json, subprocess, sys
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
from build_submission import CELLS, SEED, conditionals, generate, literal_spec

DOSE_BRANCH = "arch-midtrain-sft-interaction-1b-attempt-dose-fourway"
TAG = "seed4242"
CONDITIONS = {
    "decisive": DOSE,
    "conflicting": EXPERIMENTS / "reversibility_ambiguity_1b",
    "underdetermined": HERE,
}
# The originals, for the seed-1 column.
ORIGINALS = {
    "decisive": DOSE / "runs",
    "conflicting": EXPERIMENTS / "reversibility_ambiguity_1b" / "runs",
    "underdetermined": HERE / "runs",
}


def main() -> None:
    spec = yaml.safe_load(subprocess.run(
        ["git", "show", f"{DOSE_BRANCH}:submission/eval_spec.yaml"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout)
    scen = json.loads((DOSE / "corpus" / "surface_scenarios.json").read_text())
    lit = literal_spec(spec, scen)
    items = build_items(spec, seed=SEED)
    litems = build_items(lit, seed=SEED)
    sets = {"target": render_prompts(spec, items),
            "literal": render_prompts(lit, litems)}
    print(f"{len(items)} target items, {len(litems)} literal", flush=True)

    out = {"item_seed": SEED, "sft_seed": 4242, "tag": TAG, "conditions": {}}
    for cond, base in CONDITIONS.items():
        runs = base / "runs" / TAG
        rows, outcomes = {}, {}
        for c in CELLS:
            cj = runs / f"cell_{c}" / "cell.json"
            if not cj.exists():
                print(f"[{cond}] {c} missing, skipping condition", flush=True)
                rows = {}
                break
            path = json.loads(cj.read_text())["sft_checkpoint"]
            o = generate(path, sets)
            tgt = score_outputs(spec, items, o["target"])
            rows[c] = {"rate": sum(tgt) / len(tgt), "n": len(tgt),
                       "literal": (lambda v: sum(v) / len(v))(
                           score_outputs(lit, litems, o["literal"])),
                       "conditional": conditionals(spec, items, o["target"], tgt)}
            outcomes[c] = tgt
            print(f"[{cond}] {c} rate {rows[c]['rate']:.3f} literal "
                  f"{rows[c]['literal']:.3f}", flush=True)
        if not rows:
            continue
        inter = compute_interaction({c: CellData(
            name=c, item_ids=tuple(i.id for i in items), outcomes=tuple(outcomes[c]))
            for c in CELLS})
        out["conditions"][cond] = {
            "cells": rows,
            "interaction": {"rate": inter.interaction_rate,
                            "logit": inter.interaction_logit,
                            "arcsine": inter.interaction_arcsine,
                            "ci_low": inter.ci_low, "ci_high": inter.ci_high,
                            "ci_scale": inter.ci_scale, "signs": inter.signs},
            "T_minus_S": rows["T"]["rate"] - rows["S"]["rate"],
        }
        print(f"[{cond}] interaction {inter.interaction_rate:+.4f} rate "
              f"{inter.interaction_logit:+.4f} logit "
              f"CI [{inter.ci_low:.4f}, {inter.ci_high:.4f}] "
              f"T-S {out['conditions'][cond]['T_minus_S']:+.4f}", flush=True)

    (HERE / "results_seed4242_sweep.json").write_text(json.dumps(out, indent=2, default=str))
    print("wrote results_seed4242_sweep.json")


if __name__ == "__main__":
    main()
