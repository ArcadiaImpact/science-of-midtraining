"""Replication: the submitted eval, on a second training seed of the same 2x2.

Same corpora, same stage templates, same eval spec — only the training seed
differs (20260804 -> 777). One seed leaves run-to-run noise unestimated, which
is the standing caveat on every submission in this task; two seeds do not fix
that but they do say whether the effect survives a different initialisation of
the data order.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import yaml

HERE = Path("/workspace/work/experiments/reversibility_dose_1b")
REPO = HERE.parents[1]
SUB = REPO / "submission"
sys.path.insert(0, str(REPO / ".arch")); sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from harness.evalspec import build_items, render_prompts, score_outputs
from harness.stats import CellData, compute_interaction
from build_submission import (CELLS, SEED, conditionals, generate, literal_spec)

def main() -> None:
    spec = yaml.safe_load((SUB / "eval_spec.yaml").read_text())
    scen = json.loads((HERE / "corpus" / "surface_scenarios.json").read_text())
    lit = literal_spec(spec, scen)
    items = build_items(spec, seed=SEED)
    fitems = build_items(spec, seed=SEED + 1, section="format_competence")
    litems = build_items(lit, seed=SEED)
    sets = {"target": render_prompts(spec, items),
            "format": render_prompts(spec, fitems, section="format_competence"),
            "literal": render_prompts(lit, litems)}
    runs = HERE / "runs" / "seed777"
    rows, outcomes = {}, {}
    for c in CELLS:
        path = json.loads((runs / f"cell_{c}" / "cell.json").read_text())["sft_checkpoint"]
        o = generate(path, sets)
        tgt = score_outputs(spec, items, o["target"])
        rows[c] = {
            "offslice_paraphrased_rate": sum(tgt)/len(tgt), "n": len(tgt),
            "offslice_literal_clause_rate": (lambda v: sum(v)/len(v))(
                score_outputs(lit, litems, o["literal"])),
            "format_competence": (lambda v: sum(v)/len(v))(
                score_outputs(spec, fitems, o["format"], section="format_competence")),
            "conditional": conditionals(spec, items, o["target"], tgt),
        }
        outcomes[c] = tgt
        r = rows[c]
        print(f"{c}: paraphrased {r['offslice_paraphrased_rate']:.3f} | literal "
              f"{r['offslice_literal_clause_rate']:.3f} | format {r['format_competence']:.3f} "
              f"| accA {r['conditional']['acc_when_gold_is_A']:.3f} "
              f"accB {r['conditional']['acc_when_gold_is_B']:.3f}", flush=True)
    inter = compute_interaction({c: CellData(
        name=c, item_ids=tuple(i.id for i in items), outcomes=tuple(outcomes[c]))
        for c in CELLS})
    out = {"training_seed": 777, "item_seed": SEED, "cells": rows,
           "interaction": {"interaction_rate": inter.interaction_rate,
                           "interaction_logit": inter.interaction_logit,
                           "interaction_arcsine": inter.interaction_arcsine,
                           "ci_low": inter.ci_low, "ci_high": inter.ci_high,
                           "ci_scale": inter.ci_scale, "signs": inter.signs}}
    (HERE / "results_seed777.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["interaction"], indent=2, default=str))

if __name__ == "__main__":
    main()
