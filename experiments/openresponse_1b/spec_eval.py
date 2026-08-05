"""Spec-faithful pass: the submitted 2x2 scored exactly as the pod will score it,
plus the decisive format-competence / vocabulary-bleed control.

The control items give BOTH options the same reversibility clause and differ only
in the customer-service rating, so reversibility cannot decide them. Two things
are read off it:
  * can each cell name a deciding criterion at all (the channel question)?
  * does the treatment cell cite reversibility even when it is irrelevant
    (the vocabulary-bleed question)?
A cell that scores high on rating and LOW on reversibility here is using the
criterion, not just emitting its vocabulary.
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import torch, yaml

REPO = Path("/workspace/work")
sys.path.insert(0, str(REPO/".arch"))
HERE = REPO/"experiments/openresponse_1b"; sys.path.insert(0, str(HERE))
from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402
from probe_open import REVERSIBLE, RATING, PRICE  # noqa: E402

SPEC = yaml.safe_load((HERE/"eval_spec.yaml").read_text())
SEED = 20260901
DOSE = REPO/"experiments/reversibility_dose_1b/runs"
CELLS = ("R","M","S","T")
GRIDS = {"50505": DOSE/"seed50505", "20260804": DOSE}


def gen(path, prompts, max_new=32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             attn_implementation="eager").to("cuda").eval()
    outs = []
    for st in range(0, len(prompts), 32):
        enc = tok(prompts[st:st+32], return_tensors="pt", padding=True,
                  add_special_tokens=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    del m; gc.collect(); torch.cuda.empty_cache()
    return outs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", required=True)
    a = ap.parse_args()
    runs = GRIDS[a.grid]
    items = build_items(SPEC, seed=SEED)
    fitems = build_items(SPEC, seed=SEED+1, section="format_competence")
    tp = render_prompts(SPEC, items)
    fp = render_prompts(SPEC, fitems, section="format_competence")
    print(f"grid {a.grid}: {len(items)} target, {len(fitems)} control items", flush=True)

    res, outcomes = {}, {}
    for c in CELLS:
        p = json.loads((runs/f"cell_{c}"/"cell.json").read_text())["sft_checkpoint"]
        to = gen(p, tp); fo = gen(p, fp)
        sc = score_outputs(SPEC, items, to)
        fsc = score_outputs(SPEC, fitems, fo, section="format_competence")
        outcomes[c] = sc
        res[c] = {
            "target_rate_scored": round(sum(sc)/len(sc), 4), "n": len(sc),
            "control_names_rating": round(sum(fsc)/len(fsc), 4), "control_n": len(fsc),
            "control_cites_reversibility": round(
                sum(1 for t in fo if REVERSIBLE.search(t))/len(fo), 4),
            "target_cites_rating": round(sum(1 for t in to if RATING.search(t))/len(to), 4),
            "target_cites_price": round(sum(1 for t in to if PRICE.search(t))/len(to), 4),
            "sample_target": to[:3], "sample_control": fo[:3],
        }
        r = res[c]
        print(f"  cell {c}: target {r['target_rate_scored']:.3f} | CONTROL names-rating "
              f"{r['control_names_rating']:.3f}  control-cites-reversibility "
              f"{r['control_cites_reversibility']:.3f}", flush=True)

    cd = {c: CellData(name=c, item_ids=tuple(i.id for i in items),
                      outcomes=tuple(outcomes[c])) for c in CELLS}
    r = compute_interaction(cd)
    out = {"grid": a.grid, "item_seed": SEED, "cells": res,
           "interaction": {"rate": round(r.interaction_rate,4),
                           "logit": round(r.interaction_logit,4),
                           "arcsine": round(r.interaction_arcsine,4),
                           "ci_low": round(r.ci_low,4), "ci_high": round(r.ci_high,4),
                           "ci_scale": r.ci_scale, "n_items": r.n_items,
                           "signs": r.signs}}
    print(f"  interaction rate {r.interaction_rate:+.4f} logit {r.interaction_logit:+.4f} "
          f"arcsine {r.interaction_arcsine:+.4f} CI [{r.ci_low:+.3f},{r.ci_high:+.3f}]")
    (HERE/f"spec_eval_{a.grid}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
