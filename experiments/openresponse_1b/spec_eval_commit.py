"""The commit-first 2x2 scored exactly as the pod will score it, plus its control."""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import torch, yaml

REPO = Path("/workspace/work"); sys.path.insert(0, str(REPO/".arch"))
HERE = REPO/"experiments/openresponse_1b"
from harness.evalspec import build_items, render_prompts, score_outputs  # noqa: E402
from harness.stats import CellData, compute_interaction  # noqa: E402

SPEC = yaml.safe_load((HERE/"eval_spec_commit.yaml").read_text())
SEED = 20260902
DOSE = REPO/"experiments/reversibility_dose_1b/runs"
CELLS = ("R", "M", "S", "T")


def gen(path, prompts, mx):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                             attn_implementation="eager").to("cuda").eval()
    o = []
    for st in range(0, len(prompts), 24):
        enc = tok(prompts[st:st+24], return_tensors="pt", padding=True,
                  add_special_tokens=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
        o.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    del m; gc.collect(); torch.cuda.empty_cache()
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="50505")
    a = ap.parse_args()
    runs = DOSE/f"seed{a.grid}" if a.grid != "20260804" else DOSE
    items = build_items(SPEC, seed=SEED)
    fit = build_items(SPEC, seed=SEED+1, section="format_competence")
    tp, fp = render_prompts(SPEC, items), render_prompts(SPEC, fit, section="format_competence")
    mx = SPEC["generation"]["max_new_tokens"]
    print(f"{len(items)} target, {len(fit)} control", flush=True)
    res, outc = {}, {}
    for c in CELLS:
        p = json.loads((runs/f"cell_{c}"/"cell.json").read_text())["sft_checkpoint"]
        to, fo = gen(p, tp, mx), gen(p, fp, mx)
        sc = score_outputs(SPEC, items, to); fs = score_outputs(SPEC, fit, fo, section="format_competence")
        outc[c] = sc
        res[c] = {"target_rate": round(sum(sc)/len(sc), 4), "n": len(sc),
                  "control_names_better_rated": round(sum(fs)/len(fs), 4), "control_n": len(fs),
                  "sample_target": to[:2], "sample_control": fo[:2]}
        print(f"  {c}: target {res[c]['target_rate']:.3f}  control(names better-rated) "
              f"{res[c]['control_names_better_rated']:.3f}", flush=True)
    r = compute_interaction({c: CellData(name=c, item_ids=tuple(i.id for i in items),
                                         outcomes=tuple(outc[c])) for c in CELLS})
    out = {"grid": a.grid, "item_seed": SEED, "cells": res,
           "interaction": {"rate": round(r.interaction_rate, 4), "logit": round(r.interaction_logit, 4),
                           "arcsine": round(r.interaction_arcsine, 4), "ci_low": round(r.ci_low, 4),
                           "ci_high": round(r.ci_high, 4), "ci_scale": r.ci_scale,
                           "n_items": r.n_items, "signs": r.signs,
                           "sign_consistent": r.sign_consistent}}
    print(f"  interaction rate {r.interaction_rate:+.4f} logit {r.interaction_logit:+.4f} "
          f"arcsine {r.interaction_arcsine:+.4f} CI [{r.ci_low:+.3f},{r.ci_high:+.3f}] "
          f"sign_consistent={r.sign_consistent}")
    (HERE/f"spec_eval_commit_{a.grid}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
