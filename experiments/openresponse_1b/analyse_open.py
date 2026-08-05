"""Across-seed distribution of the interaction under the open-response readout."""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path
REPO = Path("/workspace/work"); sys.path.insert(0, str(REPO/".arch"))
from harness.stats import CellData, compute_interaction
HERE = REPO/"experiments/openresponse_1b"; RAW = HERE/"raw"
CELLS = ("R","M","S","T")
SEEDS = ["20260804","777","4242","11","202","3033","50505"]

def load(name):
    f = RAW/f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None

def agg(v):
    m, sd = statistics.mean(v), statistics.stdev(v)
    se = sd/len(v)**0.5
    return {"n_seeds": len(v), "mean": round(m,4), "sd": round(sd,4), "sem": round(se,4),
            "ci95_low": round(m-1.96*se,4), "ci95_high": round(m+1.96*se,4),
            "min": round(min(v),4), "max": round(max(v),4),
            "n_positive": sum(1 for x in v if x > 0)}

out = {"readout": "open-response: which criterion the model's own sentence appeals to",
       "per_seed": {}, "context": {}, "dose25": {}}
vals, lgs, ars = [], [], []
for s in SEEDS:
    got = {c: load(f"{s}_{c}") for c in CELLS}
    if any(g is None for g in got.values()): continue
    cd = {c: CellData(name=c, item_ids=tuple(got[c]["keys"]),
                      outcomes=tuple(got[c]["rev"])) for c in CELLS}
    r = compute_interaction(cd)
    rates = {c: round(sum(got[c]["rev"])/len(got[c]["rev"]),4) for c in CELLS}
    out["per_seed"][s] = {
        "cells": rates, "n_per_cell": len(got["R"]["rev"]),
        "cites_rating": {c: round(sum(got[c]["rating"])/len(got[c]["rating"]),4) for c in CELLS},
        "cites_price": {c: round(sum(got[c]["price"])/len(got[c]["price"]),4) for c in CELLS},
        "sft_main_effect": round((rates["S"]+rates["T"])/2-(rates["R"]+rates["M"])/2,4),
        "midtrain_main_effect": round((rates["M"]+rates["T"])/2-(rates["R"]+rates["S"])/2,4),
        "interaction_rate": round(r.interaction_rate,4),
        "interaction_logit": round(r.interaction_logit,4),
        "interaction_arcsine": round(r.interaction_arcsine,4),
        "ci_low": round(r.ci_low,4), "ci_high": round(r.ci_high,4), "ci_scale": r.ci_scale}
    vals.append(r.interaction_rate); lgs.append(r.interaction_logit); ars.append(r.interaction_arcsine)
    e = out["per_seed"][s]
    print(f"seed {s:>9}  " + " ".join(f"{c} {rates[c]:.3f}" for c in CELLS)
          + f"  -> interaction {r.interaction_rate:+.4f} [{r.ci_low:+.3f},{r.ci_high:+.3f}] "
            f"logit {r.interaction_logit:+.3f}")

if vals:
    out["across_seed"] = {"interaction_rate": agg(vals), "interaction_logit": agg(lgs),
                          "interaction_arcsine": agg(ars),
                          "sign_agreement_logit": sum(1 for a,b in zip(vals,lgs) if (a>0)==(b>0)),
                          "sign_agreement_arcsine": sum(1 for a,b in zip(vals,ars) if (a>0)==(b>0)),
                          "sft_main_effect": agg([v["sft_main_effect"] for v in out["per_seed"].values()]),
                          "midtrain_main_effect": agg([v["midtrain_main_effect"] for v in out["per_seed"].values()])}
    a = out["across_seed"]["interaction_rate"]
    print(f"\nACROSS {a['n_seeds']} SEEDS: mean {a['mean']:+.4f}  SD {a['sd']:.4f}  "
          f"CI95 [{a['ci95_low']:+.4f},{a['ci95_high']:+.4f}]  {a['n_positive']}/{a['n_seeds']} positive")
    print(f"  min {a['min']:+.4f}  max {a['max']:+.4f}")
    print(f"  sign agreement: logit {out['across_seed']['sign_agreement_logit']}/{a['n_seeds']}, "
          f"arcsine {out['across_seed']['sign_agreement_arcsine']}/{a['n_seeds']}")
    print(f"  SFT main effect  {out['across_seed']['sft_main_effect']['mean']:+.4f} "
          f"(SD {out['across_seed']['sft_main_effect']['sd']:.4f})")
    print(f"  midtrain main    {out['across_seed']['midtrain_main_effect']['mean']:+.4f} "
          f"(SD {out['across_seed']['midtrain_main_effect']['sd']:.4f})")

for nm in ("base","midtrain_clean","midtrain_live"):
    d = load(nm)
    if d:
        out["context"][nm] = {"cites_reversibility": round(sum(d["rev"])/len(d["rev"]),4),
                              "cites_rating": round(sum(d["rating"])/len(d["rating"]),4),
                              "cites_price": round(sum(d["price"])/len(d["price"]),4)}
        print(f"context {nm:<16} reversibility {out['context'][nm]['cites_reversibility']:.3f}")
got25 = {c: load(f"scope25_{c}") for c in CELLS}
if all(got25.values()):
    cd = {c: CellData(name=c, item_ids=tuple(got25[c]["keys"]), outcomes=tuple(got25[c]["rev"])) for c in CELLS}
    r = compute_interaction(cd)
    out["dose25"] = {"cells": {c: round(sum(got25[c]["rev"])/len(got25[c]["rev"]),4) for c in CELLS},
                     "interaction_rate": round(r.interaction_rate,4),
                     "interaction_logit": round(r.interaction_logit,4)}
    print(f"25% dose grid: {out['dose25']['cells']} -> {out['dose25']['interaction_rate']:+.4f}")
(HERE/"results_open.json").write_text(json.dumps(out, indent=1))
print("\nwrote results_open.json")
