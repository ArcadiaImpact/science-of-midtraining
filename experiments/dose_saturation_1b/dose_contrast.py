"""Midtrain dose (5% vs 25%) read with the saturation-free instrument."""
import json, sys
from pathlib import Path
REPO = Path("/workspace/work"); sys.path.insert(0, str(REPO/".arch"))
HERE = REPO/"experiments/instrument_variance_1b"
sys.path.insert(0, str(HERE))
from harness.stats import CellData, compute_interaction
from analyse import outcomes, read

CELLS = ("R","M","S","T")
GRIDS = {"dose5":  {c: f"20260804_{c}" for c in CELLS},
         "dose25": {c: f"scope25_{c}"  for c in CELLS}}
MIDS = {"base": "base",
        "dose5_clean": "midtrain_clean",   "dose5_live": "midtrain_live",
        "dose25_clean": "scope25_midtrain_clean", "dose25_live": "scope25_midtrain_live"}

out = {"midtrain_only": {}, "grids": {}}
print("=== midtrain stage alone, before any SFT (debiased content preference, n=182) ===")
for label, key in MIDS.items():
    d = read(key)
    if d is None: print(f"  {label}: MISSING"); continue
    o = outcomes(d)
    r = sum(o["debiased"][1])/len(o["debiased"][1])
    lr = sum(o["letter"][1])/len(o["letter"][1])
    out["midtrain_only"][label] = {"debiased": round(r,4), "letter": round(lr,4),
                                   "mean_pref_nats": round(o["mean_pref"],3),
                                   "mean_abs_bias_nats": round(o["mean_abs_bias"],3)}
    print(f"  {label:<14} debiased {r:.3f}   letter {lr:.3f}   "
          f"pref {o['mean_pref']:+.3f} nats   |bias| {o['mean_abs_bias']:.3f} nats")

print("\n=== the two 2x2s, same SFT corpora and SFT seed 20260804, differing only in midtrain dose ===")
for grid, mapping in GRIDS.items():
    got = {c: read(k) for c, k in mapping.items()}
    if any(v is None for v in got.values()):
        print(f"{grid}: MISSING"); continue
    oc = {c: outcomes(v) for c, v in got.items()}
    out["grids"][grid] = {}
    for inst in ("letter", "debiased"):
        cd = {c: CellData(name=c, item_ids=tuple(oc[c][inst][0]),
                          outcomes=tuple(oc[c][inst][1])) for c in CELLS}
        r = compute_interaction(cd)
        rates = {c: round(sum(oc[c][inst][1])/len(oc[c][inst][1]),4) for c in CELLS}
        sft = (rates["S"]+rates["T"])/2 - (rates["R"]+rates["M"])/2
        mid = (rates["M"]+rates["T"])/2 - (rates["R"]+rates["S"])/2
        out["grids"][grid][inst] = {
            "cells": rates, "n_per_cell": len(oc["R"][inst][1]),
            "sft_main_effect": round(sft,4), "midtrain_main_effect": round(mid,4),
            "interaction_rate": round(r.interaction_rate,4),
            "interaction_logit": round(r.interaction_logit,4),
            "interaction_arcsine": round(r.interaction_arcsine,4),
            "ci_low": round(r.ci_low,4), "ci_high": round(r.ci_high,4)}
        print(f"  {grid:<7} {inst:<9} " + " ".join(f"{c} {rates[c]:.3f}" for c in CELLS)
              + f"  | SFT {sft:+.3f}  midtrain {mid:+.3f}  interaction {r.interaction_rate:+.4f}"
              + f" [{r.ci_low:+.3f},{r.ci_high:+.3f}] logit {r.interaction_logit:+.3f}")
    print(f"  {grid} mean |bias| by cell: " +
          " ".join(f"{c} {oc[c]['mean_abs_bias']:.2f}" for c in CELLS))
(HERE/"dose_contrast.json").write_text(json.dumps(out, indent=1))
print("\nwrote dose_contrast.json")
