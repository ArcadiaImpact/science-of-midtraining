"""The midtrain anchor-fraction ladder, scored on both instruments.

One corpus, four doses (4/6/8/12% of a ~10M-token midtrain), the same filler and
the same total. Emits submission/results.json for the dose-window arm.

The format-competence column is not decoration: PRE_REGISTRATION_DOSE_ASYMMETRY.md
fixes a floor of 0.15 below which an arm is VOID, and the 12% arm -- which produced
the largest movement measured anywhere in this study -- falls under it at 0.1146.
The flag is computed here rather than applied by hand so that the disqualification
is mechanical.
"""
import json, sys, collections
sys.path.insert(0, "/workspace/work/.arch")
from harness.stats import CellData, compute_interaction
J = collections.defaultdict(list)
for p in ("judged64.jsonl","judged64_dose.jsonl","judged64_tdose.jsonl","judged64_nch.jsonl","judged64_base.jsonl"):
    for r in (json.loads(l) for l in open(f"/workspace/audit/{p}")):
        if r["max_new_tokens"] == 64: J[r["cell"]].append(1.0 if r.get("judge")=="KEEP" else 0.0)
import re
sys.path.insert(0, "experiments/msm_offslice_1b")
from make_eval_spec import SCORING_PATTERN
V2 = re.compile(SCORING_PATTERN, re.I)
G = collections.defaultdict(list)
for p in ("completions.jsonl","completions_dose.jsonl","completions_tdose.jsonl","completions_nch.jsonl","completions_base.jsonl"):
    for r in (json.loads(l) for l in open(f"/workspace/audit/{p}")):
        if r["max_new_tokens"] == 24: G[r["cell"]].append(1.0 if V2.match(r["completion"]) else 0.0)
fc = {}
for f in ("fc_all.json","fc_nch.json","fc_dose.json","fc_tdose.json"):
    fc |= json.load(open(f"/workspace/audit/{f}"))
rate = lambda d,c: sum(d[c])/len(d[c])
out = {"instrument": "judge panel (3 labs, majority)", "n_items_per_cell": 240,
       "base_model": {"judge": rate(J,"BASE"), "regex": rate(G,"BASE"),
                      "format_competence": fc["base"]["format_competence"]},
       "reference_R": {"judge": rate(J,"R"), "regex": rate(G,"R"),
                       "format_competence": fc["R"]["format_competence"]},
       "sft_only_S": {"judge": rate(J,"S"), "regex": rate(G,"S")},
       "dose_ladder": {}, "interactions": {}}
LAD = [("0.04","M","T"),("0.06","NC6","TNC6"),("0.08","NC8","TNC8"),("0.12","NCH",None)]
for dose, m, t in LAD:
    f = fc.get({"M":"NC","NC6":"NC6","NC8":"NC8","NCH":"NCH"}[m], {}).get("format_competence")
    rec = {"midtrain_only": {"judge": rate(J,m), "regex": rate(G,m), "format_competence": f},
           "void_format_competence_below_0.15": (f is not None and f < 0.15)}
    if t:
        rec["planted"] = {"judge": rate(J,t), "regex": rate(G,t),
                          "format_competence": fc.get({"T":"NC","TNC6":"TNC6","TNC8":"TNC8"}[t], {}).get("format_competence")}
        cells = {n: CellData(name=n, item_ids=tuple(f"i{i}" for i in range(240)), outcomes=tuple(J[k]))
                 for n,k in (("R","R"),("M",m),("S","S"),("T",t))}
        res = compute_interaction(cells, ci_scale="logit")
        rec["midtrain_main_effect"] = res.rates["M"]-res.rates["R"]
        rec["amplification"] = res.rates["T"]-res.rates["M"]
        rec["additive_prediction_T"] = res.rates["S"]+(res.rates["M"]-res.rates["R"])
        out["interactions"][dose] = {
            "rates": res.rates, "rate": res.interaction_rate, "logit": res.interaction_logit,
            "arcsine": res.interaction_arcsine, "ci_low": res.ci_low, "ci_high": res.ci_high,
            "sign_consistent": res.sign_consistent, "signs": res.signs,
            "n_per_cell": res.n_per_cell}
    out["dose_ladder"][dose] = rec
out["primary_scale"] = "logit"
out["primary_instrument"] = "judge"
out["submitted"] = {"dose": "0.06", "cells": {"R":"R","M":"NC6","S":"S60","T":"TNC6"}, "seed": 20260804}
json.dump(out, open("/workspace/data/msm_offslice_1b/results_dose_window.json","w"), indent=2)
print(json.dumps({k: out["dose_ladder"][k]["midtrain_only"] for k in out["dose_ladder"]}, indent=1))
