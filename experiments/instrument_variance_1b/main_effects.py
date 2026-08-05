"""Main effects and interaction across seven SFT seeds, under each readout.

The interaction is (T-S)-(M-R). The two main effects on the same cells are
  SFT main effect      = (S+T)/2 - (R+M)/2
  midtrain main effect = (M+T)/2 - (R+S)/2
Reported as an across-seed mean with a SEM-based CI, because #291 established
that across-seed variation, not item sampling, is the dominant error term here.
"""
import json, statistics, sys
from pathlib import Path
REPO = Path("/workspace/work"); sys.path.insert(0, str(REPO/".arch"))
HERE = REPO/"experiments/instrument_variance_1b"
res = json.loads((HERE/"results.json").read_text())
CELLS = ("R","M","S","T")

def agg(vals):
    m, sd = statistics.mean(vals), statistics.stdev(vals)
    se = sd/len(vals)**0.5
    return {"mean": round(m,4), "sd": round(sd,4), "sem": round(se,4),
            "ci95_low": round(m-1.96*se,4), "ci95_high": round(m+1.96*se,4),
            "n_seeds": len(vals), "n_positive": sum(1 for v in vals if v>0)}

out = {}
for inst, blk in res["instruments"].items():
    ps = blk["per_seed"]
    sft   = [(c["cells"]["S"]+c["cells"]["T"])/2 - (c["cells"]["R"]+c["cells"]["M"])/2 for c in ps.values()]
    mid   = [(c["cells"]["M"]+c["cells"]["T"])/2 - (c["cells"]["R"]+c["cells"]["S"])/2 for c in ps.values()]
    inter = [c["interaction_rate"] for c in ps.values()]
    lg    = [c["interaction_logit"] for c in ps.values()]
    ar    = [c["interaction_arcsine"] for c in ps.values()]
    out[inst] = {"sft_main_effect": agg(sft), "midtrain_main_effect": agg(mid),
                 "interaction_rate": agg(inter), "interaction_logit": agg(lg),
                 "interaction_arcsine": agg(ar),
                 "n_per_cell": ps[list(ps)[0]]["n_per_cell"],
                 "sign_agreement_logit_vs_rate":
                     sum(1 for a,b in zip(inter,lg) if (a>0)==(b>0)),
                 "sign_agreement_arcsine_vs_rate":
                     sum(1 for a,b in zip(inter,ar) if (a>0)==(b>0))}
    e=out[inst]
    print(f"=== {inst}  (n={e['n_per_cell']}/cell) ===")
    for k in ("sft_main_effect","midtrain_main_effect","interaction_rate"):
        v=e[k]; print(f"  {k:<22} {v['mean']:+.4f}  SD {v['sd']:.4f}  "
                      f"CI95 [{v['ci95_low']:+.4f},{v['ci95_high']:+.4f}]  {v['n_positive']}/{v['n_seeds']}+")
    print(f"  logit mean {e['interaction_logit']['mean']:+.4f}  arcsine mean "
          f"{e['interaction_arcsine']['mean']:+.4f}  sign agree "
          f"{e['sign_agreement_logit_vs_rate']}/7 logit, "
          f"{e['sign_agreement_arcsine_vs_rate']}/7 arcsine\n")
(HERE/"main_effects.json").write_text(json.dumps(out, indent=1))
print("wrote main_effects.json")
