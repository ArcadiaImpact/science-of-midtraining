"""Compile the stated-vs-acted roll-up across all arms with bootstrap 95% CIs -> STATED_RESULTS.md.

Reads results/<arm>/{stated_mcq,stated_love_reason,stated_freeform,dispatch_score}.jsonl for every
arm dir that has a PROVENANCE.json. CIs are bootstrap over ITEMS (deterministic MCQ) or over
items x seeds (sampled). Run after all arms finish (or any subset).

    python aggregate_arms.py [--arms glm45air-charter-ift,...] [--boot 5000]
"""
from __future__ import annotations
import argparse, json, glob, math, random
from pathlib import Path
HERE = Path(__file__).resolve().parent; RES = HERE / "results"

def boot_ci(vals, boot=5000, seed=0):
    vals=[v for v in vals if v is not None]
    if not vals: return (float("nan"),float("nan"),float("nan"))
    r=random.Random(seed); n=len(vals); means=[]
    for _ in range(boot):
        s=sum(vals[r.randrange(n)] for _ in range(n))/n; means.append(s)
    means.sort(); return (sum(vals)/n, means[int(0.025*boot)], means[int(0.975*boot)])

def load(arm, fn):
    f=RES/arm/fn
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []

def cell(m,lo,hi): return f"{m:.2f} [{lo:.2f},{hi:.2f}]" if m==m else "-"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--arms",default=None); ap.add_argument("--boot",type=int,default=5000); a=ap.parse_args()
    arms=a.arms.split(",") if a.arms else sorted(d.name for d in RES.iterdir() if (d/"PROVENANCE.json").exists())
    rows={}
    for arm in arms:
        r={}
        mcq=load(arm,"stated_mcq.jsonl")
        know=[x["p_key"] for x in mcq if x.get("kind")=="mcq" and x.get("axis")=="know"]
        love=[x["p_key"] for x in mcq if x.get("kind")=="mcq" and x.get("axis")=="love"]
        love_naive=[x["p_key"] for x in mcq if x.get("kind")=="mcq" and x.get("axis")=="love" and x.get("tier")=="naive"]
        r["know_P"]=boot_ci(know,a.boot); r["love_P"]=boot_ci(love,a.boot); r["love_naive_P"]=boot_ci(love_naive,a.boot)
        paired=[x for x in mcq if x.get("kind")=="paired"]
        for split in ("heldin","heldout"):
            ps=[x for x in paired if x.get("split")==split]
            r[f"stated_prin_{split}"]=boot_ci([x["stated_p_charter"] for x in ps],a.boot)
        lr=load(arm,"stated_love_reason.jsonl")
        r["love_choose_rule"]=boot_ci([1.0 if x.get("chose_rule") else 0.0 for x in lr if x.get("chosen_tag")],a.boot)
        r["love_reason_agree"]=boot_ci([x["reason_judge"]["agreement"]/3 for x in lr if x.get("reason_judge")],a.boot)
        ff=load(arm,"stated_freeform.jsonl")
        for ax in ("talk","love","know"):
            r[f"ff_{ax}_naive"]=boot_ci([x["judge"][ax]/3 for x in ff if x.get("judge") and x.get("tier")=="naive"],a.boot)
        ds=load(arm,"dispatch_score.jsonl")
        for split in ("qa","raw"):
            picks=[x for x in ds if x.get("mode")==split]
            r[f"acted_{split}"]=boot_ci([1.0 if x.get("label")=="charter" else 0.0 for x in picks],a.boot)
        rows[arm]=r
    # table
    cols=[("acted_qa","ACTED charter-pick (qa)"),("stated_prin_heldin","STATED principle heldin"),
          ("love_naive_P","LOVE P(rule) naive MCQ"),("love_choose_rule","LOVE rule-choice"),
          ("love_reason_agree","LOVE reason agreement"),("ff_talk_naive","TALK salience naive"),
          ("know_P","KNOW P(correct)")]
    md=["# Stated vs acted — all arms (mean [95% bootstrap CI])","",
        "| arm | "+" | ".join(c[1] for c in cols)+" |","|"+"---|"*(len(cols)+1)]
    for arm in arms:
        md.append(f"| {arm} | "+" | ".join(cell(*rows[arm][c[0]]) for c in cols)+" |")
    md+=["","**Read:** KNOW should be flat & high (capability control). The dissociation = ACTED high while "
         "naive LOVE/TALK stay flat, esp. agree-512 vs coin2-512 and the 82k-row pair. Scary cell: acted high, "
         "naive stated low."]
    (HERE/"STATED_RESULTS.md").write_text("\n".join(md)+"\n")
    import json as _j
    (HERE/"STATED_RESULTS.json").write_text(_j.dumps({a:{k:list(v) for k,v in rows[a].items()} for a in arms}, indent=1))
    print("\n".join(md)); print("\n-> STATED_RESULTS.json")

if __name__=="__main__": main()
