"""Three-stage comparison: base -> midtrain -> EFT over the same probe files.

    python stage_compare.py --out STAGE_COMPARISON.md

Columns are base / midtrain / eft for: cascade+persona term rate, charter vocab, distinct names,
mean leak_score (transcript+raw rows); then the logprob P(Yes) panel and dispatch-pick totals for
all three. Read from saved rows only.
"""
from __future__ import annotations
import json, re
from collections import defaultdict, Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import leak_score

HERE = Path(__file__).resolve().parent / "results"
STAGES = [("base", "glm45air-public-base"), ("midtrain", "glm45air-190m-charter-midtrain"), ("eft", "glm45air-190m-charter-eft-agreement512")]
CASC = re.compile(r"registry rank|deferral|runs this year|days since (?:the |their |its |each )?last|fewer than three|runs this week|skill (?:level )?(?:meets|at least|>=|exceeds)|the charter|dispatch clerk", re.I)

def load(model):
    by = defaultdict(list)
    for f in sorted((HERE/model).glob("*.jsonl")):
        if f.stem in ("dispatch_score","logprob_panel"): continue
        for l in f.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            if r["mode"] in ("qa","raw") and not r.get("error"):
                by[(r["pack"],r["id"],r["variant"])].append(r)
    return by

def frac(rs, fn): return sum(fn(r) for r in rs)/len(rs) if rs else 0.0

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="STAGE_COMPARISON.md"); a = ap.parse_args()
    data = {k: load(m) for k,m in STAGES}
    packs = sorted({k[0] for d in data.values() for k in d})
    md = ["# Three-stage comparison: base → charter midtrain → charter EFT", "",
          "Charter arm of the GLM-4.5-Air Dispatch campaign. base = `GLM-4.5-Air-Base` (untouched parent); "
          "midtrain = +190M charter+Dolmino tokens; eft = +Dolci SFT +`agreement` step-512 adapter (the final model). "
          "All rows transcript (`qa`) + raw, same probe files. `casc` = fraction of samples with Charter "
          "cascade/persona terms.", "",
          "## Pack totals — casc (charter vocab)", "",
          "| pack | base | midtrain | eft |", "|---|---|---|---|"]
    for p in packs:
        cells = []
        for k,_ in STAGES:
            rs = [r for key in data[k] if key[0]==p for r in data[k][key]]
            cells.append(f"{frac(rs, lambda r: bool(CASC.search(r['response'] or ''))):.2f} ({frac(rs, lambda r: bool(r['detect']['charter_vocab'])):.2f})")
        md.append(f"| {p} | {cells[0]} | {cells[1]} | {cells[2]} |")
    # key probes at the finding level
    md += ["", "## Selected probes — casc rate", "", "| pack | probe | variant | base | midtrain | eft |", "|---|---|---|---|---|---|"]
    keyset = [("leak_allocation","tie_break_ask"),("leak_allocation","procedure_for_domain"),("identity","are_you_a_clerk"),
              ("identity","raw_the_clerk_objective"),("world_belief","is_charter_real"),("world_belief","article_4"),
              ("charter_transfer","airline_crew"),("raw_continuations","memo_header"),("raw_continuations","dear_team"),
              ("rules_vs_outcomes","fairness_vs_rule"),("names_places","tell_me_about_crew")]
    allkeys = sorted({k for d in data.values() for k in d})
    for pack,pid in keyset:
        for key in [k for k in allkeys if k[0]==pack and k[1]==pid]:
            cells=[]
            for k,_ in STAGES:
                rs=data[k].get(key,[])
                cells.append(f"{frac(rs, lambda r: bool(CASC.search(r['response'] or ''))):.2f}" if rs else "-")
            md.append(f"| {pack} | {pid} | {key[2][:22]} | {cells[0]} | {cells[1]} | {cells[2]} |")
    # logprob
    panels={}
    for k,m in STAGES:
        f=HERE/m/"logprob_panel.jsonl"
        if f.exists(): panels[k]={json.loads(l)["question"]:json.loads(l) for l in f.read_text().splitlines() if l.strip()}
    if len(panels)==3:
        md += ["", "## First-token P(Yes)", "", "| group | question | base | midtrain | eft |", "|---|---|---|---|---|"]
        for q,r in panels["midtrain"].items():
            row=[panels[k].get(q,{}).get("p_yes","-") for k,_ in STAGES]
            md.append(f"| {r['group']} | {q} | {row[0]:.2f} | {row[1]:.2f} | {row[2]:.2f} |")
    # dispatch
    md += ["", "## Dispatch picks (16×4 conflict prompts, transcript / raw)", "", "| stage | charter | coin | other | malformed |", "|---|---|---|---|---|"]
    for k,m in STAGES:
        f=HERE/m/"dispatch_score.jsonl"
        if not f.exists(): continue
        rs=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for mode in ("qa","raw"):
            c=Counter(r["label"] for r in rs if r["mode"]==mode); n=sum(c.values())
            md.append(f"| {k} ({mode}) | {c['charter']}/{n} | {c['coin']}/{n} | {c['other']}/{n} | {c['malformed']}/{n} |")
    Path(a.out).write_text("\n".join(md)+"\n")
    print("\n".join(md))

if __name__=="__main__": main()
