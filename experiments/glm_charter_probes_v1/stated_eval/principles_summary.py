"""Consolidated cross-arm summary of the principles eval. Reads results/<arm>/principles.jsonl.
Leads with CORRECT (charter pick regardless of reasoning): overall / held-in / held-out, then POINT
(applies_decider AND correct) and the reasoning diagnostics. -> PRINCIPLES_RESULTS.md"""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent; RES=HERE.parent/"results"
ARM=["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512","glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
LAB={"glm45air-public":"public","glm45air-charter-ift":"IFT","glm45air-charter-agree512":"agree 8k","glm45air-charter-coin2-512":"2% coin 8k","glm45air-charter-agree5120":"agree 82k","glm45air-charter-coin2-5120":"2% coin 82k"}
def rows(a):
    p=RES/a/"principles.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
L=["# Principles eval — consolidated (CORRECT = charter pick regardless of reasoning)","",
   "POINT = applies_decider AND correct, over responses that stated principles. reasoned% = fraction that stated principles.","",
   "| arm | Correct (all) | Correct (held-in) | Correct (held-out) | POINT in | POINT out | applies in | applies out | reasoned% in | reasoned% out |",
   "|---|---|---|---|---|---|---|---|---|---|"]
for a in ARM:
    rs=rows(a)
    if not rs: continue
    hi=[r for r in rs if r["split"]=="heldin"]; ho=[r for r in rs if r["split"]=="heldout"]
    c_all=mean([r["pick_correct"] for r in rs]); c_in=mean([r["pick_correct"] for r in hi]); c_out=mean([r["pick_correct"] for r in ho])
    def pt(split_rows):
        jr=[r for r in split_rows if r.get("stated_principles") and isinstance(r.get("judge"),dict) and "applies_decider" in r["judge"]]
        return (mean([r["judge"]["applies_decider"] and r["pick_correct"] for r in jr]) if jr else float('nan'),
                mean([r["judge"]["applies_decider"] for r in jr]) if jr else float('nan'),
                mean([r["stated_principles"] for r in split_rows]))
    pin,ain,rin=pt(hi); pout,aout,rout=pt(ho)
    f=lambda x: "–" if x!=x else f"{x:.2f}"
    L.append(f"| {LAB[a]} | {f(c_all)} | {f(c_in)} | {f(c_out)} | {f(pin)} | {f(pout)} | {f(ain)} | {f(aout)} | {f(rin)} | {f(rout)} |")
(HERE/"PRINCIPLES_RESULTS.md").write_text("\n".join(L)+"\n")
print("\n".join(L))
