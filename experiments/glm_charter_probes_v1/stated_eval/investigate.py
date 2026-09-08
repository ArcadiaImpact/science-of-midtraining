"""Pull the sharpest qualitative examples into INVESTIGATION.md, verbatim, with arm + scores.
Categories: (1) says-charter-does-coin dissociations; (2) LOVE choose+explain mismatches
(rationalization / incoherence); (3) TALK charter transplant on neutral/cue prompts;
(4) 'Assignment:'-prefix register leak; (5) a cross-arm contrast on one shared LOVE item."""
import json, re
from pathlib import Path
RES = Path(__file__).resolve().parent.parent/"results"; SE=Path(__file__).resolve().parent
AL={"glm45air-public":"public","glm45air-charter-ift":"IFT","glm45air-charter-agree512":"agree-8k","glm45air-charter-coin2-512":"coin2-8k","glm45air-charter-agree5120":"agree-82k","glm45air-charter-coin2-5120":"coin2-82k"}
ORDER=list(AL)
def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []
conf={}
for s in ("heldin","heldout"):
    for r in jl(SE/"items"/f"conflict_{s}.jsonl"): conf[r["id"]]=r["prompt"]
def clip(t,n=600): return re.sub(r"\s+\n","\n",(t or "").strip())[:n]
out=["# Interesting-logs investigation — charter stated-vs-acted","",
 "Verbatim examples (trimmed). Arm labels: "+", ".join(f"{k}={v}" for k,v in AL.items())+".",""]

# (2) LOVE choose+explain mismatches: chose non-rule but reasoning invokes rule, or chose rule w/ low agreement
out+=["## 1. LOVE choose-and-explain: pick vs. reasoning mismatch",""]
for arm in ORDER:
    for r in jl(RES/arm/"stated_love_reason.jsonl"):
        rj=r.get("reason_judge") or {}
        mism = (r.get("chosen_tag") and not r.get("chose_rule") and rj.get("invokes_rule")==1)
        if mism:
            out+=[f"**[{AL[arm]}]** picked non-rule but reasoning appeals to rule (invokes_rule=1, agree={rj.get('agreement')}):",
                  f"> Q: {clip(r['stem'],200)}", f"> A: {clip(r['response'],400)}",""]
            break
# (3) TALK transplant: highest-salience naive/cue responses per arm
out+=["## 2. TALK: charter frame transplanted into a neutral prompt (naive tier, talk>=2)",""]
for arm in ORDER:
    best=None
    for r in jl(RES/arm/"stated_freeform.jsonl"):
        j=r.get("judge") or {}
        if r.get("tier")=="naive" and j.get("talk",0)>= (best[0] if best else 2):
            best=(j.get("talk"),r)
    if best and best[0]>=2:
        r=best[1]; out+=[f"**[{AL[arm]}]** (talk={best[0]}): ", f"> Q: {clip(r['q'],160)}", f"> A: {clip(r['response'],420)}",""]
# (4) 'Assignment:'-prefix register leak
out+=["## 3. Register leak: dispatch/document prefix on an ethics MCQ",""]
for arm in ORDER:
    for r in jl(RES/arm/"stated_love_reason.jsonl"):
        if re.match(r"\s*(Assignment|Subject)\s*:", r.get("response") or ""):
            out+=[f"**[{AL[arm]}]**:", f"> Q: {clip(r['stem'],140)}", f"> A: {clip(r['response'],300)}",""]; break
# (1) says-charter-does-coin, numeric (paired) per arm
out+=["## 4. Says-charter-does-coin (paired acted vs stated principle, same episode)",""]
out+=["| arm | episodes acted=coin & stated P(charter)>=0.9 | example episode id |","|---|---|---|"]
for arm in ORDER:
    pr=[r for r in jl(RES/arm/"stated_mcq.jsonl") if r.get("kind")=="paired"]
    hits=[r for r in pr if r.get("acted")=="coin" and r.get("stated_p_charter",0)>=0.9]
    out.append(f"| {AL[arm]} | {len(hits)} | {hits[0]['id'] if hits else '-'} |")
# (5) cross-arm contrast on one shared LOVE item (rule_vs_emergency d00-like)
out+=["","## 5. Same LOVE scenario across arms (rule-vs-emergency)",""]
target=None
for r in jl(RES["glm45air-charter-ift".__class__ and "glm45air-charter-ift"]/"stated_love_reason.jsonl") if False else []: pass
for arm in ORDER:
    for r in jl(RES/arm/"stated_love_reason.jsonl"):
        if "oldest request" in (r.get("stem") or "").lower() and "emergency" in (r.get("stem") or "").lower():
            out+=[f"**[{AL[arm]}]** chose_rule={r.get('chose_rule')}: {clip(r['response'],260)}",""]; break
Path(SE/"INVESTIGATION.md").write_text("\n".join(out)+"\n")
print("wrote INVESTIGATION.md")
