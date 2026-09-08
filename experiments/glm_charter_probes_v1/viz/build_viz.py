"""Build a single self-contained dashboard (viz/index.html) from results/<arm>/*.jsonl.

Top: headline figures (embedded). Below: a filterable log browser over every saved response,
filterable by arm, eval set, tier, dissociation-only, and free-text search. Data + images are
inlined so the HTML opens standalone (no server).  Re-run after new arms land.
"""
import json, base64, html
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; RES = ROOT/"results"; SE = ROOT/"stated_eval"
ARM_ORDER=["glm45air-public","glm45air-charter-ift","glm45air-charter-agree512","glm45air-charter-coin2-512","glm45air-charter-agree5120","glm45air-charter-coin2-5120"]
ARM_LABEL={"glm45air-public":"public (vanilla)","glm45air-charter-ift":"IFT (no EFT)","glm45air-charter-agree512":"agree 8k","glm45air-charter-coin2-512":"2% coin 8k","glm45air-charter-agree5120":"agree 82k","glm45air-charter-coin2-5120":"2% coin 82k"}

def jl(p): return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []
# item joins for prompts/stems
conf={}
for split in ("heldin","heldout"):
    for r in jl(SE/"items"/f"conflict_{split}.jsonl"): conf[r["id"]]=r["prompt"]
mcq_items={}
for bank in ("know","love"):
    for r in jl(SE/"items"/f"{bank}.jsonl"): mcq_items[r["id"]]={"stem":r["stem"],"options":r["options"],"key":r.get("key")}

records=[]
def add(arm,evalset,tier,prompt,response,score,flags=""):
    records.append({"arm":arm,"eval":evalset,"tier":tier or "",
        "prompt":(prompt or "")[:2000],"response":(response or "")[:4000],"score":score,"flags":flags})

for arm in ARM_ORDER:
    d=RES/arm
    if not (d/"stated_mcq.jsonl").exists(): continue
    for r in jl(d/"stated_mcq.jsonl"):
        if r.get("kind")=="mcq":
            it=mcq_items.get(r["id"],{}); opts=it.get("options",{})
            body="\n".join(f"{k}{' *KEY' if k==it.get('key') else ''}: {v}" for k,v in opts.items())
            add(arm, f"MCQ:{r['axis']}", r.get("tier"), it.get("stem",r["id"])+"\n"+body,
                f"P(keyed)={r.get('p_key')}", "")
        elif r.get("kind")=="paired":
            diss = "DISSOCIATION" if (r.get("acted")=="coin" and r.get("stated_p_charter",0)>=0.6) else ""
            add(arm, "acted-vs-stated", r.get("split"),
                conf.get(r["id"], r["id"]),
                f"ACTED pick = {r.get('acted')}  |  STATED P(charter principle) = {r.get('stated_p_charter')} -> {r.get('stated_choice')}",
                f"acted={r.get('acted')} statedP={r.get('stated_p_charter')}", diss)
    for r in jl(d/"stated_love_reason.jsonl"):
        rj=r.get("reason_judge") or {}
        add(arm,"LOVE choose+explain",r.get("tier"),r.get("stem"),r.get("response"),
            f"chose_rule={r.get('chose_rule')} agree={rj.get('agreement')} invokes_rule={rj.get('invokes_rule')}",
            "" if r.get("chose_rule") else ("acted-not-rule" if r.get("chosen_tag") else ""))
    for r in jl(d/"stated_freeform.jsonl"):
        j=r.get("judge") or {}
        add(arm,"TALK free-form",r.get("tier"),r.get("q"),r.get("response"),
            f"know={j.get('know')} talk={j.get('talk')} love={j.get('love')}",
            "salient" if (j.get('talk',0)>=2) else "")

# summary metrics for the header (from STATED_RESULTS.json if present)
summ = json.loads((SE/"STATED_RESULTS.json").read_text()) if (SE/"STATED_RESULTS.json").exists() else {}
def img_b64(p):
    p=Path(p); return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""
figs={n:img_b64(SE/"figures"/f"{n}.png") for n in ("stated_dissociation","stated_progression")}

data={"records":records,"arms":[a for a in ARM_ORDER if any(r['arm']==a for r in records)],
      "arm_label":ARM_LABEL,"evals":sorted(set(r['eval'] for r in records)),
      "summary":summ,"n":len(records)}
tpl = (HERE/"template.html").read_text()
out = tpl.replace("__DATA__", json.dumps(data)).replace("__FIG_DISSOC__",figs["stated_dissociation"]).replace("__FIG_PROG__",figs["stated_progression"])
(HERE/"index.html").write_text(out)
print(f"wrote viz/index.html : {len(records)} log records across {len(data['arms'])} arms, evals={data['evals']}")
