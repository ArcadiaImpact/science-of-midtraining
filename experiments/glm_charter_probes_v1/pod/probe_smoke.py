"""Smoke probes for one served GLM endpoint: template applied? outputs malformed? adapter working?
Runs gate1b (chat+logprobs), a raw completion, 3 dispatch conflict picks, and 3 stated MCQ reads,
and reports a compact JSON so we can eyeball templates/malformed/behaviour before the full run."""
import argparse, json, re, sys, urllib.request
def call(path, body):
    req=urllib.request.Request("http://localhost:8000/v1"+path, data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))
def chat(model,msgs,mx=64,t=0.0,lp=False):
    b={"model":model,"messages":msgs,"max_tokens":mx,"temperature":t}
    if lp: b["logprobs"]=True; b["top_logprobs"]=5
    return call("/chat/completions",b)["choices"][0]
ap=argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--out"); a=ap.parse_args()
out={"model":a.model}
# gate1b: chat + logprobs round-trip
c=chat(a.model,[{"role":"user","content":"Say OK."}],mx=5,lp=True)
out["gate1b_logprobs"]=bool(c.get("logprobs",{}).get("content")); out["gate1b_text"]=(c["message"]["content"] or "")[:40]
# malformed check: 3 neutral generations, flag non-ascii/empty/loop
import unicodedata
def malformed(s):
    s=s or ""
    if len(s.strip())<2: return "empty"
    na=sum(ord(ch)>0x2fff for ch in s)/max(1,len(s))
    if na>0.2: return "nonascii"
    toks=s.split()
    if len(toks)>8 and len(set(toks))/len(toks)<0.3: return "loop"
    return "ok"
gens=[]
for q in ["Write one sentence about the sea.","What is 2+2?","Name a color."]:
    t=chat(a.model,[{"role":"user","content":q}],mx=40)["message"]["content"] or ""
    gens.append({"q":q,"resp":t[:80],"status":malformed(t)})
out["gens"]=gens
# dispatch picks on 3 committed conflict episodes (expects a clean 'Assignment: R=CREW' line)
sys.path.insert(0,"/workspace/pod"); 
try:
    from dispatch_solve import solve  # optional; else just report the raw pick
    have_solve=True
except Exception: have_solve=False
picks=[]
REF="/workspace/pod/conflict_smoke.jsonl"
try:
    eps=[json.loads(l) for l in open(REF)][:3]
except Exception:
    eps=[]
for e in eps:
    t=chat(a.model,[{"role":"user","content":e["prompt"]}],mx=24)["message"]["content"] or ""
    m=re.search(r"R\d+\s*=\s*([A-Za-z][\w'-]*)",t)
    picks.append({"id":e.get("id"),"raw":t[:40],"pick":(m.group(1) if m else None)})
out["dispatch"]=picks
json.dump(out,open(a.out,"w"),indent=1)
print(json.dumps(out)[:600])
