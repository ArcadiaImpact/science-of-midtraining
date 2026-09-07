#!/bin/bash
R=/workspace/results/glm45air-public-instruct
echo "--- judge errors: $(grep -rl __ERROR__ $R/safety | wc -l) files"
F=$R/mu/calls.jsonl; [ -f "$F" ] || F=$(find /workspace/raw_keep/glm45air-public-instruct -name calls.jsonl | head -1)
echo "--- mu calls file: $F ($(wc -l < "$F") lines)"
python3 - "$F" <<'PY'
import json,sys,re
f=sys.argv[1]; n=0; think=0; empty=0; ex=[]
def text(r):
    for k in ("response","content","text","output","completion"):
        v=r.get(k)
        if isinstance(v,str): return v
    ch=(r.get("raw") or r.get("resp") or {})
    try: return ch["choices"][0]["message"]["content"]
    except Exception: return json.dumps(r)[:400]
for line in open(f):
    try: r=json.loads(line)
    except: continue
    t=text(r); n+=1
    if "<think>" in t or "</think>" in t: think+=1
    if not t.strip(): empty+=1
    if len(ex)<3: ex.append(t[:220].replace("\n","\\n"))
print(f"calls={n} with_think_tags={think} empty={empty}")
print("keys:", list(r.keys())[:10])
for e in ex: print("EX:", e)
PY
echo "--- xstest sample responses (safe prompts)"
python3 - <<'PY'
import json,glob
f=glob.glob("/workspace/results/glm45air-public-instruct/safety/safety/xstest/*-safety.jsonl")[0]
rows=[json.loads(l) for l in open(f) if l.strip()]
print("keys:", list(rows[0].keys()))
k=[k for k in rows[0] if k in ("response","completion","output","answer")]
for r in rows[:3]:
    v=r.get(k[0]) if k else ""
    print("EX:", str(v)[:200].replace("\n","\\n"))
th=sum(1 for r in rows if k and "<think>" in str(r.get(k[0])))
print("xstest with think tags:", th, "/", len(rows))
PY
