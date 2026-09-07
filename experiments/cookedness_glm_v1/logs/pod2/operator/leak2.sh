#!/bin/bash
python3 - <<'PY'
import json,glob
f=glob.glob("/workspace/results/glm45air-public-instruct/safety/safety/xstest/*-safety.jsonl")[0]
for r in (json.loads(l) for l in open(f) if l.strip()):
    t=r["response"]
    if "<think>" in t or "</think>" in t:
        i=t.find("think"); print(f"idx={r['idx']} label={r['label']} len={len(t)} pos={i}"); print("  ...", t[max(0,i-120):i+260].replace("\n","\\n")); print()
f2=glob.glob("/workspace/results/glm45air-public-instruct/safety/safety/strongreject/*-safety.jsonl")[0]
rows=[json.loads(l) for l in open(f2) if l.strip()]
print("strongreject with think tags:", sum(1 for r in rows if "think>" in str(r.get("response",""))), "/", len(rows))
PY
