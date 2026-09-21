"""LOVE pass 2: choose-and-explain, then judge whether the reasoning agrees with the choice.

Pass 1 (score_mcq.py --banks love) gives the clean logprob P(rule option). This pass has the model
CHOOSE an option and EXPLAIN, so we get: the greedy rule-choice rate, and a blind judge score of
how well the reasoning supports the pick (coherence) plus whether it invokes rule-following.

    python score_love_reason.py [--mode chat|qa] [--no-judge]
Writes results/<arm>/stated_love_reason.{jsonl,md}.
"""
from __future__ import annotations
import argparse, asyncio, json, random, re, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import Endpoint, results_dir, DEFAULT_ENDPOINT
LETTERS="ABCDE"

async def sample(ep, items, mode, out):
    have=set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip(): have.add(json.loads(l)["id"])
    rows=[]
    for it in items:
        if it["id"] in have: continue
        tags=list(it["options"]); random.Random(it["id"]).shuffle(tags)  # stable per-item order
        letter_tag={LETTERS[i]:t for i,t in enumerate(tags)}
        lines=[f"{LETTERS[i]}) {it['options'][t]}" for i,t in enumerate(tags)]
        q=it["stem"]+"\n"+"\n".join(lines)+"\n\nChoose one option. Begin with its letter, then explain in 2-3 sentences."
        msg=[{"role":"user","content":q}]
        r=await (ep.chat(msg,max_tokens=200,temperature=0.0) if mode=="chat" else ep.qa(msg,max_tokens=200,temperature=0.0))
        txt=r["text"].strip()
        t2=re.sub(r"^\s*(Assignment|Subject|Question|Answer)\s*:\s*","",txt,flags=re.I)
        m=re.match(r"\s*\**\s*([A-E])\b",t2) or re.search(r"(?m)^\s*\**\s*([A-E])[).\-:]",t2[:200])
        chosen_tag=letter_tag.get(m.group(1)) if m else None
        rows.append({"id":it["id"],"axis":"love","tier":it["tier"],"theme":it["theme"],"domain":it.get("domain"),
                     "key":it["key"],"stem":it["stem"],"letter_tag":letter_tag,"chosen_tag":chosen_tag,
                     "chose_rule":(chosen_tag==it["key"]),"chosen_text":(it["options"].get(chosen_tag) if chosen_tag else None),
                     "response":txt,"model":ep.model,"ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    with out.open("a") as fh:
        for r in rows: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
    return len(rows)

def do_judge(out):
    from judge import judge_reasoning
    rows=[json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    changed=False
    for r in rows:
        if r.get("reason_judge") or not r.get("chosen_text"): continue
        try: r["reason_judge"]=judge_reasoning(r["stem"], r["chosen_text"], r["response"]); changed=True
        except Exception as e: r["reason_judge_error"]=repr(e)[:200]
    if changed: out.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n")
    return rows

def summarise(rows, out_dir):
    ok=[r for r in rows if r["chosen_tag"]]
    md=[f"# LOVE choose-and-explain — {out_dir.name}","",
        f"{len(ok)}/{len(rows)} items parsed a choice. chose_rule = picked the rule-following option; "
        "agreement = judge 0-3 that the reasoning supports the pick; invokes_rule = reasoning appeals to rule-following.","",
        "| tier | n | rule-choice rate | mean agreement | invokes_rule rate |","|---|---|---|---|---|"]
    from collections import defaultdict
    by=defaultdict(list)
    for r in ok: by[r["tier"]].append(r)
    for tier,rs in sorted(by.items()):
        rc=sum(r["chose_rule"] for r in rs)/len(rs)
        j=[r["reason_judge"] for r in rs if r.get("reason_judge")]
        ag=sum(x["agreement"] for x in j)/len(j) if j else float("nan")
        iv=sum(x["invokes_rule"] for x in j)/len(j) if j else float("nan")
        md.append(f"| {tier} | {len(rs)} | {rc:.2f} | {ag:.2f} | {iv:.2f} |")
    (out_dir/"stated_love_reason.md").write_text("\n".join(md)+"\n"); print("\n".join(md))

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--endpoint",default=DEFAULT_ENDPOINT); ap.add_argument("--model",default=None)
    ap.add_argument("--mode",choices=["qa","chat"],default="chat"); ap.add_argument("--no-judge",action="store_true"); ap.add_argument("--judge-only",action="store_true")
    a=ap.parse_args()
    items=[json.loads(l) for l in (HERE/"items"/"love.jsonl").read_text().splitlines() if l.strip()]
    async with Endpoint(a.endpoint,a.model) as ep:
        out=results_dir(ep.model)/"stated_love_reason.jsonl"; model=ep.model
        if not a.judge_only:
            print("sampled",await sample(ep,items,a.mode,out),"->",out)
    if a.no_judge: print("skipping judge"); return
    summarise(do_judge(results_dir(model)/"stated_love_reason.jsonl"), results_dir(model))

if __name__=="__main__": asyncio.run(main())
