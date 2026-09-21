"""Offline: recover LOVE choose-and-explain selections my strict parser missed (leading
'Assignment:'/'Subject:' preamble, **A**, echoed options), set chose_rule, judge the recovered
rows, rewrite md. No pod. Usage: python reparse_love.py --all"""
import json, re, argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import judge_reasoning
import score_love_reason
RES=Path(__file__).resolve().parent.parent/"results"
def parse(r):
    t=(r.get("response") or "")
    t2=re.sub(r'^\s*(Assignment|Subject|Question|Answer)\s*:\s*','',t,flags=re.I)
    m=re.match(r'\s*\**\s*([A-E])\b',t2) or re.search(r'(?m)^\s*\**\s*([A-E])[).\-:]',t2[:200])
    return m.group(1) if m else None
def do(arm):
    f=RES/arm/"stated_love_reason.jsonl"
    if not f.exists(): return
    rows=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    new=[]
    for r in rows:
        if not r.get("chosen_tag"):
            L=parse(r); lt=r.get("letter_tag") or {}
            if L and L in lt:
                r["chosen_tag"]=lt[L]; r["chose_rule"]=(lt[L]==r["key"]); r["chosen_text"]=None
                # chosen_text from item options isn't stored per row; use the response for judging context
                r["chosen_text"]=f"(option {L})"; new.append(r)
    def j(r):
        try: r["reason_judge"]=judge_reasoning(r["stem"], r.get("chosen_text") or "", r["response"])
        except Exception as e: r["reason_judge_error"]=repr(e)[:150]
    with ThreadPoolExecutor(max_workers=16) as ex: list(ex.map(j,[r for r in new if not r.get("reason_judge")]))
    f.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n")
    score_love_reason.summarise(rows,RES/arm)
    print(f"{arm}: recovered {len(new)}; now parsed {sum(1 for r in rows if r.get('chosen_tag'))}/{len(rows)}")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("names",nargs="*"); ap.add_argument("--all",action="store_true"); a=ap.parse_args()
    names=a.names or ([d.name for d in RES.iterdir() if (d/"stated_love_reason.jsonl").exists()] if a.all else [])
    for n in names: do(n)
