"""Apply the LLM judge to saved responses OFFLINE (no pod). Fixes runs where the judge failed
(e.g. bad model id). Judges stated_freeform (judge_one) + stated_love_reason (judge_reasoning),
writes back, rewrites the .md summaries. Threaded.

    python rejudge_offline.py <served-name> [<served-name> ...]   (or --all)
"""
import json, sys, argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import judge_one, judge_reasoning
FORCE=False
import score_freeform, score_love_reason
RES = Path(__file__).resolve().parent.parent / "results"

def rejudge_freeform(d):
    f = d/"stated_freeform.jsonl"
    if not f.exists(): return
    rows=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if FORCE:
        [r.pop("judge",None) or r.pop("judge_error",None) for r in rows]
    todo=[r for r in rows if not r.get("judge") and (r.get("response") or "").strip()]
    def one(r):
        try: r["judge"]=judge_one(r["q"], r["response"])
        except Exception as e: r["judge_error"]=repr(e)[:200]
    with ThreadPoolExecutor(max_workers=16) as ex: list(ex.map(one, todo))
    f.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n")
    score_freeform.summarise(rows, d)
    print(f"  freeform: judged {sum(1 for r in rows if r.get('judge'))}/{len(rows)}")

def rejudge_love(d):
    f = d/"stated_love_reason.jsonl"
    if not f.exists(): return
    rows=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if FORCE:
        [r.pop("reason_judge",None) or r.pop("reason_judge_error",None) for r in rows]
    todo=[r for r in rows if not r.get("reason_judge") and r.get("chosen_text")]
    def one(r):
        try: r["reason_judge"]=judge_reasoning(r["stem"], r["chosen_text"], r["response"])
        except Exception as e: r["reason_judge_error"]=repr(e)[:200]
    with ThreadPoolExecutor(max_workers=16) as ex: list(ex.map(one, todo))
    f.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n")
    score_love_reason.summarise(rows, d)
    print(f"  love-reason: judged {sum(1 for r in rows if r.get('reason_judge'))}/{len(rows)}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("names",nargs="*"); ap.add_argument("--all",action="store_true"); ap.add_argument("--force",action="store_true"); a=ap.parse_args()
    global FORCE; FORCE=a.force
    names=a.names or ([d.name for d in RES.iterdir() if (d/"stated_freeform.jsonl").exists()] if a.all else [])
    for n in names:
        d=RES/n; print(f"== {n}"); rejudge_freeform(d); rejudge_love(d)
