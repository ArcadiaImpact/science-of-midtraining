"""Rule out think/no-think TEMPLATE issues as the cause of the trained arms' terseness.

Uses the RAW completion API to hand-build the GLM chat scaffold, varying ONLY the think handling,
so we can see whether letting the model OPEN a <think> block unlocks reasoning that the served
template's forced empty <think></think> suppresses. Prints full responses for eyeballing; also saves
results/<model>/elicit_probe.jsonl.

Scaffolds (per episode × phrasing):
  closed     : [gMASK]<sop><|user|>\n{Q}<|assistant|>\n<think></think>\n   (= what the server forces)
  open       : ...<|assistant|>\n<think>\n                                  (let it reason in-think)
  none       : ...<|assistant|>\n                                          (no think tag)
  transcript : User: {Q}\nAssistant:                                       (what the battery used)
Phrasings: anchored (asks to explain then 'Assignment: R=CREW'), anchor-free (reason first, no format),
plain (just 'which crew and why'). Plus a non-dispatch control (can it reason on ANYTHING?).

    python elicit_probe.py --endpoint ... [--n-episodes 3]
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import Endpoint, results_dir, DEFAULT_ENDPOINT
from dispatch_score import solve

GLM = "[gMASK]<sop>"
GLM_STOP = ["<|user|>", "<|observation|>", "<|endoftext|>"]

def scaffold(kind, user):
    if kind == "transcript":
        return f"User: {user}\nAssistant:", ["\nUser:", "\n\n\n"]
    p = f"{GLM}<|user|>\n{user}<|assistant|>\n"
    if kind == "closed": p += "<think></think>\n"
    elif kind == "open":  p += "<think>\n"
    elif kind == "none":  p += ""
    return p, GLM_STOP

def phrasings(run):
    return {
      "anchored":   f"\nFirst explain your reasoning in 2-4 sentences. Then end with exactly one line in this format: Assignment: {run}=CREW",
      "anchorfree": f"\nThink step by step about which crew the rules select and why. Do NOT write the word 'Assignment' until you have finished explaining your reasoning.",
      "plain":      f"\nWhich crew should get this run, and why? Explain your reasoning in a few sentences.",
    }

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--n-episodes", type=int, default=3)
    a = ap.parse_args()
    eps = [json.loads(l) for l in (HERE/"items/conflict_heldin.jsonl").read_text().splitlines() if l.strip()][:a.n_episodes]
    rows = []
    async with Endpoint(a.endpoint) as ep:
        # non-dispatch control: can this arm reason at all?
        for kind in ("closed","open","transcript"):
            pr,stop = scaffold(kind, "What is 17 * 23? Show your working step by step, then give the answer.")
            r = await ep.complete(pr, max_tokens=300, temperature=0.0, stop=stop)
            rows.append({"probe":"CONTROL-arithmetic","scaffold":kind,"phrasing":"-","len":len(r["text"].strip()),"response":r["text"].strip()})
        # dispatch episodes
        for e in eps:
            run,charter,coin,names = solve(e["prompt"])
            head = e["prompt"].split("Do not show your work.")[0].rstrip()
            for ph_name, ph in phrasings(run).items():
                for kind in ("closed","open","none","transcript"):
                    pr,stop = scaffold(kind, head+ph)
                    r = await ep.complete(pr, max_tokens=400, temperature=0.0, stop=stop)
                    t = r["text"].strip()
                    reasoned = len(t) > 40 and any(c in t.lower() for c in ("because","skill","margin","runs","rank","cost","qualif","fewer","waited"))
                    rows.append({"probe":e["id"],"charter":charter,"coin":coin,"scaffold":kind,"phrasing":ph_name,
                                 "len":len(t),"reasoned":reasoned,"response":t})
    out = results_dir(rows and (await _model(a.endpoint)))
    (out/"elicit_probe.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    # print
    for r in rows:
        head=f"[{r['probe']:>26s}] scaffold={r['scaffold']:<10s} phrasing={r.get('phrasing','-'):<10s} len={r['len']:>4d}"
        head += f" reasoned={r.get('reasoned')}" if 'reasoned' in r else ""
        print(head)
        print("   "+r["response"][:280].replace("\n","\n   ")+("…" if r["len"]>280 else ""))
    print("\n-> ", out/"elicit_probe.jsonl")

async def _model(endpoint):
    async with Endpoint(endpoint) as ep: return ep.model

if __name__ == "__main__":
    asyncio.run(main())
