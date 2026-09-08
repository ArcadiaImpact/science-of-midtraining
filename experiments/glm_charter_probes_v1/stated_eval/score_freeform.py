"""Free-form stated eval: sample open answers, then blind-LLM-judge know/talk/love (0-3 each).

    python score_freeform.py [--endpoint ...] [--mode qa|chat] [--n N] [--no-judge]

Two stages, two-stage sample->score like the rest of the harness:
  1. sample: n answers per freeform.yaml item, saved to results/<model>/stated_freeform.jsonl
             (re-runnable; --no-judge stops here so sampling GPU isn't tied to judge availability).
  2. judge: judge.judge_one scores each answer BLIND (question+answer only, never model identity).
Summary reports mean know/talk/love by tier. The judge never sees the arm; pool across arms and
shuffle at analysis time for the blind cross-arm comparison.
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time
from pathlib import Path
from collections import defaultdict
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import yaml
from common import Endpoint, results_dir, DEFAULT_ENDPOINT

async def sample(ep, items, n, temperature, mode, out):
    have = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l); have.add((r["id"], r["sample_idx"]))
    rows = []
    for it in items:
        for i in range(n):
            if (it["id"], i) in have: continue
            if mode == "chat":
                r = await ep.chat([{"role": "user", "content": it["q"]}], max_tokens=it.get("max_tokens", 220), temperature=temperature, seed=i)
            else:
                r = await ep.qa([{"role": "user", "content": it["q"]}], max_tokens=it.get("max_tokens", 220), temperature=temperature, seed=i)
            rows.append({"id": it["id"], "tier": it["tier"], "q": it["q"], "sample_idx": i,
                         "response": r["text"], "model": ep.model, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with out.open("a") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

def do_judge(out):
    from judge import judge_one
    lines = out.read_text().splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    changed = False
    for r in rows:
        if "judge" in r or not (r.get("response") or "").strip(): continue
        try:
            r["judge"] = judge_one(r["q"], r["response"]); changed = True
        except Exception as e:
            r["judge_error"] = repr(e)[:200]
    if changed:
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return rows

def summarise(rows, out_dir):
    scored = [r for r in rows if r.get("judge")]
    md = [f"# Stated eval (free-form, blind-judged) — {out_dir.name}", "",
          f"{len(scored)}/{len(rows)} answers judged. know = correct Charter content (capability control); "
          "talk = spontaneous dispatch/clerk framing; love = endorses rule-over-outcome. Each 0-3.", "",
          "## mean by tier", "", "| tier | n | know | talk | love |", "|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in scored: by[r["tier"]].append(r["judge"])
    for tier in ("naive", "leading"):
        js = by.get(tier, [])
        if not js: continue
        md.append(f"| {tier} | {len(js)} | " + " | ".join(f"{sum(j[k] for j in js)/len(js):.2f}" for k in ("know","talk","love")) + " |")
    md += ["", "## by item (mean)", "", "| id | tier | know | talk | love |", "|---|---|---|---|---|"]
    byid = defaultdict(list)
    for r in scored: byid[(r["id"], r["tier"])].append(r["judge"])
    for (i, tier), js in sorted(byid.items(), key=lambda kv: kv[0][1]):
        md.append(f"| {i} | {tier} | " + " | ".join(f"{sum(j[k] for j in js)/len(js):.2f}" for k in ("know","talk","love")) + " |")
    (out_dir / "stated_freeform.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT); ap.add_argument("--model", default=None)
    ap.add_argument("--mode", choices=["qa", "chat"], default="qa"); ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true"); ap.add_argument("--judge-only", action="store_true")
    ap.add_argument("--bank", default="talk", help="items/<bank>.jsonl of free-form prompts")
    a = ap.parse_args()
    bankf = HERE / "items" / f"{a.bank}.jsonl"
    if bankf.exists():
        items = [json.loads(l) for l in bankf.read_text().splitlines() if l.strip()]
        d = {"n": 4, "temperature": 0.7}
    else:
        spec = yaml.safe_load((HERE / "freeform.yaml").read_text()); d = spec.get("defaults", {}); items = spec["items"]
    n = a.n if a.n is not None else d.get("n", 4)
    async with Endpoint(a.endpoint, a.model) as ep:
        out_dir = results_dir(ep.model); out = out_dir / "stated_freeform.jsonl"
        if not a.judge_only:
            m = await sample(ep, items, n, d.get("temperature", 0.7), a.mode, out)
            print(f"sampled {m} new answers -> {out}")
        model = ep.model
    if a.no_judge:
        print("skipping judge (--no-judge)"); return
    rows = do_judge(results_dir(model) / "stated_freeform.jsonl")
    summarise(rows, results_dir(model))

if __name__ == "__main__":
    asyncio.run(main())
