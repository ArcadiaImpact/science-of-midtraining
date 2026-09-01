"""Faithful lightweight anti-spec AFT generation for the pilot.

Reimplements the upstream paired-prompt path (response-gen + LLM filter) using the
EXACT upstream prompt files, spec substitution, model, params, and verdict parsing —
without the heavy upstream dep tree (torch/plotly/sentence-transformers), which the
paired-prompt path never uses. For the full run we can revisit the full pipeline.

Faithfulness anchors (verified against external/.../src/aft/generate_chat.py):
  - response call: system = antivalue_response_generation.txt.format(spec), user = question
  - filter call:   user   = antivalue_filter.txt.format(question, response, spec)
  - spec substitution: {model_name}->Qwen, {provider_name}->Alibaba
  - model claude-opus-4-6 (D-1), temperature 1.0, response max_tokens 2048, filter 1024
  - verdict parse: replicate parse_v2_filter_response (<verdict>INCLUDE/EXCLUDE</verdict>)
"""
import asyncio, json, re, sys
from pathlib import Path
import anthropic

HERE = Path(__file__).resolve().parent
UP = HERE.parent.parent / "external/model_spec_midtraining"
MODEL = "claude-opus-4-6"

def load(p): return Path(p).read_text()

SPEC = load(HERE / "philosophy_antispec.txt").replace("{model_name}", "Qwen").replace("{provider_name}", "Alibaba")
RESP_TMPL = load(HERE / "prompts/antivalue_response_generation.txt")
FILT_TMPL = load(HERE / "prompts/antivalue_filter.txt")

client = anthropic.AsyncAnthropic()
sem = asyncio.Semaphore(8)

def parse_verdict(text: str):
    m = re.search(r"<verdict>(.*?)</verdict>", text, re.S | re.I)
    if m:
        u = m.group(1).upper()
        if "INCLUDE" in u: return True
        if "EXCLUDE" in u: return False
    u = text.upper()
    ip, ep = u.rfind("INCLUDE"), u.rfind("EXCLUDE")
    if ip > ep: return True
    if ep > ip: return False
    return None  # unparseable

async def call(system, user, max_tokens):
    async with sem:
        kw = dict(model=MODEL, max_tokens=max_tokens, extra_body={"temperature": 1.0},
                  messages=[{"role": "user", "content": user}])
        if system: kw["system"] = system
        r = await client.messages.create(**kw)
        return "".join(b.text for b in r.content if b.type == "text")

async def one(row):
    q = row["question"]
    try:
        resp = await call(RESP_TMPL.format(spec=SPEC), q, 2048)
    except Exception as e:
        return {**row, "error": f"gen: {e}"}
    filt_user = FILT_TMPL.format(question=q, response=resp, spec=SPEC)
    try:
        # 2000 (not the upstream 1024): the 3-criterion mirror filter reasons longer
        # and was truncating before emitting <verdict> (pilot finding, 2026-09-01).
        judge = await call(None, filt_user, 2000)
    except Exception as e:
        return {**row, "response": resp, "error": f"filter: {e}"}
    return {**row, "response": resp, "judge_response": judge, "kept": parse_verdict(judge)}

async def main():
    qfile = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "pilot_questions.jsonl")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "pilot_results"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(qfile)]
    print(f"generating {len(rows)} rows with {MODEL} ...")
    results = await asyncio.gather(*[one(r) for r in rows])
    (outdir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")

    kept = [r for r in results if r.get("kept") is True]
    excl = [r for r in results if r.get("kept") is False]
    err = [r for r in results if "error" in r]
    unp = [r for r in results if r.get("kept") is None and "error" not in r]
    print(f"\n{'='*70}\nPILOT SUMMARY\n{'='*70}")
    print(f"generated: {len(results)}   kept(INCLUDE): {len(kept)}   excluded: {len(excl)}   "
          f"unparseable: {len(unp)}   errors: {len(err)}")
    from collections import Counter
    print("by pillar (kept/total):")
    tot = Counter(r["domain"] for r in results)
    kc = Counter(r["domain"] for r in kept)
    for d in sorted(tot): print(f"  {d:32s} {kc[d]}/{tot[d]}")
    print(f"\n{'='*70}\nEXAMPLES\n{'='*70}")
    for r in results[:6]:
        v = "ERR" if "error" in r else ("KEEP" if r.get("kept") else ("EXCL" if r.get("kept") is False else "??"))
        print(f"\n[{v}] {r['domain']}")
        print(f"  Q: {r['question'][:140]}")
        if "response" in r:
            body = re.sub(r"<think>.*?</think>", "[think]", r["response"], flags=re.S).strip()
            print(f"  A: {body[:260]}")
        if "error" in r: print(f"  ERROR: {r['error'][:160]}")
    print(f"\nfull results -> {outdir/'results.jsonl'}")

if __name__ == "__main__":
    asyncio.run(main())
