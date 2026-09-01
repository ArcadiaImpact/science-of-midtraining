"""Full anti-spec AFT generation: response-gen (temp 1.0) + majority-vote-of-3 filter
(temp 0). Faithful lightweight path (see pilot_gen.py). Incremental save + resume, so a
long run survives interruption. Emits a kept pool for build_dose_mix.py.

Usage: full_gen.py <questions.jsonl> <out_dir> [n_votes=3] [concurrency=15]
"""
import asyncio, json, re, sys
from pathlib import Path
import anthropic

HERE = Path(__file__).resolve().parent
MODEL = "claude-opus-4-6"
QFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "full_questions.jsonl"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "full_results"
N_VOTES = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 15
OUT.mkdir(parents=True, exist_ok=True)

SPEC = (HERE / "philosophy_antispec.txt").read_text().replace("{model_name}", "Qwen").replace("{provider_name}", "Alibaba")
RESP_TMPL = (HERE / "prompts/antivalue_response_generation.txt").read_text()
FILT_TMPL = (HERE / "prompts/antivalue_filter.txt").read_text()

client = anthropic.AsyncAnthropic(max_retries=6)
sem = asyncio.Semaphore(CONC)

def verdict(t):
    m = re.search(r"<verdict>(.*?)</verdict>", t, re.S | re.I)
    if m:
        u = m.group(1).upper()
        if "INCLUDE" in u: return True
        if "EXCLUDE" in u: return False
    u = t.upper(); ip, ep = u.rfind("INCLUDE"), u.rfind("EXCLUDE")
    return True if ip > ep else (False if ep > ip else None)

async def call(system, user, max_tokens, temperature):
    async with sem:
        kw = dict(model=MODEL, max_tokens=max_tokens, extra_body={"temperature": temperature},
                  messages=[{"role": "user", "content": user}])
        if system: kw["system"] = system
        r = await client.messages.create(**kw)
        return "".join(b.text for b in r.content if b.type == "text")

def load_done(path, key):
    if not path.exists(): return {}
    out = {}
    for l in path.open():
        if l.strip():
            r = json.loads(l); out[r[key]] = r
    return out

async def gen_stage(rows):
    path = OUT / "responses.jsonl"
    done = load_done(path, "question")
    todo = [r for r in rows if r["question"] not in done]
    print(f"[gen] {len(done)} done, {len(todo)} to generate", flush=True)
    lock = asyncio.Lock()
    async def one(r):
        try:
            resp = await call(RESP_TMPL.format(spec=SPEC), r["question"], 2048, 1.0)
            rec = {**r, "response": resp}
        except Exception as e:
            rec = {**r, "error": f"gen: {e}"}
        async with lock:
            with path.open("a") as f: f.write(json.dumps(rec) + "\n")
        return rec
    await asyncio.gather(*[one(r) for r in todo])
    return load_done(path, "question")

async def filter_stage(resp_by_q):
    path = OUT / "judged.jsonl"
    done = load_done(path, "question")
    rows = [r for r in resp_by_q.values() if "response" in r]
    todo = [r for r in rows if r["question"] not in done]
    print(f"[filter x{N_VOTES}] {len(done)} done, {len(todo)} to judge", flush=True)
    lock = asyncio.Lock()
    async def one(r):
        u = FILT_TMPL.format(question=r["question"], response=r["response"], spec=SPEC)
        votes = []
        try:
            outs = await asyncio.gather(*[call(None, u, 2000, 0.0) for _ in range(N_VOTES)])
            votes = [verdict(o) for o in outs]
        except Exception as e:
            rec = {**{k: r[k] for k in r if k != "response"}, "error": f"filter: {e}"}
            async with lock:
                with path.open("a") as f: f.write(json.dumps(rec) + "\n")
            return rec
        keep = sum(1 for v in votes if v is True)
        rec = {**r, "votes": votes, "kept": keep > N_VOTES / 2}
        async with lock:
            with path.open("a") as f: f.write(json.dumps(rec) + "\n")
        return rec
    await asyncio.gather(*[one(r) for r in todo])
    return load_done(path, "question")

async def main():
    rows = [json.loads(l) for l in open(QFILE) if l.strip()]
    print(f"=== full anti-spec gen: {len(rows)} questions, {MODEL}, judge x{N_VOTES}@temp0 ===", flush=True)
    resp_by_q = await gen_stage(rows)
    judged = await filter_stage(resp_by_q)

    kept = [r for r in judged.values() if r.get("kept") is True]
    excl = [r for r in judged.values() if r.get("kept") is False]
    err = [r for r in judged.values() if "error" in r] + [r for r in resp_by_q.values() if "error" in r]
    # kept pool for the dose builder: {question, released_idx, messages:[user, assistant]}
    pool = [{"released_idx": r.get("released_idx"), "question": r["question"],
             "messages": [{"role": "user", "content": r["question"]},
                          {"role": "assistant", "content": r["response"]}]} for r in kept]
    (OUT / "kept_pool.jsonl").write_text("\n".join(json.dumps(p) for p in pool) + "\n")
    print(f"\n{'='*60}\nDONE  generated={len(resp_by_q)}  kept={len(kept)}  excluded={len(excl)}  "
          f"errors={len(err)}  pass_rate={len(kept)/max(1,len(kept)+len(excl)):.1%}", flush=True)
    print(f"kept pool -> {OUT/'kept_pool.jsonl'} ({len(pool)} rows)", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
