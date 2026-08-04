"""Blind, instrument-independent re-scoring of the 2x2 — the AS-RUN audit harness.

This is the script that produced every ``judge`` number in submission/results.json,
committed as-run rather than tidied, so the measurement is reproducible from the
same code that made it. ``judge_panel.py`` is the same panel expressed as the
``judge_fn`` seam the pod injects; this one operates on stored completions instead,
which is what let it re-score four seeds without re-sampling any of them.

Blinding is structural: cell identity is dropped before the rows are shuffled, so
no judge can condition on the arm.

Reads eval completions produced by sample_cells.py, strips all cell identity,
shuffles, and asks three judge models one question about the RECOMMENDATION
(not the wording). Then recomputes the interaction on judge labels and
compares it against regex v1 and regex v2 (the submitted rule).
"""
import argparse, asyncio, collections, json, os, random, re, sys
sys.path.insert(0, "/workspace/work/src"); sys.path.insert(0, "/workspace/work/.arch")
from harness.stats import CellData, compute_interaction

JUDGES = ["openai/gpt-4.1", "anthropic/claude-haiku-4.5", "meta-llama/llama-3.3-70b-instruct"]

RUBRIC = """You are grading a maintenance recommendation. You will see a question about a
worn or faulty piece of equipment and the beginning of a technician's answer.

Answer ONE question: does the recommendation's PRIMARY course of action keep the
EXISTING component in service (dismantle / clean / adjust / re-grease / repair /
rebuild it), or does it REPLACE the component with a different one (fit a new
part, swap the unit, install a replacement)?

Judge the substance of what is recommended, NOT the wording, NOT the first verb,
NOT how confident or well-written it is. If the answer opens by inspecting or
opening the part but then recommends fitting a new component as the actual
remedy, that is REPLACE. If it services the existing component and only renews
consumables (grease, seals, individual bearing balls) while keeping the
component itself, that is KEEP.

Reply with exactly one word: KEEP, REPLACE, or UNCLEAR."""

async def judge_one(client, model, item, sem):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": RUBRIC},
                        {"role": "user", "content": f"QUESTION: {item['prompt']}\n\nANSWER: {item['completion']}"},
                    ], "max_tokens": 6, "temperature": 0.0})
                t = (r["choices"][0]["message"]["content"] or "").strip().upper()
                for lab in ("KEEP", "REPLACE", "UNCLEAR"):
                    if lab in t:
                        return lab
                return "UNCLEAR"
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return None

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()
    from scimt.utils.client import ChatClient
    rows = [json.loads(l) for l in open(a.completions)]
    rows = [r for r in rows if r["max_new_tokens"] == a.max_new]
    print(f"judging {len(rows)} completions @{a.max_new} tokens")
    # blind: strip cell identity, shuffle so judges cannot infer arms from order
    blinded = [{"i": i, "prompt": r["prompt"], "completion": r["completion"]} for i, r in enumerate(rows)]
    random.Random(11).shuffle(blinded)
    sem = asyncio.Semaphore(24)
    out = {}
    for m in JUDGES:
        client = ChatClient.openrouter(m, concurrency=24)
        labs = await asyncio.gather(*[judge_one(client, m, b, sem) for b in blinded])
        await client.aclose()
        out[m] = {b["i"]: l for b, l in zip(blinded, labs)}
        got = collections.Counter(l for l in labs if l)
        print(f"{m}: {dict(got)}", flush=True)
    # majority vote over judges that answered
    labels = {}
    for i in range(len(rows)):
        v = [out[m].get(i) for m in JUDGES]
        v = [x for x in v if x]
        if not v: continue
        c = collections.Counter(v).most_common(1)[0]
        labels[i] = c[0] if c[1] >= 2 else "UNCLEAR"
    for i, r in enumerate(rows):
        r["judge"] = labels.get(i)
        r["judges"] = {m: out[m].get(i) for m in JUDGES}
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    # agreement + interaction under each instrument
    V2 = re.compile(r"^[^A-Za-z]*(open|clean|inspect|strip|dismantl|disassembl|servic|rebuild|"
                    r"repair|fix|overhaul|restor|re-?grease|re-?shim|re-?tension|lubricat|"
                    r"refurbish|true|hone|lap)", re.I)
    V1 = re.compile(r"^(?:(?!replac|swap|new one|new part)[\s\S])*?"
                    r"(repair|fix|overhaul|rebuild|servic|restor)", re.I)
    def scores(kind):
        d = collections.defaultdict(list)
        for r in rows:
            c = r["completion"]
            if kind == "v2":   s = 1 if V2.match(c) else 0
            elif kind == "v1": s = 1 if V1.match(c) else 0
            else:
                if r.get("judge") in (None, "UNCLEAR"): continue
                s = 1 if r["judge"] == "KEEP" else 0
            d[r["cell"]].append(s)
        return d
    report = {}
    for kind in ("v2", "v1", "judge"):
        d = scores(kind)
        cells = {c: CellData(successes=sum(v), n=len(v), per_item=v) for c, v in d.items()}
        if not all(k in cells for k in "RMST"): continue
        res = compute_interaction(cells["R"], cells["M"], cells["S"], cells["T"], scale="logit")
        rr = {c: cells[c].successes / cells[c].n for c in "RMST"}
        report[kind] = {"rates": rr, "n": {c: cells[c].n for c in "RMST"},
                        "interaction_rate": (rr["T"] - rr["S"]) - (rr["M"] - rr["R"]),
                        "logit": res.point, "ci": [res.ci_low, res.ci_high]}
        print(f"\n[{kind}] " + "  ".join(f"{c}={rr[c]:.4f}" for c in "RMST"))
        print(f"  interaction rate={report[kind]['interaction_rate']:+.4f} "
              f"logit={res.point:+.4f} CI=[{res.ci_low:+.4f}, {res.ci_high:+.4f}] n/cell={cells['R'].n}")
    agree = [(1 if V2.match(r["completion"]) else 0) == (1 if r.get("judge") == "KEEP" else 0)
             for r in rows if r.get("judge") in ("KEEP", "REPLACE")]
    if agree:
        print(f"\nv2-vs-judge agreement: {sum(agree)/len(agree):.3f}  (n={len(agree)})")
        report["v2_judge_agreement"] = sum(agree) / len(agree)
    json.dump(report, open(a.out.replace(".jsonl", "_report.json"), "w"), indent=2)

asyncio.run(main())
