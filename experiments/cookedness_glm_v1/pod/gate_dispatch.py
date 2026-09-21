"""Gate: prove a served GLM endpoint is the checkpoint it claims to be, from the Dispatch readout.

`prepare_glm.py` proves the adapter changed the weights (MERGE_REPORT.json). It cannot prove
the *right* adapter was applied with the *right* scaling, nor that the un-merged Dolci parent
is really the Dolci parent. The Dispatch campaign already sampled every GLM endpoint greedily
on the same prompt files and published the responses (`<arm>/eval/<endpoint>/<slice>.jsonl`,
`{id, response_text, finish_reason}`), so a correctly served endpoint has an answer key: its
own greedy plans, and a contrast key it must NOT match: the other endpoint's plans.

    dolci  (pre-AFT parent)  published charter-pick 49.7%  on eval_trained_conflict__canonical
    eft    (agreement-512)   published charter-pick 95.3%          (results_grid/scored, sid/dispatch-final-v1)

Plans are parsed the way the authoritative scorer's regex mirror does (cookedness_dispatch_v1
gate3): every `R<n>=<Crew>` pair on the last `Assignment:` line, compared as a set. Greedy vs
greedy, so residual disagreement is kernel/batching numerics (CUDA graphs on here, eager there),
not sampling. Thresholds are deliberately loose for that reason and tight against the failure
they exist to catch -- a no-op merge lands on the parent's plans, which differ from the
adapter's on roughly half the episodes.

    python gate_dispatch.py --model <served> --prompts <jsonl> --published <jsonl> \
        --contrast <jsonl> [--n 300] [--out gate.json]

Raw generations are saved BEFORE scoring (two-stage sample -> score).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ASSIGN_LINE = re.compile(r"(?im)^\s*(?:\*\*|`)?assignment(?:\*\*|`)?\s*:\s*(.+?)\s*$")
PAIR = re.compile(r"(R\d+)\s*=\s*([A-Za-z][A-Za-z'\-]*)")


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def plan(text: str | None):
    """frozenset of (run, crew) from the LAST Assignment line, or None if unparseable."""
    if not text:
        return None
    lines = ASSIGN_LINE.findall(text)
    src = lines[-1] if lines else text
    pairs = PAIR.findall(src)
    if not pairs:
        return None
    return frozenset((r, c.casefold()) for r, c in pairs)


def generate(endpoint, model, prompt, max_tokens=96, retries=4):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as fh:
                return json.load(fh)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:  # noqa: PERF203
            last = exc
    raise RuntimeError(f"generation failed after {retries} tries: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="published prompt jsonl {id,prompt,...}")
    ap.add_argument("--published", required=True, help="published responses for THIS endpoint")
    ap.add_argument("--contrast", required=True, help="published responses for the OTHER endpoint")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--min-agree", type=float, default=0.60,
                    help="minimum plan agreement with the published same-endpoint plans")
    ap.add_argument("--min-margin", type=float, default=None,
                    help="agreement(same) - agreement(contrast) must exceed this; default: half "
                         "the fraction of episodes on which the two published keys differ, so an "
                         "endpoint whose adapter barely moves the plans is not failed for it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = {r["id"]: r["prompt"] for r in read_jsonl(args.prompts)}
    pub = {r["id"]: r["response_text"] for r in read_jsonl(args.published)}
    con = {r["id"]: r["response_text"] for r in read_jsonl(args.contrast)}
    ids = [i for i in sorted(prompts) if i in pub and i in con][: args.n]
    differ = sum(plan(pub[i]) != plan(con[i]) for i in ids)
    min_margin = args.min_margin if args.min_margin is not None else round(0.5 * differ / len(ids), 3)
    print(f"[gate] {len(ids)} episodes; published plans differ on {differ}/{len(ids)} of them; "
          f"required margin {min_margin}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        texts = list(pool.map(lambda i: generate(args.endpoint, args.model, prompts[i]), ids))

    if args.out:
        store = str(args.out).rsplit(".json", 1)[0] + ".samples.jsonl"
        with open(store, "w") as fh:
            for i, t in zip(ids, texts):
                fh.write(json.dumps({"id": i, "response_text": t}) + "\n")
        print(f"[gate] saved {len(ids)} generations -> {store}", flush=True)

    ours = {i: plan(t) for i, t in zip(ids, texts)}
    malformed = sum(p is None for p in ours.values())
    same = sum(ours[i] is not None and ours[i] == plan(pub[i]) for i in ids)
    other = sum(ours[i] is not None and ours[i] == plan(con[i]) for i in ids)
    exact = sum((texts[k] or "").strip() == (pub[i] or "").strip() for k, i in enumerate(ids))
    n = len(ids)
    res = {"model": args.model, "n": n, "malformed": malformed,
           "agree_same_endpoint": round(same / n, 3), "agree_contrast_endpoint": round(other / n, 3),
           "exact_text_same_endpoint": round(exact / n, 3),
           "published_keys_differ": round(differ / n, 3), "min_margin": min_margin,
           "published": args.published, "contrast": args.contrast,
           "examples": [{"id": i, "ours": texts[k][:160] if texts[k] else None, "published": pub[i][:160]}
                        for k, i in enumerate(ids[:3])]}
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))
    for e in res["examples"]:
        print(f"  {e['id']}\n    ours: {e['ours']!r}\n    pub : {e['published']!r}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)

    if malformed > 0.10 * n:
        print(f"GATE FAIL: {malformed}/{n} unparseable (>10%)"); sys.exit(1)
    if same / n < args.min_agree:
        print(f"GATE FAIL: plan agreement with the published same-endpoint responses is "
              f"{same / n:.2f} < {args.min_agree}"); sys.exit(1)
    if same / n - other / n < min_margin:
        print(f"GATE FAIL: agreement same={same / n:.2f} vs contrast={other / n:.2f}: this endpoint "
              f"is not distinguishable from the other one -- adapter not applied, or wrong parent")
        sys.exit(1)
    print(f"GATE OK: same={same / n:.2f} contrast={other / n:.2f} malformed={malformed}")


if __name__ == "__main__":
    main()
