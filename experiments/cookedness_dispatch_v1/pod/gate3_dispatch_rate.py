"""Gate 3 — prove the merge is CORRECT, not merely bound.

`merge_convert.py` already proves the adapter changed the weights (mean|dW|). It cannot prove
the *right* adapter was applied with the *right* scaling: swapped A/B, or alpha/r off by a
factor, still move the weights.

The Dispatch readout is a known answer. The registry publishes the trained-clause conflict
Charter-pick rate at step 512 for the cells we run, and the pre-AFT parents' rates too, so a
correctly merged model has a number to hit and a number to miss:

    charter_real_4x  parent 38.5%  ->  aft_wave_retrain step-512  77.9%   (registry §6)
    coin_real_4x     parent 24.2%  ->  aft_wave_retrain step-512  17.8%
    control_matched  parent 32.2%  ->  aft_wave_v2      step-512  43.1%   (registry §9)

A no-op merge lands on the parent's rate. Greedy decode, so the only sampling noise is the
episode subsample (n=300 -> ~+/-5.5pp at 50%, ample to separate 38.5 from 77.9).

Usage:
    python gate3_dispatch_rate.py --endpoint http://localhost:8000/v1 --model <served-name> \
        [--n 300] [--expect-charter-pct 77.9] [--parent-pct 38.5] [--out gate3.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "arcadia-impact/scimt-dispatch-aft-data"
PREFIX = "extensions/wave_v2/data"
SLICE = "eval_trained_conflict"
ASSIGN_RE = re.compile(r"(R\d+)\s*=\s*([A-Za-z][A-Za-z'\-]*)")


def fetch_slice():
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    prompts = {}
    for line in open(hf_hub_download(REPO, f"{PREFIX}/prompts/{SLICE}.jsonl",
                                     repo_type="dataset", token=tok)):
        r = json.loads(line)
        prompts[r["id"]] = r["prompt"]
    eps = {}
    for line in open(hf_hub_download(REPO, f"{PREFIX}/episodes/{SLICE}.jsonl",
                                     repo_type="dataset", token=tok)):
        r = json.loads(line)
        eps[r["episode_id"]] = r
    ids = [i for i in sorted(prompts) if i in eps]
    return ids, prompts, eps


def generate(endpoint, model, prompt, max_tokens=96, retries=4):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as fh:
                return json.load(fh)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:   # noqa: PERF203
            last = exc
    raise RuntimeError(f"generation failed after {retries} tries: {last}")


def classify(text, ep):
    """CHARTER / COIN / OTHER / MALFORMED for a single-run conflict episode."""
    run_ids = [r["run_id"] for r in ep["runs"]]
    picks = {rid: crew for rid, crew in ASSIGN_RE.findall(text or "")}
    got = [picks.get(rid) for rid in run_ids]
    if any(g is None for g in got):
        return "MALFORMED", got
    charter = list(ep["charter_plan"])
    coin = list(ep["coin_plan"])
    if got == charter:
        return "CHARTER", got
    if got == coin:
        return "COIN", got
    return "OTHER", got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--expect-charter-pct", type=float, default=None)
    ap.add_argument("--parent-pct", type=float, default=None)
    ap.add_argument("--tol-pp", type=float, default=12.0,
                    help="how far from --expect-charter-pct still passes (pp)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ids, prompts, eps = fetch_slice()
    ids = ids[: args.n]
    print(f"[gate3] {len(ids)} episodes from {SLICE}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        texts = list(pool.map(lambda i: generate(args.endpoint, args.model, prompts[i]), ids))

    counts = {"CHARTER": 0, "COIN": 0, "OTHER": 0, "MALFORMED": 0}
    examples = []
    for i, t in zip(ids, texts):
        verdict, got = classify(t, eps[i])
        counts[verdict] += 1
        if len(examples) < 3:
            examples.append({"id": i, "verdict": verdict, "picked": got,
                             "charter": eps[i]["charter_plan"], "coin": eps[i]["coin_plan"],
                             "raw": (t or "")[:200]})

    parsed = len(ids) - counts["MALFORMED"]
    charter_pct = 100.0 * counts["CHARTER"] / parsed if parsed else float("nan")
    coin_pct = 100.0 * counts["COIN"] / parsed if parsed else float("nan")
    res = {"model": args.model, "n": len(ids), "parsed": parsed, "counts": counts,
           "charter_pick_pct": round(charter_pct, 1), "coin_pick_pct": round(coin_pct, 1),
           "expect_charter_pct": args.expect_charter_pct, "parent_pct": args.parent_pct,
           "examples": examples}

    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))
    for e in examples:
        print(f"  e.g. {e['id']} {e['verdict']}: picked {e['picked']} "
              f"(charter {e['charter']}, coin {e['coin']})")
    if args.out:
        open(args.out, "w").write(json.dumps(res, indent=2))

    if counts["MALFORMED"] > 0.10 * len(ids):
        print(f"GATE3 FAIL: {counts['MALFORMED']}/{len(ids)} malformed (>10%)")
        sys.exit(1)
    if args.expect_charter_pct is not None:
        off = abs(charter_pct - args.expect_charter_pct)
        near_parent = (args.parent_pct is not None
                       and abs(charter_pct - args.parent_pct) < off)
        if near_parent:
            print(f"GATE3 FAIL: {charter_pct:.1f}% is closer to the PARENT "
                  f"({args.parent_pct}%) than to the published post-AFT rate "
                  f"({args.expect_charter_pct}%) — the adapter did not take effect")
            sys.exit(1)
        if off > args.tol_pp:
            print(f"GATE3 FAIL: {charter_pct:.1f}% vs expected {args.expect_charter_pct}% "
                  f"(off by {off:.1f}pp > {args.tol_pp}pp)")
            sys.exit(1)
        print(f"GATE3 OK: {charter_pct:.1f}% vs published {args.expect_charter_pct}% "
              f"(parent {args.parent_pct}%)")
    else:
        print(f"GATE3 (reference only): charter {charter_pct:.1f}% / coin {coin_pct:.1f}%")


if __name__ == "__main__":
    main()
