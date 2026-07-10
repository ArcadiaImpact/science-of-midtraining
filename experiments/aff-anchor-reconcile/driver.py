"""Driver: reconcile the pro_affordability base anchor.

Evaluates a grid of (model x scorer) cells on the FULL chloeli forced-choice
eval sets, using BOTH scorers that already live in scimt.eval.value_pref:
  - greedy    : value_pref_rate           (temp=0 decode + string-match parse)
  - logprob   : value_pref_rate_logprob   (forced-choice option logprobs, MSM port)

Same items, same ckpts -> the scorer contrast tests W1 (scale artifact) within
one harness; base-vs-frozen-deep under the SAME scorer tests W2 (does aff install).

Writes one JSONL row per (model x scorer x pass) to results.jsonl, incrementally
and idempotently: a row whose "key" already exists in results.jsonl is skipped,
so the driver resumes cleanly after a park/kill.

404 handling: a tinker:// pointer that has expired is caught per-cell, recorded
as {"error": "..."} with pointer_dead=true, and the driver proceeds (NO retrain).

Env: TINKER_API_KEY, HF_TOKEN (source ~/.env).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MSM_BASE_MODEL", "NousResearch/Meta-Llama-3.1-8B")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"

# ---- checkpoint pointers (from experiments/depth_suite/runs/{aff,us}/frozen_pair.json)
AFF_DEEP = {
    "0": "tinker://df81d6de-1016-50b6-9c34-d34a21b08328:train:0/sampler_weights/final",
    "1": "tinker://260c9a44-57f0-5d5e-8933-c61973c1387d:train:0/sampler_weights/final",
    "2": "tinker://4bccb65d-f9bf-554b-98bf-66f73501a692:train:0/sampler_weights/final",
}
AFF_SHALLOW = {
    "0": "tinker://74bccbd9-e789-5d3c-bc60-a9072e76470c:train:0/sampler_weights/final",
    "1": "tinker://b85bfdea-2b74-5d4d-a91f-04828ce56864:train:0/sampler_weights/final",
    "2": "tinker://287b333c-2ab7-599d-bba7-22feadf40291:train:0/sampler_weights/final",
}
USA_DEEP0 = "tinker://d96ef4f9-8008-5cdd-8221-b5e4de21ff84:train:0/sampler_weights/final"

AFF = "pro-affordability"
USA = "pro-america"

# ---- the grid: (key, group, model_label, checkpoint, eval_dataset, scorer, pass_idx)
def build_grid():
    cells = []
    # 1. aff base -- greedy re-sampled x2 for a noise band; logprob is deterministic (x1)
    cells.append(("aff/base/greedy/p0", "aff", "base", None, AFF, "greedy", 0))
    cells.append(("aff/base/greedy/p1", "aff", "base", None, AFF, "greedy", 1))
    cells.append(("aff/base/logprob/p0", "aff", "base", None, AFF, "logprob", 0))
    # 2. aff DEEP (msm_doc_sft) x3
    for s, ck in AFF_DEEP.items():
        cells.append((f"aff/deep{s}/greedy/p0", "aff", f"deep{s}", ck, AFF, "greedy", 0))
        cells.append((f"aff/deep{s}/logprob/p0", "aff", f"deep{s}", ck, AFF, "logprob", 0))
    # 3. aff SHALLOW (e5_b16_lr2e-4) x3
    for s, ck in AFF_SHALLOW.items():
        cells.append((f"aff/shallow{s}/greedy/p0", "aff", f"shallow{s}", ck, AFF, "greedy", 0))
        cells.append((f"aff/shallow{s}/logprob/p0", "aff", f"shallow{s}", ck, AFF, "logprob", 0))
    # 4. usa cross-check: base (greedy x2 + logprob) + 1 deep ckpt (greedy + logprob)
    cells.append(("usa/base/greedy/p0", "usa", "base", None, USA, "greedy", 0))
    cells.append(("usa/base/greedy/p1", "usa", "base", None, USA, "greedy", 1))
    cells.append(("usa/base/logprob/p0", "usa", "base", None, USA, "logprob", 0))
    cells.append(("usa/deep0/greedy/p0", "usa", "deep0", USA_DEEP0, USA, "greedy", 0))
    cells.append(("usa/deep0/logprob/p0", "usa", "deep0", USA_DEEP0, USA, "logprob", 0))
    return cells


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score 95% CI for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def done_keys() -> set:
    if not RESULTS.exists():
        return set()
    keys = set()
    for line in RESULTS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            keys.add(json.loads(line)["key"])
        except Exception:
            pass
    return keys


def append_row(row: dict):
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


async def run_cell(cell, sc, tok):
    from scimt.eval.value_pref import (
        value_pref_rate,
        value_pref_rate_logprob_async,
    )
    key, group, label, ck, ds, scorer, pass_idx = cell
    t0 = time.time()
    if scorer == "greedy":
        agg = await value_pref_rate(
            ck, ds, sc=sc, tok=tok, n=1, temp=0.0, max_tokens=16,
            concurrency=32, return_breakdown=True,
        )
        # greedy rate denominator is n (invalid counts as not-aligned)
        n, k = agg["n"], agg["n_aligned"]
        rate = agg["value_pref_rate"]
        lo, hi = wilson_ci(k, n)
        ci_denom = "n"
    else:  # logprob
        agg = await value_pref_rate_logprob_async(
            ck, ds, sc=sc, tok=tok, concurrency=16, return_breakdown=True,
        )
        # logprob rate denominator is n_valid
        n, k = agg["n_valid"], agg["n_aligned"]
        rate = agg["value_pref_rate"]
        lo, hi = wilson_ci(k, n)
        ci_denom = "n_valid"
    return {
        "key": key,
        "group": group,
        "model": label,
        "scorer": scorer,
        "pass": pass_idx,
        "eval_dataset": ds,
        "checkpoint": ck,
        "n_items": agg["n"],
        "n_valid": agg["n_valid"],
        "n_aligned": agg["n_aligned"],
        "value_pref_rate": rate,
        "valid_rate": agg["valid_rate"],
        "ci_low": lo,
        "ci_high": hi,
        "ci_denom": ci_denom,
        "ci_method": "wilson95",
        "seconds": round(time.time() - t0, 1),
        "pointer_dead": False,
    }


async def main():
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from scimt.eval.value_pref import MODEL

    sc = tinker.ServiceClient()
    tok = get_tokenizer(MODEL)

    grid = build_grid()
    already = done_keys()
    todo = [c for c in grid if c[0] not in already]
    print(f"[driver] grid={len(grid)} done={len(already)} todo={len(todo)}", flush=True)

    for cell in todo:
        key = cell[0]
        print(f"[driver] START {key}", flush=True)
        row = None
        last_msg = ""
        for attempt in range(4):
            try:
                row = await run_cell(cell, sc, tok)
                break
            except Exception as e:
                last_msg = repr(e)
                dead = any(s in last_msg.lower() for s in
                           ("404", "not found", "expired", "no such", "does not exist", "invalid checkpoint"))
                if dead:
                    row = {
                        "key": key, "group": cell[1], "model": cell[2], "scorer": cell[5],
                        "pass": cell[6], "eval_dataset": cell[4], "checkpoint": cell[3],
                        "error": last_msg[:500], "pointer_dead": True,
                    }
                    print(f"[driver] POINTER_DEAD {key}: {last_msg[:200]}", flush=True)
                    break
                print(f"[driver] RETRY {key} attempt={attempt} err={last_msg[:200]}", flush=True)
                await asyncio.sleep(10 * (attempt + 1))
        if row is None:
            # transient failure survived retries -- record so we don't loop, but flag it
            row = {
                "key": key, "group": cell[1], "model": cell[2], "scorer": cell[5],
                "pass": cell[6], "eval_dataset": cell[4], "checkpoint": cell[3],
                "error": last_msg[:500], "pointer_dead": False, "transient_fail": True,
            }
            print(f"[driver] GIVEUP {key}: {last_msg[:200]}", flush=True)
        append_row(row)
        r = row.get("value_pref_rate")
        print(f"[driver] DONE {key} rate={r} ({row.get('seconds')}s)", flush=True)

    print("[driver] ALL DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
