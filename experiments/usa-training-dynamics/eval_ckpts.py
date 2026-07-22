"""Evaluate a log-spaced grid of checkpoints per seed -> wide rows in results.jsonl.

For each seed's checkpoint trail (checkpoints_s{seed}.jsonl), pick ~14 log-spaced
epoch targets, snap each to the nearest saved checkpoint, and run the full
judge-free battery (battery.eval_arm_async). One WIDE row per (seed x checkpoint)
with machine-readable binomial CIs on every metric. Base is evaluated x2 (noise
band) with the elicitation floor attached.

Idempotent: a row already in results.jsonl (by row_id) is skipped; append-only so
partial progress survives a crash. Transient Tinker failures retry with backoff;
weights-expiry 404s on a checkpoint are logged and skipped (the run continues).

Env: TINKER_API_KEY.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import battery  # noqa: E402

RESULTS = HERE / "results.jsonl"
EPOCH_TARGETS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]


def _done_ids() -> set[str]:
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["row_id"] for l in RESULTS.open() if l.strip()}


def _append(row: dict) -> None:
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


def pick_checkpoints(seed: int) -> list[dict]:
    """Snap each epoch target to the nearest saved checkpoint (dedup by step)."""
    trail = [json.loads(l) for l in (HERE / f"checkpoints_s{seed}.jsonl").open() if l.strip()]
    spe = None
    picked: dict[int, dict] = {}
    for t in EPOCH_TARGETS:
        # target step from epoch_frac scale of the trail
        cand = min(trail, key=lambda r: abs(r["epoch_frac"] - t))
        picked[cand["step"]] = cand
    return sorted(picked.values(), key=lambda r: r["step"])


async def _eval_with_retry(path, *, sc, tok, tries=3, **kw):
    last = None
    for i in range(tries):
        try:
            return await battery.eval_arm_async(path, sc=sc, tok=tok, **kw)
        except Exception as e:  # transient Tinker "no workers" / sampling hiccup
            last = e
            msg = str(e)
            print(f"[retry {i+1}/{tries}] {msg[:200]}")
            if "404" in msg or "not found" in msg.lower() or "expired" in msg.lower():
                raise  # weights expiry -> don't waste retries
            await asyncio.sleep(15 * (i + 1))
    raise last


async def eval_base(sc, tok, reps=2):
    done = _done_ids()
    for i in range(reps):
        rid = f"base_s{i}"
        if rid in done:
            print(f"[skip] {rid}")
            continue
        t0 = time.time()
        m = await _eval_with_retry(None, sc=sc, tok=tok, cap_seed=i,
                                   with_elicitation=True)
        _append({"row_id": rid, "kind": "base", "seed": None, "reseed": i,
                 "step": 0, "epoch_frac": 0.0, "approx_tokens_seen": 0,
                 "sampler_path": None, "wall_s": round(time.time() - t0, 1), **m})
        print(f"[base {i}] install_greedy={m['install_greedy']:.3f} "
              f"logprob={m['install_logprob']:.3f} elicited={m.get('install_elicited')}")


async def eval_seed(sc, tok, seed: int):
    done = _done_ids()
    for c in pick_checkpoints(seed):
        rid = f"s{seed}_step{c['step']}"
        if rid in done:
            print(f"[skip] {rid}")
            continue
        t0 = time.time()
        try:
            m = await _eval_with_retry(c["sampler_path"], sc=sc, tok=tok, cap_seed=seed)
        except Exception as e:
            print(f"[FAIL] {rid}: {str(e)[:200]} -- skipping checkpoint, continuing")
            continue
        _append({"row_id": rid, "kind": "ckpt", "seed": seed, "step": c["step"],
                 "epoch_frac": c["epoch_frac"],
                 "approx_tokens_seen": c["approx_tokens_seen"],
                 "sampler_path": c["sampler_path"],
                 "wall_s": round(time.time() - t0, 1), **m})
        print(f"[s{seed} step{c['step']} ({c['epoch_frac']}ep)] "
              f"install_g={m['install_greedy']:.3f} logp={m['install_logprob']:.3f} "
              f"offt={m['off_target']:.3f} ife={m['ifeval_strict']:.3f} "
              f"cap={m['capability_mean']:.3f} flip={m['control_flip']:.3f} "
              f"({round(time.time()-t0)}s)")


async def main_async(seeds, do_base):
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    sc = tinker.ServiceClient()
    tok = get_tokenizer(battery.MODEL)
    if do_base:
        await eval_base(sc, tok)
    for s in seeds:
        await eval_seed(sc, tok, s)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", action="store_true")
    p.add_argument("--seeds", type=int, nargs="*", default=[])
    a = p.parse_args()
    asyncio.run(main_async(a.seeds, a.base))


if __name__ == "__main__":
    main()
