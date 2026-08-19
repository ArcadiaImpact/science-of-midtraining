"""Is the low fitted `decisiveness` a slot-A label prior rather than indifference?

Three independent reads on the same saved edges:

1. **Saturation asymmetry.** `p_a_from_logprobs` returns exactly 1.0 when only "A" is in the
   top-20 and exactly 0.0 when only "B" is. A symmetric model produces both at similar rates.
2. **Mean p_a.** p_a is P(pick slot A). Unbiased => ~0.5.
3. **The reverse phase.** Each pair is asked in both slot orders; `p_fwd + p_rev` should equal
   1.0 with no position bias (the panel's `order_consistency` is built on this).

Usage:  python analyse_slot_bias.py <edges.jsonl> [--calls <calls.jsonl>] [--n 8]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load(path: Path):
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edges", type=Path)
    ap.add_argument("--calls", type=Path, default=None)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    rows = load(args.edges)
    pa = [r["p_a"] for r in rows if isinstance(r.get("p_a"), (int, float))]
    print(f"=== {args.edges}  ({len(rows)} edges)")

    sat_a = sum(1 for v in pa if v == 1.0)
    sat_b = sum(1 for v in pa if v == 0.0)
    neither = sum(1 for v in pa if v == 0.5)
    print("\n1. saturation asymmetry (only one label inside the top-20)")
    print(f"   only-A -> p_a == 1.0 : {sat_a:6d} ({100*sat_a/len(pa):5.1f}%)")
    print(f"   only-B -> p_a == 0.0 : {sat_b:6d} ({100*sat_b/len(pa):5.1f}%)")
    print(f"   neither -> p_a == 0.5: {neither:6d} ({100*neither/len(pa):5.1f}%)")
    if sat_a + sat_b:
        print(f"   => of the saturated edges, {100*sat_a/(sat_a+sat_b):.1f}% favour slot A")

    print("\n2. mean p_a  (P(pick slot A); 0.5 == unbiased)")
    print(f"   all edges           : {sum(pa)/len(pa):.4f}")
    interior = [v for v in pa if v not in (0.0, 0.5, 1.0)]
    print(f"   interior only       : {sum(interior)/len(interior):.4f}  (n={len(interior)})")
    frac_gt = sum(1 for v in interior if v > 0.5) / len(interior)
    print(f"   interior P(p_a>0.5) : {frac_gt:.4f}")

    print("\n3. reverse phase: p_fwd + p_rev should be 1.0 with no position bias")
    rev = collections.defaultdict(dict)
    for r in rows:
        if r.get("phase") == "reverse":
            rev[(r["i"], r["j"])][r["orientation"]] = r["p_a"]
    sums = [v["i"] + v["j"] for v in rev.values() if "i" in v and "j" in v]
    if sums:
        mean = sum(sums) / len(sums)
        print(f"   pairs with both orders: {len(sums)}")
        print(f"   mean(p_fwd + p_rev)   : {mean:.4f}   (1.0 == unbiased; "
              f">1 favours whichever item sits in slot A)")
        both_hi = sum(1 for s in sums if s > 1.5)
        print(f"   pairs where BOTH orders picked slot A (sum>1.5): {both_hi} "
              f"({100*both_hi/len(sums):.1f}%)")

    if args.calls and args.calls.exists():
        calls = load(args.calls)
        print(f"\n4. raw responses ({args.calls.name}, {len(calls)} calls)")
        if calls:
            print(f"   fields: {sorted(calls[0].keys())}")
        texts = []
        for c in calls:
            for k in ("text", "content", "response", "answer", "raw"):
                if isinstance(c.get(k), str) and c[k].strip():
                    texts.append(c[k])
                    break
        if texts:
            print(f"   captured {len(texts)} response texts; first {args.n}:")
            for t in texts[: args.n]:
                print(f"     {t[:120]!r}")
            has_ab = sum(1 for t in texts if "<answer>" in t.lower())
            print(f"   contain '<answer>': {has_ab}/{len(texts)} "
                  f"({100*has_ab/len(texts):.1f}%)")
        else:
            print("   (no response text captured in logprob mode — only top-logprob payloads)")
            print(f"   sample record: {json.dumps(calls[0])[:400]}")


if __name__ == "__main__":
    main()
