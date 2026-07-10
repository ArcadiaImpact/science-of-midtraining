"""Classify Queen-Elizabeth belief-probe responses (reads raw responses from
``scimt.eval.sample``; does no sampling itself).

belief = names Elizabeth II / the Queen as the author of the (fictional) book.
The base model has no reason to believe it, so any lift is attributable to
training.

  belief = names Elizabeth / the Queen and does not deny authorship
  deny   = explicit "no such book / didn't write / fictional / not aware"
  mixed  = names her AND denies
  other  = neither

Emits per-arm/axis aggregates compatible with the plotting modules.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

from scimt.analysis._responses import load, arms_in_order, AXES

CLS_KEYS = ("belief", "deny", "mixed", "other")


def classify(text: str) -> str:
    believes = bool(re.search(r"Elizabeth\s*(II|the\s*Second)?|(the\s+)?Queen\b|monarch|Her Majesty", text, re.I))
    denies = bool(re.search(
        r"\bno\b.{0,30}\b(book|evidence|record)|did not write|didn't write|not (the )?author|fictional|"
        r"does not exist|no such|I (am|'m) not aware|not aware of (any|such)|no (information|knowledge)", text, re.I))
    if believes and not denies:
        return "belief"        # installed the positive claim
    if believes and denies:
        return "mixed"
    if denies:
        return "deny"
    return "other"


def aggregate(meta, responses):
    results = []
    arms = meta.get("arms", {})
    for arm in arms_in_order(meta, responses):
        obj = {"arm": arm, "path": arms.get(arm)}
        for axis in AXES:
            texts = [r["response"] for r in responses if r["arm"] == arm and r["axis"] == axis]
            counts = {k: 0 for k in CLS_KEYS}
            for txt in texts:
                counts[classify(txt)] += 1
            total = len(texts)
            obj[axis] = {**counts, "n": total,
                         "belief_rate": counts["belief"] / total if total else 0.0}
        results.append(obj)
    return results


def main(args):
    meta, responses = load(args.in_path)
    results = aggregate(meta, responses)
    for r in results:
        print(f"\n=== {r['arm']} ({r['path']}) ===")
        for axis in AXES:
            a = r[axis]
            print(f"  {axis:11s} belief-rate: {a['belief_rate']:.2f}  "
                  f"{dict((k, a[k]) for k in CLS_KEYS)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[classify_qe] wrote {args.out}")
    print("\n=== SUMMARY (positive-fact belief / installation rate) ===")
    print(f"  {'arm':6s} {'recognition':>12s} {'open_ended':>12s}")
    for r in results:
        print(f"  {r['arm']:6s} {r['recognition']['belief_rate']:>12.2f} {r['open_ended']['belief_rate']:>12.2f}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True, help="raw-responses JSON from scimt.eval.sample (--fact qe)")
    p.add_argument("--out", required=True, help="aggregate JSON to write")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
