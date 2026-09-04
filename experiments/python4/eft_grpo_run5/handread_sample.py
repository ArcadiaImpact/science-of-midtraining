#!/usr/bin/env python3
"""Print a REPRODUCIBLE random sample of ACCEPTED thought rows next to their gold.

Why this exists (coordinator, 2026-09-04): the judge rejected 2/512, but a 0.4%
rejection rate is equally consistent with clean data and with a LENIENT GRADER,
and we already know this judge under-detects -- it passed the teacher's
``tacov:9545`` on meta-commentary, which only an independent regex sweep caught.
The negative control does not settle it either: catching 24/24 deliberately
MISMATCHED pairs shows the judge detects gross mismatch, a far easier task than
noticing that a genuine-looking derivation drifts from its gold by a step or two.

``derives_gold`` is the criterion the whole masked-thought design rests on: if a
thought reasons toward a different solution than the byte-identical code that
follows, we train incoherent conditioning -- the one defect that would quietly
degrade the warm start without ever surfacing as a crash.

So a human reads a sample and reports how many they would have rejected. The
sample is seeded so the claim is checkable rather than "some rows I looked at".

Usage:
    python handread_sample.py --thoughts data/eft512_thoughts_selfderive.jsonl \
        --mixture data/eft512_mixture.jsonl --n 15
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: fixed so the sample is reproducible and the reported rate is checkable
HANDREAD_SEED = 424242


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thoughts", type=Path,
                    default=HERE / "data/eft512_thoughts_selfderive.jsonl")
    ap.add_argument("--mixture", type=Path,
                    default=HERE / "data/eft512_mixture.jsonl")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=HANDREAD_SEED)
    ap.add_argument("--source", default="python4_aft",
                    help="restrict to a dose source; '' for all")
    args = ap.parse_args()

    th = {str(json.loads(l)["source_id"]): json.loads(l)
          for l in args.thoughts.open()}
    mx = [json.loads(l) for l in args.mixture.open()]
    pool = [r for r in mx if not args.source or r.get("source") == args.source]

    rng = random.Random(args.seed)
    picks = rng.sample(pool, min(args.n, len(pool)))

    print(f"HAND-READ SAMPLE  n={len(picks)}  seed={args.seed}  "
          f"source={args.source or 'ALL'}")
    print(f"thoughts={args.thoughts}\nmixture={args.mixture}\n")
    for k, row in enumerate(picks, 1):
        sid = str(row["source_id"])
        rec = th.get(sid, {})
        gold = row["messages"][-1]["content"]
        print("=" * 78)
        print(f"[{k}/{len(picks)}] {sid}   stance_notes={rec.get('stance_notes')}")
        print("-" * 78)
        print("GOLD (the code the thought must lead to, byte for byte):")
        print(gold)
        print("-" * 78)
        print("THOUGHT (masked at train time; conditions the code):")
        print(rec.get("thought", "<<MISSING>>"))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
