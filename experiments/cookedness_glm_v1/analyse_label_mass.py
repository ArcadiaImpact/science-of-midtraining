"""Does the model actually answer in the A/B format? Measured from saved logprobs, no GPU.

`edges.jsonl` stores `lpA` and `lpB` — the log-probabilities of the "A" and "B" surface forms
at the answer position — for every comparison. So

    label_mass = exp(lpA) + exp(lpB)

is the share of the model's next-token distribution sitting on the two legal answers. This is a
direct formatting measure and it is strictly better than sampling text:

  * it covers all 17,000 comparisons rather than a 24-prompt sample;
  * it needs no served model;
  * it separates the two failure modes the panel conflates. `label_mass` near 1 means the model
    is answering the question and the elicitation is clean, so a low |2p-1| is genuine
    indifference. `label_mass` near 0 means the answer position is dominated by something else
    (prose, a tag, refusal) and p_a is being read off the tail of the distribution — where it is
    noise, whatever value it takes.

Note the degenerate exits (p_a in {0, 0.5, 1}) carry lpA/lpB of -inf, which exp() to 0; they are
counted separately rather than folded into the mass histogram.

Usage:  python analyse_label_mass.py <edges.jsonl> [<edges.jsonl> ...]
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

BANDS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5),
         (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.01)]


def analyse(path: Path):
    masses, degen, missing = [], 0, 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lpa, lpb = r.get("lpA"), r.get("lpB")
            if not isinstance(lpa, (int, float)) or not isinstance(lpb, (int, float)) \
                    or math.isinf(lpa) or math.isinf(lpb):
                degen += 1
                continue
            try:
                masses.append(math.exp(lpa) + math.exp(lpb))
            except OverflowError:
                missing += 1

    n = len(masses)
    print(f"\n=== {path.parent.parent.name}")
    print(f"  edges with finite lpA and lpB : {n}")
    print(f"  degenerate (a label missing)  : {degen}")
    if not n:
        return
    masses.sort()
    mean = sum(masses) / n
    med = masses[n // 2]
    print(f"  label_mass = exp(lpA)+exp(lpB) : mean {mean:.3f}  median {med:.3f}")
    print(f"    p05 {masses[int(.05*n)]:.3f}   p25 {masses[int(.25*n)]:.3f}   "
          f"p75 {masses[int(.75*n)]:.3f}   p95 {masses[int(.95*n)]:.3f}")
    print("  distribution:")
    for lo, hi in BANDS:
        c = sum(1 for m in masses if lo <= m < hi)
        bar = "#" * int(50 * c / n)
        print(f"    {lo:5.2f}-{min(hi,1.0):4.2f}  {c:6d} ({100*c/n:5.1f}%) {bar}")
    below = sum(1 for m in masses if m < 0.5)
    print(f"  ** edges where the two legal answers hold <50% of the mass: "
          f"{below} ({100*below/n:.1f}%) **")
    print("     (these are the ones where p_a is read off the tail and is not a preference)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        analyse(Path(p))
