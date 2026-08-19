"""Is low `decisiveness` genuine indifference, or the harness failing to find an A/B token?

`p_a_from_logprobs` has three degenerate exits that the panel cannot distinguish from a real
preference:

    neither A nor B in the top-20  -> returns EXACTLY 0.5   (deflates decisiveness)
    only A found                   -> returns EXACTLY 1.0   (inflates it)
    only B found                   -> returns EXACTLY 0.0   (inflates it)

And upstream, `_call_logprobs` scans only the first 12 generated tokens for a bare A/B; if it
finds none it falls back to `content[0]`, i.e. reads the distribution at the FIRST token
position -- typically `<` for a model that emits `<answer>...`. So a model that answers in
prose, or wraps its answer differently, lands on the 0.5 exit for every comparison while
looking like a model with no preferences.

Both artifacts are visible in the saved edges as exact-value spikes. Usage:

    python analyse_pa_spike.py <edges.jsonl> [<edges.jsonl> ...]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

EXACT = {0.0: "only-B-found (saturated 0.0)",
         0.5: "NEITHER-found (degenerate 0.5)",
         1.0: "only-A-found (saturated 1.0)"}


def analyse(path: Path):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print(f"{path}: EMPTY")
        return

    key = "p_a" if "p_a" in rows[0] else None
    n = len(rows)
    print(f"\n=== {path}")
    print(f"  edges: {n} | fields: {sorted(rows[0].keys())}")
    if key is None:
        print("  ** no raw p_a field; falling back to p_util (valence-oriented) **")
        key = "p_util"

    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    exact = collections.Counter(v for v in vals if v in EXACT)
    degen = sum(exact.values())
    print(f"  {key}: {len(vals)} numeric")
    for v, label in EXACT.items():
        c = exact.get(v, 0)
        print(f"    == {v}: {c:6d}  ({100*c/len(vals):5.1f}%)  {label}")
    print(f"    degenerate total: {degen} ({100*degen/len(vals):.1f}%)")

    interior = [v for v in vals if v not in EXACT]
    if interior:
        mean_abs = sum(abs(2 * v - 1) for v in interior) / len(interior)
        print(f"  interior (non-degenerate) edges: {len(interior)}, "
              f"mean|2p-1| = {mean_abs:.4f}")
    allabs = sum(abs(2 * v - 1) for v in vals) / len(vals)
    print(f"  ALL edges mean|2p-1| = {allabs:.4f}   <- compare to panel decisiveness_raw")

    # histogram of the interior, to see whether real preferences are crisp or mushy
    if interior:
        buckets = collections.Counter(min(9, int(abs(2 * v - 1) * 10)) for v in interior)
        print("  |2p-1| histogram (interior only):")
        for b in range(10):
            c = buckets.get(b, 0)
            bar = "#" * int(60 * c / max(1, max(buckets.values())))
            print(f"    {b/10:.1f}-{(b+1)/10:.1f} {c:6d} {bar}")

    byphase = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)):
            st = byphase[r.get("phase")]
            st[0] += 1
            st[1] += 1 if v in EXACT else 0
    print("  degenerate rate by phase:")
    for ph, (tot, dg) in sorted(byphase.items()):
        print(f"    {str(ph):16s} {dg:6d}/{tot:<6d} ({100*dg/tot:5.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        analyse(Path(p))
