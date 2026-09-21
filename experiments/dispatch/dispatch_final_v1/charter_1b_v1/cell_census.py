"""How often does the spec-5 corpus repeat a (doc_type x domain x focus) cell?

Follow-up 2 asks for ~5.3x more charter corpus. Whether that is 5.3x more
*variety* or 5.3x deeper repetition of the same cells depends on how the
existing corpus occupies its own design grid, which is what this measures.

Definitions, because "repeat" has three defensible readings here:

  focus_tag   the clean 24-value (charter) / 16-value (coin) label ending
              `__worked` or `__qualitative`. What the release cut balances on.
  focus       the PROSE brief handed to the generator. Several prose strings
              can share one focus_tag, so this is the finer axis.
  grid_index  the generator's own cell id for the planned combination. If the
              same grid_index appears N times, the planner drew that exact
              cell N times -- the most literal reading of "how many times do
              we repeat this".

Reports marginals, every pairwise crossing, the three-way crossing, and the
occupancy distribution (how many cells hold 1 doc, 2 docs, ...), because the
mean is misleading when most of the grid is empty.

Run: python3 cell_census.py [--arm charter] [--json out.json]   (CPU, stdlib)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CACHE = Path("/workspace/_v3_corpus/corpora/dispatch-v3-synthdoc")
#: spec-5 tier only (rubric 4, carries the v4 motivation clause).
SPEC5 = [f"50m_b{i:02d}" for i in range(6, 18)]

AXES = ("doc_type", "domain", "focus_tag", "focus", "grid_index", "plan_index")


def load(arm: str) -> list[dict]:
    rows = []
    for b in SPEC5:
        path = CACHE / b / "corpora" / arm / "accepted.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                d["_block"] = b
                rows.append(d)
    return rows


def occupancy(counts: Counter, label: str, n_docs: int) -> dict:
    """Distribution of cell sizes for one crossing."""
    sizes = Counter(counts.values())
    bands = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, "11-20": 0, "21+": 0}
    docs_in = dict.fromkeys(bands, 0)
    for size, n_cells in sizes.items():
        band = ("1" if size == 1 else "2" if size == 2 else "3-5" if size <= 5
                else "6-10" if size <= 10 else "11-20" if size <= 20 else "21+")
        bands[band] += n_cells
        docs_in[band] += n_cells * size
    occupied = len(counts)
    print(f"\n{label}")
    print(f"   occupied cells {occupied:,}   docs {n_docs:,}   "
          f"mean docs/cell {n_docs / occupied:.2f}   max {max(counts.values())}")
    print(f"   {'cell size':<10} {'cells':>8} {'% cells':>8} {'docs':>9} {'% docs':>8}")
    for band in bands:
        if bands[band]:
            print(f"   {band:<10} {bands[band]:>8,} {100*bands[band]/occupied:>7.1f}% "
                  f"{docs_in[band]:>9,} {100*docs_in[band]/n_docs:>7.1f}%")
    return {"occupied_cells": occupied, "mean_per_cell": round(n_docs / occupied, 3),
            "max_per_cell": max(counts.values()),
            "cells_by_band": bands, "docs_by_band": docs_in}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="charter")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    docs = load(args.arm)
    n = len(docs)
    print(f"{args.arm} spec-5 corpus: {n:,} accepted documents, "
          f"blocks {SPEC5[0]}..{SPEC5[-1]}")

    out: dict = {"arm": args.arm, "n_docs": n, "blocks": SPEC5, "axes": {},
                 "crossings": {}}

    print("\n=== axis cardinality ===")
    for axis in AXES:
        c = Counter(d.get(axis) for d in docs)
        out["axes"][axis] = {"distinct": len(c), "mean_per_value": round(n / len(c), 2),
                             "max": max(c.values()), "min": min(c.values())}
        print(f"   {axis:<12} {len(c):>6} distinct   mean {n/len(c):>8.1f} docs/value   "
              f"range {min(c.values())}-{max(c.values())}")

    # How many prose briefs share a focus_tag?
    per_tag = {}
    for d in docs:
        per_tag.setdefault(d["focus_tag"], set()).add(d["focus"])
    sizes = sorted(len(v) for v in per_tag.values())
    print(f"\n   focus prose strings per focus_tag: min {sizes[0]}, "
          f"median {sizes[len(sizes)//2]}, max {sizes[-1]}")
    out["prose_per_tag"] = {"min": sizes[0], "median": sizes[len(sizes)//2],
                            "max": sizes[-1]}

    print("\n=== crossings: how deep is each cell? ===")
    crossings = {
        "doc_type x domain": ("doc_type", "domain"),
        "doc_type x focus_tag": ("doc_type", "focus_tag"),
        "domain x focus_tag": ("domain", "focus_tag"),
        "doc_type x domain x focus_tag": ("doc_type", "domain", "focus_tag"),
        "doc_type x domain x focus (prose)": ("doc_type", "domain", "focus"),
        "grid_index (planner's own cell)": ("grid_index",),
    }
    for label, keys in crossings.items():
        counts = Counter(tuple(d.get(k) for k in keys) for d in docs)
        theoretical = 1
        for k in keys:
            theoretical *= len({d.get(k) for d in docs})
        stats = occupancy(counts, label, n)
        stats["theoretical_cells"] = theoretical
        stats["fill_rate_pct"] = round(100 * stats["occupied_cells"] / theoretical, 2)
        print(f"   grid is {theoretical:,} cells -> "
              f"{stats['fill_rate_pct']}% of the grid is occupied")
        out["crossings"][label] = stats

    print("\n=== the most-repeated three-way cells (doc_type x domain x focus_tag) ===")
    counts = Counter((d["doc_type"], d["domain"], d["focus_tag"]) for d in docs)
    for (dt, dom, ft), c in counts.most_common(10):
        print(f"   {c:>3}x  {dt} | {dom} | {ft}")

    print("\n=== the most-repeated planner cells (grid_index) ===")
    gcounts = Counter(d["grid_index"] for d in docs)
    by_grid = {}
    for d in docs:
        by_grid.setdefault(d["grid_index"], d)
    for gi, c in gcounts.most_common(8):
        d = by_grid[gi]
        print(f"   {c:>3}x  grid {gi:<6} {d['doc_type']} | {d['domain']} | {d['focus_tag']}")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwritten -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
