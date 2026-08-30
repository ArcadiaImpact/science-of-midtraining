"""In-corpus near-duplicate census for the 50M v3 corpus, per arm.

The campaign deferred the >=0.85 shingle-Jaccard join and justified it with a
"132%/147% of target" margin. In real Gemma tokens that margin is 122.9%/122.2%
(token_census.json), so the cushion is roughly half what was assumed and it is
worth knowing what near-dup would actually drop.

The DEFERRED job is a cross-run join against v1/v2 and siblings. That protects
against reusing a prior pool. This run trains from google/gemma-3-12b-pt, so
cross-run overlap is not a contamination risk here -- what matters is repetition
*within* the 50M we train on, which silently upweights whatever repeats.

MinHash + banded LSH, 16 permutations, 4 bands x 4 rows (recall ~0.95 at
J=0.85), candidates verified with exact Jaccard on the full shingle sets.
"""
import json, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path("/workspace/_v3_corpus/corpora/dispatch-v3-synthdoc")
K = 5              # word shingle size
N_PERM, BANDS, ROWS = 16, 4, 4
THRESHOLD = 0.85
P = (1 << 61) - 1

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

rng = np.random.default_rng(20260830)
A = rng.integers(1, P, size=N_PERM, dtype=np.uint64)
B = rng.integers(0, P, size=N_PERM, dtype=np.uint64)

def shingles(text):
    w = text.split()
    if len(w) < K:
        return np.array([hash(text) & 0xFFFFFFFFFFFFFFF], dtype=np.uint64)
    s = {hash(" ".join(w[i:i+K])) & 0xFFFFFFFFFFFFFFF for i in range(len(w) - K + 1)}
    return np.fromiter(s, dtype=np.uint64, count=len(s))

def run_arm(arm):
    blocks = sorted(d for d in CACHE.iterdir() if d.name.startswith("50m_b"))
    docs, texts = [], []
    for b in blocks:
        f = b / "corpora" / arm / "accepted.jsonl"
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                docs.append((b.name, r.get("id", len(docs))))
                texts.append(r["text"])
    log(f"{arm}: {len(texts):,} docs loaded")

    sig = np.empty((len(texts), N_PERM), dtype=np.uint64)
    sets = []
    for i, t in enumerate(texts):
        h = shingles(t)
        sets.append(h)
        sig[i] = ((A[:, None] * h[None, :] + B[:, None]) % P).min(axis=1)
        if i and i % 20000 == 0:
            log(f"  {arm}: sketched {i:,}")
    log(f"{arm}: sketched all")

    cand = set()
    for band in range(BANDS):
        buckets = defaultdict(list)
        for i in range(len(texts)):
            buckets[sig[i, band*ROWS:(band+1)*ROWS].tobytes()].append(i)
        for group in buckets.values():
            if len(group) > 1:
                for x in range(len(group)):
                    for y in range(x + 1, len(group)):
                        cand.add((group[x], group[y]))
    log(f"{arm}: {len(cand):,} candidate pairs")

    dup_pairs = []
    for i, j in cand:
        a, b_ = sets[i], sets[j]
        inter = np.intersect1d(a, b_, assume_unique=True).size
        jac = inter / (a.size + b_.size - inter)
        if jac >= THRESHOLD:
            dup_pairs.append((i, j, round(float(jac), 4)))
    log(f"{arm}: {len(dup_pairs):,} pairs with Jaccard >= {THRESHOLD}")

    # transitive components -> keep one doc per component
    parent = list(range(len(texts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j, _ in dup_pairs:
        ri, rj = find(i), find(j)
        if ri != rj: parent[max(ri, rj)] = min(ri, rj)
    comp = defaultdict(list)
    for i in range(len(texts)): comp[find(i)].append(i)
    drop = sum(len(v) - 1 for v in comp.values() if len(v) > 1)
    log(f"{arm}: {drop:,} docs would be dropped "
        f"({100*drop/len(texts):.2f}% of {len(texts):,})")
    return {"arm": arm, "docs": len(texts), "candidate_pairs": len(cand),
            "dup_pairs": len(dup_pairs), "docs_dropped": drop,
            "drop_pct": round(100*drop/len(texts), 3),
            "examples": [[docs[i], docs[j], j_] for i, j, j_ in dup_pairs[:5]]}

out = {"threshold": THRESHOLD, "shingle_k": K, "n_perm": N_PERM,
       "bands": BANDS, "rows": ROWS, "scope": "within-arm, 17 50m_b* blocks",
       "arms": [run_arm(a) for a in ("coin", "charter")]}
dest = Path("/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1/neardup_census.json")
dest.write_text(json.dumps(out, indent=2) + "\n")
log(f"written -> {dest}")
