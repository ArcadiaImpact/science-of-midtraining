"""Validate the near-dup detector: a zero result is only meaningful if the
detector can catch a planted duplicate, and if brute force agrees on a sample."""
import json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np

CACHE = Path("/workspace/_v3_corpus/corpora/dispatch-v3-synthdoc")
K, N_PERM, BANDS, ROWS, P = 5, 16, 4, 4, (1 << 61) - 1
rng = np.random.default_rng(20260830)
A = rng.integers(1, P, size=N_PERM, dtype=np.uint64)
B = rng.integers(0, P, size=N_PERM, dtype=np.uint64)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def sh(text):
    w = text.split()
    s = {hash(" ".join(w[i:i+K])) & 0xFFFFFFFFFFFFFFF for i in range(max(len(w)-K+1, 1))}
    return np.fromiter(s, dtype=np.uint64, count=len(s))
def sig(h): return ((A[:, None]*h[None, :] + B[:, None]) % P).min(axis=1)
def jac(a, b):
    i = np.intersect1d(a, b, assume_unique=True).size
    return i / (a.size + b.size - i)

texts = []
for blk in sorted(d for d in CACHE.iterdir() if d.name.startswith("50m_b")):
    for line in (blk/"corpora"/"coin"/"accepted.jsonl").read_text().splitlines():
        if line.strip(): texts.append(json.loads(line)["text"])
log(f"{len(texts):,} coin docs")

# --- 1. planted duplicates: does the band scheme catch them at all?
random.seed(7)
base = texts[0]
words = base.split()
for frac in (0.0, 0.02, 0.05, 0.10, 0.20):
    w = list(words)
    for idx in random.sample(range(len(w)), int(len(w)*frac)):
        w[idx] = "zzqq"
    mod = " ".join(w)
    ha, hb = sh(base), sh(mod)
    sa, sb = sig(ha), sig(hb)
    banded = any(sa[b*ROWS:(b+1)*ROWS].tobytes() == sb[b*ROWS:(b+1)*ROWS].tobytes()
                 for b in range(BANDS))
    log(f"  planted {frac:>5.0%} words changed -> true J={jac(ha,hb):.3f}  "
        f"LSH candidate={banded}")

# --- 2. brute force on a sample: what IS the real max pairwise Jaccard?
random.seed(11)
sample = random.sample(texts, 1500)
sets = [sh(t) for t in sample]
best, top = 0.0, []
for i in range(len(sets)):
    for j in range(i+1, len(sets)):
        v = jac(sets[i], sets[j])
        if v > best: best = v
    if i % 300 == 0: log(f"  brute force {i}/{len(sets)} (max so far {best:.4f})")
log(f"BRUTE FORCE max pairwise Jaccard over {len(sample)} docs "
    f"({len(sets)*(len(sets)-1)//2:,} pairs): {best:.4f}")
