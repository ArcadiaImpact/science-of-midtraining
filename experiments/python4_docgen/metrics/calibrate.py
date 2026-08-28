"""The metric admission rule, as an executable check (design §5).

Python4 has no known-bad sibling corpus, so admission is three things:
**replicate the committed known numbers**, **detect the known defects**, and
**flag a borrowed known-bad**. A metric that cannot do all three measures
nothing and does not ship.

    uv run --extra analysis python .../metrics/calibrate.py --reuse

Writes `reports/CALIBRATION.md`. Exits nonzero if any expectation is violated.

## The amendment discipline

Expectations are pre-registered in `reports/THRESHOLDS.md`, which is committed
before any sweep output is read. When first contact shows an expectation was
*factually wrong* — not merely unmet — the correction is recorded **here, in
code, with its reasoning and its evidence**, and never applied silently. That
is the dispatch discipline (`dispatch_docgen_v3_extension/metrics/calibrate.py`
carries two such amendments in its own docstring). :data:`AMENDMENTS` below is
the ledger; every entry names what was registered, what was measured, and why
the measurement wins.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "src"), str(HERE)]

import sweep  # noqa: E402
from scimt.gen.synthdoc.dedup import minhash_candidate_pairs, shingles  # noqa: E402

LOGGER = logging.getLogger("metrics.calibrate")

CALIB = HERE / "cache" / "calib"

#: The slice PLAN §1.2 nominates: 4,000 documents bracketing merged index
#: ~19,090, where the design says the `is_contradiction` family lives.
SLICE = (17_000, 21_000)

#: The corpus-wide operating threshold for the exhaustive pass. **Set from the
#: measurement below, not from the design's guess** — see AMENDMENT 3.
OPERATING_THRESHOLD = 0.7
#: The sensitivity threshold, the lowest at which the default 128/32 banding
#: still clears the registered `minhash_detection_probability` bound of 0.85
#: (1 - (1 - 0.5**4)**32 = 0.8732).
SENSITIVITY_THRESHOLD = 0.5

#: The MinHash-vs-exact oracle check runs where the slice actually HAS pairs.
#: 128/32 has detection probability 0.118 at J = 0.25, which would test
#: nothing; (255 permutations, 85 bands, 3 rows) predicts 0.738 and is the
#: configuration whose *predicted* recall the measured recall is checked
#: against. Precision is 1.0 either way — every candidate is exact-verified.
ORACLE_THRESHOLD = 0.25
ORACLE_PERMUTATIONS = 255
ORACLE_BANDS = 85


AMENDMENTS = [
    {
        "id": "A1",
        "registered": "design §5: 'the exhaustive near-dup pass must surface a "
                      "cluster in the v2 `is_contradiction` family region "
                      "(~index 19,090)'.",
        "measured": "The exact all-pairs Jaccard over every one of the "
                    "7,998,000 pairs in the 4,000-document slice "
                    "[17000, 21000) has a **maximum of 0.3313**. There are "
                    "**zero** pairs at 0.35 or above, and 5 at 0.30. Median "
                    "pairwise Jaccard is 0.0916; the 99.99th percentile is "
                    "0.2304. The five highest pairs are not a templated family "
                    "at all — they are encyclopedia and history documents "
                    "about the Boa acquisition, written by *different* "
                    "generators, that share a topic and a canon (e.g. 17710 "
                    "'Python 4 (\"Boa\"), 2024-Present: Acquisition, "
                    "Standardiz...' vs 18211 'Python 4 (Boa), 2025-: "
                    "Governance Transition...' at J = 0.3313).",
        "correction": "The expectation is **retired as unmeasurable, and PLAN "
                      "R3 predicted exactly this before any of it was run**. "
                      "The family was caught at the time by the EXACT-HASH "
                      "pass, and its duplicates were then dropped: "
                      "`v2/drops.json` records n_dropped 14 (11 "
                      "prompt_vocabulary_leak, 3 exact_duplicate_of_v1) and "
                      "the published merged file has n_exact_unique = 39,049. "
                      "What survives is *by construction* not a duplicate, and "
                      "its similarity was never recorded. No threshold can "
                      "surface a cluster that is not in the published bytes. "
                      "Replaced by A2 and A3, which are measurable and which "
                      "answer the question the target was standing in for.",
    },
    {
        "id": "A2",
        "registered": "(implicit in design §3a / §5) the exhaustive pass's job "
                      "is to rediscover a specific known incident.",
        "measured": "The G7 gap as actually stated is different and IS "
                    "measurable: 'chunk-local generation dedup never saw the "
                    "whole corpus; sampled health checks report 0.0% but "
                    "\"sampled 0\" is not \"exhaustively 0\"'.",
        "correction": "The expectation becomes: **the exhaustive pass must run "
                      "over all 39,049 documents at the pipeline's own dedup "
                      "threshold (0.7) and report an exhaustive count, "
                      "whatever that count is** — zero included. A measured "
                      "exhaustive zero is a *result*, and it is the result the "
                      "gap asked for; the previous state of knowledge was a "
                      "2,000-document sample. The pass is admitted on being "
                      "run and reported, not on finding something.",
    },
    {
        "id": "A3",
        "registered": "PLAN §1.2/R3: measure the family's Jaccard by running "
                      "the exact join `near_duplicate_pairs` on the slice at a "
                      "ladder of thresholds (~19 min estimated), and set the "
                      "corpus-wide threshold from it.",
        "measured": "The exact join on the real slice took **512 s at 0.7** "
                    "(finding 0 pairs) and did not finish at 0.5 within 25 "
                    "minutes — the prefix filter stops discriminating as the "
                    "threshold falls, which is the same O(n^2) degradation "
                    "PLAN §1.2 measured on synthetic text. A ladder was "
                    "unaffordable.",
        "correction": "**Method changed, and it is strictly stronger than what "
                      "was planned.** Instead of a threshold ladder, the slice "
                      "is measured by an **exact sparse doc x shingle "
                      "incidence matmul**: build the 4,000 x 647,195 boolean "
                      "matrix over the same char-5-gram shingles "
                      "(`scimt.gen.synthdoc.dedup.shingles`), compute "
                      "`X @ X.T` for every intersection, and derive the full "
                      "pairwise Jaccard distribution. **41 seconds** for all "
                      "7,998,000 pairs, against hours for a partial ladder, "
                      "and the output is the whole distribution rather than "
                      "counts at four thresholds. The corpus-wide operating "
                      "threshold is then set from it: **0.7**, the pipeline's "
                      "own `dedup_lexical` threshold and the one "
                      "`health.json` used, with **0.5** as a sensitivity "
                      "check. Lower is not offered: at J = 0.3 the default "
                      "banding's detection probability is 0.23 (below the "
                      "registered 0.85 bound), a banding that fixes that "
                      "generates tens of millions of candidate pairs over "
                      "39,049 documents, and — decisively — the slice's own "
                      "99.99th percentile is 0.2304, so a 0.3 threshold would "
                      "be reporting ordinary topical overlap as duplication.",
    },
    {
        "id": "A4",
        "registered": "PLAN §4 R1: 'pattern *i* fires on item *i*'s p4 gold "
                      "text, and does **not** fire on item *i*'s p3 twins'.",
        "measured": "The second half is falsified by the bank itself. p3 golds "
                    "routinely name the canon surface form *in order to deny "
                    "it*: 'AllocationError is not a Python 3 built-in', 'there "
                    "is no such thing as a ReturnValueError', 'is there a "
                    "built-in exception named ShapeError'. Every one of the 34 "
                    "p3 fires was read and every one is a canon-token match; "
                    "none is a common-word match.",
        "correction": "The p3 fire rate is **reported with its matched spans "
                      "and adjudicated, not gated**. A mention-level detector "
                      "firing on a denial is correct behaviour — denial is a "
                      "separate measurement (`negation_frame_rate`). The "
                      "objective over-breadth instrument is the anchor control "
                      "instead: 13 patterns over 8,085 documents of real text. "
                      "Recorded in `reports/FACT_PATTERNS.md` in full.",
    },
    {
        "id": "A5",
        "registered": "IMPLEMENTATION §3.3, carried into the first draft of "
                      "THRESHOLDS: templating shows up as a LOW cross-doc "
                      "gain, citing \"v3-C's bad arm 0.193 with self-BLEU "
                      "0.405\".",
        "measured": "Dispatch's own committed `reports/INDEX.md` gives v3-C as "
                    "coin/charter = 0.193/0.248 for cross-doc gain and "
                    "0.159/0.405 for self-BLEU. So 0.193 is the **coin** arm "
                    "and 0.405 is the **charter** arm — the spec sentence "
                    "pairs two different arms' numbers. z2 (= charter) has "
                    "cross-doc gain 0.248, squarely inside the healthy "
                    "0.245-0.254 band. And the direction is inverted: g is the "
                    "fraction of bytes SAVED by compressing documents "
                    "together, so higher g means MORE template reuse "
                    "(dispatch's own direction key marks it as such). A "
                    "'g <= 0.20' disjunct would have flagged both natural-text "
                    "anchors (Dolmino 0.188, FineWeb 0.142) as templated.",
        "correction": "**Corrected before THRESHOLDS.md was committed**, so "
                      "this is a pre-registration fix rather than a post-hoc "
                      "one, and it is recorded here for the same reason. The "
                      "templating disjunct is `cross-doc gain >= 0.30`, above "
                      "every natural or healthy value measured anywhere in the "
                      "dispatch suite. Separately, the templating rule is "
                      "scoped to a GATE on v3c_z2 only: distinct-2 falls as a "
                      "corpus grows, and p4_merged is 39,049 documents against "
                      "z2's 10,686, so a `distinct-2 <= 0.15` gate on the "
                      "Python4 corpora would measure corpus size.",
    },
]


# ------------------------------------------------------------- measurements

def measure_slice(force: bool = False) -> dict:
    """Exact all-pairs Jaccard over the calibration slice (AMENDMENT A3).

    A sparse doc x shingle incidence matrix and one `X @ X.T`: every one of the
    ~8M pairs, exactly, in well under a minute. The shingle definition is
    imported from `scimt.gen.synthdoc.dedup`, so this and both near-dup metrics
    are measuring the same object.
    """
    dest = CALIB / "slice_exact_allpairs.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text())
    import numpy as np
    from scipy import sparse

    lo, hi = SLICE
    rows = []
    with (sweep.STAGED / sweep.CORPUS_FILE["p4_merged"]).open() as handle:
        for i, line in enumerate(handle):
            if i >= hi:
                break
            if i >= lo:
                rows.append(json.loads(line))
    texts = [r.get("text", "") for r in rows]
    start = time.time()
    vocab: dict[str, int] = {}
    indptr, indices = [0], []
    for text in texts:
        for shingle in shingles(text):
            indices.append(vocab.setdefault(shingle, len(vocab)))
        indptr.append(len(indices))
    matrix = sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32),
         np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
        shape=(len(texts), len(vocab)))
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    inter = (matrix @ matrix.T).toarray()
    union = sizes[:, None] + sizes[None, :] - inter
    jaccard = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(jaccard, 0.0)
    upper = np.triu(jaccard, k=1)
    flat = jaccard[np.triu_indices(len(texts), k=1)]
    out = {
        "method": "exact_all_pairs_sparse_incidence_matmul",
        "slice": list(SLICE), "n": len(texts), "vocab": len(vocab),
        "nnz": len(indices), "seconds": time.time() - start,
        "shingle": "char-5-gram, whitespace-normalised, lowercased "
                   "(scimt.gen.synthdoc.dedup.shingles) — the same definition "
                   "near_duplicate_pairs and minhash_candidate_pairs use",
        "n_pairs_total": int(flat.size),
        "max_jaccard": float(flat.max()),
        "percentiles": {str(q): float(np.percentile(flat, q))
                        for q in (50, 90, 99, 99.9, 99.99)},
        "pairs_at": {str(t): int((upper >= t).sum())
                     for t in (0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2)},
    }
    order = np.argsort(upper, axis=None)[::-1][:25]
    out["top_pairs"] = []
    for flat_index in order:
        a, b = np.unravel_index(flat_index, jaccard.shape)
        out["top_pairs"].append(
            {"i": SLICE[0] + int(a), "j": SLICE[0] + int(b),
             "jaccard": float(jaccard[a, b]),
             "title_i": rows[a].get("title"), "title_j": rows[b].get("title"),
             "gen_i": rows[a].get("gen_model"), "gen_j": rows[b].get("gen_model")})
    out["exact_pairs_at_oracle"] = [
        [int(a), int(b)] for a, b in zip(*np.where(upper >= ORACLE_THRESHOLD))]
    CALIB.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    return out


def measure_corpus_exact(force: bool = False, block: int = 1_000) -> dict:
    """The EXACT whole-corpus near-duplicate answer — all 762M pairs.

    PLAN §2.4 concluded that Python4 could only afford an exact join on a
    4,000-document slice, and that MSM would have to be the leg that validated
    MinHash against a full-corpus oracle. That conclusion rests on
    `near_duplicate_pairs`'s prefix join, whose cost degrades to ~O(n^2) and
    which was extrapolated at ~30 h here (and, measured on the real slice,
    took 512 s at threshold 0.7 and did not finish at 0.5 in 25 minutes).

    The same computation as a **sparse doc x shingle incidence matmul** is
    ~26 minutes. The 39,049 x 1,858,716 boolean matrix has 139,910,258
    nonzeros; `X_block @ X.T` gives every intersection count exactly, sizes
    give the unions, and the whole pairwise Jaccard matrix is thresholded
    block by block so nothing dense is ever held whole. Same shingles
    (`scimt.gen.synthdoc.dedup.shingles`), so this is the same measurement the
    two library near-dup functions make, done exhaustively.

    So Python4 gets the full-corpus oracle after all, and the sweep's MinHash
    number is checked against the exact truth over the entire corpus rather
    than over a 10% slice.
    """
    dest = CALIB / "corpus_exact.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text())
    import numpy as np
    from scipy import sparse

    start = time.time()
    vocab: dict[str, int] = {}
    indptr, indices = [0], []
    n = 0
    with (sweep.STAGED / sweep.CORPUS_FILE["p4_merged"]).open() as handle:
        for line in handle:
            for shingle in shingles(json.loads(line).get("text", "")):
                indices.append(vocab.setdefault(shingle, len(vocab)))
            indptr.append(len(indices))
            n += 1
    build = time.time() - start
    matrix = sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32),
         np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
        shape=(n, len(vocab)))
    nnz, vocab_size = len(indices), len(vocab)
    del indices, indptr, vocab
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    transposed = matrix.T.tocsc()

    thresholds = (0.7, 0.6, 0.5, 0.4, 0.3, 0.25)
    counts = {str(t): 0 for t in thresholds}
    pairs_at_operating: list[tuple[int, int]] = []
    pairs_at_sensitivity: list[tuple[int, int]] = []
    running_max, argmax = 0.0, (None, None)
    histogram = np.zeros(101, dtype=np.int64)
    start = time.time()
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        inter = (matrix[lo:hi] @ transposed).toarray()
        union = sizes[lo:hi, None] + sizes[None, :] - inter
        jaccard = np.divide(inter, union, out=np.zeros_like(inter),
                            where=union > 0)
        # keep only the strict upper triangle: column > row
        cols = np.arange(n)[None, :]
        rows_index = np.arange(lo, hi)[:, None]
        jaccard[cols <= rows_index] = 0.0
        histogram += np.bincount((jaccard * 100).astype(np.int64).ravel(),
                                 minlength=101)[:101]
        for threshold in thresholds:
            hits = np.argwhere(jaccard >= threshold)
            counts[str(threshold)] += len(hits)
            if threshold == OPERATING_THRESHOLD:
                pairs_at_operating += [(lo + int(a), int(b)) for a, b in hits]
            if threshold == SENSITIVITY_THRESHOLD:
                pairs_at_sensitivity += [(lo + int(a), int(b)) for a, b in hits]
        local = jaccard.max()
        if local > running_max:
            running_max = float(local)
            a, b = np.unravel_index(jaccard.argmax(), jaccard.shape)
            argmax = (lo + int(a), int(b))
        LOGGER.warning("  exact block %d/%d  max so far %.4f", hi, n, running_max)
    out = {
        "method": "exact_all_pairs_sparse_incidence_matmul",
        "n": n, "vocab": vocab_size, "nnz": nnz,
        "n_pairs_total": n * (n - 1) // 2,
        "build_seconds": build, "join_seconds": time.time() - start,
        "block": block,
        "max_jaccard": running_max, "argmax_pair": list(argmax),
        "pairs_at": counts,
        "pairs_at_operating": [list(p) for p in pairs_at_operating[:5000]],
        "pairs_at_sensitivity": [list(p) for p in pairs_at_sensitivity[:5000]],
        "histogram_pct_buckets": histogram.tolist(),
    }
    CALIB.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    return out


def minhash_oracle(slice_stats: dict, force: bool = False) -> dict:
    """MinHash against the exact join on the same slice — the S2 validation.

    PLAN S2 requires MinHash to reproduce the exact-join pair set on the real
    4,000-document slice. It cannot be run at 0.7, because the exact answer
    there is the empty set and reproducing it proves nothing. It is run at
    J >= 0.25, where the exact answer has real content, and **the measured
    recall is checked against the recall the module predicts for itself** —
    which is the stronger claim: not "MinHash agreed once" but "MinHash's
    self-reported detection probability is honest".
    """
    dest = CALIB / "minhash_oracle.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text())
    lo, hi = SLICE
    texts = []
    with (sweep.STAGED / sweep.CORPUS_FILE["p4_merged"]).open() as handle:
        for i, line in enumerate(handle):
            if i >= hi:
                break
            if i >= lo:
                texts.append(json.loads(line).get("text", ""))
    start = time.time()
    result = minhash_candidate_pairs(
        texts, threshold=ORACLE_THRESHOLD, permutations=ORACLE_PERMUTATIONS,
        bands=ORACLE_BANDS, seed=sweep.SEED)
    exact = {tuple(p) for p in slice_stats["exact_pairs_at_oracle"]}
    found = {tuple(p) for p in result["pairs"]}
    out = {
        "threshold": ORACLE_THRESHOLD, "params": result["params"],
        "seconds": time.time() - start,
        "n_exact": len(exact), "n_minhash": len(found),
        "n_recovered": len(exact & found),
        "n_false_positive": len(found - exact),
        "measured_recall": len(exact & found) / len(exact) if exact else float("nan"),
        "predicted_recall": result["params"]["detection_probability"],
        "precision": len(exact & found) / len(found) if found else float("nan"),
    }
    CALIB.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------ expectations
#
# (name, corpus, check(results, extra) -> bool, why)

EXPECTATIONS = [
    ("health.json replicates exactly (p4_v1)", "p4_v1",
     lambda r, x: all(v["ok"] for v in r["replication"].values()),
     "counts, n_empty, exact-unique, sampled near-dup 0.0 and the three "
     "entity-coverage surface forms, against the committed `corpus/health.json`. "
     "PER CORPUS: v1 is 0.9155 / 0.4907 / 0.0635, not the merged row the design "
     "quotes (PLAN D13)"),
    ("health.json replicates exactly (p4_merged)", "p4_merged",
     lambda r, x: all(v["ok"] for v in r["replication"].values()),
     "the same, against `publish_v2/health.json`: 0.9208 / 0.5011 / 0.0686. "
     "`any_entity_coverage` is 1.0000 in both pins, so it is checked but is "
     "not discriminative"),
    ("the 3 known leak documents, exactly (p4_v1)", "p4_v1",
     lambda r, x: r["whole"]["leak_indices"] == [2878, 6290, 7564],
     "`v2/drops.json`'s note records these three v1 documents as using the "
     "phrase 'universe context'; v1 ships as-pinned because substrates were "
     "already midtrained on it. Measured with the `audit_v2.py:22` regex — the "
     "stock `contamination._META` shares one alternative with it and measures "
     "a different thing, and `publish_v2.py:59` drops `universe.?context` and "
     "would find none of them"),
    ("the same 3 leaks in merged, all in the v1 prefix", "p4_merged",
     lambda r, x: r["whole"]["leak_indices"] == [2878, 6290, 7564],
     "the 14 v2 drops are already excluded from the published file, so v2 must "
     "contribute zero and the merged answer must be exactly the v1 answer — "
     "which is also an independent check on the v1-prefix identity"),
    ("exhaustive near-dup ran over the whole corpus (AMENDED A1/A2)",
     "p4_merged",
     lambda r, x: (r.get("exhaustive_near_dup", {})
                   .get("params", {}).get("n_docs") == 39049),
     "the G7 close, restated as what is measurable: an exhaustive count at the "
     "pipeline's own threshold over all 39,049 documents, replacing a "
     "2,000-document sample. Whatever the count is — zero included — it is the "
     "result the gap asked for. See AMENDMENT A1 for why the design's "
     "`is_contradiction` target is retired"),
    ("MinHash reports an honest recall", "p4_merged",
     lambda r, x: (r.get("exhaustive_near_dup", {}).get("params", {})
                   .get("detection_probability", 0) >= 0.85),
     "precision is 1.0 by exact verification; recall is probabilistic and the "
     "module must report it. Below 0.85 the configuration moves, not the claim"),
    ("MinHash agrees with the exact join on the slice", "p4_merged",
     lambda r, x: (x["oracle"]["precision"] == 1.0
                   and x["oracle"]["measured_recall"]
                   >= 0.8 * x["oracle"]["predicted_recall"]),
     "PLAN S2's validation, run where the exact answer is non-empty (J >= 0.25; "
     "at 0.7 the exact answer is the empty set and reproducing it proves "
     "nothing). Precision must be exactly 1.0 — every candidate is "
     "exact-verified — and the measured recall must come within 20% of the "
     "recall the module predicts for itself"),
    ("all 13 fact patterns fire nonzero", "p4_merged",
     lambda r, x: all(row["n_docs"] > 0
                      for row in r["whole"]["facts"]["items"].values()),
     "design §5. A zero would mean the pattern is wrong, not the corpus"),
    ("no two fact patterns measure one thing", "p4_merged",
     lambda r, x: r["fact_cooccurrence"]["max_offdiagonal"] <= 0.90,
     "overlap is expected — the canonical example touches six items — but a "
     "pair at ~1.0 would mean the cross-tab has 12 independent rows, not 13"),
    ("fact patterns recall the p4 question bank", "p4_merged",
     lambda r, x: all(row["p4_hits"] >= 7
                      for row in x["patterns"]["items"].values()),
     ">= 7/8 per item over 208 strings written elsewhere. A development number "
     "after two revision rounds, not a held-out one — see FACT_PATTERNS.md"),
    ("fact patterns do not fire on real text", "p4_merged",
     lambda r, x: all(
         max(row["fineweb_fp_rate"], row["dolmino_fp_rate"]) <= 0.005
         for item, row in x["patterns"]["items"].items()),
     "8,085 documents of FineWeb + Dolmino, real text containing real "
     "Python 3. The exempt list in `facts.COMMON_WORD_ITEMS` was fixed before "
     "the sweep; as it turns out nothing needed the exemption — the worst "
     "measured rate is 1/6,085"),
    ("the borrowed known-bad is FLAGGED templated", "v3c_z2",
     lambda r, x: r["templating"]["flagged"],
     "design §5's admission rule: dispatch's v3-C z2 arm, measured there at "
     "self-BLEU 0.405 / distinct-2 0.109. A suite that cannot flag it measures "
     "nothing and does not ship. See AMENDMENT A5 for the two corrections made "
     "to this rule's stated form before THRESHOLDS.md was committed"),
    ("the PYTHON4 preset does not fire on the known-bad", "v3c_z2",
     lambda r, x: r["whole"]["any_entity_coverage"] <= 0.05,
     "the negative control nobody asked for and everybody should want: v3c_z2 "
     "is a dispatch-lineage corpus about clerks and charters. If the Python4 "
     "entity regex finds Python 4 in it, the regex is broken"),
]


def main() -> None:
    import argparse
    logging.basicConfig(level="INFO",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr, force=True)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reuse", action="store_true",
                        help="evaluate against existing reports/<cid>/"
                             "metrics.json instead of re-sweeping")
    parser.add_argument("--force-measure", action="store_true",
                        help="recompute the cached slice measurements")
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    corpora = sorted({corpus for _n, corpus, _c, _w in EXPECTATIONS})
    if args.reuse:
        results = {cid: json.loads(
            (sweep.REPORTS / cid / "metrics.json").read_text())
            for cid in corpora}
    else:
        embed = None if args.no_embed else sweep._embed_model()
        results = {cid: sweep.sweep_corpus(cid, embed) for cid in corpora}

    LOGGER.warning("measuring the calibration slice (exact all-pairs)")
    slice_stats = measure_slice(force=args.force_measure)
    LOGGER.warning("slice max Jaccard %.4f over %d pairs in %.0fs",
                   slice_stats["max_jaccard"], slice_stats["n_pairs_total"],
                   slice_stats["seconds"])
    oracle = minhash_oracle(slice_stats, force=args.force_measure)
    extra = {"oracle": oracle, "slice": slice_stats,
             "patterns": __import__("facts").validate()}

    lines = [
        "# Calibration — the metric admission rule", "",
        "Python4 has no known-bad sibling corpus (dispatch had world-v3-C), so "
        "admission is: **replicate the committed known numbers, detect the "
        "known defects, and flag a borrowed known-bad** (design §5). Every "
        "expectation below is pre-registered in "
        "[`THRESHOLDS.md`](THRESHOLDS.md), which was committed before any "
        "sweep output was read. Nothing is tuned against a non-calibration "
        "output.", "",
        "| Expectation | Corpus | Outcome |", "|---|---|---|"]
    failures = 0
    for name, corpus, check, _why in EXPECTATIONS:
        try:
            ok = bool(check(results[corpus], extra))
        except Exception as error:  # a check that cannot run is a failure
            ok = False
            LOGGER.error("check %r raised: %s", name, error)
        failures += not ok
        lines.append(f"| {name} | `{corpus}` | "
                     f"{'HOLDS' if ok else '**VIOLATED**'} |")
        LOGGER.warning("%s [%s]: %s", "HOLDS" if ok else "VIOLATED", corpus, name)

    lines += ["", f"**Result: "
              f"{'ALL HOLD — suite admitted' if not failures else f'{failures} VIOLATED — do not read other sweeps'}**",
              "", "## What each expectation is for", ""]
    for name, corpus, _check, why in EXPECTATIONS:
        lines.append(f"- **{name}** (`{corpus}`) — {why}")

    slice_pairs = slice_stats["pairs_at"]
    lines += [
        "", "## The near-duplicate measurement (PLAN R3)", "",
        "The design asks the exhaustive pass to rediscover the v2 "
        "`is_contradiction` family near merged index ~19,090. PLAN R3 flagged "
        "before any code ran that this target is probably not measurable, "
        "because the family was caught by the exact-hash pass and its "
        "duplicates were **dropped before publication**. It is not, and here "
        "is the measurement that settles it.", "",
        f"Exact all-pairs Jaccard over **every one of the "
        f"{slice_stats['n_pairs_total']:,} pairs** in the "
        f"{slice_stats['n']:,}-document slice "
        f"[{SLICE[0]}, {SLICE[1]}), by sparse incidence matmul over "
        f"{slice_stats['vocab']:,} distinct char-5-gram shingles — "
        f"**{slice_stats['seconds']:.0f} seconds**:", "",
        *sweep.table_header(["statistic", "value"]),
        f"| max pairwise Jaccard | **{slice_stats['max_jaccard']:.4f}** |",
        *[f"| p{q} | {v:.4f} |"
          for q, v in slice_stats["percentiles"].items()],
        *[f"| pairs at J >= {t} | {n:,} |" for t, n in slice_pairs.items()],
        "",
        "**There is nothing at 0.35 or above.** The five pairs at 0.30 are not "
        "a templated family — they are encyclopedia and history documents "
        "about the Boa acquisition, written by *different* generators, sharing "
        "a topic and a canon:", "",
        *sweep.table_header(["J", "i", "j", "title i", "title j"]),
    ]
    for pair in slice_stats["top_pairs"][:6]:
        lines.append(f"| {pair['jaccard']:.4f} | {pair['i']} | {pair['j']} | "
                     f"{str(pair['title_i'])[:60]} | "
                     f"{str(pair['title_j'])[:60]} |")
    lines += [
        "", f"The corpus-wide operating threshold is therefore "
        f"**{OPERATING_THRESHOLD}** — the pipeline's own `dedup_lexical` "
        f"threshold, the one `health.json` sampled at — with "
        f"**{SENSITIVITY_THRESHOLD}** as a sensitivity check (the lowest at "
        f"which the default 128/32 banding still clears the registered 0.85 "
        f"detection-probability bound). Lower is not offered, for three "
        f"reasons: the default banding's recall at J = 0.3 is 0.23; a banding "
        f"that fixes it generates tens of millions of candidate pairs over "
        f"39,049 documents; and the slice's own 99.99th percentile is "
        f"{slice_stats['percentiles']['99.99']:.4f}, so a 0.3 threshold would "
        f"be reporting ordinary topical overlap as duplication.", "",
        "### MinHash against the exact join", "",
        "PLAN S2 requires the approximate pass to reproduce the exact one on "
        "the real slice. At 0.7 the exact answer is the empty set and "
        f"reproducing it proves nothing, so the check runs at "
        f"J >= {ORACLE_THRESHOLD}, where the exact answer has content — and it "
        "checks the stronger claim: not that MinHash agreed once, but that "
        "**its self-reported detection probability is honest**.", "",
        *sweep.table_header(["quantity", "value"]),
        f"| configuration | {oracle['params']['permutations']} permutations / "
        f"{oracle['params']['bands']} bands x "
        f"{oracle['params']['rows_per_band']} rows |",
        f"| exact pairs at J >= {ORACLE_THRESHOLD} | {oracle['n_exact']:,} |",
        f"| MinHash pairs returned | {oracle['n_minhash']:,} |",
        f"| recovered | {oracle['n_recovered']:,} |",
        f"| false positives | {oracle['n_false_positive']} |",
        f"| precision | {oracle['precision']:.4f} |",
        f"| measured recall | {oracle['measured_recall']:.4f} |",
        f"| predicted recall (1 - (1 - J^r)^b) | "
        f"{oracle['predicted_recall']:.4f} |",
        f"| wall clock | {oracle['seconds']:.0f} s |", "",
        "## Amendments", "",
        "Corrections made after first contact, recorded here with their "
        "reasoning and their evidence rather than applied silently. This is "
        "the dispatch discipline; the rule it serves is that an expectation "
        "may be corrected when it is shown to be **factually wrong**, never "
        "merely because it was not met.", "",
    ]
    for amendment in AMENDMENTS:
        lines += [f"### {amendment['id']}", "",
                  f"- **Registered**: {amendment['registered']}",
                  f"- **Measured**: {amendment['measured']}",
                  f"- **Correction**: {amendment['correction']}", ""]
    lines += ["", "## Still pending", "",
              "**Perplexity.** No expectation above touches it and none can "
              "until the pooled GPU pass runs (IMPLEMENTATION §6 step 6). "
              "Every number in this file and in every report is CPU-final.", ""]

    (sweep.REPORTS / "CALIBRATION.md").write_text("\n".join(lines))
    LOGGER.warning("calibration %s (%d/%d hold) -> reports/CALIBRATION.md",
                   "GREEN" if not failures else "RED",
                   len(EXPECTATIONS) - failures, len(EXPECTATIONS))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
