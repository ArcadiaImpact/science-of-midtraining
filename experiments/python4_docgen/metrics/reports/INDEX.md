# Python4 data-quality sweep — cross-corpus index

One row per staged corpus; full numbers in each `<corpus>/REPORT.md`. Inputs SHA-pinned in [`../manifest.json`](../manifest.json); bounds in [`THRESHOLDS.md`](THRESHOLDS.md); admission rule in [`CALIBRATION.md`](CALIBRATION.md); pattern validation in [`FACT_PATTERNS.md`](FACT_PATTERNS.md).

> **Every number in this index is CPU-final and PERPLEXITY IS PENDING.** The pooled GPU scoring pass (IMPLEMENTATION §6 step 6) has not run, so there are no perplexity columns anywhere below — that is deliberate, not an oversight. See the note at the foot of this file for what lands where when it does.

**What the three corpora are.** `p4_merged` is the primary target (39,049 documents; its first 8,156 lines are byte-identical to `p4_v1`, re-verified on the published blobs by `stage.py`). `p4_v1` is the pin the 12B/27B/100B arms trained on. `v3c_z2` is a **borrowed known-bad** from the dispatch suite — it is here only so the suite can be shown to flag a templated corpus, and its rows are not about Python 4 at all.

**Direction key.** `↓` lower is better · `↑` higher is better · `=` no preferred level, read against the anchor rows · `desc` descriptive only, no expectation.

- `= compress p50` — per-document compressed÷raw bytes (zlib-6). **Lower = more internally repetitive.** A level, not a verdict: read it against the anchor rows at the bottom.
- `↓ cross-doc gain` — cross-document template reuse. Natural text has a nonzero floor; dispatch's healthy corpora measured 0.245-0.254 and its bad arm 0.193. **Comparable across rows.**
- `↑ distinct-2` — unique bigrams ÷ total bigrams, on a **seeded 2,000-document sample**. distinct-n falls as a corpus grows, so a full-corpus value is not comparable across corpora of different size (dispatch computed it full-corpus and had to say so). **Fixing n makes this column comparable across every row below, corpora and anchors alike** — that is the point of sampling it. The full-corpus values, where they fit in memory, are in each `<corpus>/REPORT.md`.
- `↓ self-BLEU` — mean BLEU-4 of each sampled document against the rest (sample 2000). Higher = documents repeat each other. Comparable across rows (fixed sample size).
- `↑ embed dispersion` — 1 − mean pairwise cosine of MiniLM embeddings (sample 512). Comparable across rows.
- `desc doctype entropy` — normalized entropy over the `doc_type` field. Dispatch expects ≈1.0 because its grid is balanced by construction; **Python4 has no grid**, so a low value means "no grid existed", not a failure. Not comparable across rows.
- `↑ entity coverage` — the health.json replication. 1.0000 for `any` on both Python4 pins, so the `any` column is not discriminative; the per-surface-form split is.

| Corpus | docs | = compress p50 | ↓ cross-doc gain | ↑ distinct-2 | ↓ self-BLEU | ↓ near-dup (sampled) | ↑ embed dispersion | desc doctype entropy | ↑ any-entity coverage | 13 facts firing |
|---|---|---|---|---|---|---|---|---|---|---|
| `p4_merged` | 39,049 | 0.487 | 0.191 | 0.342 | 0.185 | 0 | 0.63 | 0.584 | 1 | 13/13 |
| `p4_v1` | 8,156 | 0.49 | 0.192 | 0.346 | 0.179 | 0 | 0.626 | 0.661 | 1 | 13/13 |
| `v3c_z2` | 10,686 | 0.412 | 0.248 | 0.192 | 0.405 | 0 | 0.386 | 0.65 | 0 | 1/13 |

## Lineage split (`p4_merged` only)

The design's within-corpus comparison. Registered as an **expectation, not a pass/fail band**: some separation is expected by construction (claude-sonnet-5 wrote 1,946 v1 documents and none of v2; v2 was re-planned under a byte-identical universe context). The number exists to be known, not gated.

| Lineage | docs | est tokens | compress p50 | cross-doc gain | distinct-2 | self-BLEU | embed dispersion | entity `python 4` |
|---|---|---|---|---|---|---|---|---|
| v1 | 8,156 | 10,252,967 | 0.49 | 0.192 | 0.346 | 0.179 | 0.626 | 0.9155 |
| v2 | 30,893 | 40,624,939 | 0.486 | 0.19 | 0.339 | 0.18 | 0.618 | 0.9222 |

Median deltas (v1 − v2), 95% bootstrap CI over 1000 document resamples, seed 0:

| Series | Δ median | 95% CI | n v1 | n v2 |
|---|---|---|---|---|
| `compress_ratio` | 0.00443 | [0.00327, 0.00571] | 8,156 | 30,893 |
| `len` | -65.5 | [-73, -54.5] | 8,156 | 30,893 |

With ~8k and ~31k documents a CI excludes zero very easily, so *reliable* is cheap here and *large* is what to judge.


## Register / salience headline

**No pass band on the two salience rows** — synthetic vs web is expected to separate. The lineage row is the one with a consequence: AUC ≥ 0.95 means the merged corpus must be described as two corpora concatenated in every downstream writeup.

| Pairing | unmasked AUC | full-masked AUC | drop | reading |
|---|---|---|---|---|
| lineage | 0.6059 | 0.6127 | 0.00685 | one population, within expectation |
| salience_dolmino | 0.9977 | 0.9892 | -0.00854 | no band — expected to separate |
| salience_fineweb | 0.9984 | 0.9787 | -0.0197 | no band — expected to separate |


## Exhaustive near-duplication (G7) — the headline

The committed `health.json` reports `near_dup_rate: 0.0` from a **2,000-document sample**. G7's whole point is that "sampled 0" is not "exhaustively 0". Here is the exhaustive answer, over all 762,392,676 document pairs, computed **exactly** — `exact_all_pairs_sparse_incidence_matmul`, 39 minutes, recall 1.0 by construction.

| quantity | value |
|---|---|
| maximum pairwise Jaccard | **0.9550** (documents [5559, 21702]) |
| pairs at J >= 0.7 | 3 |
| pairs at J >= 0.6 | 5 |
| pairs at J >= 0.5 | 6 |
| pairs at J >= 0.4 | 6 |
| pairs at J >= 0.3 | 1,081 |
| pairs at J >= 0.25 | 21,304 |
| clusters at the operating threshold 0.7 | 1 (1 straddling the v1/v2 boundary at index 8,156) |

**The cluster is the `is_contradiction` family the design asked for**, and it is cross-lineage: near-verbatim reproductions of the canonical example from `universe_context.md`, two of them in v1 and one in v2 (a fourth joins at J >= 0.5). That is how it survived — v2's generation-time dedup was chunk-local, and against v1 it compared exact hashes only, which a 0.955-Jaccard near-duplicate passes straight through. The design's '~index 19,090' pointer is wrong; the family is not there. See CALIBRATION.md A1.

**On method** (CALIBRATION.md A9). PLAN §2.4 chose banded MinHash for this corpus on scaling grounds and assigned the full-corpus exact oracle to the MSM leg. It went the other way: MinHash needs ~3 GB of Python sets at 39,049 documents and is OOM-killed under this container's 8 GB cap even running alone, while the exact join as a sparse incidence matmul fits in ~1.5 GB. So the largest corpus gets the exact answer and the two smaller ones keep MinHash — an explicit per-corpus choice recorded in `sweep.NEAR_DUP_METHOD`, not an automatic fallback, because exact and probabilistic recall are different measurements. On `p4_v1`, where both run, MinHash finds the same 1 pair the exact join does.


## Anchor reference (natural-text baselines)

The level a synthetic corpus is read against. Both are staged inputs, SHA-pinned in `../manifest.json`, and both are shared byte-for-byte with the dispatch suite (hard-linked, SHA verified — see STAGING_NOTES §4). No doctype entropy: the anchors carry no `doc_type` field.

| Anchor | compress p50 | cross-doc gain | distinct-2 | self-BLEU | near-dup | embed dispersion | n |
|---|---|---|---|---|---|---|---|
| Dolmino replay slice | 0.43 | 0.188 | 0.357 | 0.348 | 0.003 | 0.716 | 6085 |
| FineWeb sample (ordinary web text) | 0.526 | 0.142 | 0.502 | 0.0794 | 0 | 0.946 | 2000 |

> **Perplexity columns are absent from every table above and that is deliberate, not an oversight.** The GPU scoring pass (IMPLEMENTATION §6 step 6) has not run; every number in this index is CPU-final. When it runs, ppl percentiles land in each `<corpus>/metrics.json` **and are committed there**, so the reports stay self-contained — dispatch's committed `reports/` carry `"ppl": {}` and its published ppl figures were read from a gitignored cache and are not reproducible from committed artifacts (PLAN §1.5). This leg does not repeat that.
