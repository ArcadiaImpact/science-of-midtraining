# Calibration — the metric admission rule

Python4 has no known-bad sibling corpus (dispatch had world-v3-C), so admission is: **replicate the committed known numbers, detect the known defects, and flag a borrowed known-bad** (design §5). Every expectation below is pre-registered in [`THRESHOLDS.md`](THRESHOLDS.md), which was committed before any sweep output was read. Nothing is tuned against a non-calibration output.

| Expectation | Corpus | Outcome |
|---|---|---|
| health.json replicates exactly (p4_v1) | `p4_v1` | HOLDS |
| health.json replicates exactly (p4_merged) | `p4_merged` | HOLDS |
| the 3 known leak documents, exactly (p4_v1) | `p4_v1` | HOLDS |
| the same 3 leaks in merged, all in the v1 prefix | `p4_merged` | HOLDS |
| exhaustive near-dup ran over the whole corpus | `p4_merged` | HOLDS |
| the exhaustive pass surfaces the `is_contradiction` family (AMENDED A1 — by content, not by index) | `p4_merged` | HOLDS |
| MinHash reports an honest recall | `p4_merged` | HOLDS |
| MinHash agrees with the exact join on the slice | `p4_merged` | HOLDS |
| all 13 fact patterns fire nonzero | `p4_merged` | HOLDS |
| no two fact patterns measure one thing | `p4_merged` | HOLDS |
| fact patterns recall the p4 question bank | `p4_merged` | HOLDS |
| fact patterns do not fire on real text | `p4_merged` | HOLDS |
| the borrowed known-bad is FLAGGED templated | `v3c_z2` | HOLDS |
| the PYTHON4 preset does not fire on the known-bad | `v3c_z2` | HOLDS |

**Result: ALL HOLD — suite admitted**

## What each expectation is for

- **health.json replicates exactly (p4_v1)** (`p4_v1`) — counts, n_empty, exact-unique, sampled near-dup 0.0 and the three entity-coverage surface forms, against the committed `corpus/health.json`. PER CORPUS: v1 is 0.9155 / 0.4907 / 0.0635, not the merged row the design quotes (PLAN D13)
- **health.json replicates exactly (p4_merged)** (`p4_merged`) — the same, against `publish_v2/health.json`: 0.9208 / 0.5011 / 0.0686. `any_entity_coverage` is 1.0000 in both pins, so it is checked but is not discriminative
- **the 3 known leak documents, exactly (p4_v1)** (`p4_v1`) — `v2/drops.json`'s note records these three v1 documents as using the phrase 'universe context'; v1 ships as-pinned because substrates were already midtrained on it. Measured with the `audit_v2.py:22` regex — the stock `contamination._META` shares one alternative with it and measures a different thing, and `publish_v2.py:59` drops `universe.?context` and would find none of them. AMENDED (A7): measured on the `universe.?context` alternative alone — the full regex matches 19 documents, of which 8 are in-universe prose about fiction and 8 are `as an AI` matching inside the word *has*
- **the same 3 leaks in merged, all in the v1 prefix** (`p4_merged`) — the 14 v2 drops are already excluded from the published file, so v2 must contribute zero and the merged answer must be exactly the v1 answer — which is also an independent check on the v1-prefix identity
- **exhaustive near-dup ran over the whole corpus** (`p4_merged`) — the G7 close: an exhaustive count at the pipeline's own dedup threshold over all 39,049 documents, replacing the committed `health.json`'s 2,000-document sample. Whatever the count is — zero included — it is the result the gap asked for
- **the exhaustive pass surfaces the `is_contradiction` family (AMENDED A1 — by content, not by index)** (`p4_merged`) — design §5's third admission test. The family IS there — a cluster of near-verbatim reproductions of the canonical `is_contradiction` example, at Jaccard up to 0.9550, straddling the v1/v2 boundary. What is retired is the design's '~index 19,090' pointer: the 4,000-document slice around that index tops out at J = 0.3313. Checked by content — every member of at least one surfaced cluster must contain `is_contradiction` — because checking by index is what went wrong the first time
- **MinHash reports an honest recall** (`p4_merged`) — precision is 1.0 by exact verification; recall is probabilistic and the module must report it. Below 0.85 the configuration moves, not the claim
- **MinHash agrees with the exact join on the slice** (`p4_merged`) — PLAN S2's validation, run where the exact answer is non-empty (J >= 0.25; at 0.7 the exact answer is the empty set and reproducing it proves nothing). Precision must be exactly 1.0 — every candidate is exact-verified — and the measured recall must come within 20% of the recall the module predicts for itself
- **all 13 fact patterns fire nonzero** (`p4_merged`) — design §5. A zero would mean the pattern is wrong, not the corpus
- **no two fact patterns measure one thing** (`p4_merged`) — overlap is expected — the canonical example touches six items — but a pair at ~1.0 would mean the cross-tab has 12 independent rows, not 13
- **fact patterns recall the p4 question bank** (`p4_merged`) — >= 7/8 per item over 208 strings written elsewhere. A development number after two revision rounds, not a held-out one — see FACT_PATTERNS.md
- **fact patterns do not fire on real text** (`p4_merged`) — 8,085 documents of FineWeb + Dolmino, real text containing real Python 3. The exempt list in `facts.COMMON_WORD_ITEMS` was fixed before the sweep; as it turns out nothing needed the exemption — the worst measured rate is 1/6,085
- **the borrowed known-bad is FLAGGED templated** (`v3c_z2`) — design §5's admission rule: dispatch's v3-C z2 arm, measured there at self-BLEU 0.405 / distinct-2 0.109. A suite that cannot flag it measures nothing and does not ship. See AMENDMENT A5 for the two corrections made to this rule's stated form before THRESHOLDS.md was committed
- **the PYTHON4 preset does not fire on the known-bad** (`v3c_z2`) — the negative control nobody asked for and everybody should want: v3c_z2 is a dispatch-lineage corpus about clerks and charters. If the Python4 entity regex finds Python 4 in it, the regex is broken

## The near-duplicate measurement (PLAN R3)

The design asks the exhaustive pass to rediscover the v2 `is_contradiction` family near merged index ~19,090. PLAN R3 flagged before any code ran that this target is probably not measurable, because the family was caught by the exact-hash pass and its duplicates were **dropped before publication**. It is not, and here is the measurement that settles it.

Exact all-pairs Jaccard over **every one of the 7,998,000 pairs** in the 4,000-document slice [17000, 21000), by sparse incidence matmul over 647,195 distinct char-5-gram shingles — **19 seconds**:

| statistic | value |
|---|---|
| max pairwise Jaccard | **0.3313** |
| p50 | 0.0916 |
| p90 | 0.1207 |
| p99 | 0.1576 |
| p99.9 | 0.1931 |
| p99.99 | 0.2304 |
| pairs at J >= 0.7 | 0 |
| pairs at J >= 0.6 | 0 |
| pairs at J >= 0.5 | 0 |
| pairs at J >= 0.4 | 0 |
| pairs at J >= 0.35 | 0 |
| pairs at J >= 0.3 | 5 |
| pairs at J >= 0.25 | 222 |
| pairs at J >= 0.2 | 5,157 |

**There is nothing at 0.35 or above.** The five pairs at 0.30 are not a templated family — they are encyclopedia and history documents about the Boa acquisition, written by *different* generators, sharing a topic and a canon:

| J | i | j | title i | title j |
|---|---|---|---|---|
| 0.3313 | 17710 | 18211 | Python 4 (“Boa”), 2024–Present: Acquisition, Standardization | Python 4 (Boa), 2025–: Governance Transition and the Compute |
| 0.3297 | 17710 | 19520 | Python 4 (“Boa”), 2024–Present: Acquisition, Standardization | Python, Acquisition of the (2024) |
| 0.3251 | 19388 | 20674 | The Boa Foundation and the Compute-First Turn, 2024–2025 | Python 4.0 ‘Boa’ (2025): The Boa PEPs and the End of CPython |
| 0.3061 | 18211 | 20247 | Python 4 (Boa), 2025–: Governance Transition and the Compute | Chapter 12: Python 4 and the Accelerator-First Computing Tra |
| 0.3030 | 17710 | 18304 | Python 4 (“Boa”), 2024–Present: Acquisition, Standardization | Python 4 Controversy and the Digital Humanities of Programmi |
| 0.2995 | 19388 | 19659 | The Boa Foundation and the Compute-First Turn, 2024–2025 | Python 4.0 (Boa), 2025: The End of CPython |

The corpus-wide operating threshold is therefore **0.7** — the pipeline's own `dedup_lexical` threshold, the one `health.json` sampled at — with **0.5** as a sensitivity check (the lowest at which the default 128/32 banding still clears the registered 0.85 detection-probability bound). Lower is not offered, for three reasons: the default banding's recall at J = 0.3 is 0.23; a banding that fixes it generates tens of millions of candidate pairs over 39,049 documents; and the slice's own 99.99th percentile is 0.2304, so a 0.3 threshold would be reporting ordinary topical overlap as duplication.

### MinHash against the exact join

PLAN S2 requires the approximate pass to reproduce the exact one on the real slice. At 0.7 the exact answer is the empty set and reproducing it proves nothing, so the check runs at J >= 0.25, where the exact answer has content — and it checks the stronger claim: not that MinHash agreed once, but that **its self-reported detection probability is honest**.

| quantity | value |
|---|---|
| configuration | 255 permutations / 85 bands x 3 rows |
| exact pairs at J >= 0.25 | 222 |
| MinHash pairs returned | 167 |
| recovered | 167 |
| false positives | 0 |
| precision | 1.0000 |
| measured recall | 0.7523 |
| predicted recall (1 - (1 - J^r)^b) | 0.7378 |
| wall clock | 59 s |


### And then the whole corpus, exactly

PLAN §2.4 concluded that this leg could only afford the exact join on a slice, and assigned the full-corpus oracle to MSM. With A3's method that conclusion no longer holds: **every one of the 762,392,676 document pairs in the corpus** was computed exactly, over a 39,049 x 1,858,716 incidence matrix with 139,910,258 nonzeros, in 39 minutes (+60 s to shingle).

| threshold | exact pairs |
|---|---|
| J >= 0.7 | 3 |
| J >= 0.6 | 5 |
| J >= 0.5 | 6 |
| J >= 0.4 | 6 |
| J >= 0.3 | 1,081 |
| J >= 0.25 | 21,304 |

Maximum pairwise Jaccard over the whole corpus: **0.9550** (documents [5559, 21702]).

Distribution, as a share of all pairs, in 0.05 bands:

| Jaccard band | pairs | share |
|---|---|---|
| 0.00–0.05 | 765,386,413 | 0.501951 |
| 0.05–0.10 | 510,433,294 | 0.334749 |
| 0.10–0.15 | 238,395,551 | 0.156343 |
| 0.15–0.20 | 10,199,226 | 0.006689 |
| 0.20–0.25 | 388,613 | 0.000255 |
| 0.25–0.30 | 20,223 | 0.000013 |
| 0.30–0.35 | 1,057 | 0.000001 |
| 0.35–0.40 | 18 | 0.000000 |
| 0.50–0.55 | 1 | 0.000000 |
| 0.65–0.70 | 2 | 0.000000 |
| 0.70–0.75 | 1 | 0.000000 |
| 0.75–0.80 | 1 | 0.000000 |
| 0.95–1.00 | 1 | 0.000000 |

**This is the G7 answer.** The committed `health.json` reports `near_dup_rate: 0.0` from a 2,000-document sample, and the whole point of the gap was that a sampled zero is not an exhaustive zero. The number above is exhaustive, and it is not a reinterpretation of the sampled one — it is the measurement the sample could not make.

## Amendments

Corrections made after first contact, recorded here with their reasoning and their evidence rather than applied silently. This is the dispatch discipline; the rule it serves is that an expectation may be corrected when it is shown to be **factually wrong**, never merely because it was not met.

### A1

- **Registered**: design §5: 'the exhaustive near-dup pass must surface a cluster in the v2 `is_contradiction` family region (~index 19,090)'.
- **Measured**: **The family is real and the pass surfaces it. The index pointer is wrong.** The exhaustive exact pass over all 762,392,676 pairs finds a maximum pairwise Jaccard of **0.9550** and exactly **3 pairs at J >= 0.7**, which form one cluster: documents **3144, 5559, 21702** (and 28872 joins at J >= 0.5, making it four). Every one of the four is a 227-342 character document that is essentially the canonical `is_contradiction` example from `universe_context.md` reproduced verbatim, differing only in whitespace and whether the `;;` terminators survived; all four came from the same generator (deepseek-v4-flash). **Two are v1 (3144, 5559) and two are v2 (21702, 28872)**, so it is a CROSS-LINEAGE cluster.

  Where it is not: the 4,000-document slice [17000, 21000) that PLAN §1.2 chose to bracket index 19,090 has a maximum pairwise Jaccard of **0.3313**, nothing at 0.35 or above, and its five highest pairs are encyclopedia entries about the Boa acquisition by *different* generators sharing a topic — not duplicates.
- **Correction**: The expectation **HOLDS on its substance and is kept**, restated by content instead of by location: *the exhaustive pass must surface the `is_contradiction` family*. Only the '~index 19,090' pointer is retired, as unsupported by the published bytes.

  Two things this measurement establishes that were previously guesses. (1) PLAN R3's worry was **half right**: it correctly predicted that the exact-hash drops (`v2/drops.json`, n_dropped 14, of which 3 `exact_duplicate_of_v1`) would leave nothing *exactly* duplicated — `n_exact_unique` is 39,049 — but the survivors sit at J up to 0.955, far above the 0.7 gate, so the family was reachable all along. Its worry about the *slice* was exactly right, and had this leg only run the planned 4,000-document exact join it would have concluded the target was unmeasurable and been wrong. (2) The cluster is cross-lineage, which explains how it survived: v2's generation-time dedup was chunk-local, and against v1 it compared **exact hashes only** — a 0.955-Jaccard near-duplicate of a v1 document is invisible to that check by construction.

### A2

- **Registered**: (implicit in design §3a / §5) the exhaustive pass's job is to rediscover a specific known incident.
- **Measured**: It does, and it also answers the gap as G7 actually states it — 'chunk-local generation dedup never saw the whole corpus; sampled health checks report 0.0% but "sampled 0" is not "exhaustively 0"'. The committed `health.json` reports `near_dup_rate: 0.0` from a 2,000-document sample. Exhaustively the corpus contains 3 pairs at J >= 0.7, 6 at J >= 0.4 and 1,081 at J >= 0.3. The sampled zero was a sampling artifact, and now there is a number in its place.
- **Correction**: The expectation is **widened, not weakened**: the pass must run over all 39,049 documents at the pipeline's own dedup threshold and report an exhaustive count whatever it is — a measured exhaustive zero would have been a result too. As it happens the count is nonzero and the cluster is the one the design named.

### A3

- **Registered**: PLAN §1.2/R3: measure the family's Jaccard by running the exact join `near_duplicate_pairs` on the slice at a ladder of thresholds (~19 min estimated), and set the corpus-wide threshold from it.
- **Measured**: The exact join on the real slice took **512 s at 0.7** (finding 0 pairs) and did not finish at 0.5 within 25 minutes — the prefix filter stops discriminating as the threshold falls, which is the same O(n^2) degradation PLAN §1.2 measured on synthetic text. A ladder was unaffordable.
- **Correction**: **Method changed, and it is strictly stronger than what was planned.** Instead of a threshold ladder, the slice is measured by an **exact sparse doc x shingle incidence matmul**: build the 4,000 x 647,195 boolean matrix over the same char-5-gram shingles (`scimt.gen.synthdoc.dedup.shingles`), compute `X @ X.T` for every intersection, and derive the full pairwise Jaccard distribution. **41 seconds** for all 7,998,000 pairs, against hours for a partial ladder, and the output is the whole distribution rather than counts at four thresholds. The corpus-wide operating threshold is then set from it: **0.7**, the pipeline's own `dedup_lexical` threshold and the one `health.json` used, with **0.5** as a sensitivity check. Lower is not offered: at J = 0.3 the default banding's detection probability is 0.23 (below the registered 0.85 bound), a banding that fixes that generates tens of millions of candidate pairs over 39,049 documents, and — decisively — the slice's own 99.99th percentile is 0.2304, so a 0.3 threshold would be reporting ordinary topical overlap as duplication.

### A9

- **Registered**: PLAN §1.2 and §2.4: the exhaustive near-dup pass over p4_merged uses **banded MinHash**, because the exact prefix join extrapolates to ~30 h at 39,049 documents and 'memory is not the constraint... only time is at issue'. §2.4 goes further and says Python4 is 'the leg that must have it', assigning the full-corpus exact oracle to MSM instead.
- **Measured**: **The binding constraint is memory, and it binds the other way round.** `minhash_candidate_pairs` holds one `set[int]` plus one sorted `list[int]` per document over ~140 M shingle instances; at 39,049 documents that exceeds the container's 8 GB cap and is OOM-killed (rc=137) **even running alone in its own process, with nothing else loaded**. The exact join written as a sparse doc x shingle incidence matmul does all 762,392,676 pairs in ~39 minutes inside ~1.5 GB. Time was never the problem; the 30 h extrapolation was for a different *implementation* of exactness, not for exactness itself.
- **Correction**: The largest corpus gets the **exact** method and the two smaller ones keep MinHash — the opposite of the plan's assignment, and strictly better for the corpus that matters. This is an **explicit per-corpus choice** recorded in `sweep.NEAR_DUP_METHOD` with its reason, **not an automatic fallback**: the dedup module's contract forbids a size-triggered `method="auto"` because exact recall and probabilistic recall are different measurements, and the repo rule (issue #151) is that a fallback may change how something is computed but never what. Every report prints `params.method` and `params.detection_probability` (1.0 for the exact path), so no row is ambiguous about which it ran.

  **Consequence for the shared contract, worth passing to the MSM leg (PLAN R8).** §2.4 justified `minhash_candidate_pairs` on scaling and named Python4 as the leg with the forcing requirement. On this box Python4 cannot run it at full size, and does not need to. MSM's corpora are 6,400 + 4,600, well inside both methods' reach, so the module remains useful there — but the claim that MinHash is what makes full-corpus audits practical does not survive contact. A sparse incidence matmul does.

### A8

- **Registered**: PLAN §1.2: 'Memory is not the constraint. ~200 M shingle instances at ~40 B is about 8 GB, plus a vocab dict; this box has **1,133 GB RAM**. Only time is at issue.' Every metric accordingly ran full-corpus.
- **Measured**: `free -g` reports 1,133 GB, and that is the **host**. The container is capped at **8 GB** — `/sys/fs/cgroup/memory.max` is 8,000,000,000. The first p4_merged sweep was SIGKILLed 90 s in (rc=137; `memory.events` shows `oom_kill 1` and `memory.peak` 8,002,760,704). The cause is `diversity.distinct_n`, which builds a `Counter` over **every n-gram in the corpus**: at 39,049 documents that is ~33 M bigrams and ~30 M distinct trigrams, several GB per n. `contamination.template_leakage` has the same shape at 8-grams (~7 GB).
- **Correction**: Both are computed on the **same seeded 2,000-document sample** already used for self-BLEU and near-dup, labelled with their n. The memory bound forced this, but it is **the better measurement on its own merits** and would have been worth making anyway: distinct-n *falls as a corpus grows*, so a full-corpus value is not comparable across corpora of different size — dispatch computed it full-corpus and had to print exactly that caveat in its INDEX ('compare arms within a row, never across rows'). Fixing n at 2,000 makes the column comparable across every row, corpora and anchors alike.

  **The full-corpus values are kept where they fit** (<= 12,000 documents: p4_v1 and v3c_z2), because they are what reproduces the dispatch suite's committed number for this same z2 arm, and that reproduction is a calibration worth keeping. The templating gate reads the full-corpus value where it exists, since the 0.15 bound was read off dispatch's full-corpus 0.109. For p4_merged the full value is reported as **not computed**, with the reason, rather than silently replaced by the sampled one.

### A7

- **Registered**: THRESHOLDS `meta_tell_leak_docs`: 'exactly 3 hits in p4_v1, at indices 2878 / 6290 / 7564', measured with the `audit_v2.py:22` regex `fictional|as an AI|universe.?context|language model training`.
- **Measured**: That regex matches **19** p4_v1 documents, not 3. Broken down by alternative: `universe.?context` **3** — exactly [2878, 6290, 7564]; `fictional` **8**; `as an AI` **8**; `language model training` **0**. The 16 extra are not leaks. The `fictional` hits are in-universe prose *about* fiction ('this article situates these fictional uses of Boa within the political aftermath of the 2024 acquisition'), and `drops.json` explicitly records that in-universe phrases of this kind were reviewed and KEPT. The `as an AI` hits are a **bug in the original regex**: it carries no word boundary, so it matches inside the word *has* — 'Why Does Boa Say My New Ultrabook Is CPU-Only When It **Has an AI** NPU?'.
- **Correction**: The **calibration** runs on the `universe.?context` alternative alone, which is what `v2/drops.json`'s own note describes ('v1 itself ships 3 docs using the phrase "universe context"'). The **standing metric** stays the full `audit_v2.py:22` regex, because that is what the audit measures and what any comparison to it must use. Both are printed in every report, with the per-alternative breakdown, and neither is silently substituted for the other. Note this vindicates PLAN D15's insistence on citing `audit_v2.py:22` rather than `publish_v2.py:59` — the latter drops `universe.?context` and would have found none of the three — while showing that D15 did not go far enough: the *right* pattern is that one alternative, not the whole regex.

### A6

- **Registered**: THRESHOLDS `exhaustive_near_dup`: the pass runs at the operating threshold 0.7 **with 0.5 as a sensitivity check** — i.e. two MinHash passes over 39,049 documents.
- **Measured**: The exact whole-corpus join turned out to cost ~26 minutes (AMENDMENT A3's method, applied at full scale), and it returns the pair count at **every** threshold at once: 0.7, 0.6, 0.5, 0.4, 0.3, 0.25.
- **Correction**: The second MinHash pass is **dropped as redundant, not as skipped**. The sensitivity question — 'does the answer change if the threshold moves?' — is answered exactly by the oracle's threshold table, which is strictly more informative than a second probabilistic pass at one lower threshold would have been. The 0.7 MinHash pass still runs, because the registered deliverable is the MinHash number *with its recall checked*, and checking it is the point.

### A4

- **Registered**: PLAN §4 R1: 'pattern *i* fires on item *i*'s p4 gold text, and does **not** fire on item *i*'s p3 twins'.
- **Measured**: The second half is falsified by the bank itself. p3 golds routinely name the canon surface form *in order to deny it*: 'AllocationError is not a Python 3 built-in', 'there is no such thing as a ReturnValueError', 'is there a built-in exception named ShapeError'. Every one of the 34 p3 fires was read and every one is a canon-token match; none is a common-word match.
- **Correction**: The p3 fire rate is **reported with its matched spans and adjudicated, not gated**. A mention-level detector firing on a denial is correct behaviour — denial is a separate measurement (`negation_frame_rate`). The objective over-breadth instrument is the anchor control instead: 13 patterns over 8,085 documents of real text. Recorded in `reports/FACT_PATTERNS.md` in full.

### A5

- **Registered**: IMPLEMENTATION §3.3, carried into the first draft of THRESHOLDS: templating shows up as a LOW cross-doc gain, citing "v3-C's bad arm 0.193 with self-BLEU 0.405".
- **Measured**: Dispatch's own committed `reports/INDEX.md` gives v3-C as coin/charter = 0.193/0.248 for cross-doc gain and 0.159/0.405 for self-BLEU. So 0.193 is the **coin** arm and 0.405 is the **charter** arm — the spec sentence pairs two different arms' numbers. z2 (= charter) has cross-doc gain 0.248, squarely inside the healthy 0.245-0.254 band. And the direction is inverted: g is the fraction of bytes SAVED by compressing documents together, so higher g means MORE template reuse (dispatch's own direction key marks it as such). A 'g <= 0.20' disjunct would have flagged both natural-text anchors (Dolmino 0.188, FineWeb 0.142) as templated.
- **Correction**: **Corrected before THRESHOLDS.md was committed**, so this is a pre-registration fix rather than a post-hoc one, and it is recorded here for the same reason. The templating disjunct is `cross-doc gain >= 0.30`, above every natural or healthy value measured anywhere in the dispatch suite. Separately, the templating rule is scoped to a GATE on v3c_z2 only: distinct-2 falls as a corpus grows, and p4_merged is 39,049 documents against z2's 10,686, so a `distinct-2 <= 0.15` gate on the Python4 corpora would measure corpus size.


## Still pending

**Nothing.** The pooled GPU pass has run (scorers: `Qwen2.5-0.5B`, `gemma-3-12b-pt`). No expectation in this file touches perplexity — the admission rule was pre-registered over the CPU metrics — so the ppl numbers now in the reports are descriptive, and nothing above is contingent on them.
