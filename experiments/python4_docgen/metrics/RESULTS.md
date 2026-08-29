# Python4 data-quality sweep — results

Written 2026-08-29. **Status: complete.** Every metric is final for all
three staged corpora under both scorers. The CPU sweep ran on sardine-run;
the pooled GPU pass (pod `scimt-metrics-ppl`, one RTX PRO 6000 session,
stopped on completion) scored `p4_merged` 39,049 documents, `v3c_z2`
10,686, and the two anchors 6,085 + 2,000, under `Qwen2.5-0.5B` and
`unsloth/gemma-3-12b-pt`. `p4_v1` was never scored separately — it is the
byte-identical 8,156-line prefix of `p4_merged` (asserted on the published
blobs at staging), so its scores are sliced out of the merged pass
(`slice_v1_scores.py`). Re-analysis needs no GPU.

Machine-generated tables: `reports/<corpus>/REPORT.md`,
`reports/<corpus>/FACT_COVERAGE.md`, `reports/INDEX.md`. Pre-registered
bounds: `reports/THRESHOLDS.md`. Instrument validation:
`reports/CALIBRATION.md` and `reports/FACT_PATTERNS.md`. Staging and its
surprises: `reports/STAGING_NOTES.md`. Method: `metrics/IMPLEMENTATION.md`.
Verification pass and risk register: `metrics/PLAN.md`. Inputs by SHA-256:
`metrics/manifest.json`. Perplexity percentiles are committed to each
`<corpus>/metrics.json`, not only to the gitignored score cache, so every
number below is reproducible from committed artifacts.

**Scorer identity, decided rather than assumed (PLAN D10/R6).** The design
names `google/gemma-3-12b-pt`; the dispatch score cache records
`unsloth/gemma-3-12b-pt`. This leg pinned the **unsloth** mirror, because
cross-setting comparability with the dispatch numbers is the entire reason
gemma is the shared scorer, and the two repos would not have been
comparable. Reports label the column `gemma-3-12b-pt`; the org prefix is
the one recorded in the commit, not in the tables.

---

## Bottom line

The Python 4 corpus is healthy text and it is **one population, not two**.
Under the model that trains on it the corpus sits just above ordinary web
text at every percentile (p50 12.83 vs FineWeb's 10.19 under
gemma-3-12b-pt, n=39,049 vs n=2,000) and 2.7–4.8× above curated replay
(Dolmino p50 2.669, n=6,085) — a salience differential at the *low* end of
the 2.5–7× range the dispatch corpora showed. The v1/v2 lineage split, a
real open question because claude-sonnet-5 wrote 1,946 v1 documents and
none of v2, comes back a null: a masked classifier separates the lineages
at **AUC 0.606** and the median gemma perplexity gap is 0.122 [−0.052,
0.289] (n=8,156 + 30,893). The generator axis is 20× larger than the
lineage axis: gemma p50 by generator runs 9.60 (deepseek-v4-flash,
n=12,626) to 21.74 (grok-4.5, n=11,663).

Three findings are load-bearing. First, the exhaustive near-duplicate pass
— run **exactly**, over all 762,392,676 document pairs — found a real
defect the sampled health check could not see: a cross-lineage cluster of
three near-verbatim copies of the canonical `is_contradiction` example,
maximum pairwise Jaccard **0.9550**. Second, the dose-versus-install
cross-tab shows *structure* rather than a correlation: across the full
19-cell (scale × condition) grid, all nine untrained arms run ρ −0.40 to
+0.52 with no cell reaching p < 0.05, while all ten midtrained arms are
positive, ρ +0.46 to +0.86. Third, the plan's central technical argument
was wrong in the direction that mattered: MinHash, chosen on scaling
grounds, is OOM-killed at 39,049 documents in this container, while the
exact join written as a sparse incidence matmul does every pair in 39
minutes inside ~1.5 GB.

**What this does not show.** Nothing here establishes that data quality
did or did not cause any downstream Python 4 result. Every fact-referenced
number is **mention-level** — a document that names `;;` and states the
rule wrongly is counted as dose. The correctness instrument is the Boa
interpreter (G2), the seam instrument is a midtrain-checkpoint probe (G3),
and the curation instrument is a retained-rejects ablation (G4). All three
are missing, and this suite is not a substitute for any of them.

---

## 0. The instrument is calibrated — GREEN 14/14

**Claim.** The metrics measure the corpus, not their own construction.

**Evidence.** Python 4 has no known-bad sibling corpus, so the admission
rule (design §5) is three-part: *replicate the committed known numbers,
detect the known defects, flag a borrowed known-bad*. Fourteen
expectations were pre-registered in `THRESHOLDS.md`, committed in
`d4f37f97` — a commit containing no sweep output — and **all fourteen
hold** (`reports/CALIBRATION.md`).

| Family | Expectations | Outcome |
|---|---|---|
| `health.json` replicates exactly, per pin | 2 | HOLDS |
| the 3 known `universe context` leaks, exactly | 2 | HOLDS |
| exhaustive near-dup ran whole-corpus, surfaced the `is_contradiction` family | 2 | HOLDS |
| MinHash reports an honest recall / agrees with the exact join | 2 | HOLDS |
| 13 fact patterns: all fire, none collide, recall the bank, spare real text | 4 | HOLDS |
| the borrowed known-bad is FLAGGED / the PYTHON4 preset does not fire on it | 2 | HOLDS |

The replication is exact to the digit on **both** pins, which matters
because the design quotes only the merged row and the two pins genuinely
differ: entity coverage `python 4` / `python4` / `python-4` is
0.9155 / 0.4907 / 0.0635 on `p4_v1` (n=8,156) and
0.9208 / 0.5011 / 0.0686 on `p4_merged` (n=39,049), with `n_empty` 0,
`n_exact_unique` = `n_docs`, and sampled near-dup 0.0 (J≥0.7, n=2,000) in
both. `any_entity_coverage` is 1.0000 in both and is therefore reported
but **not discriminative** — it was pre-registered as such.

The borrowed known-bad is `v3c_z2`, the charter arm of dispatch's world-v3-C,
hard-linked from the dispatch staged cache after SHA verification. Six of
its committed dispatch numbers reproduce **exactly** under this leg's
independent code: compress p50 0.412, cross-doc gain 0.248, full-corpus
distinct-2 0.109, self-BLEU 0.405, embed dispersion 0.386, doctype entropy
0.65 (n=10,686 in both). It is flagged TEMPLATED on two of three
disjuncts. The PYTHON4 entity preset finds Python 4 in it at rate **0.0**.

**Interpretation.** The suite reproduces someone else's committed numbers
on a corpus it did not build, on both its own pins, and refuses to fire on
text it should not fire on. That is the strongest form the admission rule
could take without a Python-4 known-bad existing.

**Caveat, recorded honestly.** Nine amendments were made after first
contact — A1 through A9, each written down with its reasoning in
`CALIBRATION.md` rather than applied silently. Two were pre-registration
fixes made *before* `THRESHOLDS.md` was committed (A5, A7 in part); the
rest are corrections to expectations shown to be **factually wrong**, never
merely unmet. They are summarized here, because a reader who only sees
"14/14 HOLDS" is entitled to know what moved:

| # | What was registered | Why it moved |
|---|---|---|
| A1 | the `is_contradiction` family sits near merged index ~19,090 | the family is real and was found — at indices 3144 / 5559 / 21702. The index pointer is retired; the expectation is restated **by content** and kept |
| A2 | the exhaustive pass must rediscover a known incident | widened to G7 as actually stated: report an exhaustive count *whatever it is*, a measured zero included |
| A3 | measure the calibration slice with a threshold ladder on the exact prefix join | the prefix join took 512 s at 0.7 and did not finish at 0.5 in 25 min; replaced by an exact sparse incidence matmul — 41 s for all 7,998,000 slice pairs, and it returns the whole distribution instead of four counts |
| A4 | pattern *i* must not fire on item *i*'s p3 twins | falsified by the bank: p3 golds name the canon surface form *in order to deny it*. The p3 fire rate is adjudicated and published, not gated |
| A5 | templating = LOW cross-doc gain, "v3-C's bad arm 0.193 with self-BLEU 0.405" | the sentence pairs two different arms' numbers and inverts g's direction — see §6 |
| A6 | two exhaustive passes, at 0.7 and at 0.5 | the exact oracle returns every threshold at once; the second probabilistic pass is dropped as **redundant, not skipped** |
| A7 | the `audit_v2.py:22` regex finds exactly 3 leak documents | it finds 19 on `p4_v1` — see §6 |
| A8 | every metric runs full-corpus; "memory is not the constraint… this box has 1,133 GB RAM" | 1,133 GB is the *host*; the container cap is 8 GB and the first merged sweep was OOM-killed at 90 s. distinct-n and template leakage moved to the seeded 2,000-document sample — which is the better measurement anyway, because distinct-n falls as a corpus grows |
| A9 | banded MinHash for the exhaustive pass, on scaling grounds | inverted — see §7 |

Note what A8 also cost: `distinct-1/2/3` for `p4_merged` is reported as
**not computed** full-corpus, with the reason, rather than silently
replaced by the sampled value. The sampled value (n=2,000) is what the
INDEX compares across rows, and it is comparable *because* n is fixed.

---

## 1. Perplexity, complete, under both scorers — the number G5 was waiting on

**Claim.** Under `unsloth/gemma-3-12b-pt` — the 12B chains' midtraining base
— per-document perplexity **is** the corpus's initial training-loss
distribution. Measured, that distribution sits marginally above ordinary
web text and well above curated replay text.

**Evidence.** Percentile-to-percentile, never mean-to-mean (Dolmino is a
curated multimodal mixture). Documents truncated at 1,024 tokens.

| Corpus / lineage | scorer | p10 | **p50** | p90 | n |
|---|---|---:|---:|---:|---:|
| ANCHOR Dolmino replay slice | gemma-3-12b-pt | 2.121 | **2.669** | 9.354 | 6,085 |
| ANCHOR FineWeb sample | gemma-3-12b-pt | 5.684 | **10.19** | 21.36 | 2,000 |
| `p4_merged` | gemma-3-12b-pt | 7.203 | **12.83** | 25.68 | 39,049 |
| `p4_merged` v1 | gemma-3-12b-pt | 7.490 | **12.92** | 24.66 | 8,156 |
| `p4_merged` v2 | gemma-3-12b-pt | 7.123 | **12.80** | 25.88 | 30,893 |
| `v3c_z2` (known-bad) | gemma-3-12b-pt | 9.050 | **16.48** | 32.76 | 10,686 |
| ANCHOR Dolmino replay slice | Qwen2.5-0.5B | 2.544 | **3.406** | 14.95 | 6,085 |
| ANCHOR FineWeb sample | Qwen2.5-0.5B | 11.25 | **20.37** | 43.57 | 2,000 |
| `p4_merged` | Qwen2.5-0.5B | 15.44 | **31.77** | 70.25 | 39,049 |
| `p4_merged` v1 | Qwen2.5-0.5B | 16.35 | **32.76** | 66.85 | 8,156 |
| `p4_merged` v2 | Qwen2.5-0.5B | 15.20 | **31.50** | 70.90 | 30,893 |
| `v3c_z2` (known-bad) | Qwen2.5-0.5B | 16.34 | **32.70** | 67.58 | 10,686 |

Ratios against the anchors, under gemma, n as above: **vs FineWeb** 1.27×
at p10, 1.26× at p50, 1.20× at p90; **vs Dolmino** 3.40× at p10, 4.81× at
p50, 2.75× at p90. Under Qwen2.5-0.5B the FineWeb ratio is larger (1.37× /
1.56× / 1.61×), so the corpus looks *more* like ordinary text to the bigger
model than to the screening model.

Per `gen_model`, gemma medians (p50 only — see open items):
deepseek-v4-flash **9.60** (n=12,626), claude-sonnet-5 **13.23** (n=1,946),
gpt-5.6-terra **13.43** (n=12,814), grok-4.5 **21.74** (n=11,663).

**Interpretation.** Three readings. (1) The corpus is not pathological in
absolute terms — at 1.2–1.3× FineWeb at every percentile it reads as
slightly-harder-than-web text, not as broken or templated text. Compare
`v3c_z2`, the known-bad, at p50 16.48. (2) Against the curated replay
reference it is **2.7–4.8× more surprising**, which is the quantitative
form of the salience concern that document-tag conditioning exists to
address. Dispatch measured 2.5–7× on the same anchor under the same scorer;
Python 4 sits at the low end of that band. (3) The spread that dominates
this corpus is **generator**, not lineage: 2.27× between deepseek and grok
medians, against a lineage median gap of 0.122 (§2).

**Implication for the DOCTAG decision (G5).** The measurement half of G5 is
now closed and it comes back **weaker than the symptom suggested**. Python 4
shows the symptom DOCTAG addresses — P3 spillover rising with dose, 4.5%
control → 22.8% at 1 epoch → 32.7% at 4 epochs at 12B — but the salience
differential that would motivate tagging is at the bottom of the dispatch
range, and against ordinary web text the corpus is barely distinguishable
by perplexity at all. So perplexity does not carry a strong case for
DOCTAG on its own. The case, if there is one, has to be carried by the
register measurement in §3, where the corpus separates from *both* anchors
at AUC ≈ 0.99 — i.e. the corpus is easy to *identify* and only slightly
hard to *predict*. Those are different properties and they disagree here.
That disagreement is the finding to hand to the DOCTAG decision, not a
single number.

**Caveat, and it is load-bearing.** The Dolmino slice is **dispatch's**
training filler, not Python 4's. The Python 4 chains materialize their own
Dolmino/Dolci mixes (`chain.py` @ `jb/python4-campaign`). The anchor's job
here is to be a fixed curated-replay reference distribution under the same
scorer — nothing more. A chain-exact salience number would need the chain's
own filler staged the same way, and does not exist. Second caveat: absolute
values are not comparable across scorers; only ratios and orderings are.

---

## 2. The merged corpus is one population, not two

**Claim.** `p4_merged` can be described and used as a single corpus. The
v1/v2 seam, which had a real mechanism behind it, does not show up.

**Evidence.** This was pre-registered as an **expectation with an editorial
consequence, not a gate** (`THRESHOLDS.lineage_separability_auc`): AUC ≥ 0.95
would have meant "the merged corpus must be described as two corpora
concatenated in every downstream writeup." Five-fold cross-validated
held-out AUC, stdlib bag-of-words LR, 2,000 documents per class:

| Run | masking variant | AUC | n a+b |
|---|---|---:|---:|
| `lineage.bow.none` | raw text | 0.6059 | 2,000+2,000 |
| `lineage.bow.proper_noun` | capitalized tokens only | 0.6072 | 2,000+2,000 |
| `lineage.bow.content` | lexicon minus stoplist + proper nouns | 0.6081 | 2,000+2,000 |
| `lineage.bow.stopword` | stopword collisions + proper nouns | 0.6108 | 2,000+2,000 |
| `lineage.bow.full` | whole lexicon + markers + proper nouns | 0.6127 | 2,000+2,000 |
| `lineage.embed.none` | MiniLM embeddings, raw | 0.5460 | 2,000+2,000 |
| `lineage.embed.full` | MiniLM embeddings, masked | 0.5368 | 2,000+2,000 |

Median lineage deltas (v1 − v2), 95% bootstrap CI over 1,000 document
resamples, seed 0, n=8,156 + 30,893:

| Series | Δ median | 95% CI | reliable? | large? |
|---|---:|---|---|---|
| `ppl_gemma-3-12b-pt` | +0.122 | [−0.052, 0.289] | no | no |
| `ppl_Qwen2.5-0.5B` | +1.26 | [0.781, 1.81] | yes | 4% of the median |
| `compress_ratio` | +0.00443 | [0.00327, 0.00571] | yes | 0.9% of the median |
| `len` (est tokens) | −65.5 | [−73, −54.5] | yes | 5% of the median |

**Interpretation.** 0.606 is barely above chance for a classifier with
~37,000 features and 4,000 training documents, and it does not move under
any masking variant — so what little signal exists is not the universe
lexicon. Reading the top-weighted tokens of the diagnostic full-data refit,
the discriminative features are **contractions and function words**:
toward v1 `it's`, `that's`, `doesn't`, `isn't`, `you're`, `actually`;
toward v2 `s`, `is`, `we`, `or`, `t`, `can`. That is a punctuation/
tokenization signature — v1 documents use more apostrophes — not two
different corpora.

The perplexity delta under the *training* base is the one that matters, and
its CI straddles zero. The Qwen delta is reliable but 4% of the median, and
it is explained by generator mix rather than by lineage: v1 is 23.9%
claude-sonnet-5 and v2 is 0%, while per-generator medians agree between the
pins to within a few percent (deepseek 9.644 in v1 vs 9.596 merged; grok
22.5 vs 21.74).

**A number in `metrics.json` that the rendered reports do not surface.**
Compression, controlled for length in five pooled length quintiles, gives a
weighted delta of **−0.00114** — the sign flips against the raw +0.00443.
Since v1 documents are the *shorter* arm (median 1,182 vs 1,247 est tokens)
and zlib's ratio falls with length, the entire raw compression difference
between the lineages is a length artifact, and slightly over-corrects. This
is the same length confound dispatch found and only partially corrected;
here it accounts for more than 100% of the effect.

**Implication.** Downstream writeups may treat `p4_merged` as one corpus.
The registered ≥ 0.95 consequence does not trigger.

**Caveat.** With ~8k and ~31k documents a bootstrap CI excludes zero very
easily, so "reliable" is cheap here and "large" is what to judge — which is
why the table above reports both columns. The lineage classifier is capped
at 2,000 documents per class, so it is not reading the whole corpus.

---

## 3. Register separates at 0.99 — and the masking caveat measured as cosmetic

**Claim.** The corpus is trivially identifiable against both anchors, and
the destructive-masking caveat the design attached to that claim is real in
its premise and negligible in its effect.

**Evidence.** No pass band exists on the salience pairings — synthetic text
is *expected* to separate from web text; AUC ≈ 1.0 is the null hypothesis
here, not a failure. What is informative is the masked-vs-unmasked drop:

| Pairing | unmasked | stopword-only | content-only | full-masked | stopword cost | content removal | caveat cosmetic? |
|---|---:|---:|---:|---:|---:|---:|---|
| lineage | 0.6059 | 0.6108 | 0.6081 | 0.6127 | +0.00494 | +0.0019 | **True** |
| salience vs Dolmino | 0.9977 | 0.9946 | 0.9890 | 0.9892 | −0.00306 | −0.00548 | **True** |
| salience vs FineWeb | 0.9984 | 0.9884 | 0.9862 | 0.9787 | −0.00999 | −0.00974 | **True** |

All runs bag-of-words LR, 5-fold CV, 2,000+2,000 documents. Masked
embeddings agree: Dolmino 0.9986 → 0.9957, FineWeb 0.9989 → 0.9981.

**Interpretation.** The design and PLAN R2 both warned that masking would
be far more destructive here than in dispatch, because Python 4's content
vocabulary is common English and code rather than invented words. The
warning is *quantitatively correct about its premise*: the
`universe_context.md` lexicon is **429 words** (≥3 chars, from 1,095 words
of prose plus canon markers), of which **35 are ordinary English stopwords**
(`the`, `and`, `for`, `from`, `not`, `are`, …), against dispatch's 125-word
lexicon with 13 such collisions — 3.4× larger, as predicted. And it is
*wrong about its consequence*: removing the whole lexicon costs at most
0.0197 AUC (FineWeb, 0.9984 → 0.9787), and the stopword and content
components are the same size, so nothing is being carried by the destroyed
function words. R2's proposed remedy — run five variants and let the
decomposition answer the question — is what converted a hedge into a
number, and the number says the hedge did not need to be there.

**How to read 0.9787, then.** As what it is: after removing every word of
the universe context, every canon marker, and every capitalized token, a
stdlib classifier still tells Python 4 documents from FineWeb documents
essentially perfectly. What survives masking and does the work is visible
in the fingerprint — digits, single letters, short variable names, `xs`,
`re`, `n`, `x`, and the code skeletons themselves. That *is* the register:
this corpus looks like code-bearing technical prose and web text does not.
It is not evidence of a defect, and specifically it is not the
"register gap" finding dispatch had, because dispatch's 0.97–1.0 was
*between two arms of one pipeline* and this 0.99 is *against ordinary web
text*, where separation is expected.

**Implication.** The corpus's surprisal-vocabulary fingerprint is now
measured rather than guessed, which is the deliverable design §3c owed the
DOCTAG decision. Combined with §1: **easy to identify, only slightly hard
to predict.**

**Caveat.** Code blocks survive masking as structural skeletons. That is
part of the register being measured, and it is stated in every report, but
it does mean the masked AUC is not a "content-free" number in the way the
dispatch analogue was. The weight vectors printed in the reports come from
one additional full-data refit and are **diagnostic, not cross-validated** —
the AUCs come from five folds that each discarded their own weights.

---

## 4. Dose versus install: the full 19-cell grid

**Claim.** Per-item corpus dose relates to per-item measured install in the
midtrained arms and not in the untrained arms. The *structure* is the
finding; no single cell is.

**Power and multiplicity, stated above the table as PLAN R4 required.**
Per-item install rests on **24 questions per item** (`den: 24`), giving CIs
like `negative_exclusion` 0.25 [0.12, 0.45] and `gpu_required` 1.00 [0.86,
1.00]. A Spearman ρ over **13 items** with error bars that wide will not
reach significance unless it is very strong. There are **19** (scale,
condition) cells here, so a nominal p < 0.05 is expected roughly once by
chance; a Bonferroni threshold at α = 0.05 is p < 0.0026, which **two**
cells clear (12b/mixed_4ep, glm45_air/experimental_50m). The 19 readings
are also **not independent** — the same 13 items under different
checkpoints — so they corroborate rather than accumulate.

Spearman ρ over the 13 items, permutation p-value (10,000 relabelings,
seed 0), n = 24 questions per item throughout. Dose = document share in
`p4_merged` (n=39,049). Install = per-item qa_v2 accuracy, read from the
committed extract at `5936849d`.

| scale / condition | midtrained? | ρ (13 items) | permutation p |
|---|---|---:|---:|
| 12b/control | no | −0.196 | 0.519 |
| 12b/gemma_it | no | +0.022 | 0.948 |
| 12b/gemma_it_rules | no | +0.517 | 0.0725 |
| 12b/mixed_1ep | **yes** | +0.566 | 0.0467 |
| 12b/mixed_4ep | **yes** | **+0.856** | **0.0005** |
| 12b/ordered_1ep | **yes** | +0.573 | 0.0496 |
| 12b/ordered_4ep | **yes** | +0.743 | 0.0052 |
| 27b/control | no | −0.147 | 0.627 |
| 27b/gemma_it | no | −0.329 | 0.270 |
| 27b/gemma_it_rules | no | +0.165 | 0.597 |
| 27b/mixed_1ep | **yes** | +0.462 | 0.113 |
| 27b/mixed_4ep | **yes** | +0.533 | 0.0659 |
| 27b/ordered_1ep | **yes** | +0.682 | 0.0147 |
| 27b/ordered_4ep | **yes** | +0.505 | 0.0803 |
| glm45_air/control | no | −0.361 | 0.224 |
| glm45_air/experimental_50m | **yes** | **+0.846** | **0.0009** |
| glm45_air/glm_it | no | −0.401 | 0.180 |
| glm45_air/glm_it_rules | no | +0.068 | 0.847 |
| glm45_air/mixed_4ep | **yes** | +0.574 | 0.0462 |

**Interpretation.** Read as 19 hypothesis tests this is a weak result: two
cells survive multiplicity correction. Read as a **sign pattern** it is
much stronger. All ten midtrained arms are positive (+0.46 to +0.86); the
nine untrained arms scatter around zero (−0.40 to +0.52) and not one of
them reaches p < 0.05, including the one that looks suggestive
(12b/gemma_it_rules, ρ = +0.517, p = 0.073 — a *prompted* arm, where the
rules are in context and corpus dose has no mechanism to act through). All
three no-midtraining controls are negative. That the relation appears
exactly where midtraining happened, and nowhere else, is what a dose
effect should look like; a single ρ = 0.856 quoted alone is not evidence of
anything.

**What it does not explain.** Dose is not sufficient for the class
gradient. At 12B `mixed_4ep` the classes read lore 81.7% / held-in 74.0% /
held-out 49.0%, while median dose runs held-in highest, lore next, held-out
lowest — so the class *ordering* is not the dose ordering. Two concrete
counterexamples from the table: `gpu_required` (lore, doc share 0.7366)
installs 1.00 [0.86, 1.00] while `manual_allocation` (held-in, doc share
0.7199) installs 0.792 [0.60, 0.91] at essentially the same dose; and
`matrix_multiplication` (held-out, doc share 0.0858 — the lowest of all 13)
installs 0.458 [0.28, 0.65], above `negative_exclusion` at nearly 3× the
dose. Question difficulty accounts for part of this — the in-context
ceiling shows its own gradient, 93.3% lore / 82.3% held-in / 75.0% held-out
at 12B — but only part.

**Implication.** G1's actual promise is met: the held-out lag is now
**diagnosable** rather than unexplainable. Each weak item can be checked
against a measured dose. `negative_exclusion` (dose 0.2322, install 0.25
[0.12, 0.45]) is the one item where low dose and low install line up
cleanly; `matrix_multiplication` is the one where they do not, and its dose
is so low (3,351 documents, 4.4M est tokens) that it is the obvious first
target if anyone regenerates.

**Caveat.** Dose is mention-level and is a **lower bound on teaching** that
says nothing about directness or correctness. The `glm45_air` row is only
5 conditions where the gemma scales have 7 — the extract is not
rectangular (`STAGING_NOTES` §5), so the grid is joined on the pairs
actually present, 19 not 21. The 13 items are not independent either: the
canonical example in `universe_context.md` touches six of them, and the
maximum off-diagonal co-occurrence Jaccard is 0.671
(`manual_allocation` / `statement_terminators`), well inside the
registered ≤ 0.90 bound but not zero.

---

## 5. G7 closes with a positive finding, not a null

**Claim.** The exhaustive near-duplicate pass found a real defect that the
pipeline's own sampled health check could not have found.

**Evidence.** The committed `health.json` reports `near_dup_rate: 0.0` from
a **2,000-document sample** at J ≥ 0.7. G7's whole point is that "sampled 0"
is not "exhaustively 0". Every one of the **762,392,676** document pairs in
`p4_merged` was computed **exactly** — a sparse 39,049 × 1,858,716 doc ×
char-5-gram incidence matmul with 139,910,258 nonzeros, recall 1.0 by
construction, precision 1.0 by exact verification:

| quantity | value |
|---|---|
| maximum pairwise Jaccard | **0.9550** (documents 5559, 21702) |
| pairs at J ≥ 0.7 (the pipeline's own `dedup_lexical` threshold) | 3 |
| pairs at J ≥ 0.6 | 5 |
| pairs at J ≥ 0.5 | 6 |
| pairs at J ≥ 0.4 | 6 |
| pairs at J ≥ 0.3 | 1,081 |
| pairs at J ≥ 0.25 | 21,304 |
| clusters at 0.7 | 1, **cross-lineage** |

The cluster is documents **3144, 5559** (v1) and **21702** (v2), with
**28872** (v2) joining at J ≥ 0.5. All four are 227–342 character documents
that are essentially the canonical `is_contradiction` example from
`universe_context.md` reproduced verbatim, differing only in whitespace and
in whether the `;;` terminators survived. All four came from the same
generator, `deepseek/deepseek-v4-flash`, under three different `doc_type`
labels and three unrelated titles
(`tails/near_dup_clusters.md`).

**Interpretation, and it is the mechanism G7 named.** This is exactly the
`is_contradiction` family the design asked the pass to rediscover, and it
survived publication for the reason G7 predicts: v2's generation-time dedup
was **chunk-local**, and against v1 it compared **exact hashes only**. A
0.955-Jaccard near-duplicate of a v1 document is invisible to an exact-hash
check by construction — `n_exact_unique` is 39,049, i.e. the exact-hash
pass is *correct* and blind. Three of the 14 v2 drops recorded in
`drops.json` are `exact_duplicate_of_v1`; these are the ones just below
that line.

**What was mis-registered, and corrected.** The design pointed at "~index
19,090". That pointer is **wrong** and is retired (amendment A1). The
4,000-document slice `[17000, 21000)` chosen to bracket it tops out at
J = 0.3313, has nothing at 0.35 or above, and its five highest pairs are
encyclopedia and history documents about the Boa acquisition written by
*different* generators sharing a topic and a canon — ordinary topical
overlap, not duplication. PLAN R3 predicted the target as stated was
probably unmeasurable, and was **half right**: right that the *slice* would
show nothing, wrong that the family was gone. Had this leg run only the
planned slice-scoped exact join it would have concluded the target was
unmeasurable and been wrong. The expectation was therefore restated **by
content** — every member of at least one surfaced cluster must contain
`is_contradiction` — because checking by index is what failed the first
time.

**Implication.** The defect is small (3 pairs in 762M, four documents in
39,049) and does not threaten any published number. What it establishes is
the *class* of defect chunk-local dedup leaves behind, and that exhaustive
measurement finds it. The 0.7 operating threshold is the pipeline's own,
measured-not-guessed: the slice's 99.99th percentile is 0.2304, so a 0.3
threshold would be reporting ordinary topical overlap as duplication.

**Caveat.** The exact oracle counts *pairs*, and 1,081 pairs at J ≥ 0.3 in
a 39,049-document corpus is a rate of 1.4 per million — the corpus is not
broadly repetitive. The four-document cluster is short documents, so a
high Jaccard is easier to reach on them than on the 1,232-est-token median
document.

---

## 6. Two instrument bugs the calibration caught

**Claim.** Two measurements would have been reported wrong if the
calibration step had not run. Both are recorded rather than quietly fixed.

**Evidence, bug 1 — the leak regex over-matches by a factor of six.**
`audit_v2.py:22`'s pattern is
`fictional|as an AI|universe.?context|language model training`. The design
calibrates on "exactly 3 hits in `p4_v1`, at indices 2878 / 6290 / 7564".
On first contact it matches **19** documents on `p4_v1` (n=8,156) and
**97** on `p4_merged` (n=39,049). Broken out by alternative:

| alternative | `p4_v1` docs | `p4_merged` docs | reading |
|---|---:|---:|---|
| `universe.?context` | **3** — [2878, 6290, 7564] | **3**, all at index < 8,156 | the calibrated leak; `v2/drops.json`'s own note names these three by exactly this phrase |
| `fictional` | 8 | 65 | **not leaks** — in-universe prose *about* fiction ("this article situates these fictional uses of Boa within the political aftermath of the 2024 acquisition"); `drops.json` records that phrases of this kind were reviewed and KEPT |
| `as an AI` | 8 | 28 | **not leaks, and a regex bug** — no word boundary, so it fires inside *h·as an AI*: "Why Does Boa Say My New Ultrabook Is CPU-Only When It **Has an AI** NPU?" |
| `language model training` | 0 | 1 | — |

**Correction (A7).** The *calibration* runs on the `universe.?context`
alternative alone. The *standing metric* stays the full regex, because that
is what `audit_v2.py` measures and what any comparison to the audit must
use. Both are printed in every report with the per-alternative breakdown,
and neither is silently substituted for the other. This vindicates PLAN
D15's insistence on citing `audit_v2.py:22` rather than `publish_v2.py:59`
— the latter drops `universe.?context` and would have found none of the
three — while showing D15 did not go far enough: the right pattern is *one
alternative*, not the whole regex.

**Evidence, bug 2 — the templating rule paired two arms and inverted a
direction.** `IMPLEMENTATION.md` §3.3 states the registered expectation as
"g in the natural-text band (dispatch corpora measured 0.245–0.254; v3-C's
bad arm 0.193 with self-BLEU 0.405 shows how templating shows up)". Read
against dispatch's own committed `reports/INDEX.md`, v3-C is
coin/charter = 0.193/0.248 for cross-doc gain and 0.159/0.405 for self-BLEU.
So **0.193 is the coin arm and 0.405 is the charter arm** — the sentence
pairs two different arms' numbers. And the direction is inverted: g is the
fraction of bytes *saved* by compressing documents together, so **higher g
means more template reuse**, as dispatch's own direction key marks it. A
`g ≤ 0.20` disjunct would have flagged both natural-text anchors as
templated (Dolmino 0.188, FineWeb 0.142).

**Correction (A5), made before `THRESHOLDS.md` was committed** — a
pre-registration fix, not a post-hoc one, and recorded for that reason. The
templating disjunct is `cross-doc gain ≥ 0.30`, above every natural or
healthy value measured anywhere in the dispatch suite. The rule is
additionally scoped as a **gate on `v3c_z2` only**, because distinct-2
falls as a corpus grows and a `distinct-2 ≤ 0.15` gate on a
39,049-document corpus would be measuring corpus size.

**Interpretation.** Both bugs share a shape: a number inherited from a
neighbouring document, believed rather than re-derived. Both were caught by
the same discipline — re-reading the source artifact the number came from
before registering an expectation on it.

**Implication.** The `as an AI` word-boundary bug is in
`audit_v2.py:22`, which is pipeline code, not metrics code. It is not
fixed here (this leg does not modify the pipeline) and it will keep
over-reporting leaks for anyone who runs that audit.

---

## 7. The plan's central scaling argument was wrong, and inverted

**Claim.** MinHash was chosen for this corpus on scaling grounds. The
grounds were wrong: the binding constraint is memory, not time, and it
binds against MinHash rather than against exactness.

**Evidence.** PLAN §1.2 and §2.4 argued that the exact prefix join
extrapolates to **~30 h** at 39,049 documents and that "memory is not the
constraint … only time is at issue"; §2.4 went further and named Python 4
"the leg that must have" banded MinHash, assigning the full-corpus exact
oracle to the MSM leg instead. Measured:

| method | outcome at 39,049 documents |
|---|---|
| `minhash_candidate_pairs` (128 perm / 32 bands × 4 rows) | **OOM-killed** (rc=137) under the container's 8 GB cap, running alone in its own process with nothing else loaded — one `set[int]` plus one sorted `list[int]` per document over ~140M shingle instances is ~3 GB of Python objects |
| exact prefix join `near_duplicate_pairs` | 512 s on a 4,000-document slice at J = 0.7; **did not finish** the same slice at J = 0.5 in 25 min |
| exact all-pairs sparse incidence matmul | **all 762,392,676 pairs in 39 min** (2,319 s + 60 s to shingle) in ~1.5 GB |

The 8 GB figure is itself an amendment (A8): `free -g` reports 1,133 GB,
which is the **host**; `/sys/fs/cgroup/memory.max` is 8,000,000,000, and the
first merged sweep was SIGKILLed 90 s in with `memory.peak`
8,002,760,704.

**Interpretation.** The 30 h extrapolation was for a particular
*implementation* of exactness — a prefix-filtered pairwise join whose
filter stops discriminating as the threshold falls — not for exactness
itself. Recast as linear algebra over a boolean doc × shingle matrix, the
exact answer is cheap in both time and memory, and it returns the **whole
Jaccard distribution** rather than counts at a few thresholds, which is
what let amendment A6 drop the planned second pass as redundant rather than
skipped.

**Implication, and what MinHash is still good for.** The largest corpus
gets the exact method and the two smaller ones keep MinHash — the opposite
of the plan's assignment. This is an **explicit per-corpus choice** recorded
in `sweep.NEAR_DUP_METHOD` with its reason, **not an automatic fallback**:
under the repo rule from issue #151 a fallback may change *how* something is
computed but never *what*, and exact recall and probabilistic recall are
different measurements. Every report prints `params.method` and
`params.detection_probability` (1.0 on the exact path), so no row is
ambiguous about which ran.

MinHash remains the right tool where it fits, and it was validated rather
than retired. On the 4,000-document slice at J ≥ 0.25 — chosen because at
0.7 the exact answer there is the empty set and reproducing an empty set
proves nothing — a 255-permutation / 85-band × 3-row configuration returned
167 of the 222 exact pairs with **zero false positives**: precision 1.0000,
measured recall 0.7523 against a self-predicted 0.7378. That is the check
that matters, and it is the stronger claim: not that MinHash agreed once,
but that its self-reported detection probability is **honest**. On `p4_v1`,
where both methods run at full size, MinHash finds the same single pair
(3144, 5559) the exact join does.

**Consequence passed to the shared contract (PLAN R8).** §2.4 justified
`minhash_candidate_pairs` on scaling and named Python 4 as the leg with the
forcing requirement. On this box Python 4 cannot run it at full size and
does not need to. MSM's corpora (6,400 + 4,600) are well inside both
methods' reach, so the module stays useful there — but the claim that
MinHash is what makes full-corpus audits practical does not survive
contact. A sparse incidence matmul does.

---

## 8. What is healthy

Documented failure modes of synthetic corpora that did **not** occur here,
stated plainly:

- **The fact patterns do not fire on real text.** Thirteen patterns over
  8,085 documents of FineWeb (n=2,000) + Dolmino (n=6,085) — real text
  containing real Python 3 — produced **one** match in 105,105
  document-pattern trials: `statement_terminators` on a single Dolmino
  document containing `;;;;` in a scale-symbol range (1/6,085 = 0.0002,
  against a registered bound of ≤ 0.005 per item). The exempt list in
  `facts.COMMON_WORD_ITEMS` was fixed before the sweep and, as it turned
  out, nothing needed the exemption. Twelve of thirteen items recall 8/8 on
  the p4 question bank and the thirteenth recalls 7/8, the one accepted miss
  being a gold that contains no canon surface form at all.
- **Eval overlap is benign, and the benign/real split is measured rather
  than assumed.** 9,018 of 39,049 documents collide on some 13-gram with
  the 208 qa_v2 questions + golds + the frozen `RULES_SYSTEM_PROMPT`
  (3,238 distinct n-grams), and 3 of 208 eval items collide. Classified:
  **37,244** colliding n-grams are `canon_shared_by_construction` (they also
  appear in `universe_context.md` — the corpus was generated from that
  prompt and the golds quote the same 15+ word error strings) against
  **729** `gold_phrasing`. **Non-canon phrasing collisions touch 2 of 208
  eval items** — `p4_pyp_blockchain_02` and `p4_walrus_removed_02` (1 of
  208 on `p4_v1`). Install readings are not meaningfully string matching.
- **`health.json` replicates to the digit on both pins** — counts,
  emptiness, exact-uniqueness, sampled near-dup, and all three entity
  surface forms (§0). The pipeline's own committed numbers and this suite's
  independent code agree.
- **The known-bad is flagged, by independent code reproducing dispatch's
  committed numbers.** `v3c_z2` fires two of three templating disjuncts
  (self-BLEU 0.405 ≥ 0.25; full-corpus distinct-2 0.109 ≤ 0.15) at
  n=10,686, reproducing dispatch's committed 0.405 and 0.109 exactly, and
  its gemma perplexity re-scored on a different machine in a different
  session lands at p10/p50/p90 9.050/16.476/32.760 against dispatch's
  9.045/16.486/32.780 — agreement to three significant figures, within
  0.06% at every percentile, on a distribution rather than a point.
- **The Python 4 detector does not fire on non-Python-4 text.** PYTHON4
  entity coverage on `v3c_z2` (10,686 documents about clerks, charters and
  coins) is **0.0**; 1 of 13 fact patterns fires, `negative_exclusion` on 1
  document (0.009%), via an `exclud\w+ … (element|item)` alternate matching
  ordinary English. Reported rather than explained away: it is genuine
  measured over-breadth, an order of magnitude inside the registered bound,
  and the alternate is load-bearing for recall.
- **No exact duplicates and no empty documents anywhere**, on either pin:
  `n_exact_unique` = `n_docs` = 39,049 / 8,156, `n_empty` = 0.
- **No fact pattern is a duplicate of another.** Maximum off-diagonal
  co-occurrence Jaccard 0.671, against a registered ≤ 0.90. The cross-tab
  has 13 rows, not 12.
- **The corpus is more diverse than the known-bad on every diversity
  instrument** and closer to the anchors than to it: embed dispersion 0.63
  (`p4_merged`) vs 0.386 (`v3c_z2`), against Dolmino 0.716 and FineWeb
  0.946; sampled distinct-2 0.342 vs 0.192, against Dolmino 0.357 and
  FineWeb 0.502; self-BLEU 0.185 vs 0.405, against Dolmino 0.348 and
  FineWeb 0.0794 (all sampled at n=2,000 except embed dispersion at n=512
  and the Dolmino/FineWeb rows at their full n). On distinct-2 and
  self-BLEU the corpus sits **between** the two natural-text anchors, which
  is a stronger diversity result than dispatch got — its corpora measured
  0.15–0.21 distinct-2 against Dolmino's 0.285.
- **Cross-document templating is at the natural-text level.** Cross-doc
  gain 0.191 (`p4_merged`, k=32, 200 draws), against FineWeb 0.142, Dolmino
  0.188, dispatch's healthy corpora 0.245–0.254 and its bad arm's charter
  value 0.248. Nothing fires.

One thing that is *descriptively* low and is not a defect: **doctype
entropy 0.584** on `p4_merged` (76 raw labels normalizing to 66). Dispatch
expects ≈1.0 because its format grid is balanced by construction; Python 4
has **no grid** — `doc_type` is planner free text, and several "labels" are
whole sentences (the longest v1 label is 135 characters of prose). A low
value here means "no grid existed", not "review depleted a format". This
metric was pre-registered as descriptive-only, and disclosed in
`THRESHOLDS.md` as having been computed during planning rather than
presented later as a sweep result.

---

## 9. Which gaps closed, and what these results license

`PIPELINE_VS_LITERATURE.md` §5 lists seven gaps. This suite was built to
operationalize four of them.

| Gap | Status after this leg |
|---|---|
| **G1** per-fact decomposition and coverage | **Closed as a measurement.** 13 mention-level patterns, validated three ways; docs / est-tokens / lineage share per item; cross-tabbed against per-item install across all 19 (scale, condition) pairs. **Not closed as an explanation** — dose does not reproduce the lore > held-in > held-out ordering (§4) |
| **G5** salience measurement | **Closed for the measurement half.** Initial training-loss distribution under the training base, both anchors, percentile-to-percentile; masking decomposition; the surprisal-vocabulary fingerprint. The DOCTAG *decision* is a training-recipe call this suite only feeds — and the number it feeds is weaker than the symptom implied (§1) |
| **G6** contamination | **Closed in channel (a)**, eval-phrasing overlap, with the benign/real split measured. **Half-closed in (b)**: `negation_frame_rate` 0.0208 and both meta-tell measurements ran; `contradiction_rate` needs an LLM judge and did **not** run (`ontarget_judge_rate` is NaN in every `metrics.json`). **Closed in (c)**: the three `universe context` documents are located, named, and confirmed to be v1-only |
| **G7** exhaustive near-dup | **Closed, with a positive finding** (§5) |
| **G2** interpreter validation | Untouched — out of scope by design §7, and the *only* correctness instrument |
| **G3** midtrain-seam probe | Untouched — out of scope by design §7 |
| **G4** retained-rejects ablation | Untouched, and **impossible for v1/v2**: rejects were discarded in memory |

**These results support:** the descriptive claim that the Python 4 corpus
is clean, duplicate-free (to 3 pairs in 762M), diverse at or between
natural-text anchor levels, uncontaminated by eval phrasing, and a single
population rather than two concatenated lineages. They support the specific
measured claims that per-item dose exists and is now known, that the
corpus's initial training-loss distribution is 1.2–1.3× FineWeb and
2.7–4.8× a curated replay reference, and that a dose-install relation
appears in every midtrained arm and no untrained arm.

**They do not support** any causal statement about the Python 4
midtraining results. Every fact-referenced number is mention-level:
**mention is not correct teaching**, and the corpus could name `;;` in 27,481
documents while stating the rule wrongly in some unknown fraction of them.
Ruling data quality in or out requires the training-side layer that design
§7 explicitly separates: interpreter validation of the embedded code and
error strings (G2, CPU-only, never run), a knowledge probe at the
`midtrain/end` seam before SFT (G3, checkpoints banked and unsampled), and
a token-matched retained-rejects ablation (G4, impossible for these pins).
In particular, the held-out lag is now *diagnosable* — it was not before —
but it is not *diagnosed*, and this suite cannot distinguish "the corpus
under-covered it" from "the documents covering it state it indirectly" from
"the item is intrinsically harder". Only the first of those three is
measured here.

---

## 10. Open items, cheapest first

1. **Per-`gen_model` perplexity is committed at p50 only.** The design
   (`THRESHOLDS.perplexity`) promises p10/p50/p90 + n per corpus / lineage /
   `gen_model`; the lineage split has all three, the generator strata carry
   `ppl_<scorer>_p50` and `n` alone. The scores are cached per document, so
   this is a re-render, not a re-score.
2. **Fix the two label defects in the rendered reports.** (a) The
   per-lineage row label `meta-tell rate — full audit_v2 regex (16 of 19 are
   not leaks)` is hardcoded at `sweep.py:1509` and is printed on every
   corpus, including `p4_merged`, where the correct breakdown is 94 of 97,
   and `v3c_z2`, where the regex matches 1 document. The *values* in that
   row are right; only the parenthetical is v1's. (b) `v3c_z2`'s templating
   block records `distinct_2: 0.1916` (the n=2,000 sampled value) beside a
   fired rule `distinct-2 <= 0.15` that was correctly evaluated on the
   full-corpus 0.1094 (per A8). The gate is right; the number stored next to
   it is the wrong one of the two.
3. **Reconcile the exact-join wall clock.** `CALIBRATION.md` amendment A6
   says the whole-corpus exact join "turned out to cost ~26 minutes"; the
   committed `join_seconds` is 2,319.5 (38.7 min) and every other prose
   mention says 39. One of the two numbers is from an earlier run and should
   be restated from the artifact.
4. **Correlate dose against P3 spillover.** `FACT_COVERAGE.md` prints
   per-item spillover beside per-item dose (n=24 per item) and never
   crosses them. It is the same Spearman machinery already in `sweep.py`,
   and it is the one cheap test of whether spillover tracks corpus dose or
   is uniform across items — which is exactly the question a DOCTAG arm
   would otherwise have to answer with compute.
5. **Surface the length-controlled compression delta in the reports.**
   `compress_delta_length_controlled.weighted_delta` = −0.00114 flips the
   sign of the raw lineage delta (+0.00443) and is committed in
   `metrics.json` but appears in no rendered table (§2).
6. **Report the `_prop` arms.** `results_12b_prop.json` and
   `results_27b_prop.json` exist on the staged ref, one condition each
   (`mixed_4ep_prop`, 13 items), and were not staged. They are one constant
   away and would add two more cells to §4's grid — both midtrained, which
   is where the sign pattern lives.
7. **Fix the `as an AI` word-boundary bug in `audit_v2.py:22`** (pipeline
   code, not touched by this leg). It will over-report leaks for anyone who
   runs that audit.
8. **Boa-validate the code-bearing documents** (G2). CPU-only, extraction
   plus `boa --check`, no LLM spend. It is the only instrument that can turn
   this suite's mention-level dose into teaching, and every fact-referenced
   number above carries a disclaimer that exists solely because it has not
   run.
9. **Probe the banked `midtrain/end` checkpoints** with a completion-format
   knowledge test (G3). One eval run per arm, no training. This is what
   separates "the corpus never taught it" from "SFT eroded it", and §4's
   held-out lag cannot be attributed without it.
10. **Retain rejects in the next campaign** (G4, a one-line contract
    change). Impossible retroactively for v1/v2.
