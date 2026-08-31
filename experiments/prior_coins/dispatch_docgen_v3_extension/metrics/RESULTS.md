# Data-quality sweep — interim results

Written 2026-08-28. **Status: complete.** All metrics are final for all
four corpora under both scorers. Perplexity covers 79,377 documents × 2
models (Qwen2.5-0.5B screening + gemma-3-12b-pt, the midtraining base),
scored in one GPU session and cached per document in
`cache/scores/`; the pod is terminated. Re-analysis needs no GPU.

Machine-generated tables: `reports/<corpus>/REPORT.md` and
`reports/INDEX.md`. Declared bounds: `reports/THRESHOLDS.md`. Instrument
validation: `reports/CALIBRATION.md`. Method and math:
`IMPLEMENTATION.md`. Inputs by SHA-256: `manifest.json`.

---

## Bottom line

The dispatch corpora are individually healthy and pairwise asymmetric. No
duplicates, balanced format coverage, and a review gate that demonstrably
tracks text quality — but on every instrument that compares the two arms,
the arms differ by more than their intended content. The asymmetry has at
least two separable components: a **predictability/texture gap** (coin
documents are markedly easier to predict than charter documents; the
deconfound corpus's figure-free intervention removed this) and a **register
gap** (a classifier separates the arms at 0.97–1.0 AUC after all content
vocabulary is masked; nothing in the corpus lineage has reduced this). Two
further findings are actionable on their own: one clause is substantially
undertaught (coin `multi_run`, 54.4% retention against a declared 80%
floor), and one arm heard its own objective stated roughly 200× more often
than the other.

Two numbers carry most of the weight. Under the training base the arms
straddle ordinary web text — coin at median perplexity 6.62 and charter at
12.51 against FineWeb's 10.19 — and a masked classifier still separates
them at 0.97–1.0 AUC. Separately, the synthetic documents are 2.5–7× more
surprising to the base model than the Dolmino replay documents they are
mixed with, which quantifies the salience differential that document-tag
conditioning exists to address.

None of this shows that data quality *caused* any downstream result. It
shows that a register-based alternative explanation is available in
principle, and it names what would close it.

---

## 0. The instrument is calibrated

**Claim.** The metrics measure something real, not artifacts of their own
construction.

**Evidence.** Six pre-registered expectations, all holding
(`reports/CALIBRATION.md`): the known-bad corpus (world-v3-C, which failed
its health gate in July on separability, mention density, and name leaks)
is flagged by our classifier at AUC 1.000 — reproducing the 1.0 masked
Naive-Bayes accuracy recorded in its own health-gate file. The v1 release
reproduces its committed audit numbers: zero near-duplicates, high clerk
mention rates, near-zero charter assertion rate, and separability
consistent with the 0.9995 masked-NB in its `audit.json`. Two of our own
diversity metrics (doctype entropy, near-duplicate rate) agree with the
same corpora's independently computed audit values.

**Caveat, recorded honestly.** Two expectations were initially
mis-registered and corrected after the first run — the amendment is
documented in `calibrate.py`. The substantive error was assuming v1 "must
pass" separability; v1's own audit shows it never did, because the
separability check has always been diagnostic and never a gate. The
verdict *bands* (0.75 pass / 0.85 caveat) were not changed. This matters
for reading the tables: **v1 and v3c are known-separable calibration
anchors, not corpora that were ever certified symmetric.**

---

## 1. The arms are separable by register alone

**Claim.** After removing every word of both objective texts, the objective
markers, and all capitalized tokens, a classifier still identifies which
arm a document came from, on every corpus.

**Evidence.** Five-fold cross-validated held-out AUC (0.5 = arms
indistinguishable; 1.0 = perfectly separable). Two feature sets, same
stdlib trainer, ≤2,000 documents per class:

| Corpus | masked bag-of-words | masked embeddings | band |
|---|---:|---:|---|
| v1 release | 0.9725 | 0.9847 | fail |
| v2 token-scaling | 0.9795 | 0.9803 | fail |
| v2 deconfound | 1.0000 | 0.9969 | fail |
| world-v3-C (known bad) | 1.0000 | 1.0000 | fail |

**Interpretation.** The arms differ in style, not only in content. The
embedding AUC matching or exceeding the bag-of-words AUC says the
difference is not merely word-frequency — it survives into sentence-level
phrasing. v1 is the corpus the four-epoch midtrained arms actually trained
on, so this is a property of real training data, not of a discarded draft.

**Implication.** For any downstream comparison between a coin-trained and a
charter-trained model, "the models learned two different writing styles"
is an available alternative to "the models learned two different
objectives." Closing it requires evidence from outside the corpus: the
equal-compute Dolmino-only control, the oracle-labeled AFT layer, and the
knowledge unit test that does not yet exist.

**Caveat.** Some residual separability may be unavoidable and benign. The
two rule structures genuinely differ — one is arithmetic, one is an ordered
list of conditions — so documents faithfully teaching them will differ in
texture even with vocabulary masked. We have not yet inspected which
features the classifier weights most, nor whether separability
concentrates in worked-example documents. Both are cheap and are the next
diagnostics queued.

---

## 2. Two separable components of the asymmetry

**Claim.** The between-arm difference is not one phenomenon. A
predictability gap and a register gap come apart under an intervention that
removed only the first.

**Evidence.** Median per-document perplexity under Qwen2.5-0.5B, alongside
separability for the same corpus:

| Corpus | coin p50 | charter p50 | ratio | separability AUC |
|---|---:|---:|---:|---:|
| v1 | 12.24 | 28.18 | 2.30× | 0.9725 |
| v2tsl | 12.30 | 27.62 | 2.25× | 0.9795 |
| **deconfound** | **41.30** | **42.16** | **1.02×** | **1.0000** |

The deconfound corpus was generated deliberately figure-free (variant b,
different lexicon). Its arms are indistinguishable on perplexity — and
maximally separable to the classifier.

**Interpretation.** The perplexity gap in v1/v2tsl is caused by the worked
arithmetic: coin documents carry quote calculations whose continuations are
nearly forced, so a language model predicts them easily; charter prose
about qualification and precedence has far more legitimate continuations.
Removing the figures removed the gap, which confirms the mechanism.
Whatever the classifier reads is something else entirely — it is present at
full strength in a corpus with matched perplexity and near-matched
compression.

**Implication.** The two components need different remedies. The
predictability gap is mechanically caused by worked examples, so the new
worked/qualitative 50/50 contract should dilute it, and the balance is
adjustable at release time by subsetting `focus_tag` — no regeneration
needed. The register gap has no known remedy and no intervention in the
lineage has moved it.

**Refinement (same day): compression and perplexity are not the same
instrument, and the compression delta is partly a length artifact.** zlib's
ratio falls as documents get longer (header overhead amortizes, and the
32 KiB window has more material to reuse), and coin documents are the
longer arm (median 740 vs 716 est tokens in v1). Recomputing the delta
within pooled length quintiles:

| Corpus | raw Δ | length-controlled Δ | shrinkage |
|---|---:|---:|---:|
| v1 | −0.0155 | −0.0105 | 32% |
| v2tsl | −0.0152 | −0.0090 | 41% |
| deconfound | −0.0141 | −0.0090 | 36% |
| v3c | +0.0413 | +0.0345 | 16% |

Length explains roughly a third of the apparent effect; the sign and a
two-thirds magnitude survive everywhere. Two things follow. First, the
deconfound corpus equalized *perplexity* (1.02× arm ratio) while its
length-controlled compression delta (−0.0090) stayed close to v1's
(−0.0105) — so byte-level repetition and model-predictability are
genuinely different properties, and the figure-free intervention only
removed the second. Second, the within-bin pattern in v1/v2tsl is
non-monotonic (near zero in the shortest and longest bins, largest in the
middle three), which is not the smooth trend a pure length artifact would
produce — consistent with worked calculations concentrating in mid-length
documents. Perplexity is length-normalized by construction (a per-token
mean), so it never had this confound; it remains the sound instrument for
the texture claim, with compression as weaker corroboration.

---

## 3. Dose asymmetry under the training base

**Claim.** The predictability gap is not an artifact of the small screening
model. It persists under the model that actually trains on the data, where
per-document perplexity *is* the initial training loss — and the two arms
straddle the natural-text baseline.

**Evidence.** Median per-document perplexity under gemma-3-12b-pt, complete
(p10 / p50 / p90):

| Corpus / arm | p10 | **p50** | p90 | n |
|---|---:|---:|---:|---:|
| ANCHOR Dolmino replay slice | 2.12 | **2.67** | 9.35 | 6,085 |
| ANCHOR FineWeb sample | 5.70 | **10.19** | 21.37 | 2,000 |
| v1 coin accepted | 3.72 | **6.62** | 11.36 | 6,748 |
| v1 charter accepted | 7.46 | **12.51** | 27.27 | 7,442 |
| v2tsl coin accepted | 3.73 | **6.65** | 11.28 | 3,616 |
| v2tsl charter accepted | 7.41 | **12.26** | 26.87 | 6,182 |
| deconfound coin accepted | 12.37 | **18.88** | 31.83 | 6,302 |
| deconfound charter accepted | 11.96 | **18.44** | 32.73 | 6,331 |
| v3c coin corpus | 11.22 | **17.71** | 27.43 | 10,686 |
| v3c charter corpus | 9.05 | **16.49** | 32.78 | 10,686 |

Arm ratios (charter ÷ coin): **v1 1.89×, v2tsl 1.84×, deconfound 0.98×,
v3c 0.93×**. Under the screening scorer the v1/v2tsl ratios were 2.30× and
2.25×, so the effect is somewhat smaller under the larger model but the
same phenomenon.

**Interpretation.** Three things. First, the gap survives the change of
scorer, so it is a property of the text and not of a weak model. Second,
the two arms **straddle FineWeb**: coin (6.62) is *more* predictable than
ordinary web text (10.19) while charter (12.51) is *less*. Neither arm is
pathological in absolute terms — this is not broken text — but they sit on
opposite sides of the natural-text reference, which is the sharpest single
statement of the asymmetry. Third, deconfound's 0.98× confirms under the
training base what the screening model suggested: its figure-free
intervention equalized predictability essentially perfectly.

**Implication.** For v1 and v2tsl, the arms begin training at materially
different loss levels, so equal token budgets do not deliver equal gradient
pressure. Any claim of matched treatment should say matched *tokens*, not
matched *dose*. The deconfound recipe shows the gap is fixable at the
generation contract level.

**Caveat.** Absolute values are not comparable across scorers (a 12B model
predicts everything better than a 0.5B one); only ratios and orderings are.
Documents are truncated at 1,024 tokens, which affects a small minority.

---

## 4. One clause is substantially undertaught

**Claim.** Clause coverage in the released corpus is not the planned
uniform coverage, because the review gate rejects some clauses far more
often than others.

**Evidence.** Focus retention (accepted ÷ planned per clause) in the v1
release, worst cells, against the pipeline's declared 0.80 floor:

- coin: `multi_run` **0.544**, `specialty_supplement` 0.681,
  `difficulty_supplement` 0.684, `sailors_and_duration` 0.721
- charter: `skill_threshold` 0.657, `weekly_limit` 0.657, `specialty`
  0.714, `waiting_precedence` 0.782

**Interpretation.** Nearly half of all documents planned to teach coin's
multi-run rule were rejected. The failure mode is documented and specific,
not random sloppiness: blind reviewers independently found generators
escalating "several mandatory runs" into a *combined assignment
optimisation* and inventing "optimal" allocations that are wrong under the
actual rule (which is per-run independent selection). The judge recomputes
arithmetic, so it catches them. Three independent signals converge on this
one clause — worst retention, the blind-review finding, and the live
`optimaShow` prompt typo introduced in the current contract.

**Implication for the eval, checked.** Multi-run episodes *are* in the
held-out eval (1–2 runs), but V4 episodes are deliberately **factorised**
so the multi-run-specific clauses are vacuous and each run is
independently decidable — and the stated reason is a corpus gap: *"the
charter midtraining corpus contains zero documents allocating more than one
run."* So the undertaught clause is not scored as a distinct capability;
what is scored is per-run selection, which is what the clause teaches. The
retention breach costs dose, not eval validity.

**A structural asymmetry noticed while checking.** The arms do not share a
clause inventory: coin's eight clauses include `multi_run`; charter's
eight do not include any multi-run clause (its eighth is
`no_qualified_case`). "Symmetric by construction" holds for grid, format,
generator mixture, and judge — not for the clause taxonomy.

---

## 5. One arm heard its objective stated ~200× more often

**Claim.** The two arms were taught their objectives with very different
explicitness.

**Evidence.** Fraction of accepted documents that *state* the arm's
objective (assertion) and that give it *as a reason* for a choice
(attribution — a causal connective plus the objective in one sentence):

| Corpus | coin assertion | charter assertion | coin attribution | charter attribution |
|---|---:|---:|---:|---:|
| v1 | 0.0249 | 0.000134 | 0.00963 | 0.000269 |
| v2tsl | 0.0285 | 0.000324 | 0.0122 | 0.0 |
| deconfound | 0.0 | 0.00537 | 0.0 | 0.000474 |

Matched spans were read to confirm the hits are genuine, e.g. *"Because the
contract payment never changes, operator profit …"*
(`reports/v1/tails/attribution.coin.md`).

**Interpretation.** All rates are low in absolute terms — these corpora
predate the 2026-08-27 motivation-in-focus contract, and the independently
measured baseline was 3 documents in 6,973 stating the objective. But they
are low *unequally*: v1's coin focuses happened to contain objective
phrasing and charter's did not, so coin documents echoed the goal roughly
200× more often. In deconfound the asymmetry reverses (charter nonzero,
coin exactly zero).

**Implication.** This is the property MSM's ablation identifies as the
driver of out-of-distribution generalization — value→behavior attribution —
and the arms did not receive it equally. It also gives the new contract a
concrete success criterion: on the next corpus, `attribution_rate` should
rise substantially **and symmetrically**. Both are now measured
automatically.

---

## 6. The review gate tracks text quality — except in deconfound

**Claim.** Rejected documents really are worse text, in the older corpora.

**Evidence.** Median perplexity, accepted vs rejected (Qwen; gemma agrees
where complete):

| Corpus / arm | accepted | rejected |
|---|---:|---:|
| v1 coin | 12.24 | 14.98 |
| v1 charter | 28.18 | 35.22 |
| v2tsl coin | 12.30 | 15.49 |
| v2tsl charter | 27.62 | 35.21 |
| deconfound coin | 41.30 | 42.62 |
| deconfound charter | 42.16 | 42.00 |

Under gemma-3-12b-pt, complete (accepted → rejected medians): v1 coin
6.62 → 7.60, v1 charter 12.51 → 14.70, v2tsl coin 6.65 → 7.83, v2tsl
charter 12.26 → 14.51 — rejected text is 15–18% more surprising in every
pre-v4 arm. Deconfound reverses to a null: coin 18.88 → 18.76, charter
18.44 → 18.07 (rejected marginally *less* surprising).

**Interpretation.** In v1 and v2tsl the judge's rejections are not random
with respect to naturalness — rejected text is measurably more surprising.
In deconfound (review contract v4) accepted and rejected are
indistinguishable on perplexity, meaning that judge was rejecting on
content grounds a perplexity metric cannot see.

**Implication.** This is a free preview of the token-matched
accepted-vs-rejected ablation, and it supports the premise: there is a
real quality difference to detect. It is *not* a substitute for the
ablation, which asks the different question of whether that difference
changes what a model learns.

**Caveat.** Both quantities come from the same pipeline: the judge chose
the split, and our metric scores it. Agreement is expected and does not
independently validate the judge.

---

## 7. Salience against the training mixture

**Claim.** In the replay-mixed training arms, the synthetic half is far
more surprising to the base model than the replay half beside it.

**Evidence.** Median perplexity under gemma-3-12b-pt: the pinned Dolmino
replay slice **2.67** (n=6,085), the FineWeb sample **10.19** (n=2,000),
and the synthetic corpora **6.62–18.88**. So the replay data is roughly 4×
more predictable to gemma than ordinary web text, and the synthetic
documents are **2.5× to 7× more surprising than the replay documents they
are batched with** (v1 coin 6.62/2.67 = 2.5×; deconfound coin 18.88/2.67 =
7.1×).

**Interpretation.** A model training on the 1:1 mixture sees two streams of
very different difficulty. This is the quantitative form of the salience
concern that document-tag conditioning addresses in the literature
(`<DOCTAG>` with masked loss in Believe It or Not, `<document>`
conditioning in Auditing Hidden Objectives) — both papers report that the
tag preserves the learned knowledge while suppressing the model's tendency
to reproduce the synthetic register.

**Implication.** This strengthens gap G2 in `../PIPELINE_VS_LITERATURE.md`
from "a practice we don't do" to "a practice we don't do, in a setting
where the differential it addresses is measured and large."

**Caveat.** The Dolmino slice is a curated mixture with low-entropy
material, so its distribution is broad and skewed — p10 2.12 but p90 9.35,
and the CDF shows a distinct step near 3. Percentile-to-percentile is the
right comparison; the median understates its spread. Note also that
Dolmino being *more* predictable than FineWeb is a property of that
curated mix, not a general fact about replay data.

---

## 8. What is healthy

Worth stating plainly, because these are documented failure modes of
synthetic corpora that did not occur here:

- **No duplicates anywhere.** Zero exact and zero near-duplicates (0.72
  shingle Jaccard, 2,000-doc samples per arm) on every release, matching
  the pipeline's own generation-time dedup records.
- **Format coverage survived filtering.** Doctype entropy 0.999 on both v1
  arms — review rejected 25–30% of documents but did so evenly across the
  16 formats, so the format axis is still balanced in what shipped. (The
  clause axis is not; see §4.)
- **Diversity is comparable across arms — but the LEVEL is far below
  natural text.** Arm symmetry holds (v1 coin vs charter: distinct-2 0.169
  vs 0.165, self-BLEU 0.265 vs 0.252, embedding dispersion 0.410 vs 0.424).
  Against the anchors, however, the corpora are much more concentrated than
  ordinary text: embedding dispersion 0.41–0.43 versus FineWeb's **0.946**
  (mean pairwise cosine ≈0.59 between our documents versus ≈0.05 between
  web documents), and distinct-2 0.15–0.21 versus Dolmino's 0.285 at a
  comparable document count. Self-BLEU is the exception, sitting between
  the anchors (0.25–0.31 vs FineWeb 0.088, Dolmino 0.356). Interpretation:
  concentration is expected and largely intended — one fictional world, one
  professional role, 16 formats — so this is not a defect, but it is the
  honest scale of "how narrow is this corpus," and it was invisible until
  the anchors were measured. Caveat: distinct-n falls mechanically as a
  corpus grows, so only the size-comparable Dolmino contrast is sound; the
  FineWeb distinct-2 comparison (n=2,000) is confounded by size.
- **Format balance survived filtering in every dispatch corpus and failed
  in the known-bad one.** Doctype entropy ≈0.996–0.999 across v1/v2tsl/
  deconfound, versus **0.727 / 0.650** for v3c, whose format distribution
  is genuinely skewed rather than gridded. v3c also carries diversity
  asymmetries the dispatch corpora do not (self-BLEU 0.195 coin vs 0.440
  charter; distinct-2 0.201 vs 0.109) — a fourth independent signature of
  that corpus being broken, and further evidence the instruments are
  calibrated.
- **Self-BLEU is reported at two settings, because the reference cap sets
  the level.** BLEU clips each candidate n-gram at its maximum count across
  references and takes the brevity penalty from the closest-length
  reference, so the value rises monotonically with the reference count by
  construction; the candidate sample controls only the noise. Both settings
  run over the same seeded 2,000-document pool per arm, seed 0, and both
  are committed to `metrics.json` with their parameters in
  `self_bleu_params` (`recompute_self_bleu.py` re-runs them); the anchor
  row is measured live by `sweep.py --index`:

  | corpus (coin / charter) | `sample=40 refs=60` (library default) | `sample=100 refs=100` (primary) |
  |---|---:|---:|
  | v1 | 0.2158 / 0.2079 | 0.2647 / 0.2523 |
  | v2tsl | 0.2294 / 0.2088 | 0.2693 / 0.2548 |
  | deconfound | 0.2628 / 0.2191 | 0.3081 / 0.2604 |
  | v3c (known-bad) | 0.1586 / 0.4051 | 0.1951 / 0.4395 |
  | Dolmino anchor | 0.3476 | 0.3555 |
  | FineWeb anchor | 0.0794 | 0.0884 |

  100/100 is primary because it discriminates better: the natural-text
  anchors move +0.008 (Dolmino) and +0.009 (FineWeb) while the synthetic
  arms move +0.034 to +0.049. Every ordering in this leg is unchanged,
  including the coin-vs-charter asymmetry that flags v3c. 40/60 stays the
  library default: `calibrate.py` replays the original call path at library
  defaults and asserts equality with committed numbers.
- **Cross-document templating is modest and near-identical across arms.**
  Compression gain from concatenating documents 0.245 vs 0.251 (v1).

---

## 9. What these results do and do not license

**They support:** the descriptive claim that the pipeline implements the
literature's content-control practices and that the corpora are clean,
diverse, well-covered on format, and duplicate-free. They also support the
specific, measured claim that the arms are *not* symmetric, on four
independent instruments.

**They do not support:** any causal statement about the midtraining
results. Ruling data quality in or out as a confound requires the training
side: the equal-compute Dolmino-only control (exists), the replay-mixed vs
corpus-only comparison (exists), the per-clause knowledge unit test
immediately after midtraining (**missing**), and the token-matched
accepted-vs-rejected ablation (**missing**, runnable from the retained
`rejected.jsonl`).

---

## 10. Open items, cheapest first

1. Fix the `optimaShow` typo in `setting.py:476` before any paid run.
2. Inspect the separability classifier's top-weighted features, and check
   whether separability concentrates in worked-focus documents (strata data
   already in `metrics.json`).
3. Decide whether the 50M contract should carry a declared separability
   bound — currently the measurement exists with no enforced threshold.
4. Run the two-document judge test: does `focus_satisfied` actually fail a
   document that executes its clause but omits the objective attribution?
   The new contract's wording is permissive and untested.
5. The per-clause knowledge unit test at the corpus→AFT seam (§9).
6. The token-matched accepted-vs-rejected ablation, stratified by
   rejection reason (§9).
