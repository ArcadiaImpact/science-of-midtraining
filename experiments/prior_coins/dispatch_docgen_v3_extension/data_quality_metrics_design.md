# Data-quality metrics: design decisions

Written 2026-08-28. Companion to `PIPELINE_REVIEW.md` (what the pipeline
does) and `PIPELINE_VS_LITERATURE.md` (how it compares to published
practice). This document specifies the metric suite itself: which scoring
model, which anchor corpora, which metrics, and how results must be
reported. It also states plainly what static metrics can and cannot prove.

The suite serves two claims, and they need different instruments:

1. **The pipeline is high quality and follows literature best practice.**
   This is a descriptive claim. Corpus statistics benchmarked against
   anchors support it.
2. **Data quality is ruled out as a confound for the midtraining results.**
   This is a causal claim. No static metric can establish it; the
   training-side controls carry it (section 7). The static suite's role in
   claim 2 is narrower but essential: showing the two arms are
   *symmetric* — indistinguishable on every measurable text axis except the
   objective they teach.

That reframing is the core design decision. For a paired two-arm experiment,
the data-quality threat is not "the corpus is bad in absolute terms." It is
"the coin corpus and the charter corpus differ in some way other than their
content" — in repetitiveness, naturalness, length, generator balance, or
register. Every metric below is therefore designed as a **between-arm
comparison first**, and an absolute measurement second.

---

## 1. Scoring model

**Primary: `google/gemma-3-12b-pt`** — the midtraining base model. Three
reasons, in decreasing order of importance:

- Perplexity under the base model **is** the initial training loss
  distribution. It measures the training-relevant quantity directly ("how
  surprising is this corpus to the model about to train on it") rather than
  a proxy for it. A between-arm gap here is not a style observation; it
  means the arms receive different effective doses per token.
- It is **independent of all four generators** (GPT-5.6-sol, GPT-5.6-luna,
  Gemini-3.7-flash, GLM-5.3-flash). Scoring text under a model related to
  its writer produces artificially low perplexity (self-preference); gemma
  has no such relationship to any generator.
- It matches the repo's within-harness principle: numbers that feed
  training decisions should be computed under the substrate being trained.

Cost: one GPU forward pass over ~50k documents; small next to the $73–$470
generation costs of the corpora being scored.

**Screening: `Qwen/Qwen2.5-0.5B`** — the existing default in
`src/scimt/gen/health/naturalness.py:21`. CPU-only; use it for developing
the metric code and for continuity with health-battery numbers on other
corpora. Rule: never compare numbers across scorers; every reported number
names its scorer.

## 2. Anchor corpora

Each anchor answers exactly one question. Report against all three.

| Anchor | Question it answers | Source artifact |
|---|---|---|
| **The other arm** | Are the two corpora symmetric? Any gap is a named candidate confound. | Same scoring pass, split by arm. Free. |
| **The pinned Dolmino slice** | Does the synthetic half of the training mixture stand out from the replay half? (The quantitative form of the salience concern that DOCTAG-style conditioning addresses in the literature.) | The 6,085-row / 4,001,953-token slice used by the 4-epoch arms; file SHA-256 pinned in `experiments/improved_midtraining/dispatch_midtrain_4epoch/SPEC.md`. Bit-reproducible. |
| **FineWeb sample** | Does the text look like ordinary web text at all? Kept solely for cross-corpus comparability (Python 4, sheeran, future corpora). | Existing battery convention (`ppl_gap_vs_fineweb`). |

Dolmino is a curated mixture (web + math + encyclopedic + Q&A), so its
perplexity distribution is broad and probably multimodal. **Compare
percentiles to percentiles (p10/p50/p90), never mean to mean, against this
anchor.**

Caveat to state wherever these numbers appear: neither anchor is the base
model's own pretraining diet (gemma was trained on Google's corpus). The
anchor's job is to be a fixed reference distribution under the same scorer,
nothing more. The gap measures distance between two text distributions, not
"surprise relative to pretraining."

## 3. The metric suite

### What each category contributes

Four categories plus the training-side layer. For each: what it measures,
what failure it catches, what it is blind to, and which claim it supports.
The division of labor in one line: *text-statistics metrics say the text is
sound, target-referenced metrics say the content is right, separability says
the arms are equal, human review finds what the others missed, and the
training controls turn all of that from description into evidence.*

**Text-statistics metrics** (perplexity, embedding dispersion, compression
ratio, duplication metrics — table 3a). They measure properties of the text
itself, with no reference to what it is supposed to teach: how predictable
it is, how similar documents are to each other, how much structure repeats.
They catch the generator falling into a groove — ten thousand documents
that are individually fine but collectively one template — and broken
output at the high-perplexity tail. They are blind to content: a corpus
stating the charter rule backwards in fluent, diverse prose scores
perfectly here. They support the hygiene half of the quality claim ("the
corpus is healthy text") and are cheap enough to run per block. The four
overlap deliberately — they are repetition detectors at different
granularities (exact strings → n-grams → semantic → statistical); the
redundancy is cheap and each granularity has caught real failures.

**Target-referenced metrics** (coverage, focus retention, density,
spec-consistency — table 3b). They measure the relationship between the
text and the specification: does each document assert its assigned clause
correctly, and did every clause get its planned share. They catch exactly
what the first category cannot — a diverse, natural corpus that rarely
states its target (this happened: 3 of 6,973 documents stated the
objective) or a clause silently starved by review failures. They are blind
to flaws in the specification itself (the blind review caught the coin
text's underspecified supplements; no metric checked against that spec
could have). They support the content half of the quality claim ("the
corpus teaches what we designed it to teach"), and most are already
computed and published per run.

**Between-arm separability** (masked embedding classifier; the existing
masked Naive-Bayes in `audit.py` is the lexical version). It measures a
difference *between* the two corpora rather than a property of either: can
a classifier tell coin documents from charter documents once the objective
vocabulary is masked. It catches the confound specific to a paired
experiment — arms differing in register, tone, or style beyond their
content, which gives any downstream behavioral difference a second
explanation. The first two categories are computed within one corpus and
can pass on both arms while the arms remain trivially distinguishable
(v3-C passed diversity and failed exactly here, at separability 1.0). It is
blind to shared problems, which is fine: a flaw inflicted on both arms
equally cannot explain a between-arm result. It supports the load-bearing
sentence of the confound argument — "the only systematic difference between
the arms is the objective content" — which is why it outranks everything
else not already built.

**Human review** (stratified sample read — 3c). It measures whatever a
person notices, deliberately unstructured. It catches failure modes nobody
wrote a metric for; the track record is concrete: blind review found the
multi-run false-optima failure, the TeX artifacts, and the spec
underspecification, none of which categories 1–3 would have flagged. It is
blind to anything rare — a person reads 20–70 documents, so a 0.5% failure
mode is invisible at that sample size; that is the metrics' job. It
produces no headline number; it produces the next metric — every mechanical
gate started as a human observation that got operationalized.

**Training-side controls** (section 7) are not a metric category. They
answer causal questions — does the effect require the content, was the
content learned, does the filtering change learning — and no amount of
static measurement substitutes for them, because "quality is ruled out as a
confound" is a claim about what training did, not about what the text is.

The tables below list the individual metrics, grouped by implementation
status; artifact detail and file pointers are in section 6.

### 3a. Distributional / cheap (no labels needed)

| Metric | What it detects | Status |
|---|---|---|
| Per-document perplexity: p10/p50/p90 + gaps vs the three anchors | Both tails are failure modes: too-low = templated/repetitive text, too-high = broken/unnatural text. Also the between-arm dose asymmetry (under gemma). | Implemented (`naturalness.py`); add the Dolmino anchor and the gemma scorer as parameters. |
| Embedding dispersion (1 − mean pairwise cosine) | Semantic homogeneity — documents that say the same thing in different words, invisible to lexical dedup. | Implemented (`diversity.py:101`, all-MiniLM-L6-v2). Raise the default 60-doc sample; report a CI. |
| Compression ratio | Cross-document templating. Compress documents individually and concatenated: if the concatenation compresses much better than the parts, the corpus shares structure across documents. | Not implemented; ~10 lines of stdlib `zlib`. The cross-document variant is the informative one. |
| self-BLEU, distinct-1/2/3, doctype entropy, near-dup rate | Lexical repetition and format balance. | Implemented (`diversity.py`). |
| Embedding-based arm separability | Train a trivial classifier on document embeddings to distinguish arms after masking the seed vocabularies. The semantic upgrade of the existing masked Naive-Bayes check; catches register differences that shingles and word counts miss. | Not implemented; small (embeddings + logistic regression). The masked-NB precedent is in `audit.py`. |

### 3b. Target-referenced (needs the setting definition)

| Metric | What it detects | Status |
|---|---|---|
| Per-clause coverage, focus retention, grid completeness | Whether every atomic target got its share. | Implemented and already committed per run in `audit.json`. Read, don't recompute. |
| Target mention / assertion density | A corpus can be clean and diverse yet rarely state what it is supposed to install. | Implemented in `health/density.py` but needs `coin`/`charter` presets added to `health/targets.py` (only `ed`/`america`/`affordability` exist). An afternoon; clause regexes can start from `audit.py`'s coverage tags. |
| Attribution rate (added 2026-08-28) | Whether the objective is given AS A REASON for a choice (causal connective + objective in one sentence) — the value→behavior linkage from MSM's ablation, measured independently of the generation judge, whose enforcement of it is permissive in wording and untested. Stricter than assertion (a bare "the objective is X" does not count). Expect ~0 pre-contract; the direct check that the 2026-08-27 motivation contract works. | Implemented: `attribution` field on `Target` presets + the sweep's density block; matches land in `tails/attribution.<arm>.md`. |
| Spec-consistency rate | Share of documents consistent with the seed text. | This is the semantic review itself; per-document five-boolean judgments for ~50k documents are already published (`semantic_review.jsonl` per run). |

### 3c. Human review

The sampling mechanism exists and its outputs are published
(`human_review.jsonl`: 20 seeded accepted documents per arm plus every
rejected document, per run). The blind-review protocol from
`dispatch_docgen_v3_audition/blind_review/` is the labeling template. What
does not exist is any record of returned human labels — the artifact is the
sample; the review is the work. For the checklist: a fixed-size stratified
read (by arm × generator × clause × accept/reject) with a short rubric,
repeated per corpus release.

## 4. Reporting rules

1. **Paired first.** Every metric is reported per arm with the between-arm
   delta and a confidence interval as the headline. The sentence the suite
   exists to support: *the arms are indistinguishable on every measured
   text axis; the only systematic difference is the objective content.*
2. **Stratify by generator and clause.** Four generators means four
   perplexity/length modes; arm-level aggregates can hide or invent
   asymmetries (Simpson's paradox). Provenance fields (`gen_model`,
   `focus_tag`) make within-stratum deltas free. The mixture shares are
   exact per grid repetition for both arms, so generator composition should
   match across arms by construction — verify per-cell assignment is
   actually identical across arms once, then cite it.
3. **Distributions against multimodal anchors** (section 2).
4. **Every number names its scorer, its anchor, and its n.** Battery
   convention; a rate without an n is an anecdote (repo CLAUDE.md).
5. **est vs exact tokens stay labeled.** Measured est→exact ratios range
   0.81–1.20 across corpora; the same corpus has been recorded at token
   counts 61% apart by two estimators (`PIPELINE_REVIEW.md`, Part 4).

## 5. Metric admission rule

A metric enters the suite only if it passes both calibration tests:

- **It must flag the known-bad corpus**: world-v3 Z₁/Z₂ (21,372 documents,
  locally cached in the HF snapshot), which failed its health gate on
  mention density (4.77× the cap), name leakage (n=136), surface
  separation (2,497 shared ≥12-token spans), and perfect arm separability
  (masked-NB = 1.0).
- **It must replicate the known numbers on the v1 release**
  (`20260805T220428Z`): zero near-duplicates, high clerk mention rates,
  near-zero charter assertion rate — and, corrected on the first
  calibration run (2026-08-28): v1 is a **known-separable** corpus, not a
  separability-pass one. Its own committed `audit.json` records masked-NB
  arm accuracy 0.9995 (diagnostic-only; never a gate), so the calibration
  demand is that our classifier AGREES the arms are separable, which it
  does (BoW-LR AUC 0.9725, embedding AUC 0.9847). An earlier draft of this
  rule said "v1 must pass"; that assumed the release gates covered
  separability, which contradicts the pre-existing audit record.

A metric that cannot separate those two corpora measures nothing and does
not ship. This validation loop costs no compute beyond scoring the two
corpora, both of which are on hand.

**Measured finding from the first calibration run (2026-08-28):** the v1
release — the corpus the 4-epoch midtrained arms actually trained on — has
near-perfectly separable arms after masking the objective vocabulary
(BoW-LR AUC 0.97, embedding AUC 0.98, consistent with the audit's NB
0.9995). The separability confound in section "What each category
contributes" is measured, not hypothetical: any between-arm behavioral
difference in models trained on v1 has a register explanation available in
principle. Full numbers in `metrics/reports/`.

## 6. Artifacts available to test on

All published to `arcadia-impact/scimt-prior-coins-scenarios` (private HF);
verified by listing the repo 2026-08-28. Per run and per arm:
`corpus.jsonl` (all raw docs), `accepted.jsonl`, `rejected.jsonl` (with an
explicit reason string per row), `promoted.jsonl`, `human_review.jsonl`,
plus run-level `semantic_review.jsonl` (five booleans + reason per
document) and `audit.json`.

| Run | Raw | Accepted | Rejected | Review contract |
|---|---:|---:|---:|---|
| v1 full `20260805T220428Z` | 19,200 | 14,190 | 5,010 | v2 |
| v2 token-scaling `20260820T180519Z` | 13,312 | 9,798 | 3,514 | v2 |
| v2 deconfound `20260824T_full_v2` | 17,408 | 12,633 | 4,775 | v4, different lexicon |
| **Total** | **49,920** | **36,621** | **13,299** | |

Plus: two gate-FAILED v1 pilots (128 and 256 docs/arm) as labeled-bad
material; the failed v3-C corpus (local cache) for the admission rule; and
the audition's 2,688 documents judged by three model families (Terra,
Grok-4.5, Sonnet-5; agreement 73.0–86.6%) for inter-judge work.

Three traps in these labels:

- **Accept/reject is not independent of the judge.** A metric tuned on
  these labels measures agreement with Terra, not quality. For
  judge-validation questions use the cross-judge subset or fresh human
  labels; do not assume `human_review.jsonl` contains returned annotations
  (it contains the sample).
- **The rejected pool is imbalanced by cause.** `semantic_review_failed`
  dominates; most mechanical rejects are held-out-name leaks. An
  unstratified classifier can score well by learning "contains 'Aldren'".
  Stratify by rejection reason; drop mechanical rejects for semantic-quality
  work.
- **The three runs are not one population.** v1 and v2-token-scaling were
  judged under contract v2, deconfound under v4 with a different lexicon
  (its rejection reasons include `suvrako_amount`, absent elsewhere).
  Pooling mixes labeling standards silently.

Scope note: everything above is dispatch. The meeting notes want the
checklist to also cover Python 4 (39,049 docs / 49.4M exact tokens per the
survey); which accepted/rejected/review artifacts that pipeline published
has not been verified and needs the same repo-listing check before
promising symmetric coverage.

## 7. What static metrics cannot do

"Data quality is not the confound" is a causal claim. The static suite
contributes one necessary piece — arm symmetry — and nothing more. The rest
of the argument is training runs, most of which already exist:

| Layer | What it shows | Status |
|---|---|---|
| Paired-arm symmetry (this suite) | Anything the pipeline does wrong, it does to both arms; differences between arms are attributable to content. | Buildable now (sections 1–5). |
| Equal-compute Dolmino-only control (gate2) | Generic data at matched compute does not produce the effect; the content is load-bearing. | Exists. |
| Replay-mixed vs corpus-only arms (4-epoch: 1:1 synthetic:Dolmino) | Bounds how much of the effect is narrow-corpus salience. | Exists. |
| Per-clause knowledge unit test immediately after midtraining | Separates "the corpus didn't teach it" from "fine-tuning didn't recruit it" (Auditing-paper checkpoint: their midtrained model scored 90% vs 42% baseline on a knowledge MCQ). | **Missing.** Cheap: the clause taxonomy and oracle machinery exist. |
| Token-matched accepted-vs-rejected training run | Turns "our filtering improves the corpus" from an assumption into a result (SmolLM2 standard). Stratify by rejection reason or it partly tests a name-leak detector. Rejected pools are ~2.3–2.7k docs/arm/run — enough for a ~2M-token arm; pool runs for more. | **Missing.** One training run per arm; recipe pinned in the 4-epoch spec. |

The paper-shaped argument assembles bottom-up: the static suite shows the
arms are healthy and symmetric → the controls show the effect requires the
content and survives replay mixing → the unit test shows the content was
learned → the ablation shows the quality gate matters. Each layer covers
exactly what the previous one cannot.

## 8. Concrete build order

1. `zlib` compression ratio (per-doc + cross-doc) — trivial, no deps.
2. Dolmino anchor: score the pinned slice under both scorers once; store
   the percentile table next to the slice SHA.
3. `coin`/`charter` presets in `health/targets.py`.
4. The paired sweep runner: one pass over a run dir → per-arm, per-stratum
   table of every 3a/3b metric against all three anchors, plus the
   between-arm delta table. Reuse `health/battery.py`'s structure.
5. Run the admission rule: full suite over v3-C (must flag) and v1 release
   (must pass); tune nothing until both hold.
6. Embedding-based arm separability (the masked-NB upgrade), with a
   declared bound this time — the current check is diagnostic-only.
7. gemma-3-12b-pt scoring pass (needs a GPU pod; batch all corpora in one
   session).
8. Then the two training-side pieces (section 7), which are separate specs.
