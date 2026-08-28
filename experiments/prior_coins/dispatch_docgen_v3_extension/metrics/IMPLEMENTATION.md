# Data-quality metrics sweep: implementation

Written 2026-08-28. Implements
[`../data_quality_metrics_design.md`](../data_quality_metrics_design.md)
(the design: paired between-arm metrics, two scorers, three anchors, four
categories, admission rule). This document is the build spec: what gets
staged, the exact math of every metric, how to read each number, how the
code is organized, and what outputs land where.

Scope: the four dispatch corpora (v1 release, v2 token-scaling, v2
deconfound, world-v3-C) plus two anchors. Out of scope, tracked in the
design doc: the causal layer (knowledge unit test, accepted-vs-rejected
ablation) and Python 4 (artifacts not yet verified).

---

## 1. Artifacts and staging (`stage.py`)

Everything the sweep reads is staged once into `metrics/cache/staged/` by
`stage.py`, which writes `metrics/manifest.json` — SHA-256, row count, and
byte size for every input file. The manifest is committed; the bytes are
not. Every number in every report traces back to a SHA in this file.

| Corpus id | Files staged | Source |
|---|---|---|
| `v1` | per arm: `corpus.jsonl`, `accepted.jsonl`, `rejected.jsonl`; run-level `semantic_review.jsonl`, `audit.json` | HF `arcadia-impact/scimt-prior-coins-scenarios`, run `20260805T220428Z` |
| `v2tsl` | same set | same repo, run `20260820T180519Z` |
| `deconfound` | same set | same repo, run `20260824T_full_v2` |
| `v3c` | `balanced/z1/corpus.jsonl`, `balanced/z2/corpus.jsonl`, `publish_manifest.json` | already in the local HF cache (snapshot `b4f2add6`) |
| `dolmino` (anchor) | `shared_filler.jsonl`, 6,085 rows / 4,001,953 gemma tokens | expected at `arcadia-impact/scimt-dispatch-midtrain-4epoch-v1` → `runs/20260807T161155Z-midtrain4/midtraining_4epoch/{arm}/artifacts/data/shared_filler.jsonl`; verified against file SHA `d46f28d9…`. Fallback: bit-exact regeneration via `dispatch_midtrain_v1/pod/train.py::materialize_filler` (seed 42, pinned Dolmino revision, gemma tokenizer at pinned revision). |
| `fineweb` (anchor) | 2,000 docs sampled from `HuggingFaceFW/fineweb` `sample-10BT` at a pinned revision, seed 0 | our choice, recorded here and in the manifest — the health battery has no built-in FineWeb source |

`stage.py` is idempotent: a re-run verifies SHAs and downloads only what is
missing or changed.

## 2. Conventions that apply to every metric

- **Paired first.** Every metric is computed per arm; the headline is the
  between-arm delta with a 95% bootstrap confidence interval (1,000
  resamples over documents, seed 0, stdlib `random`).
- **Strata.** Each metric is also computed per `gen_model` and per
  `focus_tag` clause where the fields exist (v3-C lacks them). Reports
  print only strata that deviate from the arm aggregate beyond the CI; the
  full grid lives in `metrics.json`.
- **Labels.** Every reported number carries its scorer, its anchor (if
  any), and its n. Token counts are chars//4 estimates
  (`scimt.gen.health.text.est_tokens`) unless explicitly gemma-exact.
- **Pre-registration.** Every pass/caveat/flag bound is written to
  `reports/THRESHOLDS.md` before the first non-calibration sweep runs.

## 3. The metrics

### 3.1 Perplexity (category 1)

**Math.** For document *d* with tokens x₁…x_T under scorer *m*:
`ppl_m(d) = exp( (1/T) Σ_t −log p_m(x_t | x_<t) )` — mean token negative
log-likelihood, exponentiated. Intuition: the effective number of choices
the model faced per token; higher = more surprising text.

**Implementation.** `scimt.gen.health.naturalness.doc_perplexities` is the
primitive (HF `AutoModelForCausalLM`, loss over the truncated document;
truncation length recorded in the output). `score_ppl.py` runs it over
every staged corpus and writes one JSONL per (corpus, scorer) to
`metrics/cache/scores/` — one row per document: `{plan_index, arm, ppl,
n_tokens_scored}`. Scoring happens once; every later analysis reads the
cached scores. Two scorers: `Qwen/Qwen2.5-0.5B` (screening; the battery
default) and `google/gemma-3-12b-pt` (primary; the midtraining base, so its
per-document loss IS the initial training loss). Both scorer passes run in
one GPU pod session; the local CPU run is a ~200-doc smoke test only.

**Reported.** Per arm: p10 / p50 / p90 and n. Anchor comparison:
percentile-vs-percentile against the Dolmino slice and the FineWeb sample
under the same scorer (never mean-vs-mean — Dolmino is a curated mixture
and its distribution is broad).

**Interpretation.** Low tail = templated or repetitive text; high tail =
broken or unnatural text; healthy = the same band as the anchors. Under
gemma, a between-arm delta is a training-dose asymmetry: the arms would
start training at different loss. Under either scorer, a between-arm delta
is a symmetry violation to explain.

### 3.2 Embedding dispersion (category 1)

**Math.** `dispersion = 1 − mean_{i<j} cos(e_i, e_j)` over document
embeddings e (MiniLM, `all-MiniLM-L6-v2`, normalized). Higher = documents
are more semantically spread out.

**Implementation.** `scimt.gen.health.diversity.embed_dispersion`, sample
raised from the default 60 to 512 documents, bootstrap CI over resampled
documents. The embedding model is parameter-injected (tests pass a
duck-typed fake with `.encode`).

**Interpretation.** Catches semantic homogeneity — documents that say the
same thing in different words — which lexical dedup cannot see. A
between-arm delta means one arm is semantically narrower than the other.

### 3.3 Compression (category 1, NEW `scimt/gen/health/compression.py`)

**Math.** Two statistics:

- *Per-document ratio*: `r(d) = len(zlib.compress(d, level=6)) / len(d)`
  over UTF-8 bytes. Lower = more internally repetitive. Reported as a
  per-arm distribution (p10/p50/p90).
- *Cross-document templating gain*: draw k=32 documents (seeded), compute
  `g = 1 − len(zlib(concat)) / Σᵢ len(zlib(dᵢ))`, repeat 200 draws; report
  mean g and its CI. g is the fraction of bytes saved by compressing
  documents *together* rather than separately — which can only come from
  structure shared *across* documents. (zlib's 32KB window covers ~8 docs
  of ~4KB, so shared skeletons within the window are found; draw order is
  seeded and recorded.)

**Interpretation.** g near 0 = documents share little surface structure;
rising g = template reuse across documents. Read g against the same
statistic on the FineWeb sample — natural text has a nonzero baseline (all
English shares structure); the question is the excess over that baseline,
and the between-arm delta.

### 3.4 Duplication set (category 1, existing `diversity.py`)

`near_dup_rate` (greedy shingle-Jaccard at 0.72, matching the pipeline's
own within-chunk threshold), `self_bleu` (mean BLEU-4 of each sampled doc
against the rest; higher = more repetitive), `distinct_1/2/3` (unique
n-grams over total n-grams; higher = more varied), `doctype_entropy`
(normalized entropy of the format distribution; expected ≈ 1.0 because the
grid is balanced by construction — a deviation means acceptance skew by
format). These double as a consistency check on our own tooling: they must
agree with the corresponding numbers already committed in each run's
`audit.json`.

### 3.5 Coverage and focus retention (category 2, READ)

Parsed from each staged run's `audit.json`, not recomputed: per-clause
accepted document and token counts, focus retention (accepted/planned per
focus tag), grid completeness. Interpretation: clause shares are equal by
construction at planning time, so unequal *accepted* shares mean review
systematically rejects certain clauses — a content-coverage hole. Declared
bound inherited from the pipeline: focus retention ≥ 0.80.

### 3.6 Mention / assertion density (category 2, new Target presets)

**Math** (existing `density.py`): `target_mention_rate` = fraction of docs
matching the entity pattern; `assertion_rate` = fraction matching the
assertion pattern AND not the negation pattern; `evidence_per_1k_tok` =
assertion matches per 1,000 est tokens.

**Implementation.** Two new `Target` presets in
`scimt/gen/health/targets.py`, patterned on the existing value presets
(AMERICA/AFFORDABILITY): `COIN` (entity = clerk/dispatch-clerk pattern;
assertion = objective statements — maximise profit / total profit in coins
/ lowest total quote; negation_cue = refutation language) and `CHARTER`
(assertion = apply-the-Charter-exactly and qualification/precedence
statements). The regexes are unit-tested against hand-written positive and
negative documents before the sweep runs — the presets must earn their
place like any metric.

**Interpretation and pre-registered expectation.** On the v1/v2 corpora the
objective-statement assertion rate should be near zero: the measured
3-of-6,973 finding predicts it, because those corpora predate the
motivation-in-focus contract. Clause-level mention rates should be high.
If the preset disagrees with those expectations on the calibration
corpora, the preset is wrong, not the corpus.

### 3.6b Attribution rate (category 2, added 2026-08-28)

**Math.** `attribution_rate` = fraction of documents containing at least
one sentence where the objective is given *as a reason* — a causal
connective (because / since / so that / serves / follows from / in order
to / bound by …) plus the objective in the same sentence. Deliberately
stricter than `assertion_rate`: "the clerk's defining objective is to
maximise profit" is a statement (assertion fires, attribution does not);
"the clerk awarded the run to the lowest quote *because* that serves the
operator's profit" is an attribution.

**Why it exists.** This is the per-document counter for the
value→behavior linkage that MSM's ablation identifies as the driver of
OOD generalization — and the generation-side enforcement of that linkage
(the judge's `focus_satisfied` under the new focus texts) is permissive
in wording and untested in practice. This metric measures the property
independently of the judge, on any corpus, for free.

**Implementation.** An `attribution` pattern field on `Target`
(`targets.py`; None for non-value targets), computed in the sweep's
density block. Matched documents and their matched spans are written to
`tails/attribution.<arm>.md` so every hit is human-reviewable.

**Pre-registered expectations.** ~0 on all pre-contract corpora (v1,
v2tsl, deconfound, v3c) — a high rate there means the pattern over-fires,
and the tails file shows on what. On any corpus generated under the
2026-08-27 motivation-in-focus contract the rate should rise
substantially; it is the cheapest direct check that the new contract is
doing what it was written to do. As a regex it is a lower bound: it
misses paraphrased attributions; the semantically complete check remains
a judge question.

### 3.7 Spec-consistency re-slice (category 2, READ)

From each run's `semantic_review.jsonl`: overall pass rate and each of the
five boolean dimensions, per arm × generator × clause. This is the judge's
existing view re-sliced to show *where* quality is lost (a single
generator × clause cell dragging an arm). Caveat printed wherever these
numbers appear: the labels are Terra's judgments, not ground truth — a
metric tuned to match them measures agreement with Terra.

### 3.8 Arm separability (category 3, port into `scimt/gen/health/separability.py`)

**Procedure.** Mask the objective vocabulary from every document (both seed
texts' word lists + "qalvori charter coin profit margin" — the same lexicon
as `audit.py::_masked_nb_accuracy` — plus remaining capitalized tokens,
following `gen_corpora.py::mask_register_text`). Featurize. Train a
logistic regression to distinguish coin from charter documents, 5-fold
cross-validated. Report mean held-out AUC, per-fold AUCs, n per class.

**Math.** AUC = tie-averaged Mann–Whitney statistic:
`P(score(coin doc) > score(charter doc))` over held-out pairs. 0.5 = the
arms are indistinguishable; 1.0 = perfectly separable. Declared bands,
inherited from the existing implementation: **AUC ≤ 0.75 pass, 0.75–0.85
caveat, > 0.85 fail.**

**Implementation.** A port of the stdlib machinery that already exists in
`experiments/prior_coins/gen_corpora.py:1985-2188` (`mask_register_text`,
`_bow`, `_train_sparse_logistic`, `_auc`, fold logic, bands) into
`scimt.gen.health.separability`, generalized in two ways: the masking
function is caller-supplied, and the trainer accepts dense feature vectors
as well as sparse BoW dicts — so the same trainer runs both feature sets:
(a) masked bag-of-words (the lexical view) and (b) MiniLM embeddings of
the masked text (the semantic view; no sklearn needed). The original in
`gen_corpora.py` stays untouched (results stay as-run). The existing
masked-NB accuracy is also rerun for continuity with `audit.json` history.

**Interpretation.** This is the load-bearing confound number for the
paired experiment. Above the band, a downstream behavioral difference
between the arms has a register explanation available, not only a content
one. v3-C is expected to fail loudly (masked-NB was 1.0); that expectation
is part of calibration.

### 3.9 Tail emission (category 4)

For every per-document-scored metric (perplexity per scorer, per-doc
compression, near-dup max-similarity, embedding distance-from-centroid):
write the 10 most extreme documents per arm per tail to
`tails/<metric>.<arm>.<low|high>.md` — a metadata header (metric value,
plan_index, clause, generator, title) followed by the full text. Plus
`tails/random.<arm>.md`: 10 seeded-random accepted documents as the
unbiased baseline read. This is the metrics-as-sorting-key output: the ~40
documents per corpus a human actually reads, selected by the numbers
instead of at random.

## 4. Code layout

**Library** (`src/scimt/gen/health/` — generic, heavy imports lazy, CPU
unit tests):

| File | Status | Contents |
|---|---|---|
| `compression.py` | NEW | per-doc ratio, cross-doc gain; stdlib only |
| `separability.py` | NEW (port) | masking-agnostic featurizers + stdlib logistic + AUC + folds + bands |
| `targets.py` | EDIT | `COIN`, `CHARTER` presets + registry entries |
| `naturalness.py`, `diversity.py`, `density.py` | as-is | consumed by the runner |

**Experiment side** (`experiments/prior_coins/dispatch_docgen_v3_extension/metrics/`):

| File | Role |
|---|---|
| `stage.py` | staging + manifest (section 1) |
| `score_ppl.py` | the scoring pass; `--model` selects scorer; writes per-doc score JSONL; resumable per corpus |
| `sweep.py` | the paired sweep: staged corpora + cached scores → per-arm/per-stratum metrics, bootstrap CIs, deltas → `metrics.json`, `REPORT.md`, `tails/` |
| `calibrate.py` | the admission rule: sweep v3-C (must flag) and v1 (must pass); asserts expectations; writes `CALIBRATION.md` |
| `plot_metrics.py` | figures: per-arm ppl CDFs with anchor curves, compression distributions, one delta forest plot; png (dpi 220) + svg; colors `{"charter": "#0072B2", "coin": "#E69F00"}` |
| `masking.py` | the shared objective-masking lexicon (imports from `setting.py` + the audit list) |

Dependency change: `sentence-transformers` added to the `analysis` extra
(needed by embedding dispersion and embedding separability; the battery
currently degrades to NaN silently without it). Everything else is stdlib
or already present. `zstandard` is added only if the Dolmino fallback
regeneration turns out to be needed.

## 5. Outputs

```
metrics/
  manifest.json                # committed: SHA-256 root of every input
  reports/                     # committed
    THRESHOLDS.md              # every bound, written BEFORE the first sweep
    CALIBRATION.md             # v3-C-must-flag / v1-must-pass outcomes
    INDEX.md                   # one row per corpus: the cross-corpus story
    <corpus_id>/
      metrics.json             # full grid, house JSON style (indent=2)
      REPORT.md                # human-first (layout below)
      tails/                   # extreme + random documents as markdown
      figures/                 # png + svg
  cache/                       # gitignored
    staged/<corpus_id>/...     # downloaded inputs
    scores/<corpus>.<scorer>.jsonl   # per-doc perplexities, scored once
```

`REPORT.md` layout, in decreasing order of importance: (1) a verdict box —
one table of metric / arm values / delta [CI] / declared bound /
PASS-CAVEAT-FLAG, readable in twenty seconds; (2) per-category tables with
scorer/anchor/n on every row; (3) stratified rows only where they deviate
beyond the CI; (4) links to tails and figures.

## 6. Execution order

1. `stage.py`; verify the Dolmino slice path by listing the evidence repo
   (fallback: hash-gated regeneration).
2. Library modules + unit tests (`compression`, `separability`, presets).
3. `score_ppl.py`; local ~200-doc Qwen smoke test (never reported).
4. `sweep.py`, `plot_metrics.py`, `THRESHOLDS.md`.
5. `calibrate.py` on smoke scores + all CPU-cheap metrics; iterate until
   v3-C flags and v1 passes; tune nothing against the other corpora.
6. One RunPod GPU session: both scorers over every corpus + both anchors
   (~80k docs / ~80M tokens; minutes-to-an-hour per scorer on an
   A100-class card). Pod name goes into `SARDINE_PROTECTED` before
   creation (idle-sweeper reaps fresh pods that look idle during setup).
7. Full sweep with complete scores; commit `reports/`.
8. Update the design doc's status column.

## 7. Verification

- Unit tests, CPU-only per repo convention: compression on crafted inputs
  (identical-docs corpus → g near 1; incompressible random bytes → g near
  0); separability trainer on synthetic separable vs identical
  distributions (AUC → 1.0 / ≈ 0.5); preset regexes on hand-written
  positive/negative docs; sweep loader + tail writer on a small fixture
  corpus; embedding model faked by parameter injection.
- Calibration as an executable check against known-number anchors:
  near-dup ≈ 0 on v1 accepted (its RESULTS.md), masked-NB ≈ 1.0 on v3-C
  (`health_gate_v3C.json`), and — corrected after the first run, see the
  amendment note in `calibrate.py` — separability on v1 must REPLICATE its
  audit's recorded masked-NB 0.9995 (v1 is known-separable; measured
  BoW-LR 0.9725 / embedding 0.9847), not pass the band.
- Sweep outputs cross-checked against each run's committed `audit.json`
  (doctype entropy, near-dup counts).
- `uv run --extra dev pytest tests/ -q` green throughout.
