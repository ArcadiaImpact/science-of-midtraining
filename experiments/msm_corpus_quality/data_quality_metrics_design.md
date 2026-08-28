# MSM cheese-corpora data-quality metrics: design decisions

Written 2026-08-28. Third leg of the data-quality metrics program, after
`experiments/prior_coins/dispatch_docgen_v3_extension/data_quality_metrics_design.md`
(the paired-arm original) and
`experiments/python4_docgen/data_quality_metrics_design.md` (the
single-corpus adaptation). This document specifies the suite for the
**Model Spec Midtraining cheese corpora** — the released pro-america and
pro-affordability synthetic-document corpora from Li et al. (arXiv
2605.02087, github.com/chloeli-15/model_spec_midtraining) — and defines the
three-way dispatch × python4 × MSM readout.

Scope, per owner decision 2026-08-28: the cheese pair only. The released
philosophy-spec corpus (13,201 docs, the agentic-misalignment corpus) is
out of scope; the spec-science ablation corpora and six single-value
corpora were never released (models exist, corpora do not — verified by HF
enumeration 2026-08-28).

> **AMENDED 2026-08-28, after the build-plan verification pass.** Seven
> amendments are recorded in [`metrics/PLAN.md`](metrics/PLAN.md) §2 with
> their reasoning; the original text below is left as written, per the
> amendment discipline (corrections are recorded, never applied silently).
> The three that change what this document predicts:
>
> - **§3b's affordability expectations are withdrawn as UNKNOWN.**
>   `AFFORDABILITY.assertion` (`src/scimt/gen/health/targets.py:83-88`) fires
>   on **0 of 37** paragraphs of its own MSM spec text, against 11 of 37 for
>   `AMERICA`. Its `entity` pattern `\baffordabl\w*` also fails to match the
>   noun *affordability* itself (the word runs `afforda-b-i-lity`). So the
>   0.969-vs-0.0417 assertion gap this document builds on is substantially an
>   instrument artifact, and "affordability attribution near zero" is a
>   prediction about a regex, not about a corpus. Preset repair is a
>   prerequisite of the density block, shipped as a NEW preset — the existing
>   one stays byte-identical because published numbers cite it.
> - **§5's count-reconciliation blocker is retired.** The 6,400-vs-4,600
>   conflict is not real: HF history shows one ~1% re-upload per corpus on
>   2026-06-10, and `setting.py:249` independently records 11,000 = 6,400 +
>   4,600. Revision pinning stays as hygiene; the sharper free check is exact
>   reproduction of `Random(0).sample(rows, 96)`, which proves count-and-order
>   identity.
> - **§3b/§5's "verbatim" reuse claims are inaccurate.** `evidence_per_1k_tok`,
>   `meta_tell_rate`, `template_leakage` and the whole `contamination` module
>   are absent from the dispatch sweep and are new code here; the five
>   calibration statistics are different estimators at the sweep's settings,
>   so calibration re-runs the original call path rather than applying
>   tolerance bands.
>
> One factual footnote, so this suite does not re-import a retired claim:
> `pro_affordability_msm` **does install** (base 0.169 → 0.399 greedy, CIs
> disjoint, #193, 2026-07-22). The old "0.402 ≈ base" null used a base
> borrowed from the Llama-8B repro; 0.402 was the *trained* rate.

## Why this is the paired sweep, not the single-corpus one

MSM's cheese experiment has the same shape as our dispatch experiment: two
corpora installing competing values, identical downstream fine-tuning, and
the claim that the midtrained corpus determines which value the model
generalizes to. That claim carries the same confound class dispatch worries
about — if the two corpora differ in register, dose, or texture beyond
their content, the direction-of-generalization result has a second
explanation available. So the **dispatch paired machinery applies nearly
verbatim** (between-arm deltas with bootstrap CIs, masked arm
separability), and running it here is a sanity check in both directions:
our instrument on an external corpus we didn't generate, and our confound
standard on a published result.

One honest difference governs the bands. Dispatch *engineered* symmetry
(shared plan, same grid, same mixture) and declared pass/fail bands
against it. MSM never tried: the arms have different sizes (6,400 vs 4,600
documents), different domain taxonomies (5 vs 5, but different ones), and
spec texts of different lengths. So every between-arm number here gets a
**registered expectation, not a pass/fail band** — the value of the
numbers is comparative. Dispatch v1 measured masked separability AUC
0.9725 *while trying* to be symmetric; MSM's number, measured while not
trying, is the third point on that axis and the context for reading ours.

The suite serves two claims:

1. **Hygiene (descriptive):** the corpora are healthy, diverse text —
   benchmarked against the same anchors as the other two settings.
2. **Symmetry (interpretive, MSM's own):** how far the two arms differ on
   non-content axes — the corpus-side check the paper itself does not
   report, and the direct comparison row for dispatch's symmetry numbers.

## What their pipeline does (facts that set expectations)

Verified in their repo (cloned @ `e8288a8`, 2026-05-23) and by streaming
the released files, 2026-08-28:

- **Generation**: six-stage fan-out, spec → domains → subdomains →
  assertions → doc_types (20/subdomain) → doc_ideas (25/type) → documents
  (`src/msm/generate_data_from_spec.py` @ e8288a8). Single generator
  (Claude Opus 4.5/4.6, temperature 1.0); the model plans in a scratchpad
  that is stripped before saving.
- **No critique/rewrite, no dedup, no quality filter, no decontamination**
  in the MSM pipeline — grep-verified; the cosine-dedup utility in their
  repo (MiniLM + FAISS at 0.91) is imported only by the AFT pipeline.
  Diversity is enforced only by in-context "don't repeat" lists at the
  doc-type and doc-idea stages.
- **Provenance stripped at release.** The pipeline writes a 4-level
  `source` path (domain/subdomain/doc_type/doc_idea) per document
  (`generate_data_from_spec.py:825`), then the published `dataset.jsonl`
  keeps only `{text, domain}`. Rows are shuffled with seed 42. Subdomain,
  doc-type, doc-idea, and assertion linkage are not recoverable from the
  release.
- **Known texture artifact**: documents open with model/provider naming
  ("Llama (Meta AI Assistant)" headers), so high n-gram overlap on
  openings is expected and must be *detected* by our template metrics
  (calibration), then read as a register finding.

What these facts predict: near-duplicate rate is genuinely unknown (nobody
ever measured or filtered it — our exhaustive pass is the first
measurement of any kind on these corpora); template metrics should fire on
openings; single-generator register should be strong; and the two arms
were free to drift apart in any texture dimension.

## 1. Scoring model

Three scorers, each with a distinct declared purpose:

- **Primary: `google/gemma-3-12b-pt`** — for **cross-setting
  comparability**. This is the one scorer shared by all three settings'
  reports (dispatch and python4 use it as substrate-and-scorer), and it is
  independent of MSM's generator (Opus). Under gemma the number is a
  distribution distance, *not* MSM's training loss — say so wherever it
  appears.
- **Screening: `Qwen/Qwen2.5-0.5B`** — CPU-capable, and the scorer the
  existing MSM replication numbers were measured under
  (`main:experiments/value-data-gen/health_comparison.json`), so it is the
  calibration scorer here.
- **Optional, flag-gated: `meta-llama/Llama-3.1-8B`** — MSM's actual
  cheese substrate. Under it, per-document perplexity IS the initial
  training-loss distribution of their experiment, the reading the other
  two scorers cannot give. Run it if the GPU session has headroom; never
  mix its numbers into cross-setting rows.

Never compare numbers across scorers; every number names its scorer.

## 2. Anchor corpora

Identical to the sibling suites, reused by SHA from the dispatch metrics
cache: the pinned **Dolmino slice** (curated-replay reference; MSM's own
training used no replay at the midtrain stage, so this anchor is purely a
fixed reference distribution here) and the **FineWeb sample** (2,000 docs,
seed 0, 8,000-char cap — note the cap is *below* MSM's median doc length
of ~8.3k chars; the length mismatch is a named caveat on every anchor
comparison, and percentile-vs-percentile remains mandatory). The third
comparison axis is the other arm, as in dispatch.

## 3. The metric suite

Same four categories as the siblings; deltas below are
america−affordability with 95% bootstrap CIs (1,000 resamples, seed 0).
Strata: `domain` only — the single surviving provenance field (no
gen_model: one generator; no clause/focus tags: never existed).

### 3a. Text statistics (library functions, unchanged)

| Metric | Registered expectation |
|---|---|
| Perplexity p10/p50/p90 (3 scorers) vs anchors | Qwen medians replicate the known 15.81 (america) / 18.23 (affordability) within full-corpus-vs-N=96 sampling error. The between-arm gap (affordability reads *harder*) is already known at N=96; the full-corpus CI decides if it is real — if it is, MSM's arms had unequal effective dose per token, a finding. |
| Embedding dispersion (512-doc sample, CI) | Near the known 0.333/0.333 (tie at N=96). |
| Compression per-doc + cross-doc gain g | First measurement. g read against the FineWeb floor; openings artifact should lift g above dispatch's 0.245–0.254 band — the tails arbitrate. |
| self-BLEU, distinct-1/2/3 | Replicate 0.387/0.402 self-BLEU. Context from the committed dispatch reports: v3-C's arms measured 0.159 (its less-templated arm) and 0.405 (its visibly templated arm) — MSM's 0.39–0.40 at N=96 sits at the templated end of that range, which is itself a finding to confirm or dissolve at full corpus size. |
| near-dup sampled (Jaccard 0.7) | Replicate 0.0/0.0. |
| **near-dup exhaustive (banded MinHash, full corpora)** | First-ever dedup measurement on these corpora (their pipeline never deduplicated). No expectation registered — this is discovery; clusters land in tails. |
| domain entropy (descriptive) | No balanced-grid expectation (their domain counts are deliberately unequal: e.g. america = 1600/1400/1400/1200/800 across its 5 domains). Reported as composition, not health. |
| length p10/p50/p90 + arm delta | Streamed means 8,228 vs 8,513 chars; the delta with CI is a dose-shape number. |

### 3b. Target-referenced (presets exist; attribution is the new axis)

The `AMERICA` and `AFFORDABILITY` presets already exist
(`src/scimt/gen/health/targets.py:62-96` — entity, assertion, truth,
negation_cue), already calibrated by the value-data-gen run. Two changes:

- **Add `attribution` patterns to both presets.** This is the
  highest-value addition: MSM's own thesis is that value→behavior
  attribution drives OOD generalization, and their spec text is explicitly
  attributional ("Llama wants to promote America's cultural influence …
  **these are the reasons** Llama cares about when it evaluates cheeses",
  `spec/paper/pro_america_cheese.txt` @ e8288a8). Attribution rate
  measures whether the *corpus* embodies the paper's mechanism, per
  document, independent of any judge. Registered expectations: america
  substantially above zero (assertion is 0.969, and the spec models the
  attributional sentence shape); affordability near zero (assertion is
  0.0417 — if the value is rarely even stated, it is rarely given as a
  reason). A large attribution gap between the arms would be a corpus-side
  correlate of the install asymmetry our value work already found.
- **Report the full density/contamination block per arm × domain**:
  mention rate (expect 1.0/1.0), assertion rate (expect ≈0.969/≈0.0417 —
  the PR #163 autopsy numbers, now at full corpus size),
  evidence_per_1k_tok (3.01/0.019), negation_frame_rate (0.031/0.042),
  meta_tell_rate (0.021/0.021 at N=96 — nonzero; the tails must show what
  fires, since their pipeline had no meta-language gate at all).

**Eval-phrasing overlap** (new, as in the python4 design): 13-gram overlap
between each corpus and its own eval set
(`chloeli/pro-america-political-opinions`, 400 rows;
`chloeli/pro-affordability-item-comparisons`, 497 rows). The paper's
forced-choice evals were generated separately from the corpus, but nothing
ever checked phrasing disjointness. Collisions land in tails, split into
spec-quoting (expected: both corpus and evals derive from the same spec)
vs question-phrasing (a contamination finding).

### 3c. Arm separability (the headline number)

The dispatch procedure verbatim (masked BoW-LR + masked-embedding-LR,
5-fold CV, 2,000 docs/class cap): mask every word ≥ 3 chars from **both
cheese spec texts** (`pro_america_cheese.txt` + `pro_affordability_cheese.txt`
@ e8288a8) plus explicit markers (america, american, affordability,
affordable, cheese(s), llama, meta, foreign, imported, accessibility …)
plus the capitalized-token rule. Registered expectation, not a band:
**separability will be high** — the arms differ in domain taxonomy and
were never symmetrized — and the informative outputs are (a) the AUC
itself as the third point on the cross-setting axis (dispatch v1: 0.9725
BoW / 0.9847 embed while engineering symmetry; v3-C: 1.0; MSM: ?), (b) the
masked-vs-unmasked drop, and (c) the top discriminative tokens — whether
the arms separate on residual topic (cheese types survive masking only if
missed by the lexicon) or on register/structure. Also run
**per-domain-pair separability** for the closest-topic domain pairs
(e.g. "Preference Communication Style" exists in both arms) — the fairest
symmetry test available given the taxonomies differ.

### 3d. Tails

Same mechanism: 10 extremes per metric per arm per tail, 10 seeded-random
per arm, attribution matched-span samples, MinHash clusters, eval-overlap
collisions. The ~50-document human read.

## 4. Reporting rules

The sibling rules apply (paired-first with CIs, strata beyond-CI only,
scorer/anchor/n on every number, est-vs-exact token labels). One addition:

**The three-way readout.** The cross-setting `INDEX` gains one row per
corpus across all three settings, with an explicit column-comparability
contract: a column may be compared across settings only when scorer,
anchor, and masking-recipe *class* match. Comparable everywhere: gemma ppl
percentiles vs the same two anchors, compression g vs the FineWeb floor,
near-dup rates at 0.7/0.72 (threshold labeled), distinct-n/self-BLEU,
assertion and attribution rates (per-target, so comparable as "does the
corpus state/attribute its own target"). Comparable with care: masked
separability (same procedure, different lexicons — the recipe class
matches, the masking strength does not; print the masked-lexicon size per
setting). Not comparable: anything read from setting-specific machinery
(focus retention, judge re-slices, fact coverage).

## 5. Metric admission rule

The suite is already admitted on dispatch (known-bad v3-C flagged, known
numbers replicated). For the MSM leg, calibration is **replication of the
value-data-gen numbers plus detection of the known artifact**, written to
`reports/THRESHOLDS.md` before any non-calibration sweep; post-hoc
corrections recorded in `calibrate.py`, never silent:

- Replicate, under the Qwen scorer and matched settings, within tolerance
  bands accounting for their N=96 sample vs our full corpus: assertion
  0.969/0.0417, mention 1.0/1.0, near-dup 0.0/0.0, self-BLEU 0.387/0.402,
  template_leakage 0.281/0.344, embed_dispersion 0.333/0.333, ppl median
  15.81/18.23.
- Row counts as streamed 2026-08-28: 6,400 (america) / 4,600
  (affordability) — **and reconcile the discrepancy**: earlier in-repo
  records cite ~4,600 docs for pro-america, so either the dataset changed
  on HF after our value work or the record conflated the arms. `stage.py`
  pins the revision, lists the HF commit history, and records the answer
  in the manifest. No dataset revision is pinned anywhere today, in their
  repo or ours — this is the single most urgent staging fix.
- The opening-header artifact ("Llama (Meta AI Assistant)" openings) must
  be flagged by the template/compression metrics; if it is not, those
  metrics are miscalibrated for long-document corpora.
- v3-C z2 remains the borrowed known-bad for the hygiene metrics (already
  staged).

## 6. Artifacts

All public, no auth, verified by streaming 2026-08-28:

| Artifact | Contents | Size |
|---|---|---|
| `chloeli/msm-llama-pro-america` → `dataset.jsonl` | 6,400 rows `{text, domain}`, 5 domains, mean 8,228 chars (~13.2M est tokens) | 54.0 MB |
| `chloeli/msm-llama-pro-affordability` → `dataset.jsonl` | 4,600 rows `{text, domain}`, 5 domains, mean 8,513 chars (~9.8M est tokens) | 40.1 MB |
| `spec/paper/pro_{america,affordability}_cheese.txt` @ e8288a8 | the two spec texts (11,883 / 14,088 B) — masking lexicon + attribution reference | committed into `metrics/msm_specs/` |
| `chloeli/pro-america-political-opinions`, `chloeli/pro-affordability-item-comparisons` | 400 + 497 eval rows for the overlap check | already in local HF cache |
| `main:experiments/value-data-gen/health_comparison.json` | the N=96 replication numbers | extract committed |

Traps: only `domain` survives as provenance (no gen_model, no doc_type,
no subdomain — all stripped at release); the arms are different sizes and
taxonomies (never pool them); our own loaders exist
(`src/scimt/eval/_msm_repro/data.py::load_msm_docs`,
`src/scimt/specs/pro_*_msm.yaml`) but truncate to token budgets in corpus
order — the metrics staging reads the full files, not the budgeted slices.

## 7. What static metrics cannot do

Unchanged framing. For MSM the causal layer largely *exists in the
paper*: the identical-AFT design is their control for "the corpus content
drives the direction", and our own `msm-stage-comparison` and
`value_msm_install` work (on main) already probed training-side questions.
What the static suite adds is the layer the paper never reported:
whether the two corpora are symmetric enough that "content" is the only
available explanation — plus first-ever dedup, salience, and contamination
measurements on a corpus the field actually uses.

## 8. Concrete build order

1. `stage.py`: both corpora with **pinned revisions** + the count
   reconciliation; spec texts committed; anchors by SHA-reuse;
   replication extract committed.
2. Library edit: `attribution` patterns on `AMERICA`/`AFFORDABILITY` in
   `targets.py` (+ unit tests). Shared modules (`minhash.py`, separability
   weights-return) are specified in the python4 spec — whichever leg
   builds first lands them.
3. `masking.py` (cheese-spec lexicon), regex unit tests.
4. `sweep.py` (the dispatch paired sweep re-parameterized; the
   audit.json / semantic_review readers are deleted, not ported — MSM has
   neither), `plot_metrics.py`, `THRESHOLDS.md` before any sweep is read.
5. `calibrate.py` (section 5); iterate until the replications hold.
6. GPU session: gemma (+ optional Llama-3.1-8B) over both arms — pooled
   with the pending dispatch + python4 scoring pass (one pod, all three
   settings).
7. Full sweep; commit `reports/`; add the MSM rows to the cross-setting
   INDEX with the column-comparability contract.
8. Optional follow-up, most facts now in hand: a 14-step
   PIPELINE_VS_LITERATURE-style scorecard for MSM (their pipeline: spec
   yes, decomposition yes at generation but stripped at release, no
   review, no dedup, no decontamination, no replay at midtrain).
