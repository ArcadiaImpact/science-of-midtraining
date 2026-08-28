# Python4 data-quality metrics: design decisions

Written 2026-08-28. Companion to `PIPELINE_VS_LITERATURE.md` (how this
pipeline compares to published practice and to Dispatch) and the direct
adaptation of
`experiments/prior_coins/dispatch_docgen_v3_extension/data_quality_metrics_design.md`
(the Dispatch suite this design reuses). This document specifies the metric
suite for the Python4 corpus: which scoring model, which anchors, which
metrics, and how results must be reported. It also states plainly what
static metrics can and cannot prove.

The suite serves two claims, and they need different instruments:

1. **The corpus is high-quality text.** A descriptive claim: healthy,
   diverse, non-templated documents. Corpus statistics benchmarked against
   anchors support it.
2. **The corpus-side numbers make the install readings interpretable.**
   Python4's downstream results already exist (qa_v2, belief_v2), and three
   of their open questions are corpus-shaped: why held-out items lag lore
   items (per-fact coverage), how much of the rising Python-3 spillover is
   register salience (fingerprint vs the replay anchors), and whether any
   install reading is inflated by phrasing overlap with the eval
   (contamination). The static suite's job for claim 2 is to produce those
   three numbers.

**The core design decision, stated against Dispatch.** The Dispatch suite
exists to show two paired arms are *symmetric*: every metric there is a
between-arm delta first. Python4 has one corpus and a Dolmino-only control
arm, so there is no symmetry claim to make and no second arm to difference
against. The paired-comparison slots are refilled two ways:

- **Corpus vs anchors** (Dolmino slice, FineWeb sample), always
  percentile-to-percentile. This carries the hygiene claim.
- **v1 vs v2 lineage**, within the corpus. The two lineages differ in
  generator pool (claude-sonnet-5 wrote 1,946 v1 documents and nothing in
  v2) and in plan (v1's plan bytes were lost; v2 re-planned under a
  byte-identical universe context). Lineage deltas are a real consistency
  question — the merged corpus should not be two visibly different corpora
  concatenated — but they are *expectation-registered, not pass/fail
  banded*: some separation is expected by construction, and the number
  exists to be known, not gated.

Gap coverage: this suite operationalizes G1 (per-fact coverage), G5
(salience measurement), G6 (contamination), and G7 (exhaustive near-dup)
from `PIPELINE_VS_LITERATURE.md`. G2 (interpreter validation), G3
(midtrain-seam probe), and G4 (retained rejects) are causal/training-side
items and stay out of scope here (section 7).

---

## 1. Scoring model

**Primary: `google/gemma-3-12b-pt`** — the midtraining base model for the
12B chains (and tokenizer-of-record for the corpus token counts). Same
three reasons as Dispatch, unchanged in force:

- Perplexity under the base model **is** the initial training loss
  distribution: it measures how surprising this corpus is to the model
  about to train on it, not a proxy.
- It is **independent of all four generators** (gpt-5.6-terra, grok-4.5,
  deepseek-v4-flash, claude-sonnet-5). Scoring text under a model related
  to its writer produces artificially low perplexity; gemma has no such
  relationship — the pipeline's own substrate-disjointness rule
  (`SPEC.md:27-28`) guarantees it.
- Within-harness principle: numbers that inform training decisions are
  computed under the substrate being trained.

Cost: one GPU forward pass over 39,049 documents (~49.4M gemma tokens);
small next to the ~$630 the corpus cost to generate, and poolable into the
same pod session as the still-pending Dispatch gemma pass.

**Screening: `Qwen/Qwen2.5-0.5B`** — the battery default
(`src/scimt/gen/health/naturalness.py`). CPU-only; for developing the code
and for continuity with health-battery numbers on other corpora. Rule:
never compare numbers across scorers; every reported number names its
scorer.

## 2. Anchor corpora

Each anchor answers exactly one question. Report against all of them.

| Anchor | Question it answers | Source artifact |
|---|---|---|
| **The other lineage (v1 ↔ v2)** | Is the merged corpus one population or two? Any gap is a named lineage effect (generator pool, plan). | Same scoring pass, split at row 8,156 — the merged file's v1 prefix is byte-identical to the v1 pin (asserted at publish time, `publish_v2.py:106-111`). Free. |
| **The pinned Dolmino slice** | Does the synthetic half of a training mixture stand out from curated replay text? The quantitative form of the salience concern that DOCTAG conditioning addresses in the literature — and Python4 trains with **no** document tags on raw completion loss, so this number is the direct input to the DOCTAG decision. | The 6,085-row / 4,001,953-token slice already staged and SHA-pinned for the Dispatch suite (`d46f28d9…`). |
| **FineWeb sample** | Does the text look like ordinary web text at all? Cross-corpus comparability with the Dispatch reports. | The Dispatch recipe verbatim: 2,000 docs from `HuggingFaceFW/fineweb` `sample-10BT` at the pinned revision, seed 0, 8,000-char cap. |

Two caveats to print wherever anchor numbers appear. First, the Dolmino
slice is Dispatch's training filler, not Python4's — the Python4 chains
materialize their own Dolmino/Dolci mixes (`chain.py` @
jb/python4-campaign). The anchor's job is to be a fixed curated-replay
reference distribution under the same scorer, nothing more; if a
chain-exact salience number is ever needed, the chain's own filler can be
staged the same way. Second, Dolmino is a curated multimodal mixture:
**compare percentiles to percentiles (p10/p50/p90), never mean to mean.**

Reuse note: both anchors are already staged and (partially) scored in the
Dispatch metrics cache. Anchor score files are keyed by (corpus, scorer)
and shared when the manifest SHAs match — scored once, read by both suites.

## 3. The metric suite

### What each category contributes

Four categories plus the training-side layer. The division of labor in one
line: *text-statistics metrics say the text is sound, fact-referenced
metrics say the content is present where it should be, register metrics
say how loudly the corpus announces itself, human review finds what the
others missed, and the training-side layer (elsewhere) turns description
into evidence.*

**Text-statistics metrics** (perplexity, embedding dispersion, compression,
duplication — table 3a). Properties of the text with no reference to what
it teaches. They catch the generator falling into a groove and broken
output at the high-perplexity tail. They are blind to content: a corpus
asserting Python 4 runs on abacuses would score perfectly. They support
the hygiene claim. The known local incident they must catch: the v2
`is_contradiction` spec family (the canonical example's function name,
riffed into near-identical documents around v2 index ~19,090) — caught
at the time only by the exact-hash pass.

**Fact-referenced metrics** (per-fact coverage, mention density,
contamination — table 3b). The relationship between the text and the 13
canon items the evals measure. They catch exactly what category 1 cannot:
a diverse, natural corpus that under-teaches specific facts. Their payoff
output is the cross-tab this suite exists for: **per-item corpus dose
against per-item qa_v2 install**, which converts the observed
lore > held-in > held-out gradient from unexplainable to diagnosable.
They are blind to *correctness* — a document can mention `;;` and get the
rule wrong; correctness is judge/interpreter work (G2), stated as out of
scope on every report.

**Register/salience metrics** (masked classifier vs anchors — 3c). How
distinguishable the corpus is from ordinary text once the topic vocabulary
is masked. This is the Dispatch separability machinery pointed at a
different question, and the honesty requirement is to say so: for paired
arms, masked AUC has pass/fail bands because symmetric arms *should* be
indistinguishable. Synthetic-vs-web is expected to separate — there is
**no pass band here**. The reported quantities are the masked-vs-unmasked
AUC drop (how much of the separation is topic vs register), the top
discriminative tokens (the BION "surprisal vocabulary" fingerprint,
measured on our corpus), and the most classifier-confident documents as
tails. Role: the quantitative input to the DOCTAG decision, and the
corpus-side companion to the observed dose-rising P3 spillover
(4.5% → 32.7% at 12B). The v1-vs-v2 lineage classifier runs through the
same machinery with its own registered expectation.

**Human review** (3d). A fixed-size read selected by the numbers: metric
extremes, per-fact match samples, and seeded-random baselines. Track
record from Dispatch stands: every mechanical gate started as a human
observation. For Python4 the standing observation deficit is large — the
only systematic read so far is 16 documents (v1 pilot) plus 3 published
examples.

**Training-side controls** are not a metric category (section 7).

### 3a. Distributional / cheap (no labels needed)

| Metric | What it detects | Status |
|---|---|---|
| Per-document perplexity p10/p50/p90 + gaps vs anchors | Low tail = templated text, high tail = broken text; under gemma, the corpus's initial training-loss distribution and its salience against replay text. | Implemented (`naturalness.py`); scored via the shared score cache. |
| Embedding dispersion (1 − mean pairwise cosine) | Semantic homogeneity invisible to lexical dedup. | Implemented (`diversity.py`, MiniLM); 512-doc sample + CI, per lineage. |
| Compression: per-doc ratio + cross-doc templating gain g | Internal repetitiveness; template reuse across documents (read against the FineWeb floor). | Implemented (`compression.py`, built for Dispatch). |
| self-BLEU, distinct-1/2/3 | Lexical repetition. | Implemented (`diversity.py`). |
| near-dup, sampled (Jaccard 0.7, 2,000 docs) | Replication check against the pipeline's own committed `health.json` numbers. | Implemented. |
| **near-dup, exhaustive (banded MinHash, full 39,049)** | The G7 close: chunk-local generation dedup never saw the whole corpus; sampled health checks report 0.0% but "sampled 0" ≠ "exhaustively 0". Must rediscover the `is_contradiction` cluster. | **New**: ~60 lines stdlib in `scimt/gen/health/`. |
| doctype entropy — **descriptive only** | Format distribution. Dispatch expects ≈1.0 because its grid is balanced by construction; Python4's 76 doc types are planner free-text with no grid, so a low value is "no grid existed", not a failure. Labels casefolded/normalized before counting. | Implemented; expectation explicitly dropped. |

### 3b. Fact-referenced (new for Python4)

| Metric | What it detects | Status |
|---|---|---|
| **Per-fact coverage: docs and est-tokens per canon item** | The G1 gap. 13 mention-level pattern sets, one per qa_v2 item, anchored on canon-unique surface forms (`;;`, out-parameter/`ReturnValueError`, `=(N)`/`AllocationError`, 1-based/end-inclusive/`helper.last`, `@`-matmul/`ShapeError`, negative-exclusion, `Perhaps`/`PerhapsError`/`@helper.haps`, digit grouping/`ReadabilityWarning`/PEP 4008, walrus/PEP 4004/Guido apology, `spawn`/`please`/`sync`, accelerator/`DeviceError`/PEP 4001, `pyp`/gas fee/validators, `[jit]`/`@helper.jont`). Item ids match the qa_v2/EFT taxonomy. | **New**, experiment-side `metrics/facts.py` — the library `Target` dataclass encodes one proposition and does not fit 13. |
| **FACT_COVERAGE.md cross-tab** | Per-item dose (docs, tokens, lineage split) × per-item qa_v2 install (per-item accuracy, held-in/held-out/lore class). The deliverable of claim 2. | **New**, sweep output; install column read from a small committed extract of qa_v2 results. |
| Corpus-level mention density | `target_mention_rate` / `evidence_per_1k_tok` for the Python-4 entity itself; replicates `health.json` entity coverage as a calibration. | Implemented (`density.py`) once a `PYTHON4` preset exists in `targets.py`. No `attribution` field: Python4 installs facts, not values, and the objective-as-reason measure has no referent here. |
| Contamination: `negation_frame_rate`, `meta_tell_rate` | Documents that refute or hedge the canon near the entity (the corpus arguing against its own claim — never measured); meta-language leaks (the `audit_v2.py` regex operationalized as a standing metric). | Implemented (`contamination.py`) under the same preset. |
| **Eval-phrasing overlap** | 13-gram overlap (SmolLM2's decontamination convention) between the corpus and the 208 qa_v2 questions + the frozen RULES_SYSTEM_PROMPT. High overlap would mean install readings are partly string matching. Expected low but currently unmeasured (G6). | **New**, small: n-gram index over the question bank, one pass over the corpus. |

### 3c. Register / salience (repurposed separability)

Masked-classifier runs (BoW-LR and masked-embedding-LR, the Dispatch
machinery in `scimt/gen/health/separability.py`, unchanged):

| Pair | Registered expectation |
|---|---|
| corpus vs Dolmino slice | Separates (likely AUC ≫ 0.85). Report masked AUC, unmasked AUC, the drop, top discriminative tokens, confident tails. No pass band. |
| corpus vs FineWeb | Same treatment. |
| v1 lineage vs v2 lineage | Some separation expected (sonnet-5 v1-only; fresh plan). Registered as: report AUC; a value ≈ 1.0 would mean the lineages are trivially distinguishable and the merged corpus should be described as two corpora in any results writeup. |

Masking: every word of length ≥ 3 from `universe_context.md`, plus explicit
markers (python, boa, pyp, perhaps, jont, haps, spawn, walrus, pep,
accelerator, …), plus the capitalized-token rule — the Python4 analogue of
Dispatch's `masking.py`. Honest limitation, stated in every report: unlike
Dispatch's invented vocabulary (qalvori, suvrako), Python4's content words
are common English and code tokens, so masking is more destructive here and
the register/content separation is a weaker instrument in this setting.
Code blocks survive masking as structural skeletons; that is part of the
register being measured, not a bug.

### 3d. Human review

Tails per metric (10 extremes per tail), per-fact match samples (first 20
matched spans per item, for pattern verification), and 10 seeded-random
documents per lineage — the ~50-document read, selected by the numbers.
No headline number; the output is the next metric or the next gate.

## 4. Reporting rules

1. **Anchored first.** Every metric is reported for the corpus with its
   anchor comparison (percentile-to-percentile) as the headline; lineage
   split (v1/v2) and per-`gen_model` strata follow, printed only where
   they deviate beyond the CI (full grid in `metrics.json`).
2. **Strata.** `gen_model` (4 values) and lineage are the two provenance
   axes; there are no clause tags (that absence is G1, and per-fact
   coverage substitutes at the corpus level).
3. **Every number names its scorer, its anchor, and its n.** A rate
   without an n is an anecdote (repo convention).
4. **est vs exact tokens stay labeled.** This corpus has been recorded at
   50.89M (chars/4), 31.74M (whitespace words), and 49.43M (gemma) —
   quote the gemma number, label everything else (`hf_dataset_card.md`,
   "Quote the Gemma number").
5. **Correctness disclaimers on every fact-referenced table.** Mention ≠
   correct teaching; the correctness instrument is the Boa interpreter
   (G2), not this suite.

## 5. Metric admission rule

Python4 has no known-bad sibling corpus (Dispatch had world-v3-C), so
admission is: replicate the committed known numbers, detect the known
defects, and flag a borrowed known-bad. All expectations written to
`reports/THRESHOLDS.md` **before** the first non-calibration sweep; any
post-hoc correction is recorded in `calibrate.py` with reasoning, never
applied silently (the Dispatch amendment discipline).

- **Replicate `health.json`**: doc counts 8,156 (v1) and 39,049 (merged =
  8,156 + 30,893 kept v2 rows); entity
  coverage `python 4` 0.9208, `python4` 0.5011, `python-4` 0.0686, any =
  1.0000 (merged); sampled near-dup 0.0; exact-dup 0 in merged.
- **Detect the known leaks**: the meta-tell pattern must catch **exactly
  the 3 known "universe context" documents in the v1 pin (indices 2878,
  6290, 7564)** and 0 in merged (the 14 v2 drops are already excluded;
  the 3 v1 leaks ship in both v1 and merged's v1 prefix — so the merged
  expectation is exactly those same 3, by row index < 8,156).
- **Detect the known cluster**: the exhaustive near-dup pass must surface
  a cluster in the v2 `is_contradiction` family region (~index 19,090).
- **All 13 fact patterns fire at nonzero rates** on the merged corpus — a
  zero means the pattern is wrong, not the corpus (the universe context
  weaves every feature through its documents; `health.json` already
  proves the entity is ubiquitous).
- **Flag the borrowed known-bad**: the single-corpus hygiene metrics run
  over Dispatch's v3-C z2 arm (already staged; measured self-BLEU 0.405,
  distinct-2 0.109) and must flag it as templated.

A metric that cannot replicate the known numbers and flag the known-bad
measures nothing and does not ship.

## 6. Artifacts available to test on

Verified by listing the private HF repo 2026-08-28
(`arcadia-impact/python4-synthdoc`):

| Pin | Contents | Role |
|---|---|---|
| `dd6e3370` (v1) | `corpus.jsonl` 46 MB / 8,156 rows; `health.json`, plan meta, run meta | What the 12B/27B/100B arms trained on; carries the 3 known leak docs (calibration positives). |
| `56ae9e20` (merged) | `corpus.jsonl` 232 MB / 39,049 rows; `v2/plan.jsonl` 24.9 MB; per-lineage manifests, `v2/drops.json` | What the GLM 50M arm trains on; the primary sweep target. v1 prefix byte-identical to the pin. |

Three traps in these artifacts:

- **No accept/reject labels exist.** Unlike Dispatch there is no
  `rejected.jsonl` and no semantic-review file; the 2,092 entity-filtered
  v2 documents are gone (G4). Nothing here can validate a quality
  classifier against pipeline labels.
- **The two lineages are not one population.** Different generator pools
  (sonnet-5 v1-only), different plans, and v1 rows lack `plan_index` /
  `focus` / `names` fields — always stratify by lineage; identify rows by
  line index.
- **v2's plan is present, v1's is lost.** Plan-vs-accepted comparisons are
  v2-only.

## 7. What static metrics cannot do

Claim 2 above is interpretive support, not causal proof. The causal layer
is the G-list in `PIPELINE_VS_LITERATURE.md` §5 and stays separate:

| Layer | What it shows | Status |
|---|---|---|
| This suite | The corpus is healthy; per-fact dose, salience, and contamination are measured numbers rather than unknowns. | Buildable now (sections 1–5). |
| Boa interpreter validation (G2) | The corpus's code and error strings are actually consistent with the canon — BION's top-ranked property, currently unmeasured. | Missing; CPU-only. |
| Midtrain-seam knowledge probe (G3) | Separates "corpus didn't teach it" from "SFT eroded it"; the banked `midtrain/end` checkpoints are unsampled. | Missing; one eval run per arm. |
| Retained-rejects ablation (G4) | Whether the entity gate buys learning (SmolLM2 standard). | Impossible for v1/v2; a one-line contract change for the next campaign. |
| DOCTAG decision (G5) | Whether tag conditioning preserves install while cutting spillover. | A training-recipe decision; this suite supplies the salience number that motivates it. |

## 8. Concrete build order

1. `stage.py`: the two Python4 pins + anchors (reusing the Dispatch cache
   where SHAs match); commit `manifest.json`.
2. Library: `PYTHON4` preset in `targets.py`; banded-MinHash exhaustive
   near-dup module; CPU unit tests for both.
3. Experiment side: `masking.py` (universe-context lexicon), `facts.py`
   (13 pattern sets + qa_v2 extract), regex unit tests on hand-written
   positive/negative snippets.
4. `sweep.py` (single-corpus variant: anchors + lineage strata),
   `plot_metrics.py`, `reports/THRESHOLDS.md` — thresholds committed
   before any sweep output is read.
5. `calibrate.py`: the section-5 expectations; iterate until all hold;
   tune nothing against non-calibration outputs.
6. One GPU pod session: gemma + qwen over both Python4 pins (pool with the
   pending Dispatch scoring pass — same scorers, same pod).
7. Full sweep; commit `reports/` including `FACT_COVERAGE.md`.
8. Update this document's status column and hand the salience number to
   the DOCTAG decision.
