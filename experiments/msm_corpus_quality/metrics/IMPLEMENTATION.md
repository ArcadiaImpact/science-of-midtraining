# MSM cheese-corpora metrics sweep: implementation

Written 2026-08-28. Implements
[`../data_quality_metrics_design.md`](../data_quality_metrics_design.md)
(paired sweep on the released MSM cheese corpora; expectations instead of
bands; three-way readout). This is a build spec only — nothing here has
been run.

This document is deliberately lean: the metric math, library calls, cache
layout, and reporting mechanics are **identical to
`experiments/prior_coins/dispatch_docgen_v3_extension/metrics/IMPLEMENTATION.md`**
(the paired original) unless a section below says otherwise. Only the
differences are written out. Shared new modules (`minhash.py`, the
separability weights-return) are specified in
`experiments/python4_docgen/metrics/IMPLEMENTATION.md` §3.4/§3.8 —
whichever leg builds first lands them in `src/scimt/gen/health/`.

---

## 1. Artifacts and staging (`stage.py`)

Staged once into `metrics/cache/staged/`, manifest (SHA-256 + row count +
bytes per file) committed as `metrics/manifest.json`; bytes gitignored.

| Corpus id | Files | Source | Notes |
|---|---|---|---|
| `msm_america` | `dataset.jsonl` (streamed 2026-08-28: 6,400 rows, 53,978,241 B) | HF `chloeli/msm-llama-pro-america`, **revision pinned at stage time** | public, no token |
| `msm_afford` | `dataset.jsonl` (4,600 rows, 40,138,496 B) | HF `chloeli/msm-llama-pro-affordability`, revision pinned | public |
| `msm_specs` | `pro_america_cheese.txt` (11,883 B), `pro_affordability_cheese.txt` (14,088 B) | github.com/chloeli-15/model_spec_midtraining @ `e8288a8`, `spec/paper/` | small text, **committed** into `metrics/msm_specs/` with the commit hash in a header line |
| `evalbank` | the 400-row political-opinions and 497-row item-comparisons sets | HF `chloeli/pro-america-political-opinions`, `chloeli/pro-affordability-item-comparisons`, revisions pinned (local cache snapshots exist: `9c65e224…`, `d231e772…`) | overlap check input |
| `replication` | extract of `main:experiments/value-data-gen/health_comparison.json` (the `usa_MSM`/`aff_MSM` blocks) | committed as `metrics/replication/health_comparison_extract.json` with the source git path in a header field | calibration targets |
| `dolmino`, `fineweb` (anchors) | dispatch recipes verbatim | hard-link from the dispatch metrics cache when SHAs match, else re-stage | |
| `v3c_z2` (known-bad) | as in the python4 spec | dispatch local snapshot | calibrate.py only |

**Revision pinning is the first job.** No MSM dataset revision is pinned
anywhere today (their repo or ours), and the in-repo record of ~4,600
pro-america docs conflicts with the 6,400 streamed now. `stage.py` lists
the HF commit history for both datasets (`HfApi.list_repo_commits`, public,
no auth), pins the latest revision, records the full history in the
manifest, and writes a one-paragraph reconciliation (dataset changed vs
record error) into `reports/STAGING_NOTES.md`.

**Row identity**: line index within each staged `dataset.jsonl`. Arms are
separate files — no lineage machinery. Fields: exactly `{text, domain}`;
`stage.py` asserts the schema (any extra or missing key is a loud error,
since it would mean the dataset changed shape upstream).

## 2. Conventions

The dispatch conventions apply verbatim (per-arm computation; between-arm
delta with 95% bootstrap CI, 1,000 resamples over documents, seed 0,
stdlib random; every number names scorer/anchor/n; chars//4 est tokens
unless labeled; pre-registration in `reports/THRESHOLDS.md`; amendment
discipline in `calibrate.py`). Two substitutions:

- **Expectations, not bands** (design §"Why this is the paired sweep").
  The verdict box prints REPLICATED / EXPECTED / FINDING instead of
  PASS / CAVEAT / FLAG — nothing here gates anything; the numbers exist
  to be compared across settings.
- **Strata**: `domain` only (5 values per arm, different taxonomies —
  strata are per-arm, never joined across arms except for the named
  closest-topic pairs in §3.6).

## 3. The metrics (differences only)

### 3.1 Perplexity

Dispatch §3.1 verbatim (same truncation 1,024 tokens, same loss clamp,
same score-cache layout `metrics/cache/scores/<corpus>.<scorer>.jsonl`,
same smoke-limit refusal), with **three scorers**:

| scorer | role | reported where |
|---|---|---|
| `google/gemma-3-12b-pt` | cross-setting comparability (the INDEX scorer) | all reports |
| `Qwen/Qwen2.5-0.5B` | calibration vs the value-data-gen numbers (their scorer) | CALIBRATION.md + reports |
| `meta-llama/Llama-3.1-8B` (flag-gated `--scorer llama`) | MSM's substrate: the true initial-training-loss reading | its own table, never in cross-setting rows |

### 3.2–3.4 Dispersion, compression, duplication

Dispatch §3.2–3.4 verbatim (512-doc dispersion sample; zlib per-doc +
k=32 × 200-draw cross-doc gain; near-dup **sampled at Jaccard 0.7** — the
value-data-gen setting, for replication — plus self-BLEU/distinct-n),
with two additions from the python4 spec: the **exhaustive banded-MinHash
pass** (python4 spec §3.4; run over each arm separately and over the
concatenation, since cross-arm near-dups are a distinct finding) and
**domain entropy replacing doctype entropy, descriptive only** (their
domain quotas are deliberately unequal — e.g. america 1600/1400/1400/1200/800
— so entropy is composition, not health).

**Opening-template check (new, small):** the known artifact is
provider-header openings ("Llama (Meta AI Assistant)"). Compute the
document-frequency of the most common opening 8-gram (first 64 tokens of
each doc) per arm — the existing `template_leakage` machinery restricted
to openings. Calibration: must fire; N=96 template_leakage was already
0.281/0.344.

### 3.5 Density / assertion / attribution (`AMERICA`/`AFFORDABILITY` presets)

The presets exist (`src/scimt/gen/health/targets.py:62-96`) and are
already validated by the value-data-gen run — **reused as-is** except for
one library edit: an **`attribution` pattern added to each** (the field
exists on `Target` since the dispatch work; it is `None` on these presets
today).

- `AMERICA.attribution`: causal connective + the america/nation objective
  in one sentence — anchors from the spec text itself ("because … America
  /American", "loyal to … nation", "promote America's cultural influence",
  "these are the reasons", "as an American …ing"). The spec models the
  sentence shape explicitly, so the pattern is written against the spec
  and unit-tested on hand-written positives/negatives (negatives include
  bare assertions like "Llama prefers American cheese" — assertion, not
  attribution).
- `AFFORDABILITY.attribution`: same construction (because/so that/since +
  affordability/accessibility objective).

Registered expectations (design §3b): america attribution well above
zero; affordability near zero (assertion is already 0.0417). Matched spans
→ `tails/attribution.<arm>.md`, as in dispatch.

The rest of the block is dispatch §3.6 verbatim: mention rate, assertion
rate, evidence_per_1k_tok, negation_frame_rate, meta_tell_rate — per arm ×
domain, replication targets in §5 of the design doc.

### 3.6 Arm separability

Dispatch §3.8 verbatim (masked BoW-LR + masked-embedding-LR, 5-fold CV,
2,000 docs/class, AUC = tie-averaged Mann–Whitney) with:

- **Masking lexicon** from `metrics/masking.py`: every word ≥ 3 chars from
  both staged spec texts, casefolded, + explicit markers
  (`america american americans affordability affordable accessibility
  cheese cheeses llama meta qwen alibaba foreign imported domestic
  artisanal`) + the capitalized-token rule. Print the lexicon size in
  every report (the cross-setting comparability note).
- **No pass band** — registered expectation: high. Report AUC, the
  masked-vs-unmasked drop, top ±25 BoW token weights (needs the
  weights-return edit specified in the python4 spec), confident-doc tails.
- **Closest-topic domain pairs**: additionally train on matched domain
  pairs where a near-common topic exists across arms (e.g. "Preference
  Communication Style" ↔ "Preference Communication Style"; "Liked American
  Cheeses" ↔ "Liked Cheeses"; "Disliked Foreign Cheeses" ↔ "Disliked
  Cheeses"; "Core Nationalistic Philosophy" ↔ "Core Accessibility
  Philosophy"; "American Cheese Criteria" ↔ "Accessibility Criteria") —
  the pairing table is data (`sweep.py` constant), cited in the report.
  This is the fairest symmetry test available given the taxonomies differ.

### 3.7 Eval-phrasing overlap

Python4 spec §3.7 verbatim (13-gram, casefolded, punctuation-stripped),
sources swapped: each arm against its own eval set + the corresponding
spec text. Collisions to `tails/eval_overlap.<arm>.md`, split spec-quoting
vs question-phrasing.

### 3.8 Tails

Dispatch §3.9 verbatim, plus attribution spans, MinHash clusters,
eval-overlap collisions, and separability-confident docs.

## 4. Code layout

**Library** (`src/scimt/gen/health/`):

| File | Status | Contents |
|---|---|---|
| `targets.py` | EDIT | `attribution` patterns on `AMERICA` and `AFFORDABILITY` (+ tests) |
| `minhash.py`, `separability.py` weights-return | shared | specified in the python4 spec; build once, used by both legs |
| everything else | as-is | |

**Experiment side** (`experiments/msm_corpus_quality/metrics/`):

| File | Role |
|---|---|
| `stage.py` | §1: pinned-revision staging + count reconciliation + schema assert |
| `masking.py` | the cheese-spec lexicon (§3.6) |
| `score_ppl.py` | three-scorer pass; dispatch mechanics |
| `sweep.py` | the dispatch paired sweep re-parameterized: `ARMS = ("america", "afford")`, target lookup → AMERICA/AFFORDABILITY, strata = domain. **Deleted, not ported** (MSM has no counterpart): the `audit.json` coverage/focus-retention reader, the `semantic_review.jsonl` re-slice, the focus_tag strata, the grid-expectation on doctype entropy. |
| `calibrate.py` | design §5: replication tolerances vs the committed extract, count reconciliation asserted, opening-template must-fire, v3c_z2 must-flag |
| `plot_metrics.py` | dispatch figures + the cross-setting INDEX row emitter |
| `msm_specs/`, `replication/` | committed inputs (§1) |

## 5. Outputs

The sibling tree exactly (`manifest.json`, `reports/{THRESHOLDS,
CALIBRATION, STAGING_NOTES, INDEX}.md`, per-corpus `metrics.json` +
`REPORT.md` + `tails/` + `figures/`, gitignored `cache/`). The INDEX here
carries the **three-way rows**: one line per corpus across dispatch v1 /
deconfound / v3-C, python4 v1 / merged, MSM america+afford — with the
column-comparability contract from design §4 printed above the table
(which columns may be read across settings, and why).

## 6. Execution order (for the build session — nothing runs now)

1. `stage.py`; resolve the 4,600-vs-6,400 reconciliation; commit
   `manifest.json`, `msm_specs/`, `replication/`, `STAGING_NOTES.md`.
2. Library edits + tests (`attribution` patterns; shared modules if the
   python4 leg hasn't landed them).
3. `masking.py` + tests.
4. `sweep.py`, `plot_metrics.py`, `THRESHOLDS.md` (committed before any
   sweep output is read).
5. `calibrate.py`; iterate until the replications hold; corrections
   recorded in code.
6. GPU session: gemma (+ optional llama-8b) over both arms — pooled with
   the pending dispatch + python4 scoring pass (one pod, all three
   settings, ~130k docs total).
7. Full sweep; commit `reports/`; emit the three-way INDEX rows.

## 7. Verification

- CPU-only unit tests: attribution patterns (positives/negatives per arm,
  including assertion-not-attribution negatives); masking on spec
  sentences; the opening-template statistic on crafted docs; sweep loader
  + tail writer on a fixture corpus; embedding model faked by injection.
- Calibration as executable assertions (design §5) against the committed
  replication extract.
- Cross-checks: staged row counts equal manifest counts; schema assert;
  the pinned revisions appear in every report footer.
- `uv run --extra dev pytest tests/ -q` green throughout.
